"""Descriptive all-group measurements; no fitted detector or parent decision."""
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import ExitStack

import torch
from safetensors import safe_open
from evidence.delta import block_delta
from evidence.sketch import row_block_bounds
from lab.artifacts import atomic_json

root = Path(os.environ['ZOO_DATA'])
repo = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--temporary-lab-root', type=Path, required=True)
args = parser.parse_args()
temporary_lab = args.temporary_lab_root.resolve()
out = root / 'runs/track-b-k5-joined-20261108-v2'
out.mkdir(exist_ok=True)
assert not (out / 'protocol.json').exists(), 'choose a fresh run, never overwrite'
torch.set_num_threads(4)
started = time.monotonic()
hashes = {}


def reference(path):
    path = path.resolve()
    if path.is_relative_to(root):
        return str(path.relative_to(root))
    if path.is_relative_to(temporary_lab):
        return 'temporary-lab/' + str(path.relative_to(temporary_lab))
    raise ValueError('unregistered input root')


def bind(path, expected=None):
    key = reference(path)
    if key not in hashes:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for data in iter(lambda: f.read(8 * 1024**2), b''):
                h.update(data)
        hashes[key] = h.hexdigest()
    if expected is not None:
        assert hashes[key] == expected, 'source pin mismatch: ' + key
    return hashes[key]


edges_path = repo / 'lab/edges.csv'
edges = {r['child']: json.loads(r['params_json']) for r in csv.DictReader(edges_path.open())}
groups = {
    'adapter': sorted(k for k in edges if '-lora-' in k),
    'refusal': sorted(k for k in edges if '-abliteration-' in k),
    'ties_dare': sorted(k for k, r in edges.items() if r.get('method') in ('ties', 'dare_ties')),
    'sft_dpo': sorted(k for k in edges if '-sft-' in k or '-dpo-' in k),
}
assert list(map(len, groups.values())) == [24, 8, 12, 16]
anchors = sorted({edges[k]['anchor'] for ks in groups.values() for k in ks})
assert len(anchors) == 4
paths, recipes = {}, {}
for mid in sum(groups.values(), []):
    candidates = [root / 'lab' / mid, temporary_lab / mid]
    path = next((p for p in candidates if (p / 'recipe.json').exists()), None)
    assert path is not None, 'missing checkpoint: ' + mid
    paths[mid] = path
    bind(path / 'recipe.json')
    r = json.loads((path / 'recipe.json').read_text())
    recipes[mid] = r.get('recipe', r)
for anchor in anchors:
    r = recipes[anchor + '-lora-dialogue-r8']
    paths[anchor] = root / r['inputs']['model_path']
assert len(paths) == 64
model_anchors = {k: (k if k in anchors else edges[k]['anchor']) for k in paths}
release = root / 'runs/track-a-k0-20261107/release-manifest.json'
bind(release)
archive = {r['path']: r['sha256'] for r in json.loads(release.read_text())['repositories'][0]['inventory']}
expected_files = {}
for mid, path in sorted(paths.items()):
    pins = recipes[mid + '-lora-dialogue-r8']['inputs']['model_pin']['weight_sha256'] if mid in anchors else None
    files = sorted(path.glob('*.safetensors'))
    assert files
    for p in files:
        expected_files[reference(p)] = pins[p.name] if pins else archive['models/' + mid + '/' + p.name]
protocol = repo / 'docs/steering/K5-joined-measurements-protocol.md'
atomic_json(out / 'protocol.json', dict(text=protocol.read_text(), protocol_sha256=bind(protocol),
    driver_sha256=bind(Path(__file__)), edges_sha256=bind(edges_path), groups=groups,
    paths={k: reference(v) for k, v in paths.items()}, expected_files=expected_files,
    backend='cpu', threads=4, gpu_reservations=[], model_weight_identities=len(paths)))


def opened(stack, path):
    result = {}
    for p in sorted(path.glob('*.safetensors')):
        bind(p, expected_files[reference(p)])
        handle = stack.enter_context(safe_open(str(p), framework='pt', device='cpu'))
        for name in handle.keys():
            assert name not in result
            result[name] = handle
    return result


def target(name):
    return name.endswith(('q_proj.weight', 'v_proj.weight'))


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def diagnostic(c, p, name):
    cs, ps = c[name].get_slice(name), p[name].get_slice(name)
    assert cs.get_shape() == ps.get_shape()
    lo, hi = row_block_bounds(cs.get_shape()[0], 256)
    child, parent = cs[lo:hi, :], ps[lo:hi, :]
    # Storage dtype is passed explicitly; never infer it from a float32 decode.
    result = block_delta(child, parent, child.dtype, compute_device='cpu')
    result.pop('top_singular_values')
    result.update(tensor=name, rows=[lo, hi], shape=cs.get_shape(), storage_dtype=str(child.dtype),
                  interpretable=result['status'] == 'ok' and result['rank_confidence'] == 'high'
                  and result['rel_update_norm'] >= .003)
    return result


rows, signatures = [], {}
for mid, path in sorted(paths.items()):
    assert time.monotonic() - started < 6600
    is_base = mid in anchors
    anchor = mid if is_base else edges[mid]['anchor']
    row = dict(model_id=mid, anchor=anchor, group=next((g for g, ids in groups.items() if mid in ids), 'base'))
    with ExitStack() as stack:
        child = opened(stack, path)
        parent = None if is_base else opened(stack, paths[anchor])
        signatures[mid] = {}
        identical, entries, dtypes = 0, 0, set()
        for name, f in sorted(child.items()):
            shape = f.get_slice(name).get_shape()
            t = f.get_tensor(name)
            if not t.is_floating_point():
                continue
            assert torch.isfinite(t).all(), 'nonfinite ' + name
            if not target(name):
                signatures[mid][name] = dict(shape=shape, dtype=str(t.dtype),
                                            native=digest(t), bf16=digest(t.to(torch.bfloat16)))
            elif parent is not None:
                assert name in parent and shape == parent[name].get_slice(name).get_shape()
                base = parent[name].get_tensor(name)
                identical += int((t.float() == base.float()).sum())
                entries += t.numel()
                dtypes.add((str(t.dtype), str(base.dtype)))
        row.update(target_entries=entries, target_identical_entries=identical,
                   target_identical_fraction=identical / entries if entries else None,
                   storage_dtype_pairs=sorted(dtypes), rank_diagnostics=[])
        if parent is not None and row['group'] in ('adapter', 'sft_dpo', 'refusal'):
            if row['group'] == 'refusal':
                chosen = sorted(n for n in child if n.endswith('down_proj.weight'))[:3]
            else:
                chosen = sorted(n for n in child if n.endswith('q_proj.weight'))[:1]
            for name in chosen:
                row['rank_diagnostics'].append(diagnostic(child, parent, name))
            if row['group'] in ('adapter', 'sft_dpo'):
                chosen = sorted(n for n in child if n.endswith('down_proj.weight'))[:3]
                row['writer_control_diagnostics'] = [diagnostic(child, parent, n) for n in chosen]
            if row['group'] == 'adapter':
                row['configured_rank'] = edges[mid]['rank']
    rows.append(row)
    atomic_json(out / 'progress.json', dict(rows=rows, input_sha256=hashes,
                                         elapsed_seconds=time.monotonic() - started))
    print(mid, row['target_identical_fraction'], row['rank_diagnostics'], flush=True)


def invariant(child_id, candidate_id, field):
    a, b = signatures[child_id], signatures[candidate_id]
    if set(a) != set(b):
        return False
    return all(a[n]['shape'] == b[n]['shape'] and a[n][field] == b[n][field]
               and (field != 'native' or a[n]['dtype'] == b[n]['dtype']) for n in a)


for row in rows:
    if row['group'] != 'adapter':
        continue
    mid, anchor = row['model_id'], row['anchor']
    pool = sorted(k for k in paths if k != mid and model_anchors[k] == anchor)
    row['native_true_parent_untouched'] = invariant(mid, anchor, 'native')
    row['bf16_true_parent_untouched'] = invariant(mid, anchor, 'bf16')
    row['parent_missing_names'] = sorted(set(signatures[mid]) ^ set(signatures[anchor]))
    row['native_support_present'] = [k for k in pool if invariant(mid, k, 'native')]
    row['native_support_absent'] = [k for k in row['native_support_present'] if k != anchor]
    row['parent_decision'] = 'NOT_TESTED_no_approved_discriminating_rule'

atomic_json(out / 'tensor-signatures.json', signatures)
atomic_json(out / 'results.json', dict(groups=groups, rows=rows, input_sha256=hashes,
    elapsed_seconds=time.monotonic() - started, backend='cpu', threads=4,
    status='DESCRIPTIVE_no_new_detector_cutoff_or_parent_rule'))
print('COMPLETE', len(rows), 'models', time.monotonic() - started, flush=True)
