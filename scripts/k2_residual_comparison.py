"""Bounded matched comparison; see predeclared K2 residual protocol."""
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from studies.baselines.paths import workspace_root
import time
import numpy as np
from safetensors import safe_open
from lab.artifacts import atomic_json
from studies.baselines.residual import signature, score

root = workspace_root(Path(__file__).resolve().parents[1])
repo = Path(__file__).resolve().parents[1]
out = root / 'runs/track-c-k2-residual-20261108'
out.mkdir(exist_ok=True)
assert not (out / 'results.json').exists(), 'Never overwrite completed run'
started = time.monotonic()
prior = root / 'runs/track-b-k5-parent-out-20261108'
protocol = json.loads((prior / 'protocol.json').read_text())
expected = json.loads((prior / 'results.json').read_text())['input_sha256']
basepath = {'olmo-1b':'inputs/pinned-base-olmo-1b-a1847dff3500',
            'q25-05':'inputs/cohort-v1-q25-05-060db6499f32',
            'q3-06':'inputs/cohort-v1-q3-06-c1899de289a0',
            'smol-360':'inputs/cohort-v1-smol-360-f8027fd0eaee'}
hashes = {}
cost = {'logical_weight_bytes':0, 'hash_scan_bytes':0}


def bind(path, require_prior=False):
    key = str(path.relative_to(root))
    if key not in hashes:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2), b''):
                h.update(block)
                cost['hash_scan_bytes'] += len(block)
        hashes[key] = h.hexdigest()
    if require_prior:
        assert key in expected and hashes[key] == expected[key], 'B input pin mismatch: '+key
    return hashes[key]


for path in [prior/'protocol.json', prior/'results.json', Path(__file__),
             repo/'studies/baselines/residual.py',
             repo/'docs/research/2026-11-08-k2-residual-protocol.md']:
    bind(path)
atomic_json(out/'protocol.json', {'text':(repo/'docs/research/2026-11-08-k2-residual-protocol.md').read_text(),
                                'queries':protocol['queries'], 'pools':protocol['pools'],
                                'input_sha256':hashes.copy(), 'pid':os.getpid(),
                                'backend':'numpy float64 CPU, BLAS threads=8'})


def extract(mid):
    path = root/'lab'/basepath.get(mid, mid)
    bind(path/'config.json')
    cfg = json.loads((path/'config.json').read_text())
    layers, width = cfg['num_hidden_layers'], cfg['hidden_size']
    vectors, concentrations = [], []
    with ExitStack() as stack:
        loc = {}
        for shard in sorted(path.glob('*.safetensors')):
            bind(shard, require_prior=True)
            handle = stack.enter_context(safe_open(str(shard), framework='pt', device='cpu'))
            for key in handle.keys():
                assert key not in loc
                loc[key] = handle
        assert loc
        for layer in range(layers):
            keys = [f'model.layers.{layer}.mlp.{role}_proj.weight' for role in ('down','up')]
            down, up = [loc[k].get_tensor(k) for k in keys]
            assert down.shape[0] == width and up.shape[1] == width
            cost['logical_weight_bytes'] += sum(t.numel()*t.element_size() for t in (down,up))
            vector, concentration = signature(down.double().numpy(), up.double().numpy())
            vectors.append(vector); concentrations.append(concentration)
            assert time.monotonic()-started < 6600, '110min limit'
    return np.stack(vectors), np.array(concentrations)


rows, rankings, extractions = [], [], []
for family in basepath:
    queries = [q for q in protocol['queries'] if q.startswith(family+'-')]
    names = sorted(set(queries).union(*(protocol['pools'][q] for q in queries)))
    profiles = {}
    for mid in names:
        t = time.monotonic()
        profiles[mid] = extract(mid)
        extractions.append({'model_id':mid, 'layers':len(profiles[mid][0]),
                            'width':int(np.sqrt(profiles[mid][0].shape[1])),
                            'seconds':time.monotonic()-t})
        print('extracted', mid, round(time.monotonic()-t, 2), flush=True)
    for query in queries:
        recipe_path = root/'lab'/query/'recipe.json'
        bind(recipe_path, require_prior=True)
        rec = json.loads(recipe_path.read_text()); rec = rec.get('recipe',rec)
        parent = rec['parents'][0]
        scores = {}
        for candidate in protocol['pools'][query]:
            value, assignment = score(profiles[query], profiles[candidate])
            scores[candidate] = value
            rows.append({'query':query, 'candidate':candidate, 'score':value,
                         'status':'tested', 'true_parent':candidate == parent,
                         'evidence_tier':'construction_records',
                         'aligned_layers':len(assignment),
                         'identity_alignment':int(np.sum(assignment == np.arange(len(assignment))))})
        best = max(scores.values())
        top = [n for n in scores if scores[n] == best]
        absent = {n:s for n,s in scores.items() if n != parent}
        rankings.append({'query':query, 'parent':parent,
                         'parent_rank':1+sum(s > scores[parent] for s in scores.values()),
                         'top_candidates':top, 'unique_correct_top':top == [parent],
                         'parent_absent_top':[n for n in absent if absent[n] == max(absent.values())],
                         'parent_decision':'unsupported', 'type_direction':'unsupported'})
    del profiles
    atomic_json(out/'progress.json', {'rows':rows, 'rankings':rankings,
                                     'extractions':extractions, 'input_sha256':hashes, 'cost':cost})
summary = {'models':len(extractions), 'queries':len(rankings), 'comparisons':len(rows),
           'unique_correct_top':sum(r['unique_correct_top'] for r in rankings),
           'complete':bool(len(rows)==126 and len(rankings)==12 and all(np.isfinite(r['score']) for r in rows)),
           'calibrated_parent_decisions':'unsupported', 'ancestry_auroc':'unsupported_no_independent_negatives'}
result = {'summary':summary, 'rows':rows, 'rankings':rankings, 'extractions':extractions,
          'input_sha256':hashes, 'cost':cost, 'elapsed_seconds':time.monotonic()-started}
atomic_json(out/'results.json', result)
print(json.dumps(summary), flush=True)
