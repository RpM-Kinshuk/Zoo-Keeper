"""Correct construction-inactive sampling with the unchanged rank gate."""
import hashlib
import json
import math
import os
from pathlib import Path
import time
from contextlib import ExitStack

import torch
from safetensors import safe_open
from evidence.delta import block_delta
from evidence.sketch import row_block_bounds
from lab.artifacts import atomic_json

root = Path(os.environ['ZOO_DATA']).resolve()
temporary = Path(os.environ['TEMPORARY_LAB_ROOT']).resolve()
repo = Path(__file__).resolve().parents[1]
out = root/'runs/track-b-k5-active-writers-20261108'
out.mkdir(exist_ok=False)
torch.set_num_threads(4)
started = time.monotonic()
hashes = {}


def reference(p):
    return str(p.relative_to(root)) if p.is_relative_to(root) else 'temporary-lab/'+str(p.relative_to(temporary))


def bind(p, expected=None):
    key = reference(p)
    if key not in hashes:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):
                h.update(b)
        hashes[key] = h.hexdigest()
    if expected is not None:
        assert hashes[key]==expected, 'pin mismatch '+key


def read(p):
    bind(p)
    return json.loads(p.read_text())


audit = read(root/'runs/track-b-k5-control-audit-20261108/results.json')
old = read(root/'runs/track-b-k5-joined-20261108-v2/results.json')
prior_protocol = read(root/'runs/track-b-k5-joined-20261108-v2/protocol.json')
selected = {r['anchor']:r for r in audit['writer_construction_coverage'] if r['method']=='heretic'}
controls = [r for r in old['rows'] if r['group'] in ('adapter','sft_dpo')]
ids = sorted([r['model_id'] for r in selected.values()]+[r['model_id'] for r in controls]+list(selected))
assert len(set(ids))==48
paths = {k:(temporary/prior_protocol['paths'][k].removeprefix('temporary-lab/') if prior_protocol['paths'][k].startswith('temporary-lab/') else root/prior_protocol['paths'][k]) for k in ids}
protocol = repo/'docs/steering/K5-active-writer-protocol.md'
atomic_json(out/'protocol.json',dict(text=protocol.read_text(),sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
    paths={k:reference(p) for k,p in paths.items()},layers={a:r['proposed_first_three_active_down_layers'] for a,r in selected.items()},
    expected_files={k:v for k,v in prior_protocol['expected_files'].items() if any(k.startswith(reference(p)+'/') for p in paths.values())}))


def opened(stack,path):
    result = {}
    for p in sorted(path.glob('*.safetensors')):
        bind(p,prior_protocol['expected_files'][reference(p)])
        f = stack.enter_context(safe_open(str(p),framework='pt',device='cpu'))
        for n in f.keys():
            assert n not in result
            result[n] = f
    return result


rows = []
for anchor, selection in sorted(selected.items()):
    layers = selection['proposed_first_three_active_down_layers']
    assert len(layers)==3 and all(k in selection['active_down_layers'] for k in layers)
    case_ids = [selection['model_id']]+sorted(r['model_id'] for r in controls if r['anchor']==anchor)
    for mid in case_ids:
        assert time.monotonic()-started<6600
        with ExitStack() as stack:
            child,parent = opened(stack,paths[mid]),opened(stack,paths[anchor])
            blocks = []
            for layer in layers:
                names = [n for n in child if n.endswith(f'layers.{layer}.mlp.down_proj.weight')]
                assert len(names)==1
                name = names[0]
                cs,ps = child[name].get_slice(name),parent[name].get_slice(name)
                shape = cs.get_shape()
                assert shape==ps.get_shape()
                lo,hi = row_block_bounds(shape[0],256)
                c,p = cs[lo:hi,:],ps[lo:hi,:]
                x = block_delta(c,p,c.dtype,compute_device='cpu')
                sv = x.pop('top_singular_values')
                delta = c.float().double()-p.to(c.dtype).float().double()
                x.update(tensor=name,layer=layer,rows=[lo,hi],shape=shape,storage_dtype=str(c.dtype),
                    delta_frobenius_norm=float(delta.norm()),
                    top_singular_values=[float(v) if math.isfinite(float(v)) else None for v in sv],
                    resolved=x['status']=='ok' and x['rank_confidence']=='high' and x['rel_update_norm']>=.003)
                x['resolved_rank_one'] = x['resolved'] and x['rank']==1
                blocks.append(x)
        row = dict(model_id=mid,anchor=anchor,positive=mid==selection['model_id'],blocks=blocks)
        rows.append(row)
        atomic_json(out/'progress.json',dict(rows=rows,input_sha256=hashes,elapsed_seconds=time.monotonic()-started))
        print(mid,[dict(rank=x['rank'],resolved=x['resolved'],status=x['status']) for x in blocks],flush=True)
positive = [x for r in rows if r['positive'] for x in r['blocks']]
negative = [x for r in rows if not r['positive'] for x in r['blocks']]
assert len(positive)==12 and len(negative)==120
summary = dict(positive_blocks=12,control_blocks=120,
    positive_resolved=sum(x['resolved'] for x in positive),positive_rank_one=sum(x['resolved_rank_one'] for x in positive),
    control_resolved=sum(x['resolved'] for x in negative),control_rank_one=sum(x['resolved_rank_one'] for x in negative),
    elapsed_seconds=time.monotonic()-started,model_weight_identities=48,
    no_threshold_change=True,no_new_detector_verdict=True)
atomic_json(out/'results.json',dict(summary=summary,rows=rows,input_sha256=hashes,
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print(json.dumps(summary),flush=True)
