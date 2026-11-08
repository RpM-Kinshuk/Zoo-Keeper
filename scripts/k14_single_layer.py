"""Execute tensor-level permutation test declared in K14 PROTOCOL.md."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from evidence.sketch import row_block_bounds, two_sided_sketch, sketch_cosine
from lab.artifacts import atomic_json
root=Path(os.environ['ZOO_DATA'])
out=root/'runs/track-b-k14-20261107'
assert (out/'PROTOCOL.md').is_file()
assert not (out/'single-layer.json').exists()
p=root/'lab/inputs/cohort-v1-smol-360-f8027fd0eaee/model.safetensors'
with safe_open(str(p),framework='pt',device='cpu') as f:
    group={kind:f.get_tensor('model.layers.0.mlp.'+kind+'_proj.weight') for kind in ['gate','up','down']}
rng=np.random.default_rng(20261028)
perm=torch.from_numpy(rng.permutation(group['up'].shape[0]))
inv=torch.argsort(perm)
changed={'gate':group['gate'][perm], 'up':group['up'][perm], 'down':group['down'][:,perm]}
exact=all(torch.equal(changed[k][inv] if k!='down' else changed[k][:,inv],v) for k,v in group.items())
assert exact
start,end=row_block_bounds(group['down'].shape[0])
a=two_sided_sketch(group['down'][start:end],64)
b=two_sided_sketch(changed['down'][start:end],64)
drift=float(np.linalg.norm(a.astype(float)-b)/np.linalg.norm(a.astype(float)))
tol=32*np.finfo('float32').eps
atomic_json(out/'single-layer.json',dict(input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()},protocol_sha256=hashlib.sha256((out/'PROTOCOL.md').read_bytes()).hexdigest(),inverse_indexing_exact=exact,relative_sketch_drift=drift,tolerance=float(tol),sketch_cosine=sketch_cosine(a,b),exact_sketch_invariance='FAIL' if drift>tol else 'PASS',scope='one exact algebraic MLP reordering, tensor sketch only; network inference unmeasured'))
print(json.dumps(dict(inverse_exact=exact,drift=drift,tolerance=float(tol),cosine=sketch_cosine(a,b))))
