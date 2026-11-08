"""Follow-up to failed linear endpoint audit; rules in docs/steering."""
import hashlib,json,math,os,re
from pathlib import Path
from contextlib import ExitStack
import torch
from safetensors import safe_open
from lab.artifacts import atomic_json
from lineage.upstream import load_records
from lineage.records import view_diagnostics
root=Path(os.environ['ZOO_DATA']);repo=Path(__file__).resolve().parents[1]
prior=root/'runs/track-b-k3-slerp-endpoints-20261107/results.json'
audit=json.loads(prior.read_text());fits=json.loads((root/'runs/track-b-k3-deltas-20261107/results.json').read_text())
out=root/'runs/track-b-k3-slerp-trig-20261107-v2';out.mkdir(parents=True,exist_ok=True)
assert not (out/'results.json').exists()
p=repo/'docs/steering/K3-slerp-trig-protocol.md'
atomic_json(out/'protocol.json',dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
rows=[]
catalogues={}
for row in audit['rows']:
 mid=row['model_id'];fit=next(f for f in fits if f['model_id']==mid)
 lane='smol360' if mid.startswith('smol') else 'qwen'
 if lane not in catalogues:
  index=root/f'runs/track-b-formal-{lane}-merges-v1/evidence/record-index.json'
  catalogues[lane]=load_records(index,root/'runs',data_root=root)
 records=catalogues[lane]
 def mid_of(r):return r.key.split('@')[0].removeprefix('lab/')
 child=next(r for r in records if mid_of(r)==mid)
 recipe=json.loads((root/'lab'/mid/'recipe.json').read_text());recipe=recipe.get('recipe',recipe)
 parents=[next(r for r in records if mid_of(r)==n) for n in recipe['parents']]
 base=next(r for r in records if r.key==recipe['parent_identities'][recipe['reference']])
 aligned=[dict(view_diagnostics(child,r,k=64)[0]) for r in parents+[base]]
 common=set.intersection(*(set(x) for x in aligned))
 fitted_names={child.names[i] for i in common}
 assert len(fitted_names)==fit['n_shared']
 paths=[root/'lab'/n/'model.safetensors' for n in recipe['parents']+[mid]]
 details=[];errors=[]
 with ExitStack() as stack:
  handles=[stack.enter_context(safe_open(str(p),framework='pt',device='cpu')) for p in paths]
  for item in row['tensors']:
   d=item['parent_cosine'];t=item['t'];key=item['tensor']
   if d is None:coeff=[1-t,t]
   elif abs(d)>.9995:coeff=[1-t,t]
   else:
    theta=math.acos(max(-1,min(1,d)));coeff=[math.sin((1-t)*theta)/math.sin(theta),math.sin(t*theta)/math.sin(theta)]
   if not item['passed']:
    a,b,c=[f.get_tensor(key).float() for f in handles];expected=coeff[0]*a+coeff[1]*b
    error=float((c-expected).abs().max());tolerance=8*torch.finfo(torch.float32).eps*max(1,float(expected.abs().max()))
    details.append(dict(tensor=key,coefficients=coeff,error=error,tolerance=tolerance,passed=error<=tolerance))
   # Existing sketches contain only matrix views; do not treat 1d norms as fitted.
   if key not in fitted_names:continue
   m=re.search(r'\.layers\.(\d+)\.',key)
   if 'layers' in fit:
    if m is None:continue
    layer=int(m.group(1));f=next(x for x in fit['layers'] if x['layer']==layer)
   else:f=fit['fit']
   errors.append(max(abs(f['coefficients'][i]-coeff[i]) for i in range(2)))
 rows.append(dict(model_id=mid,max_coefficient_error=max(errors),coefficient_pass=max(errors)<=.05,nonlinear_reproduction=details,stored_child_reproduced=all(x['passed'] for x in details),fitted_matrix_tensors=len(errors)))
print(json.dumps(rows),flush=True)
atomic_json(out/'results.json',dict(rows=rows,input_sha256={str(prior.relative_to(root)):hashlib.sha256(prior.read_bytes()).hexdigest(),'runs/track-b-k3-deltas-20261107/results.json':hashlib.sha256((root/'runs/track-b-k3-deltas-20261107/results.json').read_bytes()).hexdigest()},raw_input_hash_record='runs/track-b-k3-slerp-endpoints-20261107/results.json',driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
