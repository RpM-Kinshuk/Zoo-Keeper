"""Independent candidate pins, complete effective coverage and tied alias audit."""
import hashlib,json,os
from pathlib import Path
from contextlib import ExitStack
import torch
from safetensors import safe_open
from lab.artifacts import atomic_json
root=Path(os.environ['ZOO_DATA']);out=root/'runs/track-b-k5-parent-out-20261108';protocol=json.loads((out/'protocol.json').read_text())
source=root/'runs/track-a-compatibility-20261105/full-weight-artifact-map137.json';catalog={r['model_id']:r for r in json.loads(source.read_text())['rows']}
basepath={'olmo-1b':'inputs/pinned-base-olmo-1b-a1847dff3500','q25-05':'inputs/cohort-v1-q25-05-060db6499f32','q3-06':'inputs/cohort-v1-q3-06-c1899de289a0','smol-360':'inputs/cohort-v1-smol-360-f8027fd0eaee'}
proof={};maps={};models=sorted(set(protocol['queries'])|{n for pool in protocol['pools'].values() for n in pool})
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def path(n):return root/'lab'/basepath[n] if n in basepath else root/'lab'/n
for n in models:
 r=catalog[n];p=root/r['manifest'];assert digest(p)==r['manifest_sha256'];m=json.loads(p.read_text());proof[n]={'manifest':r['manifest'],'manifest_sha256':r['manifest_sha256'],'weight_pins':{f['filename']:f['sha256'] for f in m['files'] if f['filename'].endswith('.safetensors')}}
 maps[n]={}
 for p in sorted(path(n).glob('*.safetensors')):
  with safe_open(str(p),framework='pt',device='cpu') as f:
   for k in f.keys():maps[n][k]=(p,tuple(f.get_slice(k).get_shape()))
rows=[];tie_checked=set()
for mid in protocol['queries']:
 meta=json.loads((path(mid)/'quantization.json').read_text());aliases=meta['tied_alias_groups']
 def canonical(k):return next((min(g) for g in aliases if k in g),k)
 child={canonical(k):shape for k,(_,shape) in maps[mid].items()}
 for n in protocol['pools'][mid]:
  parent={canonical(k):shape for k,(_,shape) in maps[n].items()}
  assert child==parent,(mid,n,'effective key/shape coverage mismatch')
  for group in aliases:
   keys=[k for k in group if k in maps[n]]
   if len(keys)<2 or (n,tuple(keys)) in tie_checked:continue
   with ExitStack() as stack:
    streams={k:stack.enter_context(safe_open(str(maps[n][k][0]),framework='pt',device='cpu')) for k in keys}
    shape=maps[n][keys[0]][1]
    for first in range(0,shape[0],128):
     a=streams[keys[0]].get_slice(keys[0])[first:first+128,:]
     for key in keys[1:]:
      b=streams[key].get_slice(key)[first:first+128,:]
      assert a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),(n,keys,'conflicting tie')
   tie_checked.add((n,tuple(keys)))
  rows.append(dict(query=mid,candidate=n,effective_tensor_sets_and_shapes_equal=True))
atomic_json(out/'candidate-coverage.json',dict(status='PASS',pairs=rows,candidate_pins=proof,tied_groups_verified=len(tie_checked),source_map_sha256=digest(source),scope='effective header coverage and bit-exact tied aliases; compare recorded run hashes with these pins before final verdict'))
print('PASS effective coverage/aliases for',len(rows),'candidate pairs;',len(models),'pinned identities')
