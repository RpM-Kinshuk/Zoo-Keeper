"""Fixed exact-replay parent decision, tested with parent removed."""
import hashlib,json,os,time
from pathlib import Path
from contextlib import ExitStack
import torch
from safetensors import safe_open
from lab.artifacts import atomic_json
from lab.quantization import simulate_quantization
root=Path(os.environ['ZOO_DATA']);repo=Path(__file__).resolve().parents[1]
out=root/'runs/track-b-k5-parent-out-20261108';out.mkdir(parents=True,exist_ok=True)
assert not (out/'results.json').exists()
prior=root/'runs/track-b-k5-quantization-20261108';protocol=json.loads((prior/'protocol.json').read_text());positive=protocol['positive'];controls=protocol['controls']
basepath={'olmo-1b':'inputs/pinned-base-olmo-1b-a1847dff3500','q25-05':'inputs/cohort-v1-q25-05-060db6499f32','q3-06':'inputs/cohort-v1-q3-06-c1899de289a0','smol-360':'inputs/cohort-v1-smol-360-f8027fd0eaee'}
recipes={};pools={};paths={};hashes={};start=time.monotonic()
def bind(p):
 rel=str(p.relative_to(root))
 if rel not in hashes:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
  hashes[rel]=h.hexdigest()
 return hashes[rel]
for mid in positive:
 p=root/'lab'/mid/'recipe.json';bind(p);r=json.loads(p.read_text());r=r.get('recipe',r);recipes[mid]=r;anchor=r['anchor']
 pools[mid]=sorted([n for n in controls+positive if n.startswith(anchor+'-') and n!=mid]+[anchor])
 assert r['parents'][0] in pools[mid]
 for n in pools[mid]+[mid]:paths[n]=root/'lab'/basepath[n] if n in basepath else root/'lab'/n
p=repo/'docs/steering/K5-parent-out-protocol.md'
atomic_json(out/'protocol.json',dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),queries=positive,pools=pools))
def open_model(stack,path):
 loc={}
 for p in sorted(path.glob('*.safetensors')):
  bind(p);f=stack.enter_context(safe_open(str(p),framework='pt',device='cpu'))
  for k in f.keys():
   assert k not in loc
   loc[k]=f
 assert loc
 return loc
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
rows=[]
for mid in positive:
 r=recipes[mid];mp=root/'lab'/mid/'quantization.json';bind(mp);meta=json.loads(mp.read_text());comparisons=[]
 for candidate in pools[mid]:
  try:
   with ExitStack() as stack:
    child=open_model(stack,paths[mid]);parent=open_model(stack,paths[candidate]);mapping={}
    for name,f in child.items():
     target=name
     if name not in parent:
      group=next((a for a in meta['tied_alias_groups'] if name in a),[]);target=next((a for a in group if a in parent),None)
     assert target is not None and target in parent,'unresolved key '+name
     assert f.get_slice(name).get_shape()==parent[target].get_slice(target).get_shape(),'shape mismatch '+name
     mapping[name]=target
    matched=True;mismatch=None
    for name,f in sorted(child.items()):
     key=mapping[name];shape=f.get_slice(name).get_shape()
     for first in range(0,shape[0],128) if len(shape)==2 else [None]:
      a=parent[key].get_slice(key)[first:first+128,:] if first is not None else parent[key].get_tensor(key)
      c=f.get_slice(name)[first:first+128,:] if first is not None else f.get_tensor(name)
      expected=simulate_quantization(a,r['bits']).to(torch.bfloat16) if a.is_floating_point() and len(shape)==2 else (a.to(torch.bfloat16) if a.is_floating_point() and name not in meta.get('preserved_legacy_attention_masks',{}) else a)
      if not equal(expected,c):matched=False;mismatch=dict(tensor=name,row_start=first);break
     if not matched:break
   comparison=dict(candidate=candidate,status='complete',matches=matched,first_mismatch=mismatch)
  except (AssertionError,FileNotFoundError,KeyError) as exc:comparison=dict(candidate=candidate,status='incomplete',reason=str(exc),matches=None)
  comparisons.append(comparison)
  assert time.monotonic()-start<6600
 complete=all(c['status']=='complete' for c in comparisons);matches=[c['candidate'] for c in comparisons if c['matches']]
 absent=[n for n in matches if n not in r['parents']]
 present_answer=matches[0] if complete and len(matches)==1 else None
 absent_answer=absent[0] if complete and len(absent)==1 else None
 row=dict(model_id=mid,true_parent=r['parents'][0],complete=complete,comparisons=comparisons,present_matches=matches,absent_matches=absent,present_answer=present_answer,absent_answer=absent_answer,present_wrong=present_answer is not None and present_answer not in r['parents'],absent_wrong=absent_answer is not None)
 rows.append(row);atomic_json(out/'progress.json',dict(rows=rows,input_sha256=hashes));print(mid,'present',present_answer,'absent',absent_answer,'matches',matches,'complete',complete,flush=True)
n_present=sum(r['present_answer'] is not None for r in rows);n_absent=sum(r['absent_answer'] is not None for r in rows);errors=sum(r['present_wrong']+r['absent_wrong'] for r in rows);answers=n_present+n_absent
summary=dict(present_answers=n_present,absent_answers=n_absent,present_errors=sum(r['present_wrong'] for r in rows),absent_errors=sum(r['absent_wrong'] for r in rows),pooled_errors=errors,pooled_answers=answers,pooled_false_per_answer=errors/answers if answers else None,present_coverage=n_present/len(rows),incomplete_queries=sum(not r['complete'] for r in rows))
summary['passed']=summary['incomplete_queries']==0 and answers>0 and errors/answers<=.05 and summary['present_coverage']>=.5
atomic_json(out/'results.json',dict(summary=summary,rows=rows,input_sha256=hashes,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()));print(json.dumps(summary),flush=True)
