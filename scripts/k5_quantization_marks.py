"""Bounded K5 exact grid and quantizer replay; see declared protocol."""
import csv,hashlib,json,os,time
from contextlib import ExitStack
from pathlib import Path
import torch
from safetensors import safe_open
from lab.artifacts import atomic_json
from lab.quantization import simulate_quantization
root=Path(os.environ['ZOO_DATA']);repo=Path(__file__).resolve().parents[1]
out=root/'runs/track-b-k5-quantization-20261108';out.mkdir(parents=True,exist_ok=True)
assert not (out/'results.json').exists()
models=list(csv.DictReader((repo/'lab/models.csv').open()))
positive=sorted(r['model_id'] for r in models if '-quantization-' in r['model_id'])
controls=sorted(r['model_id'] for r in models if '-sft-' in r['model_id'] or '-dpo-' in r['model_id'])+sorted(r['model_id'] for r in models if '-lora-' in r['model_id'])[:14]
assert len(positive)==12 and len(controls)==30
p=repo/'docs/steering/K5-quantization-protocol.md'
atomic_json(out/'protocol.json',dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),positive=positive,controls=controls))
hashes={};started=time.monotonic();results=[]
def bind(p):
 rel=str(p.relative_to(root))
 if rel not in hashes:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
  hashes[rel]=h.hexdigest()
 return hashes[rel]
def read(p):bind(p);return json.loads(p.read_text())
def opened(stack,path):
 handles={};location={}
 for p in sorted(path.glob('*.safetensors')):
  bind(p);f=stack.enter_context(safe_open(str(p),framework='pt',device='cpu'));handles[p.name]=f
  for name in f.keys():
   assert name not in location
   location[name]=f
 assert location
 return location
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
for mid in positive+controls:
 is_positive=mid in positive;path=root/'lab'/mid
 try:
  with ExitStack() as stack:
   children=opened(stack,path);bits=8;parents=None;aliases=[];reproduction=True;first_mismatch=None
   if is_positive:
    rec=read(path/'recipe.json');rec=rec.get('recipe',rec);bits=rec['bits'];parent=rec['parents'][0];pin=rec['inputs'][parent];pp=root/'lab'/pin['path']
    # Explicitly verify the supplied immutable parent weights before comparing.
    for filename,h in pin['weight_sha256'].items():assert bind(pp/filename)==h,'parent pin mismatch'
    parents=opened(stack,pp);qmeta=read(path/'quantization.json');aliases=qmeta['tied_alias_groups']
   grid=True;matrices=0;rows=0;max_distinct=0;counterexample=None
   for name,f in sorted(children.items()):
    shape=f.get_slice(name).get_shape()
    sample=f.get_slice(name)[:1] if shape else f.get_tensor(name)
    floating=sample.is_floating_point()
    pkey=name
    if parents is not None and pkey not in parents:
     group=next((a for a in aliases if name in a),[]);pkey=next((a for a in group if a in parents),None)
     assert pkey is not None,'unresolved alias '+name
    if floating and len(shape)==2:
     matrices+=1
     for start in range(0,shape[0],128):
      child=f.get_slice(name)[start:start+128,:]
      assert torch.isfinite(child).all()
      sorted_values=torch.sort(child.float(),dim=1).values
      counts=1+(sorted_values[:,1:]!=sorted_values[:,:-1]).sum(dim=1)
      max_distinct=max(max_distinct,int(counts.max()));rows+=len(counts)
      if bool((counts>2**bits).any()):
       grid=False
       if counterexample is None:
        i=int(torch.nonzero(counts>2**bits)[0]);counterexample=dict(tensor=name,row=start+i,distinct=int(counts[i]))
      if parents is not None:
       source=parents[pkey].get_slice(pkey)[start:start+128,:]
       expected=simulate_quantization(source,bits).to(torch.bfloat16)
       if not exact(expected,child):reproduction=False;first_mismatch=first_mismatch or name
      if not grid and not is_positive:break
    elif parents is not None:
     child=f.get_tensor(name);source=parents[pkey].get_tensor(pkey)
     expected=source.to(torch.bfloat16) if source.is_floating_point() and name not in qmeta.get('preserved_legacy_attention_masks',{}) else source
     if not exact(expected,child):reproduction=False;first_mismatch=first_mismatch or name
    if not grid and not is_positive:break
   row=dict(model_id=mid,positive=is_positive,status='tested',bits_limit=bits,grid_compatible=grid and matrices>0,matrices_checked=matrices,rows_checked=rows,max_distinct_values=max_distinct,counterexample=counterexample,parent_reproduced=reproduction if is_positive else None,first_reproduction_mismatch=first_mismatch)
 except (FileNotFoundError,KeyError,AssertionError) as exc:
  row=dict(model_id=mid,positive=is_positive,status='NOT_MEASURED',reason=str(exc))
 results.append(row);atomic_json(out/'progress.json',dict(rows=results,input_sha256=hashes));print(mid,row.get('grid_compatible'),row.get('parent_reproduced'),row.get('reason',''),flush=True)
 assert time.monotonic()-started<6600
summary=dict(positive_detected=sum(r.get('grid_compatible',False) for r in results if r['positive']),control_flags=sum(r.get('grid_compatible',False) for r in results if not r['positive']),parents_reproduced=sum(r.get('parent_reproduced',False) for r in results if r['positive']),not_measured=sum(r['status']!='tested' for r in results))
summary['passed']=summary==dict(positive_detected=12,control_flags=0,parents_reproduced=12,not_measured=0)
atomic_json(out/'results.json',dict(summary=summary,rows=results,input_sha256=hashes,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256((repo/'lab/quantization.py').read_bytes()).hexdigest()))
print(json.dumps(summary),flush=True)
