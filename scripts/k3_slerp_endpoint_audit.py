"""Audit stored SLERP children against published linear endpoint relation."""
from pathlib import Path
import hashlib,json,os,re
from contextlib import ExitStack
import numpy as np
import torch
from safetensors import safe_open
from lab.artifacts import atomic_json
root=Path(os.environ['ZOO_DATA'])
repo=Path(__file__).resolve().parents[1]
out=root/'runs/track-b-k3-slerp-endpoints-20261107'
out.mkdir(parents=True,exist_ok=True)
assert not (out/'results.json').exists()
protocol=repo/'docs/steering/K3-slerp-exact-protocol.md'
atomic_json(out/'protocol.json',{'text':protocol.read_text(),'sha256':hashlib.sha256(protocol.read_bytes()).hexdigest()})
hashes={};rows=[]
def bind(p):
    hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
for lane,dirname in [('smol360','recipes'),('qwen','recipes-v2-frozen')]:
 for rp in sorted((root/f'runs/track-b-formal-{lane}-merges-v1'/dirname).glob('*slerp*.json')):
    bind(rp);rec=json.loads(rp.read_text());names=rec['parents']+[rec['model_id']]
    paths=[root/'lab'/n/'model.safetensors' for n in names]
    for p in paths:
        if str(p.relative_to(root)) not in hashes:bind(p)
    details=[]
    with ExitStack() as stack:
        handles=[stack.enter_context(safe_open(str(p),framework='pt',device='cpu')) for p in paths]
        assert set(handles[0].keys())==set(handles[1].keys())==set(handles[2].keys())
        keys=handles[2].keys()
        layers=sorted({int(m.group(1)) for k in keys if (m:=re.search(r'\.layers\.(\d+)\.',k))})
        for key in keys:
            a,b,c=[f.get_tensor(key).float() for f in handles]
            t=rec['t']
            if isinstance(t,list):
                m=re.search(r'\.layers\.(\d+)\.',key)
                position=layers.index(int(m.group(1)))/max(1,len(layers)-1) if m else 0
                t=float(np.interp(position,np.linspace(0,1,len(t)),t))
            expected=(1-t)*a+t*b
            error=float(torch.max(torch.abs(c-expected)))
            tolerance=8*torch.finfo(torch.float32).eps*max(1,float(expected.abs().max()))
            # Angles descriptive only; use float64 to avoid dot cancellation.
            av=a.flatten().double();bv=b.flatten().double()
            denom=float(torch.linalg.vector_norm(av)*torch.linalg.vector_norm(bv))
            cosine=float(torch.dot(av,bv)/denom) if denom else None
            details.append(dict(tensor=key,t=t,error=error,tolerance=tolerance,passed=error<=tolerance,parent_cosine=cosine))
    row=dict(model_id=rec['model_id'],tensors=details,passed=all(x['passed'] for x in details),failed_tensors=sum(not x['passed'] for x in details),max_error=max(x['error'] for x in details),min_parent_cosine=min(x['parent_cosine'] for x in details if x['parent_cosine'] is not None))
    rows.append(row);print(row['model_id'],row['passed'],row['failed_tensors'],row['max_error'],row['min_parent_cosine'],flush=True)
atomic_json(out/'results.json',dict(rows=rows,input_sha256=hashes,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),historical_kernel_source_verified=False))
