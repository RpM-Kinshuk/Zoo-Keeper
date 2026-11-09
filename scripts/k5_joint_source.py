"""Fixed joint classic-source development test, with conservative rank certificates."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from evidence.safetensors_header import header_length, parse_header, decode_plain, _DTYPE_BYTES
from lab.artifacts import atomic_json
from scripts.k5_classic_noise import radius

U64=2.**-53
TINY=math.ulp(0.)


class Unsupported(ValueError):
    pass


def up(x):
    if not math.isfinite(x): raise Unsupported('nonfinite scalar')
    return math.nextafter(x,math.inf)


def down(x):
    if not math.isfinite(x): raise Unsupported('nonfinite scalar')
    return math.nextafter(x,-math.inf)


def add(a,b): return up(a+b)
def mul(a,b): return up(a*b)
def div(a,b):
    if b<=0: raise Unsupported('zero denominator')
    return up(a/b)
def sqrt(a): return up(math.sqrt(max(0.,a)))


def dot_error(n):
    nu=mul(float(n),U64)
    den=down(1.-nu)
    if den<=.99: raise Unsupported('dot too long')
    return div(nu,den),div(mul(float(2*n),TINY),den)


def norms(x):
    x=x.double()
    sh=float((x*x).sum())
    if not math.isfinite(sh): raise Unsupported('norm overflow')
    g,a=dot_error(x.numel())
    hi=sqrt(div(add(sh,a),down(1.-g)))
    lo=math.nextafter(math.sqrt(max(0.,down(down(sh-a)/up(1.+g)))),-math.inf)
    return max(0.,lo),hi


def delta_error(c,w):
    return add(mul(U64,add(norms(c)[1],norms(w)[1])),mul(sqrt(float(c.numel())),TINY))


def budget(w,p,c):
    rr=radius(c)
    if not bool(torch.isfinite(rr).all()): raise Unsupported('unsupported rounding cell')
    q=norms(rr)[1]
    alpha=div(2.**-24,down(1.-2.**-24))
    a=add(mul(alpha,norms(p)[1]),mul(mul(alpha,add(2.,alpha)),add(norms(c)[1],q)))
    coefficient=add(add(2.,mul(3.,alpha)),mul(alpha,alpha))
    a=add(a,mul(mul(coefficient,2.**-150),sqrt(float(c.numel()))))
    h=0. if torch.equal(w,p) else add(norms(p.double()-w.double())[1],delta_error(p,w))
    return add(add(h,q),a)


def certificate(d,u,v):
    """Lower bound on rank<=1 distance; scalar arithmetic rounded outward."""
    m,n=d.shape
    uf,df,vf=norms(u)[1],norms(d)[1],norms(v)[1]
    gm,am=dot_error(m);gn,an=dot_error(n)
    # Explicit absolute underflow terms for each projected dot product.
    et=add(mul(gm,mul(uf,df)),mul(sqrt(float(2*n)),am))
    ej=add(add(mul(et,vf),mul(gn,mul(add(mul(uf,df),et),vf))),mul(2.,an))
    j=(u.T@d)@v
    if not bool(torch.isfinite(j).all()): raise Unsupported('projection overflow')
    x,y,z,t=map(float,j.reshape(-1))
    p=x*t;q=y*z
    lo=down(down(p)-up(q));hi=up(up(p)-down(q))
    det_lo=0. if lo<=0<=hi else min(abs(lo),abs(hi))
    jn=norms(j)[1]
    if det_lo==0 or jn==0: return 0.
    def op_upper(a,g,absolute,f):
        gram=a.T@a
        err=add(mul(g,mul(f,f)),absolute)
        row=max(add(abs(float(r[0])),abs(float(r[1]))) for r in gram)
        return sqrt(add(row,mul(2.,err)))
    un=op_upper(u,gm,am,uf);vn=op_upper(v,gn,an,vf)
    numerator=max(0.,down(down(det_lo/jn)-ej))
    return max(0.,down(numerator/mul(un,vn)))


def witness_upper(d,u,b):
    measured=u[:,None]*b[None,:]
    r=norms(d-measured)[1]
    sub=div(add(r,mul(sqrt(float(d.numel())),TINY)),down(1.-U64))
    product=add(mul(U64,mul(norms(u)[1],norms(b)[1])),mul(sqrt(float(d.numel())),TINY))
    return add(sub,product)


def reusable_pairs(rows):
    grouped={}
    for row in rows:
        key=(row['query'],row['candidate'])
        grouped.setdefault(key,[]).append(row)
    for key,pair in grouped.items():
        if len(pair)!=2 or {r['model'] for r in pair}!={'fp32_input','bf16_input'}:
            raise ValueError('invalid saved hypothesis pair '+str(key))
    return {key:pair for key,pair in grouped.items()
            if not any(r['state']=='unresolved' and r['reason']=='no weights' for r in pair)}


def resume_provenance(path):
    protocol=path.parent/'protocol.json';driver=path.parent/'driver.py'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    value=dict(progress_sha256=sha(path),protocol_sha256=sha(protocol),driver_sha256=sha(driver))
    if json.loads(protocol.read_text())['driver_sha256']!=value['driver_sha256']:
        raise ValueError('original frozen driver pin mismatch')
    return value


def read_range(p,begin,end,readers,stack):
    if p not in readers:readers[p]=stack.enter_context(p.open('rb'))
    f=readers[p];f.seek(begin);data=f.read(end-begin)
    if len(data)!=end-begin:raise Unsupported('truncated tensor range')
    return data


def svd_vectors(d,device):
    if device=='cpu':return torch.linalg.svd(d,full_matrices=False)
    # GPU only proposes test vectors; every guarded certificate stays CPU FP64.
    values=torch.linalg.svd(d.to(device),full_matrices=False,driver='gesvd')
    return tuple(x.cpu() for x in values)


def numerical_check(out,device='cpu'):
    assert float(torch.tensor(TINY,dtype=torch.float64)*1.)==TINY
    gen=torch.Generator().manual_seed(20261028)
    rows=[]
    for m,n in [(8,16),(32,64),(64,128)]:
        a=torch.randn(m,generator=gen,dtype=torch.float32).double()
        b=torch.randn(n,generator=gen,dtype=torch.float32).double()
        exact=a[:,None]*b[None,:]
        for scale in [0.,1e-8,1e-4,1.]:
            e=torch.randn((m,n),generator=gen,dtype=torch.float32).double()*scale
            d=exact+e
            known=add(norms(e)[1],delta_error(exact,-e))
            uu,ss,vh=svd_vectors(d,device)
            for distorted in [False,True]:
                u=uu[:,:2];v=vh[:2].T
                if distorted:
                    transform=torch.tensor([[1.,5.],[0.,.01]],dtype=torch.float64)
                    u=u@transform;v=v@transform
                lower=certificate(d,u,v)
                assert lower<=known, 'lower certificate exceeds known rank-one witness'
                rows.append(dict(shape=[m,n],noise_scale=scale,distorted=distorted,lower=lower,known_upper=known))
    d=torch.diag(torch.tensor([2.,1.,.5,.25],dtype=torch.float64))
    u,s,vh=svd_vectors(d,device)
    lower=certificate(d,u[:,:2],vh[:2].T)
    assert 0<lower<=math.sqrt(1.+.5**2+.25**2)
    atomic_json(out/'certificate-validation.json',dict(passed=True,checks=len(rows)+1,rows=rows,positive_power_check=lower))


def run(root,repo,out,stack,resume=None,seconds=900,device='cpu'):
    assert json.loads((out/'certificate-validation.json').read_text())['passed']
    assert (out/'proof-review.md').exists()
    start=time.monotonic();hashes={};read_ids=set();attempt_ids=set();payload=0
    readers={};hash_seconds=0.;stream_seconds=0.;svd_seconds=0.
    prior=json.loads(resume.read_text()) if resume else {}
    inherited=prior.get('input_sha256',{})
    reused=reusable_pairs(prior.get('pairs',[]))
    def bind(p,expected=None):
        nonlocal hash_seconds
        rel=str(p.relative_to(root))
        if rel not in hashes:
            begin=time.monotonic()
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            hashes[rel]=h.hexdigest()
            hash_seconds+=time.monotonic()-begin
            if rel in inherited and hashes[rel]!=inherited[rel]:
                raise ValueError('resumed input changed '+rel)
        if expected is not None and hashes[rel]!=expected: raise ValueError('pin mismatch '+rel)
    def read(rel):
        p=root/rel;bind(p);return json.loads(p.read_text())
    old=read('runs/track-b-k5-joined-20261108-v2/protocol.json')
    oldrows=read('runs/track-b-k5-joined-20261108-v2/results.json')['rows']
    signatures=read('runs/track-b-k5-joined-20261108-v2/tensor-signatures.json')
    quant=read('runs/track-b-k5-quantization-20261108/protocol.json')['positive']
    cat={q['model_id']:q for q in read('runs/track-a-k8-protocol-20261108/catalog/queries.json') if q['eligible_proposed_query']}
    groups=read('runs/track-b-existing-artifacts-20261108/groups.json')['groups']
    audit=read('runs/track-b-k5-control-audit-20261108/results.json')
    release=read('runs/track-a-k0-20261107/release-manifest.json')
    archive={r['path']:r['sha256'] for r in release['repositories'][0]['inventory']}
    paths={k:root/v for k,v in old['paths'].items()}
    paths.update({k:root/'lab'/k for k in quant})
    # The only temporary paths in the prior cohort are four Heretic candidates.
    for k,v in old['paths'].items():
        if v.startswith('temporary-lab/'):
            paths[k]=Path(os.environ['TEMPORARY_LAB_ROOT'])/v.removeprefix('temporary-lab/')
    anchors={r['model_id']:r['anchor'] for r in oldrows}
    anchors.update({k:next(a for a in ['olmo-1b','q25-05','q3-06','smol-360'] if k.startswith(a+'-')) for k in quant})
    assert len(paths)==76
    manifests={k:read(cat[k]['manifest']) for k in paths}
    ties={k:manifests[k]['config'].get('tie_word_embeddings',False) is True for k in paths}
    metadata={a:read(str((paths[a+'-abliteration-classic']/'abliteration.json').relative_to(root))) for a in ['olmo-1b','q25-05','q3-06','smol-360']}
    writer_sets={a:set(r['writers']) for a,r in metadata.items()}
    layers={r['anchor']:sorted({0,1,10}|set(r['proposed_first_three_active_down_layers'])) for r in audit['writer_construction_coverage'] if r['method']=='heretic'}
    fp={}
    for k in paths:
        p=root/cat[k]['row_block']['path'];bind(p,cat[k]['row_block']['sha256'])
        with safe_open(str(p),framework='numpy') as h:
            meta=h.metadata();names=json.loads(meta['names']);sha=json.loads(meta['sha256']);geometry=json.loads(meta['geometry'])
            rows=h.get_tensor('rows');shape=h.get_tensor('shape')
        dtype={t['name']:t['dtype'] for t in manifests[k]['tensors']}
        fp[k]={n:dict(hash=sha[i],geometry=geometry[i],rows=rows[i].tolist(),shape=shape[i].tolist(),dtype=dtype.get(n)) for i,n in enumerate(names)}
    indexed={}
    def index(k):
        if k in indexed:return indexed[k]
        attempt_ids.add(k);result={}
        for p in sorted(paths[k].glob('*.safetensors')):
            rel=str(p.relative_to(root)) if p.is_relative_to(root) else 'temporary-lab/'+str(p.relative_to(Path(os.environ['TEMPORARY_LAB_ROOT'])))
            expected=old['expected_files'].get(rel,archive.get('models/'+k+'/'+p.name))
            if expected is None:raise ValueError('unbound file')
            if p.is_relative_to(root):bind(p,expected)
            else:
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
                if h.hexdigest()!=expected:raise ValueError('temporary pin mismatch')
                hashes[rel]=h.hexdigest()
            with p.open('rb') as f:
                first=f.read(8);size=header_length(first);file_size=p.stat().st_size
                if 8+size>file_size:raise Unsupported('header out of file')
                entries,_=parse_header(first+f.read(size),p.name)
            for n,e in entries.items():
                if e.end>file_size or n in result:raise Unsupported('invalid tensor index')
                result[n]=(p,e)
        if not result:raise Unsupported('no weights')
        read_ids.add(k)
        indexed[k]=result;return result
    def tensor(k,n,lo=None,hi=None):
        nonlocal payload,stream_seconds
        p,e=index(k)[n];shape=e.shape;begin,end=e.start,e.end
        if e.dtype not in ('BF16','F32','F16'):raise Unsupported('unsupported floating tensor')
        if lo is not None:
            if len(shape)!=2 or not 0<=lo<hi<=shape[0]:raise Unsupported('invalid rows')
            stride=shape[1]*_DTYPE_BYTES[e.dtype];begin=e.start+lo*stride;end=e.start+hi*stride;shape=(hi-lo,shape[1])
        tick=time.monotonic();data=read_range(p,begin,end,readers,stack)
        stream_seconds+=time.monotonic()-tick
        payload+=len(data)
        return decode_plain(data,e.dtype,shape)
    def logical(k,panel):
        panel=dict(panel)
        if ties[k]:
            emb='model.embed_tokens.weight';head='lm_head.weight'
            if emb not in panel:raise Unsupported('missing tied embedding')
            if head in panel:
                a,b=panel[emb],panel[head]
                if a['shape']!=b['shape'] or a['bf16']!=b['bf16']:raise Unsupported('conflicting tied aliases')
                del panel[head]
        return panel
    def cached_nonwriter(q,c,anchor):
        a=logical(q,{n:v for n,v in signatures[q].items() if n not in writer_sets[anchor]})
        if c not in signatures:return None
        b=logical(c,{n:v for n,v in signatures[c].items() if n not in writer_sets[anchor]})
        if set(a)!=set(b):return 'cached_nonwriter_schema'
        for n in sorted(a):
            if a[n]['shape']!=b[n]['shape'] or a[n]['bf16']!=b[n]['bf16']:return 'cached_nonwriter:'+n
        return None
    def row_counterexample(q,c,anchor):
        for n,a in fp[q].items():
            if n=='lm_head.weight' or n in writer_sets[anchor] or n not in fp[c]:continue
            b=fp[c][n]
            if a['dtype']==b['dtype']=='BF16' and all(a[x]==b[x] for x in ['geometry','rows','shape']) and a['hash']!=b['hash']:
                return 'cached_BF16_row:'+n
        return None
    def nonwriter_stream(q,c,anchor):
        # Cached rows never prove equality. Complete missing q/v or uncached panels.
        a=index(q);b=index(c)
        float_types={'BF16','F32','F16'}
        an={n for n,(_,e) in a.items() if e.dtype in float_types and n not in writer_sets[anchor]}
        bn={n for n,(_,e) in b.items() if e.dtype in float_types and n not in writer_sets[anchor]}
        for k,names,idx in [(q,an,a),(c,bn,b)]:
            if ties[k]:
                emb='model.embed_tokens.weight';head='lm_head.weight'
                if emb not in names:raise Unsupported('missing effective embedding')
                if head in names:
                    if k not in signatures:
                        if idx[emb][1].shape!=idx[head][1].shape:raise Unsupported('tied shape conflict')
                        for lo in range(0,idx[emb][1].shape[0],128):
                            hi=min(lo+128,idx[emb][1].shape[0])
                            if not torch.equal(tensor(k,emb,lo,hi).float().to(torch.bfloat16),tensor(k,head,lo,hi).float().to(torch.bfloat16)):raise Unsupported('tied value conflict')
                    names.remove(head)
        if an!=bn:return 'nonwriter_schema'
        for n in sorted(an):
            if a[n][1].shape!=b[n][1].shape:return 'nonwriter_shape:'+n
            if c in signatures and n in signatures[q] and n in signatures[c]:continue
            shape=a[n][1].shape
            chunks=range(0,shape[0],128) if len(shape)==2 else [None]
            for lo in chunks:
                hi=min(lo+128,shape[0]) if lo is not None else None
                x=tensor(q,n,lo,hi).float().to(torch.bfloat16);y=tensor(c,n,lo,hi).float().to(torch.bfloat16)
                if not torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)):return 'nonwriter_bytes:'+n
        return None
    pair_rows=[];query_rows=[];inherited_pairs=0
    def checkpoint():
        atomic_json(out/'progress.json',dict(pairs=pair_rows,input_sha256=hashes,
            inherited_input_sha256=inherited,weight_identities_read=sorted(read_ids),
            index_attempts=sorted(attempt_ids),payload_bytes=payload,
            inherited_pairs=inherited_pairs,elapsed_seconds=time.monotonic()-start,
            hash_seconds=hash_seconds,stream_seconds=stream_seconds,svd_seconds=svd_seconds))
    for anchor in ['olmo-1b','q25-05','q3-06','smol-360']:
        q=anchor+'-abliteration-classic';self_group=groups[cat[q]['model_key']]
        pool=sorted(c for c in paths if anchors[c]==anchor and groups[cat[c]['model_key']]!=self_group)
        source_group=groups[cat[anchor]['model_key']]
        panel=[f'model.layers.{i}.mlp.down_proj.weight' for i in layers[anchor]]
        for c in pool:
            if time.monotonic()-start>=seconds-50:raise Unsupported('deadline approaching')
            base=dict(query=q,candidate=c,anchor=anchor)
            if (q,c) in reused:
                pair_rows.extend(reused[(q,c)]);inherited_pairs+=1;checkpoint();continue
            try:
                for k in [q,c]:
                    if any(t['dtype'].startswith('F') and t['dtype'] not in ('F16','F32') for t in manifests[k]['tensors']):
                        raise Unsupported('unsupported source floating dtype')
                reason=cached_nonwriter(q,c,anchor) or row_counterexample(q,c,anchor) or nonwriter_stream(q,c,anchor)
                if reason:
                    for model in ['fp32_input','bf16_input']:pair_rows.append(dict(**base,model=model,state='excluded',reason=reason))
                    checkpoint()
                    continue
                blocks=[];bounds={m:[] for m in ['fp32_input','bf16_input']};rounding=[]
                for n in panel:
                    shape=index(q)[n][1].shape
                    if index(c)[n][1].shape!=shape:raise Unsupported('writer shape mismatch')
                    lo=(shape[0]-min(256,shape[0]))//2;hi=lo+min(256,shape[0])
                    # Existing bounds choose a centered interval with the shared helper.
                    from evidence.sketch import row_block_bounds
                    lo,hi=row_block_bounds(shape[0],256)
                    cq=tensor(q,n,lo,hi);w=tensor(c,n,lo,hi).float()
                    if cq.dtype!=torch.bfloat16 or not torch.isfinite(cq).all() or not torch.isfinite(w).all():raise Unsupported('unsupported writer values')
                    delta=cq.double()-w.double();blocks.append((cq,w,delta));rounding.append(delta_error(cq,w))
                    for model in bounds:
                        p=w if model=='fp32_input' else w.to(torch.bfloat16).float()
                        bounds[model].append(budget(w,p,cq))
                d=torch.cat([x[2] for x in blocks],dim=1)
                tick=time.monotonic();u,s,vh=svd_vectors(d,device)
                svd_seconds+=time.monotonic()-tick
                lower=certificate(d,u[:,:2],vh[:2].T)
                de=sqrt(sum_up(mul(x,x) for x in rounding));lower=max(0.,down(lower-de))
                right=s[0]*vh[0];offset=0;upper=[];eligible=[];informativeness={m:[] for m in bounds}
                for i,(cq,w,delta) in enumerate(blocks):
                    width=delta.shape[1];r=right[offset:offset+width];offset+=width
                    upper.append(add(witness_upper(delta,u[:,0],r),rounding[i]))
                    dnlo=max(0.,down(norms(delta)[0]-rounding[i]));wnhi=norms(w)[1]
                    eligible.append(wnhi>0 and down(dnlo/wnhi)>=.003)
                    for model in bounds:informativeness[model].append(bounds[model][i]<dnlo)
                for model,bb in bounds.items():
                    aggregate=sqrt(sum_up(mul(x,x) for x in bb))
                    if lower>aggregate:state,reason='excluded','certified_shared_writer_residual'
                    elif all(eligible) and all(informativeness[model]) and all(r<=b for r,b in zip(upper,bb)):state,reason='supported','joint_conditional_evidence'
                    else:state,reason='unresolved','no_complete_informative_shared_witness'
                    pair_rows.append(dict(**base,model=model,state=state,reason=reason,lower=lower,aggregate_budget=aggregate,block_budgets=bb,witness_upper=upper,above_floor=eligible,informative=informativeness[model]))
            except (Unsupported,KeyError,OSError,AssertionError) as exc:
                for model in ['fp32_input','bf16_input']:pair_rows.append(dict(**base,model=model,state='unresolved',reason=str(exc)))
            checkpoint()
            print(q,c,pair_rows[-1]['state'],pair_rows[-1]['reason'],flush=True)
        for model in ['fp32_input','bf16_input']:
            for variant in ['present','removed']:
                rows=[r for r in pair_rows if r['query']==q and r['model']==model and (variant=='present' or groups[cat[r['candidate']]['model_key']]!=source_group)]
                support=[r['candidate'] for r in rows if r['state']=='supported'];unresolved=[r['candidate'] for r in rows if r['state']=='unresolved']
                answer=support[0] if len(support)==1 and not unresolved else None
                query_rows.append(dict(query=q,model=model,variant=variant,candidates=len(rows),groups=len({groups[cat[r['candidate']]['model_key']] for r in rows}),support=support,unresolved=unresolved,answer=answer,wrong=answer is not None and (variant=='removed' or answer!=anchor)))
    assert len(query_rows)==16
    assert len(pair_rows)==144 and len({(r['query'],r['candidate']) for r in pair_rows})==72
    assert inherited_pairs==len(reused)
    atomic_json(out/'results.json',dict(pairs=pair_rows,queries=query_rows,input_sha256=hashes,inherited_input_sha256=inherited,inherited_pairs=inherited_pairs,index_attempts=sorted(attempt_ids),weight_identities_read=sorted(read_ids),payload_bytes=payload,elapsed_seconds=time.monotonic()-start,hash_seconds=hash_seconds,stream_seconds=stream_seconds,svd_seconds=svd_seconds,svd_device=device,scope='conditional exposed development, no final rule adoption'))
    print('COMPLETE',len(pair_rows),'pair-hypothesis rows',len(read_ids),'weight identities',flush=True)


def sum_up(values):
    result=0.
    for value in values:result=add(result,value)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['validate','run'])
    parser.add_argument('--resume',type=Path);parser.add_argument('--out',default='runs/track-b-joint-source-20261109-v2')
    parser.add_argument('--seconds',type=int,default=900);parser.add_argument('--device',default='cpu',choices=['cpu','cuda:0']);args=parser.parse_args()
    torch.set_num_threads(1);root=Path(os.environ['ZOO_DATA']).resolve();repo=Path(__file__).resolve().parents[1]
    out=root/args.out;out.mkdir(exist_ok=True)
    p=repo/'docs/steering/K5-joint-source-protocol.md';driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    continuation=repo/'docs/steering/K5-joint-source-continuation.md'
    provenance=resume_provenance(args.resume) if args.resume else None
    if args.resume:
        original=json.loads((args.resume.parent/'protocol.json').read_text())
        assert original['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    if not (out/'protocol.json').exists():atomic_json(out/'protocol.json',dict(text=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),driver_sha256=driver,continuation=continuation.read_text() if args.resume else None,resume=str(args.resume) if args.resume else None,resume_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,resume_provenance=provenance,seconds=args.seconds,device=args.device))
    assert json.loads((out/'protocol.json').read_text())['driver_sha256']==driver
    frozen=json.loads((out/'protocol.json').read_text())
    assert (args.seconds,args.device)==(frozen['seconds'],frozen['device'])
    assert bool(args.resume)==bool(frozen['resume'])
    assert provenance==frozen['resume_provenance']
    if args.resume:assert hashlib.sha256(args.resume.read_bytes()).hexdigest()==frozen['resume_sha256']
    if args.stage=='validate':numerical_check(out,args.device)
    else:
        try:
            with ExitStack() as stack:run(root,repo,out,stack,args.resume,args.seconds,args.device)
        except Exception as exc:
            atomic_json(out/'failure.json',dict(type=type(exc).__name__,reason=str(exc)))
            raise


if __name__=='__main__':main()
