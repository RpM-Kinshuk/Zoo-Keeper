"""Validate a conditional rank-one arithmetic/storage bound; no detector adoption."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from lab.abliteration import classic_edit, _unit
from lab.artifacts import atomic_json
from evidence.sketch import row_block_bounds
from evidence.safetensors_header import header_length, parse_header, decode_plain, _DTYPE_BYTES

U = 2.**-24
ETA = 2.**-150
ALPHA = U/(1-U)


def norm(x):
    return float(x.double().norm())


def radius(c):
    assert c.dtype == torch.bfloat16 and torch.isfinite(c).all()
    hi = torch.nextafter(c, torch.full_like(c, math.inf)).double()
    lo = torch.nextafter(c, torch.full_like(c, -math.inf)).double()
    return torch.maximum(hi-c.double(), c.double()-lo)/2


def allowance(w, p, c):
    radii = radius(c)
    if not bool(torch.isfinite(radii).all()):
        raise ValueError('unsupported_rounding_cell')
    q = norm(radii)
    arithmetic = ALPHA*norm(p)+ALPHA*(2+ALPHA)*(norm(c)+q)+(2+3*ALPHA+ALPHA**2)*ETA*math.sqrt(c.numel())
    h = norm(p.double()-w.double())
    if not all(math.isfinite(v) for v in [q,arithmetic,h]):
        raise ValueError('unsupported_nonfinite_allowance')
    return dict(storage=q, arithmetic=arithmetic, input_conversion=h, total=h+q+arithmetic)


def guard(n, scale):
    assert math.isfinite(scale) and scale>=0
    k = (10*n+100)*2.**-53
    assert k < 1
    return math.nextafter(k/(1-k)*scale, math.inf)


def operator(p, direction):
    r = _unit(direction, p.shape[0])
    v = r @ p
    assert torch.isfinite(v).all()
    x = classic_edit({'writer':p},{0:direction},{'writer':dict(layer=0,residual_axis=0)})['writer']
    assert torch.equal(x,p-r[:,None]*v[None,:])
    return x,r,v


def stage1(out):
    assert not (out/'stage1.json').exists()
    bits = torch.arange(65536,dtype=torch.int32).to(torch.int16)
    z = bits.view(torch.bfloat16)
    z = z[torch.isfinite(z)]
    b = radius(z)
    z = z[torch.isfinite(b)]
    up = torch.nextafter(z,torch.full_like(z,math.inf))
    mid = ((z.double()+up.double())/2).float()
    probes = torch.cat([z.float(),mid,torch.nextafter(mid,torch.full_like(mid,math.inf)),
                        torch.nextafter(mid,torch.full_like(mid,-math.inf))])
    c = probes.to(torch.bfloat16)
    finite = torch.isfinite(c)
    probes,c = probes[finite],c[finite]
    b = radius(c)
    supported = torch.isfinite(b)
    error = (probes.double()-c.double()).abs()
    assert bool((error[supported]<=b[supported]).all()), 'rounding-cell counterexample'
    rejected_endpoints = 0
    for sign in [-1,1]:
        extreme = torch.tensor([[sign*torch.finfo(torch.bfloat16).max]],dtype=torch.bfloat16)
        try:
            allowance(extreme.float(),extreme.float(),extreme)
        except ValueError as exc:
            assert str(exc)=='unsupported_rounding_cell'
            rejected_endpoints += 1
        else:
            raise AssertionError('unsupported endpoint accepted')
    generator = torch.Generator().manual_seed(20261028)
    cases = []
    for shape in [(17,31),(64,128),(256,384)]:
        d = torch.randn(shape[0],generator=generator)
        for scale in [0.,2.**-149,2.**-120,2.**-60,1.,2.**60]:
            for cancellation in [False,True]:
                r = _unit(d,shape[0])
                p = ((r[:,None]*torch.randn(shape[1],generator=generator)[None,:]) if cancellation else torch.randn(shape,generator=generator))*scale
                x,r,v = operator(p,d)
                assert torch.isfinite(x).all()
                c = x.to(torch.bfloat16)
                a = allowance(p,p,c)
                oracle = -r.double()[:,None]*v.double()[None,:]
                arithmetic_error = norm(x.double()-p.double()-oracle)
                exact_a = ALPHA*norm(p)+ALPHA*(2+ALPHA)*norm(x)+(2+3*ALPHA+ALPHA**2)*ETA*math.sqrt(p.numel())
                arithmetic_guard = guard(p.numel(),norm(p)+norm(x)+norm(oracle)+exact_a)
                assert arithmetic_error<=exact_a+arithmetic_guard, 'FP32 arithmetic-bound counterexample'
                actual = norm(c.double()-p.double()-oracle)
                g = guard(p.numel(),norm(p)+norm(c)+norm(oracle)+a['total'])
                assert actual<=a['total']+g, 'operator-bound counterexample'
                cases.append(dict(shape=shape,scale=scale,cancellation=cancellation,
                    arithmetic_error=arithmetic_error,arithmetic_bound=exact_a,arithmetic_guard=arithmetic_guard,
                    actual_error=actual,bound=a['total'],evaluation_guard=g))
    result = dict(passed=True,supported_bf16_values=len(z),supported_probes=int(supported.sum()),
        unsupported_probe_endpoints=int((~supported).sum()),operator_cases=len(cases),cases=cases,
        unsupported_extreme_cells_rejected=rejected_endpoints,separate_arithmetic_checks=len(cases),
        largest_supported_rounding_ratio=float((error[supported]/b[supported]).max()),
        scope='conditional numerical validation; no historical operator or detector verdict')
    atomic_json(out/'stage1.json',result)
    print('stage1 PASS',len(z),'BF16 values,',int(supported.sum()),'probes,',len(cases),'operator cases',flush=True)


def stage2(root,repo,out):
    assert json.loads((out/'stage1.json').read_text())['passed']
    assert (out/'proof-review.md').exists(), 'fresh conditional proof review required'
    assert not (out/'stage2.json').exists()
    start = time.monotonic()
    old = json.loads((root/'runs/track-b-k5-joined-20261108-v2/protocol.json').read_text())
    release_path = root/'runs/track-a-k0-20261107/release-manifest.json'
    release = json.loads(release_path.read_text())
    pins = {r['path']:r['sha256'] for r in release['repositories'][0]['inventory']}
    hashes = {}

    def bind(p,expected=None):
        rel = str(p.relative_to(root))
        if rel not in hashes:
            print('verifying',rel,flush=True)
            h = hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):
                    h.update(b)
            hashes[rel] = h.hexdigest()
            print('verified',rel,flush=True)
        if expected is not None:
            assert hashes[rel]==expected, 'pin mismatch '+rel

    def opened(stack,path):
        index = {}
        for p in sorted(path.glob('*.safetensors')):
            bind(p,old['expected_files'][str(p.relative_to(root))])
            with p.open('rb') as f:
                first = f.read(8)
                size = header_length(first)
                file_size = p.stat().st_size
                assert 8+size<=file_size
                entries,_ = parse_header(first+f.read(size),p.name)
            assert all(e.end<=file_size for e in entries.values())
            for n,e in entries.items():
                assert n not in index
                index[n] = (p,e)
        return index

    def matrix(index,name,rows=None):
        p,e = index[name]
        assert len(e.shape)==2 and e.dtype in ('BF16','F32')
        shape = e.shape
        start,end = e.start,e.end
        if rows is not None:
            lo,hi = rows
            assert 0<=lo<hi<=shape[0]
            stride = shape[1]*_DTYPE_BYTES[e.dtype]
            start,end = e.start+lo*stride,e.start+hi*stride
            shape = (hi-lo,shape[1])
        assert e.start<=start<=end<=e.end<=p.stat().st_size
        with p.open('rb') as f:
            f.seek(start)
            data = f.read(end-start)
        return decode_plain(data,e.dtype,shape)

    for p in [repo/'scripts/k5_classic_noise.py',repo/'docs/steering/K5-classic-noise-validation.md',release_path]:
        bind(p)
    rows = []
    identities = set()
    for anchor in ['olmo-1b','q25-05','q3-06','smol-360']:
        positive = anchor+'-abliteration-classic'
        mids = [positive,anchor+'-sft-s0',anchor+'-dpo-s0',anchor+'-lora-dialogue-r8']
        identities.update(mids+[anchor])
        directory = root/old['paths'][positive]
        metadata = directory/'abliteration.json'
        bind(metadata,pins['models/'+positive+'/abliteration.json'])
        record = json.loads(metadata.read_text())
        assert record['shared_direction_all_writers'] is True
        dp = directory/'diagnostics/directions.safetensors'
        bind(dp,pins['models/'+positive+'/diagnostics/directions.safetensors'])
        with safe_open(str(dp),framework='pt',device='cpu') as f:
            assert len(list(f.keys()))==1
            direction = f.get_tensor(list(f.keys())[0])
        with ExitStack() as stack:
            parents = opened(stack,root/old['paths'][anchor])
            children = {mid:opened(stack,root/old['paths'][mid]) for mid in mids}
            chosen = sorted(n for n in children[positive] if n.endswith('down_proj.weight'))[:3]
            assert len(chosen)==3
            for name in chosen:
                assert time.monotonic()-start<6600
                print('matrix',anchor,name,flush=True)
                w = matrix(parents,name).float()
                assert w.ndim==2 and len(direction)==w.shape[0]
                lo,hi = row_block_bounds(w.shape[0],256)
                # Check byte/coordinate compatibility with the established reader.
                parent_file,_ = parents[name]
                with safe_open(str(parent_file),framework='pt',device='cpu') as check:
                    native = check.get_slice(name)[lo:lo+1,:].float()
                    assert torch.equal(w[lo:lo+1],native)
                for model in ['fp32_input','bf16_input']:
                    p = w if model=='fp32_input' else w.to(torch.bfloat16).float()
                    x,r,v = operator(p,direction)
                    simulated = x[lo:hi].to(torch.bfloat16)
                    oracle = -r[lo:hi].double()[:,None]*v.double()[None,:]
                    arithmetic_error = norm(x[lo:hi].double()-p[lo:hi].double()-oracle)
                    exact_a = ALPHA*norm(p[lo:hi])+ALPHA*(2+ALPHA)*norm(x[lo:hi])+(2+3*ALPHA+ALPHA**2)*ETA*math.sqrt(simulated.numel())
                    arithmetic_guard = guard(simulated.numel(),norm(p[lo:hi])+norm(x[lo:hi])+norm(oracle)+exact_a)
                    assert arithmetic_error<=exact_a+arithmetic_guard, 'real-matrix FP32 arithmetic bound fails'
                    bounds = allowance(w[lo:hi],p[lo:hi],simulated)
                    error = norm(simulated.double()-w[lo:hi].double()-oracle)
                    g = guard(simulated.numel(),norm(w[lo:hi])+norm(simulated)+norm(oracle)+bounds['total'])
                    assert error<=bounds['total']+g, 'real-matrix oracle violates bound'
                    rows.append(dict(anchor=anchor,kind='oracle_roundtrip',tensor=name,model=model,
                        actual_error=error,bound=bounds,evaluation_guard=g,
                        arithmetic_error=arithmetic_error,arithmetic_bound=exact_a,arithmetic_guard=arithmetic_guard))
                    for mid in mids:
                        assert children[mid][name][1].shape==tuple(w.shape), 'child matrix shape mismatch'
                        c = matrix(children[mid],name,(lo,hi))
                        assert c.dtype==torch.bfloat16 and torch.isfinite(c).all()
                        delta = c.double()-w[lo:hi].double()
                        a = allowance(w[lo:hi],p[lo:hi],c)
                        dn = norm(delta)
                        relative = dn/norm(w[lo:hi]) if norm(w[lo:hi]) else None
                        # An explicit computed rank-one witness gives an upper residual;
                        # no claim that its numerical SVD is a certified optimum.
                        u,s,vh = torch.linalg.svd(delta,full_matrices=False)
                        witness = u[:,:1]*(s[:1,None]@vh[:1,:])
                        residual = norm(delta-witness)
                        numerical = guard(delta.numel(),dn+norm(witness)+a['total'])
                        informative = a['total']+numerical < dn-numerical
                        rows.append(dict(anchor=anchor,kind='observed',model_id=mid,
                            positive=mid==positive,tensor=name,rows=[lo,hi],model=model,
                            delta_norm=dn,relative_update=relative,bound=a,
                            rank1_witness_residual=residual,numerical_guard=numerical,
                            informative=informative,above_update_floor=relative is not None and relative>=.003,
                            witness_within_bound_upper=residual+numerical<=a['total'],
                            status='conditional_diagnostic_only'))
                atomic_json(out/'progress.json',dict(rows=rows,input_sha256=hashes,elapsed_seconds=time.monotonic()-start))
        atomic_json(out/'progress.json',dict(rows=rows,input_sha256=hashes,elapsed_seconds=time.monotonic()-start))
        print('stage2 measured',anchor,flush=True)
    assert len(identities)==20
    atomic_json(out/'stage2.json',dict(rows=rows,input_sha256=hashes,weight_identities=len(identities),
        elapsed_seconds=time.monotonic()-start,no_final_rule_adopted=True))
    print('stage2 complete; conditional diagnostics only',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['stage1','stage2'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(os.environ['ZOO_DATA']).resolve()
    repo = Path(__file__).resolve().parents[1]
    out = root/'runs/track-b-classic-noise-20261109-v4'
    out.mkdir(exist_ok=True)
    protocol = repo/'docs/steering/K5-classic-noise-validation.md'
    if not (out/'protocol.json').exists():
        atomic_json(out/'protocol.json',dict(text=protocol.read_text(),
            sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    frozen = json.loads((out/'protocol.json').read_text())
    assert frozen['driver_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.stage=='stage1':
        stage1(out)
    else:
        stage2(root,repo,out)


if __name__=='__main__':
    main()
