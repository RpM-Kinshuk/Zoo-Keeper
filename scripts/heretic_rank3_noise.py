"""Synthetic validation only: known rank3 factors, FP32 product and BF16 merge."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from scripts.k5_classic_noise import norm, radius, guard
from lab.artifacts import atomic_json


def evaluate(w, b, a):
    if any(z.device.type!='cpu' for z in (w,b,a)):
        raise ValueError('unsupported_arithmetic_device')
    probe=torch.tensor(2.**-149,dtype=torch.float32)*torch.tensor(1.,dtype=torch.float32)
    if float(probe)!=2.**-149:
        raise ValueError('unsupported_arithmetic_flush_denormal')
    if w.dtype not in (torch.float32,torch.bfloat16) or b.dtype != torch.float32 or a.dtype != torch.float32:
        raise ValueError('unsupported_dtype')
    if w.ndim!=2 or b.shape!=(w.shape[0],3) or a.shape!=(3,w.shape[1]):
        raise ValueError('unsupported_rank_or_shape')
    if any(not bool(torch.isfinite(x).all()) for x in (w,b,a)):
        raise ValueError('nonfinite_input')
    p=w.to(torch.bfloat16).float()
    t=b@a
    u=t.to(torch.bfloat16)
    x=p+u.float()
    c=x.to(torch.bfloat16)
    if any(not bool(torch.isfinite(z).all()) for z in (p,t,u,x,c)):
        raise ValueError('overflow_or_nonfinite')
    ru,rc=radius(u),radius(c)
    if not bool(torch.isfinite(ru).all() and torch.isfinite(rc).all()):
        raise ValueError('unsupported_rounding_cell')
    unit,eta=2.**-24,2.**-150
    gamma=6*unit/(1-6*unit)
    oracle=b.double()@a.double()
    product_magnitude=b.double().abs()@a.double().abs()
    gemm=gamma*product_magnitude+6*eta/(1-6*unit)
    product_guard=guard(3*w.numel(),norm(product_magnitude)+norm(oracle)+norm(t))
    q_u,q_c=norm(ru),norm(rc)
    # Separate FP32 addition before BF16 output; do not assume BF16 sums are exact.
    add=unit/(1-unit)*(norm(c)+q_c)+eta/(1-unit)*math.sqrt(w.numel())
    h=norm(p.double()-w.double())
    bound=h+norm(gemm)+q_u+add+q_c
    actual=norm(c.double()-w.double()-oracle)
    numerical=guard(w.numel(),norm(c)+norm(w)+norm(oracle)+bound)
    d=norm(c.double()-w.double())
    return dict(actual_error=actual,bound_total=bound,evaluation_guard=numerical,
                input_conversion=h,factor_product_bound=norm(gemm),
                factor_product_error=norm(t.double()-oracle),product_evaluation_guard=product_guard,update_rounding=q_u,
                addition_bound=add,output_rounding=q_c,delta_norm=d,
                informative=bound+numerical<d-numerical,
                bound_holds=actual<=bound+numerical,
                factor_product_holds=norm(t.double()-oracle)<=norm(gemm)+product_guard)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out
    if (out/'results.json').exists():
        raise ValueError('refuse_to_overwrite_completed_validation')
    if not (out/'proof-review.md').exists():
        raise ValueError('independent_proof_review_required')
    torch.set_num_threads(1)
    torch.set_flush_denormal(False)
    gen=torch.Generator().manual_seed(20261028)
    start=time.monotonic();rows=[]
    for shape in [(17,31),(64,128),(256,384)]:
        for scale in [0.,2.**-140,2.**-60,1.,2.**60]:
            for kind in ['normal','cancellation','exponent_boundary']:
                b=torch.randn(shape[0],3,generator=gen)
                a=torch.randn(3,shape[1],generator=gen)*scale
                if kind=='cancellation':
                    b[:,1]=b[:,0];a[1]=-a[0];a[2]*=2.**-12
                w=torch.randn(shape,generator=gen)
                if kind=='exponent_boundary':
                    w=torch.sign(w)*torch.pow(torch.tensor(2.),torch.randint(-20,20,shape,generator=gen))
                result=evaluate(w,b,a)
                assert result['bound_holds'] and result['factor_product_holds'],'supported_bound_counterexample'
                rows.append(dict(shape=shape,scale=scale,kind=kind,**result))
    rejected=0
    for w,b,a in [(torch.ones(2,2),torch.full((2,3),1e30),torch.full((3,2),1e30)),
                  (torch.full((2,2),torch.finfo(torch.bfloat16).max),torch.zeros(2,3),torch.ones(3,2))]:
        try:evaluate(w,b,a)
        except ValueError:rejected+=1
        else:raise AssertionError('unsupported_extreme_accepted')
    result=dict(passed=True,cases=len(rows),informative=sum(r['informative'] for r in rows),
                unsupported_rejected=rejected,rows=rows,elapsed_seconds=time.monotonic()-start,
                input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),out/'PROTOCOL.md',out/'proof-review.md',Path(__file__).with_name('k5_classic_noise.py')]},
                scope='Known-factor synthetic lemma only. No endpoint-only/history/edit/source detector or weight reads.')
    atomic_json(out/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','input_sha256']}),flush=True)


if __name__=='__main__':main()
