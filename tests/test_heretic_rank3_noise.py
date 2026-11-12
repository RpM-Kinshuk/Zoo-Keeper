"""The bound must include both BF16 roundings and reject unsupported arithmetic."""
import pytest
import torch


def module():
    from scripts import heretic_rank3_noise
    return heretic_rank3_noise


def test_known_factor_bound_covers_update_rounding_and_cancellation():
    m=module()
    w=torch.tensor([[1.,-1.],[.5,2.]],dtype=torch.float32)
    b=torch.tensor([[1.,1.,1.],[1.,-1.,1.]],dtype=torch.float32)
    a=torch.tensor([[.00391,.1],[-.00390,-.1],[.00197,.003]],dtype=torch.float32)
    r=m.evaluate(w,b,a)
    assert r['actual_error']<=r['bound_total']+r['evaluation_guard']
    assert r['update_rounding']>0 and r['output_rounding']>0
    assert r['factor_product_error']<=r['factor_product_bound']+r['evaluation_guard']


def test_zero_update_is_uninformative_and_overflow_is_unsupported():
    m=module();w=torch.ones(2,2);b=torch.zeros(2,3);a=torch.ones(3,2)
    r=m.evaluate(w,b,a)
    assert not r['informative']
    with pytest.raises(ValueError):m.evaluate(w,torch.full((2,3),1e30),torch.full((3,2),1e30))
    with pytest.raises(ValueError):m.evaluate(w.double(),b,a)


def test_product_guard_remains_tight_when_base_dwarfs_subnormal_update():
    m=module();torch.set_flush_denormal(False)
    r=m.evaluate(torch.ones(2,2),torch.ones(2,3),torch.full((3,2),2.**-140))
    assert r['product_evaluation_guard']<1e-45
    assert r['factor_product_error']<=r['factor_product_bound']+r['product_evaluation_guard']


def test_flushed_subnormal_arithmetic_is_rejected():
    m=module()
    torch.set_flush_denormal(True)
    try:
        with pytest.raises(ValueError,match='arithmetic'):
            m.evaluate(torch.ones(2,2),torch.ones(2,3),torch.ones(3,2))
    finally:
        torch.set_flush_denormal(False)
