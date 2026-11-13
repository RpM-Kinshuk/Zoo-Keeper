import torch


def test_public_factors_admit_fp32_merge_and_reject_close_wrong_base():
    from scripts.documented_lora import check_band
    g = torch.Generator().manual_seed(20261028)
    parent = torch.randn(23, 37, generator=g) / 10
    a = torch.randn(16, 37, generator=g) / 100
    b = torch.randn(23, 16, generator=g) / 100
    child = parent + (b @ a) * 2
    result = check_band(parent, child, a, b)
    assert result['supported'] and result['violations'] == 0
    assert result['exact_mismatches'] == 0
    assert result['after_sse'] < result['before_sse']
    assert check_band(parent + 1e-5, child, a, b)['violations'] > 0


def test_numerical_envelope_accepts_other_summation_order_and_cancellation():
    from scripts.documented_lora import check_band
    g = torch.Generator().manual_seed(7)
    a = torch.randn(16, 7, generator=g)
    b = torch.randn(5, 16, generator=g)
    delta = sum((b[:, j:j+1] * a[j:j+1]) for j in reversed(range(16))) * 2
    parent = -((b @ a) * 2)
    child = parent + delta
    result = check_band(parent, child, a, b)
    assert result['supported'] and result['violations'] == 0


def test_unsupported_ranges_and_untouched_copy_are_not_rescued():
    from scripts.documented_lora import check_band
    p = torch.ones(2, 3)
    a = torch.ones(16, 3); b = torch.ones(2, 16)
    assert not check_band(p, p, a.double(), b)['supported']
    a[0, 0] = 2.**-60
    assert not check_band(p, p, a, b)['supported']
    assert check_band(p, p)['violations'] == 0
    wrong = p.clone(); wrong[0, 0] = torch.nextafter(wrong[0, 0], torch.tensor(float('inf')))
    assert check_band(p, wrong)['violations'] == 1
