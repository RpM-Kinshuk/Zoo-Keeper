import torch
import pytest


def test_replay_uses_recorded_scale_storage_and_rounding_cell_ambiguity():
    from scripts.documented_fp8_case import compare_band,decide
    value=torch.zeros(128,128);value[0,0]=448;value[0,1]=1
    codes=value.to(torch.float8_e4m3fn);scales=torch.ones(1,1,dtype=torch.bfloat16)
    r=compare_band(value,codes,scales)
    assert r['observed_code_mismatches']==r['replay_code_mismatches']==r['replay_scale_mismatches']==0
    close=value.clone();close[0,1]=1.001
    assert compare_band(close,codes,scales)['observed_code_mismatches']==0
    assert decide({'source':True,'relative':True})['answer'] is None
    assert decide({'source':True,'relative':False})['answer']=='source'
    assert decide({'relative':False})['answer'] is None
    close[0,1]=2
    assert compare_band(close,codes,scales)['observed_code_mismatches']==1


def test_invalid_scale_or_incomplete_tiles_do_not_become_matches():
    from scripts.documented_fp8_case import compare_band
    with pytest.raises(ValueError):compare_band(torch.zeros(127,128),torch.zeros(127,128).to(torch.float8_e4m3fn),torch.ones(1,1))
    with pytest.raises(ValueError):compare_band(torch.zeros(128,128),torch.zeros(128,128).to(torch.float8_e4m3fn),torch.zeros(1,1))
