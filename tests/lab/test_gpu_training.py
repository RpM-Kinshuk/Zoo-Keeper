import copy
import json

import pytest

from lab.launches import file_hash
from lab.scripts.gpu_training import eligible_gpu, prepare_bundle


def test_gpu_limits_are_both_strict_and_require_memory_headroom():
    row = dict(utilization_percent=79, memory_used_mib=1000, memory_total_mib=100000)
    assert eligible_gpu(row)
    assert not eligible_gpu({**row, 'utilization_percent': 80})
    assert not eligible_gpu({**row, 'memory_used_mib': 80000})
    assert not eligible_gpu({**row, 'memory_used_mib': 77000})  # <80%, <24GiB free


def test_gpu_bundle_preserves_frozen_inputs_and_binds_dependency_hashes(tmp_path):
    source = tmp_path/'cpu'; (source/'configs').mkdir(parents=True)
    original = dict(trainer='sft', recipe=dict(model_id='smol-135-launch-sft-s0', device='cpu',
        parents=['smol-135'], parent_identities={'smol-135':'hub@commit'}, steps=256, training_seed=12),
        inputs={'model_path':'lab/inputs/base','tokens_sha256':'frozen'}, selections={'training':['a']})
    sft = source/'configs'/'sft.json'; sft.write_text(json.dumps(original))
    dpo = copy.deepcopy(original); dpo['trainer']='dpo'
    dpo['recipe'].update(model_id='smol-135-launch-dpo-s0', parents=['smol-135-launch-sft-s0'], parent_identities={})
    dpo['inputs']['model_path']='lab/smol-135-launch-sft-s0'
    (source/'configs'/'dpo.json').write_text(json.dumps(dpo))
    before = {p.name:p.read_bytes() for p in (source/'configs').iterdir()}
    out=tmp_path/'gpu'; manifest=prepare_bundle(source,out)
    prepared=json.loads((out/'configs'/'smol-135-gpu-v1-sft-s0.json').read_text())
    assert prepared['inputs']==original['inputs']
    assert prepared['selections']==original['selections']
    assert prepared['recipe']['steps']==256 and prepared['recipe']['training_seed']==12
    assert prepared['recipe']['device']=='cuda:0'
    prepared_dpo=json.loads((out/'configs'/'smol-135-gpu-v1-dpo-s0.json').read_text())
    assert prepared_dpo['recipe']['parents']==['smol-135-gpu-v1-sft-s0']
    assert prepared_dpo['inputs']['model_path']=='lab/smol-135-gpu-v1-sft-s0'
    assert {p.name:p.read_bytes() for p in (source/'configs').iterdir()}==before
    record=next(r for r in manifest['configs'] if r['model_id'].endswith('sft-s0'))
    assert record['source_sha256']==file_hash(sft)
    assert record['config_sha256']==file_hash(out/record['config_path'])
    assert prepare_bundle(source,out)==manifest
    sft.write_text(sft.read_text()+' ')
    with pytest.raises(FileExistsError): prepare_bundle(source,out)
