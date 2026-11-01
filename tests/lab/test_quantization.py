"""Quantizer arithmetic, pinned source rejection, and offline checkpoint behavior."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from lab.artifacts import atomic_json, content_hash, write_checkpoint
from lab.quantization import simulate_quantization, run_quantization
from lab.registry import expand_inventory


def test_hand_computed_row_zero_and_round_even():
    value = torch.tensor([[7., -7., 0.5, 1.5, 2.5, -2.5], [0., 0., 0., 0., 0., 0.]])
    actual = simulate_quantization(value, 4)
    torch.testing.assert_close(actual, torch.tensor([[7., -7., 0., 2., 2., -2.], [0.]*6]))
    torch.testing.assert_close(value[0], torch.tensor([7., -7., .5, 1.5, 2.5, -2.5]))


@pytest.mark.parametrize('bits', [0, 1, 3, 16, True, 4.0])
def test_invalid_bits(bits):
    with pytest.raises(ValueError, match='bits'):
        simulate_quantization(torch.ones(2, 2), bits)


@pytest.mark.parametrize('value', [torch.ones(3), torch.ones(2, 2, 2), torch.ones(2, 2, dtype=torch.int64)])
def test_nonmatrix_rejected_by_matrix_api(value):
    with pytest.raises(ValueError, match='matrix'):
        simulate_quantization(value, 4)


def fixture(root, local=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(9)
    model = LlamaForCausalLM(LlamaConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=16, tie_word_embeddings=True))
    parent = 'q25-05-dpo-s0' if local else 'olmo-1b'
    path = root / parent
    model.save_pretrained(path, safe_serialization=True)
    # A floating vector, nonmatrix float, and integer tensor exercise preservation.
    state = load_file(str(path / 'model.safetensors'))
    state.update(vector=torch.tensor([.123456, 1.]), cube=torch.ones(1, 2, 3)*.123456,
                 counter=torch.tensor([1, 2], dtype=torch.int64))
    save_file(state, str(path / 'model.safetensors'), metadata={'format':'pt'})
    identity = 'lab/'+parent+'@local:'+'a'*64 if local else 'org/olmo@'+'b'*40
    weights = {'model.safetensors': hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest()}
    manifest = dict(repo_id='lab/'+parent if local else 'org/olmo', revision='local:'+'a'*64 if local else 'b'*40,
                    files=[dict(filename=k, sha256=v) for k,v in weights.items()])
    pin = dict(path=parent, model_key=identity, config_sha256=content_hash(json.loads((path/'config.json').read_text())),
               config_file_sha256=hashlib.sha256((path/'config.json').read_bytes()).hexdigest(), weight_sha256=weights,
               manifest=manifest)
    recipe = dict(model_id='quantized', edge_type='quantization_sim', parents=[parent],
                  parent_identities={parent:identity}, bits=4, inputs={parent:pin})
    return recipe, state


@pytest.mark.parametrize('local', [True, False])
def test_safe_reload_preserves_tied_effective_weights_and_lineage(tmp_path, local):
    from transformers import AutoModelForCausalLM
    recipe, before = fixture(tmp_path, local)
    target = write_checkpoint('quantized', tmp_path, recipe,
        lambda stage: run_quantization(recipe, {'root':tmp_path}, stage))
    actual = load_file(str(target/'model.safetensors'))
    assert all(t.dtype == torch.bfloat16 for t in actual.values() if t.is_floating_point())
    assert torch.equal(actual['vector'], before['vector'].bfloat16())
    assert torch.equal(actual['cube'], before['cube'].bfloat16())
    assert torch.equal(actual['counter'], before['counter'])
    loaded = AutoModelForCausalLM.from_pretrained(target, local_files_only=True, use_safetensors=True,
                                                 trust_remote_code=False, dtype=torch.bfloat16)
    assert loaded.model.embed_tokens.weight.data_ptr() == loaded.lm_head.weight.data_ptr()
    assert torch.equal(loaded.model.embed_tokens.weight, actual['model.embed_tokens.weight'])
    lineage = json.loads((target/'lineage.json').read_text())
    assert lineage == dict(name='lab/quantized', bases=['lab/q25-05-dpo-s0' if local else 'org/olmo@'+'b'*40], relation='quantization_sim')
    metrics = json.loads((target/'quantization.json').read_text())
    assert metrics['relative_weight_change'] > 0
    assert write_checkpoint('quantized', tmp_path, recipe, lambda stage: pytest.fail('resume wrote'), resume=True) == target
    with pytest.raises(ValueError, match='resume mismatch'):
        write_checkpoint('quantized', tmp_path, {**recipe, 'bits':8}, lambda stage: None, resume=True)


@pytest.mark.parametrize('attack', ['weight', 'manifest', 'identity', 'path', 'index', 'alias'])
def test_reject_changed_or_unbound_sources_atomically(tmp_path, attack):
    recipe, before = fixture(tmp_path)
    parent = recipe['parents'][0]
    pin = recipe['inputs'][parent]
    if attack == 'weight':
        (tmp_path/parent/'model.safetensors').write_bytes(b'changed')
    elif attack == 'manifest':
        pin['weight_sha256']['model.safetensors'] = 'c'*64
    elif attack == 'identity':
        pin['manifest']['revision'] = 'local:'+'c'*64
    elif attack == 'path':
        pin['path'] = '../escape'
    elif attack == 'index':
        atomic_json(tmp_path/parent/'model.safetensors.index.json', {'weight_map':{'missing':'model.safetensors'}})
    elif attack == 'alias':
        before['lm_head.weight'] = before['model.embed_tokens.weight'] + 1
        save_file(before, str(tmp_path/parent/'model.safetensors'))
        digest = hashlib.sha256((tmp_path/parent/'model.safetensors').read_bytes()).hexdigest()
        pin['weight_sha256']['model.safetensors'] = digest
        pin['manifest']['files'][0]['sha256'] = digest
    with pytest.raises(ValueError):
        write_checkpoint('quantized', tmp_path, recipe, lambda stage: run_quantization(recipe, {'root':tmp_path}, stage))
    assert not (tmp_path/'quantized').exists()


def test_dry_run_inventory_has_correct_dpo_and_base_parents(tmp_path, capsys):
    from lab.scripts.quantization_sim import main
    assert main(['--config','lab/configs/quantization_sim.json','--lab-root',str(tmp_path),'--dry-run']) == 0
    rows = json.loads(capsys.readouterr().out)['recipes']
    assert len(rows) == 12
    assert {r['bits'] for r in rows} == {2,4,8}
    for row in rows:
        assert row['parents'] == [row['anchor'] if row['anchor'] == 'olmo-1b' else row['anchor']+'-dpo-s0']
    assert list(tmp_path.iterdir()) == []


def test_missing_effective_parameter_is_rejected(tmp_path):
    recipe, state = fixture(tmp_path)
    parent = recipe['parents'][0]
    del state['model.layers.0.self_attn.q_proj.weight']
    save_file(state, str(tmp_path/parent/'model.safetensors'))
    digest = hashlib.sha256((tmp_path/parent/'model.safetensors').read_bytes()).hexdigest()
    recipe['inputs'][parent]['weight_sha256']['model.safetensors'] = digest
    recipe['inputs'][parent]['manifest']['files'][0]['sha256'] = digest
    with pytest.raises(ValueError, match='missing effective parameter'):
        run_quantization(recipe, {'root':tmp_path}, tmp_path/'stage')


def test_sharded_roundtrip_preserves_integer_matrix_and_equal_tied_aliases(tmp_path):
    recipe, state = fixture(tmp_path)
    parent = recipe['parents'][0]
    path = tmp_path/parent
    state['lm_head.weight'] = state['model.embed_tokens.weight'].clone()
    state['integer_matrix'] = torch.tensor([[1,2],[3,4]])
    (path/'model.safetensors').unlink()
    groups = [dict(list(state.items())[::2]), dict(list(state.items())[1::2])]
    mapping, hashes = {}, {}
    for i, group in enumerate(groups, 1):
        name = f'model-{i:05d}-of-00002.safetensors'
        save_file(group, str(path/name))
        hashes[name] = hashlib.sha256((path/name).read_bytes()).hexdigest()
        mapping.update({k:name for k in group})
    atomic_json(path/'model.safetensors.index.json', {'weight_map':mapping})
    pin = recipe['inputs'][parent]
    pin['weight_sha256'] = hashes
    pin['manifest']['files'] = [dict(filename=k, sha256=v) for k,v in hashes.items()]
    stage = tmp_path/'stage'
    run_quantization(recipe, {'root':tmp_path}, stage)
    actual = {k:v for p in stage.glob('*.safetensors') for k,v in load_file(str(p)).items()}
    assert torch.equal(actual['integer_matrix'], state['integer_matrix'])
    assert torch.equal(actual['lm_head.weight'], actual['model.embed_tokens.weight'])
    assert json.loads((stage/'model.safetensors.index.json').read_text())['weight_map'] == mapping
