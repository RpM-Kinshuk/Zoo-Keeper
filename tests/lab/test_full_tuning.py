"""Offline full-parameter continuation and strict provenance contracts."""
import copy
import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from tests.lab.test_lora import fixture_model, frozen_inputs
from lab.artifacts import content_hash, write_checkpoint
from lab.scripts.sft import main as sft_main
from lab.scripts.dpo import main as dpo_main
from lab.training import train_sft, train_dpo


def recipe(edge='sft'):
    parent = 'tiny-base' if edge == 'sft' else 'tiny-sft'
    return dict(model_id='tiny-' + edge, edge_type=edge, parents=[parent],
                parent_identities={parent: 'lab/' + parent + '@local:' + 'a' * 64},
                seed=20261028, steps=2, effective_batch=4, sequence_length=8,
                learning_rate=0.001, torch_dtype='float32', device='cpu',
                base_num_parameters=24000)


def memory_inputs(edge='sft', model=None):
    tokens = torch.arange(64).reshape(8, 8)
    result = dict(model=model if model is not None else fixture_model(), token_ids=tokens,
                  sample_ids=list(range(8)), heldout_ids=[8])
    if edge == 'dpo':
        rejected = tokens.clone()
        rejected[:, 4:] = rejected[:, 4:].roll(1, dims=1)
        result.update(rejected_token_ids=rejected,
                      chosen_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * 8),
                      rejected_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * 8),
                      parent_relation='sft')
    return result


def test_full_sft_and_dpo_change_nonadapter_matrix_and_freeze_reference(tmp_path):
    torch.set_num_threads(1)
    model = fixture_model()
    key = 'model.layers.0.mlp.up_proj.weight'
    before = model.state_dict()[key].clone()
    sft = write_checkpoint('tiny-sft', tmp_path, recipe(),
                           lambda out: train_sft(recipe(), memory_inputs(model=model), out))
    assert not torch.equal(before, model.state_dict()[key])
    sft_model = AutoModelForCausalLM.from_pretrained(sft, local_files_only=True, use_safetensors=True, dtype=torch.float32)
    reference = copy.deepcopy(sft_model)
    reference_before = {k: v.clone() for k, v in reference.state_dict().items()}
    before = sft_model.state_dict()[key].clone()
    inputs = {**memory_inputs('dpo', sft_model), 'reference_model': reference}
    dpo = write_checkpoint('tiny-dpo', tmp_path, recipe('dpo'),
                           lambda out: train_dpo(recipe('dpo'), inputs, out))
    assert not torch.equal(before, sft_model.state_dict()[key])
    for k, v in reference.state_dict().items():
        assert torch.equal(v, reference_before[k])
    assert all(not p.requires_grad for p in reference.parameters())
    assert json.loads((dpo / 'lineage.json').read_text()) == dict(name='lab/tiny-dpo', bases=['lab/tiny-sft'], relation='dpo')
    report = json.loads((dpo / 'training.json').read_text())
    assert report['steps_completed'] == 2 and report['reference_unchanged']
    assert report['loss_implementation'] == 'torch_logsigmoid_dpo'
    loaded = AutoModelForCausalLM.from_pretrained(dpo, local_files_only=True, use_safetensors=True, dtype=torch.bfloat16)
    assert loaded.get_input_embeddings().weight.data_ptr() == loaded.get_output_embeddings().weight.data_ptr()
    assert all(p.dtype == torch.bfloat16 for p in loaded.parameters())


@pytest.mark.parametrize('edge', ['sft', 'dpo'])
def test_size_and_overlap_refused_before_weight_loading(tmp_path, monkeypatch, edge):
    trainer = train_sft if edge == 'sft' else train_dpo
    def forbidden(*args, **kwargs):
        raise AssertionError('weights must not be loaded')
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)
    with pytest.raises(ValueError, match='1.7B'):
        trainer({**recipe(edge), 'base_num_parameters': 1700000001}, {}, tmp_path)
    inputs = memory_inputs(edge)
    inputs['heldout_ids'] = [0]
    with pytest.raises(ValueError, match='overlap'):
        trainer(recipe(edge), inputs, tmp_path)
    with pytest.raises(ValueError, match='frozen|pinned'):
        trainer(recipe(edge), {}, tmp_path)


def test_dpo_requires_exact_sft_reference_and_parent(tmp_path):
    inputs = memory_inputs('dpo')
    inputs['parent_relation'] = 'lora'
    with pytest.raises(ValueError, match='SFT'):
        train_dpo(recipe('dpo'), inputs, tmp_path)
    inputs['parent_relation'] = 'sft'
    inputs['reference_model'] = fixture_model()
    with torch.no_grad():
        next(inputs['reference_model'].parameters()).add_(1)
    with pytest.raises(ValueError, match='reference'):
        train_dpo(recipe('dpo'), inputs, tmp_path)


def test_sft_real_prepared_inputs_strict_hashes_and_preprocessing(tmp_path):
    r, inputs = frozen_inputs(tmp_path)
    r = {**r, **recipe(), 'parent_identities': r['parent_identities'], 'torch_dtype': 'bfloat16'}
    inputs['data_pin']['preprocessing'] = dict(format='causal_tokens_v1', loss='all_tokens', padding='none')
    train_sft(r, inputs, tmp_path / 'child')
    report = json.loads((tmp_path / 'child' / 'training.json').read_text())
    assert report['sample_ids'] == list(range(8))
    assert report['parameter_dtype'] == 'float32'
    assert report['torch_dtype'] == 'bfloat16'
    inputs['model_pin']['weight_sha256']['model.safetensors'] = '0' * 64
    with pytest.raises(ValueError, match='weight hash'):
        train_sft(r, inputs, tmp_path / 'tampered')


@pytest.mark.parametrize('tamper', ['weight', 'canonical_config', 'raw_config', 'source_pin', 'zero_steps', 'relation'])
def test_real_dpo_local_parent_provenance(tmp_path, tamper):
    r, inputs = frozen_inputs(tmp_path)
    r = {**r, **recipe(), 'parent_identities': r['parent_identities']}
    inputs['data_pin']['preprocessing'] = dict(format='causal_tokens_v1', loss='all_tokens', padding='none')
    portable = {k: v for k, v in inputs.items() if k != 'root'}
    r['inputs'] = portable
    sft = write_checkpoint('tiny-sft', tmp_path, r, lambda out: train_sft(r, inputs, out))
    # Extractor manifest key is supplied by the caller, never recomputed by training.
    key = 'lab/tiny-sft@local:' + 'c' * 64
    config = (sft / 'config.json').read_bytes()
    manifest = dict(model_key=key, config_sha256=content_hash(json.loads(config)),
                    config_file_sha256=hashlib.sha256(config).hexdigest(),
                    weight_sha256={'model.safetensors': hashlib.sha256((sft / 'model.safetensors').read_bytes()).hexdigest()})
    data = memory_inputs('dpo')
    token_file = tmp_path / 'prefs.safetensors'
    save_file(dict(input_ids=data['token_ids'], rejected_input_ids=data['rejected_token_ids'],
                   chosen_mask=data['chosen_mask'], rejected_mask=data['rejected_mask']), str(token_file))
    d_inputs = {**inputs, 'model_path': 'tiny-sft', 'local_parent': manifest,
                'tokens_path': 'prefs.safetensors', 'tokens_sha256': hashlib.sha256(token_file.read_bytes()).hexdigest(),
                'data_pin': {**inputs['data_pin'], 'preprocessing': dict(format='preference_tokens_v1', loss='completion_only', padding='none')}}
    d_recipe = {**recipe('dpo'), 'parent_identities': {'tiny-sft': key}}
    train_dpo(d_recipe, d_inputs, tmp_path / 'dpo')
    if tamper == 'weight':
        manifest['weight_sha256']['model.safetensors'] = '0' * 64
    elif tamper == 'canonical_config':
        manifest['config_sha256'] = '0' * 64
    elif tamper == 'raw_config':
        manifest['config_file_sha256'] = '0' * 64
    elif tamper == 'source_pin':
        d_inputs['model_pin'] = {**inputs['model_pin'], 'revision': 'd' * 40}
    elif tamper == 'zero_steps':
        report_path = sft / 'training.json'
        report = json.loads(report_path.read_text())
        report['steps_completed'] = 0
        report_path.write_text(json.dumps(report))
    else:
        lineage_path = sft / 'lineage.json'
        lineage = json.loads(lineage_path.read_text())
        lineage['relation'] = 'lora'
        lineage_path.write_text(json.dumps(lineage))
    with pytest.raises(ValueError, match='weight hash|config.*hash|SFT|source'):
        train_dpo(d_recipe, d_inputs, tmp_path / 'bad')


def test_scripts_tiny_chain_and_unfrozen_real_refusal(tmp_path, capsys):
    config = tmp_path / 'config.json'
    config.write_text('{}')
    root = tmp_path / 'lab'
    for main in [sft_main, dpo_main]:
        main(['--config', str(config), '--lab-root', str(root), '--dry-run'])
        assert len(json.loads(capsys.readouterr().out)['recipes']) == 8
        with pytest.raises(ValueError, match='blocked'):
            main(['--config', str(config), '--lab-root', str(root)])
    with pytest.raises(ValueError, match='SFT'):
        dpo_main(['--config', str(config), '--lab-root', str(root), '--tiny'])
    sft_main(['--config', str(config), '--lab-root', str(root), '--tiny'])
    dpo_main(['--config', str(config), '--lab-root', str(root), '--tiny'])
    for edge in ['sft', 'dpo']:
        report = json.loads((root / ('tiny-' + edge) / 'training.json').read_text())
        assert report['steps_completed'] == 2
        assert not list((root / ('tiny-' + edge)).rglob('*.bin'))


def test_real_sft_rejects_missing_template_protocol_and_zero_steps(tmp_path):
    r, inputs = frozen_inputs(tmp_path)
    r = {**r, **recipe(), 'parent_identities': r['parent_identities']}
    with pytest.raises(ValueError, match='preprocessing'):
        train_sft(r, inputs, tmp_path / 'bad-preprocessing')
    with pytest.raises(ValueError, match='steps'):
        train_sft({**recipe(), 'steps': 0}, memory_inputs(), tmp_path / 'zero')


def test_dpo_rejects_prompt_or_completion_mask_mismatch(tmp_path):
    inputs = memory_inputs('dpo')
    inputs['rejected_token_ids'][0, 0] = 63
    with pytest.raises(ValueError, match='prompts'):
        train_dpo(recipe('dpo'), inputs, tmp_path)
    inputs = memory_inputs('dpo')
    inputs['chosen_mask'][0, 6] = 0
    with pytest.raises(ValueError, match='contiguous'):
        train_dpo(recipe('dpo'), inputs, tmp_path)


@pytest.mark.parametrize('edge', ['sft', 'dpo'])
def test_training_changes_float_matrix_before_bf16_export(tmp_path, edge):
    trainer = train_sft if edge == 'sft' else train_dpo
    trainer(recipe(edge), memory_inputs(edge), tmp_path)
    report = json.loads((tmp_path / 'training.json').read_text())
    assert report['updated_matrix']['delta_norm_before_export'] > 0
    assert 'lora_' not in report['updated_matrix']['name']
