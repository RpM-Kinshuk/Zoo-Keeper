import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import load_file
from lab.scripts.decision import main, tiny_fixture, decision_recipes
from lab.training import train_decision, load_decision_checkpoint

CONFIG = Path('lab/configs/decision.json')


def test_six_recipes_and_dry_run(tmp_path, capsys):
    recipes = decision_recipes(json.loads(CONFIG.read_text()))
    assert len(recipes) == 6
    assert sum(r['edge_type'] == 'sft' for r in recipes) == 5
    assert [r['anchor'] for r in recipes if r['edge_type'] == 'lora'] == ['q3-06']
    assert all(r['steps'] == 32 and r['effective_batch'] == 16 and r['sequence_length'] == 128 for r in recipes)
    assert main(['--config', str(CONFIG), '--lab-root', str(tmp_path), '--dry-run']) == 0
    assert len(json.loads(capsys.readouterr().out)['recipes']) == 6
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('architecture,embedded', [('llama', False), ('modernbert', False), ('qwen3', True)])
def test_offline_checkpoint(tmp_path, architecture, embedded):
    torch.set_num_threads(1)
    recipe, inputs = tiny_fixture(architecture, embedded)
    initial = {n: p.detach().clone() for n, p in inputs['model'].named_parameters()}
    train_decision(recipe, inputs, tmp_path)
    restored = load_decision_checkpoint(tmp_path)
    restored.eval()
    with torch.no_grad():
        expected = inputs['effective_model'].eval()(input_ids=inputs['token_ids'], attention_mask=inputs['attention_mask']).logits
        actual = restored(input_ids=inputs['token_ids'], attention_mask=inputs['attention_mask']).logits
    assert actual.shape == (8, 2)
    if not embedded:
        assert any(not torch.equal(p, initial[n]) for n, p in inputs['model'].named_parameters() if p.ndim == 2 and p.shape[0] != 2)
    else:
        names = dict(inputs['effective_model'].named_parameters())
        assert all('lora_' in n or 'modules_to_save' in n for n, p in names.items() if p.requires_grad)
        assert any(torch.count_nonzero(v) for n, v in restored.state_dict().items() if 'lora_B' in n)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    state = load_file(str(tmp_path / 'model.safetensors'))
    assert any(v.shape == (2, 32) for v in state.values())
    assert any('lora_A' in k for k in state) == embedded
    if embedded:
        assert any('base_layer.weight' in k for k in state)
        metadata = json.loads((tmp_path / 'embedded_adapter.json').read_text())
        assert metadata['lora_alpha'] / metadata['r'] == 1
    assert restored.config.num_labels == 2
    assert restored.config.pad_token_id == 0
    assert json.loads((tmp_path / 'training.json').read_text())['full_parameters'] == (not embedded)


def test_invalid_labels_and_unfrozen_inputs(tmp_path):
    recipe, inputs = tiny_fixture('llama', False)
    inputs['labels'][0] = 2
    with pytest.raises(ValueError, match='label'):
        train_decision(recipe, inputs, tmp_path)
    with pytest.raises(ValueError, match='frozen|pinned'):
        train_decision(recipe, {}, tmp_path)


def test_truth_and_scale_validation():
    from lab.training import validate_decision_recipe
    recipe, _ = tiny_fixture('qwen3', True)
    with pytest.raises(ValueError, match='edge_type'):
        validate_decision_recipe({**recipe, 'edge_type': 'sft'})
    with pytest.raises(ValueError, match='rank=alpha'):
        validate_decision_recipe({**recipe, 'alpha': 16})


def test_embedded_metadata_tamper_fails(tmp_path):
    recipe, inputs = tiny_fixture('qwen3', True)
    train_decision(recipe, inputs, tmp_path)
    metadata = json.loads((tmp_path / 'embedded_adapter.json').read_text())
    metadata['lora_alpha'] = 16
    (tmp_path / 'embedded_adapter.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='metadata'):
        load_decision_checkpoint(tmp_path)


def test_frozen_native_classifier_load_and_input_integrity(tmp_path):
    import hashlib
    from safetensors.torch import save_file
    from lab.artifacts import content_hash
    from transformers import Qwen3ForCausalLM
    recipe, tiny = tiny_fixture('qwen3', True)
    base = tmp_path / 'base'
    Qwen3ForCausalLM(tiny['model'].config).save_pretrained(base, safe_serialization=True)
    token_file = tmp_path / 'tokens.safetensors'
    save_file(dict(input_ids=tiny['token_ids'], labels=tiny['labels'], attention_mask=tiny['attention_mask']), str(token_file))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    pin = dict(status='pinned', kind='model', repo_id='synthetic/base', revision='a' * 40,
        license='apache-2.0', license_path='evidence/LICENSE', license_sha256='b' * 64,
        tokenizer_sha256='c' * 64, config_sha256=content_hash(json.loads((base / 'config.json').read_text())),
        config_file_sha256=digest(base / 'config.json'),
        weight_sha256={'model.safetensors': digest(base / 'model.safetensors')})
    data_pin = dict(status='pinned', kind='dataset', repo_id='synthetic/classification', revision='b' * 40,
        license='apache-2.0', license_path='evidence/LICENSE', license_sha256='b' * 64, split='train',
        selected_index_sha256=content_hash(tiny['sample_ids']),
        preprocessing=dict(format='classification_tokens_v1', padding='right',
                           label_names=recipe['label_names'], pad_token_id=0))
    inputs = dict(root=tmp_path, model_path='base', tokens_path='tokens.safetensors', model_pin=pin,
                  data_pin=data_pin, tokens_sha256=digest(token_file), sample_ids=tiny['sample_ids'], heldout_ids=tiny['heldout_ids'])
    recipe = {**recipe, 'torch_dtype': 'bfloat16', 'parent_identities': {'tiny-base': 'synthetic/base@' + 'a' * 40}}
    train_decision(recipe, inputs, tmp_path / 'child')
    restored = load_decision_checkpoint(tmp_path / 'child')
    assert next(restored.parameters()).dtype == torch.float32
    assert restored(tiny['token_ids'], attention_mask=tiny['attention_mask']).logits.shape == (8, 2)
    inputs['tokens_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='token file hash'):
        train_decision(recipe, inputs, tmp_path / 'bad')
    inputs['data_pin']['license'] = 'unknown'
    with pytest.raises(ValueError, match='license'):
        train_decision(recipe, inputs, tmp_path / 'bad')


def test_tiny_cli_publishes_portable_resumable_checkpoints(tmp_path):
    args = ['--config', str(CONFIG), '--lab-root', str(tmp_path), '--tiny']
    assert main(args) == 0
    assert len(list(tmp_path.glob('*/model.safetensors'))) == 3
    for path in tmp_path.glob('*/recipe.json'):
        recipe = json.loads(path.read_text())['recipe']
        assert recipe['steps'] == 2
        assert recipe['edge_type'] == ('lora' if recipe['embedded_adapter'] else 'sft')
    assert main([*args, '--resume']) == 0


def test_fresh_lora_then_classifier_never_network_and_restore_offline_state(tmp_path):
    import os
    import subprocess
    import sys
    # A fresh interpreter exercises PEFT's first optional-dependency imports.
    code = r'''
import os
import runpy
import socket
import sys
from pathlib import Path
from huggingface_hub import constants
import torch
torch.set_num_threads(1)
assert os.environ['HF_HUB_OFFLINE'] == '0'
assert constants.HF_HUB_OFFLINE is False
connections = []
def reject_connect(self, address):
    connections.append(str(address))
    raise AssertionError('unexpected network connection')
socket.socket.connect = reject_connect
socket.socket.connect_ex = reject_connect
def reject_dns(*args, **kwargs):
    connections.append('DNS ' + str(args[0]))
    raise AssertionError('unexpected DNS lookup')
socket.getaddrinfo = reject_dns
lora_suite = runpy.run_path('tests/lab/test_lora.py')
lora_suite['test_tiny_lora_freezes_parent_and_saves_effective_tied_bf16'](Path(sys.argv[1]) / 'causal')
assert not connections, connections
assert constants.HF_HUB_OFFLINE is False
assert os.environ['HF_HUB_OFFLINE'] == '0'
suite = runpy.run_path('tests/lab/test_decision.py')
suite['test_frozen_native_classifier_load_and_input_integrity'](Path(sys.argv[1]) / 'classification')
assert not connections, connections
assert constants.HF_HUB_OFFLINE is False
assert os.environ['HF_HUB_OFFLINE'] == '0'
'''
    env = {**os.environ, 'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0', 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
