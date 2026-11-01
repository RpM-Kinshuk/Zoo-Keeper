"""Offline effective-weight, frozen-parent and safe serialization contracts."""
import json

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM


def fixture_model():
    torch.manual_seed(20261028)
    return LlamaForCausalLM(LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=32, tie_word_embeddings=True))


def recipe():
    return dict(model_id='tiny-lora', edge_type='lora', parents=['tiny-base'],
                parent_identities={'tiny-base': 'lab/tiny-base@local:' + 'a' * 64},
                seed=20261028, rank=8, alpha=8, steps=2, effective_batch=4,
                sequence_length=8, learning_rate=2e-4)


def test_tiny_lora_freezes_parent_and_saves_effective_tied_bf16(tmp_path):
    from lab.training import train_lora
    model = fixture_model()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    tokens = torch.arange(64).reshape(8, 8)
    train_lora(recipe(), {'model': model, 'token_ids': tokens}, tmp_path)
    factors = load_file(str(tmp_path / 'adapter' / 'adapter_model.safetensors'))
    assert len(factors) == 8
    assert any(torch.count_nonzero(v) for k, v in factors.items() if 'lora_B' in k)
    # A missing freeze would mutate the retained parent tensors before merge.
    report = json.loads((tmp_path / 'training.json').read_text())
    assert report['steps_completed'] == 2
    assert report['base_tensors_unchanged'] is True
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True,
                                                 use_safetensors=True, dtype=torch.bfloat16)
    assert loaded.get_input_embeddings().weight.data_ptr() == loaded.get_output_embeddings().weight.data_ptr()
    assert all(v.dtype == torch.bfloat16 for v in loaded.state_dict().values())
    changed = 0
    for key, weight in before.items():
        module = key.removesuffix('.weight')
        prefix = 'base_model.model.' + module
        a_key = prefix + '.lora_A.weight'
        if a_key in factors:
            expected = weight.float() + factors[prefix + '.lora_B.weight'].float() @ factors[a_key].float()
            torch.testing.assert_close(loaded.state_dict()[key], expected.bfloat16(), rtol=0, atol=0)
            changed += 1
        else:
            torch.testing.assert_close(loaded.state_dict()[key], weight.bfloat16(), rtol=0, atol=0)
    assert changed == 4
    assert not list(tmp_path.rglob('*.bin'))


def test_merge_adapter_uses_alpha_over_rank_and_rejects_unmatched_factors():
    from lab.training import merge_adapter
    base = {'q.weight': torch.zeros(2, 3)}
    factors = {'q.lora_A.weight': torch.tensor([[1., 2., 3.]]),
               'q.lora_B.weight': torch.tensor([[2.], [4.]])}
    merged = merge_adapter(base, factors, {'rank': 1, 'alpha': 2})
    torch.testing.assert_close(merged['q.weight'], torch.tensor([[4., 8., 12.], [8., 16., 24.]]))
    with pytest.raises(ValueError, match='factor|target'):
        merge_adapter(base, {**factors, 'missing.lora_A.weight': torch.ones(1, 3)}, {'rank': 1, 'alpha': 2})


def test_tiny_script_atomic_lineage_resume_and_dry_run(tmp_path, capsys):
    from lab.scripts.lora import main
    config = tmp_path / 'config.json'
    config.write_text('{}')
    root = tmp_path / 'lab'
    assert main(['--config', str(config), '--lab-root', str(root), '--dry-run']) == 0
    assert not root.exists()
    assert len(json.loads(capsys.readouterr().out)['recipes']) == 24
    assert main(['--config', str(config), '--lab-root', str(root), '--tiny']) == 0
    child = root / 'tiny-lora'
    lineage = json.loads((child / 'lineage.json').read_text())
    assert lineage == {'name': 'lab/tiny-lora', 'bases': ['lab/tiny-base'], 'relation': 'lora'}
    before = (child / 'model.safetensors').read_bytes()
    assert main(['--config', str(config), '--lab-root', str(root), '--tiny', '--resume']) == 0
    assert (child / 'model.safetensors').read_bytes() == before
    config.write_text('{"learning_rate": 0.001}')
    with pytest.raises(ValueError, match='resume mismatch'):
        main(['--config', str(config), '--lab-root', str(root), '--tiny', '--resume'])


def test_unknown_targets_and_unfrozen_data_fail_before_training(tmp_path):
    from lab.training import train_lora
    with pytest.raises(ValueError, match='target'):
        train_lora({**recipe(), 'target_modules': ['missing']},
                   {'model': fixture_model(), 'token_ids': torch.arange(64).reshape(8, 8)}, tmp_path)
    with pytest.raises(ValueError, match='frozen|pinned'):
        train_lora(recipe(), {}, tmp_path)
    assert not (tmp_path / 'model.safetensors').exists()


def frozen_inputs(tmp_path):
    """Prepared snapshot with synthetic commit identities and real local hashes."""
    import hashlib
    from safetensors.torch import save_file
    from lab.artifacts import content_hash
    model_dir = tmp_path / 'base'
    fixture_model().save_pretrained(model_dir, safe_serialization=True)
    config_bytes = (model_dir / 'config.json').read_bytes()
    ids = list(range(8))
    save_file({'input_ids': torch.arange(64).reshape(8, 8)}, str(tmp_path / 'tokens.safetensors'))
    pin = dict(status='pinned', model_id='tiny-base', kind='model', repo_id='synthetic/tiny',
               revision='a' * 40, license='apache-2.0', license_path='evidence/LICENSE',
               license_sha256='a' * 64, tokenizer_sha256='b' * 64,
               config_sha256=content_hash(json.loads(config_bytes)),
               config_file_sha256=hashlib.sha256(config_bytes).hexdigest(),
               weight_sha256={'model.safetensors': hashlib.sha256((model_dir / 'model.safetensors').read_bytes()).hexdigest()})
    data = dict(status='pinned', kind='dataset', repo_id='synthetic/tokens', revision='b' * 40,
                license='apache-2.0', license_path='evidence/LICENSE', license_sha256='a' * 64,
                split='train', selected_index_sha256=content_hash(ids), preprocessing={'fixture': 'tokens-v1'})
    inputs = dict(root=tmp_path, model_path='base', tokens_path='tokens.safetensors', model_pin=pin,
                  data_pin=data, sample_ids=ids, heldout_ids=[8],
                  tokens_sha256=hashlib.sha256((tmp_path / 'tokens.safetensors').read_bytes()).hexdigest())
    r = {**recipe(), 'parent_identities': {'tiny-base': 'synthetic/tiny@' + 'a' * 40},
         'device': 'cpu', 'torch_dtype': 'bfloat16'}
    return r, inputs


def test_prepared_snapshot_consumes_config_hash_and_device_dtype(tmp_path):
    from lab.training import _frozen_inputs, train_lora
    r, inputs = frozen_inputs(tmp_path)
    model, tokens = _frozen_inputs(r, inputs)
    assert next(model.parameters()).dtype == torch.bfloat16
    assert next(model.parameters()).device.type == 'cpu'
    assert tokens.shape == (8, 8)
    train_lora(r, inputs, tmp_path / 'child')
    report = json.loads((tmp_path / 'child' / 'training.json').read_text())
    assert report['device'] == 'cpu'
    assert report['torch_dtype'] == 'bfloat16'
    # Canonical configuration identity remains independent of source formatting.
    config_path = tmp_path / 'base' / 'config.json'
    config_path.write_text(json.dumps(json.loads(config_path.read_text()), separators=(',', ':')))
    with pytest.raises(ValueError, match='config.*hash'):
        _frozen_inputs(r, inputs)


@pytest.mark.parametrize('tamper', ['changed', 'missing_provenance', 'unlisted', 'bad_index'])
def test_prepared_snapshot_rejects_unbound_weights(tmp_path, tamper):
    from lab.training import _frozen_inputs
    r, inputs = frozen_inputs(tmp_path)
    model_path = tmp_path / 'base'
    # Isolate weight provenance from the separate raw/canonical config regression.
    from lab.schema import canonical_json
    import hashlib
    config_path = model_path / 'config.json'
    config_path.write_text(canonical_json(json.loads(config_path.read_text())))
    inputs['model_pin']['config_file_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if tamper == 'changed':
        # Same config, loadable safetensors, different weights must fail before loading.
        from safetensors.torch import save_file
        state = load_file(str(model_path / 'model.safetensors'))
        state['model.layers.0.self_attn.q_proj.weight'].add_(1)
        save_file(state, str(model_path / 'model.safetensors'), metadata={'format': 'pt'})
    elif tamper == 'missing_provenance':
        del inputs['model_pin']['weight_sha256']
    elif tamper == 'unlisted':
        (model_path / 'extra.safetensors').write_bytes((model_path / 'model.safetensors').read_bytes())
    else:
        (model_path / 'model.safetensors.index.json').write_text(json.dumps({
            'weight_map': {'model.embed_tokens.weight': 'unlisted.safetensors'}}))
    with pytest.raises(ValueError, match='weight|provenance|shard'):
        _frozen_inputs(r, inputs)
