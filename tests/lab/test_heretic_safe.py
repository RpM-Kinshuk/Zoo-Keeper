"""Installed Heretic algorithm, with all upstream loading entrypoints bypassed."""
import importlib.util

import pytest
import torch

from transformers import LlamaConfig, LlamaForCausalLM


def fixture_model():
    torch.manual_seed(20261028)
    return LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=64, max_position_embeddings=32, tie_word_embeddings=True))


def adapter():
    spec = importlib.util.find_spec('lab.heretic_safe')
    assert spec is not None, 'safe installed Heretic adapter is missing'
    from lab import heretic_safe
    return heretic_safe


def test_installed_algorithm_edits_real_peft_weights_without_any_loader(tmp_path, monkeypatch):
    safe = adapter()
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, PretrainedConfig
    from heretic.model import Model
    def forbidden(*args, **kwargs):
        raise AssertionError('upstream loader or constructor was reached')
    for cls in (AutoModelForCausalLM, AutoTokenizer, AutoProcessor):
        monkeypatch.setattr(cls, 'from_pretrained', forbidden)
    monkeypatch.setattr(PretrainedConfig, 'get_config_dict', forbidden)
    monkeypatch.setattr(Model, '__init__', forbidden)
    base = fixture_model().float().eval()
    weight = base.model.layers[0].self_attn.o_proj.weight.detach().clone()
    wrapped = safe.wrap_loaded_model(base, seed=20261028, row_normalization='none')
    directions = torch.zeros(4, 32); directions[:, 0] = 1
    parameters = {key: dict(max_weight=0.5, max_weight_position=0., min_weight=0.5,
                           min_weight_distance=2.) for key in wrapped.get_abliterable_components()}
    safe.apply_trial(wrapped, directions, direction_index=None, parameters=parameters)
    merged = safe.merge_loaded_model(wrapped)
    expected = weight.clone(); expected[0] *= 0.5
    torch.testing.assert_close(merged.model.layers[0].self_attn.o_proj.weight, expected)
    merged.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / 'model.safetensors').is_file()
    assert not list(tmp_path.rglob('*.bin'))


def test_reset_trial_does_not_accumulate_previous_edit():
    safe = adapter()
    base = fixture_model().float().eval()
    weight = base.model.layers[0].mlp.down_proj.weight.detach().clone()
    wrapped = safe.wrap_loaded_model(base, seed=20261028, row_normalization='none')
    directions = torch.zeros(4, 32); directions[:, 1] = 1
    parameters = {key: dict(max_weight=0.5, max_weight_position=0., min_weight=0.5,
                           min_weight_distance=2.) for key in wrapped.get_abliterable_components()}
    for _ in range(2):
        safe.apply_trial(wrapped, directions, direction_index=None, parameters=parameters)
    merged = safe.merge_loaded_model(wrapped)
    expected = weight.clone(); expected[1] *= 0.5
    torch.testing.assert_close(merged.model.layers[0].mlp.down_proj.weight, expected)


def test_invalid_directions_rejected_before_adapter_mutation():
    safe = adapter()
    wrapped = safe.wrap_loaded_model(fixture_model(), seed=20261028, row_normalization='none')
    with pytest.raises(ValueError, match='direction'):
        safe.apply_trial(wrapped, torch.zeros(2, 32), direction_index=None, parameters={})


def test_optimizer_runs_real_trials_and_restores_selected_trial():
    safe = adapter()
    assert hasattr(safe, 'optimize_loaded_model'), 'installed optimization interface missing'
    wrapped = safe.wrap_loaded_model(fixture_model().float(), seed=20261028, row_normalization='none')
    directions = torch.zeros(4, 32); directions[:, 0] = 1
    records = [dict(sample_id='train0', role='group_0'), dict(sample_id='train1', role='group_1')]
    seen = []
    def objective(model, training):
        assert training == records
        # Synthetic integration objective only; production objective is explicit.
        value = float(model.base_model.model.model.layers[2].self_attn.o_proj.lora_B['default'].weight.square().sum())
        seen.append(value)
        return value, value
    report = safe.optimize_loaded_model(wrapped, directions, objective=objective,
        objective_id='synthetic_test_squared_adapter_norm', training_records=records,
        heldout_ids=['validation'], seed=20261028, n_trials=3, n_startup_trials=2)
    assert len(report['optimization_trace']) == 3 and len(seen) == 3
    assert report['objective_id'] == 'synthetic_test_squared_adapter_norm'
    assert report['training_sample_ids'] == ['train0', 'train1']
    assert report['selected_trial'] in range(3)
    assert report['per_component_settings']
    selected = report['optimization_trace'][report['selected_trial']]
    restored = float(wrapped.model.base_model.model.model.layers[2].self_attn.o_proj.lora_B['default'].weight.detach().square().sum())
    assert len(set(seen)) > 1
    assert restored == pytest.approx(selected['objectives'][0])
    assert all(t['parameters']['direction_scope'] in {'global', 'per layer'} for t in report['optimization_trace'])


def test_optimizer_rejects_heldout_overlap_before_objective():
    safe = adapter()
    assert hasattr(safe, 'optimize_loaded_model'), 'installed optimization interface missing'
    def forbidden(*args):
        raise AssertionError('objective reached before disjointness check')
    with pytest.raises(ValueError, match='overlap'):
        safe.optimize_loaded_model(None, None, objective=forbidden, objective_id='explicit',
            training_records=[dict(sample_id='heldout', role='group_0')],
            heldout_ids=['heldout'], seed=1, n_trials=1)


def test_full_normalization_rank_three_resets_and_merges_real_algorithm():
    safe = adapter()
    base = fixture_model().float().eval()
    before = base.model.layers[2].self_attn.o_proj.weight.detach().clone()
    wrapped = safe.wrap_loaded_model(base, seed=20261028, row_normalization='full')
    assert wrapped.peft_config.r == 3
    directions = torch.zeros(4, 32); directions[:, 0] = 1
    parameters = {key: dict(max_weight=0.8, max_weight_position=2., min_weight=0.5,
        min_weight_distance=2.) for key in wrapped.get_abliterable_components()}
    safe.apply_trial(wrapped, directions, direction_index=None, parameters=parameters)
    module = wrapped.model.base_model.model.model.layers[2].self_attn.o_proj
    first_delta = module.get_delta_weight('default').detach().clone()
    assert first_delta.norm() > 0
    safe.apply_trial(wrapped, directions, direction_index=None, parameters=parameters)
    torch.testing.assert_close(module.get_delta_weight('default'), first_delta, rtol=0, atol=0)
    merged = safe.merge_loaded_model(wrapped)
    after = merged.model.layers[2].self_attn.o_proj.weight
    torch.testing.assert_close(after, before + first_delta)
    # Upstream full normalization preserves norms approximately after rank3 SVD.
    relative_drift = (after.norm(dim=1) / before.norm(dim=1) - 1).abs()
    assert relative_drift.max() < 0.1
