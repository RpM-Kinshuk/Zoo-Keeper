"""Offline residual projection, frozen data, and honest Heretic dispatch tests."""
import copy
import json
from importlib.metadata import version

import pytest
import torch
from safetensors.torch import load_file, save_file

from lab.abliteration import (classic_edit, mean_directions, run_abliteration,
                             installed_heretic_runner)
from lab.artifacts import content_hash, write_checkpoint
from lab.scripts.abliteration import main
from tests.lab.test_lora import fixture_model, frozen_inputs


def recipe(method='classic'):
    return dict(model_id='tiny-' + method, edge_type='abliteration', method=method,
                parents=['tiny-base'], parent_identities={'tiny-base': 'lab/tiny-base@local:' + 'a' * 64},
                seed=20261028, torch_dtype='float32', device='cpu',
                heretic_config={'n_trials': 2, 'orthogonalize_direction': True})


def split_inputs():
    return dict(sample_ids=list(range(8)), source_training_ids=[10, 11], heldout_ids=[8, 9], validation_ids=[8], test_ids=[9],
                direction_labels=[0, 0, 0, 0, 1, 1, 1, 1])


def test_classic_every_writer_bias_orientation_and_orthogonal_preservation():
    torch.manual_seed(20261028)
    r = torch.tensor([1., 2., 3., 4.]); r /= r.norm()
    state = {'a.weight': torch.randn(4, 6), 'a.bias': torch.randn(4),
             'b.weight': torch.randn(3, 4), 'untouched': torch.randn(4)}
    original = copy.deepcopy(state)
    modules = {'a.weight': {'layer': 0, 'residual_axis': 0},
               'a.bias': {'layer': 0, 'residual_axis': 0},
               'b.weight': {'layer': 0, 'residual_axis': 1}}
    result = classic_edit(state, {0: r * 7}, modules)
    q = torch.tensor([2., -1., 0., 0.])
    assert torch.norm(r @ result['a.weight']) < 1e-6
    assert abs(r @ result['a.bias']) < 1e-6
    assert torch.norm(result['b.weight'] @ r) < 1e-6
    torch.testing.assert_close(q @ result['a.weight'], q @ original['a.weight'])
    torch.testing.assert_close(result['b.weight'] @ q, original['b.weight'] @ q)
    for name in state:
        assert torch.equal(state[name], original[name])
    assert torch.equal(result['untouched'], original['untouched'])


@pytest.mark.parametrize('bad', [torch.zeros(4), torch.tensor([float('nan')] * 4), torch.ones(3)])
def test_invalid_directions_fail(bad):
    with pytest.raises(ValueError, match='direction'):
        classic_edit({'w': torch.ones(4, 2)}, {0: bad}, {'w': {'layer': 0, 'residual_axis': 0}})


def test_mean_difference_is_normalized_and_uses_both_groups():
    residuals = {0: torch.tensor([[0., 1.], [0., 3.], [4., 1.], [4., 3.]])}
    result = mean_directions(residuals, [0, 0, 1, 1])
    torch.testing.assert_close(result[0], torch.tensor([1., 0.]))
    with pytest.raises(ValueError, match='both'):
        mean_directions(residuals, [0] * 4)


def test_classic_real_local_pinned_fixture_and_tied_loadable_checkpoint(tmp_path):
    r, inputs = frozen_inputs(tmp_path)
    r = {**recipe(), 'parent_identities': r['parent_identities'], 'inputs': {'fixture': 'pinned-local'}}
    inputs.update(split_inputs())
    inputs['data_pin']['preprocessing'] = dict(format='abliteration_tokens_v1', padding='none',
        residual_position='last_token', direction_labels_sha256=content_hash(inputs['direction_labels']))
    path = write_checkpoint(r['model_id'], tmp_path, r, lambda out: run_abliteration(r, inputs, out))
    report = json.loads((path / 'abliteration.json').read_text())
    assert report['method'] == 'classic' and len(report['writers']) == 4
    assert report['validation_status'] == 'not_run_no_frozen_validation_tokens'
    assert report['direction_sample_ids'] == list(range(8))
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True, trust_remote_code=False)
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    directions = load_file(str(path / 'diagnostics' / 'directions.safetensors'))
    for name, spec in report['writers'].items():
        assert torch.norm(directions[str(spec['layer'])] @ model.state_dict()[name]) < 1e-6


@pytest.mark.parametrize('field', ['heldout_ids', 'validation_ids', 'test_ids', 'source_training_ids'])
def test_overlap_refused_before_loading(tmp_path, monkeypatch, field):
    from transformers import AutoModelForCausalLM
    def forbidden(*args, **kwargs):
        raise AssertionError('weight loader reached')
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)
    inputs = split_inputs(); inputs[field] = [0]
    with pytest.raises(ValueError, match='overlap'):
        run_abliteration(recipe(), inputs, tmp_path)


def test_injected_heretic_preserves_exact_config_and_trace(tmp_path):
    r = recipe('heretic'); inputs = split_inputs()
    trace = [{'trial': 0, 'params': {'max_weight': 0.7}, 'objectives': [0.2, 1]}]
    def fake(selected, training, output):
        assert selected == r
        assert set(training) == {'sample_ids', 'direction_labels'}
        save_file({'synthetic.weight': torch.eye(4)}, str(output / 'model.safetensors'))
        return dict(config=selected['heretic_config'], optimization_trace=trace,
                    per_layer_settings={'0': {'directions': [0, 1], 'weight': 0.7}},
                    runner_kind='injected_test', synthetic=True)
    inputs['heretic_runner'] = fake
    path = write_checkpoint(r['model_id'], tmp_path, r, lambda out: run_abliteration(r, inputs, out))
    report = json.loads((path / 'abliteration.json').read_text())
    assert report['config'] == r['heretic_config'] and report['optimization_trace'] == trace
    assert report['synthetic'] is True and report['runner_kind'] == 'injected_test'
    assert json.loads((path / 'recipe.json').read_text())['recipe'] == r
    assert json.loads((path / 'lineage.json').read_text())['relation'] == 'abliteration'


def test_missing_heretic_safe_output_and_trace_rejected(tmp_path):
    inputs = {**split_inputs(), 'heretic_runner': lambda *args: {}}
    with pytest.raises(ValueError, match='trace|config|safe'):
        run_abliteration(recipe('heretic'), inputs, tmp_path)


def test_installed_heretic_rejects_tiny_local_model_before_unsafe_loader(tmp_path):
    assert version('heretic-llm') == '1.4.0'
    model_path = tmp_path / 'model'; fixture_model().save_pretrained(model_path, safe_serialization=True)
    with pytest.raises(RuntimeError, match='1.4.0.*local_files_only.*use_safetensors.*trust_remote_code'):
        installed_heretic_runner(recipe('heretic'), {'root': tmp_path, 'model_path': 'model'}, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_script_dry_run_no_weights_tiny_and_unfrozen_refusal(tmp_path, capsys, monkeypatch):
    from transformers import AutoModelForCausalLM
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run loaded weights')
    config = tmp_path / 'config.json'; config.write_text('{}')
    args = ['--config', str(config), '--lab-root', str(tmp_path / 'lab')]
    with monkeypatch.context() as m:
        m.setattr(AutoModelForCausalLM, 'from_pretrained', forbidden)
        assert main(args + ['--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan['recipes']) == 8 and plan['launch'] == 'HUMAN'
    with pytest.raises(ValueError, match='blocked'):
        main(args)
    assert main(args + ['--tiny']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'COMPLETE_SYNTHETIC' and result['real_heretic_output'] is False
    assert (tmp_path / 'lab' / 'tiny-classic' / 'lineage.json').is_file()
    assert (tmp_path / 'lab' / 'tiny-heretic-dispatch' / 'lineage.json').is_file()


def test_classic_validation_reports_drift_without_using_validation_for_direction(tmp_path):
    import hashlib
    r, inputs = frozen_inputs(tmp_path)
    r = {**recipe(), 'parent_identities': r['parent_identities']}
    inputs.update(split_inputs())
    inputs['data_pin']['preprocessing'] = dict(format='abliteration_tokens_v1', padding='none',
        residual_position='last_token', direction_labels_sha256=content_hash(inputs['direction_labels']))
    path = tmp_path / 'validation.safetensors'
    save_file({'input_ids': torch.arange(8).reshape(1, 8)}, str(path))
    inputs.update(validation_tokens_path='validation.safetensors',
                  validation_tokens_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    run_abliteration(r, inputs, tmp_path / 'first')
    report = json.loads((tmp_path / 'first' / 'abliteration.json').read_text())
    assert report['validation_status'] == 'reported_only'
    assert report['validation_metrics']['max_abs_logit_drift'] > 0
    save_file({'input_ids': torch.arange(8, 16).reshape(1, 8)}, str(path))
    inputs['validation_tokens_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    run_abliteration(r, inputs, tmp_path / 'second')
    first = load_file(str(tmp_path / 'first' / 'diagnostics' / 'directions.safetensors'))
    second = load_file(str(tmp_path / 'second' / 'diagnostics' / 'directions.safetensors'))
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    inputs['data_pin']['preprocessing']['direction_labels_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='label hash'):
        run_abliteration(r, inputs, tmp_path / 'tampered')


def test_tiny_resume_rejects_changed_config_without_replacing_checkpoint(tmp_path, capsys):
    config = tmp_path / 'config.json'; config.write_text('{}')
    root = tmp_path / 'lab'
    args = ['--config', str(config), '--lab-root', str(root), '--tiny']
    main(args)
    before = (root / 'tiny-classic' / 'model.safetensors').read_bytes()
    main(args + ['--resume'])
    assert (root / 'tiny-classic' / 'model.safetensors').read_bytes() == before
    config.write_text('{"heretic_config":{"n_trials":5}}')
    with pytest.raises(ValueError, match='resume mismatch'):
        main(args + ['--resume'])
    assert (root / 'tiny-classic' / 'model.safetensors').read_bytes() == before
