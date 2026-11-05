"""Generation-only drift artifacts retain actual failed equivalence outcomes."""
import copy
import importlib
import json

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen2Config

from lab.transforms import transform_state
from lab.scripts.laundering import _audit


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    cfg = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
        max_position_embeddings=32, tie_word_embeddings=True,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = AutoModelForCausalLM.from_config(cfg).eval()
    with torch.no_grad():
        model.model.norm.weight.mul_(1000)
    batches = [torch.tensor([[1, 5, 9, 2], [3, 8, 2, 7]])]
    frozen = dict(model_id='base-laundering-value_rotation', method='value_rotation',
        seed=7, parents=['base'], parent_identities={'base': 'org/base@' + 'a' * 40},
        edge_type='laundering', inputs={})
    return model, frozen, batches


def api():
    return importlib.import_module('lab.transform_drift')


def recipe_for(module, frozen):
    return module.make_drift_recipe(frozen, {'path': 'runs/prior/result.json', 'sha256': 'b' * 64})


def test_real_negative_outcome_retained_and_v1_still_rejects():
    m = api(); model, frozen, batches = fixture()
    snapshot = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError, match='failed frozen-token'):
        _audit(copy.deepcopy(model), frozen, batches)
    recipe = recipe_for(m, frozen)
    edited, audit = m.audit_drift(model, recipe, batches, synthetic=True)
    assert audit['artifact_status'] == 'PENDING_SERIALIZATION'
    assert audit['construction_status'] == 'VERIFIED'
    assert audit['functional_equivalence'] == 'FAILED'
    assert not (audit['fp32']['passed'] and audit['bfloat16']['passed'])
    assert audit['fp32']['atol'] == audit['fp32']['rtol'] == 2e-5
    assert audit['bfloat16']['atol'] == audit['bfloat16']['rtol'] == 2e-2
    expected = transform_state(snapshot, model.config.to_dict(), frozen)
    assert all(torch.equal(edited.state_dict()[k], v.bfloat16()) for k, v in expected.items())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in snapshot.items())
    assert model.training is False and edited.training is False
    assert edited.get_input_embeddings().weight.data_ptr() == edited.get_output_embeddings().weight.data_ptr()


def test_atomic_safe_serialization_reload_and_immutable_resume(tmp_path):
    m = api(); model, frozen, batches = fixture(); recipe = recipe_for(m, frozen)
    edited, audit = m.audit_drift(model, recipe, batches, synthetic=True)
    target = m.save_drift(tmp_path, recipe, model, edited, audit, batches, synthetic=True)
    cohort = json.loads((target / 'cohort.json').read_text())
    assert cohort['schema_version'] == '2'
    assert cohort['functional_equivalence'] == 'FAILED'
    assert cohort['quota_credit'] == 0
    assert cohort['reloaded_bfloat16']['tokens_sha256'] == audit['bfloat16']['tokens_sha256']
    assert cohort['reloaded_state_exact'] and cohort['weight_sha256']
    assert not list(target.rglob('*.bin'))
    reloaded = AutoModelForCausalLM.from_pretrained(target, local_files_only=True,
        trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16)
    assert all(torch.equal(v, reloaded.state_dict()[k]) for k, v in edited.state_dict().items())
    assert m.save_drift(tmp_path, recipe, model, edited, audit, batches,
                        synthetic=True, resume=True) == target
    changed = copy.deepcopy(recipe); changed['seed'] += 1
    with pytest.raises(ValueError, match='mismatch'):
        m.save_drift(tmp_path, changed, model, edited, audit, batches, synthetic=True, resume=True)


def test_reload_rejects_changed_state_and_ties():
    m = api(); model, frozen, batches = fixture(); recipe = recipe_for(m, frozen)
    edited, _ = m.audit_drift(model, recipe, batches, synthetic=True)
    reloaded = copy.deepcopy(edited)
    with torch.no_grad():
        reloaded.model.layers[0].mlp.up_proj.weight[0, 0] += 1
    with pytest.raises(ValueError, match='state mismatch'):
        m.verify_reloaded(model, edited, reloaded, batches, synthetic=True)
    reloaded = copy.deepcopy(edited)
    reloaded.lm_head.weight = torch.nn.Parameter(reloaded.lm_head.weight.clone())
    with pytest.raises(ValueError, match='tied'):
        m.verify_reloaded(model, edited, reloaded, batches, synthetic=True)
    with pytest.raises(ValueError, match='state mismatch'):
        m.verify_reloaded(model, edited, copy.deepcopy(edited).float(), batches, synthetic=True)


def test_resume_rejects_relabelled_negative_and_changed_construction(tmp_path):
    m = api(); model, frozen, batches = fixture(); recipe = recipe_for(m, frozen)
    edited, audit = m.audit_drift(model, recipe, batches, synthetic=True)
    target = m.save_drift(tmp_path, recipe, model, edited, audit, batches, synthetic=True)
    sidecar = target / 'cohort.json'
    contents = json.loads(sidecar.read_text()); contents['functional_equivalence'] = 'PASSED'
    sidecar.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match='resume mismatch'):
        m.save_drift(tmp_path, recipe, model, edited, audit, batches, synthetic=True, resume=True)
    with torch.no_grad():
        edited.model.norm.weight[0] *= 2
    with pytest.raises(ValueError, match='state mismatch'):
        m.save_drift(tmp_path, recipe, model, edited, audit, batches, synthetic=True, resume=True)


def test_nonfinite_and_cpu_real_execution_fail_before_artifact(tmp_path):
    m = api(); model, frozen, batches = fixture(); recipe = recipe_for(m, frozen)
    with pytest.raises(ValueError, match='CUDA'):
        m.audit_drift(model, recipe, batches)
    with torch.no_grad():
        model.model.norm.weight[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        m.audit_drift(model, recipe, batches, synthetic=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('method', ['mlp_permutation', 'value_rotation'])
def test_same_full_transform_and_recorded_recipe_identity(method):
    m = api(); model, frozen, batches = fixture()
    frozen['method'] = method; frozen['model_id'] = 'base-laundering-' + method
    recipe = recipe_for(m, frozen)
    assert recipe['model_id'] == frozen['model_id'] + '-drift-v2'
    assert recipe['drift_protocol']['original_recipe'] == frozen
    edited, _ = m.audit_drift(model, recipe, batches, synthetic=True)
    expected = transform_state(model.state_dict(), model.config.to_dict(), frozen)
    assert all(torch.equal(edited.state_dict()[k], v.bfloat16()) for k, v in expected.items())


@pytest.mark.parametrize('metadata', ['config.json', 'generation_config.json', 'recipe.json', 'lineage.json'])
def test_resume_rejects_every_changed_checkpoint_metadata_byte(tmp_path, metadata):
    m = api(); model, frozen, batches = fixture(); recipe = recipe_for(m, frozen)
    edited, audit = m.audit_drift(model, recipe, batches, synthetic=True)
    target = m.save_drift(tmp_path, recipe, model, edited, audit, batches, synthetic=True)
    # JSON semantics and all old short-fixture logits remain identical. Bytes changed.
    path = target / metadata
    path.write_text(path.read_text() + ' \n')
    with pytest.raises(ValueError, match='resume mismatch'):
        m.save_drift(tmp_path, recipe, model, edited, audit, batches, synthetic=True, resume=True)


def test_config_behavior_change_rejected_even_if_short_logits_unchanged(tmp_path):
    m = api(); model, frozen, batches = fixture(); recipe = recipe_for(m, frozen)
    edited, audit = m.audit_drift(model, recipe, batches, synthetic=True)
    target = m.save_drift(tmp_path, recipe, model, edited, audit, batches, synthetic=True)
    path = target / 'config.json'; config = json.loads(path.read_text())
    config['max_position_embeddings'] *= 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='resume mismatch'):
        m.save_drift(tmp_path, recipe, model, edited, audit, batches, synthetic=True, resume=True)


def test_one_shot_and_empty_validation_rejected_at_every_api(tmp_path):
    m = api(); model, frozen, batches = fixture(); recipe = recipe_for(m, frozen)
    edited, audit = m.audit_drift(model, recipe, batches, synthetic=True)
    for tokens in [iter(batches), iter([]), []]:
        with pytest.raises(ValueError, match='validation batches'):
            m.audit_drift(model, recipe, tokens, synthetic=True)
    for tokens in [iter(batches), iter([]), []]:
        with pytest.raises(ValueError, match='validation batches'):
            m.verify_reloaded(model, edited, copy.deepcopy(edited), tokens, synthetic=True)
    for tokens in [iter(batches), iter([]), []]:
        with pytest.raises(ValueError, match='validation batches'):
            m.save_drift(tmp_path, recipe, model, edited, audit, tokens, synthetic=True)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('field,value', [('token_count', 0), ('token_count', 16), ('tokens_sha256', '0' * 64)])
def test_mismatched_dtype_token_contract_rejected_before_serialization(tmp_path, field, value):
    m = api(); model, frozen, batches = fixture(); recipe = recipe_for(m, frozen)
    edited, audit = m.audit_drift(model, recipe, batches, synthetic=True)
    audit['fp32'][field] = value
    with pytest.raises(ValueError, match='validation token mismatch'):
        m.save_drift(tmp_path, recipe, model, edited, audit, batches, synthetic=True)
    assert not list(tmp_path.iterdir())
