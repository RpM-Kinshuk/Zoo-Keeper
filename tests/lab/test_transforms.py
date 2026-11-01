"""Transform invariants: wrong axes, GQA inverse, or lost bias break logits."""
import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen2Config, Qwen3Config, Olmo2Config

from lab.transforms import transform_state, check_equivalence


def tiny(family='qwen2', dtype=torch.float32):
    torch.set_num_threads(1)
    torch.manual_seed(41)
    cls = dict(llama=LlamaConfig, qwen2=Qwen2Config, qwen3=Qwen3Config, olmo2=Olmo2Config)[family]
    config = cls(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                 vocab_size=64, max_position_embeddings=32, tie_word_embeddings=True,
                 bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = AutoModelForCausalLM.from_config(config).to(dtype).eval()
    # Nonzero value biases ensure the rotation must include the affine term.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith('v_proj.bias'):
                param.copy_(torch.linspace(-.2, .2, param.numel()).to(dtype))
    return model


@pytest.mark.parametrize('family', ['llama', 'qwen2', 'qwen3', 'olmo2'])
@pytest.mark.parametrize('method', ['mlp_permutation', 'rescaling', 'value_rotation'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_symmetries_preserve_gqa_logits_and_ties(family, method, dtype):
    original = tiny(family, dtype)
    state = original.state_dict()
    snapshot = {name: value.clone() for name, value in state.items()}
    rng = torch.random.get_rng_state().clone()
    result = transform_state(state, original.config.to_dict(), dict(method=method, seed=7))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(state[name], snapshot[name]) for name in state)
    assert any(not torch.equal(result[name], state[name]) for name in state)
    assert result['model.embed_tokens.weight'].data_ptr() == result['lm_head.weight'].data_ptr()
    changed = copy.deepcopy(original)
    changed.load_state_dict(result, strict=True)
    tokens = [torch.tensor([[1, 5, 9, 2], [3, 8, 2, 7]])]
    report = check_equivalence(original, changed, tokens)
    tol = 2e-5 if dtype == torch.float32 else 2e-2
    assert report['passed'], report
    assert report['atol'] == report['rtol'] == tol
    assert report['tokens_sha256']
    assert report['max_abs_error'] >= 0


def test_targeted_rank_one_norm_before_and_after_cast():
    original = tiny()
    state = original.state_dict()
    recipe = dict(method='targeted_perturbation', seed=9, relative_frobenius_norm=1e-3, target_matrices=4)
    result = transform_state(state, original.config.to_dict(), recipe)
    changed = [key for key in state if not torch.equal(state[key], result[key])]
    assert len(changed) == 4
    assert all(key.endswith(('q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'o_proj.weight')) for key in changed)
    for key in changed:
        delta = result[key] - state[key]
        assert abs((delta.norm() / state[key].norm()).item() - 1e-3) < 1e-6
        assert torch.linalg.svdvals(delta)[1] < torch.linalg.svdvals(delta)[0] * 1e-3
    # Casting is measured, never used to silently inflate the requested update.
    bf = tiny(dtype=torch.bfloat16)
    result_bf = transform_state(bf.state_dict(), bf.config.to_dict(), recipe)
    assert all(t.dtype == torch.bfloat16 for t in result_bf.values())


@pytest.mark.parametrize('change', ['family', 'shape', 'extra', 'method', 'value_norm'])
def test_unsupported_layout_fails_explicitly(change):
    model = tiny()
    config = model.config.to_dict()
    state = model.state_dict()
    recipe = dict(method='value_rotation', seed=1)
    if change == 'family':
        config['model_type'] = 'unreviewed'
    elif change == 'shape':
        state['model.layers.0.self_attn.v_proj.weight'] = torch.ones(3, 32)
    elif change == 'extra':
        state['model.layers.0.self_attn.v_proj.lora_A.weight'] = torch.ones(2, 32)
    elif change == 'method':
        recipe['method'] = 'residual_rotation'
    else:
        state['model.layers.0.self_attn.v_norm.weight'] = torch.ones(8)
    with pytest.raises(NotImplementedError, match='(unreviewed|v_proj|residual_rotation|v_norm)'):
        transform_state(state, config, recipe)


def test_cli_dry_run_and_tiny_atomic_resume(tmp_path, capsys):
    from lab.scripts.laundering import main
    config = Path('lab/configs/laundering.json')
    root = tmp_path / 'lab'
    args = ['--config', str(config), '--lab-root', str(root)]
    main(args + ['--dry-run'])
    plan = json.loads(capsys.readouterr().out)
    assert len(plan['recipes']) == 16
    assert plan['launch'] == 'HUMAN'
    assert not root.exists()
    main(args + ['--tiny'])
    capsys.readouterr()
    outputs = sorted(root.glob('tiny-*/recipe.json'))
    assert len(outputs) == 4
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    main(args + ['--tiny', '--resume'])
    capsys.readouterr()
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    for path in outputs:
        recipe = json.loads(path.read_text())['recipe']
        assert 'audit' in recipe
        assert recipe['edge_type'] == 'laundering'
        if recipe['method'] == 'targeted_perturbation':
            assert not recipe['audit']['function_preserving_exact_arithmetic']
            for norms in recipe['audit']['perturbations'].values():
                assert abs(norms['precast_relative_norm'] - 1e-3) < 1e-6
                assert norms['postcast_relative_norm'] >= 0
        else:
            assert recipe['audit']['fp32']['passed']
            assert recipe['audit']['bfloat16']['passed']
        reloaded = AutoModelForCausalLM.from_pretrained(path.parent, local_files_only=True,
                    trust_remote_code=False, use_safetensors=True)
        assert reloaded.lm_head.weight.data_ptr() == reloaded.model.embed_tokens.weight.data_ptr()
    altered = tmp_path / 'changed.json'
    altered.write_text(json.dumps(dict(json.loads(config.read_text()), seed=8)))
    with pytest.raises(ValueError, match='resume mismatch'):
        main(['--config', str(altered), '--lab-root', str(root), '--tiny', '--resume'])
    assert before == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_source_hash_mismatch_fails_before_output(tmp_path):
    from lab.scripts.laundering import load_source
    from lab.artifacts import content_hash
    model = tiny()
    source = tmp_path / 'base'
    model.save_pretrained(source, safe_serialization=True)
    pin = dict(repo_id='org/base', revision='a' * 40, config_sha256=content_hash(json.loads((source / 'config.json').read_text())),
               status='pinned', license='apache-2.0', license_path='inputs/LICENSE',
               license_sha256='b' * 64, tokenizer_sha256='c' * 64,
               weight_sha256={'model.safetensors': '0' * 64})
    inputs = dict(model_path='base', model_pin=pin)
    with pytest.raises(ValueError, match='(weight|config).*hash'):
        load_source(inputs, tmp_path, 'base', 'org/base@' + 'a' * 40)


def test_equivalence_detects_logit_drift_and_restores_mode():
    original = tiny()
    changed = copy.deepcopy(original)
    with torch.no_grad():
        changed.model.embed_tokens.weight.zero_()
    original.train()
    changed.eval()
    result = check_equivalence(original, changed, [torch.tensor([[1, 7, 4]])])
    assert not result['passed']
    assert result['max_abs_error'] > 2e-5
    assert original.training and not changed.training


def test_local_source_requires_matching_extractor_key_and_hashed_sidecar(tmp_path):
    from lab.scripts.laundering import load_source, _file_hash
    from lab.artifacts import content_hash
    model = tiny()
    source = tmp_path / 'base'
    model.save_pretrained(source, safe_serialization=True)
    pin = dict(repo_id='org/base', revision='a' * 40,
               status='pinned', license='apache-2.0', license_path='inputs/LICENSE',
               license_sha256='b' * 64, tokenizer_sha256='c' * 64,
               config_sha256=content_hash(json.loads((source / 'config.json').read_text())),
               weight_sha256={'model.safetensors': _file_hash(source / 'model.safetensors')})
    identity = 'lab/base@local:' + 'd' * 64
    manifest = dict(model_key=identity, config_sha256=pin['config_sha256'], weight_sha256=pin['weight_sha256'])
    sidecar = tmp_path / 'base-manifest.json'
    sidecar.write_text(json.dumps(manifest))
    inputs = dict(model_path='base', model_pin=pin, local_parent=manifest,
                  local_manifest_path='base-manifest.json', local_manifest_sha256=_file_hash(sidecar))
    loaded = load_source(inputs, tmp_path, 'base', identity)
    assert torch.equal(loaded.model.layers[0].self_attn.v_proj.weight, model.model.layers[0].self_attn.v_proj.weight)
    sidecar.write_text('{}')
    with pytest.raises(ValueError, match='sidecar hash'):
        load_source(inputs, tmp_path, 'base', identity)
    with pytest.raises(ValueError, match='identity mismatch'):
        load_source(inputs, tmp_path, 'base', 'lab/base@local:' + 'e' * 64)
