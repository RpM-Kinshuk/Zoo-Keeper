import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from lab.artifacts import content_hash
from lab.merges import mergekit_configs, merge_states, run_merge, prepare_recipe
from lab.registry import expand_inventory, materialize_tables


def test_inventory_merge_recipes():
    inventory = expand_inventory({})
    configs = mergekit_configs(inventory, {})
    assert len(configs) == 30
    recipes = inventory['merge']
    assert {r['method'] for r in recipes} == {'linear', 'slerp', 'ties', 'dare_ties'}
    assert {len(r['parents']) for r in recipes} == {2, 3, 4}
    assert any(r.get('t') == [0.2, 0.8] for r in recipes)
    for row in recipes:
        cfg = yaml.safe_load(configs[row['model_id']])
        assert cfg['merge_method'] == row['method']
        if row['method'] in {'ties', 'dare_ties'}:
            assert cfg['base_model'] == row['reference']
        elif row['method'] == 'slerp':
            assert cfg['base_model'] == row['parents'][0]
        else:
            assert 'base_model' not in cfg
        assert row['reference'] == row['anchor']
        ordered = sorted([r for r in inventory['lora'] if r['anchor'] == row['anchor']], key=lambda r: (r['task'], r['rank']))
        assert row['parents'] == [r['model_id'] for r in ordered][:len(row['parents'])]
        assert '/' not in configs[row['model_id']]


def states():
    gen = torch.Generator().manual_seed(21)
    return [{f'model.layers.{i}.weight': torch.randn(4, 4, generator=gen) for i in range(2)} for _ in range(5)]


def test_linear_weighted_state():
    a, b, base, *_ = states()
    actual = merge_states(dict(method='linear', parents=['a', 'b'], configured_weights=[0.25, 0.75], seed=1), [a, b], base)
    for k in a:
        torch.testing.assert_close(actual[k], 0.25*a[k]+0.75*b[k])


def test_cpu_merges_never_request_cuda_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU merge requested CUDA RNG state')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', forbidden)
    a, b, base, *_ = states()
    merge_states(dict(method='linear', parents=['a', 'b'],
                      configured_weights=[0.25, 0.75], seed=1), [a, b], base)


@pytest.mark.parametrize('t', [0., 1.])
def test_slerp_endpoints(t):
    a, b, base, *_ = states()
    actual = merge_states(dict(method='slerp', parents=['a', 'b'], t=t, seed=1), [a, b], base)
    for k in a:
        torch.testing.assert_close(actual[k], [a, b][int(t)][k])


def test_slerp_schedule_uses_layers():
    a, b, base, *_ = states()
    r = dict(method='slerp', parents=['a', 'b'], t=[0.2, 0.8], seed=1)
    actual = merge_states(r, [a, b], base)
    for i, k in enumerate(sorted(a)):
        expected = merge_states({**r, 't': [0.2, 0.8][i]}, [a, b], base)
        torch.testing.assert_close(actual[k], expected[k])


@pytest.mark.parametrize('method', ['ties', 'dare_ties'])
def test_task_arithmetic_matches_installed_mergekit(method):
    from mergekit.merge_methods import get
    from mergekit.merge_methods.base import TensorGroup
    a, b, c, d, base = states()
    recipe = dict(method=method, parents=['a', 'b', 'c', 'd'], density=0.8,
                  configured_weights=[0.25]*4, seed=12, sparsification_seed=42)
    actual = merge_states(recipe, [a, b, c, d], base)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        for k in sorted(base):
            group = TensorGroup.from_tensors([a[k], b[k], c[k], d[k], base[k]], ids=['a','b','c','d','base'], base_index=4, name=k)
            expected = get(method)([group], parameters={'weight': [0.25]*4, 'density': 0.8})[0]
            torch.testing.assert_close(actual[k], expected)


def fixture(root):
    a, b, base, *_ = states()
    rec = dict(model_id='merged', edge_type='merge', method='linear', anchor='base', reference='base',
               parents=['a','b'], configured_weights=[0.25,0.75], seed=1)
    inputs, identities = {}, {}
    for name, state in [('a',a),('b',b),('base',base)]:
        p = root/name
        p.mkdir()
        cfg = dict(model_type='llama', num_hidden_layers=2, hidden_size=4)
        (p/'config.json').write_text(json.dumps(cfg))
        save_file(state, str(p/'model.safetensors'))
        key = 'lab/'+name+'@local:'+'a'*64
        inputs[name] = dict(path=name, model_key=key, config_sha256=content_hash(cfg),
             config_file_sha256=hashlib.sha256((p/'config.json').read_bytes()).hexdigest(),
             weight_sha256={'model.safetensors': hashlib.sha256((p/'model.safetensors').read_bytes()).hexdigest()})
        identities[name] = key
    return {**rec, 'inputs': inputs, 'parent_identities': identities, 'synthetic_fixture': True}


def test_tiny_safe_checkpoint_and_hashes(tmp_path):
    rec = prepare_recipe(fixture(tmp_path))
    output = tmp_path/'out'
    run_merge(rec, tmp_path, output)
    assert load_file(str(output/'model.safetensors'))
    metadata = json.loads((output/'merge.json').read_text())
    assert metadata['yaml_sha256'] == hashlib.sha256(rec['mergekit_yaml'].encode()).hexdigest()
    assert metadata['config_sha256'] == content_hash(rec)
    assert str(tmp_path) not in json.dumps(metadata)


@pytest.mark.parametrize('mutation', ['shape','weight','extra','identity','config'])
def test_invalid_inputs_rejected_before_execution(tmp_path, monkeypatch, mutation):
    rec = fixture(tmp_path)
    if mutation == 'shape':
        p = tmp_path/'b/model.safetensors'
        save_file({'model.layers.0.weight': torch.ones(3,3), 'model.layers.1.weight':torch.ones(4,4)}, str(p))
        rec['inputs']['b']['weight_sha256']['model.safetensors'] = hashlib.sha256(p.read_bytes()).hexdigest()
    elif mutation == 'weight':
        rec['inputs']['b']['weight_sha256']['model.safetensors'] = '0'*64
    elif mutation == 'extra':
        save_file({'x':torch.ones(1)}, str(tmp_path/'b/extra.safetensors'))
    elif mutation == 'identity':
        rec['inputs']['b']['model_key'] = 'lab/wrong@local:'+'a'*64
    else:
        rec['inputs']['b']['config_sha256'] = '0'*64
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('execution before validation'))
    with pytest.raises(ValueError):
        run_merge(prepare_recipe(rec), tmp_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_table_configured_task_weights_are_explicit():
    inv = expand_inventory({})
    pins = {r['model_id']: dict(repo_id=r.get('repo_id', 'org/'+r['model_id']), revision='a'*40, license='apache-2.0', base_family=r.get('base_family','fixture')) for r in inv['bases']+inv['references']}
    _, edges = materialize_tables(inv, pins)
    for row in inv['merge']:
        selected = edges[edges.child == row['model_id']]
        assert selected[selected.role == 'base'].coefficient.tolist() == ['']
        expected = row.get('configured_weights', ['', ''])
        values = dict(zip(row['parents'], expected))
        for edge in selected[selected.role == 'parent'].itertuples():
            assert edge.coefficient == values[edge.parent]
            params = json.loads(edge.params_json)
            if row['method'] in {'ties','dare_ties'}:
                assert params['coefficient_semantics'] == 'configured_task_weight'


def test_cli_dry_run_opens_no_weights(tmp_path, monkeypatch, capsys):
    from lab.scripts.merge import main
    import safetensors
    monkeypatch.setattr(safetensors, 'safe_open', lambda *a, **k: pytest.fail('dry run opened weights'))
    assert main(['--config','lab/configs/merge.json','--lab-root',str(tmp_path),'--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan['recipes']) == 30
    assert not list(tmp_path.iterdir())


def test_cli_tiny_is_offline_complete_and_resumable(tmp_path, capsys):
    from lab.scripts.merge import main
    args = ['--config','lab/configs/merge.json','--lab-root',str(tmp_path),'--tiny']
    assert main(args) == 0
    assert main(args+['--resume']) == 0
    for method in ['linear','slerp','ties','dare_ties']:
        path = tmp_path/('tiny-merge-'+method)
        assert len(load_file(str(path/'model.safetensors'))) > 2
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True, trust_remote_code=False)
        assert model.config.num_hidden_layers == 2
        lineage = json.loads((path/'lineage.json').read_text())
        assert lineage['relation'] == 'merge'
        assert lineage['bases'][-1] == 'lab/tiny-merge-base'
        assert str(tmp_path) not in (path/'recipe.json').read_text()


def real_fixture(root):
    from lab.artifacts import _lineage
    rec = fixture(root)
    rec.pop('synthetic_fixture')
    identity = 'org/base@'+'c'*40
    rec['parent_identities']['base'] = identity
    rec['inputs']['base'].update(model_key=identity, status='pinned', repo_id='org/base', revision='c'*40,
        license='apache-2.0', license_path='inputs/license.txt', license_sha256='d'*64, tokenizer_sha256='e'*64)
    for name in rec['parents']:
        source = dict(model_id=name, anchor='base', edge_type='lora', steps=2,
                      parents=['base'], parent_identities={'base':identity}, inputs={})
        (root/name/'recipe.json').write_text(json.dumps(dict(model_id=name, recipe=source,
            config_hash=content_hash(source), input_hash=content_hash(dict(parents=source['parent_identities'], inputs={})))))
        (root/name/'lineage.json').write_text(json.dumps(_lineage(name, source)))
        (root/name/'training.json').write_text(json.dumps(dict(steps_completed=2, merged_adapter=True)))
    return rec


def normalized_config_fixture(root):
    from transformers import LlamaConfig
    rec = real_fixture(root)
    raw = dict(model_type='llama', hidden_size=4, intermediate_size=8,
               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
               vocab_size=16, rope_theta=100000., rope_scaling=None)
    exported = LlamaConfig(**raw).to_dict()
    for name in [*rec['parents'], rec['reference']]:
        config = raw if name == rec['reference'] else exported
        path = root / name / 'config.json'
        path.write_text(json.dumps(config))
        rec['inputs'][name].update(config_sha256=content_hash(config),
            config_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return rec


def test_real_inputs_accept_equivalent_serialized_config(tmp_path):
    from lab.merges import _checked_inputs
    rec = normalized_config_fixture(tmp_path)
    before = {name: (tmp_path / name / 'config.json').read_bytes()
              for name in rec['inputs']}
    assert set(_checked_inputs(rec, tmp_path)) == set(rec['inputs'])
    assert before == {name: (tmp_path / name / 'config.json').read_bytes()
                      for name in rec['inputs']}


@pytest.mark.parametrize('change', ['rope', 'heads', 'bias'])
def test_real_inputs_reject_architecture_changes_after_normalization(tmp_path, change):
    from lab.merges import _checked_inputs
    rec = normalized_config_fixture(tmp_path)
    path = tmp_path / rec['parents'][0] / 'config.json'
    config = json.loads(path.read_text())
    if change == 'rope':
        config['rope_parameters']['rope_theta'] = 200000.
    elif change == 'heads':
        config['num_attention_heads'] = 1
    else:
        config['mlp_bias'] = True
    path.write_text(json.dumps(config))
    rec['inputs'][rec['parents'][0]].update(config_sha256=content_hash(config),
        config_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='incompatible parent architecture configs'):
        _checked_inputs(rec, tmp_path)


@pytest.mark.parametrize('mutation', ['unfinished','input_hash','license','yaml_hash'])
def test_real_inputs_require_completed_provenance(tmp_path, mutation, monkeypatch):
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('execution before validation'))
    rec = real_fixture(tmp_path)
    if mutation == 'unfinished':
        (tmp_path/'a/training.json').write_text(json.dumps(dict(steps_completed=0)))
    elif mutation == 'input_hash':
        path = tmp_path/'a/recipe.json'
        data = json.loads(path.read_text()); data['input_hash'] = '0'*64
        path.write_text(json.dumps(data))
    elif mutation == 'license':
        rec['inputs']['base']['license'] = 'unknown'
    else:
        rec['mergekit_yaml_sha256'] = '0'*64
    with pytest.raises(ValueError):
        run_merge(rec, tmp_path, tmp_path/'out')


def test_real_cli_uses_transient_argument_array(tmp_path, monkeypatch):
    rec = real_fixture(tmp_path)
    called = []
    def runner(argv, **kwargs):
        called.append(argv)
        assert isinstance(argv, list) and kwargs.get('shell', False) is False
        cfg = yaml.safe_load(Path(argv[3]).read_text())
        assert 'base_model' not in cfg
        assert cfg['models'][0]['model'] == str(tmp_path/'a')
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        target = Path(argv[4])
        save_file({'model.layers.0.weight':torch.ones(4,4)}, str(target/'model.safetensors'))
        (target/'config.json').write_text(json.dumps({'_name_or_path':str(tmp_path/'a'),'model_type':'llama'}))
        (target/'mergekit_config.yml').write_text(Path(argv[3]).read_text())
    monkeypatch.setattr('subprocess.run', runner)
    run_merge(rec, tmp_path, tmp_path/'out')
    assert len(called) == 1 and not Path(called[0][3]).exists()
    assert str(tmp_path) not in (tmp_path/'out/config.json').read_text()
    assert not list((tmp_path/'out').glob('*.yml'))
