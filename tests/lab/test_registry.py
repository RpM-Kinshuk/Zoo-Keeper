import json
from pathlib import Path

import pandas as pd
import pytest
from safetensors.numpy import save_file
import numpy as np

from lab.schema import LAB_SCHEMA_VERSION, MODEL_COLUMNS, EDGE_COLUMNS
from lab.registry import (validate_tables, expand_inventory, materialize_tables,
                          topological_order, write_tables, read_tables, verify_tables)
from lab.artifacts import write_checkpoint


def test_registry_contract(tables):
    models, edges = tables
    assert LAB_SCHEMA_VERSION == '1'
    assert MODEL_COLUMNS == list(models.columns)
    assert EDGE_COLUMNS == list(edges.columns)
    validate_tables(models, edges)
    assert topological_order(models, edges).index('b') < topological_order(models, edges).index('c')


@pytest.mark.parametrize('case,offender', [
    ('duplicate', 'b'), ('missing', 'absent'), ('self', 'c'), ('cycle', 'b'),
    ('independent', 'i'), ('unknown', 'c'), ('nonfinite', 'c'),
    ('nan', 'c'), ('json', 'c'), ('absolute', 'c'), ('escape', 'c'),
])
def test_invalid_tables(tables, case, offender):
    models, edges = tables
    if case == 'duplicate':
        models = pd.concat([models, models.iloc[[0]]], ignore_index=True)
    elif case == 'missing':
        edges.loc[0, 'parent'] = 'absent'
    elif case == 'self':
        edges.loc[0, 'parent'] = 'c'
    elif case == 'cycle':
        edges.loc[1] = ['b', 'c', 'sft', 'parent', '', '{}']
    elif case == 'independent':
        edges.loc[0, 'child'] = 'i'
    elif case == 'unknown':
        edges.loc[0, 'edge_type'] = 'new_type'
    elif case == 'nonfinite':
        edges.loc[0, ['edge_type', 'coefficient']] = ['merge', float('inf')]
    elif case == 'nan':
        edges.loc[0, ['edge_type', 'coefficient']] = ['merge', float('nan')]
    elif case == 'json':
        edges.loc[0, 'params_json'] = '{invalid'
    elif case == 'absolute':
        models.loc[1, 'path'] = '/absolute'
    elif case == 'escape':
        models.loc[1, 'path'] = '../escape'
    with pytest.raises(ValueError, match=offender):
        validate_tables(models, edges)


def inventory():
    return expand_inventory(json.loads((Path(__file__).parents[2] / 'lab/configs/inventory.json').read_text()))


def test_inventory_counts():
    inv = inventory()
    counts = {'lora': 24, 'sft': 8, 'dpo': 8, 'merge': 30, 'abliteration': 8,
              'quantization_sim': 12, 'laundering': 16, 'decision': 6}
    assert {key: len(inv[key]) for key in counts} == counts
    children = [row for key in counts for row in inv[key]]
    assert len({r['model_id'] for r in children}) == 112
    assert len(inv['bases']) == 7
    decoders = [r for r in inv['bases'] if r['model_id'] != 'modernbert']
    assert len(decoders) == 6
    assert len({r['base_family'] for r in decoders}) == 4
    assert len(inv['references']) == 47
    assert all(row['status'] == 'unresolved' for row in inv['references'])
    for anchor in ['q25-05', 'q3-06', 'smol-360']:
        sft = next(r for r in inv['sft'] if r['anchor'] == anchor and r['seed_index'] == 0)
        dpo = next(r for r in inv['dpo'] if r['parents'] == [sft['model_id']])
        quant = next(r for r in inv['quantization_sim'] if r['anchor'] == anchor and r['bits'] == 8)
        assert sft['parents'] == [anchor]
        assert quant['parents'] == [dpo['model_id']]
    assert {r['edge_type'] for r in inv['decision']} == {'sft', 'lora'}
    assert children == [r for key in counts for r in expand_inventory({})[key]]
    assert sorted(r['seed'] for r in children) == list(range(20261028, 20261028 + 112))
    for merge in inv['merge']:
        assert len(merge['parents']) in [2, 3, 4]
        assert merge['reference'] == merge['anchor']
        assert all(p.startswith(merge['anchor'] + '-lora-') for p in merge['parents'])
        if merge['method'] != 'linear':
            assert not merge.get('coefficients')


def test_materialize_requires_real_pins():
    inv = inventory()
    with pytest.raises(ValueError, match='unresolved'):
        materialize_tables(inv, {})
    pins = {r['model_id']: {'repo_id': r.get('repo_id', 'test/' + r['model_id']),
                           'revision': 'a' * 40, 'license': 'apache-2.0',
                           'base_family': r.get('base_family', 'synthetic-reference')}
            for r in inv['bases'] + inv['references']}
    models, edges = materialize_tables(inv, pins)
    assert len(models) == 166
    assert models.model_id.tolist() == sorted(models.model_id)
    assert list(edges[['child', 'parent', 'role']].itertuples(index=False, name=None)) == sorted(
        edges[['child', 'parent', 'role']].itertuples(index=False, name=None))
    validate_tables(models, edges)
    assert not set(models.loc[models.kind == 'independent', 'model_id']) & set(edges.child)
    for row in edges.itertuples():
        if row.role == 'base' or row.edge_type != 'merge' or json.loads(row.params_json)['method'] == 'slerp':
            assert row.coefficient == ''
    pins['q25-05']['revision'] = 'main'
    with pytest.raises(ValueError, match='q25-05'):
        materialize_tables(inv, pins)


def recipe():
    return {'edge_type': 'sft', 'parents': ['b'],
            'parent_identities': {'b': 'org/base@' + 'a' * 40},
            'seed': 20261028, 'inputs': {'dataset_hash': 'c' * 64}}


def marker_writer(directory):
    save_file({'marker': np.array([1], dtype=np.float32)}, directory / 'model.safetensors')


def test_atomic_resume(tmp_path):
    def broken(directory):
        marker_writer(directory)
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        write_checkpoint('c', tmp_path, recipe(), broken)
    assert not (tmp_path / 'c').exists()
    output = write_checkpoint('c', tmp_path, recipe(), marker_writer)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert json.loads((output / 'lineage.json').read_text()) == {
        'name': 'lab/c', 'bases': ['org/base@' + 'a' * 40], 'relation': 'sft'}
    metadata = json.loads((output / 'recipe.json').read_text())
    assert metadata['parent_identities'] == recipe()['parent_identities']
    assert str(tmp_path) not in (output / 'recipe.json').read_text()
    assert metadata['config_hash'] and metadata['input_hash']
    def must_not_run(directory):
        pytest.fail('resume invoked writer')
    assert write_checkpoint('c', tmp_path, recipe(), must_not_run, resume=True) == output
    for changed in [dict(recipe(), seed=5), dict(recipe(), inputs={'dataset_hash': 'd' * 64}),
                    dict(recipe(), parent_identities={'b': 'org/base@' + 'b' * 40})]:
        with pytest.raises(ValueError, match='c'):
            write_checkpoint('c', tmp_path, changed, marker_writer, resume=True)
    with pytest.raises(ValueError, match='c'):
        write_checkpoint('c', tmp_path, recipe(), marker_writer)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_portable_local_lineage(tmp_path):
    rec = recipe()
    rec['parents'] = ['b', 'a']
    rec['reference'] = 'ref'
    rec['edge_type'] = 'merge'
    rec['parent_identities'] = {p: f'lab/{p}@local:' + 'c' * 64 for p in ['a', 'b', 'ref']}
    out = write_checkpoint('c', tmp_path, rec, marker_writer)
    assert json.loads((out / 'lineage.json').read_text())['bases'] == ['lab/b', 'lab/a', 'lab/ref']


def test_symlink_escape_and_verify(tmp_path, tables):
    root = tmp_path / 'lab'
    root.mkdir()
    external = tmp_path / 'external'
    external.mkdir()
    (root / 'c').symlink_to(external, target_is_directory=True)
    models, edges = tables
    validate_tables(models, edges)
    with pytest.raises(ValueError, match='c'):
        verify_tables(models, edges, root)
    with pytest.raises(ValueError, match='c'):
        write_checkpoint('c', root, recipe(), marker_writer)
    (root / 'c').unlink()
    report = verify_tables(models, edges, root)
    assert report['missing_ids'] == ['c']
    write_checkpoint('c', root, recipe(), marker_writer)
    assert verify_tables(models, edges, root)['completed_ids'] == ['c']


def test_atomic_table_snapshot(tmp_path, tables, monkeypatch):
    import lab.registry as registry
    models, edges = tables
    write_tables(models, edges, tmp_path)
    old_pointer = (tmp_path / 'current.json').read_bytes()
    original = registry.os.replace
    def interrupted(src, dst):
        if Path(dst).name == 'current.json':
            raise RuntimeError('interrupted publication')
        return original(src, dst)
    altered = models.copy()
    altered.loc[1, 'license'] = 'mit'
    monkeypatch.setattr(registry.os, 'replace', interrupted)
    with pytest.raises(RuntimeError, match='publication'):
        write_tables(altered, edges, tmp_path)
    assert (tmp_path / 'current.json').read_bytes() == old_pointer
    read_models, read_edges = read_tables(tmp_path)
    pd.testing.assert_frame_equal(read_models, models)
    pd.testing.assert_frame_equal(read_edges, edges)


@pytest.mark.parametrize('bad', ['/absolute', '../escape', 'a/../../b', 'a\\b'])
def test_artifact_bad_id(tmp_path, bad):
    with pytest.raises(ValueError):
        write_checkpoint(bad, tmp_path, recipe(), marker_writer)


def test_verify_parent_pin_matches_registry(tmp_path, tables):
    changed = recipe()
    changed['parent_identities'] = {'b': 'org/base@' + 'b' * 40}
    write_checkpoint('c', tmp_path, changed, marker_writer)
    report = verify_tables(*tables, tmp_path)
    assert report['completed_ids'] == []
    assert report['incomplete'][0]['model_id'] == 'c'
    assert 'pin differs' in report['incomplete'][0]['reason']


@pytest.mark.parametrize('use_hub_identity', [False, True])
def test_verify_local_parent_requires_registered_local_identity(tmp_path, tables, use_hub_identity):
    models, edges = tables
    models.loc[len(models)] = ['d', 'derived', 'd', 'tiny', 'apache-2.0']
    edges.loc[len(edges)] = ['d', 'c', 'dpo', 'parent', '', '{}']
    write_checkpoint('c', tmp_path, recipe(), marker_writer)
    identity = 'org/unrelated@' + 'a' * 40 if use_hub_identity else 'lab/c@local:' + 'b' * 64
    child_recipe = {'edge_type': 'dpo', 'parents': ['c'], 'parent_identities': {'c': identity}}
    write_checkpoint('d', tmp_path, child_recipe, marker_writer)
    report = verify_tables(models, edges, tmp_path)
    if use_hub_identity:
        assert report['completed_ids'] == ['c']
        assert report['incomplete'][0]['model_id'] == 'd'
        assert 'local identity' in report['incomplete'][0]['reason']
    else:
        assert report['completed_ids'] == ['c', 'd']
        assert report['incomplete'] == []


@pytest.mark.parametrize('swap_roles', [False, True])
def test_verify_merge_preserves_contributor_and_reference_roles(tmp_path, tables, swap_roles):
    models, edges = tables
    models.loc[len(models)] = ['m', 'derived', 'm', 'tiny', 'apache-2.0']
    for parent, role in [('c', 'parent'), ('i', 'parent'), ('b', 'base')]:
        edges.loc[len(edges)] = ['m', parent, 'merge', role, '', '{}']
    write_checkpoint('c', tmp_path, recipe(), marker_writer)
    merge_recipe = {'edge_type': 'merge', 'parents': ['b', 'i'] if swap_roles else ['c', 'i'],
                    'reference': 'c' if swap_roles else 'b', 'parent_identities': {
                        'b': 'org/base@' + 'a' * 40,
                        'c': 'lab/c@local:' + 'b' * 64,
                        'i': 'org/independent@' + 'b' * 40}}
    write_checkpoint('m', tmp_path, merge_recipe, marker_writer)
    report = verify_tables(models, edges, tmp_path)
    if swap_roles:
        assert report['completed_ids'] == ['c']
        assert report['incomplete'][0]['model_id'] == 'm'
        assert 'roles' in report['incomplete'][0]['reason']
    else:
        assert report['completed_ids'] == ['c', 'm']
        assert report['incomplete'] == []
