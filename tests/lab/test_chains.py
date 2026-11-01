import pandas as pd

from lab.chains import chain_paths, verify_minimum
from lab.registry import expand_inventory, materialize_tables
from lab.schema import MODEL_COLUMNS, EDGE_COLUMNS


def planned_tables():
    inventory = expand_inventory({})
    pins = {r['model_id']: {'repo_id': r.get('repo_id', 'test/' + r['model_id']),
                           'revision': 'a' * 40, 'license': 'apache-2.0',
                           'base_family': r.get('base_family', 'synthetic-reference')}
            for r in inventory['bases'] + inventory['references']}
    return materialize_tables(inventory, pins)


def test_three_counted_chains_reuse_generated_outputs():
    models, edges = planned_tables()
    completed = set(models.model_id)
    report = verify_minimum(models, edges, completed)
    assert report['checks']['chains']['count'] == 3
    assert report['checks']['chains']['verdict'] == 'PASS'
    assert report['checks']['lora']['count'] == 24
    assert report['checks']['sft']['count'] == 8
    assert report['checks']['decision']['count'] == 6
    assert report['public_verdict'] == 'PASS'
    assert report['checks']['laundering_combined']['verdict'] == 'PENDING_C_ATTESTATION'
    assert report['status'] == 'INCOMPLETE'
    assert len(models) == 166


def test_missing_intermediate_is_not_a_complete_chain():
    models, edges = planned_tables()
    report = verify_minimum(models, edges, set(models.model_id) - {'q25-05-sft-s0'})
    assert report['checks']['chains']['count'] == 2
    assert report['checks']['chains']['verdict'] == 'FAIL'
    assert 'q25-05-sft-s0' in report['missing_ids']
    assert report['public_verdict'] == 'FAIL'


def test_reference_only_edge_does_not_increase_depth():
    models = pd.DataFrame([
        ['b0', 'base', 'org/b0@' + 'a'*40, 'family', 'mit'],
        ['b1', 'base', 'org/b1@' + 'b'*40, 'family', 'mit'],
        ['c', 'derived', 'c', 'family', 'mit'],
        ['m', 'derived', 'm', 'family', 'mit'],
    ], columns=MODEL_COLUMNS)
    edges = pd.DataFrame([
        ['c', 'b1', 'sft', 'parent', '', '{}'],
        ['m', 'c', 'merge', 'parent', 1.0, '{}'],
        ['m', 'b0', 'merge', 'base', '', '{}'],
    ], columns=EDGE_COLUMNS)
    paths = chain_paths(models, edges)
    assert ['b1', 'c', 'm'] in paths
    assert not any(p[0] == 'b0' and p[-1] == 'm' for p in paths)
