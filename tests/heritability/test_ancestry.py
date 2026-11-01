import numpy as np
import pandas as pd
import pytest

from studies.heritability.ancestry import assemble_edges, build_groups, network_covariance, select_primary_deployments


def test_network_covariance():
    edges = pd.DataFrame([
        {'parent_id':'r','child_id':'a','coefficient':1.,'branch_length':1.},
        {'parent_id':'r','child_id':'b','coefficient':1.,'branch_length':1.},
        {'parent_id':'a','child_id':'m','coefficient':.5,'branch_length':1.},
        {'parent_id':'b','child_id':'m','coefficient':.5,'branch_length':1.},
    ])
    k = network_covariance(['r','a','b','m'], edges)
    assert np.allclose(k, [[1,1,1,1],[1,2,1,1.5],[1,1,2,1.5],[1,1.5,1.5,2.5]])
    assert np.linalg.eigvalsh(k).min() >= -1e-12
    order = ['m', 'r', 'b']
    assert np.allclose(network_covariance(order, edges), k[np.ix_([3, 0, 2], [3, 0, 2])])


def test_declared_edges_preserve_direction_and_report_unsupported_descendants():
    atlas = {'edges': [{'source': 'org/r', 'target': 'org/a'},
                       {'source': 'org/r', 'target': 'org/b'},
                       {'source': 'org/a', 'target': 'org/u'},
                       {'source': 'org/b', 'target': 'org/u'},
                       {'source': 'org/u', 'target': 'org/v'}]}
    pins = pd.DataFrame({'repo_id': ['org/r'], 'revision': ['a' * 40]})
    edges, excluded = assemble_edges(atlas, pd.DataFrame(), pins)
    assert ('org/r', 'org/a') in set(zip(edges.parent_id, edges.child_id))
    assert set(edges.evidence_tier) == {'declared'}
    assert edges.loc[edges.parent_id.eq('org/r'), 'parent_key'].eq('org/r@' + 'a' * 40).all()
    assert set(excluded.loc[excluded.reason.eq('merge_coefficients_unavailable'), 'record_id']) == {'org/u', 'org/v'}
    assert len(edges) == 5


def test_grouping_unites_ancestry_and_duplicates_before_splits():
    models = pd.DataFrame({'model_key': ['r@'+'a'*40, 'a@'+'a'*40, 's@'+'b'*40, 'b@'+'b'*40],
                           'repo_id': ['r', 'a', 's', 'b']})
    edges = pd.DataFrame({'parent_id': ['r', 's'], 'child_id': ['a', 'b']})
    duplicates = pd.DataFrame({'model_key': [models.model_key[1], models.model_key[3]],
                               'duplicate_group': ['same', 'same']})
    groups = build_groups(models, edges, duplicates)
    assert groups.split_group.nunique() == 1
    assert groups.root_group.nunique() == 2
    assert len(groups) == 4
    assert groups.set_index('model_key').loc[models.model_key[1], 'duplicate_group'] == 'same'


def test_merge_joins_both_root_groups():
    models = pd.DataFrame({'model_key': ['r@'+'a'*40, 's@'+'b'*40, 'm@'+'c'*40],
                           'repo_id': ['r', 's', 'm']})
    edges = pd.DataFrame({'parent_id': ['r', 's'], 'child_id': ['m', 'm']})
    assert build_groups(models, edges, pd.DataFrame()).split_group.nunique() == 1


def test_missing_nodes_and_duplicate_deployments_stay_countable():
    models = pd.DataFrame({'model_key': ['a@'+'a'*40, 'a@'+'a'*40, 'b@'+'b'*40],
                           'repo_id': ['a', 'a', 'b']})
    edges = pd.DataFrame({'parent_id': ['unobserved'], 'child_id': ['a']})
    groups = build_groups(models, edges, pd.DataFrame())
    assert len(groups) == 2
    assert groups.split_group.nunique() == 2


@pytest.mark.parametrize('coefficient', [-.1, 1.2, float('inf')])
def test_invalid_merge_coefficients_are_rejected(coefficient):
    edges = pd.DataFrame({'parent_id': ['a', 'b'], 'child_id': ['m', 'm'],
                           'coefficient': [coefficient, 1-coefficient], 'branch_length': [1., 1.]})
    with pytest.raises(ValueError, match='coefficient'):
        network_covariance(['a', 'b', 'm'], edges)


def test_unknown_merge_and_descendant_are_not_silently_simplified():
    edges = pd.DataFrame({'parent_id': ['r', 's', 'm'], 'child_id': ['m', 'm', 'c'],
                           'coefficient': [np.nan, np.nan, 1.], 'branch_length': [1., 1., 1.]})
    with pytest.raises(ValueError, match='coefficient'):
        network_covariance(['r', 's', 'm', 'c'], edges)


def test_cycles_and_invalid_lengths_are_rejected():
    edges = pd.DataFrame({'parent_id': ['r', 's'], 'child_id': ['s', 'r']})
    with pytest.raises(ValueError, match='cycle'):
        network_covariance(['r', 's'], edges)
    with pytest.raises(ValueError, match='cycle'):
        assemble_edges({'edges': [{'source':'r', 'target':'s'}, {'source':'s', 'target':'r'}]}, pd.DataFrame(), pd.DataFrame())
    with pytest.raises(ValueError, match='branch'):
        network_covariance(['r', 's'], pd.DataFrame({'parent_id':['r'], 'child_id':['s'], 'branch_length':[-1.]}))


def test_empty_network_and_single_root():
    assert network_covariance([], pd.DataFrame()).shape == (0, 0)
    assert np.array_equal(network_covariance(['r'], pd.DataFrame()), [[1.]])


def test_unknown_metadata_parent_does_not_join_unrelated_roots():
    metadata = pd.DataFrame({'model_id': ['a', 'b'], 'source_model': ['unknown', 'unknown'],
                             'base_model_relation': ['unknown', 'unknown']})
    edges, excluded = assemble_edges({'edges': []}, metadata, pd.DataFrame())
    assert edges.empty
    assert set(excluded.reason) == {'parent_unavailable'}
    assert len(excluded) == 2


def test_casefolded_ancestry_matching_preserves_output_keys():
    models = pd.DataFrame({'model_key':['ORG/A@'+'a'*40,'org/r@'+'b'*40], 'repo_id':['ORG/A','org/r']})
    edges = pd.DataFrame({'parent_id':['org/r'],'child_id':['org/a']})
    groups = build_groups(models,edges,pd.DataFrame())
    assert groups.split_group.nunique() == 1
    assert set(groups.model_key) == set(models.model_key)


def test_repeated_edge_evidence_is_order_independent_and_conflicts_rejected():
    records=[{'source':'r','target':'c'},
             {'source':'r','target':'c','branch_length':2.,'branch_length_source':'configured'}]
    with pytest.raises(ValueError,match='conflicting.*edge'):
        assemble_edges({'edges':records},pd.DataFrame(),pd.DataFrame())
    raw=pd.DataFrame({'parent_id':['r','R'],'child_id':['c','C'],'branch_length':[1.,2.]})
    with pytest.raises(ValueError,match='conflicting.*edge'):
        network_covariance(['r','c'],raw)
    records=[{'source':'r','target':'m'}, {'source':'r','target':'m','coefficient':.4,'coefficient_source':'config'},
             {'source':'s','target':'m','coefficient':.6,'coefficient_source':'config'}]
    a,_=assemble_edges({'edges':records},pd.DataFrame(),pd.DataFrame())
    b,_=assemble_edges({'edges':records[::-1]},pd.DataFrame(),pd.DataFrame())
    pd.testing.assert_frame_equal(a,b)
    assert a.loc[a.parent_id.eq('r'),'coefficient'].iloc[0] == .4


def test_missing_child_is_audited_and_primary_deployment_is_explicit():
    _,excluded=assemble_edges({'edges':[]},pd.DataFrame({'model_id':[''],'source_model':['r']}),pd.DataFrame())
    assert 'missing_endpoint' in set(excluded.reason)
    scores=pd.DataFrame({'model_key':['a@'+'a'*40]*2, 'benchmark':['v2']*2,'task':['math']*2,
                         'precision':['bf16','fp16'],'quantization':['none']*2,'harness':['h1']*2,'record_id':['a','b']})
    primary,sensitivity,excluded=select_primary_deployments(scores,{'precision':'bf16','quantization':'none','harness':None})
    assert list(primary.record_id) == ['a']
    assert list(sensitivity.record_id) == ['b']
    assert 'nonprimary_deployment' in set(excluded.reason)
