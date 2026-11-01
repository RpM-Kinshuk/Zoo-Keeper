"""Offline synthetic tests; no fixture represents an actual model family."""
from itertools import combinations
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from lineage.tree import assemble_tree, pivot_flags, tree_diagnostics


def edge_frame(rows):
    return pd.DataFrame(rows, columns=['parent_key', 'child_key', 'cost', 'direction_status'])


def is_acyclic(frame):
    graph = nx.DiGraph()
    graph.add_edges_from(zip(frame.parent_key, frame.child_key))
    return nx.is_directed_acyclic_graph(graph)


@pytest.fixture
def tree_fixture():
    labels = ['r', 'a', 'b', 'c', 'd', 'e']
    truth = [('r', 'a'), ('r', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'e')]
    graph = nx.Graph()
    graph.add_weighted_edges_from([(u, v, i + 1) for i, (u, v) in enumerate(truth)])
    distances = np.array([[nx.shortest_path_length(graph, u, v, weight='weight')
                           for v in labels] for u in labels], dtype=float)
    cyclic = edge_frame([('r', 'a', 5., 'resolved'), ('r', 'b', 6., 'resolved'),
                         ('r', 'c', 7., 'resolved'), ('a', 'b', 0., 'resolved'),
                         ('b', 'c', 0., 'resolved'), ('c', 'a', 0., 'resolved')])
    return SimpleNamespace(inputs=dict(distances=distances, labels=labels, truth_edges=truth,
                                      predicted_edges=truth),
                           cyclic_inputs=dict(nodes=['r', 'a', 'b', 'c'], edges=cyclic, roots=['r']))


def test_tree_metric_and_cycle_contraction(tree_fixture):
    out = tree_diagnostics(**tree_fixture.inputs)
    assert out['four_point_violation_fraction'] == 0
    assert out['normalized_rf'] == 0
    graph = assemble_tree(**tree_fixture.cyclic_inputs)
    assert len(graph) == len(tree_fixture.cyclic_inputs['nodes']) - 1
    assert is_acyclic(graph)
    assert set(zip(graph.parent_key, graph.child_key)) == {('r', 'a'), ('a', 'b'), ('b', 'c')}
    assert graph.cost.sum() == 5


def test_pivot_thresholds_are_reported_not_decided():
    assert pivot_flags(four_point=0.21, normalized_rf=0.1) == ['four_point_gt_0.20']
    assert pivot_flags(four_point=0.1, normalized_rf=0.21) == ['normalized_rf_gt_0.20']
    assert pivot_flags(four_point=0.2, normalized_rf=0.2) == []
    assert pivot_flags(four_point=None, normalized_rf=None) == []


def test_non_tree_quartet():
    distances = np.ones((4, 4)) - np.eye(4)
    distances[0, 1] = distances[1, 0] = 3
    out = tree_diagnostics(distances, list('abcd'), [])
    assert out['four_point_violation_fraction'] == 1
    assert out['four_point_violations'] == out['four_point_quartets'] == 1
    assert out['normalized_rf'] is None
    assert out['rf_status'] == 'not_assembled'


def test_sibling_cosine():
    distances = np.array([[0, 1, 1], [1, 0, 2], [1, 2, 0]], dtype=float)
    out = tree_diagnostics(distances, ['r', 'a', 'b'], [('r', 'a'), ('r', 'b')])
    assert out['sibling_cosines'][0]['raw_cosine'] == 0
    distances[0, 1] = distances[1, 0] = 0
    out = tree_diagnostics(distances, ['r', 'a', 'b'], [('r', 'a'), ('r', 'b')])
    assert out['sibling_cosines'][0]['raw_cosine'] is None
    assert out['sibling_cosines'][0]['reason'] == 'zero_length_update'


def test_reticulation_removed_before_tree_fit():
    labels = ['r', 'a', 'b', 'm', 'd']
    truth = [('r', 'a'), ('r', 'b'), ('a', 'm'), ('b', 'm'), ('m', 'd')]
    out = tree_diagnostics(np.ones((5, 5)) - np.eye(5), labels, truth)
    assert out['excluded_reticulate_nodes'] == ['d', 'm']
    assert out['reticulate_fraction'] == 2 / 5
    assert out['four_point_quartets'] == 0


def test_symmetric_similarity_cannot_direct():
    out = assemble_tree(['r', 'a'], edge_frame([('r', 'a', 0., 'unresolved')]),
                        roots=['r'], times={'r': '2020-01-01', 'a': '2021-01-01'})
    assert out.empty
    assert out.attrs['direction_coverage'] == 0
    assert out.attrs['unresolved_edges'][0]['reason'] == 'unresolved_direction'
    assert out.attrs['unreachable_nodes'] == ['a']


def test_missing_prediction_leaves_are_explicit(tree_fixture):
    out = tree_diagnostics(**{**tree_fixture.inputs, 'predicted_edges': [('r', 'a')]})
    assert out['normalized_rf'] is None
    assert out['rf_status'] == 'incomplete_leaf_coverage'
    assert out['missing_predicted_leaves'] == ['c', 'd', 'e']


def test_rf_compares_predictions_not_truth_distances():
    labels = list('abcdxy')
    truth = [('x', 'a'), ('x', 'b'), ('x', 'y'), ('y', 'c'), ('y', 'd')]
    predicted = [('x', 'a'), ('x', 'c'), ('x', 'y'), ('y', 'b'), ('y', 'd')]
    out = tree_diagnostics(np.ones((6, 6)) - np.eye(6), labels, truth,
                           predicted_edges=predicted)
    assert out['normalized_rf'] == 1
    assert out['truth_split_count'] == out['predicted_split_count'] == 1


def test_sample_is_bounded_deterministic_and_tolerance_relative():
    labels = [str(i) for i in range(30)]
    distances = np.ones((30, 30)) - np.eye(30)
    first = tree_diagnostics(distances, labels, [])
    assert first == tree_diagnostics(distances, labels, [])
    assert first['four_point_quartets'] == 10000
    assert first['four_point_total_quartets'] == 27405
    assert first['four_point_sampled'] is True
    distances = np.ones((4, 4)) - np.eye(4)
    distances[0, 1] += 1e-7
    distances[1, 0] += 1e-7
    assert tree_diagnostics(distances, list('abcd'), [])['four_point_violations'] == 0


def test_assembly_roots_leaf_constraints_ties_and_coverage():
    edges = edge_frame([('r', 'a', 1., 'resolved'), ('s', 'a', 1., 'resolved'),
                        ('a', 'b', 0., 'resolved'), ('b', 'r', -5., 'resolved')])
    first = assemble_tree(['r', 's', 'a', 'b'], edges, roots=['s', 'r'], leaf_keys={'a'})
    second = assemble_tree(['b', 'a', 's', 'r'], edges.iloc[::-1], roots=['r', 's'], leaf_keys={'a'})
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 1
    assert first.attrs['unreachable_nodes'] == ['b']
    assert any(row['reason'] == 'leaf_parent' for row in first.attrs['excluded_edges'])
    assert not set(first.child_key) & {'r', 's'}


@pytest.mark.parametrize('distance', [np.array([[0, np.nan], [np.nan, 0]]),
                                      np.array([[0, -1], [-1, 0]]),
                                      np.array([[0, 1], [2, 0]])])
def test_invalid_distances_rejected(distance):
    with pytest.raises(ValueError):
        tree_diagnostics(distance, ['a', 'b'], [])
