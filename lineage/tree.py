"""Observed-node tree diagnostics and directed minimum-cost forests.

Distances supplied to sibling diagnostics must be *squared* sketch distances.
Tree additivity is a diagnostic, never a recovery guarantee. Edge costs are
frozen upstream negative-log parent plausibilities, optionally including
explicitly recorded prior assumptions; timestamps never establish direction.
"""
from itertools import combinations
from math import comb

import networkx as nx
import numpy as np
import pandas as pd

_SEED = 20261028
_MAX_QUARTETS = 10000
_COLUMNS = ['parent_key', 'child_key', 'cost', 'direction_status']


def _graph(edges):
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError('diagnostic edges must form a directed acyclic graph')
    return graph


def _reticulate(graph):
    excluded = set()
    for node, degree in graph.in_degree():
        if degree > 1:
            excluded.add(node)
            excluded.update(nx.descendants(graph, node))
    return excluded


def _splits(graph, leaves):
    """Canonical unrooted nontrivial splits, including degree-two suppression."""
    undirected = graph.to_undirected()
    if not nx.is_tree(undirected):
        return None
    result = set()
    for a, b in list(undirected.edges):
        undirected.remove_edge(a, b)
        side = set(nx.node_connected_component(undirected, a)) & leaves
        other = leaves - side
        undirected.add_edge(a, b)
        if min(len(side), len(other)) > 1:
            sides = [tuple(sorted(side)), tuple(sorted(other))]
            result.add(min(sides))
    return result


def tree_diagnostics(distances, labels, truth_edges, *, predicted_edges=None, tolerance=1e-6):
    """Return diagnostic counts, observed-leaf nRF, and raw sibling cosines.

    Reticulate nodes and descendants in either supplied DAG are excluded from
    all tree-only metrics. Quartet sampling is uniform without replacement and
    seeded. Missing predictions leave nRF null rather than scoring truth itself.
    """
    distance = np.asarray(distances, dtype=np.float64)
    n = len(labels)
    if len(set(labels)) != n or distance.shape != (n, n):
        raise ValueError('unique labels and a matching square distance matrix are required')
    if (not np.isfinite(distance).all() or (distance < 0).any()
            or not np.allclose(distance, distance.T, rtol=0, atol=1e-12)
            or not np.allclose(np.diag(distance), 0, rtol=0, atol=1e-12)):
        raise ValueError('distances must be finite, nonnegative, symmetric, and zero-diagonal')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    truth = _graph(truth_edges)
    predicted = None if predicted_edges is None else _graph(predicted_edges)
    excluded = _reticulate(truth)
    if predicted is not None:
        excluded |= _reticulate(predicted)
    observed = set(labels) - excluded
    indices = [i for i, key in enumerate(labels) if key in observed]
    total = comb(len(indices), 4) if len(indices) >= 4 else 0
    sampled = total > _MAX_QUARTETS
    if sampled:
        rng = np.random.default_rng(_SEED)
        quartets = set()
        while len(quartets) < _MAX_QUARTETS:
            quartets.add(tuple(sorted(rng.choice(indices, 4, replace=False).tolist())))
        quartets = sorted(quartets)
    else:
        quartets = list(combinations(indices, 4))
    violations = 0
    for a, b, c, d in quartets:
        sums = sorted([distance[a, b] + distance[c, d], distance[a, c] + distance[b, d],
                       distance[a, d] + distance[b, c]])
        violations += int(sums[2] - sums[1] > tolerance * max(1., sums[2]))
    out = dict(four_point_violation_fraction=violations / len(quartets) if quartets else None,
               four_point_violations=violations, four_point_quartets=len(quartets),
               four_point_total_quartets=total, four_point_sampled=sampled,
               four_point_seed=_SEED, normalized_rf=None, rf_status='not_assembled',
               reticulate_fraction=len(set(labels) & excluded) / n if n else None,
               excluded_reticulate_nodes=sorted(set(labels) & excluded),
               observed_node_count=len(observed), missing_predicted_leaves=[],
               extra_predicted_leaves=[], truth_split_count=None, predicted_split_count=None,
               sibling_cosines=[], distance_kind='squared_sketch', recovery_claim=False)
    truth = truth.subgraph(set(truth) - excluded).copy()
    if predicted is not None:
        predicted = predicted.subgraph(set(predicted) - excluded).copy()
        leaves = {key for key in observed & set(truth) if truth.out_degree(key) == 0}
        predicted_leaves = {key for key in observed & set(predicted) if predicted.out_degree(key) == 0}
        out['missing_predicted_leaves'] = sorted(leaves - predicted_leaves)
        out['extra_predicted_leaves'] = sorted(predicted_leaves - leaves)
        out['leaf_coverage'] = len(leaves & predicted_leaves) / len(leaves) if leaves else None
        if not leaves or predicted_leaves != leaves:
            out['rf_status'] = 'incomplete_leaf_coverage'
        else:
            truth_splits, predicted_splits = _splits(truth, leaves), _splits(predicted, leaves)
            if truth_splits is None or predicted_splits is None:
                out['rf_status'] = 'not_a_tree'
            else:
                denominator = len(truth_splits) + len(predicted_splits)
                out.update(normalized_rf=(len(truth_splits ^ predicted_splits) / denominator
                                          if denominator else 0.), rf_status='ok',
                           truth_split_count=len(truth_splits), predicted_split_count=len(predicted_splits))
    else:
        out['leaf_coverage'] = None
    position = {key: i for i, key in enumerate(labels)}
    for parent in sorted(truth):
        if parent not in observed:
            continue
        for a, b in combinations(sorted(set(truth.successors(parent)) & observed), 2):
            d1, d2 = distance[position[parent], position[a]], distance[position[parent], position[b]]
            d12 = distance[position[a], position[b]]
            cosine = float((d1 + d2 - d12) / (2 * np.sqrt(d1 * d2))) if d1 > 0 and d2 > 0 else None
            out['sibling_cosines'].append(dict(parent_key=parent, child_keys=[a, b],
                                               raw_cosine=cosine,
                                               reason='ok' if cosine is not None else 'zero_length_update'))
    return out


def assemble_tree(nodes, edges, *, roots, times=None, leaf_keys=None):
    """Minimum-cost directed forest rooted only at allowed observed roots.

    Only ``direction_status='resolved'`` certifies asymmetric evidence. Unknown
    rows remain in ``attrs['unresolved_edges']``. A super-root forces permitted
    roots; unreachable observed nodes remain explicit abstentions. Sorted node
    and edge insertion provides deterministic key-order resolution of equal
    costs without perturbing the supplied scientific costs. Times are recorded
    as unused unless their prior has already been incorporated into input cost.
    """
    nodes, roots = sorted(set(nodes)), sorted(set(roots))
    if not set(roots) <= set(nodes):
        raise ValueError('roots must belong to nodes')
    if not set(_COLUMNS) <= set(edges.columns):
        raise ValueError('directed edges require parent_key, child_key, cost, direction_status')
    leaves = set() if leaf_keys is None else set(leaf_keys)
    if not leaves <= set(nodes):
        raise ValueError('leaf_keys must belong to nodes')
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    unresolved, excluded = [], []
    rows = edges[_COLUMNS].to_dict('records')
    for row in sorted(rows, key=lambda r: (r['parent_key'], r['child_key'], str(r['direction_status']), float(r['cost']))):
        parent, child = row['parent_key'], row['child_key']
        if parent not in graph or child not in graph:
            raise ValueError('edge endpoint missing from nodes')
        if not np.isfinite(float(row['cost'])):
            raise ValueError('edge cost must be finite')
        if row['direction_status'] != 'resolved':
            unresolved.append({**row, 'reason': 'unresolved_direction'})
            continue
        reason = ('self_edge' if parent == child else 'allowed_root' if child in roots
                  else 'leaf_parent' if parent in leaves else None)
        if reason:
            excluded.append({**row, 'reason': reason})
            continue
        if not graph.has_edge(parent, child) or row['cost'] < graph[parent][child]['weight']:
            graph.add_edge(parent, child, weight=float(row['cost']), row=row)
    reachable = set(roots)
    for root in roots:
        reachable.update(nx.descendants(graph, root))
    fitted = graph.subgraph(reachable).copy()
    # Object identity guarantees this synthetic node cannot collide with a key.
    super_root = object()
    fitted.add_node(super_root)
    for root in roots:
        fitted.add_edge(super_root, root, weight=0.)
    selected = []
    if reachable:
        arborescence = nx.minimum_spanning_arborescence(fitted, attr='weight', preserve_attrs=True)
        selected = [attributes['row'] for parent, _, attributes in arborescence.edges(data=True)
                    if parent is not super_root]
    result = pd.DataFrame(sorted(selected, key=lambda r: (r['parent_key'], r['child_key'])), columns=_COLUMNS)
    result.attrs.update(unresolved_edges=unresolved, excluded_edges=excluded,
                        unreachable_nodes=sorted(set(nodes) - reachable),
                        direction_coverage=(sum(r['direction_status'] == 'resolved' for r in rows) / len(rows)
                                            if rows else 0.),
                        node_coverage=len(reachable) / len(nodes) if nodes else None,
                        times_used_for_direction=False, times_supplied=times is not None,
                        cost_assumption='frozen upstream negative-log plausibility plus declared priors',
                        algorithm='Chu-Liu/Edmonds', roots=roots)
    return result


def pivot_flags(*, four_point, normalized_rf):
    """Report strict threshold crossings; a person makes any pivot decision."""
    return (["four_point_gt_0.20"] if four_point is not None and four_point > .20 else []) + (
        ["normalized_rf_gt_0.20"] if normalized_rf is not None and normalized_rf > .20 else [])
