"""Declared ancestry, conservative split groups and network Brownian covariance."""

from collections import defaultdict
import hashlib
import heapq
import json

import numpy as np
import pandas as pd


EDGE_COLUMNS = ['parent_id', 'child_id', 'parent_key', 'child_key', 'relation',
                'evidence_tier', 'coefficient', 'coefficient_source',
                'branch_length', 'branch_length_source', 'source_hash']
GROUP_COLUMNS = ['model_key', 'duplicate_group', 'root_group', 'split_group', 'group_evidence']


def _text(value):
    return '' if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)) else str(value).strip()


def _matching_edges(edges):
    """Canonical matching identities with deterministic duplicate evidence handling."""
    if edges.empty:
        return pd.DataFrame(columns=list(edges.columns) or ['parent_id', 'child_id'])
    edges = edges.copy()
    for column in ['parent_id', 'child_id']:
        edges[column] = edges[column].map(lambda value: _text(value).casefold())
    rows = []
    for _, group in edges.groupby(['parent_id', 'child_id'], sort=True):
        chosen = group.iloc[0].to_dict()
        for column in ['coefficient', 'branch_length']:
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors='coerce').dropna().unique()
            if len(values) > 1:
                raise ValueError('conflicting repeated edge ' + column)
            chosen[column] = values[0] if len(values) else np.nan
            proof_column = column + '_source'
            if proof_column in group:
                proof = sorted({_text(value) for value in group[proof_column] if _text(value)})
                chosen[proof_column] = ';'.join(proof)
        if 'source_hash' in group:
            chosen['source_provenance'] = json.dumps(sorted(set(group.source_hash)), separators=(',', ':'))
            chosen['source_hash'] = hashlib.sha256(chosen['source_provenance'].encode()).hexdigest()
        # Display IDs are kept separately from the casefolded matching identity.
        for column in ['parent_display', 'child_display']:
            if column in group:
                chosen[column] = sorted(group[column])[0]
        rows.append(chosen)
    return pd.DataFrame(rows)


def _topology(nodes, edges):
    parents, children = defaultdict(set), defaultdict(set)
    universe = set(nodes)
    for row in edges.itertuples():
        parent, child = _text(row.parent_id), _text(row.child_id)
        if not parent or not child:
            raise ValueError('empty ancestry endpoint')
        universe.update([parent, child])
        parents[child].add(parent)
        children[parent].add(child)
    indegree = {node: len(parents[node]) for node in universe}
    ready = [node for node, degree in indegree.items() if not degree]
    heapq.heapify(ready)
    order = []
    while ready:
        node = heapq.heappop(ready)
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if not indegree[child]:
                heapq.heappush(ready, child)
    if len(order) != len(universe):
        raise ValueError('ancestry cycle')
    return order, parents


def _merge_support(order, parents, edges):
    bad = set()
    reasons = {}
    for node in order:
        ps = parents[node]
        if any(parent in bad for parent in ps):
            bad.add(node)
            reasons[node] = 'merge_coefficients_unavailable'
        elif len(ps) > 1:
            rows = edges.loc[edges.child_id.eq(node)]
            weights = pd.to_numeric(rows.coefficient, errors='coerce').to_numpy(float)
            if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1.):
                bad.add(node)
                reasons[node] = 'merge_coefficients_unavailable'
    return reasons


def assemble_edges(atlas: dict, metadata: pd.DataFrame, pins: pd.DataFrame):
    """Keep bare Atlas identities; endpoint pins never verify declared relationships."""
    pin_map = defaultdict(set)
    for row in pins.to_dict('records'):
        repo = _text(row.get('repo_id'))
        revision = _text(row.get('revision'))
        if repo and len(revision) == 40 and all(c in '0123456789abcdefABCDEF' for c in revision):
            pin_map[repo.casefold()].add(repo + '@' + revision.lower())
    canonical_atlas = dict(atlas)
    canonical_atlas['edges'] = sorted(atlas.get('edges', []), key=lambda row: json.dumps(row, sort_keys=True))
    source_hash = hashlib.sha256(json.dumps(canonical_atlas, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    records, exclusions = [], []
    for row in atlas.get('edges', []):
        records.append(dict(row, parent_id=_text(row.get('source', row.get('parent_id'))),
                            child_id=_text(row.get('target', row.get('child_id'))), source_hash=source_hash))
    metadata_hash = hashlib.sha256(metadata.to_json(orient='records').encode()).hexdigest()
    for row in metadata.to_dict('records'):
        parent, child = _text(row.get('source_model')), _text(row.get('model_id'))
        if parent.casefold() in {'', 'unknown', 'none', 'nan', 'null'}:
            exclusions.append({'stage':'ancestry', 'record_id':child, 'reason':'parent_unavailable', 'detail':'metadata has no declared parent'})
            continue
        if not child:
            exclusions.append({'stage':'ancestry', 'record_id':'', 'reason':'missing_endpoint', 'detail':parent})
            continue
        if parent and child:
            records.append({'parent_id': parent, 'child_id': child,
                            'relation': _text(row.get('base_model_relation')) or 'unknown', 'source_hash':metadata_hash})
    output = []
    for row in records:
        parent, child = row['parent_id'], row['child_id']
        if not parent or not child:
            exclusions.append({'stage':'ancestry', 'record_id':child, 'reason':'missing_endpoint', 'detail':parent})
            continue
        keys = []
        for node in [parent, child]:
            values = pin_map[node.casefold()]
            keys.append(next(iter(values)) if len(values) == 1 else '')
        coefficient = row.get('coefficient', np.nan)
        coefficient_source = _text(row.get('coefficient_source'))
        # Numerical weights without provenance are not configured merge truth.
        if not coefficient_source:
            coefficient = np.nan
        output.append(dict(zip(EDGE_COLUMNS, [parent, child, *keys,
            _text(row.get('relation', row.get('base_model_relation'))) or 'unknown',
            'declared', coefficient, coefficient_source,
            row.get('branch_length', 1.), row.get('branch_length_source', 'unit'), row['source_hash']])))
    edges = pd.DataFrame(output, columns=EDGE_COLUMNS)
    edges['parent_display'] = edges.parent_id
    edges['child_display'] = edges.child_id
    edges = _matching_edges(edges).sort_values(['child_id', 'parent_id']).reset_index(drop=True)
    order, parents = _topology([], edges)
    for node in order:
        if len(parents[node]) == 1:
            edges.loc[edges.child_id.eq(node), ['coefficient', 'coefficient_source']] = [1., 'single_parent']
    for node, reason in _merge_support(order, parents, edges).items():
        exclusions.append({'stage':'ancestry', 'record_id':node, 'reason':reason, 'detail':'primary network fit excludes node and affected descendants'})
    edges['parent_id'] = edges.pop('parent_display')
    edges['child_id'] = edges.pop('child_display')
    return edges, pd.DataFrame(exclusions, columns=['stage', 'record_id', 'reason', 'detail'])


def build_groups(models: pd.DataFrame, edges: pd.DataFrame, duplicates: pd.DataFrame):
    """Join every shared ancestry root and duplicate link before splitting."""
    repo_for_key = {}
    for row in models.to_dict('records'):
        key = _text(row.get('model_key'))
        if key:
            repo_for_key[key] = (_text(row.get('repo_id')) or key.rsplit('@', 1)[0]).casefold()
    edges = _matching_edges(edges)
    order, parents = _topology(repo_for_key.values(), edges)
    roots = {}
    for node in order:
        roots[node] = set().union(*(roots[p] for p in parents[node])) if parents[node] else {node}
    leaders = {node: node for node in order}

    def find(node):
        while leaders[node] != node:
            leaders[node] = leaders[leaders[node]]
            node = leaders[node]
        return node

    def union(left, right):
        a, b = sorted([find(left), find(right)])
        leaders[b] = a

    for row in edges.itertuples():
        union(row.parent_id, row.child_id)
    duplicate_for_key = {}
    for row in duplicates.to_dict('records'):
        key, group = _text(row.get('model_key')), _text(row.get('duplicate_group'))
        if key in repo_for_key and group:
            if key in duplicate_for_key and duplicate_for_key[key] != group:
                raise ValueError('conflicting duplicate group')
            duplicate_for_key[key] = group
    grouped = defaultdict(list)
    for key, group in duplicate_for_key.items():
        grouped[group].append(repo_for_key[key])
    for members in grouped.values():
        for node in members[1:]:
            union(members[0], node)

    def label(kind, members):
        text = json.dumps(sorted(members), separators=(',', ':'))
        return kind + ':' + hashlib.sha256(text.encode()).hexdigest()[:20]

    components = defaultdict(set)
    for node in order:
        components[find(node)].add(node)
    result = []
    for key, node in sorted(repo_for_key.items()):
        result.append([key, duplicate_for_key.get(key, label('unique', [key])),
                       label('root', roots[node]), label('split', components[find(node)]),
                       'declared_ancestry' + (';duplicate_evidence' if key in duplicate_for_key else '')])
    return pd.DataFrame(result, columns=GROUP_COLUMNS)


def network_covariance(nodes: list[str], edges: pd.DataFrame):
    """Return covariance in requested order, retaining unobserved ancestors internally."""
    nodes = [node.casefold() for node in nodes]
    if len(set(nodes)) != len(nodes):
        raise ValueError('duplicate covariance nodes')
    edges = _matching_edges(edges)
    order, parents = _topology(nodes, edges)
    if not nodes:
        return np.empty((0, 0))
    # Only the ancestors of requested nodes contribute to their marginal covariance.
    needed = set(nodes)
    for node in reversed(order):
        if node in needed:
            needed.update(parents[node])
    order = [node for node in order if node in needed]
    positions = {node:i for i, node in enumerate(order)}
    a = np.zeros((len(order), len(order)))
    for node in order:
        i = positions[node]
        if not parents[node]:
            a[i, i] = 1.
            continue
        rows = edges.loc[edges.child_id.eq(node)]
        lengths = pd.to_numeric(rows.get('branch_length', pd.Series(1., index=rows.index)), errors='coerce').to_numpy(float)
        if not np.isfinite(lengths).all() or (lengths < 0).any() or not np.allclose(lengths, lengths[0]):
            raise ValueError('invalid or conflicting branch length')
        if len(rows) == 1:
            raw = rows.iloc[0].get('coefficient', 1.)
            if pd.notna(raw) and (not np.isfinite(float(raw)) or not np.isclose(float(raw), 1.)):
                raise ValueError('single-parent coefficient must be one')
            weights = [1.]
        else:
            weights = pd.to_numeric(rows.get('coefficient', pd.Series(np.nan, index=rows.index)), errors='coerce').to_numpy(float)
            if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(sum(weights), 1.):
                raise ValueError('merge coefficient unavailable or invalid')
        for weight, parent in zip(weights, rows.parent_id):
            a[i] += weight * a[positions[parent]]
        a[i, i] = np.sqrt(lengths[0])
    selected = a[[positions[node] for node in nodes]]
    return selected @ selected.T


def select_primary_deployments(scores: pd.DataFrame, config: dict):
    """Select one declared deployment per checkpoint/task; retain alternatives."""
    selected, sensitivity, exclusions = [], [], []
    if not config.get('precision') or not config.get('quantization'):
        raise ValueError('primary deployment precision and quantization must be explicit')
    for _, group in scores.groupby(['model_key', 'benchmark', 'task'], dropna=False, sort=True):
        matches = group.precision.eq(config['precision']) & group.quantization.eq(config['quantization'])
        if config.get('harness'):
            matches &= group.harness.eq(config['harness'])
        else:
            matches &= group.harness.fillna('').ne('')
        candidates = group.loc[matches]
        if len(candidates) == 1:
            selected.append(candidates)
            other = group.loc[~group.index.isin(candidates.index)]
            reason = 'nonprimary_deployment'
        else:
            other = group
            reason = 'primary_deployment_unavailable' if candidates.empty else 'primary_deployment_ambiguous'
        sensitivity.append(other)
        for row in other.itertuples():
            exclusions.append({'stage':'primary_deployment', 'record_id':row.record_id,
                               'reason':reason, 'detail':'retained for sensitivity/descriptive output'})
    primary = pd.concat(selected, ignore_index=True) if selected else scores.iloc[:0].copy()
    alternative = pd.concat(sensitivity, ignore_index=True) if sensitivity else scores.iloc[:0].copy()
    return primary, alternative, pd.DataFrame(exclusions, columns=['stage','record_id','reason','detail'])
