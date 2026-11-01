"""Unconstrained aligned-sketch merge fits and contiguous layer uncertainty."""

import re

import numpy as np

from .records import FingerprintRecord, sketch_block, view_diagnostics


def _layer(name):
    match = re.search(r'(?:^|\.)(?:layers|layer|blocks|block|h)\.(\d+)(?:\.|$)', name)
    if match:
        return name[:match.start(1)], int(match.group(1))
    return None


def _column_groups(x):
    """Connected components of absolute normalized column correlations."""
    normalized = x / np.linalg.norm(x, axis=0)
    correlations = np.abs(normalized.T @ normalized)
    remaining, groups = set(range(x.shape[1])), []
    while remaining:
        component, pending = set(), [min(remaining)]
        while pending:
            i = pending.pop()
            if i in component:
                continue
            component.add(i)
            pending.extend(j for j in remaining if correlations[i, j] >= .999 and j not in component)
        remaining -= component
        groups.append(sorted(component))
    return groups


def _solve(x, y):
    coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float('inf')
    identified = rank == x.shape[1] and condition <= 1e6
    return coefficients, int(rank), condition, identified


def _bootstrap_intervals(pieces, columns, *, bootstrap, block_layers, rng):
    """Resample moving blocks of complete layers; never split a view or layer.

    QR reduction of each layer's augmented [X, y] preserves its least-squares
    objective, avoiding repeated copies of the large sketch arrays.
    """
    layers = sorted(pieces)
    blocks = [layers[i:i + block_layers] for i in range(len(layers) - block_layers + 1)
              if all(layers[j][0] == layers[i][0] and layers[j][1] == layers[i][1] + j - i
                     for j in range(i, i + block_layers))]
    if not blocks:
        return None, 'insufficient_contiguous_layers'
    supported_layers = {layer for block in blocks for layer in block}
    if supported_layers != set(layers):
        return None, 'insufficient_contiguous_layer_blocks'
    reduced = {}
    for layer in layers:
        x = np.concatenate([piece[0][:, columns] for piece in pieces[layer]])
        y = np.concatenate([piece[1] for piece in pieces[layer]])
        reduced[layer] = np.linalg.qr(np.column_stack((x, y)), mode='r')
    samples = []
    for _ in range(bootstrap):
        sampled = []
        while len(sampled) < len(layers):
            sampled.extend(blocks[int(rng.integers(len(blocks)))])
        data = np.concatenate([reduced[layer] for layer in sampled[:len(layers)]])
        coefficients, _, _, identified = _solve(data[:, :-1], data[:, -1])
        if not identified:
            return None, 'nonidentifiable_resample'
        samples.append(coefficients)
    return np.quantile(samples, [.025, .975], axis=0).T.tolist(), 'ok'


def _fit_group(views, keys, *, bootstrap, block_layers, rng):
    empty = dict(coefficients=None, intervals=None, coefficient_kind='linear',
                 condition_number=None, rank=0, collinear_groups=[], relative_residual=None,
                 n_views=len(views), n_layers=0, status='missing_data',
                 interval_status='missing_data', group_attributions=[])
    if not views:
        return empty
    x = np.concatenate([view[1] for view in views])
    y = np.concatenate([view[2] for view in views])
    coefficients, rank, condition, identified = _solve(x, y)
    components = _column_groups(x)
    collinear = [[keys[i] for i in component] for component in components if len(component) > 1]
    identified = identified and not collinear
    pieces = {}
    for name, vx, vy in views:
        layer = _layer(name)
        if layer is not None:
            pieces.setdefault(layer, []).append((vx, vy))
    complete_layers = all(_layer(name) is not None for name, _, _ in views)
    result = dict(empty, condition_number=condition, rank=rank, collinear_groups=collinear,
                  relative_residual=float(np.linalg.norm(y - x @ coefficients) / np.linalg.norm(y)),
                  n_layers=len(pieces), status='identified' if identified else 'nonidentifiable')
    if identified:
        result['coefficients'] = coefficients.tolist()
    # Attribution for correlated columns is the combined effect expressed in
    # the first member's sketch units, not an allocation to individual parents.
    columns = [component[0] for component in components]
    grouped, _, _, grouped_identified = _solve(x[:, columns], y)
    if not complete_layers or len(pieces) < max(4, block_layers):
        interval_status, intervals = 'insufficient_layers', None
    elif bootstrap == 0:
        interval_status, intervals = 'disabled', None
    elif not grouped_identified:
        interval_status, intervals = 'nonidentifiable', None
    else:
        intervals, interval_status = _bootstrap_intervals(
            pieces, columns, bootstrap=bootstrap, block_layers=block_layers, rng=rng)
    result['interval_status'] = interval_status
    if identified:
        result['intervals'] = intervals
    if collinear and grouped_identified:
        result['group_attributions'] = [
            dict(members=[keys[i] for i in component], reference_key=keys[component[0]],
                 coefficient=float(grouped[j]), interval=None if intervals is None else intervals[j])
            for j, component in enumerate(components)]
    return result


def fit_merge(child: FingerprintRecord, parents: list[FingerprintRecord], base: FingerprintRecord,
              *, groups: dict[str, list[str]] | None = None, bootstrap: int = 200,
              block_layers: int = 2, seed: int = 20261028) -> dict:
    """Fit signed raw sketches without bounds, an intercept or a sum constraint.

    Parent order determines column order; a base already listed as a parent is
    represented once with both roles. Missing views are excluded with reasons.
    Intervals are percentile intervals from complete contiguous layer blocks.
    Nonlinear recipe coefficients are effective projections, never recovered
    configured weights. Collinear groups are fixed before all resampling.
    """
    if not isinstance(bootstrap, (int, np.integer)) or bootstrap < 0:
        raise ValueError('bootstrap must be a nonnegative integer')
    if not isinstance(block_layers, (int, np.integer)) or block_layers < 1:
        raise ValueError('block_layers must be a positive integer')
    records, regressors, positions = [], [], {}
    for record, role in [(parent, 'parent') for parent in parents] + [(base, 'base')]:
        if record.key not in positions:
            positions[record.key] = len(records)
            records.append(record)
            regressors.append(dict(key=record.key, roles=[]))
        roles = regressors[positions[record.key]]['roles']
        if role not in roles:
            roles.append(role)
    aligned, reasons = [], []
    for record in records:
        pairs, diagnostics = view_diagnostics(child, record, k=64)
        aligned.append(dict(pairs))
        reasons.extend(f'{record.key}:{reason}' for reason in diagnostics)
    common = sorted(set.intersection(*(set(mapping) for mapping in aligned)), key=lambda i: child.names[i])
    views = []
    for index in common:
        target = sketch_block(child, index, 64).ravel()
        columns = [sketch_block(record, mapping[index], 64).ravel()
                   for record, mapping in zip(records, aligned)]
        views.append((child.names[index], np.column_stack(columns), target))
    if groups is None:
        requested = {name: [name] for name in sorted(child.names)}
    else:
        requested = {name: list(names) for name, names in sorted(groups.items())}
        unknown = set(name for names in requested.values() for name in names) - set(child.names)
        if unknown:
            raise ValueError(f'unknown child view names: {sorted(unknown)}')
    requested['all'] = list(child.names)
    keys = [record.key for record in records]
    recipe = child.meta.get('merge_recipe', {})
    method = str(recipe.get('method', '')).lower()
    kind = 'effective' if any(name in method for name in ('slerp', 'ties', 'dare')) else 'linear'
    rng = np.random.default_rng(seed)
    fitted = {}
    for name, names in requested.items():
        selected = set(names)
        result = _fit_group([view for view in views if view[0] in selected], keys,
                            bootstrap=bootstrap, block_layers=block_layers, rng=rng)
        result['coefficient_kind'] = kind
        if 'coefficients' in recipe:
            configured = recipe['coefficients']
            effective = result['coefficients']
            comparable = isinstance(configured, (list, tuple)) and len(configured) == len(keys)
            result['configured_comparison'] = dict(
                configured=configured, regressor_keys=keys,
                difference=(np.asarray(effective) - np.asarray(configured)).tolist()
                if comparable and effective is not None else None,
                status='comparable' if comparable and effective is not None else 'unavailable')
        fitted[name] = result
    output = dict(fitted['all'])
    output.update(groups=fitted, regressors=regressors,
                  coverage=dict(n_requested=len(child.names), n_shared=len(common),
                                fraction=len(common) / len(child.names) if child.names else 0.,
                                reasons=sorted(set(reasons))))
    return output
