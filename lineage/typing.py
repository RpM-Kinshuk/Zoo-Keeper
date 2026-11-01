"""Matched-parent extraction summaries and explicit shallow development rules.

No ranks or raw-delta statistics are estimated from weight sketches. Structural
metadata must identify its parent; callers supply measured/verified evidence.
"""

import copy
import fnmatch
import json
import math

import numpy as np

from .records import view_diagnostics


def _values(delta, field, n):
    value = np.asarray(delta.get(field, []))
    if value.ndim == 0:
        return [value.item()] * n
    return value.tolist() if len(value) == n else None


def _finite(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(value)


def pair_features(child, parent) -> dict:
    """Extract only aligned, matched-parent summaries; nulls retain reasons."""
    out = dict(child_key=child.key, parent_key=parent.key, rank=None,
               spectral_gap=None, frac_identical=None, changed_fraction=None,
               changed_modules=[], added_modules=sorted(set(child.names) - set(parent.names)),
               removed_modules=sorted(set(parent.names) - set(child.names)),
               rank_status='missing_data', n_views=0, coverage=0., reasons=[],
               adapter_factors=None, storage_dtype=child.meta.get('storage_dtype'),
               delta_flatness=None, leaf_prior=None, representation_supported=True,
               identical=False)
    reasons = out['reasons']
    if child.meta.get('parent') != parent.key:
        out['rank_status'] = 'parent_mismatch'
        reasons.append('parent_mismatch')
        return out
    if child.meta.get('representation', 'matrix') not in ('matrix', 'dense', 'adapter'):
        out['representation_supported'] = False
        reasons.append('unsupported_representation')
        return out
    pairs, diagnostics = view_diagnostics(child, parent)
    reasons.extend(diagnostics)
    out['n_views'] = len(pairs)
    # New modules have no parent delta and are accounted for separately.
    expected = len(set(child.names) & set(parent.names))
    out['coverage'] = len(pairs) / expected if expected else 0.
    if not pairs:
        reasons.append('no_shared_views')
        return out
    n = len(child.names)
    arrays = {field: _values(child.delta, field, n)
              for field in ('rank', 'status', 'confidence', 'frac_identical', 'top_sv')}
    if any(value is None or len(value) == 0 for value in arrays.values()):
        reasons.append('missing_delta_summaries')
        return out
    ranks, gaps, fractions, changed = [], [], [], []
    blocked = False
    for i, _ in pairs:
        name, status = child.names[i], arrays['status'][i]
        frac = arrays['frac_identical'][i]
        if not _finite(frac) or not 0 <= frac <= 1:
            reasons.append(f'{name}:invalid_frac_identical')
            blocked = True
        else:
            fractions.append(float(frac))
        if status == 'identical':
            if arrays['rank'][i] != 0 or frac != 1:
                reasons.append(f'{name}:inconsistent_identical_summary')
                blocked = True
            continue
        if status != 'ok':
            reasons.append(f'{name}:{status}')
            blocked = True
            continue
        changed.append(name)
        rank = arrays['rank'][i]
        if not _finite(rank) or rank <= 0 or int(rank) != rank or arrays['confidence'][i] != 'high':
            reasons.append(f'{name}:missing_or_low_confidence_rank')
            blocked = True
            continue
        rank = int(rank)
        ranks.append(rank)
        sv = np.asarray(arrays['top_sv'][i], dtype=float)
        if sv.ndim != 1 or len(sv) <= rank:
            reasons.append('gap_outside_observed_spectrum')
            continue
        if not np.isfinite(sv).all() or np.any(sv < 0) or np.any(np.diff(sv) > 0) or sv[rank] <= 0:
            reasons.append(f'{name}:invalid_or_unresolved_spectrum')
            continue
        gaps.append(float(sv[rank - 1] / sv[rank]))
    out['changed_modules'] = sorted(changed)
    out['changed_fraction'] = len(changed) / len(pairs)
    out['frac_identical'] = float(np.mean(fractions)) if len(fractions) == len(pairs) else None
    out['identical'] = not changed and not blocked
    out['rank_status'] = 'unavailable' if blocked else 'ok'
    if ranks and not blocked:
        out['rank'] = max(ranks)
        if len(gaps) == len(changed):
            out['spectral_gap'] = min(gaps)
    factors = child.meta.get('adapter_factors', {})
    if (factors.get('parent') == parent.key and factors.get('verified') is True
            and sorted(factors.get('modules', [])) == sorted(changed)
            and len(factors.get('ranks', [])) == len(changed)
            and changed and all(_finite(r) and r > 0 and int(r) == r for r in factors['ranks'])):
        out['adapter_factors'] = copy.deepcopy(factors)
    quant = child.meta.get('quantization_evidence', {})
    if quant.get('parent') == parent.key and quant.get('provenance') == 'development_frozen':
        for field in ('delta_flatness', 'leaf_prior'):
            if _finite(quant.get(field)):
                out[field] = float(quant[field])
    if out['delta_flatness'] is None:
        reasons.append('raw_delta_flatness_unavailable')
    return out


def _validate_rules(rules):
    fields = ('adapter_max_rank', 'min_views', 'min_changed_fraction', 'min_gap', 'min_coverage',
              'adapter_modules', 'head_modules', 'finetune_modules',
              'quantization_max_flatness', 'quantization_min_prior', 'quantization_storage')
    if any(field not in rules for field in fields):
        raise ValueError('rule JSON must explicitly define thresholds, modules and coverage')
    for field in fields:
        if field.endswith('_modules') or field == 'quantization_storage':
            if not isinstance(rules[field], list) or not all(isinstance(v, str) for v in rules[field]):
                raise ValueError(f'invalid rule list: {field}')
        elif not _finite(rules[field]) or rules[field] < 0:
            raise ValueError(f'invalid rule threshold: {field}')
    for field in ('min_coverage', 'min_changed_fraction', 'quantization_min_prior'):
        if rules[field] > 1:
            raise ValueError(f'invalid fraction threshold: {field}')
    for field in ('adapter_max_rank', 'min_views'):
        if rules[field] < 1 or int(rules[field]) != rules[field]:
            raise ValueError(f'invalid integer threshold: {field}')


def _matches(names, patterns):
    return bool(names) and all(any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
                               for name in names)


def type_edge(features: dict, rules: dict) -> dict:
    """Apply an explicit shallow rule tree; every unresolved leaf abstains."""
    _validate_rules(rules)
    f = features
    result = dict(edge_type='unknown', status='abstained', reasons=[], features=copy.deepcopy(f))

    def finish(edge, reason):
        result.update(edge_type=edge, status='typed' if edge != 'unknown' else 'abstained',
                      reasons=[reason])
        return result

    if not f.get('representation_supported') or f.get('rank_status') != 'ok':
        result['reasons'] = list(f.get('reasons', [])) or ['missing_rank_evidence']
        return result
    if f['n_views'] < rules['min_views'] or f['coverage'] < rules['min_coverage']:
        return finish('unknown', 'insufficient_coverage')
    changed, added = f['changed_modules'], f['added_modules']
    if f['removed_modules']:
        return finish('unknown', 'unresolved_removed_modules')
    if f['identical'] and _matches(added, rules['head_modules']):
        return finish('finetune', 'head_only_addition')
    if f['identical']:
        return finish('unknown', 'identical_no_update')
    if added:
        return finish('unknown', 'unresolved_added_modules')
    if f['changed_fraction'] < rules['min_changed_fraction']:
        return finish('unknown', 'insufficient_changed_fraction')
    if (f['storage_dtype'] in rules['quantization_storage']
            and f['delta_flatness'] is not None and 0 <= f['delta_flatness'] <= rules['quantization_max_flatness']
            and f['leaf_prior'] is not None and rules['quantization_min_prior'] <= f['leaf_prior'] <= 1):
        return finish('quantized', 'storage_flatness_and_development_prior')
    if _matches(changed, rules['adapter_modules']):
        factors = f['adapter_factors']
        if factors and max(factors['ranks']) <= rules['adapter_max_rank']:
            return finish('adapter', 'verified_adapter_factors')
        if (f['rank'] is not None and f['rank'] <= rules['adapter_max_rank']
                and f['spectral_gap'] is not None and f['spectral_gap'] >= rules['min_gap']):
            return finish('adapter', 'rank_gap_and_module_pattern')
    if _matches(changed, rules['finetune_modules']):
        return finish('finetune', 'finetune_module_pattern')
    return finish('unknown', 'unresolved_module_or_structural_evidence')


def fit_rules(training_rows, *, split: dict, provenance: dict, candidate_rules: list[dict]) -> dict:
    """Select a shallow rule JSON by empirical development exact-truth error.

    Rows contain features, edge_type, roots (all parents for merges), truth_tier.
    split maps complete root IDs to partitions. Provenance must attest duplicate
    grouping and whole-root splitting. Caller supplies candidate numerical cutoffs
    and module lists; no scientific thresholds are hidden here. Ties use canonical
    JSON order; unknown is a regular class. Returned JSON is frozen with fit roots.
    """
    if not training_rows or not candidate_rules:
        raise ValueError('training rows and candidate rule JSONs are required')
    if any(provenance.get(key) is not True for key in ('complete_root_components', 'duplicates_grouped')):
        raise ValueError('complete root and duplicate provenance is required')
    roots = set()
    labels = {'finetune', 'adapter', 'merge', 'quantized', 'abliteration', 'unknown'}
    for row in training_rows:
        if row.get('truth_tier') != 'lab_exact' or row.get('edge_type') not in labels or not row.get('roots'):
            raise ValueError('development exact-truth rows with all roots are required')
        if any(split.get(root) != 'development' for root in row['roots']):
            raise ValueError('all fitting roots must be development; cross-split merges are quarantined')
        roots.update(row['roots'])
    candidates = []
    for rules in candidate_rules:
        _validate_rules(rules)
        if 'fit_provenance' in rules or 'frozen' in rules:
            raise ValueError('candidate rules must be unfitted JSON')
        canonical = json.dumps(rules, sort_keys=True, allow_nan=False)
        errors = sum(type_edge(row['features'], rules)['edge_type'] != row['edge_type'] for row in training_rows)
        candidates.append((errors, canonical))
    errors, canonical = min(candidates)
    result = json.loads(canonical)
    result.update(frozen=True, fit_provenance=dict(partition='development', roots=sorted(roots),
                  n_rows=len(training_rows), errors=errors, truth_tier='lab_exact',
                  complete_root_components=True, duplicates_grouped=True,
                  objective='misclassification_count', candidate_count=len(candidates)))
    return result
