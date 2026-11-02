"""Offline scoring of nested lineage relations against an explicit universe.

External references need evaluation-only labels. An absent Hub declaration never
supplies a negative. Diagnostic TPR90 thresholds are descriptive; only a supplied
frozen threshold produces operational threshold metrics. Negative infinity is
serialized as the JSON string ``"-inf"``.
"""
import math
from numbers import Real

import numpy as np
import pandas as pd
from zoo.schema import LINEAGE_COLUMNS, model_key

from .registry import validate_tables
from .schema import HUB_PATH, LOCAL_KEY

ESTIMANDS = ('parent', 'ancestor', 'same_root')
TIERS = ('lab_exact', 'verified_hub', 'declared')
_TYPE_MAP = {'lora': 'adapter', 'sft': 'finetune', 'dpo': 'finetune',
             'quantization_sim': 'quantized'}


def _missing(value):
    return value is None or value is pd.NA or (isinstance(value, Real) and pd.isna(value))


def _boolean(value, name, *, nullable=False):
    if nullable and _missing(value):
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} requires boolean values')
    return bool(value)


def _identities(models, manifests):
    by_key, by_repo = {}, {}
    for manifest in manifests:
        if not isinstance(manifest, dict):
            raise ValueError('manifest must be an object')
        try:
            repo, revision = manifest['repo_id'], manifest['revision']
            key = model_key(repo, revision)
        except (KeyError, TypeError) as exc:
            raise ValueError('manifest requires repo_id and pinned revision') from exc
        if not (HUB_PATH.fullmatch(key) or LOCAL_KEY.fullmatch(key)):
            raise ValueError('manifest identity must be a portable pinned repository')
        if manifest.get('model_key', key) != key:
            raise ValueError('manifest model_key conflicts with repo_id/revision')
        if key in by_key:
            raise ValueError('duplicate manifest mapping: ' + key)
        by_key[key] = manifest
        by_repo.setdefault(repo, []).append(key)
    result = {}
    for row in models.itertuples(index=False):
        if HUB_PATH.fullmatch(row.path):
            matches = [row.path] if row.path in by_key else []
        else:
            matches = [key for key in by_repo.get('lab/' + row.model_id, [])
                       if LOCAL_KEY.fullmatch(key)]
        if len(matches) != 1:
            raise ValueError(f'{row.model_id}: missing or duplicate manifest mapping')
        result[row.model_id] = matches[0]
    if len(set(result.values())) != len(result):
        raise ValueError('model IDs have duplicate manifest mappings')
    return result


def _construction(models, edges):
    parents = {name: set() for name in models.model_id}
    direct, edge_types = set(), {}
    for row in edges.itertuples(index=False):
        parents[row.child].add(row.parent)
        if row.role == 'parent':
            direct.add((row.child, row.parent))
            edge_types[row.child, row.parent] = row.edge_type
    ancestors, roots = {}, {}

    def visit(name):
        if name in ancestors:
            return
        for parent in parents[name]:
            visit(parent)
        ancestors[name] = set(parents[name])
        roots[name] = set() if parents[name] else {name}
        for parent in parents[name]:
            ancestors[name].update(ancestors[parent])
            roots[name].update(roots[parent])

    for name in sorted(parents):
        visit(name)
    public = set(models.loc[models.kind.isin(['base', 'derived']), 'model_id'])
    complete = {name for name in public if ancestors[name] <= public}
    return direct, ancestors, roots, complete, edge_types


def _claims(predictions):
    if list(predictions.columns) != list(LINEAGE_COLUMNS):
        raise ValueError('predictions require the exact zoo.schema.LINEAGE_COLUMNS')
    result, originals = {}, {}
    for row in predictions.to_dict('records'):
        claim = row['child_key'], row['parent_key'], row['estimand']
        if row['estimand'] not in ESTIMANDS:
            raise ValueError('unknown prediction estimand')
        value = row['score']
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or math.isnan(value) or value == math.inf):
            raise ValueError('score must be numeric, finite or negative infinity')
        normalized = {k: None if _missing(v) else v for k,v in row.items()}
        if claim in originals and originals[claim] != normalized:
            raise ValueError('conflicting duplicate prediction claims')
        originals[claim] = normalized
        result[claim] = row
    return result


def _at_threshold(rows, threshold, n_positive, n_queries):
    selected = [r for r in rows if r['truth'] is not None and r['score'] >= threshold]
    tp = sum(r['truth'] for r in selected)
    fp = len(selected) - tp
    return {'threshold': '-inf' if threshold == -math.inf else threshold,
            'tpr': tp / n_positive if n_positive else None,
            'false_matches': fp,
            'false_matches_per_query': fp / n_queries if n_queries else None}


def _metrics(rows, estimand, threshold):
    labeled = [r for r in rows if r['truth'] is not None]
    positives = [r['score'] for r in labeled if r['truth']]
    n = len(rows)
    p, neg = len(positives), len(labeled) - len(positives)
    queries = len({r['query'] for r in rows})
    scored = sum(r['score'] > -math.inf for r in rows)
    status = 'undefined_no_positives' if not p else 'undefined_no_negatives' if not neg else 'ok'
    auroc = None
    if p and neg:
        # pandas' average ranks implement the Mann-Whitney midrank AUROC,
        # including abstention ties at negative infinity.
        ranks = pd.Series([r['score'] for r in labeled]).rank(method='average')
        rank_sum = sum(float(rank) for rank,r in zip(ranks, labeled) if r['truth'])
        auroc = (rank_sum - p * (p + 1) / 2) / (p * neg)
    tpr90 = None
    if p:
        cutoff = sorted(positives, reverse=True)[math.ceil(.9 * p) - 1]
        tpr90 = _at_threshold(rows, cutoff, p, queries)
    report = dict(n_pairs=n, n_labeled=len(labeled), n_unknown=n-len(labeled),
                  n_positive=p, n_negative=neg, n_queries=queries,
                  n_eligible=sum(r['eligible'] for r in rows), n_scored=scored,
                  n_abstentions=n-scored,
                  eligible_fraction=sum(r['eligible'] for r in rows)/n if n else None,
                  scored_fraction=scored/n if n else None,
                  abstention_fraction=(n-scored)/n if n else None,
                  positive_abstentions=sum(s == -math.inf for s in positives),
                  auroc=auroc, status=status, auroc_status=status,
                  tpr_status='ok' if p else 'undefined_no_positives',
                  threshold_at_tpr90=tpr90['threshold'] if tpr90 else None,
                  attained_tpr90=tpr90['tpr'] if tpr90 else None,
                  false_matches_at_tpr90=tpr90['false_matches'] if tpr90 else None,
                  false_matches_per_query_at_tpr90=tpr90['false_matches_per_query'] if tpr90 else None)
    if threshold is not None:
        report['frozen_threshold'] = _at_threshold(rows, threshold, p, queries)
    if estimand != 'same_root':
        lookup = {(r['query'],r['candidate']): r for r in rows}
        pairs = [(r, lookup.get((r['candidate'],r['query']))) for r in labeled if r['truth']]
        paired = [(a,b) for a,b in pairs if b is not None and b['truth'] is False]
        available = [(a,b) for a,b in paired if a['score'] > -math.inf and b['score'] > -math.inf]
        report.update(n_direction_pairs=len(paired), n_direction_scored=len(available),
                      direction_availability=len(available)/len(paired) if paired else None,
                      direction_accuracy=sum(a['score'] > b['score'] for a,b in available)/len(available)
                      if available else None)
    return report


def score_lab(models: pd.DataFrame, edges: pd.DataFrame, predictions: pd.DataFrame,
              manifests: list[dict], universe: pd.DataFrame, *, threshold: float | None = None) -> dict:
    """Score only the declared pairs; missing/ineligible claims abstain.

    Coverage and query denominators include unknown-label and ineligible rows.
    Unknown labels do not enter class counts or AUROC. Self-pairs are excluded.
    Identical duplicate prediction claims collapse; universe duplicates fail.
    """
    if threshold is not None and (isinstance(threshold, (bool, np.bool_))
                                 or not isinstance(threshold, Real) or not math.isfinite(threshold)):
        raise ValueError('frozen threshold must be finite')
    validate_tables(models, edges)
    identities = _identities(models, manifests)
    ids = {key: name for name,key in identities.items()}
    claims = _claims(predictions)
    required = {'query_key', 'candidate_key', 'eligible'}
    if not required <= set(universe.columns):
        raise ValueError('universe requires query_key,candidate_key,eligible')
    if universe.duplicated(['query_key', 'candidate_key']).any():
        raise ValueError('duplicate universe pair')
    direct, ancestors, roots, complete, types = _construction(models, edges)
    grouped = {tier: {estimand: [] for estimand in ESTIMANDS} for tier in TIERS}
    self_pairs, type_examples, type_correct, type_eligible = 0, 0, 0, 0
    for row in universe.to_dict('records'):
        q, c = row['query_key'], row['candidate_key']
        if q not in ids or c not in ids:
            raise ValueError('universe keys require exact manifest mappings')
        eligible = _boolean(row['eligible'], 'eligible')
        explicit = {e: _boolean(row.get('truth_' + e), 'truth_' + e, nullable=True) for e in ESTIMANDS}
        if ((explicit['parent'] is True and explicit['ancestor'] is False)
                or (explicit['ancestor'] is True and explicit['same_root'] is False)
                or (explicit['parent'] is True and explicit['same_root'] is False)):
            raise ValueError('explicit labels conflict with nested relations')
        tier = row.get('truth_tier')
        if _missing(tier):
            if any(v is not None for v in explicit.values()):
                raise ValueError('explicit labels require truth_tier')
            tier = 'lab_exact'
        if tier not in TIERS:
            raise ValueError('unknown truth_tier')
        if q == c:
            self_pairs += 1
            continue
        child, parent = ids[q], ids[c]
        known = child in complete and parent in complete
        truth = {'parent': (child,parent) in direct,
                 'ancestor': parent in ancestors[child],
                 'same_root': bool(roots[child] & roots[parent])} if known else {}
        for e in ESTIMANDS:
            if known and explicit[e] is not None and explicit[e] != truth[e]:
                raise ValueError('explicit label conflicts with complete construction truth')
            if tier == 'lab_exact' and not known and explicit[e] is not None:
                raise ValueError('external labels require verified_hub or declared truth_tier')
            label = truth.get(e) if tier == 'lab_exact' and known else explicit[e]
            claim = claims.get((q,c,e))
            value = float(claim['score']) if claim is not None and eligible else -math.inf
            grouped[tier][e].append(dict(query=q, candidate=c, eligible=eligible,
                                        truth=label, score=value))
        edge_type = types.get((child,parent))
        if tier == 'lab_exact' and known and edge_type is not None and edge_type != 'laundering':
            type_eligible += 1
            claim = claims.get((q,c,'parent'))
            if eligible and claim is not None and claim['score'] > -math.inf:
                type_examples += 1
                type_correct += claim['edge_type'] == _TYPE_MAP.get(edge_type, edge_type)
    return {'schema_version': '1', 'diagnostic_tpr_target': .9,
            'threshold_policy': 'diagnostic_tpr90; operational metrics require frozen_threshold',
            'self_pairs_excluded': self_pairs,
            'tiers': {tier: {e: _metrics(grouped[tier][e], e, threshold) for e in ESTIMANDS}
                      for tier in TIERS},
            'type_accuracy': {'accuracy': type_correct/type_examples if type_examples else None,
                              'n_examples': type_examples, 'n_eligible_examples': type_eligible,
                              'coverage': type_examples/type_eligible if type_eligible else None,
                              'excluded_edge_types': ['laundering']}}
