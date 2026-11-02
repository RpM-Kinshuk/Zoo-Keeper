"""Offline evaluation with nullable truth and immutable, pre-evaluation freezes."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from .baseline import baseline_reports

ESTIMANDS = ('parent', 'ancestor', 'same_root')
TRUTH_COLUMNS = {'query_key', 'candidate_key', 'estimand', 'label', 'truth_tier', 'root_group', 'split'}
KEYS = ['query_key', 'candidate_key']
SCORER_MODULES = ('lineage.evaluate', 'lineage.baseline', 'lineage.cli',
                  'lab.score', 'lab.registry', 'lab.schema', 'lab.artifacts',
                  'zoo.schema', 'lineage', 'lab', 'zoo')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_hash(path):
    path = Path(path)
    if path.is_dir():
        files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob('*')) if p.is_file()}
        return hashlib.sha256(canonical(files).encode()).hexdigest()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def scorer_identity():
    """Bind scoring sources by portable module name and numerical runtime."""
    import pyarrow
    components = {}
    for name in SCORER_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin or not Path(spec.origin).is_file():
            raise ValueError('scorer source unavailable: ' + name)
        components[name] = file_hash(spec.origin)
    runtime = dict(python=sys.version, implementation=sys.implementation.name,
                   machine=platform.machine(), byteorder=sys.byteorder,
                   numpy=np.__version__, pandas=pd.__version__, pyarrow=pyarrow.__version__)
    return components, runtime


def freeze(config, inputs, out):
    """Create a new freeze directory; never replace even an identical freeze.

    inputs maps descriptive names to existing local files/directories. Method
    identity is a supplied inference method hash when present; otherwise it is
    explicitly the evaluation config hash, never claimed as inference provenance.
    """
    config = json.loads(canonical(config))
    out = Path(out)
    hashes = {name: {'path': os.path.relpath(Path(path).resolve(), out.resolve()),
                     'path_base': 'freeze_directory', 'sha256': file_hash(path)}
              for name,path in sorted(inputs.items())}
    config_hash = hashlib.sha256(canonical(config).encode()).hexdigest()
    components, runtime = scorer_identity()
    scorer_hash = hashlib.sha256(canonical(dict(components=components,runtime=runtime)).encode()).hexdigest()
    protocol = dict(schema_version='2', config=config, config_hash=config_hash,
                    method_hash=config.get('method_hash') or config_hash,
                    method_hash_scope='supplied_inference_method' if config.get('method_hash') else 'evaluation_config_only',
                    scorer_hash=scorer_hash, scorer_components=components,
                    runtime_identity=runtime, frozen_before_evaluation=True,
                    threshold_policy='fixed deployment threshold; evaluation TPR90 is descriptive only',
                    prospective_cutoff='2026-12-04', prospective_status='not_evaluated',
                    pool_policy='fixed nested seeded candidate pools retaining all labeled positives',
                    extrapolation_status='not_run', evidence_ladder=['metadata','T0','T1','T2','black-box'])
    out.mkdir(parents=True, exist_ok=False)
    for name,value in [('protocol.json',protocol), ('input_hashes.json',hashes)]:
        with (out / name).open('x') as stream:
            stream.write(canonical(value) + '\n')
    return protocol


def verify_freeze(config, inputs, out):
    out = Path(out)
    protocol = json.loads((out / 'protocol.json').read_text())
    hashes = json.loads((out / 'input_hashes.json').read_text())
    if canonical(config) != canonical(protocol['config']):
        raise ValueError('evaluation config differs from frozen config')
    if 'scorer_components' not in protocol or 'runtime_identity' not in protocol:
        raise ValueError('freeze lacks scorer component/runtime binding; create a new versioned freeze')
    components, runtime = scorer_identity()
    if components != protocol['scorer_components']:
        raise ValueError('evaluation implementation changed after freeze')
    if runtime != protocol['runtime_identity']:
        raise ValueError('evaluation runtime changed after freeze')
    identity_hash = hashlib.sha256(canonical(dict(components=components,runtime=runtime)).encode()).hexdigest()
    if identity_hash != protocol['scorer_hash']:
        raise ValueError('evaluation implementation changed after freeze: inconsistent scorer hash')
    for name,path in inputs.items():
        if name not in hashes or hashes[name]['sha256'] != file_hash(path):
            raise ValueError('input not frozen or changed after freeze: ' + name)
    for name,row in hashes.items():
        path = out / row['path'] if row.get('path_base') == 'freeze_directory' else Path(row['path'])
        if row['sha256'] != file_hash(path):
            raise ValueError('frozen input changed: ' + name)
    return protocol


def _groups(value):
    if value is None or value is pd.NA:
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return list(value)
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, str) and value.startswith('['):
        return json.loads(value)
    return [value]


def validate_splits(truth):
    namespaces = {'root': {}, 'duplicate': {}}
    for field in ('root_group', 'duplicate_group', 'merge_roots'):
        if field not in truth:
            continue
        seen = namespaces['duplicate' if field == 'duplicate_group' else 'root']
        for row in truth.to_dict('records'):
            for group in _groups(row[field]):
                if group in seen and seen[group] != row['split']:
                    raise ValueError(field + ' crosses evaluation splits: ' + str(group))
                seen[group] = row['split']


def _threshold(rows, threshold, queries):
    labeled = rows[rows.label.notna()]
    selected = labeled[labeled.score >= threshold]
    p = int(labeled.label.eq(1).sum())
    fp = int(selected.label.eq(0).sum())
    return dict(threshold='-inf' if threshold == -math.inf else threshold,
                n_queries=len(queries), tpr=int(selected.label.eq(1).sum()) / p if p else None,
                false_matches=fp, false_matches_per_query=fp / len(queries) if queries else None)


def _metrics(rows, threshold, queries, estimand, costs=None):
    labeled = rows[rows.label.notna()]
    p, n = int(labeled.label.eq(1).sum()), int(labeled.label.eq(0).sum())
    auroc = None
    status = 'undefined_no_positives' if not p else 'undefined_no_negatives' if not n else 'ok'
    if p and n:
        ranks = labeled.score.rank(method='average')
        auroc = float((ranks[labeled.label.eq(1)].sum() - p*(p+1)/2)/(p*n))
    available = rows.score.gt(-math.inf)
    answered_queries = set(rows.loc[available, 'query_key'])
    eligible = rows.eligible.eq(True)
    eligible_queries = set(rows.loc[eligible, 'query_key'])
    result = dict(n_pairs=len(rows), positive_pairs=p, negative_pairs=n,
                  unknown_truth_pairs=int(rows.label.isna().sum()), n_queries=len(queries),
                  n_eligible=int(eligible.sum()), n_eligible_queries=len(eligible_queries),
                  n_scored=int(available.sum()), n_abstentions=int((~available).sum()),
                  candidate_coverage=float(available.sum()/len(rows)) if len(rows) else None,
                  eligible_candidate_coverage=float(available.sum()/eligible.sum()) if eligible.any() else None,
                  query_coverage=len(answered_queries)/len(queries) if queries else None,
                  abstention_fraction=float((~available).sum()/len(rows)) if len(rows) else None,
                  auroc=auroc, auroc_status=status,
                  at_frozen_threshold=_threshold(rows, threshold, queries), diagnostic_tpr90=None)
    if p:
        cutoff = float(labeled.loc[labeled.label.eq(1), 'score'].sort_values(ascending=False).iloc[math.ceil(.9*p)-1])
        ranked = _threshold(rows, cutoff, queries)
        scored_positives = int((labeled.label.eq(1) & labeled.score.gt(-math.inf)).sum())
        if cutoff == -math.inf:
            result['diagnostic_tpr90'] = dict(status='unattainable', threshold=None,
                n_queries=len(queries), tpr=None, false_matches=None, false_matches_per_query=None,
                n_scored_positives=scored_positives, maximum_finite_tpr=scored_positives/p,
                reason='finite-scored positives cannot attain TPR 0.9',
                ranking_extension=dict(ranked, status='all_abstention_tie_included'))
        else:
            result['diagnostic_tpr90'] = dict(ranked, status='ok', n_scored_positives=scored_positives,
                                            maximum_finite_tpr=scored_positives/p)
    result.update(direction_accuracy=None, direction_resolved=0, direction_eligible=0,
                  direction_status='symmetric_estimand' if estimand == 'same_root' else 'no_resolved_pairs')
    if estimand != 'same_root':
        lookup = {(r.query_key,r.candidate_key):r for r in rows.itertuples()}
        for r in labeled[labeled.label.eq(1)].itertuples():
            reverse = lookup.get((r.candidate_key,r.query_key))
            if reverse is not None and pd.notna(reverse.label) and reverse.label == 0:
                result['direction_eligible'] += 1
                if r.score > -math.inf and reverse.score > -math.inf:
                    result['direction_resolved'] += 1
                    result.setdefault('_direction_correct', 0)
                    result['_direction_correct'] += int(r.score > reverse.score)
        if result['direction_resolved']:
            result['direction_accuracy'] = result.pop('_direction_correct') / result['direction_resolved']
            result['direction_status'] = 'ok'
    if costs is not None:
        result['costs'] = _cost_report(costs, queries, len(answered_queries))
    return result


def _cost_report(costs, queries, n_answers):
    if 'query_key' not in costs:
        raise ValueError('costs require query_key')
    if costs.query_key.duplicated().any():
        raise ValueError('duplicate cost query')
    costs = costs[costs.query_key.isin(queries)].copy()
    missing = sorted(set(queries) - set(costs.query_key))
    ok = not missing and len(costs) == len(queries)
    if 'status' not in costs or not costs.status.eq('ok').all():
        ok = False
    totals = {}
    for column in ('bytes_read','seconds','query_bytes','reference_bytes','verification_bytes'):
        values = pd.to_numeric(costs[column], errors='coerce') if column in costs else pd.Series(dtype=float)
        known = len(values) == len(queries) and values.notna().all() and np.isfinite(values).all() and values.ge(0).all()
        totals[column] = float(values.sum()) if known else None
    amortized = None
    if totals['reference_bytes'] is not None and 'reference_amortization_queries' in costs:
        denominator = pd.to_numeric(costs.reference_amortization_queries, errors='coerce')
        if denominator.notna().all() and np.isfinite(denominator).all() and denominator.gt(0).all():
            amortized = float((costs.reference_bytes/denominator).sum())
    complete = ok and all(v is not None for v in totals.values())
    return dict(status='ok' if complete else 'incomplete', missing_queries=missing,
                n_queries=len(queries), n_answers=n_answers, totals=totals,
                reference_bytes_unamortized=totals['reference_bytes'],
                reference_bytes_amortized=amortized,
                bytes_per_answer=totals['bytes_read']/n_answers if ok and totals['bytes_read'] is not None and n_answers else None,
                bytes_per_answer_amortized=(totals['query_bytes']+totals['verification_bytes']+amortized)/n_answers
                if ok and amortized is not None and totals['query_bytes'] is not None and totals['verification_bytes'] is not None and n_answers else None)


def evaluate(predictions, truth, universe, costs, *, config):
    """Return JSON metrics plus curves/coverage DataFrames for artifact writing.

    Truth is supplied explicitly. Every universe pair is retained even when no
    prediction or label exists; ineligible/missing predictions rank at -inf.
    """
    if not TRUTH_COLUMNS <= set(truth):
        raise ValueError('truth requires ' + ','.join(sorted(TRUTH_COLUMNS)))
    if not set(KEYS + ['eligible']) <= set(universe):
        raise ValueError('universe requires query_key,candidate_key,eligible')
    if universe.duplicated(KEYS).any() or truth.duplicated(KEYS + ['estimand']).any():
        raise ValueError('duplicate universe/truth pair')
    if not universe.eligible.map(lambda x: isinstance(x,(bool,np.bool_))).all():
        raise ValueError('eligible requires strict boolean values')
    if truth.truth_tier.isna().any() or truth.split.isna().any():
        raise ValueError('truth_tier and split must be explicit')
    if not truth.label.dropna().isin([0,1]).all() or not truth.estimand.isin(ESTIMANDS).all():
        raise ValueError('invalid truth label/estimand')
    validate_splits(truth)
    threshold = config.get('deployment_threshold')
    if isinstance(threshold,bool) or threshold is None or not np.isfinite(threshold):
        raise ValueError('finite deployment_threshold required')
    predictions = predictions.rename(columns={'child_key':'query_key','parent_key':'candidate_key'}).copy()
    if not set(KEYS + ['estimand','score']) <= set(predictions):
        raise ValueError('predictions require child_key,parent_key,estimand,score')
    if predictions.duplicated(KEYS+['estimand']).any():
        raise ValueError('duplicate prediction claim')
    scores = pd.to_numeric(predictions.score, errors='coerce')
    if scores.isna().any() or scores.eq(math.inf).any() or not predictions.estimand.isin(ESTIMANDS).all():
        raise ValueError('invalid prediction score/estimand')
    predictions['score'] = scores
    universe = universe[universe.query_key.ne(universe.candidate_key)].copy()
    pairs = set(map(tuple, universe[KEYS].to_numpy()))
    if any(tuple(row) not in pairs for row in truth[KEYS].to_numpy()):
        raise ValueError('truth pair outside evaluation universe')
    queries = sorted(set(universe.query_key))
    frames, by_estimand, by_tier = {}, {}, {}
    for estimand in ESTIMANDS:
        labels = truth[truth.estimand.eq(estimand)][KEYS+['label','truth_tier']]
        rows = universe.drop(columns=['truth_tier'], errors='ignore').merge(labels,on=KEYS,how='left')
        claims = predictions[predictions.estimand.eq(estimand)][KEYS+['score']]
        rows = rows.merge(claims,on=KEYS,how='left')
        rows['score'] = rows.score.where(rows.eligible, -math.inf).fillna(-math.inf)
        rows['truth_tier'] = rows.truth_tier.fillna('unknown')
        rows['estimand'] = estimand
        frames[estimand] = rows
        by_estimand[estimand] = _metrics(rows, threshold, queries, estimand, costs)
        for tier,group in rows.groupby('truth_tier'):
            by_tier.setdefault(tier,{})[estimand] = _metrics(group,threshold,queries,estimand,costs)
    # Sampling uses only frozen universe/labels, never method predictions.
    all_rows = pd.concat(list(frames.values()),ignore_index=True)
    candidates = sorted(set(universe.candidate_key))
    positive_keys = set(truth.loc[truth.label.eq(1),'candidate_key'])
    rng = np.random.default_rng(config.get('seed',0))
    remaining = list(rng.permutation(sorted(set(candidates)-positive_keys)))
    ordered = sorted(positive_keys) + remaining
    available = len(candidates)
    sizes = config.get('pool_sizes', [2,4,8] if available < 100 else [100,500,1000,5000,10000,20000])
    sizes = sorted({int(s) for s in sizes if len(positive_keys) <= int(s) <= available} | {available})
    curves = []
    for size in sizes:
        chosen = set(ordered[:size])
        for estimand,frame in frames.items():
            subset = frame[frame.candidate_key.isin(chosen)]
            for tier in ['all'] + sorted(set(frame.truth_tier)):
                group = subset if tier == 'all' else subset[subset.truth_tier.eq(tier)]
                m = _metrics(group,threshold,queries,estimand,costs)
                d = m['diagnostic_tpr90'] or {}
                extension = d.get('ranking_extension', {})
                curves.append(dict(pool_size=size, available_candidates=available, estimand=estimand,truth_tier=tier,
                    scale='toy' if available < 100 else 'measured', n_queries=len(queries),n_pairs=len(group),
                    n_abstentions=m['n_abstentions'],auroc=m['auroc'],auroc_status=m['auroc_status'],
                    false_matches_per_query_at_tpr90=d.get('false_matches_per_query'),
                    diagnostic_tpr90_status=d.get('status', 'undefined_no_positives'),
                    threshold_at_tpr90=float(d['threshold']) if d.get('threshold') is not None else None,
                    ranking_extension_threshold_at_tpr90=float(extension['threshold']) if extension else None,
                    ranking_extension_false_matches_per_query_at_tpr90=extension.get('false_matches_per_query'),
                    ranking_extension_false_matches_at_tpr90=extension.get('false_matches'),
                    ranking_extension_tpr=extension.get('tpr'),
                    candidate_coverage=m['candidate_coverage'],
                    query_coverage=m['query_coverage'], abstention_fraction=m['abstention_fraction'],
                    direction_accuracy=m['direction_accuracy'],direction_resolved=m['direction_resolved'],
                    direction_eligible=m['direction_eligible'],
                    deployed_tpr=m['at_frozen_threshold']['tpr'],
                    deployed_false_matches_per_query=m['at_frozen_threshold']['false_matches_per_query'],
                    bytes_per_answer=m['costs']['bytes_per_answer'],
                    bytes_per_answer_amortized=m['costs']['bytes_per_answer_amortized'],
                    cost_status=m['costs']['status']))
    primary = by_estimand['parent']
    n_answers = len(set(frames['parent'].loc[frames['parent'].score.gt(-math.inf),'query_key']))
    baseline = baseline_reports(config)
    return dict(schema_version='1', by_estimand=by_estimand, by_truth_tier=by_tier,
                at_frozen_threshold=primary['at_frozen_threshold'],
                unknown_truth_pairs=primary['unknown_truth_pairs'],negative_pairs=primary['negative_pairs'],
                costs=_cost_report(costs,queries,n_answers),baselines=baseline,
                gate={'status':'NOT_EVALUABLE','reason':'requires measured 5000-reference pool, T2 comparisons, and matched real baseline artifacts; a person decides gates'},
                pool_protocol={'seed':config.get('seed',0),'candidate_order':ordered,'pool_sizes':sizes,
                               'scale':'toy' if available < 100 else 'measured','extrapolation':'not_run'},
                curves=pd.DataFrame(curves),coverage=all_rows)
