"""Random-item exchange rates, matched selection pairs and cluster intervals.

Binary item equivalence requires explicit ``scores.kind == 'binary'``. A
composite requires documented per-item ``item_variance`` and
``item_variance_source``; score sampling variance is not per-item variance.
The numeric primitive accepts the caller's supplied per-item variance directly.
Selection assumes independent candidate items unless aligned binary response
dicts (``item_responses``, keyed by item ID) establish their covariance.
"""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import hashlib
import json

import numpy as np
import pandas as pd

from .statistics import cluster_bootstrap


DEPLOYMENT = ['model_key', 'benchmark', 'task', 'harness', 'precision', 'quantization']
STRATA = ['benchmark', 'task', 'harness', 'precision', 'quantization', 'setting', 'fold', 'analysis_tier']
FAMILY_STRATA = ['within_family', 'across_family', 'ancestry_ambiguous', 'ancestry_unavailable']
PAIR_COLUMNS = ['method', *STRATA, 'family_stratum', 'model_i', 'model_j', 'actual_difference', 'predicted_difference', 'residual_difference', 'independent_item_variance', 'item_variance', 'item_covariance', 'item_assumption', 'split_group_i', 'split_group_j', 'ranking_accuracy', 'regret', 'candidate_pairs', 'sampled_pairs', 'sampling_seed']


def _text(value):
    return '' if value is None or (np.isscalar(value) and pd.isna(value)) else str(value).strip()


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _bool(value):
    return value is True or isinstance(value, np.bool_) and bool(value) or _text(value).lower() == 'true'


def _rate(actual, predicted, item_variance, *, pairwise=False, weights=None):
    y, prediction, q = [np.asarray(value, dtype=float) for value in (actual, predicted, item_variance)]
    if y.ndim != 1 or prediction.shape != y.shape or q.shape != y.shape:
        raise ValueError('actual, predicted and item_variance must be matching vectors')
    weights = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != y.shape or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('weights must be a finite nonnegative matching vector')
    finite = np.isfinite(y) & np.isfinite(prediction)
    observed = (y >= (-1 if pairwise else 0)) & (y <= 1)
    valid = finite & observed & (weights > 0)
    output = dict(status='unavailable', reason=None, n=int(valid.sum()), n_input=len(y),
                  excluded_nonfinite=int((~finite).sum()), invalid_observations=int((finite & ~observed).sum()),
                  out_of_range_predictions=int((valid & ((prediction < (-1 if pairwise else 0)) | (prediction > 1))).sum()),
                  mae=None, mse=None, rmse=None, q=None, k=None, total_items=None,
                  strict_mae_upper=None, universal_mae_upper=None,
                  historical_approximation=None, approximation_label='historical_rmse_1.25_mae_approximation_not_bound',
                  mae_bound_status='unavailable', k_unit='items_per_candidate' if pairwise else 'items_per_model',
                  universal_numerator_bound=.5 if pairwise else .25,
                  universal_bound_assumption='independent_binary_candidate_items' if pairwise else 'binary_items',
                  adaptive_factor_2=None, adaptive_factor_4=None,
                  adaptive_label='sensitivity_not_measured_irt_savings')
    if not valid.any():
        return dict(output, reason='insufficient_support')
    y, prediction, q, weights = y[valid], prediction[valid], q[valid], weights[valid]
    weights = weights/weights.sum()
    residual = prediction-y
    mae = float(weights @ np.abs(residual))
    mse = float(weights @ residual**2)
    output.update(mae=mae, mse=mse, rmse=float(np.sqrt(mse)))
    if mae == 0:
        output['mae_bound_status'] = 'unbounded'
    else:
        output.update(mae_bound_status='finite', universal_mae_upper=output['universal_numerator_bound']/mae**2,
                      historical_approximation=output['universal_numerator_bound']/(1.25*mae)**2)
    if not np.isfinite(q).all() or (q < 0).any():
        return dict(output, reason='item_equivalence_unavailable')
    numerator = float(weights @ q)
    output['q'] = numerator
    if numerator > output['universal_numerator_bound'] + 1e-12:
        output.update(universal_mae_upper=None, historical_approximation=None,
                      universal_bound_assumption='binary_bound_not_applicable_to_supplied_variance')
    if mae > 0:
        output['strict_mae_upper'] = numerator/mae**2
    if mse == 0:
        return dict(output, status='perfect_on_sample', reason='perfect_on_sample',
                    strict_mae_upper=None, universal_mae_upper=None,
                    historical_approximation=None, mae_bound_status='unbounded')
    k = numerator/mse
    return dict(output, status='ok', k=k, total_items=(2 if pairwise else 1)*k,
                adaptive_factor_2=k/2, adaptive_factor_4=k/4)


def exchange_rate(actual: np.ndarray, predicted: np.ndarray, item_variance: np.ndarray, *, pairwise=False) -> dict:
    """Price measured residual MSE in caller-documented random binary items.

    Pairwise k is items per candidate and total_items is 2*k. Supplied q is
    authoritative; no approximation replaces observed RMSE. Missing q retains
    descriptive error but makes item equivalence unavailable.
    """
    return _rate(actual, predicted, item_variance, pairwise=pairwise)


class _PairIndex:
    """Rank eligible pairs with posting lists, using O(n) storage."""

    def __init__(self, rows):
        self.rows = rows
        self.postings = defaultdict(list)
        for i, row in enumerate(rows):
            root, split, duplicate = (_text(row.get(key)) for key in ('root_group', 'split_group', 'duplicate_group'))
            features = [('all', '')]
            if root:
                features.extend([('known', ''), ('root', root)])
                if split:
                    features.extend([('both', ''), ('both_root', root), ('both_split', split), ('both_root_split', (root, split))])
            for kind, value in features:
                self.postings[(kind, value, None)].append(i)
                if duplicate:
                    self.postings[(kind, value, duplicate)].append(i)

    def count(self, i, stop, stratum, duplicate=None):
        start = i+1
        row = self.rows[i]
        root, split = _text(row.get('root_group')), _text(row.get('split_group'))

        def count(kind, value=''):
            positions = self.postings.get((kind, value, duplicate), ())
            return bisect_right(positions, stop)-bisect_left(positions, start)

        if stratum == 'within_family':
            return count('root', root) if root else 0
        across = (count('both')-count('both_root', root)-count('both_split', split)+count('both_root_split', (root, split))) if root and split else 0
        if stratum == 'across_family':
            return across
        if stratum == 'ancestry_ambiguous':
            return count('known')-count('root', root)-across if root else 0
        return count('all')-count('known') if root else count('all')

    def eligible(self, i, stop, stratum):
        count = self.count(i, stop, stratum)
        duplicate = _text(self.rows[i].get('duplicate_group'))
        return count-self.count(i, stop, stratum, duplicate) if duplicate else count

    def sample(self, stratum, cap, rng):
        n = len(self.rows)
        cumulative = np.cumsum([self.eligible(i, n-1, stratum) for i in range(n)], dtype=np.int64)
        total = int(cumulative[-1]) if n else 0
        if total <= cap:
            ranks = range(total)
        else:
            # Floyd's algorithm samples integer ranks uniformly, without an
            # O(total) permutation or a materialized quadratic candidate list.
            chosen = set()
            for upper in range(total-cap, total):
                candidate = int(rng.integers(0, upper+1))
                chosen.add(upper if candidate in chosen else candidate)
            ranks = sorted(chosen)
        for rank in ranks:
            i = int(np.searchsorted(cumulative, rank, side='right'))
            offset = rank-(int(cumulative[i-1]) if i else 0)
            left, right = i+1, n-1
            while left < right:
                middle = (left+right)//2
                if self.eligible(i, middle, stratum) > offset:
                    right = middle
                else:
                    left = middle+1
            yield i, left, total, min(total, cap)


def _item_variance(row):
    if _text(row.get('kind')) == 'binary':
        p = _number(row.get('actual'))
        return p*(1-p) if p is not None and 0 <= p <= 1 else None
    variance = _number(row.get('item_variance'))
    if variance is not None and variance >= 0 and _text(row.get('item_variance_source')):
        return variance
    return None


def _response_covariance(a, b):
    left, right = a.get('item_responses'), b.get('item_responses')
    if not isinstance(left, dict) or not left or not isinstance(right, dict) or left.keys() != right.keys():
        return None
    keys = sorted(left)
    try:
        x, y = np.asarray([left[key] for key in keys], dtype=float), np.asarray([right[key] for key in keys], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isin(x, [0., 1.]).all() or not np.isin(y, [0., 1.]).all():
        return None
    if not np.isclose(x.mean(), a['actual'], atol=1e-12, rtol=0) or not np.isclose(y.mean(), b['actual'], atol=1e-12, rtol=0):
        return None
    return float(np.mean((x-x.mean())*(y-y.mean())))


def _pair(a, b):
    delta = float(a['actual']-b['actual'])
    prediction = float(a['prediction']-b['prediction'])
    independent = float(a['actual']*(1-a['actual'])+b['actual']*(1-b['actual']))
    q_a, q_b = _item_variance(a), _item_variance(b)
    variance = q_a+q_b if q_a is not None and q_b is not None else None
    covariance = _response_covariance(a, b) if _text(a.get('kind')) == _text(b.get('kind')) == 'binary' else None
    assumption = 'independent_item_approximation' if variance is not None else 'item_equivalence_unavailable'
    if covariance is not None:
        variance = max(0., variance-2*covariance) if variance is not None else None
        assumption = 'documented_shared_binary_item_responses'
    ranking = 1. if delta == 0 else .5 if prediction == 0 else float(np.sign(delta) == np.sign(prediction))
    return dict(model_i=a['model_key'], model_j=b['model_key'], actual_difference=delta,
                predicted_difference=prediction, residual_difference=prediction-delta,
                independent_item_variance=independent, item_variance=variance,
                item_covariance=covariance, item_assumption=assumption,
                split_group_i=_text(a.get('split_group')), split_group_j=_text(b.get('split_group')),
                ranking_accuracy=ranking, regret=abs(delta)*(1-ranking))


def selection_pairs(predictions: pd.DataFrame, *, max_pairs=100000, seed=20261028) -> pd.DataFrame:
    """One unordered pair per task/method/fold and audited ancestry stratum.

    Common model support is used across methods within each deployment/fold.
    Unknown ancestry is labeled, never assigned an arbitrary root. Shared split
    ancestry with different root-set labels is ambiguous, not across-family.
    Returned attrs expose input exclusions and sampling counts; each row also
    carries candidate/sample counts. Algebraic independent_item_variance is a
    diagnostic only; item_variance requires the explicit task-kind evidence.
    """
    if not isinstance(max_pairs, int) or max_pairs < 1:
        raise ValueError('max_pairs must be a positive integer')
    frame = predictions.copy()
    if any(key not in frame for key in ('model_key', 'actual', 'prediction', 'method', 'task', 'fold')):
        raise ValueError('selection requires model_key, actual, prediction, method, task and fold')
    for key in STRATA:
        if key not in frame:
            frame[key] = ''
        frame[key] = frame[key].map(_text)
    frame['model_key'] = frame.model_key.map(_text)
    frame['actual'] = pd.to_numeric(frame.actual, errors='coerce')
    frame['prediction'] = pd.to_numeric(frame.prediction, errors='coerce')
    valid = np.isfinite(frame.actual) & frame.actual.between(0, 1) & np.isfinite(frame.prediction) & frame.model_key.ne('')
    exclusions = [{'reason': 'invalid_pair_input', 'count': int((~valid).sum())}] if not valid.all() else []
    frame = frame.loc[valid].copy()
    output, sampling = [], []
    for values, group in frame.groupby(STRATA, sort=True, dropna=False):
        metadata = dict(zip(STRATA, values))
        methods = {}
        for method, rows in group.groupby('method', sort=True):
            duplicated = rows.model_key.duplicated(keep=False)
            if duplicated.any():
                exclusions.append(dict(reason='ambiguous_pair_model', method=method, count=int(duplicated.sum())))
            methods[method] = rows.loc[~duplicated].set_index('model_key', drop=False)
        support = sorted(set.intersection(*(set(rows.index) for rows in methods.values()))) if methods else []
        for method, rows in methods.items():
            if len(rows) != len(support):
                exclusions.append(dict(reason='baseline_support_mismatch', method=method, count=len(rows)-len(support)))
        if len(support) < 2:
            continue
        # Roots, duplicate labels and actuals must agree on matched methods.
        compatible = []
        for model in support:
            records = [rows.loc[model].to_dict() for rows in methods.values()]
            if any(any(_text(record.get(key)) != _text(records[0].get(key)) for key in ('root_group', 'split_group', 'duplicate_group', 'actual')) for record in records[1:]):
                exclusions.append(dict(reason='matched_model_metadata_mismatch', model_key=model, count=1))
            else:
                compatible.append(model)
        support = compatible
        first = methods[sorted(methods)[0]].loc[support].to_dict('records')
        index = _PairIndex(first)
        # Method is deliberately absent from the seed: matched methods reuse
        # identical sampled pairs, including independent calls on one method.
        for stratum in FAMILY_STRATA:
            salt = json.dumps([seed, *values, stratum], separators=(',', ':')).encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(salt).digest()[:8], 'little'))
            sampled = list(index.sample(stratum, max_pairs, rng))
            candidate_count = sampled[0][2] if sampled else 0
            sampling.append(dict(metadata, family_stratum=stratum, candidate_pairs=candidate_count, sampled_pairs=len(sampled), max_pairs=max_pairs, seed=seed))
            for method, rows in sorted(methods.items()):
                by_model = rows.to_dict('index')
                for i, j, total, count in sampled:
                    pair = _pair(by_model[support[i]], by_model[support[j]])
                    output.append(dict(pair, **metadata, method=method, family_stratum=stratum,
                                       candidate_pairs=total, sampled_pairs=count, sampling_seed=seed))
    result = pd.DataFrame(output, columns=PAIR_COLUMNS)
    if not result.empty:
        result = result.sort_values(STRATA+['family_stratum', 'model_i', 'model_j', 'method']).reset_index(drop=True)
    result.attrs.update(exclusions=exclusions, sampling=sampling, sampling_design='seeded_uniform_unordered_pairs_without_replacement')
    return result


def _bootstrap_fields(result):
    fields = {f'bootstrap_{key}': result.get(key) for key in ('status', 'reason', 'n_clusters', 'n_bootstrap', 'n_successful', 'n_failed', 'failure_rate', 'metric_successes', 'failures')}
    for metric in ('k', 'incremental_k', 'mse'):
        interval = result.get('intervals', {}).get(metric)
        fields[f'{metric}_lower'], fields[f'{metric}_upper'] = interval if interval else (None, None)
    return fields


def _bootstrap(source, statistic, n_bootstrap, seed):
    if source.empty or 'split_group' not in source or source.split_group.map(_text).eq('').any():
        return _bootstrap_fields(dict(status='unavailable', reason='missing_source_split_groups', n_bootstrap=n_bootstrap, intervals={}))
    return _bootstrap_fields(cluster_bootstrap(source, statistic, n_bootstrap=n_bootstrap, seed=seed))


def _estimation_stat(sample, method, baseline):
    own = sample.loc[sample.method.eq(method)]
    rate = exchange_rate(own.actual.to_numpy(float), own.prediction.to_numpy(float), own['_item_variance'].to_numpy(float))
    result = dict(k=rate['k'], mse=rate['mse'])
    if baseline and method != baseline:
        other = sample.loc[sample.method.eq(baseline)]
        base = exchange_rate(other.actual.to_numpy(float), other.prediction.to_numpy(float), other['_item_variance'].to_numpy(float))
        result['incremental_k'] = rate['k']-base['k'] if rate['k'] is not None and base['k'] is not None else None
    return result


def _selection_stat(pairs, source, sample, method, baseline):
    original_counts = source.groupby('split_group').size()
    multiplicity = sample.groupby('split_group').size().div(original_counts).fillna(0)

    def estimate(name):
        own = pairs.loc[pairs.method.eq(name)]
        left = own.split_group_i.map(multiplicity).fillna(0).to_numpy()
        right = own.split_group_j.map(multiplicity).fillna(0).to_numpy()
        # Each source draw repeats its internal pair block once. Products
        # apply only to pair blocks between distinct resampled source groups.
        weights = np.where(own.split_group_i.eq(own.split_group_j), left, left*right)
        return _rate(own.actual_difference.to_numpy(float), own.predicted_difference.to_numpy(float), own.item_variance.to_numpy(float), pairwise=True, weights=weights)

    rate = estimate(method)
    result = dict(k=rate['k'], mse=rate['mse'])
    if baseline and method != baseline:
        base = estimate(baseline)
        result['incremental_k'] = rate['k']-base['k'] if rate['k'] is not None and base['k'] is not None else None
    return result


def exchange_table(predictions: pd.DataFrame, scores: pd.DataFrame, *, n_bootstrap=2000, seed=20261028, max_pairs=100000) -> pd.DataFrame:
    """Build estimation/selection cells, matched increments and coverage reasons.

    No prediction normalization is inverted. Ineligible provenance is retained
    as a separate historical descriptive tier; unsupported scales and missing
    score joins receive explicit unavailable cells. Bootstrap resamples complete
    source split groups. Selection uses source-group multiplicities on the fixed
    prespecified pair sample, never bootstrap draws of independent pair rows.
    """
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError('n_bootstrap must be a positive integer')
    frame = predictions.copy()
    audits, retained = [], []
    blocked_identities = set()

    def audit(row, reason, **extra):
        audits.append(dict({key: row.get(key, '') for key in STRATA+['method']}, estimand='coverage', family_stratum='',
                           status='unavailable', reason=reason, k=None, incremental_k=None, **extra))

    # Partial legacy identities are allowed only in descriptive coverage. A
    # complete verified cell still requires all six deployment identity fields.
    lookup = defaultdict(list)
    for row in scores.to_dict('records'):
        identity = tuple(_text(row.get(key)).casefold() if key == 'model_key' else _text(row.get(key)) for key in DEPLOYMENT)
        lookup[identity].append(row)
    for row in frame.to_dict('records'):
        for key in STRATA+DEPLOYMENT+['method']:
            row.setdefault(key, '')
        raw = _text(row.get('prediction_scale')) == 'raw_accuracy'
        actual, prediction = _number(row.get('actual')), _number(row.get('prediction'))
        identity = tuple(_text(row.get(key)).casefold() if key == 'model_key' else _text(row.get(key)) for key in DEPLOYMENT)
        complete = all(identity)
        row['analysis_tier'] = 'verified_protocol_eligible' if complete and _bool(row.get('revision_verified')) and row.get('provenance_status') == 'eligible' and row.get('setting') in {'within_family', 'root_holdout'} else 'historical_descriptive'
        if not raw:
            audit(row, 'raw_prediction_unavailable')
            continue
        if actual is None or not 0 <= actual <= 1 or prediction is None:
            audit(row, 'invalid_observation' if actual is None or not 0 <= actual <= 1 else 'nonfinite_prediction')
            continue
        matched = lookup.get(identity, [])
        # Scores can be supplied once per method by small fixtures; equal
        # score evidence is a single observation, conflicting evidence is not.
        signatures = {(_number(score.get('accuracy')), _text(score.get('kind')), _number(score.get('item_variance')), _text(score.get('item_variance_source'))) for score in matched}
        if len(signatures) != 1:
            audit(row, 'score_join_ambiguous' if matched else 'score_join_unavailable')
            continue
        group_evidence = {key: {_text(item.get(key)) for item in matched if _text(item.get(key))}
                          for key in ('root_group', 'duplicate_group', 'split_group')}
        if any(len(values) > 1 for values in group_evidence.values()):
            row['analysis_tier'] = 'historical_descriptive'
            audit(row, 'conflicting_score_group_evidence')
            blocked_identities.add(identity)
            continue
        canonical_groups = {key: next(iter(values)) if values else '' for key, values in group_evidence.items()}
        if any(_text(row.get(key)) and value and _text(row.get(key)) != value for key, value in canonical_groups.items()):
            row['analysis_tier'] = 'historical_descriptive'
            audit(row, 'matched_model_metadata_mismatch')
            blocked_identities.add(identity)
            continue
        score = matched[0]
        observed = _number(score.get('accuracy'))
        if observed is None or not np.isclose(observed, actual, rtol=0, atol=1e-12):
            audit(row, 'actual_score_mismatch')
            continue
        # Task evidence and group metadata come from scores when available.
        for key in ('kind', 'item_variance', 'item_variance_source', 'item_responses'):
            if key in score:
                row[key] = score[key]
        for key in ('root_group', 'duplicate_group', 'split_group'):
            if not _text(row.get(key)):
                row[key] = canonical_groups[key]
        if not all(_bool(item.get('revision_verified')) for item in matched) or not all(_text(row.get(key)) for key in ('root_group', 'duplicate_group', 'split_group')):
            row['analysis_tier'] = 'historical_descriptive'
        row.update(actual=actual, prediction=prediction, _item_variance=_item_variance(dict(row, actual=actual)), _deployment_identity=identity)
        retained.append(row)
    compatible = []
    for row in retained:
        if row['_deployment_identity'] in blocked_identities:
            row['analysis_tier'] = 'historical_descriptive'
            audit(row, 'matched_model_metadata_mismatch')
        else:
            compatible.append(row)
    retained = compatible
    frame = pd.DataFrame(retained)
    output = []
    if frame.empty:
        return pd.DataFrame(audits, columns=list(dict.fromkeys(STRATA+['method', 'estimand', 'family_stratum', 'status', 'reason', 'k', 'incremental_k'])))
    for key in STRATA:
        frame[key] = frame[key].map(_text)
    for values, group in frame.groupby(STRATA, sort=True, dropna=False):
        metadata = dict(zip(STRATA, values))
        # Conflicting or repeated predictions cannot become pseudo-replicates.
        ambiguous = group.duplicated(['method', 'model_key'], keep=False)
        if ambiguous.any():
            for row in group.loc[ambiguous].to_dict('records'):
                audit(row, 'ambiguous_prediction_identity')
            group = group.loc[~ambiguous]
        group = group.sort_values(['method', 'model_key'])
        methods = sorted(group.method.unique())
        if not methods:
            continue
        baseline = 'no_weight' if 'no_weight' in methods else None
        if baseline:
            support = set.intersection(*(set(rows.model_key) for _, rows in group.groupby('method')))
            missing = group.loc[~group.model_key.isin(support)]
            for method, rows in missing.groupby('method', sort=True):
                audit(dict(metadata, method=method), 'baseline_support_mismatch', n_excluded=len(rows))
            group = group.loc[group.model_key.isin(support)].copy()
            compatible = []
            for model, records in group.groupby('model_key', sort=True):
                evidence = records.to_dict('records')
                if any(any(_text(row.get(key)) != _text(evidence[0].get(key)) for key in ('actual', 'root_group', 'split_group', 'duplicate_group')) for row in evidence[1:]):
                    for row in evidence:
                        audit(row, 'matched_model_metadata_mismatch', n_excluded=1)
                else:
                    compatible.append(model)
            group = group.loc[group.model_key.isin(compatible)].copy()
        if group.empty:
            continue
        estimates = {}
        for method, rows in group.groupby('method', sort=True):
            estimates[method] = exchange_rate(rows.actual.to_numpy(float), rows.prediction.to_numpy(float), rows['_item_variance'].to_numpy(float))
        for method, estimate in sorted(estimates.items()):
            own = group.loc[group.method.eq(method)]
            binary = own.kind.eq('binary').all() if 'kind' in own else False
            if not binary:
                estimate.update(universal_mae_upper=None, historical_approximation=None,
                                universal_bound_assumption='binary_bound_not_applicable_to_task_design')
            base = estimates.get(baseline, {})
            incremental = estimate['k']-base['k'] if method != baseline and estimate['k'] is not None and base.get('k') is not None else None
            interval = _bootstrap(group, lambda sample, method=method: _estimation_stat(sample, method, baseline), n_bootstrap, seed)
            output.append(dict(metadata, method=method, estimand='estimation', family_stratum='', **estimate,
                               incremental_k=incremental, incremental_bound_status='no_bound_from_subtracting_bounds',
                               item_assumption='binary_items' if binary else 'documented_per_item_variance_design' if estimate['q'] is not None else 'item_equivalence_unavailable',
                               comparison_status='matched_support' if baseline else 'baseline_unavailable', **interval))
        pairs = selection_pairs(group, max_pairs=max_pairs, seed=seed)
        for reason in pairs.attrs.get('exclusions', []):
            audit(dict(metadata, method=reason.get('method', '')), reason['reason'], n_excluded=reason.get('count', 0))
        if pairs.empty:
            for method in methods:
                audit(dict(metadata, method=method), 'selection_pairs_unavailable')
            continue
        for stratum, all_pairs in pairs.groupby('family_stratum', sort=True):
            pair_estimates = {}
            for method, own in all_pairs.groupby('method', sort=True):
                pair_estimates[method] = exchange_rate(own.actual_difference.to_numpy(float), own.predicted_difference.to_numpy(float), own.item_variance.to_numpy(float), pairwise=True)
            for method, own in all_pairs.groupby('method', sort=True):
                estimate = pair_estimates[method]
                base = pair_estimates.get(baseline, {})
                incremental = estimate['k']-base['k'] if method != baseline and estimate['k'] is not None and base.get('k') is not None else None
                # Source rows (one method) retain every group member; only
                # their resampling multiplicities weight the shared pair set.
                source = group.loc[group.method.eq(method)].copy()
                interval = _bootstrap(source, lambda sample, method=method: _selection_stat(all_pairs, source, sample, method, baseline), n_bootstrap, seed)
                assumptions = sorted(set(own.item_assumption))
                if assumptions != ['independent_item_approximation'] or ('kind' in group and not group.kind.eq('binary').all()):
                    estimate.update(universal_mae_upper=None, historical_approximation=None,
                                    universal_bound_assumption='independent_binary_bound_not_applicable_to_task_design')
                output.append(dict(metadata, method=method, estimand='selection', family_stratum=stratum, **estimate,
                                   incremental_k=incremental, ranking_accuracy=float(own.ranking_accuracy.mean()), regret=float(own.regret.mean()),
                                   candidate_pairs=int(own.candidate_pairs.iloc[0]), sampled_pairs=len(own), sampling_seed=seed,
                                   item_assumption=';'.join(assumptions), bootstrap_pair_design='source_split_group_product_multiplicities_on_fixed_uniform_pair_sample',
                                   bootstrap_within_group_rule='repeat_internal_pair_block_once_per_source_draw',
                                   incremental_bound_status='no_bound_from_subtracting_bounds', comparison_status='matched_support' if baseline else 'baseline_unavailable', **interval))
    result = pd.DataFrame(output+audits)
    if not result.empty:
        result = result.sort_values(STRATA+['estimand', 'family_stratum', 'method', 'reason'], na_position='last').reset_index(drop=True)
    return result
