"""Known-noise REML and observed-parent numerical primitives.

Callers construct verified compatible pairs, align covariance rows, and stratify
by benchmark/task/harness, relation evidence tier and scale. These routines never
pool tasks or infer measurement evidence. Familial fractions describe a specified
covariance model, not causal heritability. All moments use population divisors.
"""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, null_space
from scipy.optimize import minimize, minimize_scalar


def _design(y, x):
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or x.shape[0] != len(y):
        raise ValueError('y and x must have shapes (n,) and (n,p)')
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError('y and x must be finite')
    if x.shape[1] < 1 or np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError('x must have full column rank')
    return y, x


def reml_nll(y: np.ndarray, x: np.ndarray, v: np.ndarray) -> float:
    """Restricted Gaussian negative log likelihood, including its constant."""
    y, x = _design(y, x)
    n, p = x.shape
    if n <= p:
        raise ValueError('REML requires n > p')
    v = np.asarray(v, dtype=float)
    if v.shape != (n, n) or not np.isfinite(v).all() or not np.allclose(v, v.T):
        raise ValueError('v must be a finite symmetric covariance matrix')
    factor = cho_factor(v, lower=True, check_finite=False)
    vx = cho_solve(factor, x, check_finite=False)
    information = cho_factor(x.T @ vx, lower=True, check_finite=False)
    beta = cho_solve(information, x.T @ cho_solve(factor, y, check_finite=False), check_finite=False)
    residual = y - x @ beta
    quadratic = residual @ cho_solve(factor, residual, check_finite=False)
    logdet = 2*np.log(np.diag(factor[0])).sum()
    logdet_x = 2*np.log(np.diag(information[0])).sum()
    return float(.5*(logdet + logdet_x + quadratic + (n-p)*np.log(2*np.pi)))


def _measurement_d(value, n):
    d = np.asarray(value, dtype=float)
    if d.shape == (n,):
        diagonal = d
    elif d.shape == (n, n) and np.array_equal(d, np.diag(np.diag(d))):
        diagonal = np.diag(d)
    else:
        raise ValueError('measurement_variance must be a diagonal vector or diagonal matrix')
    if not np.isfinite(diagonal).all() or (diagonal < 0).any():
        raise ValueError('measurement_variance must be finite and nonnegative')
    return np.diag(diagonal)


def fit_reml(y, x, kernel, measurement_variance, *, covariance_model='family', families=None) -> dict:
    """Fit V=tau2*K_normalized+sigma2*I+D with explicit zero boundaries.

    K is divided by its mean diagonal before fitting, so reported tau2 is the
    mean latent familial variance. ``kernel_normalization`` reports the original
    mean diagonal divisor, including when it is very small. Identifiability
    is tested on the normalized covariance. ``families`` supplies the known count;
    otherwise unique rows count families only for ``covariance_model='family'``.
    Unsupported residual designs and optimizer failures return no estimates.
    Optimizer uses relative component units, ftol=1e-12, gtol=1e-7, maxiter=500.
    """
    y, x = _design(y, x)
    n, p = x.shape
    d = _measurement_d(measurement_variance, n)
    k = np.asarray(kernel, dtype=float)
    if k.shape != (n, n) or not np.isfinite(k).all() or not np.allclose(k, k.T):
        raise ValueError('kernel must be finite, symmetric and aligned with y')
    if np.linalg.eigvalsh(k).min() < -1e-9*max(1., np.abs(k).max()):
        raise ValueError('kernel must be positive semidefinite')
    normalization = float(np.trace(k)/n)
    if families is None and covariance_model == 'family':
        families = int(len(np.unique(k, axis=0)))
    result = dict(status='unavailable', reason=None, converged=False,
                  boundary=False, n=n, families=families,
                  covariance_model=covariance_model, kernel_normalization=normalization,
                  tau2=None, sigma2=None, fixed_effects=None, familial_fraction=None,
                  measurement_variance_fraction=None, noise_ceiling=None,
                  mean_measurement_variance=float(np.trace(d)/n))
    if n <= p:
        return dict(result, reason='insufficient_residual_support')
    if normalization <= 0:
        return dict(result, reason='nonidentifiable_kernel')
    k = k/normalization
    q = null_space(x.T)
    residual_k = q.T @ k @ q
    projected_mean = np.trace(residual_k)/(n-p)
    if np.allclose(residual_k, projected_mean*np.eye(n-p), rtol=1e-8, atol=1e-10*max(1., np.abs(k).max())):
        return dict(result, reason='nonidentifiable_kernel')
    identity = np.eye(n)
    residual = y-x @ np.linalg.lstsq(x, y, rcond=None)[0]
    unit = max(float(residual @ residual/(n-p)), float(np.trace(d)/n), 1e-8)

    def objective(components):
        try:
            return reml_nll(y, x, unit*(components[0]*k+components[1]*identity)+d)
        except np.linalg.LinAlgError:
            return np.inf

    candidates = []
    optimizers = []
    for start in ([.3, .7], [.7, .3]):
        fit = minimize(objective, start, method='L-BFGS-B', bounds=[(0., None), (0., None)],
                       options=dict(ftol=1e-12, gtol=1e-7, maxiter=500))
        optimizers.append(bool(fit.success))
        if fit.success and np.isfinite(fit.fun):
            candidates.append((float(fit.fun), np.asarray(fit.x)))
    # Separate one-dimensional searches include each zero-component face.
    # Grow the bracket until its minimum is interior; do not silently cap a fit.
    for free in (0, 1):
        upper = 10.
        for _ in range(8):
            def boundary_objective(value):
                components = np.zeros(2)
                components[free] = value
                return objective(components)
            fit = minimize_scalar(boundary_objective, bounds=(0., upper), method='bounded', options={'xatol':1e-10, 'maxiter':500})
            if fit.success and fit.x < upper*.99:
                components = np.zeros(2)
                components[free] = fit.x
                candidates.append((float(fit.fun), components))
                break
            upper *= 10
    origin = objective([0., 0.])
    if np.isfinite(origin):
        candidates.append((origin, np.zeros(2)))
    if not any(optimizers) or not candidates:
        return dict(result, reason='optimizer_failure')
    best_value = min(item[0] for item in candidates)
    # Equivalent optima prefer the explicit boundary over numerical epsilon.
    eligible = [item for item in candidates if item[0] <= best_value+1e-8]
    value, components = min(eligible, key=lambda item: (np.count_nonzero(item[1]), item[0]))
    tau2, sigma2 = (float(v*unit) for v in components)
    v = tau2*k+sigma2*identity+d
    factor = cho_factor(v, lower=True)
    vx = cho_solve(factor, x)
    beta = np.linalg.solve(x.T @ vx, x.T @ cho_solve(factor, y))
    latent = tau2+sigma2
    total = latent+result['mean_measurement_variance']
    return dict(result, status='ok', reason=None if latent > 0 else 'zero_latent_variance',
                converged=True, boundary=bool((components == 0).any()),
                tau2=tau2, sigma2=sigma2, fixed_effects=beta.tolist(), nll=value,
                familial_fraction=tau2/latent if latent > 0 else None,
                measurement_variance_fraction=result['mean_measurement_variance']/total if total > 0 else None,
                noise_ceiling=latent/total if total > 0 else None)


def parent_offspring(pairs: pd.DataFrame) -> dict:
    """Population-moment slopes on caller-verified single-parent pairs.

    The naive slope uses all finite observed pairs. Corrected moments use pairs
    with known finite nonnegative parent variance. Paired error covariance is
    subtracted only if available for every retained pair; otherwise the entire
    correction assumes independent parent/child errors, explicitly labeled.
    Repeated parents require ``cluster_bootstrap`` by split_group in the caller.
    """
    required = ['parent_accuracy', 'child_accuracy']
    numeric = pairs.reindex(columns=required+['parent_variance', 'error_covariance']).apply(pd.to_numeric, errors='coerce').astype(float)
    finite = np.isfinite(numeric[required]).all(axis=1)
    observed = numeric.loc[finite]
    valid_variance = np.isfinite(observed.parent_variance) & observed.parent_variance.ge(0)
    corrected = observed.loc[valid_variance]
    result = dict(status='unavailable', reason=None, n_input=len(pairs), n=len(observed),
                  excluded_pairs=int((~finite).sum()), n_corrected=len(corrected),
                  excluded_variance=int((~valid_variance).sum()), naive_slope=None,
                  corrected_slope=None, intercept=None, naive_intercept=None,
                  corrected_intercept=None, variance_denominator=None,
                  error_assumption='independent_parent_child_errors',
                  error_covariance_available=int(np.isfinite(corrected.error_covariance).sum()))
    if len(observed) < 2:
        return dict(result, reason='insufficient_pairs')
    parent = observed.parent_accuracy.to_numpy()
    child = observed.child_accuracy.to_numpy()
    variance = float(np.var(parent))
    if variance > 0:
        slope = float(np.mean((parent-parent.mean())*(child-child.mean()))/variance)
        result.update(naive_slope=slope, naive_intercept=float(child.mean()-slope*parent.mean()))
    if len(corrected) < 2:
        return dict(result, reason='insufficient_corrected_pairs')
    parent = corrected.parent_accuracy.to_numpy()
    child = corrected.child_accuracy.to_numpy()
    denominator = float(np.var(parent)-corrected.parent_variance.mean())
    result['variance_denominator'] = denominator
    if denominator <= 0:
        return dict(result, reason='nonpositive_corrected_denominator')
    covariance = 0.
    if np.isfinite(corrected.error_covariance).all():
        covariance = float(corrected.error_covariance.mean())
        result['error_assumption'] = 'documented_paired_error_covariance'
    numerator = float(np.mean((parent-parent.mean())*(child-child.mean()))-covariance)
    slope = numerator/denominator
    intercept = float(child.mean()-slope*parent.mean())
    return dict(result, status='ok', corrected_slope=slope, intercept=intercept, corrected_intercept=intercept)


def prepare_scale(scores: pd.DataFrame, scale='raw') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select known-variance support and optionally apply documented logit correction.

    Input is assembled scores, where variance_source starting ``binomial:``
    attests documented ordinary binary counts. Logit uses (p*n+.5)/(n+1),
    recomputes corrected binomial variance then applies the delta method. A
    composite's documented raw variance never authorizes binomial logit variance.
    Retained rows preserve order/index for covariance alignment; exclusions carry
    original row_index and record_id. Call separately for raw/logit sensitivity.
    """
    if scale not in {'raw', 'logit'}:
        raise ValueError('scale must be raw or logit')
    retained, exclusions = [], []
    for index, row in scores.iterrows():
        row = row.to_dict()
        try:
            p = float(row.get('accuracy', np.nan))
        except (TypeError, ValueError):
            p = np.nan
        try:
            variance = float(row.get('measurement_variance', np.nan))
        except (TypeError, ValueError):
            variance = np.nan
        reason = None
        if not np.isfinite(p) or not 0 <= p <= 1:
            reason = 'invalid_accuracy'
        elif scale == 'raw' and (not np.isfinite(variance) or variance < 0):
            reason = 'variance_unavailable'
        elif scale == 'logit':
            try:
                n = float(row.get('n_items', np.nan))
            except (TypeError, ValueError):
                n = np.nan
            if not str(row.get('variance_source', '')).startswith('binomial:') or not np.isfinite(n) or n <= 0 or not n.is_integer():
                reason = 'logit_requires_documented_binomial_counts'
            else:
                corrected = (p*n+.5)/(n+1)
                row['accuracy'] = float(np.log(corrected/(1-corrected)))
                binomial_variance = corrected*(1-corrected)/(n+1)
                row['measurement_variance'] = float(binomial_variance/(corrected*(1-corrected))**2)
        if reason:
            exclusions.append(dict(stage='statistics', row_index=index, record_id=row.get('record_id', ''), reason=reason))
        else:
            row.update(scale=scale, correction='binomial_jeffreys_half_count' if scale == 'logit' else 'none')
            retained.append((index, row))
    output = pd.DataFrame([row for _, row in retained], index=[index for index, _ in retained], columns=list(scores.columns)+[name for name in ['scale','correction'] if name not in scores])
    return output, pd.DataFrame(exclusions, columns=['stage','row_index','record_id','reason'])


def skill_summary(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Paired finite support; population target SD, untruncated R squared.

    For comparisons the caller must first align the identical target set across
    methods; reported exclusions make accidental support differences visible.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError('actual and predicted must be matching vectors')
    finite = np.isfinite(actual) & np.isfinite(predicted)
    y, prediction = actual[finite], predicted[finite]
    result = dict(n=len(y), excluded=int((~finite).sum()), mae=None, rmse=None, mae_sd=None, r2=None, status='unavailable', reason=None)
    if not len(y):
        return dict(result, reason='insufficient_support')
    error = prediction-y
    mae, mse = float(np.mean(np.abs(error))), float(np.mean(error**2))
    variance = float(np.var(y))
    return dict(result, status='ok', mae=mae, rmse=float(np.sqrt(mse)),
                mae_sd=mae/np.sqrt(variance) if variance > 0 else None,
                r2=1-mse/variance if variance > 0 else None,
                reason=None if variance > 0 else 'constant_target')


def cluster_bootstrap(frame: pd.DataFrame, statistic, *, cluster='split_group', n_bootstrap=2000, seed=20261028) -> dict:
    """Resample whole clusters, preserving each repeated draw's complete rows.

    ``statistic(sample)`` returns a dict of scalar estimates (None/nonfinite
    means a failed replicate for that metric). Pass only metrics to bootstrap,
    e.g. lambda s: {'slope': parent_offspring(s)['corrected_slope']}.
    Intervals require >=90% success separately for each metric. ``n_failed``
    counts replicates with any failed metric; per-metric counts are also exposed.
    Exceptions from a replicate count as failures and retain their type/message.
    The private _bootstrap_draw column distinguishes independent copies of a
    source cluster; source cluster labels stay unchanged. One cluster provides
    no independent resampling support and is unavailable.
    """
    if cluster not in frame or frame[cluster].isna().any() or frame[cluster].astype(str).str.strip().eq('').any():
        raise ValueError(f'{cluster} must contain a nonmissing cluster for every row')
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError('n_bootstrap must be a positive integer')
    groups = [group for _, group in frame.groupby(cluster, sort=True)]
    output = dict(status='unavailable', reason=None, n_clusters=len(groups), n_bootstrap=n_bootstrap,
                  n_successful=0, n_failed=0, failure_rate=None, intervals={}, metric_successes={}, failures={})
    if len(groups) < 2:
        return dict(output, reason='insufficient_clusters')
    point_frame = frame.assign(_bootstrap_draw=pd.factorize(frame[cluster], sort=True)[0])
    point = statistic(point_frame)
    if not isinstance(point, dict) or not point:
        raise ValueError('statistic must return a nonempty dict of scalar estimates')
    values = {key: [] for key in point}
    failures = Counter()
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(n_bootstrap):
        sample = pd.concat([groups[i].assign(_bootstrap_draw=draw) for draw, i in enumerate(rng.integers(0, len(groups), len(groups)))], ignore_index=True)
        try:
            estimate = statistic(sample)
        except Exception as exc:
            failures[f'{type(exc).__name__}: {exc}'] += 1
            continue
        complete = True
        for key in values:
            value = estimate.get(key)
            if value is not None and np.isscalar(value) and not isinstance(value, (str, bytes)) and np.isfinite(value):
                values[key].append(float(value))
            else:
                complete = False
        if complete:
            successes += 1
        else:
            failures['unavailable_metric'] += 1
    intervals = {key: [float(v) for v in np.percentile(sample, [2.5, 97.5])] if len(sample)/n_bootstrap >= .9 else None for key, sample in values.items()}
    return dict(output, status='ok', reason=None, n_successful=successes,
                n_failed=n_bootstrap-successes, failure_rate=1-successes/n_bootstrap,
                intervals=intervals, metric_successes={key: len(sample) for key, sample in values.items()},
                failures=dict(sorted(failures.items())))
