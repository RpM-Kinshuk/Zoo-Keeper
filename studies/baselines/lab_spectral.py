"""Bounded native-module adaptations of two published similarity baselines.

Consumes existing full-SVD spectra; never opens weights or changes truth/splits.
PYTHONPATH must include the existing Model-Research checkout for Trace's helper.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


METHODS = ('Trace-native', 'Spectral-Signatures-native')


def _digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hill_alpha(eigs):
    values = np.sort(np.asarray(eigs, dtype=np.float64))
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid full spectrum')
    k = len(values) // 2
    if values[-k-1] <= 0:
        raise ValueError('Invalid zero Hill threshold')
    denominator = np.log(values[-k:] / values[-k-1]).sum()
    if denominator <= 0 or not np.isfinite(denominator):
        raise ValueError('Flat or invalid Hill tail')
    return float(1 + k / denominator)


def similarity(left, right, method):
    if method not in METHODS:
        raise ValueError('Unknown method')
    if not left or set(left) != set(right):
        raise ValueError('Incompatible native module sets')
    roles = sorted(left)
    a, b = (np.stack([profile[key] for key in roles], axis=1) for profile in (left, right))
    if min(len(a), len(b)) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid profiles')
    if method == 'Trace-native':
        from spectral_research.study.reassessment_lineage import _unit_profiles
        depth = max(len(a), len(b))
        a, b = (_unit_profiles(value[None], depth)[0] for value in (a, b))
        return float(np.clip(np.sum(a * b) / len(roles), -1, 1))
    if len(a) != len(b):
        raise ValueError('Unequal depth; no layer alignment fitted')
    if (np.ptp(a, axis=0) == 0).any() or (np.ptp(b, axis=0) == 0).any():
        raise ValueError('Constant role; Spearman undefined')
    return float(np.mean([spearmanr(a[:, i], b[:, i]).statistic for i in range(len(roles))]))


def score_pairs(pairs, profiles, method):
    result = pairs.copy()
    scores, statuses, reasons = [], [], []
    for row in pairs.itertuples(index=False):
        try:
            if not row.eligible:
                raise ValueError('Frozen pair ineligible')
            score = similarity(profiles[row.query_key], profiles[row.candidate_key], method)
            scores.append(score); statuses.append('tested'); reasons.append('')
        except (KeyError, ValueError) as exc:
            scores.append(np.nan); statuses.append('not_run'); reasons.append(str(exc))
    return result.assign(method=method, score=scores, status=statuses, reason=reasons)


def _profiles(path, cost=None):
    """Return native role/depth profiles and exact logical eig-array bytes read."""
    energy, hill, skipped = {}, {}, []
    logical_bytes = 0
    if cost is not None:
        cost['logical_eigenvalue_bytes'] = 0
    with h5py.File(path, 'r') as handle:
        cfg = json.loads(handle.attrs['measurement_config_json'])
        if not cfg.get('use_svd') or cfg.get('spectrum_storage') != 'full':
            raise ValueError('Requires saved full-SVD spectrum')
        names = handle['layers/longname'].asstr()[:]
        hill_error = None
        for i, name in enumerate(names):
            match = re.search(r'(?:layers|h|block)\.(\d+)\.(.+)', name)
            if match is None:
                skipped.append({'name': name, 'reason': 'not_transformer_layer_matrix'})
                continue
            depth, role = int(match[1]), match[2]
            eigs = np.asarray(handle['eigs'][i])
            logical_bytes += eigs.nbytes
            if cost is not None:
                cost['logical_eigenvalue_bytes'] = logical_bytes
            if not np.isfinite(eigs).all() or (eigs < 0).any():
                raise ValueError('Invalid raw eigenvalues')
            if depth in energy.setdefault(role, {}):
                raise ValueError('Duplicate native role/depth')
            energy[role][depth] = float(np.sqrt(eigs.sum(dtype=np.float64)))
            try:
                hill.setdefault(role, {})[depth] = hill_alpha(eigs)
            except ValueError as exc:
                hill_error = str(exc)
        profiles = []
        for index, values in enumerate((energy, hill)):
            if index == 1 and hill_error:
                profiles.append(None)
                continue
            if not values:
                raise ValueError('No native matrix profiles')
            depths = sorted(next(iter(values.values())))
            if depths != list(range(len(depths))) or any(sorted(row) != depths for row in values.values()):
                raise ValueError('Missing or noncontiguous native depths')
            profiles.append({role: np.array([row[d] for d in depths]) for role, row in values.items()})
    return profiles, logical_bytes, skipped, hill_error


def _write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def run(data_root, model_index, control_root, output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True)
    from spectral_research.study import reassessment_lineage
    bindings = {'code': {'adapter_sha256': _digest(Path(__file__)),
                         'trace_helper_sha256': _digest(Path(reassessment_lineage.__file__))}}
    def bind(path):
        relative = path.absolute().relative_to(data_root.absolute()).as_posix()
        if relative not in bindings:
            bindings[relative] = {'sha256': _digest(path), 'hash_scan_bytes': path.stat().st_size}
        return relative
    bind(model_index)
    models = pd.read_parquet(model_index).set_index('model_key', verify_integrity=True)
    universes = {}
    for split in ('development', 'test'):
        path = control_root / split / 'universe.parquet'
        bind(path)
        universes[split] = pd.read_parquet(path)
    training = control_root.parent.parent / 'training-12' / 'universe.parquet'
    if training.is_file():
        bind(training)
        universes['training'] = pd.read_parquet(training)
    keys = sorted(set().union(*(set(u.query_key) | set(u.candidate_key) for u in universes.values())))
    profiles = {method: {} for method in METHODS}
    audit = []
    for key in keys:
        row = {'model_key': key}
        try:
            source = models.loc[key, 'full_weight_spectra']
            path = data_root / source
            if Path(source).is_absolute() or not path.resolve().is_relative_to(data_root.resolve()):
                raise ValueError('Spectrum path escapes data root')
            row['spectra'] = bind(path)
            values, payload, skipped, hill_error = _profiles(path, row)
            row.update(logical_eigenvalue_bytes=payload, skipped=skipped, hill_error=hill_error)
            for method, value in zip(METHODS, values):
                if value is not None:
                    profiles[method][key] = value
            row['status'] = 'read'
        except (KeyError, OSError, ValueError, TypeError) as exc:
            row.update(status='not_run', reason=str(exc))
        audit.append(row)
    # Freeze all data before labels are scored. Source paths remain relative.
    _write(output / 'inputs.json', json.dumps(bindings, indent=2, sort_keys=True) + '\n')
    _write(output / 'profile-audit.json', json.dumps(audit, indent=2, sort_keys=True) + '\n')
    metrics = []
    for split, pairs in universes.items():
        for method in METHODS:
            scored = score_pairs(pairs, profiles[method], method)
            _write(output / f'{split}-{method}.csv', scored.to_csv(index=False))
            valid = scored.loc[scored.status.eq('tested')]
            target = valid.truth_same_root.astype(bool)
            auroc = float(roc_auc_score(target, valid.score)) if target.nunique() == 2 else None
            metrics.append(dict(split=split, method=method, pairs=len(pairs), answered_pairs=len(valid),
                                same_family_auroc=auroc, immediate_parent='not applicable',
                                direction='not applicable', edge_type='not applicable'))
    report = dict(metrics=metrics, models=audit, new_weight_bytes=0,
                  logical_eigenvalue_bytes=sum(r.get('logical_eigenvalue_bytes', 0) for r in audit),
                  historical_extraction_bytes=None, historical_extraction_seconds=None,
                  hash_scan_bytes=sum(r.get('hash_scan_bytes', 0) for r in bindings.values()),
                  elapsed_seconds=time.monotonic()-start,
                  scope='native-module adaptations; cached full-SVD spectra; no thresholds fitted')
    _write(output / 'report.json', json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for arg in ('data-root', 'model-index', 'control-root', 'out'):
        parser.add_argument('--' + arg, type=Path, required=True)
    args = parser.parse_args()
    report = run(args.data_root, args.model_index, args.control_root, args.out)
    print(json.dumps({key: value for key, value in report.items() if key != 'models'}, indent=2))
