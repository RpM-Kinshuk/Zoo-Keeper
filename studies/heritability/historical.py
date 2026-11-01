"""Audit existing predictions without fitting, rescaling, or inventing provenance.

``columns`` maps canonical field names to CSV headers (for example ``fold`` to
``outer_fold`` and ``method`` to ``feature_block``). Other mapping keys provide
explicit constants, including prediction_scale/semantics and deployment fields.
Only ``raw_accuracy`` with binary_accuracy, raw_accuracy, or documented_accuracy
semantics is accepted. Normalized/unknown targets live in the audit's separate
``descriptive_predictions`` records, never the returned Raw prediction table.

A manifest uses ``artifacts: {relative_path: {sha256: digest}}``. It must cover
predictions, protocol and split tables. Protocol ``inputs`` may be an inventory
``{files: [{path, sha256}]}``, a list of those records, or a path-to-hash mapping.
Hashes establish artifact consistency, not authorship of an old manifest.
"""

from collections import Counter
import json
import math
from pathlib import Path
import re

import pandas as pd

from .data import EXCLUSION_COLUMNS, _model_key, _sha256, _text


DEPLOYMENT = ['model_key', 'benchmark', 'task', 'harness', 'precision', 'quantization']
PREDICTION_COLUMNS = DEPLOYMENT + ['method', 'setting', 'fold', 'actual', 'prediction',
    'prediction_scale', 'prediction_semantics', 'split_group', 'revision_verified',
    'provenance_status', 'source_hash']
DISCOVERY_COLUMNS = ['root', 'path', 'size_bytes', 'sha256', 'status']
_GROUPS = ['split_group', 'root_group', 'duplicate_group', 'model_key']
_SKIPPED = {'__pycache__', 'node_modules', 'venv', 'env', 'cache', 'caches'}


def _relevant(path):
    return path.suffix.lower() in {'.csv', '.json'} and any(
        word in path.stem.lower() for word in
        ('prediction', 'manifest', 'config', 'protocol', 'split', 'target', 'input', 'score', 'leaderboard'))


def _safe_path(root, relative):
    """Refuse traversal, environment files and symlinks, including parent links."""
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('artifact paths must be relative and stay inside the run')
    path = Path(root)
    for part in path.parts:
        if part.startswith('.env'):
            raise ValueError('environment paths are forbidden')
    # Root ancestors also matter when an explicitly supplied root is a symlink.
    if any(part.is_symlink() for part in [path, *path.parents]):
        raise ValueError('symlink paths are forbidden')
    for part in relative.parts:
        if part.startswith('.env'):
            raise ValueError('environment paths are forbidden')
        path = path / part
        if path.is_symlink():
            raise ValueError('symlink paths are forbidden')
    return path


def discover_runs(search_roots: list[Path]) -> pd.DataFrame:
    """Search only supplied roots, at most eight directory levels/20,000 entries.

    Root labels are positional, so all stored paths are portable. Only relevant
    CSV/JSON candidates are hashed; no weights or source scripts are opened.
    """
    records = []
    for index, supplied in enumerate(search_roots):
        root = Path(supplied)
        label = f'root_{index}'
        def record(path, status, size=None, digest=None):
            records.append(dict(root=label, path=path, size_bytes=size, sha256=digest, status=status))
        try:
            _safe_path(root, '.')
        except ValueError:
            record('.', 'unsafe_path')
            continue
        if not root.exists():
            record('.', 'not_found')
            continue
        pending = [(root, 0)]
        visited, found = 0, False
        while pending:
            directory, depth = pending.pop()
            try:
                entries = [directory] if directory.is_file() else sorted(directory.iterdir(), reverse=True)
            except OSError:
                record(directory.relative_to(root).as_posix(), 'unreadable')
                continue
            for path in entries:
                visited += 1
                if visited > 20000:
                    break
                if path.name.startswith('.') or path.name in _SKIPPED or path.is_symlink():
                    continue
                if path.is_dir():
                    if depth < 8:
                        pending.append((path, depth + 1))
                    else:
                        record(path.relative_to(root).as_posix(), 'depth_limit')
                    continue
                if not _relevant(path):
                    continue
                found = True
                relative = path.relative_to(root).as_posix() if path != root else path.name
                try:
                    size = path.stat().st_size
                    record(relative, 'ok' if size else 'missing_zero_bytes', size, _sha256(path))
                except OSError:
                    record(relative, 'unreadable')
            if visited > 20000:
                record('.', 'entry_limit')
                break
        if not found and not any(row['root'] == label for row in records):
            record('.', 'no_candidates')
    return pd.DataFrame(records, columns=DISCOVERY_COLUMNS).sort_values(['root', 'path', 'status']).reset_index(drop=True)


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _canonical_key(key, repo='', revision=''):
    key = _text(key)
    if key:
        if '@' not in key:
            return ''
        repo, revision = key.rsplit('@', 1)
    return _model_key(_text(repo).casefold(), _text(revision).lower())


def _is_verified(value):
    return value is True or _text(value).lower() == 'true'


def _input_records(inputs):
    if isinstance(inputs, list):
        return inputs
    if isinstance(inputs, dict):
        if 'files' in inputs:
            return _input_records(inputs['files'])
        return [dict(value, path=value.get('path', key)) if isinstance(value, dict)
                else dict(path=key, sha256=value) for key, value in inputs.items()]
    return []


def audit_predictions(run: Path, scores: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return Raw rows (eligible or labeled unverified), exclusions and coverage.

    Primary consumers must select provenance_status == 'eligible'. A verified
    join requires a unique, independently verified score at the exact checkpoint,
    benchmark, task, harness, precision and quantization, plus matching actual.
    Baseline support compares full deployment/task/setting/fold/group row
    multisets using only rows that passed the other provenance checks.
    """
    run = Path(run)
    exclusions, descriptive, records = [], [], []
    input_count, verified_joins = 0, 0
    artifacts = {}
    global_reasons = set()

    def exclude(label, reason, detail):
        exclusions.append(dict(stage='historical', record_id=str(label), reason=reason, detail=str(detail)))

    def load_path(name):
        try:
            path = _safe_path(run, name)
        except (ValueError, TypeError):
            exclude('run', 'unsafe_artifact_path', 'artifact must be relative, nonsecret and nonsymlinked')
            return None
        if not path.is_file():
            exclude(str(name), 'not_found', str(name))
            return None
        if not path.stat().st_size:
            exclude(str(name), 'missing_zero_bytes', str(name))
            return None
        if path.suffix.lower() not in {'.csv', '.json'}:
            exclude(str(name), 'unsupported_artifact', 'only CSV/JSON audit inputs are supported')
            return None
        artifacts[str(name)] = _sha256(path)
        return path

    def finish():
        table = pd.DataFrame(records, columns=PREDICTION_COLUMNS)
        if len(table):
            table = table.sort_values(DEPLOYMENT + ['method', 'fold']).reset_index(drop=True)
        table['revision_verified'] = table['revision_verified'].astype(bool)
        excluded = pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS).drop_duplicates().sort_values(EXCLUSION_COLUMNS).reset_index(drop=True)
        eligible = int(table.provenance_status.eq('eligible').sum())
        denominator = input_count
        audit = dict(input_predictions=input_count, retained_raw_predictions=len(table), eligible_predictions=eligible,
            raw_status='available' if eligible else 'unavailable', protocol_verified=protocol_verified,
            artifacts=dict(sorted(artifacts.items())), descriptive_predictions=descriptive,
            prediction_to_score_join=dict(numerator=verified_joins, denominator=denominator,
                fraction=verified_joins / denominator if denominator else None,
                denominator_definition='all input prediction rows, including normalized, unpinned and missing deployment evidence'),
            out_of_range_predictions=sum(not 0 <= row['prediction'] <= 1 for row in records),
            reasons=dict(sorted(Counter(excluded.reason).items())))
        return table, excluded, audit

    protocol_verified = False
    prediction_name = mapping.get('prediction_file', 'predictions.csv')
    prediction_path = load_path(prediction_name)
    if prediction_path is None:
        return finish()
    try:
        frame = pd.read_csv(prediction_path, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
        exclude(prediction_name, 'invalid_prediction_table', prediction_name)
        return finish()
    input_count = len(frame)

    # Read the original freeze, then validate its hashes; never execute a config.
    protocol = {}
    manifest = {}
    for field, destination in [('protocol_file', protocol), ('manifest_file', manifest)]:
        name = mapping.get(field)
        if not name:
            global_reasons.add('protocol_unverifiable')
            exclude('run', 'protocol_unverifiable', f'{field} missing')
            continue
        path = load_path(name)
        if path is None:
            global_reasons.add('protocol_unverifiable')
            continue
        try:
            content = json.loads(path.read_text())
            if not isinstance(content, dict):
                raise ValueError('not an object')
            destination.update(content)
        except (ValueError, UnicodeError):
            global_reasons.add('protocol_unverifiable')
            exclude(name, 'protocol_unverifiable', 'invalid JSON object')

    proofs = manifest.get('artifacts', {})
    if not isinstance(proofs, dict):
        proofs = {}
    required = [prediction_name] + [mapping[key] for key in ('protocol_file', 'split_file') if mapping.get(key)]
    for name in sorted(set(required) | {str(key) for key in proofs if Path(str(key)).suffix.lower() in {'.csv', '.json'}}):
        path = load_path(name)
        proof = proofs.get(name, {})
        expected = proof.get('sha256') if isinstance(proof, dict) else proof
        if not expected or not re.fullmatch('[0-9a-fA-F]{64}', str(expected)):
            reason = 'artifact_hash_unavailable'
        elif path is None or artifacts.get(str(name)) != str(expected).lower():
            reason = 'artifact_hash_mismatch'
        else:
            continue
        global_reasons.add(reason)
        exclude(name, reason, 'frozen artifact hash is absent or does not match')
    for proof in _input_records(protocol.get('inputs', [])):
        if not isinstance(proof, dict) or not proof.get('path'):
            global_reasons.add('artifact_hash_unavailable')
            exclude('inputs', 'artifact_hash_unavailable', 'input requires path and sha256')
            continue
        name = proof['path']
        path = load_path(name)
        expected = proof.get('sha256')
        reason = 'artifact_hash_unavailable' if not expected else 'artifact_hash_mismatch'
        if path is None or artifacts.get(str(name)) != expected:
            global_reasons.add(reason)
            exclude(name, reason, 'protocol input hash is absent or does not match')
    protocol_verified = not global_reasons

    splits = pd.DataFrame()
    if mapping.get('split_file'):
        path = load_path(mapping['split_file'])
        if path is not None:
            try:
                splits = pd.read_csv(path, dtype=str, keep_default_na=False)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError):
                pass
    split_columns = mapping.get('split_columns', {})
    splits = splits.rename(columns={source: canonical for canonical, source in split_columns.items()})
    membership = {}
    if not {'fold', 'partition', 'split_group'}.issubset(splits.columns):
        global_reasons.add('split_unverifiable')
        exclude('run', 'split_unverifiable', 'explicit fold/train/test group membership is required')
    else:
        for fold, block in splits.groupby('fold', sort=True):
            train = block[block.partition.eq('train')]
            test = block[block.partition.eq('test')]
            reasons = set()
            if train.empty or test.empty or train.split_group.eq('').any() or test.split_group.eq('').any():
                reasons.add('split_unverifiable')
            for group in _GROUPS:
                if group in block:
                    normalize = _canonical_key if group == 'model_key' else _text
                    train_groups = {normalize(value) for value in train[group]} - {''}
                    test_groups = {normalize(value) for value in test[group]} - {''}
                    if train_groups & test_groups:
                        reasons.add('historical_leakage')
            membership[fold] = (test, reasons)

    score_lookup = {}
    for score in scores.to_dict('records'):
        identity = tuple(_canonical_key(score.get('model_key', ''), score.get('repo_id', ''), score.get('revision', ''))
                         if key == 'model_key' else _text(score.get(key, '')) for key in DEPLOYMENT)
        score_lookup.setdefault(identity, []).append(score)
    columns = mapping.get('columns', {})
    for index, source in enumerate(frame.to_dict('records'), 1):
        label = f'{prediction_name}:{index}'
        row = {key: _text(source.get(columns.get(key, key), mapping.get(key, '')))
               for key in PREDICTION_COLUMNS + ['repo_id', 'revision', 'root_group', 'duplicate_group']}
        row['source_hash'] = artifacts[str(prediction_name)]
        reasons = set(global_reasons)
        scale, semantics = row['prediction_scale'], row['prediction_semantics']
        if scale != 'raw_accuracy' or semantics not in {'binary_accuracy', 'raw_accuracy', 'documented_accuracy'}:
            reason = 'normalized_target_not_raw' if 'normaliz' in scale.lower() or 'normaliz' in semantics.lower() else 'unsupported_prediction_semantics'
            exclude(label, reason, 'no inverse normalization or target replacement is permitted')
            row.update(provenance_status='historical_unverified', setting='historical_unverified', revision_verified=False)
            descriptive.append(row)
            continue
        actual, prediction = _number(row['actual']), _number(row['prediction'])
        if actual is None or not 0 <= actual <= 1 or prediction is None:
            exclude(label, 'invalid_observation' if actual is None or not 0 <= actual <= 1 else 'invalid_prediction', 'finite Raw observations in [0, 1] and finite predictions required')
            continue
        row.update(actual=actual, prediction=prediction)
        canonical = _canonical_key(row['model_key'], row['repo_id'], row['revision'])
        if not row['model_key']:
            row['model_key'] = _model_key(row['repo_id'], row['revision'])
        identity = (canonical,) + tuple(row[key] for key in DEPLOYMENT[1:])
        matches = score_lookup.get(identity, []) if all(identity) else []
        verified = False
        if len(matches) != 1:
            reasons.add('score_join_ambiguous' if len(matches) > 1 else 'score_join_unavailable')
        else:
            observed = _number(matches[0].get('accuracy'))
            if observed is None or not math.isclose(actual, observed, rel_tol=0, abs_tol=1e-12):
                reasons.add('actual_score_mismatch')
            elif not _is_verified(matches[0].get('revision_verified', False)):
                reasons.add('deployment_unverified')
            else:
                verified = True
                verified_joins += 1
        row['revision_verified'] = verified
        if not canonical:
            reasons.add('unpinned_model')
        test, split_reasons = membership.get(row['fold'], (pd.DataFrame(), {'split_unverifiable'}))
        reasons.update(split_reasons)
        if not row['split_group'] or test.empty or row['split_group'] not in set(test.get('split_group', [])):
            reasons.add('split_unverifiable')
        for group in ['root_group', 'duplicate_group', 'model_key']:
            if group in test and row.get(group):
                matched = test[test.split_group.eq(row['split_group'])]
                normalize = _canonical_key if group == 'model_key' else _text
                if matched.empty or normalize(row[group]) not in {normalize(value) for value in matched[group]}:
                    reasons.add('split_unverifiable')
        if row['setting'] not in {'root_holdout', 'within_family'}:
            reasons.add('split_unverifiable')
        if not row['method'] or not row['fold']:
            reasons.add('missing_prediction_identity')
        row['_reasons'], row['_label'] = reasons, label
        records.append(row)

    baseline = mapping.get('baseline_method')
    support_columns = DEPLOYMENT + ['setting', 'fold', 'split_group', 'actual']
    supports = {row['method']: Counter() for row in records}
    for row in records:
        # A row with failed provenance cannot supply a baseline observation.
        if not row['_reasons']:
            identity = tuple(_canonical_key(row[key]) if key == 'model_key' else row[key] for key in support_columns)
            supports[row['method']][identity] += 1
    matched_methods = set()
    if baseline in supports and supports[baseline]:
        matched_methods = {method for method, support in supports.items()
                           if method != baseline and support == supports[baseline]}
    # A third method with incomplete support does not invalidate an otherwise
    # complete baseline/comparator pair. Every eligible method still has a pair.
    paired_methods = matched_methods | {baseline} if matched_methods else set()
    bad_methods = set(supports) - paired_methods
    for row in records:
        if row['method'] in bad_methods:
            row['_reasons'].add('baseline_support_mismatch')
        for reason in sorted(row['_reasons']):
            exclude(row['_label'], reason, 'descriptive Raw row retained; excluded from primary historical comparisons')
        row['provenance_status'] = 'historical_unverified' if row['_reasons'] else 'eligible'
        if row['_reasons']:
            row['setting'] = 'historical_unverified'
    return finish()
