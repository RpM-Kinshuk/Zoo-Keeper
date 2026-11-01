"""Offline assembly, immutable freezes and reproducible study analysis.

Source descriptors and stored artifacts use paths relative to a documented source
root or run. Runtime source roots are never serialized. No model code is loaded.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from . import ancestry, data, historical, report, statistics
from .protocol import canonical_json, freeze_protocol


def _json(path):
    return json.loads(data._safe_file(path).read_text())


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_json(path, value):
    report.atomic_write(path, canonical_json(_clean(value)))


def _write_table(path, frame):
    report.atomic_write(path, frame.to_csv(index=False, lineterminator='\n', float_format='%.17g'))


def _path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or any(part.startswith('.env') for part in relative.parts):
        raise ValueError('input paths must be relative, nonsecret and stay within their root')
    root = Path(root)
    path = root/relative
    if any(part.is_symlink() for part in [path, *path.parents]):
        raise ValueError('symlink input paths are unsupported')
    return path


def _source(config_path, relative, source_root=None):
    """Resolve explicit portable source names; never traverse or search for files."""
    config_path = Path(config_path)
    base = config_path.parent.parent if config_path.parent.name == 'configs' else config_path.parent
    roots = [base]
    if source_root is not None:
        roots.append(Path(source_root))
    for root in roots:
        candidate = _path(root, relative)
        if candidate.exists():
            return candidate
    return _path(base, relative)


def _copy(source, destination):
    content = Path(source).read_bytes()
    destination = Path(destination)
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError('immutable snapshot differs: '+destination.name)
    else:
        report.atomic_write(destination, content)


def _inventory(root, paths):
    return data.inventory_inputs([_path(root, name) for name in sorted(set(paths))], data_root=Path(root))


def _verify(root, inventory):
    for entry in inventory['files']:
        path = _path(root, entry['path'])
        if entry['status'] != 'ok':
            if path.exists():
                raise ValueError('previously unavailable input changed: '+entry['path'])
            continue
        if not path.is_file() or data._sha256(path) != entry['sha256'] or path.stat().st_size != entry['size_bytes']:
            raise ValueError('input hash mismatch: '+entry['path'])


def _read_table(path):
    return pd.read_csv(data._safe_file(path), keep_default_na=False, dtype={name:str for name in ['revision','model_key','repo_id','model_id','source_model','fold']})


def _verified(frame):
    return frame.revision_verified.astype(str).str.lower().eq('true')


def _registry_tasks(registry):
    tasks = {}
    for schema in registry['schemas'].values():
        for task in schema['tasks']:
            key = (task['benchmark'], task['task'], task.get('harness', ''))
            if key in tasks and tasks[key].get('kind') != task.get('kind'):
                raise ValueError('conflicting task kinds in registry')
            tasks[key] = task
    return tasks


def _enrich(scores, registry, pins):
    scores = scores.copy()
    tasks = _registry_tasks(registry)
    scores['kind'] = [tasks.get((row.benchmark, row.task, row.harness), {}).get('kind', '') for row in scores.itertuples()]
    for name in ['item_variance', 'item_variance_source']:
        scores[name] = [tasks.get((row.benchmark, row.task, row.harness), {}).get(name, '') for row in scores.itertuples()]
    # Fixed effects require independent, documented counts for every retained row.
    count_map = {}
    for pin in pins.to_dict('records'):
        if pin.get('parameter_count_source') and pin.get('parameter_count'):
            key = data._model_key(str(pin.get('repo_id', '')), str(pin.get('revision', '')))
            count_map.setdefault(key, set()).add((float(pin['parameter_count']), str(pin['parameter_count_source'])))
    scores['parameter_count'] = [next(iter(count_map[key]))[0] if len(count_map.get(key, set())) == 1 else np.nan for key in scores.model_key]
    scores['parameter_count_source'] = [next(iter(count_map[key]))[1] if len(count_map.get(key, set())) == 1 else '' for key in scores.model_key]
    return scores


def _raw_edges(atlas, metadata):
    records = [dict(parent_id=row.get('source', row.get('parent_id', '')), child_id=row.get('target', row.get('child_id', '')), relation=row.get('relation', 'unknown')) for row in atlas.get('edges', [])]
    records += [dict(parent_id=row.get('source_model', ''), child_id=row.get('model_id', ''), relation=row.get('base_model_relation', 'unknown')) for row in metadata.to_dict('records')]
    frame = pd.DataFrame(records, columns=['parent_id', 'child_id', 'relation'])
    frame['evidence_tier'] = 'declared'
    return frame.sort_values(['parent_id', 'child_id']).reset_index(drop=True)


def _assemble(config_path, out, source_root=None):
    config = _json(config_path)
    out = Path(out)
    registry_source = _source(config_path, config['task_registry'], source_root)
    registry = _json(registry_source)
    names = [source['path'] for source in config['sources']]+[config['task_registry']]
    names += [config[key] for key in ['graph','metadata','pinned_models','duplicates','source_manifest'] if config.get(key)]
    if config.get('historical_provenance', {}).get('config'):
        names.append(config['historical_provenance']['config'])
    for schema in registry['schemas'].values():
        for task in schema['tasks']:
            for field in ['denominator_evidence', 'variance_evidence']:
                if isinstance(task.get(field), dict) and task[field].get('path'):
                    names.append(task[field]['path'])
    source_rows = []
    manifest = _json(_source(config_path, config['source_manifest'], source_root)) if config.get('source_manifest') else {'files': []}
    expected = {entry['path']:entry for entry in manifest['files']}
    for name in sorted(set(names)):
        original = _source(config_path, name, source_root)
        destination = _path(out/'source_inputs', name)
        if original.is_file() and original.stat().st_size:
            digest = data._sha256(original)
            if name in expected and expected[name]['sha256'] != digest:
                raise ValueError('source manifest hash mismatch: '+name)
            _copy(original, destination)
        source_rows.append('source_inputs/'+name)
    _write_json(out/'source_inventory.json', _inventory(out, source_rows))
    registry = dict(registry, evidence_root=out/'source_inputs')
    pins = _read_table(_path(out/'source_inputs', config['pinned_models'])) if config.get('pinned_models') else pd.DataFrame()
    sources = [dict(source, path=_path(out/'source_inputs', source['path'])) for source in config['sources']]
    scores, exclusions, coverage = data.assemble_scores(sources, pins, registry)
    scores = _enrich(scores, registry, pins)
    atlas_path = _path(out/'source_inputs', config['graph']) if config.get('graph') else None
    atlas = _json(atlas_path) if atlas_path and atlas_path.is_file() else {'nodes': [], 'edges': []}
    metadata_path = _path(out/'source_inputs', config['metadata']) if config.get('metadata') else None
    metadata = _read_table(metadata_path) if metadata_path and metadata_path.is_file() else pd.DataFrame(columns=['model_id','source_model','base_model_relation'])
    raw_edges = _raw_edges(atlas, metadata)
    _write_table(out/'raw_declared_edges.csv', raw_edges)
    self_parent = metadata.source_model.fillna('').astype(str).str.casefold().eq(metadata.model_id.fillna('').astype(str).str.casefold()) & metadata.model_id.fillna('').ne('')
    quarantine = pd.DataFrame([dict(stage='ancestry', record_id=str(row.model_id), reason='self_parent_declaration', detail='invalid metadata declaration retained in raw metadata snapshot') for row in metadata.loc[self_parent].itertuples()], columns=data.EXCLUSION_COLUMNS)
    duplicates = _read_table(_path(out/'source_inputs', config['duplicates'])) if config.get('duplicates') else pd.DataFrame()
    try:
        if not atlas_path or not atlas_path.is_file():
            raise ValueError('ancestry graph unavailable')
        edges, edge_exclusions = ancestry.assemble_edges(atlas, metadata.loc[~self_parent], pins)
        groups = ancestry.build_groups(scores, edges, duplicates)
        coverage['ancestry_status'] = 'available'
    except ValueError as exc:
        reason = 'ancestry_cycle' if 'cycle' in str(exc).lower() else 'graph_unavailable'
        edge_exclusions = pd.DataFrame([dict(stage='ancestry', record_id='graph', reason=reason, detail='declared graph rejected; no independent roots inferred')])
        edges = raw_edges.reindex(columns=ancestry.EDGE_COLUMNS, fill_value='')
        groups = pd.DataFrame([dict(model_key=key, duplicate_group='', root_group='', split_group='unresolved_ancestry', group_evidence=reason) for key in sorted(set(scores.model_key)-{''})], columns=ancestry.GROUP_COLUMNS)
        coverage['ancestry_status'] = reason
    primary, sensitivity, deployment_exclusions = ancestry.select_primary_deployments(scores, config['primary_deployment'])
    exclusions = pd.concat([exclusions, quarantine, edge_exclusions, deployment_exclusions], ignore_index=True).reindex(columns=data.EXCLUSION_COLUMNS).fillna('').sort_values(data.EXCLUSION_COLUMNS).reset_index(drop=True)
    coverage.update(primary_observations=len(primary), sensitivity_observations=len(sensitivity), self_parent_declarations=int(self_parent.sum()))
    files = {'scores.csv':scores, 'edges.csv':edges, 'groups.csv':groups, 'primary_scores.csv':primary, 'sensitivity_scores.csv':sensitivity, 'exclusions.csv':exclusions}
    for name, frame in files.items():
        _write_table(out/name, frame)
    _write_json(out/'coverage.json', coverage)
    _write_json(out/'config.json', config)
    names = list(files)+['raw_declared_edges.csv','coverage.json','config.json','source_inventory.json']+source_rows
    _write_json(out/'assembly_manifest.json', _inventory(out, names))
    return 0


def _historical(run_root, assembled, out):
    assembled, out, run_root = Path(assembled), Path(out), Path(run_root)
    _verify(assembled, _json(assembled/'assembly_manifest.json'))
    config = _json(assembled/'config.json')
    scores = _read_table(assembled/'scores.csv').merge(_read_table(assembled/'groups.csv'), on='model_key', how='left')
    discovery = historical.discover_runs([run_root])
    predictions, exclusions, audit = historical.audit_predictions(run_root, scores, config.get('historical_mapping', {}))
    _write_table(out/'discovery.csv', discovery)
    _write_table(out/'predictions.csv', predictions)
    _write_table(out/'exclusions.csv', exclusions)
    _write_json(out/'audit.json', audit)
    _write_json(out/'mapping.json', config.get('historical_mapping', {}))
    names = ['discovery.csv','predictions.csv','exclusions.csv','audit.json','mapping.json']
    for name in audit['artifacts']:
        original = _path(run_root, name)
        if data._sha256(original) != audit['artifacts'][name]:
            raise ValueError('historical artifact changed during audit')
        destination = _path(out/'source_inputs', name)
        _copy(original, destination)
        names.append('source_inputs/'+name)
    _write_json(out/'audit_manifest.json', _inventory(out, names))
    return 0


def _freeze(config_path, assembled, out):
    assembled, out = Path(assembled), Path(out)
    assembly_inventory = _json(assembled/'assembly_manifest.json')
    _verify(assembled, assembly_inventory)
    config = _json(config_path)
    if config != _json(assembled/'config.json'):
        raise ValueError('config differs from assembly; reassemble before freezing')
    # Every control actually used by Task 3 is recorded in the protocol.
    frozen_config = dict(config, effective_optimizer=dict(method='L-BFGS-B', ftol=1e-12, gtol=1e-7, maxiter=500, starts=[[.3,.7],[.7,.3]], boundary_xatol=1e-10, boundary_maxiter=500, equivalent_nll_tolerance=1e-8))
    copies = [(assembled, entry['path'], 'inputs/'+entry['path']) for entry in assembly_inventory['files']]
    if config.get('historical_audit'):
        audit_root = _source(config_path, config['historical_audit'])
        if (audit_root/'audit_manifest.json').is_file():
            inventory = _json(audit_root/'audit_manifest.json')
            _verify(audit_root, inventory)
            copies += [(audit_root, entry['path'], 'inputs/historical/'+entry['path']) for entry in inventory['files']]
            copies.append((audit_root, 'audit_manifest.json', 'inputs/historical/audit_manifest.json'))
    # Build expected inventory first, refusing existing differing snapshots.
    records = []
    for root, source_name, destination_name in copies:
        source = _path(root, source_name)
        status = 'ok' if source.is_file() and source.stat().st_size else 'missing'
        records.append(dict(path=destination_name, status=status, sha256=data._sha256(source) if status == 'ok' else None, size_bytes=source.stat().st_size if source.is_file() else None))
    inventory = {'files':sorted(records, key=lambda entry: entry['path'])}
    if (out/'protocol.json').exists():
        expected = canonical_json(dict(config=frozen_config, inputs=inventory))
        if (out/'protocol.json').read_bytes() != expected:
            raise ValueError('existing frozen protocol differs')
        _verify(out, inventory)
    for root, source_name, destination_name in copies:
        source = _path(root, source_name)
        if source.is_file() and source.stat().st_size:
            _copy(source, _path(out, destination_name))
    digest = freeze_protocol(frozen_config, inventory, out/'protocol.json')
    _write_json(out/'inputs.json', inventory)
    report.atomic_write(out/'protocol.sha256', digest+'\n')
    return 0


def _design(frame, use_parameters):
    if not use_parameters:
        return np.ones((len(frame), 1))
    log_count = np.log(pd.to_numeric(frame.parameter_count).to_numpy(float))
    return np.column_stack([np.ones(len(frame)), log_count-log_count.mean()])



def _canonical_model_key(value):
    repo, separator, revision = str(value).rpartition('@')
    return repo.casefold()+separator+revision.lower() if separator else str(value)


def _bootstrap_covariance(sample, kernel):
    """Each resampled split-group draw gets its own latent covariance block."""
    positions = sample['_position'].to_numpy(int)
    draws = sample['_bootstrap_draw'].to_numpy(int)
    return kernel[np.ix_(positions, positions)]*np.equal.outer(draws, draws)


def _exchange_coverage(exchange, registry):
    """Retain unavailable requested cells at benchmark/task/harness identity."""
    records = exchange.to_dict('records')
    covered = {(row.get('benchmark'),row.get('task'),row.get('harness')) for row in records}
    for benchmark,task,harness in _registry_tasks(registry):
        if (benchmark,task,harness) not in covered:
            records.append(dict(benchmark=benchmark,task=task,harness=harness,estimand='coverage',method='',status='unavailable',reason='historical_task_missing',k=None))
    return pd.DataFrame(records)


def _skill_errors(predictions, registry, *, unavailable_methods=None, missing_reason='audited_raw_support_unavailable'):
    """Raw errors on shared audited checkpoint support across tasks AND methods.

    Comparability strata keep benchmark, harness, precision, quantization,
    setting and fold separate. Only tasks with audited Raw observations enter
    the intersection; unavailable tasks get explicit rows and are named apart.
    Full per-task descriptive errors and support counts are retained separately.
    """
    strata = ['benchmark','harness','precision','quantization','setting','fold']
    fields = strata+['task','method','model_key','actual','prediction','revision_verified','provenance_status','prediction_scale']
    frame = predictions.reindex(columns=fields).copy()
    for name in strata+['task','method','model_key']:
        frame[name] = frame[name].fillna('').astype(str)
    for name in ['actual','prediction']:
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    frame['_key'] = frame.model_key.map(_canonical_model_key)
    eligible = frame.provenance_status.eq('eligible') & frame.prediction_scale.eq('raw_accuracy') & _verified(frame)
    eligible &= np.isfinite(frame.actual) & np.isfinite(frame.prediction) & frame.actual.between(0,1)
    audited = frame.loc[eligible].copy()
    audited = audited.loc[~audited.duplicated(strata+['task','method','_key'], keep=False)]
    records = []
    for identity, group in audited.groupby(strata, sort=True, dropna=False):
        base = dict(zip(strata, identity))
        methods = sorted(group.method.unique())
        tasks = sorted(group.task.unique())
        supports = [set(group.loc[group.task.eq(task) & group.method.eq(method), '_key']) for task in tasks for method in methods]
        common = set.intersection(*supports) if supports else set()
        full_stratum = frame.loc[np.logical_and.reduce([frame[name].eq(value) for name,value in base.items()])]
        for task in tasks:
            for method in methods:
                cell = group.loc[group.task.eq(task) & group.method.eq(method)]
                selected = cell.loc[cell._key.isin(common)].sort_values('_key')
                full = full_stratum.loc[full_stratum.task.eq(task) & full_stratum.method.eq(method)]
                descriptive = statistics.skill_summary(full.actual.to_numpy(float), full.prediction.to_numpy(float))
                summary = statistics.skill_summary(selected.actual.to_numpy(float), selected.prediction.to_numpy(float))
                if not len(selected):
                    summary['reason'] = 'common_audited_support_unavailable'
                records.append(dict(base, task=task, method=method, prediction_scale='raw_accuracy', analysis_tier='verified_protocol_eligible', common_tasks=';'.join(tasks), common_methods=';'.join(methods), n_observed=len(full), n_audited=len(cell), n_common=len(selected), excluded_from_common_support=len(full)-len(selected), descriptive_mae=descriptive['mae'], descriptive_rmse=descriptive['rmse'], **summary))
    covered = {(row['benchmark'],row['task'],row['harness'],row['method']) for row in records}
    # Ineligible Raw rows remain descriptive, with no scale-free comparison claim.
    for identity, cell in frame.groupby(strata+['task','method'], sort=True, dropna=False):
        base = dict(zip(strata+['task','method'], identity))
        key = tuple(base[name] for name in ['benchmark','task','harness','method'])
        exact = any(all(row.get(name) == value for name,value in base.items()) for row in records)
        if exact:
            continue
        descriptive = statistics.skill_summary(cell.actual.to_numpy(float), cell.prediction.to_numpy(float))
        records.append(dict(base, prediction_scale='raw_accuracy', analysis_tier='historical_descriptive', status='unavailable', reason=missing_reason, n=0, n_common=0, n_observed=len(cell), n_audited=0, excluded_from_common_support=len(cell), mae=None, rmse=None, mae_sd=None, r2=None, descriptive_mae=descriptive['mae'], descriptive_rmse=descriptive['rmse']))
        covered.add(key)
    methods = sorted(set(frame.method)-{''}) or sorted(set(unavailable_methods or [])) or ['']
    for benchmark,task,harness in _registry_tasks(registry):
        for method in methods:
            if (benchmark,task,harness,method) not in covered:
                records.append(dict(benchmark=benchmark,task=task,harness=harness,method=method,precision='',quantization='',setting='',fold='',prediction_scale='raw_accuracy',analysis_tier='unavailable',status='unavailable',reason=missing_reason,n=0,n_common=0,n_observed=0,n_audited=0,excluded_from_common_support=0,mae=None,rmse=None,mae_sd=None,r2=None))
    return pd.DataFrame(records).sort_values(strata+['task','method']).reset_index(drop=True)


def _pairs(frame, edges):
    records = []
    canonical_keys = frame.model_key.map(_canonical_model_key)
    counts = edges.assign(_parent=edges.parent_id.str.casefold()).groupby(edges.child_id.str.casefold())._parent.nunique()
    for edge in edges.itertuples():
        if counts.get(str(edge.child_id).casefold(), 0) != 1 or not edge.parent_key or not edge.child_key:
            continue
        parent, child = frame.loc[canonical_keys.eq(_canonical_model_key(edge.parent_key))], frame.loc[canonical_keys.eq(_canonical_model_key(edge.child_key))]
        if len(parent) != 1 or len(child) != 1:
            continue
        parent, child = parent.iloc[0], child.iloc[0]
        if parent.split_group != child.split_group:
            continue
        records.append(dict(parent_accuracy=parent.accuracy, child_accuracy=child.accuracy, parent_variance=parent.measurement_variance, error_covariance=np.nan, split_group=child.split_group))
    return pd.DataFrame(records, columns=['parent_accuracy','child_accuracy','parent_variance','error_covariance','split_group'])


def _fit_tables(scores, primary, groups, edges, registry, assembly_exclusions, config, coverage):
    primary = primary.merge(groups, on='model_key', how='left')
    tasks = _registry_tasks(registry)
    fits, parents, exclusions, diagnostics = [], [], [], []
    bootstrap_count, seed = config['bootstrap_replicates'], config['seed']
    for (benchmark, task, harness), metadata in sorted(tasks.items()):
        observed = primary.loc[primary.benchmark.eq(benchmark) & primary.task.eq(task) & primary.harness.eq(harness)]
        verified = observed.loc[_verified(observed)]
        for scale in ['raw', 'logit']:
            retained, scale_exclusions = statistics.prepare_scale(verified, scale)
            if len(scale_exclusions):
                scale_exclusions['detail'] = 'excluded from '+scale+' corrected analysis'
                exclusions.append(scale_exclusions[data.EXCLUSION_COLUMNS])
            retained = retained.loc[retained.root_group.fillna('').ne('')].reset_index(drop=True)
            counts = pd.to_numeric(retained.parameter_count, errors='coerce')
            use_parameters = bool(len(retained) and counts.notna().all() and counts.gt(0).all() and retained.parameter_count_source.fillna('').ne('').all())
            design_label = 'intercept_centered_log_parameters' if use_parameters else 'intercept_only_parameters_unavailable'
            base = dict(benchmark=benchmark, task=task, harness=harness, precision=config['primary_deployment']['precision'], quantization=config['primary_deployment']['quantization'], relation_tier=config.get('lineage_evidence_tier', 'declared'), scale=scale, design=design_label, n_observed=len(observed), n_verified=len(verified), excluded_support=len(observed)-len(retained))
            for model in ['family','network']:
                subset = retained.copy()
                unavailable = None
                if coverage['ancestry_status'] != 'available':
                    unavailable = coverage['ancestry_status']
                elif subset.empty:
                    unavailable = 'no_eligible_observations'
                if model == 'network' and len(subset):
                    unsupported = set(assembly_exclusions.loc[assembly_exclusions.reason.eq('merge_coefficients_unavailable'), 'record_id'].astype(str).str.casefold())
                    subset = subset.loc[~subset.repo_id.str.casefold().isin(unsupported)].reset_index(drop=True)
                    if subset.repo_id.str.casefold().duplicated().any():
                        unavailable = 'multiple_revisions_per_network_node'
                    if subset.empty:
                        unavailable = 'network_support_unavailable'
                if unavailable:
                    result = dict(status='unavailable', reason=unavailable, converged=False, n=len(subset), families=int(subset.split_group.nunique()) if len(subset) else 0, familial_fraction=None)
                else:
                    try:
                        if model == 'family':
                            labels = subset.split_group.to_numpy()
                            kernel = np.equal.outer(labels, labels).astype(float)
                        else:
                            kernel = ancestry.network_covariance(list(subset.repo_id), edges)
                        result = statistics.fit_reml(subset.accuracy.to_numpy(float), _design(subset, use_parameters), kernel, pd.to_numeric(subset.measurement_variance).to_numpy(float), covariance_model=model, families=int(subset.split_group.nunique()))
                    except (ValueError, np.linalg.LinAlgError):
                        result = dict(status='unavailable', reason='covariance_or_design_unsupported', converged=False, n=len(subset), families=int(subset.split_group.nunique()), familial_fraction=None)
                    if result['status'] == 'ok':
                        subset['_position'] = np.arange(len(subset))
                        def estimate(sample):
                            fitted = statistics.fit_reml(sample.accuracy.to_numpy(float), _design(sample, use_parameters), _bootstrap_covariance(sample, kernel), pd.to_numeric(sample.measurement_variance).to_numpy(float), covariance_model=model, families=int(sample['_bootstrap_draw'].nunique()))
                            return {'familial_fraction':fitted['familial_fraction']}
                        bootstrap = statistics.cluster_bootstrap(subset, estimate, n_bootstrap=bootstrap_count, seed=seed)
                        diagnostics.append(dict(base, covariance_model=model, bootstrap=bootstrap))
                        interval = bootstrap['intervals'].get('familial_fraction')
                        result.update(familial_fraction_lower=interval[0] if interval else None, familial_fraction_upper=interval[1] if interval else None, bootstrap_failures=bootstrap['n_failed'], bootstrap_failure_rate=bootstrap['failure_rate'])
                row = {**base, **result, 'covariance_model':model}
                for name in ['fixed_effects']:
                    if name in row:
                        row[name] = json.dumps(row[name], separators=(',', ':'))
                fits.append(row)
            pair_support = verified if scale == 'raw' else retained
            pair_table = _pairs(pair_support, edges) if coverage['ancestry_status'] == 'available' else _pairs(pair_support, edges.iloc[:0])
            result = statistics.parent_offspring(pair_table)
            if result['corrected_slope'] is not None:
                bootstrap = statistics.cluster_bootstrap(pair_table, lambda sample: {'corrected_slope':statistics.parent_offspring(sample)['corrected_slope']}, n_bootstrap=bootstrap_count, seed=seed)
                interval = bootstrap['intervals'].get('corrected_slope')
                result.update(corrected_slope_lower=interval[0] if interval else None, corrected_slope_upper=interval[1] if interval else None, bootstrap_failures=bootstrap['n_failed'], bootstrap_failure_rate=bootstrap['failure_rate'])
                diagnostics.append(dict(base, covariance_model='observed_parent', bootstrap=bootstrap))
            parents.append(dict(base, **result))
    return pd.DataFrame(fits), pd.DataFrame(parents), pd.concat(exclusions, ignore_index=True) if exclusions else pd.DataFrame(columns=data.EXCLUSION_COLUMNS), diagnostics


def _analyze(run, resume=False):
    run = Path(run)
    protocol = _json(run/'protocol.json')
    if hashlib.sha256((run/'protocol.json').read_bytes()).hexdigest() != (run/'protocol.sha256').read_text().strip():
        raise ValueError('protocol hash mismatch')
    inputs = _json(run/'inputs.json')
    if inputs != protocol['inputs']:
        raise ValueError('input inventory differs from frozen protocol')
    _verify(run, inputs)
    config = protocol['config']
    assembly_config = _json(run/'inputs/config.json')
    if {key:value for key,value in config.items() if key != 'effective_optimizer'} != assembly_config:
        raise ValueError('config differs from frozen assembly')
    scores, primary = _read_table(run/'inputs/scores.csv'), _read_table(run/'inputs/primary_scores.csv')
    edges, groups = _read_table(run/'inputs/edges.csv'), _read_table(run/'inputs/groups.csv')
    excluded = _read_table(run/'inputs/exclusions.csv')
    registry = _json(_path(run/'inputs/source_inputs', config['task_registry']))
    coverage = _json(run/'inputs/coverage.json')
    fits, parents, new_exclusions, bootstrap_diagnostics = _fit_tables(scores, primary, groups, edges, registry, excluded, config, coverage)
    prediction_path = run/'inputs/historical/predictions.csv'
    if prediction_path.is_file():
        predictions = _read_table(prediction_path)
        audit = _json(run/'inputs/historical/audit.json')
        coverage.update(prediction_to_score_join=audit['prediction_to_score_join'], historical_status=audit['raw_status'])
        from .exchange import exchange_table
        enriched = scores.merge(groups, on='model_key', how='left')
        exchange = exchange_table(predictions, enriched, n_bootstrap=config['bootstrap_replicates'], seed=config['seed'], max_pairs=config.get('max_pairs', 100000))
        excluded = pd.concat([excluded, _read_table(run/'inputs/historical/exclusions.csv')], ignore_index=True)
    else:
        coverage['historical_status'] = 'historical_predictions_missing'
        exchange = pd.DataFrame([dict(benchmark=benchmark, task=task, harness=harness, estimand=estimand, method='', status='unavailable', reason='historical_predictions_missing', k=None) for benchmark,task,harness in _registry_tasks(registry) for estimand in ['estimation','selection']])
    exchange = _exchange_coverage(exchange, registry)
    methods = sorted({row.get('method', '') for row in audit.get('descriptive_predictions', [])}) if prediction_path.is_file() else None
    skills = _skill_errors(predictions if prediction_path.is_file() else pd.DataFrame(), registry, unavailable_methods=methods, missing_reason='audited_raw_support_unavailable' if prediction_path.is_file() else 'historical_predictions_missing')
    excluded = pd.concat([excluded, new_exclusions], ignore_index=True).reindex(columns=data.EXCLUSION_COLUMNS).fillna('').drop_duplicates().sort_values(data.EXCLUSION_COLUMNS)
    partial = fits.status.ne('ok').any() or parents.status.ne('ok').any() or exchange.status.ne('ok').any() or skills.status.ne('ok').any() or coverage['historical_status'] != 'available'
    coverage.update(status='PARTIAL' if partial else 'COMPLETE', heritability_rows=len(fits), parent_rows=len(parents), exchange_rows=len(exchange), skill_error_rows=len(skills), reasons=dict(sorted(Counter(excluded.reason).items())))
    diagnostics = dict(seed=config['seed'], bootstrap_replicates=config['bootstrap_replicates'], convergence_failures=int((fits.reason == 'optimizer_failure').sum()), fits=fits.to_dict('records'), bootstrap=bootstrap_diagnostics)
    for name, frame in [('heritability.csv',fits),('parent_offspring.csv',parents),('exchange.csv',exchange),('skill_errors.csv',skills),('exclusions.csv',excluded)]:
        _write_table(run/name, frame)
    _write_json(run/'coverage.json', coverage)
    _write_json(run/'fit_diagnostics.json', diagnostics)
    report.write_report(run, coverage, diagnostics, fits, parents, exchange, skills)
    return 0


def _dry_run(out, without_historical=False):
    out = Path(out)
    seed = 20261028
    rng = np.random.default_rng(seed)
    rows, pins, links = [], [], []
    for family in range(8):
        effect = rng.normal(0, .11)
        for child in range(3):
            repo = f'synthetic/f{family}' if child == 0 else f'synthetic/f{family}c{child}'
            revision = f'{family*3+child+1:040x}'
            accuracy = float(np.round(np.clip(.5+effect+rng.normal(0,.035), .05,.95)*200)/200)
            rows.append(dict(model_id=repo, revision=revision, precision='bf16', quantization='none', raw=accuracy, composite=.5))
            pins.append(dict(repo_id=repo, revision=revision, harness='synthetic-h1', precision='bf16', quantization='none', evidence_source='independent_synthetic_manifest'))
            if child:
                links.append(dict(source=f'synthetic/f{family}', target=repo, relation='finetune'))
    _write_table(out/'sources/scores.csv', pd.DataFrame(rows))
    _write_table(out/'sources/pins.csv', pd.DataFrame(pins))
    _write_table(out/'sources/metadata.csv', pd.DataFrame(columns=['model_id','source_model','base_model_relation']))
    _write_json(out/'sources/atlas.json', {'nodes':[row['model_id'] for row in rows], 'edges':links})
    _write_json(out/'sources/denominator.json', {'kind':'synthetic_binomial','n_items':200})
    proof = dict(path='sources/denominator.json', sha256=data._sha256(out/'sources/denominator.json'))
    registry = {'schemas':{'synthetic':{'columns':dict(repo_id='model_id',revision='revision',precision='precision',quantization='quantization'), 'tasks':[
        dict(column='raw',benchmark='synthetic-v1',task='math',harness='synthetic-h1',scale='raw_accuracy',kind='binary',n_items=200,denominator_evidence=proof),
        dict(column='composite',benchmark='synthetic-v1',task='composite',harness='synthetic-h1',scale='raw_accuracy',kind='composite')]}}}
    _write_json(out/'registry.json', registry)
    mapping = dict(prediction_file='predictions.csv',protocol_file='protocol.json',manifest_file='artifact_manifest.json',split_file='splits.csv',baseline_method='no_weight',setting='root_holdout',prediction_scale='raw_accuracy',prediction_semantics='binary_accuracy')
    config = dict(seed=seed,bootstrap_replicates=20,bootstrap_unit='split_group',dry_run=dict(synthetic=True,bootstrap_override=20), sources=[dict(path='sources/scores.csv',name='synthetic_scores',schema='synthetic')],task_registry='registry.json',graph='sources/atlas.json',metadata='sources/metadata.csv',pinned_models='sources/pins.csv',primary_deployment=dict(precision='bf16',quantization='none',harness='synthetic-h1'),lineage_evidence_tier='declared',historical_mapping=mapping,primary_scale='raw_accuracy',sensitivity_scale='corrected_binomial_logit')
    if not without_historical:
        config['historical_audit'] = 'historical'
    _write_json(out/'config.json', config)
    _assemble(out/'config.json', out/'assembly')
    if not without_historical:
        scores = _read_table(out/'assembly/scores.csv').merge(_read_table(out/'assembly/groups.csv'), on='model_key')
        scores = scores.loc[scores.task.eq('math')].reset_index(drop=True)
        predictions, split_rows = [], []
        families = sorted(scores.split_group.unique())
        for fold in [0,1]:
            test_groups = set(families[fold::2])
            for group in families:
                split_rows.append(dict(fold=str(fold),split_group=group,partition='test' if group in test_groups else 'train'))
            for row in scores.loc[scores.split_group.isin(test_groups)].to_dict('records'):
                for method, error in [('weight',.02),('no_weight',.06)]:
                    predictions.append({**{key:row[key] for key in historical.DEPLOYMENT},'method':method,'fold':str(fold),'actual':row['accuracy'],'prediction':row['accuracy']+error,'split_group':row['split_group'],'duplicate_group':row['duplicate_group'],'root_group':row['root_group']})
        _write_table(out/'synthetic_history/predictions.csv', pd.DataFrame(predictions))
        _write_table(out/'synthetic_history/splits.csv', pd.DataFrame(split_rows))
        _write_json(out/'synthetic_history/protocol.json', dict(seed=17,synthetic=True,refit=False))
        _write_json(out/'synthetic_history/artifact_manifest.json', {'artifacts':{name:{'sha256':data._sha256(out/'synthetic_history'/name)} for name in ['predictions.csv','splits.csv','protocol.json']}})
        _historical(out/'synthetic_history', out/'assembly', out/'historical')
    _freeze(out/'config.json', out/'assembly', out/'final')
    return _analyze(out/'final')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    assemble = commands.add_parser('assemble')
    assemble.add_argument('--config', required=True)
    assemble.add_argument('--source-root')
    assemble.add_argument('--out', required=True)
    history = commands.add_parser('historical')
    history.add_argument('--run-root', required=True)
    history.add_argument('--assembled', required=True)
    history.add_argument('--out', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--config', required=True)
    freeze.add_argument('--assembled', required=True)
    freeze.add_argument('--out', required=True)
    analyze = commands.add_parser('analyze')
    analyze.add_argument('--run', required=True)
    analyze.add_argument('--resume', action='store_true')
    dry = commands.add_parser('dry-run')
    dry.add_argument('--out', required=True)
    dry.add_argument('--without-historical', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'assemble':
            return _assemble(args.config, args.out, args.source_root)
        if args.command == 'historical':
            return _historical(args.run_root, args.assembled, args.out)
        if args.command == 'freeze':
            return _freeze(args.config, args.assembled, args.out)
        if args.command == 'analyze':
            return _analyze(args.run, args.resume)
        return _dry_run(args.out, args.without_historical)
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
