"""Local detected lineage, with explicit direction and calibration boundaries."""
from collections import defaultdict
import hashlib
import json
import math
import re

import networkx as nx
import numpy as np
import pandas as pd
from zoo.schema import LINEAGE_COLUMNS

from .abliteration import projected_abliteration
from .calibration import STATISTIC, apply_null, query_control
from .merge import fit_merge
from .records import compatible_views
from .retrieval import exclusion_reason, retrieve
from .tree import assemble_tree
from .typing import pair_features, type_edge
from .upstream import median_similarity

METHOD_VERSION = 'p1-0'
DEFAULT_CONFIG = dict(method_version=METHOD_VERSION, sketch_k=64, reporting_threshold=0.,
                      retrieval_limit=50, ancestor_decay=.95, alpha=.05,
                      residual_views=[], typing_rules=None, merge_proposals=[],
                      assemble_tree=False, roots=[], direction_method_hash=None)


def clean(value):
    """JSON scalars only; unknown nonfinite diagnostics remain explicit nulls."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return clean(value.item())
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def configuration(config):
    result = {**DEFAULT_CONFIG, **config}
    if result['method_version'] != METHOD_VERSION or result['sketch_k'] != 64:
        raise ValueError('unsupported method version or sketch size')
    if (type(result['retrieval_limit']) is not int or result['retrieval_limit'] < 1
            or not math.isfinite(result['reporting_threshold'])
            or not 0 < result['ancestor_decay'] < 1 or not 0 < result['alpha'] < 1):
        raise ValueError('invalid engine thresholds or retrieval limit')
    # Configs are frozen inputs, never silently convert NaN thresholds to null.
    json.dumps(result, allow_nan=False)
    if result['direction_method_hash'] is not None and not re.fullmatch(r'[0-9a-f]{64}', result['direction_method_hash']):
        raise ValueError('direction method requires a SHA256 hash')
    return result


def record_hash(record):
    facts = {key: record.meta.get(key) for key in (
        'tier', 'row_block', 'sketch_seed', 'evidence_version', 'parent',
        'duplicate_group', 'direction_evidence', 'adapter_factors',
        'quantization_evidence', 'merge_recipe', 'storage_dtype',
        'manifest_sha256', 'fingerprint_sha256', 'family')}
    return digest(dict(key=record.key, arch=record.arch, names=record.names,
                       geometry=record.geometry, shapes=record.shapes, rows=record.rows,
                       sketches_sha256=hashlib.sha256(record.sketches.tobytes()).hexdigest(),
                       delta=record.delta, meta=facts))


def _direction(child, parent, features, cfg):
    evidence = child.meta.get('direction_evidence', {})
    if (evidence.get('parent') == parent.key and evidence.get('status') == 'asymmetric'
            and evidence.get('provenance') == 'development_frozen'
            and isinstance(evidence.get('method_hash'), str)
            and evidence['method_hash'] == cfg['direction_method_hash']
            and isinstance(evidence.get('score'), (int, float))
            and math.isfinite(evidence['score']) and 0 < evidence['score'] <= 1):
        return float(evidence['score']), dict(evidence)
    if features.get('adapter_factors'):
        return None, {'reason': 'adapter_factors_need_frozen_direction_scorer'}
    return None, {'reason': 'no_asymmetric_direction_evidence'}


def infer(records, *, config, calibration=None):
    cfg = configuration(config)
    records = sorted(records, key=lambda record: record.key)
    by_key = {record.key: record for record in records}
    if len(by_key) != len(records):
        raise ValueError('duplicate record identities')
    method_hash = digest(cfg)
    universe_hash = digest([(r.key, r.arch, r.meta.get('tier'), r.meta.get('duplicate_group'))
                            for r in records])
    diagnostics = dict(method_version=METHOD_VERSION, method_hash=method_hash,
                       candidate_universe_hash=universe_hash,
                       input_hash=digest([(r.key, record_hash(r)) for r in records]),
                       n_records=len(records), abstentions=[], pairs=[], merges=[],
                       direction_coverage=0., assembly=None,
                       reporting_scope='detected development scores; calibrated discoveries only when supported')
    rows, parent_rows = [], []
    pools, scored = {}, 0

    def abstain(child, parent, reason, **extra):
        diagnostics['abstentions'].append(dict(query_key=child, candidate_key=parent,
                                                reason=reason, **clean(extra)))

    def edge(child, parent, estimand, score, edge_type, evidence, statistic=None):
        if score < cfg['reporting_threshold']:
            abstain(child.key, parent.key, 'below_reporting_threshold', estimand=estimand)
            return None
        artifact = (calibration or {}).get(estimand)
        p_value, calibrated = None, False
        result = dict(status='uncalibrated', reason='missing_matching_null', validity='unvalidated')
        if artifact and estimand != 'ancestor':
            settings = {**cfg.get('calibration_settings', {}), **STATISTIC,
                        **{key: child.meta.get(key) for key in
                           ('tier', 'row_block', 'sketch_seed', 'evidence_version')},
                        'method_hash': method_hash, 'candidate_universe_hash': universe_hash,
                        'architecture': child.arch}
            if (artifact.get('estimand') == estimand
                    and artifact.get('settings', {}).get('method_hash') == method_hash
                    and artifact.get('settings', {}).get('candidate_universe_hash') == universe_hash):
                result = apply_null(statistic, artifact, family=child.meta.get('family', child.arch),
                                    settings=settings)
                p_value = result.get('p_value')
                calibrated = result.get('validity') == 'exchangeable' and p_value is not None
        evidence.update(space='sketch', calibration_mode=result['status'],
                        calibration_reason=result.get('reason'),
                        calibration_statistic='median_cosine' if estimand != 'ancestor' else None,
                        method_hash=method_hash, input_hash=diagnostics['input_hash'])
        row = dict(child_key=child.key, parent_key=parent.key, estimand=estimand,
                   edge_type=edge_type, evidence_tier='detected', evidence=evidence,
                   score=float(score), p_value=p_value, calibrated=calibrated,
                   method_version=METHOD_VERSION)
        rows.append(row)
        return row

    for child in records:
        if child.meta.get('tier') != 'row_blocks':
            abstain(child.key, None, 'unsupported_or_unmeasured_tier')
            continue
        pool = [candidate for candidate in records if not exclusion_reason(child, candidate)
                and candidate.arch == child.arch and candidate.meta.get('tier') == 'row_blocks']
        pools[child.key] = len(pool)
        candidates = retrieve(child, records, limit=cfg['retrieval_limit'])
        if not candidates:
            abstain(child.key, None, 'no_eligible_candidates')
        retrieved = {row['parent_key'] for row in candidates}
        for candidate in records:
            reason = exclusion_reason(child, candidate)
            if not reason and candidate.arch != child.arch:
                reason = 'architecture_mismatch'
            if not reason and candidate.meta.get('tier') != 'row_blocks':
                reason = 'candidate_unmeasured_tier'
            if not reason and candidate.key not in retrieved:
                reason = 'not_shortlisted' if compatible_views(child, candidate) else 'incompatible_views'
            if reason:
                abstain(child.key, candidate.key, reason)
        for candidate in candidates:
            parent = by_key[candidate['parent_key']]
            score = median_similarity(child, parent)
            if score is None or not math.isfinite(score):
                abstain(child.key, parent.key, 'no_compatible_scored_views')
                continue
            scored += 1
            features = pair_features(child, parent)
            typed = type_edge(features, cfg['typing_rules']) if cfg['typing_rules'] else {
                'edge_type': 'unknown', 'status': 'unfitted_rules'}
            stacked = projected_abliteration(child, parent, residual_views=cfg['residual_views'])
            diagnostics['pairs'].append(clean(dict(query_key=child.key, candidate_key=parent.key,
                median_cosine=score, n_shared=len(compatible_views(child, parent)),
                typing=typed, stacked=stacked)))
            edge(child, parent, 'same_root', score, 'unknown',
                 dict(reason='symmetric_similarity', n_shared=len(compatible_views(child, parent))), score)
            directed_score, direction = _direction(child, parent, features, cfg)
            if directed_score is None:
                abstain(child.key, parent.key, direction['reason'], estimand='parent')
                continue
            row = edge(child, parent, 'parent', directed_score, typed['edge_type'],
                       dict(reason='asymmetric_direction_evidence', direction=direction,
                            typing=typed, stacked=stacked), score)
            if row is not None:
                parent_rows.append(row)

    # Calibration is applied per estimand and query across the full eligible
    # pool. Unretrieved and uncalibrated candidates contribute zero e-values.
    grouped = defaultdict(list)
    for row in rows:
        if row['calibrated']:
            grouped[(row['child_key'], row['estimand'])].append(row)
    rejected_rows = set()
    for (query, estimand), group in grouped.items():
        rejected = query_control(np.asarray([r['p_value'] for r in group]),
                                 alpha=cfg['alpha'], pool_size=pools[query], validity='exchangeable')
        for row, selected in zip(group, rejected):
            row['evidence']['query_control_rejected'] = bool(selected)
            row['evidence']['full_pool_size'] = pools[query]
            if not selected:
                rejected_rows.add(id(row))
                abstain(query, row['parent_key'], 'not_rejected_by_query_control', estimand=estimand)
    rows = [row for row in rows if id(row) not in rejected_rows]
    parent_rows = [row for row in parent_rows if id(row) not in rejected_rows]
    graph = nx.DiGraph((r['parent_key'], r['child_key']) for r in parent_rows)
    cyclic = [component for component in nx.strongly_connected_components(graph) if len(component) > 1]
    bad = {id(row) for row in parent_rows if any(
        {row['parent_key'], row['child_key']} <= component for component in cyclic)}
    for row in parent_rows:
        if id(row) in bad:
            abstain(row['child_key'], row['parent_key'], 'conflicting_direction_cycle')
    parent_rows = [row for row in parent_rows if id(row) not in bad]
    rows = [row for row in rows if id(row) not in bad]
    graph = nx.DiGraph()
    for row in parent_rows:
        graph.add_edge(row['parent_key'], row['child_key'], score=row['score'])
    diagnostics['direction_coverage'] = len(parent_rows) / scored if scored else 0.
    # Conservative widest-path closure, computed from supported parent evidence.
    # It carries no borrowed calibration guarantee and never reads truth tables.
    for parent in sorted(graph):
        scores = {parent: 1.}
        for node in nx.topological_sort(graph):
            if node in scores:
                for child in graph.successors(node):
                    value = min(scores[node], graph[node][child]['score']) * cfg['ancestor_decay']
                    scores[child] = max(scores.get(child, 0.), value)
        for child, value in sorted(scores.items()):
            if child != parent:
                edge(by_key[child], by_key[parent], 'ancestor', value, 'unknown',
                     dict(reason='supported_parent_path', calibration_scope='uncalibrated_closure'))
    for proposal in cfg['merge_proposals']:
        keys = [proposal['child'], *proposal['parents'], proposal['base']]
        if any(key not in by_key for key in keys):
            raise ValueError('merge proposal references absent records')
        result = fit_merge(by_key[proposal['child']], [by_key[k] for k in proposal['parents']],
                           by_key[proposal['base']], bootstrap=proposal.get('bootstrap', 200))
        diagnostics['merges'].append(clean(dict(proposal=proposal, result=result,
             scope='effective regression evidence; no direction inferred from fit')))
    if cfg['assemble_tree']:
        proposals = pd.DataFrame([dict(parent_key=r['parent_key'], child_key=r['child_key'],
                                        cost=-math.log(max(r['score'], 1e-12)), direction_status='resolved')
                                   for r in parent_rows],
                                  columns=['parent_key', 'child_key', 'cost', 'direction_status'])
        tree = assemble_tree(sorted(by_key), proposals, roots=cfg['roots'])
        diagnostics['assembly'] = clean(dict(edges=tree.to_dict('records'), attributes=tree.attrs))
    for row in rows:
        row['evidence'] = canonical(row['evidence'])
    frame = pd.DataFrame(rows, columns=LINEAGE_COLUMNS)
    frame['calibrated'] = frame.calibrated.astype(bool)
    for column in ('score', 'p_value'):
        frame[column] = pd.to_numeric(frame[column]).astype(float)
    frame = frame.sort_values(['child_key', 'parent_key', 'estimand', 'evidence'], kind='stable').reset_index(drop=True)
    return frame, clean(diagnostics)
