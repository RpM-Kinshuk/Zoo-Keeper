import json
import os

import pandas as pd
import pytest
from zoo.schema import LINEAGE_COLUMNS

from lineage.engine import infer
from lineage.io import write_edges
from lineage.testing import replace_record


@pytest.fixture
def method_config():
    return {'method_version': 'p1-0', 'sketch_k': 64, 'reporting_threshold': 0.,
            'retrieval_limit': 50, 'ancestor_decay': .95, 'direction_method_hash': 'a' * 64}


def test_inferred_rows_keep_frozen_schema(synthetic_pool, method_config):
    edges, diagnostics = infer(synthetic_pool.records, config=method_config)
    assert list(edges.columns) == LINEAGE_COLUMNS
    assert len(edges) > 0
    assert set(edges.evidence_tier) == {'detected'}
    assert not edges.calibrated.any()
    assert edges.p_value.isna().all()
    assert edges.method_version.eq('p1-0').all()
    assert set(edges.estimand) == {'same_root'}
    assert diagnostics['direction_coverage'] == 0
    assert diagnostics['abstentions']


def test_direction_requires_asymmetric_evidence(aligned_pair, method_config):
    parent, child = aligned_pair
    child = replace_record(child, parent=parent.key)
    edges, _ = infer([parent, child], config=method_config)
    assert not edges.estimand.isin(['parent', 'ancestor']).any()
    child = replace_record(child, direction_evidence={
        'parent': parent.key, 'status': 'asymmetric', 'score': .8,
        'provenance': 'development_frozen', 'method_hash': 'a' * 64})
    edges, diagnostics = infer([parent, child], config=method_config)
    direct = edges[edges.estimand.eq('parent')]
    assert list(zip(direct.child_key, direct.parent_key)) == [(child.key, parent.key)]
    assert direct.iloc[0].score == .8
    ancestor = edges[edges.estimand.eq('ancestor')].iloc[0]
    assert ancestor.score < direct.iloc[0].score
    assert not ancestor.calibrated
    assert diagnostics['direction_coverage'] > 0


def test_no_common_views_is_a_diagnostic_not_an_edge(aligned_pair, method_config):
    a, b = aligned_pair
    b = replace_record(b, names=tuple('other.' + name for name in b.names))
    edges, diagnostics = infer([a, b], config=method_config)
    assert edges.empty
    assert any(row['reason'] == 'no_eligible_candidates' for row in diagnostics['abstentions'])


def test_direction_method_must_match_frozen_configuration(aligned_pair, method_config):
    parent, child = aligned_pair
    child = replace_record(child, direction_evidence={
        'parent': parent.key, 'status': 'asymmetric', 'score': .8,
        'provenance': 'development_frozen', 'method_hash': 'b' * 64})
    edges, _ = infer([parent, child], config=method_config)
    assert not edges.estimand.isin(['parent', 'ancestor']).any()


def test_missing_or_mismatched_null_never_claims_calibration(aligned_pair, method_config):
    edges, _ = infer(list(aligned_pair), config=method_config,
                     calibration={'same_root': {'settings': {'method_hash': 'wrong'}}})
    assert not edges.calibrated.any()
    assert edges.p_value.isna().all()


def test_query_control_counts_unscorable_candidates_in_full_pool(aligned_pair, method_config, monkeypatch):
    import numpy as np
    import lineage.engine as engine
    a, b = aligned_pair
    absent = replace_record(b, key='lab/missing@' + 'c' * 40,
                            names=tuple('missing.' + name for name in b.names))
    records = [a, b, absent]
    _, provenance = infer(records, config=method_config)
    artifact = {'estimand': 'same_root', 'settings': {
        'method_hash': provenance['method_hash'],
        'candidate_universe_hash': provenance['candidate_universe_hash']}}
    monkeypatch.setattr(engine, 'apply_null', lambda *a, **kw: {
        'status': 'calibrated', 'p_value': 1e-8, 'validity': 'exchangeable'})
    seen = []
    def control(values, *, pool_size, **kw):
        seen.append(pool_size)
        return np.ones(len(values), dtype=bool)
    monkeypatch.setattr(engine, 'query_control', control)
    infer(records, config=method_config, calibration={'same_root': artifact})
    assert seen == [2, 2]


@pytest.mark.parametrize('setting,value', [('row_block', 128), ('row_block', 512),
                                          ('sketch_seed', 7), ('evidence_version', 'other')])
def test_calibration_uses_measured_settings(aligned_pair, method_config, monkeypatch, setting, value):
    import numpy as np
    import lineage.engine as engine
    changed = {setting: value}
    if setting == 'row_block':
        changed['rows'] = np.tile([0, value], (6, 1))
    records = [replace_record(r, **changed) for r in aligned_pair]
    cfg = dict(method_config, calibration_settings={
        'sketch_seed': 20261028, 'evidence_version': 'synthetic-v1'})
    _, provenance = infer(records, config=cfg)
    settings = dict(engine.STATISTIC, sketch_seed=20261028, evidence_version='synthetic-v1',
                    method_hash=provenance['method_hash'],
                    candidate_universe_hash=provenance['candidate_universe_hash'])
    artifact = {'estimand': 'same_root', 'settings': settings}
    def apply(score, artifact, *, settings, **kwargs):
        if any(settings.get(k) != v for k,v in artifact['settings'].items()):
            return {'status': 'uncalibrated', 'validity': 'unvalidated', 'p_value': None}
        return {'status': 'calibrated', 'validity': 'exchangeable', 'p_value': 1e-8}
    monkeypatch.setattr(engine, 'apply_null', apply)
    monkeypatch.setattr(engine, 'query_control', lambda p, **kw: np.ones(len(p), dtype=bool))
    edges, _ = infer(records, config=cfg, calibration={'same_root': artifact})
    assert len(edges) > 0
    assert not edges.calibrated.any()
    assert edges.p_value.isna().all()


def test_interrupted_output_retains_previous(synthetic_pool, method_config, tmp_path, monkeypatch):
    edges, _ = infer(synthetic_pool.records, config=method_config)
    out = tmp_path / 'edges.parquet'
    write_edges(edges, out, provenance={'input_hash': 'a'})
    before, destination = out.read_bytes(), out.resolve()
    monkeypatch.setattr(os, 'replace', lambda *args: (_ for _ in ()).throw(OSError('interrupted')))
    with pytest.raises(OSError, match='interrupted'):
        write_edges(edges, out, provenance={'input_hash': 'b'})
    assert out.read_bytes() == before
    assert out.resolve() == destination


def test_empty_output_has_exact_arrow_types(tmp_path):
    import pyarrow.parquet as pq
    out = tmp_path / 'empty.parquet'
    write_edges(pd.DataFrame(columns=LINEAGE_COLUMNS), out, provenance={})
    schema = pq.read_schema(out)
    assert schema.names == LINEAGE_COLUMNS
    assert str(schema.field('calibrated').type) == 'bool'
    assert str(schema.field('p_value').type) == 'double'


def test_cli_dry_run_and_resume_reject_changed_input(tmp_path):
    from lineage.cli import main
    target = tmp_path / 'run'
    assert main(['dry-run', '--out', str(target)]) == 0
    edge_file = target / 'edges.parquet'
    before = edge_file.resolve()
    assert main(['dry-run', '--out', str(target), '--resume']) == 0
    assert edge_file.resolve() == before
    config = tmp_path / 'changed.json'
    config.write_text(json.dumps({'method_version': 'p1-0', 'reporting_threshold': .9}))
    with pytest.raises(ValueError, match='resume'):
        main(['dry-run', '--out', str(target), '--resume', '--config', str(config)])


def test_lineage_into_zoo_build(tmp_path):
    from lineage.cli import main
    from zoo.build import build_zoo, open_tables
    from zoo.testing import make_universe
    universe = make_universe(tmp_path)
    edges = tmp_path / 'p1' / 'edges.parquet'
    assert main(['infer', '--manifests', str(universe.manifests),
                 '--fingerprints', str(universe.fingerprints), '--data-root', str(tmp_path),
                 '--out', str(edges)]) == 0
    predictions = pd.read_parquet(edges)
    assert len(predictions) > 0
    build_zoo(universe.manifests, [universe.fingerprints], tmp_path / 'built',
              data_root=tmp_path, lineage_edges=edges)
    merged = open_tables(tmp_path / 'built')['lineage']
    actual = merged[merged.method_version.eq('p1-0')].copy()
    actual['p_value'] = pd.to_numeric(actual.p_value).astype(float)
    predictions['p_value'] = pd.to_numeric(predictions.p_value).astype(float)
    pd.testing.assert_frame_equal(actual[LINEAGE_COLUMNS].reset_index(drop=True),
                                  predictions[LINEAGE_COLUMNS].reset_index(drop=True),
                                  check_dtype=False)
