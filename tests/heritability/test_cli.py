import hashlib
import json

import pandas as pd
import pytest

from studies.heritability.cli import main


def test_tiny_dry_run_uses_full_pipeline_and_portable_freeze(tmp_path):
    root=tmp_path/'tiny'
    assert main(['dry-run','--out',str(root)]) == 0
    run=root/'final'
    assert (run/'report.md').is_file()
    for name in ['heritability.csv','parent_offspring.csv','exchange.csv','fit_diagnostics.json','coverage.json','protocol.json','inputs.json','exclusions.csv']:
        assert (run/name).is_file()
    payload=json.loads((run/'protocol.json').read_text())
    assert payload['config']['bootstrap_replicates'] == 20
    assert payload['config']['dry_run']['synthetic'] is True
    assert str(tmp_path) not in (run/'protocol.json').read_text()
    summary=pd.read_csv(run/'heritability.csv')
    assert {'family','network'} <= set(summary.covariance_model)
    assert {'raw','logit'} <= set(summary.scale)
    assert pd.read_csv(run/'exchange.csv').shape[0] > 0


def test_resume_refuses_corrupt_inputs_and_keeps_previous_report(tmp_path):
    root=tmp_path/'tiny'
    assert main(['dry-run','--out',str(root)]) == 0
    run=root/'final'
    report=(run/'report.md').read_bytes()
    assert main(['analyze','--run',str(run),'--resume']) == 0
    p=run/'inputs/scores.csv'
    p.write_bytes(p.read_bytes()+b'\n')
    assert main(['analyze','--run',str(run),'--resume']) != 0
    assert (run/'report.md').read_bytes() == report


def test_missing_historical_predictions_still_yields_honest_partial_report(tmp_path):
    root=tmp_path/'tiny'
    assert main(['dry-run','--out',str(root),'--without-historical']) == 0
    run=root/'final'
    assert 'PARTIAL' in (run/'report.md').read_text()
    exchange=pd.read_csv(run/'exchange.csv')
    assert 'historical_predictions_missing' in set(exchange.reason)
    assert not pd.read_csv(run/'heritability.csv').empty


def test_identical_dry_runs_have_identical_statistical_tables(tmp_path):
    for name in ['one','two']:
        assert main(['dry-run','--out',str(tmp_path/name)]) == 0
    for name in ['heritability.csv','parent_offspring.csv','exchange.csv']:
        assert (tmp_path/'one/final'/name).read_bytes() == (tmp_path/'two/final'/name).read_bytes()


def test_report_failure_preserves_previous_complete_report(tmp_path,monkeypatch):
    from studies.heritability import report
    root=tmp_path/'tiny'
    assert main(['dry-run','--out',str(root)]) == 0
    run=root/'final'
    original=(run/'report.md').read_bytes()
    def fail(*args,**kwargs):
        raise OSError('simulated report failure')
    monkeypatch.setattr(report,'write_report',fail)
    assert main(['analyze','--run',str(run)]) != 0
    assert (run/'report.md').read_bytes() == original


def test_freeze_rejects_config_mutation(tmp_path):
    root=tmp_path/'tiny'
    assert main(['dry-run','--out',str(root)]) == 0
    config=root/'config.json'
    payload=json.loads(config.read_text())
    payload['seed']=1
    config.write_text(json.dumps(payload))
    assert main(['freeze','--config',str(config),'--assembled',str(root/'assembly'),'--out',str(root/'final')]) != 0


def test_assembly_quarantines_self_parent_declarations_without_losing_scores(tmp_path):
    root = tmp_path/'tiny'
    assert main(['dry-run', '--out', str(root)]) == 0
    config = json.loads((root/'config.json').read_text())
    metadata_path = root/config['metadata']
    metadata = pd.read_csv(metadata_path)
    metadata = pd.concat([metadata, pd.DataFrame([dict(model_id='synthetic/f0', source_model='SYNTHETIC/F0', base_model_relation='declared')])], ignore_index=True)
    metadata.to_csv(metadata_path, index=False)
    assembly = root/'quarantined'
    assert main(['assemble', '--config', str(root/'config.json'), '--out', str(assembly)]) == 0
    excluded = pd.read_csv(assembly/'exclusions.csv')
    assert 'self_parent_declaration' in set(excluded.reason)
    assert len(pd.read_csv(assembly/'scores.csv')) == len(pd.read_csv(root/'assembly/scores.csv'))
    assert json.loads((assembly/'coverage.json').read_text())['ancestry_status'] == 'available'


def test_freeze_refuses_mutated_assembly_before_copying(tmp_path):
    root = tmp_path/'tiny'
    assert main(['dry-run', '--out', str(root)]) == 0
    score_path = root/'assembly/scores.csv'
    score_path.write_bytes(score_path.read_bytes()+b'\n')
    target = root/'changed'
    assert main(['freeze', '--config', str(root/'config.json'), '--assembled', str(root/'assembly'), '--out', str(target)]) != 0
    assert not (target/'protocol.json').exists()


def test_synthetic_pipeline_produces_actual_eligible_fits_and_audited_predictions(tmp_path):
    root = tmp_path/'tiny'
    assert main(['dry-run', '--out', str(root)]) == 0
    fits = pd.read_csv(root/'final/heritability.csv')
    math = fits.loc[fits.task.eq('math')]
    assert len(math) == 4
    assert math.converged.all()
    assert math.n.eq(24).all()
    parents = pd.read_csv(root/'final/parent_offspring.csv')
    assert parents.loc[parents.task.eq('math'), 'n'].eq(16).all()
    exchange = pd.read_csv(root/'final/exchange.csv')
    assert ((exchange.analysis_tier == 'verified_protocol_eligible') & (exchange.status == 'ok')).any()
    skills = pd.read_csv(root/'final/skill_errors.csv')
    assert {'mae_sd','r2','n_observed','excluded_from_common_support'} <= set(skills.columns)
    assert 'Per-skill prediction errors' in (root/'final/report.md').read_text()


def test_numeric_only_pinned_revisions_preserve_leading_zero_identity(tmp_path):
    root = tmp_path/'tiny'
    assert main(['dry-run', '--out', str(root)]) == 0
    config = json.loads((root/'config.json').read_text())
    for relative in [config['pinned_models'], config['sources'][0]['path']]:
        path = root/relative
        frame = pd.read_csv(path, dtype=str).iloc[:3]
        frame.to_csv(path, index=False)
    assembly = root/'numeric_sha'
    assert main(['assemble', '--config', str(root/'config.json'), '--out', str(assembly)]) == 0
    assert pd.read_csv(assembly/'scores.csv').revision_verified.all()


def test_other_ancestry_cycles_remain_partial_without_fabricated_roots(tmp_path):
    root = tmp_path/'tiny'
    assert main(['dry-run', '--out', str(root)]) == 0
    config = json.loads((root/'config.json').read_text())
    graph_path = root/config['graph']
    graph = json.loads(graph_path.read_text())
    graph['edges'].append(dict(source='synthetic/f0c1', target='synthetic/f0', relation='declared'))
    graph_path.write_text(json.dumps(graph))
    assembly = root/'cycle'
    assert main(['assemble', '--config', str(root/'config.json'), '--out', str(assembly)]) == 0
    groups = pd.read_csv(assembly/'groups.csv', keep_default_na=False)
    assert groups.root_group.eq('').all()
    assert groups.split_group.nunique() == 1
    assert 'ancestry_cycle' in set(pd.read_csv(assembly/'exclusions.csv').reason)
    final = root/'cycle_final'
    assert main(['freeze', '--config', str(root/'config.json'), '--assembled', str(assembly), '--out', str(final)]) == 0
    assert main(['analyze', '--run', str(final)]) == 0
    assert 'PARTIAL' in (final/'report.md').read_text()
    assert pd.read_csv(final/'heritability.csv').reason.eq('ancestry_cycle').all()


def test_resampled_covariance_separates_independent_copies_of_one_family():
    import numpy as np
    from studies.heritability import cli
    sample = pd.DataFrame(dict(_position=[0,1,0,1], _bootstrap_draw=[0,0,1,1]))
    kernel = np.array([[2.,1.],[1.,2.]])
    covariance = cli._bootstrap_covariance(sample, kernel)
    assert np.array_equal(covariance[:2,:2], kernel)
    assert np.array_equal(covariance[2:,2:], kernel)
    assert not covariance[:2,2:].any()


def test_observed_parent_matching_casefolds_only_repository_identity():
    from studies.heritability import cli
    sha = 'a'*40
    frame = pd.DataFrame(dict(model_key=['Org/Parent@'+sha,'Org/Child@'+sha], accuracy=[.3,.5], measurement_variance=[.01,.01], split_group=['s','s']))
    edges = pd.DataFrame(dict(parent_id=['org/parent'], child_id=['org/child'], parent_key=['org/parent@'+sha], child_key=['org/child@'+sha]))
    assert len(cli._pairs(frame, edges)) == 1


def test_missing_exchange_harness_is_reported_separately():
    from studies.heritability import cli
    exchange = pd.DataFrame([dict(benchmark='v2',task='math',harness='h1',status='ok')])
    registry = {'schemas':{'v2':{'tasks':[dict(benchmark='v2',task='math',harness='h1'),dict(benchmark='v2',task='math',harness='h2')]}}}
    covered = cli._exchange_coverage(exchange, registry)
    missing = covered.loc[covered.harness.eq('h2')]
    assert len(missing) == 1
    assert missing.iloc[0].reason == 'historical_task_missing'


def test_skill_errors_use_common_models_across_tasks_and_methods():
    from studies.heritability import cli
    rows = []
    for task, keys in [('a',['m1','m2','m3']),('b',['m1','m2'])]:
        for method in ['weight','no_weight']:
            for index, key in enumerate(keys):
                if task == 'a' and method == 'no_weight' and key == 'm3':
                    continue
                rows.append(dict(model_key='org/'+key+'@'+'a'*40,benchmark='v2',task=task,harness='h1',precision='bf16',quantization='none',setting='root_holdout',fold='0',method=method,actual=.2+.2*index,prediction=.3+.2*index,revision_verified=True,provenance_status='eligible',prediction_scale='raw_accuracy'))
    registry = {'schemas':{'v2':{'tasks':[dict(benchmark='v2',task='a',harness='h1'),dict(benchmark='v2',task='b',harness='h1'),dict(benchmark='v2',task='absent',harness='h1')]}}}
    result = cli._skill_errors(pd.DataFrame(rows), registry)
    available = result.loc[result.task.isin(['a','b'])]
    assert available.n.eq(2).all()
    assert available.mae_sd.tolist() == pytest.approx([1.]*4)
    assert available.r2.tolist() == pytest.approx([0.]*4, abs=1e-12)
    row = available.loc[available.task.eq('a') & available.method.eq('weight')].iloc[0]
    assert row.n_observed == 3 and row.excluded_from_common_support == 1
    assert result.loc[result.task.eq('absent'), 'status'].eq('unavailable').all()
    constant = pd.DataFrame(rows)
    constant['actual'] = .4
    result = cli._skill_errors(constant, registry)
    supported = result.loc[result.n.gt(0)]
    assert supported.mae_sd.isna().all() and supported.r2.isna().all()
    assert supported.reason.eq('constant_target').all()
