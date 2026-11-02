"""Evaluation contracts: explicit truth, complete denominators, frozen artifacts."""
import json

import pandas as pd
import pytest

from lineage.evaluate import evaluate, freeze


@pytest.fixture
def evaluation_fixture():
    pairs = [('q1', 'p1', 1, .9), ('q1', 'n1', 0, .8),
             ('q2', 'p2', 1, .7), ('q2', 'n2', 0, .1)]
    return dict(predictions=pd.DataFrame([dict(child_key=q, parent_key=c, estimand='parent', score=s)
                                         for q,c,_,s in pairs]),
                truth=pd.DataFrame([dict(query_key=q, candidate_key=c, estimand='parent', label=y,
                                         truth_tier='lab_exact', root_group=q, split='test')
                                    for q,c,y,_ in pairs]),
                universe=pd.DataFrame([dict(query_key=q, candidate_key=c, eligible=True) for q,c,_,_ in pairs]),
                costs=pd.DataFrame([dict(query_key=q, bytes_read=100, seconds=.1, status='ok', reason='',
                    query_bytes=40, reference_bytes=40, verification_bytes=20, reference_amortization_queries=2)
                    for q in ['q1', 'q2']]), config={'deployment_threshold': .75, 'pool_sizes': [2,4,8], 'seed': 7})


def test_false_parents_denominators_and_missing_rows(evaluation_fixture):
    out = evaluate(**evaluation_fixture)
    assert out['at_frozen_threshold']['false_matches_per_query'] == .5
    assert out['at_frozen_threshold']['tpr'] == .5
    assert out['at_frozen_threshold']['n_queries'] == 2
    evaluation_fixture['predictions'] = evaluation_fixture['predictions'].iloc[:2]
    out = evaluate(**evaluation_fixture)
    assert out['by_estimand']['parent']['candidate_coverage'] == .5
    assert out['by_estimand']['parent']['query_coverage'] == .5
    assert out['curves'].n_queries.eq(2).all()


def test_unknown_hub_truth_is_not_a_negative(evaluation_fixture):
    evaluation_fixture['truth'].loc[1, 'label'] = None
    out = evaluate(**evaluation_fixture)
    assert out['unknown_truth_pairs'] == 1
    assert out['negative_pairs'] == 1
    assert out['gate']['status'] == 'NOT_EVALUABLE'
    assert all(v['status'] == 'not_run' for v in out['baselines'].values())


def test_duplicate_groups_and_merge_roots_do_not_cross_split(evaluation_fixture):
    truth = evaluation_fixture['truth']
    truth['duplicate_group'] = ['dup', 'other', 'dup', 'other2']
    truth.loc[2:, 'split'] = 'train'
    with pytest.raises(ValueError, match='duplicate_group'):
        evaluate(**evaluation_fixture)


def test_root_holdout_includes_all_merge_parents(evaluation_fixture):
    truth = evaluation_fixture['truth']
    truth['root_group'] = [['a','b'], ['a'], ['b'], ['b']]
    truth.loc[2:, 'split'] = 'train'
    with pytest.raises(ValueError, match='root_group'):
        evaluate(**evaluation_fixture)


def test_pool_curves_include_unretrieved_candidates(evaluation_fixture):
    evaluation_fixture['predictions'] = evaluation_fixture['predictions'].iloc[:1]
    out = evaluate(**evaluation_fixture)
    assert set(out['curves'].pool_size) == {2,4}
    largest = out['curves'].query('pool_size == 4').iloc[0]
    assert largest.n_pairs == 4
    assert largest.n_abstentions == 3
    assert largest.scale == 'toy'


def test_test_tpr_threshold_does_not_tune_deployment(evaluation_fixture):
    evaluation_fixture['predictions']['score'] = [.8, .7, .8, .2]
    out = evaluate(**evaluation_fixture)
    assert out['at_frozen_threshold']['threshold'] == .75
    assert out['by_estimand']['parent']['diagnostic_tpr90']['threshold'] == .8
    assert out['by_estimand']['parent']['diagnostic_tpr90']['tpr'] == 1
    assert out['by_estimand']['parent']['auroc'] == 1


def test_ineligible_claim_abstains_and_missing_cost_is_unknown(evaluation_fixture):
    evaluation_fixture['universe'].loc[0, 'eligible'] = False
    evaluation_fixture['costs'] = evaluation_fixture['costs'].iloc[:1]
    out = evaluate(**evaluation_fixture)
    assert out['at_frozen_threshold']['tpr'] == 0
    assert out['costs']['status'] == 'incomplete'
    assert out['costs']['bytes_per_answer'] is None
    assert out['costs']['missing_queries'] == ['q2']


def test_freeze_is_immutable(tmp_path):
    source = tmp_path / 'truth.json'
    source.write_text('{"label":1}')
    out = tmp_path / 'frozen'
    frozen = freeze({'deployment_threshold': .75}, {'truth': source}, out)
    assert len(frozen['method_hash']) == 64
    assert json.loads((out / 'protocol.json').read_text())['config']['deployment_threshold'] == .75
    with pytest.raises(FileExistsError):
        freeze({'deployment_threshold': .7}, {'truth': source}, out)
    assert json.loads((out / 'protocol.json').read_text())['config']['deployment_threshold'] == .75


def test_auroc_undefined_without_certified_negative(evaluation_fixture):
    evaluation_fixture['truth'].loc[evaluation_fixture['truth'].label.eq(0), 'label'] = None
    out = evaluate(**evaluation_fixture)
    assert out['by_estimand']['parent']['auroc'] is None
    assert out['by_estimand']['parent']['auroc_status'] == 'undefined_no_negatives'


def test_cli_requires_freeze_and_preserves_frozen_inputs(tmp_path, evaluation_fixture):
    from lineage.cli import main
    paths = {}
    for name in ('predictions','truth','universe','costs'):
        paths[name] = str(tmp_path / (name + '.parquet'))
        evaluation_fixture[name].to_parquet(paths[name],index=False)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(evaluation_fixture['config']))
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps(paths))
    out = tmp_path / 'out'
    argv = ['evaluate','--config',str(config),'--out',str(out)]
    for name,path in paths.items():
        argv.extend(['--'+name,path])
    with pytest.raises(FileNotFoundError):
        main(argv)
    assert main(['freeze','--config',str(config),'--inputs',str(inputs),'--out',str(out)]) == 0
    assert main(argv) == 0
    assert json.loads((out / 'metrics.json').read_text())['at_frozen_threshold']['tpr'] == .5
    assert pd.read_parquet(out / 'coverage.parquet').shape[0] == 12
    with pytest.raises(FileExistsError):
        main(argv)
    pd.DataFrame().to_parquet(paths['truth'])
    with pytest.raises(ValueError, match='changed'):
        main(argv)


def test_cost_components_and_amortization_are_measured(evaluation_fixture):
    out = evaluate(**evaluation_fixture)
    assert out['costs']['reference_bytes_unamortized'] == 80
    assert out['costs']['reference_bytes_amortized'] == 40
    assert out['costs']['bytes_per_answer'] == 100
    assert out['by_truth_tier']['lab_exact']['parent']['costs']['bytes_per_answer_amortized'] == 80
    assert out['curves'].query("estimand == 'parent' and truth_tier == 'all'").bytes_per_answer.eq(100).all()


def test_cli_public_lab_checks_supplied_truth_against_construction(tmp_path):
    from lineage.cli import main
    from tests.lab.test_score import fixture_graph
    models, edges, predictions, manifests, universe = fixture_graph()
    lab_dir = tmp_path / 'lab'
    lab_dir.mkdir()
    paths = {}
    for name,frame in [('models',models),('edges',edges),('predictions',predictions),('universe',universe)]:
        paths[name] = str(lab_dir / (name + '.parquet'))
        frame.to_parquet(paths[name],index=False)
    paths['manifests'] = str(lab_dir / 'manifests.json')
    (lab_dir / 'manifests.json').write_text(json.dumps(manifests))
    # d -> c is a true parent; deliberately claiming 0 must fail the public scorer.
    truth = pd.DataFrame([dict(query_key=universe.iloc[1].query_key,
        candidate_key=universe.iloc[1].candidate_key,estimand='parent',label=0,
        truth_tier='lab_exact',root_group='root',split='test')])
    paths['truth'] = str(lab_dir / 'truth.parquet')
    truth.to_parquet(paths['truth'],index=False)
    paths['costs'] = str(lab_dir / 'costs.parquet')
    pd.DataFrame({'query_key':universe.query_key.unique(),'status':'unknown'}).to_parquet(paths['costs'],index=False)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'deployment_threshold':.75}))
    inputs = tmp_path / 'inputs.json'
    inputs.write_text(json.dumps(paths))
    out = tmp_path / 'out'
    main(['freeze','--config',str(config),'--inputs',str(inputs),'--out',str(out)])
    argv = ['evaluate','--config',str(config),'--out',str(out)]
    for name,path in paths.items():
        argv.extend(['--'+name,path])
    with pytest.raises(ValueError,match='conflicts with complete construction truth'):
        main(argv)
    assert not (out / 'metrics.json').exists()


def test_curve_parquet_preserves_abstention_thresholds(tmp_path, evaluation_fixture):
    evaluation_fixture['truth'].loc[2:, 'truth_tier'] = 'verified_hub'
    evaluation_fixture['predictions'] = evaluation_fixture['predictions'].iloc[:2]
    out = evaluate(**evaluation_fixture)
    path = tmp_path / 'curves.parquet'
    out['curves'].to_parquet(path,index=False)
    curves = pd.read_parquet(path)
    diagnostic = curves.query("estimand == 'parent' and truth_tier == 'verified_hub'")
    assert diagnostic.threshold_at_tpr90.isna().all()
    assert diagnostic.diagnostic_tpr90_status.eq('unattainable').all()
    assert diagnostic.ranking_extension_threshold_at_tpr90.eq(float('-inf')).all()


def test_root_and_merge_root_fields_share_holdout_namespace(evaluation_fixture):
    truth = evaluation_fixture['truth']
    truth['root_group'] = ['A', 'A', 'B', 'B']
    truth['merge_roots'] = [[], [], ['A', 'C'], ['A', 'C']]
    truth.loc[:1, 'split'] = 'development'
    with pytest.raises(ValueError, match='root.*crosses evaluation splits'):
        evaluate(**evaluation_fixture)
    # Duplicate IDs have their own namespace and can equal another split's root.
    truth['merge_roots'] = [[], [], [], []]
    truth['duplicate_group'] = ['B', 'B', 'A', 'A']
    evaluate(**evaluation_fixture)


@pytest.mark.parametrize('changed_module', ['lineage.baseline', 'lineage.cli', 'lab.score',
                                           'lab.registry', 'lab.schema', 'zoo.schema'])
def test_freeze_detects_changed_scorer_dependency(tmp_path, monkeypatch, changed_module):
    import importlib.util
    from types import SimpleNamespace
    from lineage.evaluate import verify_freeze
    original_find = importlib.util.find_spec
    original_spec = original_find(changed_module)
    copied_source = tmp_path / 'scorer_dependency.py'
    copied_source.write_bytes(__import__('pathlib').Path(original_spec.origin).read_bytes())

    def find_spec(name, *args, **kwargs):
        if name == changed_module:
            return SimpleNamespace(origin=str(copied_source))
        return original_find(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'find_spec', find_spec)
    source = tmp_path / 'truth.json'
    source.write_text('{"label":1}')
    config = {'deployment_threshold': .75}
    out = tmp_path / 'frozen'
    freeze(config, {'truth': source}, out)
    copied_source.write_text(copied_source.read_text() + '\n# changed scorer dependency\n')
    with pytest.raises(ValueError, match='implementation changed'):
        verify_freeze(config, {'truth': source}, out)


def test_unattainable_tpr90_keeps_abstention_ranking_extension(evaluation_fixture):
    evaluation_fixture['predictions'] = evaluation_fixture['predictions'].iloc[:0]
    out = evaluate(**evaluation_fixture)
    diagnostic = out['by_estimand']['parent']['diagnostic_tpr90']
    assert diagnostic['status'] == 'unattainable'
    assert diagnostic['threshold'] is None
    assert diagnostic['tpr'] is None
    assert diagnostic['false_matches_per_query'] is None
    assert diagnostic['n_scored_positives'] == 0
    assert diagnostic['maximum_finite_tpr'] == 0
    extension = diagnostic['ranking_extension']
    assert extension['threshold'] == '-inf'
    assert extension['tpr'] == 1
    assert extension['false_matches'] == 2
    assert extension['false_matches_per_query'] == 1
    assert extension['n_queries'] == 2
    parent_curve = out['curves'].query("estimand == 'parent' and truth_tier == 'all'")
    assert parent_curve.diagnostic_tpr90_status.eq('unattainable').all()
    assert parent_curve.false_matches_per_query_at_tpr90.isna().all()
    assert parent_curve.query('pool_size == 4').ranking_extension_false_matches_per_query_at_tpr90.eq(1).all()


def test_freeze_relocation_and_missing_component_binding(tmp_path):
    from lineage.evaluate import verify_freeze
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    source = bundle / 'truth.json'
    source.write_text('{"label":1}')
    config = {'deployment_threshold': .75}
    out = bundle / 'frozen'
    freeze(config, {'truth':source}, out)
    saved = json.loads((out / 'input_hashes.json').read_text())
    assert saved['truth']['path'] == '../truth.json'
    moved = tmp_path / 'relocated'
    bundle.rename(moved)
    verify_freeze(config, {'truth':moved/'truth.json'}, moved/'frozen')
    protocol_path = moved / 'frozen' / 'protocol.json'
    legacy = json.loads(protocol_path.read_text())
    legacy.pop('scorer_components')
    protocol_path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match='lacks scorer component'):
        verify_freeze(config, {'truth':moved/'truth.json'}, moved/'frozen')
