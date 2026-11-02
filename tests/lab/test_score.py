"""Small, hand-calculated construction and external-truth scoring examples."""
import json

import pandas as pd
import pytest
from zoo.schema import LINEAGE_COLUMNS

from lab.schema import EDGE_COLUMNS, MODEL_COLUMNS
from lab.score import score_lab


def fixture_graph():
    models = pd.DataFrame([
        ['b', 'base', 'org/b@' + 'a'*40, 'tiny', 'mit'],
        ['c', 'derived', 'c', 'tiny', 'mit'],
        ['d', 'derived', 'd', 'tiny', 'mit'],
        ['i', 'base', 'org/i@' + 'b'*40, 'tiny', 'mit'],
    ], columns=MODEL_COLUMNS)
    edges = pd.DataFrame([
        ['c', 'b', 'sft', 'parent', '', '{}'],
        ['d', 'c', 'dpo', 'parent', '', '{}'],
    ], columns=EDGE_COLUMNS)
    manifests = [dict(repo_id='org/b', revision='a'*40),
                 dict(repo_id='org/i', revision='b'*40),
                 dict(repo_id='lab/c', revision='local:' + 'c'*64),
                 dict(repo_id='lab/d', revision='local:' + 'd'*64)]
    keys = {name: repo + '@' + rev for name, repo, rev in
            [('b', 'org/b', 'a'*40), ('i', 'org/i', 'b'*40),
             ('c', 'lab/c', 'local:'+'c'*64), ('d', 'lab/d', 'local:'+'d'*64)]}
    universe = pd.DataFrame([(keys[q], keys[p], True) for q,p in
                            [('d','b'), ('d','c'), ('d','i'), ('b','d'), ('c','d')]],
                           columns=['query_key', 'candidate_key', 'eligible'])
    predictions = []
    for estimand, values in [('parent', [.1, .9, .05, .1, .1]),
                             ('ancestor', [.8, .9, .05, .1, .1]),
                             ('same_root', [.9, .9, .05, .9, .9])]:
        for (q,p,_), value in zip(universe.itertuples(index=False, name=None), values):
            predictions.append([q,p,estimand,'finetune','fingerprint','{}',value,None,False,'test'])
    return models, edges, pd.DataFrame(predictions, columns=LINEAGE_COLUMNS), manifests, universe


def metric(result, estimand='parent', tier='lab_exact'):
    return result['tiers'][tier][estimand]


def test_nested_truth_and_direction_use_strict_ancestry():
    args = fixture_graph()
    result = score_lab(*args)
    for estimand, positives in [('parent', 1), ('ancestor', 2), ('same_root', 4)]:
        got = metric(result, estimand)
        assert got['auroc'] == 1
        assert got['n_positive'] == positives
        assert got['false_matches_per_query_at_tpr90'] == 0
        assert got['n_queries'] == 3
    assert metric(result)['direction_accuracy'] == 1
    assert metric(result, 'ancestor')['direction_accuracy'] == 1
    assert 'direction_accuracy' not in metric(result, 'same_root')
    predictions = args[2].copy()
    selected = (predictions.estimand == 'parent') & predictions.parent_key.str.startswith('lab/c@')
    reverse = (predictions.estimand == 'parent') & predictions.child_key.str.startswith('lab/c@')
    predictions.loc[selected, 'score'] = .1
    predictions.loc[reverse, 'score'] = .9
    assert metric(score_lab(args[0], args[1], predictions, args[3], args[4]))['direction_accuracy'] == 0


def test_missing_positives_and_ineligible_rows_stay_in_denominators():
    models, edges, predictions, manifests, universe = fixture_graph()
    predictions = predictions[~((predictions.estimand == 'parent') &
                                predictions.parent_key.str.startswith('lab/c@'))]
    universe.loc[2, 'eligible'] = False
    got = metric(score_lab(models, edges, predictions, manifests, universe, threshold=.5))
    assert got['n_positive'] == 1
    assert got['n_negative'] == 4
    assert got['n_queries'] == 3
    assert got['eligible_fraction'] == .8
    assert got['scored_fraction'] == .6
    assert got['abstention_fraction'] == .4
    assert got['positive_abstentions'] == 1
    assert got['threshold_at_tpr90'] == '-inf'
    assert got['attained_tpr90'] == 1
    assert got['false_matches_per_query_at_tpr90'] == pytest.approx(4/3)
    assert got['frozen_threshold']['tpr'] == 0
    assert got['direction_availability'] == 0


def test_midrank_ties_and_undefined_classes_are_json_safe():
    models, edges, predictions, manifests, universe = fixture_graph()
    predictions['score'] = float('-inf')
    result = score_lab(models, edges, predictions, manifests, universe)
    got = metric(result)
    assert got['auroc'] == .5
    assert got['false_matches_per_query_at_tpr90'] == pytest.approx(4/3)
    assert got['scored_fraction'] == 0
    json.dumps(result, allow_nan=False)
    empty = metric(score_lab(models, edges, predictions, manifests, universe.iloc[[2]]))
    assert empty['auroc'] is None
    assert empty['status'] == 'undefined_no_positives'
    assert empty['attained_tpr90'] is None
    positive_only = metric(score_lab(models, edges, predictions, manifests, universe.iloc[[1]]))
    assert positive_only['auroc'] is None
    assert positive_only['status'] == 'undefined_no_negatives'
    assert positive_only['n_queries'] == 1


def test_external_absent_edges_remain_unknown_and_explicit_tiers_are_separate():
    models, edges, predictions, manifests, universe = fixture_graph()
    models.loc[models.model_id == 'i', 'kind'] = 'independent'
    result = score_lab(models, edges, predictions, manifests, universe)
    assert metric(result)['n_unknown'] == 1
    assert metric(result)['n_negative'] == 3
    assert metric(result)['n_pairs'] == 5
    universe['truth_tier'] = pd.NA
    universe.loc[2, ['truth_tier','truth_parent','truth_ancestor','truth_same_root']] = ['verified_hub', False, False, True]
    result = score_lab(models, edges, predictions, manifests, universe)
    assert metric(result)['n_pairs'] == 4
    assert metric(result, 'same_root', 'verified_hub')['n_positive'] == 1
    assert metric(result, 'ancestor', 'verified_hub')['n_negative'] == 1
    assert metric(result, 'parent', 'declared')['n_pairs'] == 0


def test_merge_base_context_and_root_intersection_are_not_transitive():
    models, edges, predictions, manifests, universe = fixture_graph()
    models.loc[len(models)] = ['m','derived','m','tiny','mit']
    edges.loc[len(edges)] = ['m','c','merge','parent','','{}']
    edges.loc[len(edges)] = ['m','i','merge','base','','{}']
    manifests.append(dict(repo_id='lab/m',revision='local:'+'e'*64))
    m = 'lab/m@local:'+'e'*64
    b, i = manifests[0]['repo_id']+'@'+'a'*40, manifests[1]['repo_id']+'@'+'b'*40
    universe = pd.DataFrame([(m,i,True),(m,b,True),(b,i,True)], columns=universe.columns)
    result = score_lab(models, edges, predictions.iloc[:0], manifests, universe)
    assert metric(result)['n_positive'] == 0
    assert metric(result, 'ancestor')['n_positive'] == 2
    assert metric(result, 'same_root')['n_positive'] == 2
    assert metric(result, 'same_root')['n_negative'] == 1


@pytest.mark.parametrize('fault', ['universe_duplicate', 'conflicting_prediction', 'conflicting_truth',
                                  'nonboolean_label', 'missing_manifest', 'duplicate_manifest',
                                  'path_guess', 'nan_score', 'extra_prediction_column'])
def test_rejects_ambiguous_or_invalid_evidence(fault):
    models, edges, predictions, manifests, universe = fixture_graph()
    if fault == 'universe_duplicate':
        universe = pd.concat([universe, universe.iloc[[0]]])
    elif fault == 'conflicting_prediction':
        duplicate = predictions.iloc[[0]].copy()
        duplicate['score'] = .4
        predictions = pd.concat([predictions, duplicate])
    elif fault == 'conflicting_truth':
        universe['truth_parent'] = pd.NA
        universe.loc[1, 'truth_parent'] = False
        universe['truth_tier'] = 'lab_exact'
    elif fault == 'nonboolean_label':
        universe['truth_parent'] = 'false'
    elif fault == 'missing_manifest':
        manifests.pop()
    elif fault == 'duplicate_manifest':
        manifests.append(manifests[-1].copy())
    elif fault == 'path_guess':
        manifests[-1]['repo_id'] = '/tmp/d'
    elif fault == 'nan_score':
        predictions.loc[0, 'score'] = float('nan')
    else:
        predictions['eligible'] = True
    with pytest.raises(ValueError):
        score_lab(models, edges, predictions, manifests, universe)


def test_agreeing_claims_collapse_and_type_accuracy_excludes_laundering():
    models, edges, predictions, manifests, universe = fixture_graph()
    edges.loc[edges.child == 'd', 'edge_type'] = 'laundering'
    predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    result = score_lab(models, edges, predictions, manifests, universe)
    assert metric(result)['n_pairs'] == 5
    assert result['type_accuracy']['n_examples'] == 0
    # Include c->b so sft->finetune is tested while laundering remains excluded.
    universe.loc[len(universe)] = [manifests[2]['repo_id']+'@'+manifests[2]['revision'],
                                   manifests[0]['repo_id']+'@'+manifests[0]['revision'], True]
    row = predictions.iloc[0].copy()
    row[['child_key','parent_key','score']] = [universe.iloc[-1].query_key, universe.iloc[-1].candidate_key,.9]
    predictions.loc[len(predictions)] = row
    result = score_lab(models, edges, predictions, manifests, universe)
    assert result['type_accuracy']['n_examples'] == 1
    assert result['type_accuracy']['accuracy'] == 1


def test_unlisted_prediction_pairs_cannot_improve_results():
    models, edges, predictions, manifests, universe = fixture_graph()
    baseline = score_lab(models, edges, predictions, manifests, universe)
    extra = predictions.iloc[[0]].copy()
    extra['child_key'] = manifests[2]['repo_id']+'@'+manifests[2]['revision']
    extra['parent_key'] = manifests[0]['repo_id']+'@'+manifests[0]['revision']
    extra['score'] = 1000
    assert score_lab(models, edges, pd.concat([predictions, extra]), manifests, universe) == baseline


@pytest.mark.parametrize('eligible', ['False', 0, None])
def test_eligibility_is_strictly_boolean(eligible):
    args = list(fixture_graph())
    args[-1]['eligible'] = args[-1]['eligible'].astype(object)
    args[-1].loc[0, 'eligible'] = eligible
    with pytest.raises(ValueError, match='boolean'):
        score_lab(*args)


def test_external_labels_cannot_contradict_nested_relations():
    args = list(fixture_graph())
    args[0].loc[args[0].model_id == 'i','kind'] = 'independent'
    args[-1] = args[-1].iloc[[2]].copy()
    args[-1]['truth_tier'] = 'declared'
    args[-1]['truth_parent'] = True
    args[-1]['truth_ancestor'] = False
    with pytest.raises(ValueError, match='conflict'):
        score_lab(*args)


def test_score_cli_writes_atomic_strict_json_and_preserves_output_on_error(tmp_path):
    from lab.cli import main
    models, edges, predictions, manifests, universe = fixture_graph()
    models.to_csv(tmp_path/'models.csv', index=False)
    edges.to_csv(tmp_path/'edges.csv', index=False)
    predictions.to_parquet(tmp_path/'predictions.parquet', index=False)
    universe.to_parquet(tmp_path/'universe.parquet', index=False)
    directory = tmp_path/'manifests'
    directory.mkdir()
    for index, manifest in enumerate(manifests):
        (directory/f'{index}.json').write_text(json.dumps(manifest))
    config = tmp_path/'config.json'
    config.write_text('{"schema_version":"1","threshold":0.5}')
    output = tmp_path/'scores.json'
    argv = ['score','--models',str(tmp_path/'models.csv'),'--edges',str(tmp_path/'edges.csv'),
            '--predictions',str(tmp_path/'predictions.parquet'),'--manifests',str(directory),
            '--universe',str(tmp_path/'universe.parquet'),'--config',str(config),'--output',str(output)]
    assert main(argv) == 0
    result = json.loads(output.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert metric(result)['frozen_threshold']['threshold'] == .5
    previous = output.read_bytes()
    config.write_text('{"schema_version":"1","threshold":NaN}')
    with pytest.raises(SystemExit):
        main(argv)
    assert output.read_bytes() == previous
    assert not list(tmp_path.glob('.scores.json-*'))
