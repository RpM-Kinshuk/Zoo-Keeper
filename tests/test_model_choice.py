import pandas as pd
import pytest


def fixture():
    scores = pd.DataFrame({'held': [.1, .9, .6], 'other1': [.9, .1, .5], 'other2': [.8, .2, .5]}, index=['a', 'b', 'c'])
    metadata = pd.DataFrame({'reported_params_b': [1., 3., 2.], 'historical_downloads': [10., 20., 30.]}, index=scores.index)
    return scores, metadata


def test_target_column_never_changes_policy_choices():
    from scripts.model_choice import evaluate_scenario
    scores, metadata = fixture()
    first = evaluate_scenario(scores, metadata, 'held', None)
    scores['held'] = [.99, .01, .02]
    second = evaluate_scenario(scores, metadata, 'held', None)
    assert {k:v['selected'] for k,v in first['policies'].items()} == {k:v['selected'] for k,v in second['policies'].items()}
    assert first['policies']['other_scores']['selected'] == 'a'
    assert first['policies']['largest']['selected'] == 'b'
    assert first['policies']['downloads']['selected'] == 'c'
    assert first['policies']['other_scores']['regret'] != second['policies']['other_scores']['regret']


def test_budget_is_shared_and_uniform_regret_is_exact():
    from scripts.model_choice import evaluate_scenario
    scores, metadata = fixture()
    result = evaluate_scenario(scores, metadata, 'held', 2.)
    assert result['candidates'] == ['a', 'c']
    assert all(v['selected'] in result['candidates'] for v in result['policies'].values())
    assert result['uniform']['expected_regret'] == pytest.approx(.25)
    assert result['policies']['largest']['regret'] == 0
    assert evaluate_scenario(scores, metadata, 'held', 1.)['status'] == 'insufficient_candidates'


def test_ties_are_stable_and_missing_target_is_not_imputed():
    from scripts.model_choice import evaluate_scenario
    scores, metadata = fixture()
    scores.loc[:, ['other1','other2']] = .5
    metadata[:] = 1.
    result = evaluate_scenario(scores.iloc[::-1], metadata.iloc[::-1], 'held', None)
    assert all(v['selected'] == 'a' for v in result['policies'].values())
    scores.loc['a','held'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        evaluate_scenario(scores, metadata, 'held', None)
