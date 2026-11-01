import numpy as np
import pandas as pd
import pytest

from studies.heritability.exchange import exchange_rate, selection_pairs, exchange_table


def test_strict_and_measured_exchange():
    r = exchange_rate(np.array([.5,.5]), np.array([.6,.4]), np.array([.25,.25]))
    assert r['rmse'] == pytest.approx(.1)
    assert r['k'] == pytest.approx(25.)
    assert r['strict_mae_upper'] == pytest.approx(25.)
    assert r['k'] <= r['strict_mae_upper'] + 1e-10


@pytest.fixture
def pair_predictions():
    return pd.DataFrame({'model_key':['a@'+'a'*40,'b@'+'b'*40], 'actual':[.4,.6],
                         'prediction':[.5,.7], 'root_group':['r','r'], 'duplicate_group':['a','b'],
                         'split_group':['r','r'], 'method':['weight']*2, 'task':['math']*2,'fold':['0']*2})


def test_selection_cancels_shared_error(pair_predictions):
    pairs = selection_pairs(pair_predictions)
    assert len(pairs) == 1
    assert pairs.iloc[0].residual_difference == pytest.approx(0.)
    assert pairs.iloc[0].independent_item_variance == pytest.approx(.48)


def test_pair_item_units_and_duplicate_exclusion(pair_predictions):
    pair_predictions.loc[1,'prediction'] = .8
    pair = selection_pairs(pair_predictions).iloc[0]
    result = exchange_rate(np.array([0.]),np.array([pair.residual_difference]),np.array([pair.independent_item_variance]),pairwise=True)
    assert result['k'] == pytest.approx(48.)
    assert result['total_items'] == pytest.approx(96.)
    pair_predictions['duplicate_group']='same'
    assert selection_pairs(pair_predictions).empty


def test_heavy_tails_perfect_prediction_and_out_of_range():
    r=exchange_rate(np.array([.5]*4),np.array([.5,.5,.5,1.5]),np.array([.25]*4))
    assert r['rmse'] > r['mae']
    assert r['k'] <= r['strict_mae_upper']
    assert r['out_of_range_predictions'] == 1
    r=exchange_rate(np.array([.5]),np.array([.5]),np.array([.25]))
    assert r['status'] == 'perfect_on_sample'
    assert r['k'] is None
    assert r['strict_mae_upper'] is None
    assert r['mae_bound_status'] == 'unbounded'


def test_pair_sampling_is_deterministic_and_reused_for_methods():
    n=80
    rows=pd.DataFrame({'model_key':[f'm{i}@'+ 'a'*40 for i in range(n)],'actual':np.linspace(.1,.9,n),
                       'prediction':np.linspace(.15,.95,n),'root_group':[f'r{i%4}' for i in range(n)],
                       'split_group':[f'r{i%4}' for i in range(n)],'duplicate_group':[f'd{i}' for i in range(n)],
                       'method':['weight']*n,'task':['math']*n,'fold':['0']*n})
    pairs=selection_pairs(rows,max_pairs=10)
    again=selection_pairs(rows.sample(frac=1,random_state=1),max_pairs=10)
    pd.testing.assert_frame_equal(pairs,again)
    assert pairs.groupby('family_stratum').size().max() <= 10
    other=selection_pairs(rows.assign(method='no_weight'),max_pairs=10)
    assert list(zip(pairs.model_i,pairs.model_j)) == list(zip(other.model_i,other.model_j))


def test_negative_increment_and_missing_baseline_support(pair_predictions):
    weighted=pair_predictions.assign(prediction=[.1,.9],revision_verified=True,provenance_status='eligible',
                                       prediction_scale='raw_accuracy',benchmark='v2',harness='h1',precision='bf16',quantization='none',setting='root_holdout')
    baseline=weighted.assign(method='no_weight',prediction=[.45,.55])
    scores=weighted.rename(columns={'actual':'accuracy'}).assign(kind='binary')
    table=exchange_table(pd.concat([weighted,baseline],ignore_index=True),scores,n_bootstrap=20)
    row=table.query("estimand == 'estimation' and method == 'weight'").iloc[0]
    assert row.incremental_k < 0
    short=exchange_table(pd.concat([weighted,baseline.iloc[:1]],ignore_index=True),scores,n_bootstrap=20)
    assert 'baseline_support_mismatch' in set(short.reason)


def test_composite_item_equivalence_is_unavailable(pair_predictions):
    rows=pair_predictions.assign(revision_verified=True,provenance_status='eligible',prediction_scale='raw_accuracy')
    scores=rows.rename(columns={'actual':'accuracy'}).assign(kind='composite')
    table=exchange_table(rows,scores,n_bootstrap=20)
    assert table.k.isna().all()
    assert 'item_equivalence_unavailable' in set(table.reason)


def test_supplied_variance_and_pairwise_universal_units():
    rate = exchange_rate(np.array([.5]), np.array([.6]), np.array([.1]))
    assert rate['k'] == pytest.approx(10)
    rate = exchange_rate(np.array([0.]), np.array([.1]), np.array([.48]), pairwise=True)
    assert rate['universal_numerator_bound'] == .5
    assert rate['universal_mae_upper'] == pytest.approx(50)
    assert rate['strict_mae_upper'] == pytest.approx(48)
    assert rate['adaptive_factor_2'] == pytest.approx(24)
    tiny = exchange_rate(np.array([0.]), np.array([1e-17]), np.array([0.]))
    assert tiny['status'] == 'ok'
    assert tiny['k'] == 0


def test_documented_shared_item_covariance(pair_predictions):
    rows = pair_predictions.assign(actual=.5, kind='binary')
    rows['item_responses'] = [{'a': 1, 'b': 1, 'c': 0, 'd': 0}, {'a': 0, 'b': 0, 'c': 1, 'd': 1}]
    pair = selection_pairs(rows).iloc[0]
    assert pair.item_covariance == pytest.approx(-.25)
    assert pair.item_variance == pytest.approx(1.)
    assert pair.item_assumption == 'documented_shared_binary_item_responses'


def test_unknown_kind_does_not_authorize_binary_equivalence(pair_predictions):
    rows = pair_predictions.assign(prediction_scale='raw_accuracy')
    scores = rows.rename(columns={'actual': 'accuracy'})
    table = exchange_table(rows, scores, n_bootstrap=20)
    assert table.k.isna().all()
    assert 'item_equivalence_unavailable' in set(table.reason)


def test_pair_rank_index_matches_exhaustive_eligible_pairs():
    from itertools import combinations
    rows = pd.DataFrame({'model_key': list('abcdefghi'), 'actual': np.linspace(.1,.9,9), 'prediction': np.linspace(.12,.92,9),
                         'root_group': ['r1','r1','r2','r2','r3','r3','','','r4'],
                         'split_group': ['s1','s1','s1','s2','s3','s3','s4','',''],
                         'duplicate_group': ['d1','d2','d3','d3','d4','d5','d1','d6','d7'],
                         'method': 'weight', 'task': 'math', 'fold': '0'})
    expected = {}
    for a,b in combinations(rows.to_dict('records'),2):
        if a['duplicate_group'] == b['duplicate_group']:
            continue
        if not a['root_group'] or not b['root_group']:
            family = 'ancestry_unavailable'
        elif a['root_group'] == b['root_group']:
            family = 'within_family'
        elif a['split_group'] and b['split_group'] and a['split_group'] != b['split_group']:
            family = 'across_family'
        else:
            family = 'ancestry_ambiguous'
        expected[(a['model_key'],b['model_key'])] = family
    pairs = selection_pairs(rows)
    assert {(r.model_i,r.model_j): r.family_stratum for r in pairs.itertuples()} == expected


def test_whole_source_cluster_bootstrap_and_score_verification():
    rows = pd.DataFrame({'model_key': [f'm{i}@'+ 'a'*40 for i in range(6)], 'actual': np.linspace(.2,.7,6),
                         'prediction': [.22,.33,.43,.56,.67,.79], 'root_group': ['r0','r0','r1','r1','r2','r2'],
                         'split_group': ['r0','r0','r1','r1','r2','r2'], 'duplicate_group': list('abcdef'),
                         'method': 'weight', 'task': 'math', 'fold': '0', 'benchmark': 'v2', 'harness': 'h1',
                         'precision': 'bf16', 'quantization': 'none', 'setting': 'root_holdout',
                         'revision_verified': True, 'provenance_status': 'eligible', 'prediction_scale': 'raw_accuracy'})
    scores = rows.rename(columns={'actual':'accuracy'}).assign(kind='binary')
    first = exchange_table(rows, scores, n_bootstrap=20)
    second = exchange_table(rows.sample(frac=1,random_state=3), scores, n_bootstrap=20)
    pd.testing.assert_frame_equal(first, second)
    selection = first.query("estimand == 'selection'")
    assert set(selection.bootstrap_n_clusters) == {3}
    assert set(selection.bootstrap_pair_design) == {'source_split_group_product_multiplicities_on_fixed_uniform_pair_sample'}
    unverified = exchange_table(rows, scores.assign(revision_verified=False), n_bootstrap=20)
    assert set(unverified.analysis_tier) == {'historical_descriptive'}


def test_matched_comparison_rejects_conflicting_source_groups(pair_predictions):
    weighted = pair_predictions.assign(revision_verified=True, provenance_status='eligible', prediction_scale='raw_accuracy', benchmark='v2', harness='h1', precision='bf16', quantization='none', setting='root_holdout')
    baseline = weighted.assign(method='no_weight', split_group='wrong')
    scores = weighted.rename(columns={'actual':'accuracy'}).assign(kind='binary')
    table = exchange_table(pd.concat([weighted, baseline]), scores, n_bootstrap=20)
    assert 'matched_model_metadata_mismatch' in set(table.reason)
    assert table.k.isna().all()


def test_documented_composite_item_variance_is_honored(pair_predictions):
    rows = pair_predictions.assign(prediction_scale='raw_accuracy')
    scores = rows.rename(columns={'actual':'accuracy'}).assign(kind='composite', item_variance=.1, item_variance_source='design.json@sha256')
    table = exchange_table(rows, scores, n_bootstrap=20)
    estimation = table.query("estimand == 'estimation'").iloc[0]
    assert estimation.q == pytest.approx(.1)
    assert estimation.k == pytest.approx(10)


def test_source_bootstrap_repeats_internal_pair_blocks_once_per_draw():
    from studies.heritability.exchange import _selection_stat

    source = pd.DataFrame({'split_group': ['g0','g0','g1','g1','g2','g2']})
    sample = pd.concat([source.iloc[:2], source.iloc[:2], source.iloc[2:4]], ignore_index=True)
    pairs = pd.DataFrame({'method': 'weight', 'actual_difference': 0.,
                          'predicted_difference': [.1,.3,.7,.2], 'item_variance': .4,
                          'split_group_i': ['g0','g1','g2','g0'],
                          'split_group_j': ['g0','g1','g2','g1']})
    internal = _selection_stat(pairs.iloc[:3], source, sample, 'weight', None)
    # Multiplicities (2,1,0) repeat internal blocks in ratio 2:1, not 4:1.
    assert internal['mse'] == pytest.approx((2*.1**2+.3**2)/3)
    assert internal['k'] == pytest.approx(.4/internal['mse'])
    together = _selection_stat(pairs, source, sample, 'weight', None)
    # The cross-group block still receives m_g0*m_g1 = 2 comparisons.
    assert together['mse'] == pytest.approx((2*.1**2+.3**2+2*.2**2)/5)


@pytest.mark.parametrize('field', ['root_group', 'duplicate_group', 'split_group'])
def test_both_methods_cannot_override_matched_score_groups(pair_predictions, field):
    weighted = pair_predictions.assign(revision_verified=True, provenance_status='eligible', prediction_scale='raw_accuracy', benchmark='v2', harness='h1', precision='bf16', quantization='none', setting='root_holdout')
    baseline = weighted.assign(method='no_weight')
    scores = weighted.rename(columns={'actual':'accuracy'}).assign(kind='binary')
    both = pd.concat([weighted, baseline], ignore_index=True)
    both[field] = 'contradicts_audited_scores'
    table = exchange_table(both, scores, n_bootstrap=20)
    assert table.k.isna().all()
    assert not table.estimand.isin(['estimation', 'selection']).any()
    assert 'matched_model_metadata_mismatch' in set(table.reason)


def test_conflicting_matched_score_group_evidence_is_excluded(pair_predictions):
    rows = pair_predictions.assign(prediction_scale='raw_accuracy')
    scores = rows.rename(columns={'actual':'accuracy'}).assign(kind='binary')
    conflicting = pd.concat([scores, scores.assign(duplicate_group='different_audit', split_group='different_split')], ignore_index=True)
    table = exchange_table(rows, conflicting, n_bootstrap=20)
    assert table.k.isna().all()
    assert not table.estimand.isin(['estimation', 'selection']).any()
    assert 'conflicting_score_group_evidence' in set(table.reason)
