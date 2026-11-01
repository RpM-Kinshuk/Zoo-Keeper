"""Numerical contracts for known-noise familial estimation."""

import numpy as np
import pandas as pd
import pytest

from studies.heritability.statistics import (
    cluster_bootstrap, fit_reml, parent_offspring, prepare_scale, reml_nll,
    skill_summary,
)


def test_reml_matches_direct_restricted_gaussian_expression():
    y = np.array([.2, .6, .3, .9])
    x = np.column_stack([np.ones(4), [-1., 0., 1., 2.]])
    v = np.array([[2., .2, .1, 0.], [.2, 1., .3, 0.], [.1, .3, 1.5, .1], [0., 0., .1, 1.]])
    inv = np.linalg.inv(v)
    p = inv - inv @ x @ np.linalg.inv(x.T @ inv @ x) @ x.T @ inv
    expected = .5 * (np.linalg.slogdet(v)[1] + np.linalg.slogdet(x.T @ inv @ x)[1] + y @ p @ y + 2*np.log(2*np.pi))
    assert reml_nll(y, x, v) == pytest.approx(expected, abs=1e-10)


def test_reml_recovers_40_by_5_synthetic_variances():
    rng = np.random.default_rng(20261028)
    z = np.repeat(np.eye(40), 5, axis=0)
    # The plan specifies two draws; D decomposes the observed residual .01
    # into known noise .002 plus latent residual .008.
    y = .5 + z @ rng.normal(0, np.sqrt(.02), 40) + rng.normal(0, np.sqrt(.01), 200)
    result = fit_reml(y, np.ones((200, 1)), z @ z.T, np.full(200, .002))
    assert result['converged']
    assert result['tau2'] >= 0 and result['sigma2'] >= 0
    assert result['familial_fraction'] == pytest.approx(2/3, abs=.15)
    assert result['n'] == 200 and result['families'] == 40


def test_separate_measurement_draw_matches_balanced_analytic_reml():
    rng = np.random.default_rng(20261028)
    z = np.repeat(np.eye(40), 5, axis=0)
    y = .5 + z @ rng.normal(0, np.sqrt(.02), 40) + rng.normal(0, np.sqrt(.01), 200) + rng.normal(0, np.sqrt(.002), 200)
    result = fit_reml(y, np.ones((200, 1)), z @ z.T, np.full(200, .002))
    # Independent balanced-ANOVA solutions for this exact realized sample.
    assert result['tau2'] == pytest.approx(.012104385174333867, abs=1e-7)
    assert result['sigma2'] == pytest.approx(.011918337295718026, abs=1e-7)
    assert result['measurement_variance_fraction'] == pytest.approx(.0768559117, abs=1e-6)
    assert result['noise_ceiling'] == pytest.approx(1-result['measurement_variance_fraction'])


def test_zero_family_variance_selects_boundary_and_accepts_matrix_d():
    z = np.repeat(np.eye(6), 4, axis=0)
    y = np.tile([-.1, .1, -.1, .1], 6)
    result = fit_reml(y, np.ones((24, 1)), z @ z.T, np.eye(24)*.001)
    assert result['converged'] and result['boundary']
    assert result['tau2'] == 0


def test_kernel_normalization_preserves_latent_fraction():
    kernel = np.kron(np.eye(4), np.ones((4, 4)))
    y = np.repeat([.2, .4, .6, .8], 4) + np.tile([-.03, .03, -.02, .02], 4)
    a = fit_reml(y, np.ones((16, 1)), kernel, np.full(16, .0001))
    b = fit_reml(y, np.ones((16, 1)), kernel*7, np.full(16, .0001))
    assert a['familial_fraction'] == pytest.approx(b['familial_fraction'], abs=1e-6)
    assert b['kernel_normalization'] == 7


def test_tiny_positive_kernel_scaling_preserves_identifiability_and_components():
    kernel = np.kron(np.eye(4), np.ones((4, 4)))
    y = np.repeat([.2, .4, .6, .8], 4) + np.tile([-.03, .03, -.02, .02], 4)
    reference = fit_reml(y, np.ones((16, 1)), kernel, np.full(16, .0001))
    scaled = fit_reml(y, np.ones((16, 1)), kernel*1e-12, np.full(16, .0001))
    assert scaled['converged']
    assert scaled['families'] == reference['families']
    assert scaled['familial_fraction'] == pytest.approx(reference['familial_fraction'], abs=1e-6)
    assert scaled['tau2'] == pytest.approx(reference['tau2'], abs=1e-7)
    assert scaled['sigma2'] == pytest.approx(reference['sigma2'], abs=1e-7)
    assert scaled['kernel_normalization'] == pytest.approx(1e-12, rel=1e-12, abs=0)


@pytest.mark.parametrize('d', [np.ones(3), np.array([.1, .1, -.1, .1]), np.eye(4)*-.1, np.ones((4, 4))*.1])
def test_invalid_known_noise_rejected(d):
    with pytest.raises(ValueError, match='measurement'):
        fit_reml(np.arange(4.), np.ones((4, 1)), np.eye(4), d)


def test_singular_design_rejected_and_insufficient_residual_support_unavailable():
    with pytest.raises(ValueError, match='rank'):
        reml_nll(np.arange(4.), np.ones((4, 2)), np.eye(4))
    result = fit_reml(np.arange(2.), np.eye(2), np.eye(2), np.ones(2)*.01)
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'insufficient_residual_support'


@pytest.mark.parametrize('kernel', [np.ones((5, 5)), np.eye(5), np.zeros((5, 5))])
def test_nonidentifiable_kernels_are_unavailable(kernel):
    result = fit_reml(np.arange(5.), np.ones((5, 1)), kernel, np.ones(5)*.01)
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'nonidentifiable_kernel'
    assert result['familial_fraction'] is None


def test_parent_measurement_correction():
    pairs = pd.DataFrame({'parent_accuracy':[.2,.4,.6,.8],
                          'child_accuracy':[.3,.4,.5,.6],
                          'parent_variance':[.01]*4,
                          'error_covariance':[0.]*4})
    result = parent_offspring(pairs)
    assert result['naive_slope'] == pytest.approx(.5)
    assert result['corrected_slope'] == pytest.approx(.625)
    assert result['variance_denominator'] == pytest.approx(.04)
    assert result['naive_intercept'] == pytest.approx(.2)
    assert result['corrected_intercept'] == pytest.approx(.1375)


def test_parent_paired_error_correction_and_missing_covariance_assumption():
    pairs = pd.DataFrame(dict(parent_accuracy=[.2,.4,.6,.8], child_accuracy=[.3,.4,.5,.6], parent_variance=[.01]*4, error_covariance=[.005]*4))
    assert parent_offspring(pairs)['corrected_slope'] == pytest.approx(.5)
    pairs.loc[0, 'error_covariance'] = np.nan
    result = parent_offspring(pairs)
    assert result['error_assumption'] == 'independent_parent_child_errors'
    assert result['corrected_slope'] == pytest.approx(.625)
    assert result['error_covariance_available'] == 3


def test_parent_unavailable_denominator_and_counted_missing_variances():
    pairs = pd.DataFrame(dict(parent_accuracy=[.2,.4,.6,.8], child_accuracy=[.3,.4,.5,.6], parent_variance=[.1,.1,np.nan,.1]))
    result = parent_offspring(pairs)
    assert result['naive_slope'] == pytest.approx(.5)
    assert result['corrected_slope'] is None
    assert result['reason'] == 'nonpositive_corrected_denominator'
    assert result['n_corrected'] == 3 and result['excluded_variance'] == 1


def test_binomial_logit_endpoints_are_finite_and_composite_not_fabricated():
    scores = pd.DataFrame(dict(record_id=['a','b','c'], accuracy=[0.,1.,.4], n_items=[10,10,10], measurement_variance=[0.,0.,.02], variance_source=['binomial:proof','binomial:proof','documented:proof']))
    retained, exclusions = prepare_scale(scores, 'logit')
    assert list(retained.record_id) == ['a','b']
    assert retained.accuracy.to_numpy() == pytest.approx([-np.log(21), np.log(21)])
    assert retained.measurement_variance.to_numpy() == pytest.approx([484/231, 484/231])
    assert retained.correction.eq('binomial_jeffreys_half_count').all()
    assert list(exclusions.reason) == ['logit_requires_documented_binomial_counts']


def test_raw_scale_excludes_and_counts_unavailable_variance():
    scores = pd.DataFrame(dict(record_id=['a','b','c'], accuracy=[.2,.3,1.2], measurement_variance=[.01,np.nan,.01]))
    retained, excluded = prepare_scale(scores)
    assert list(retained.record_id) == ['a']
    assert list(excluded.reason) == ['variance_unavailable','invalid_accuracy']


def test_constant_target_scale_free_errors_unavailable_and_negative_r2_preserved():
    constant = skill_summary(np.ones(3), np.array([0.,1.,2.]))
    assert constant['rmse'] == pytest.approx(np.sqrt(2/3))
    assert constant['r2'] is None and constant['mae_sd'] is None
    assert constant['reason'] == 'constant_target'
    result = skill_summary(np.array([0.,1.]), np.array([2.,2.]))
    assert result['r2'] == -9
    assert result['mae_sd'] == 3


def test_skill_summary_reports_pairwise_excluded_support():
    result = skill_summary(np.array([0.,1.,np.nan]), np.array([0.,2.,.5]))
    assert result['n'] == 2 and result['excluded'] == 1
    assert result['mae'] == .5


def test_cluster_bootstrap_is_deterministic_and_resamples_whole_groups():
    frame = pd.DataFrame(dict(split_group=['a','a','b','b','c','c'], value=[0.,0.,1.,1.,2.,2.]))
    def statistic(sample):
        assert all(size % 2 == 0 for size in sample.groupby('split_group').size())
        return {'mean': sample.value.mean()}
    a = cluster_bootstrap(frame, statistic, n_bootstrap=20, seed=17)
    b = cluster_bootstrap(frame, statistic, n_bootstrap=20, seed=17)
    assert a == b
    assert a['n_successful'] == 20 and a['n_failed'] == 0
    assert a['intervals']['mean'][0] <= 1 <= a['intervals']['mean'][1]


def test_bootstrap_counts_failures_and_withholds_interval_below_90_percent():
    frame = pd.DataFrame(dict(split_group=['a','b','c'], value=[0.,1.,2.]))
    def statistic(sample):
        return {'slope': None if sample.value.nunique() < 3 else sample.value.mean()}
    result = cluster_bootstrap(frame, statistic, n_bootstrap=20, seed=17)
    assert result['n_failed'] > 2
    assert result['intervals']['slope'] is None
    assert result['failure_rate'] > .1
    with pytest.raises(ValueError, match='split_group'):
        cluster_bootstrap(frame.drop(columns='split_group'), statistic, n_bootstrap=20)


def test_bootstrap_one_cluster_does_not_invent_precision():
    frame = pd.DataFrame(dict(split_group=['a','a'], value=[0.,1.]))
    result = cluster_bootstrap(frame, lambda sample: {'mean': sample.value.mean()}, n_bootstrap=20)
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'insufficient_clusters'


def test_empty_parent_pairs_are_unavailable_without_numeric_dtype_assumptions():
    pairs = pd.DataFrame(columns=['parent_accuracy', 'child_accuracy', 'parent_variance'])
    result = parent_offspring(pairs)
    assert result['status'] == 'unavailable'
    assert result['reason'] == 'insufficient_pairs'
    assert result['n'] == 0 and result['n_corrected'] == 0
    assert result['corrected_slope'] is None


def test_cluster_bootstrap_labels_independent_draws_of_repeated_source_clusters():
    frame = pd.DataFrame(dict(split_group=['a','a','b','b'], value=[0.,0.,1.,1.]))
    duplicates = []
    def statistic(sample):
        assert '_bootstrap_draw' in sample
        assert sample._bootstrap_draw.nunique() == 2
        for _, draw in sample.groupby('_bootstrap_draw'):
            assert len(draw) == 2 and draw.split_group.nunique() == 1
        duplicates.append(sample.split_group.nunique() < 2)
        return {'mean':sample.value.mean()}
    result = cluster_bootstrap(frame, statistic, n_bootstrap=20, seed=17)
    assert result['n_successful'] == 20
    assert any(duplicates)
