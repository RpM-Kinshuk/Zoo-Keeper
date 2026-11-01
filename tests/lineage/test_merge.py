"""Offline scientific recovery and abstention contracts for merge fits."""

from types import SimpleNamespace

import numpy as np
import pytest

from lineage.merge import fit_merge
from lineage.testing import SEED, make_record, replace_record


def fixture_for(*, depth=False, collinear=False):
    rng = np.random.default_rng(SEED)
    arrays = rng.normal(size=(3, 16, 64, 64)).astype(np.float32)
    if collinear:
        arrays[1] = arrays[0]
    parents = [make_record('lab/p%d@' % i + str(i + 1) * 40, arrays[i]) for i in range(2)]
    base = make_record('lab/base@' + 'b' * 40, arrays[2])
    weights = np.repeat([.2] * 4 + [.8] * 4, 2)[:, None, None] if depth else .7
    child = make_record('lab/child@' + 'c' * 40, weights * arrays[0] + .4 * arrays[1] - .2 * arrays[2])
    return SimpleNamespace(inputs=dict(child=child, parents=parents, base=base))


@pytest.fixture
def merge_fixture():
    return fixture_for()


@pytest.fixture
def collinear_merge():
    return fixture_for(collinear=True)


def test_unconstrained_coefficients_and_depth(merge_fixture):
    out = fit_merge(**merge_fixture.inputs)
    assert np.max(np.abs(np.asarray(out['groups']['all']['coefficients']) - [.7, .4, -.2])) < 1e-5
    assert abs(sum(out['groups']['all']['coefficients']) - .9) < 1e-5
    assert out['status'] == 'identified'
    assert out['relative_residual'] < 1e-6
    assert len(out['groups']) == 17  # one per tensor, plus all


def test_collinear_parents_are_not_attributed(collinear_merge):
    out = fit_merge(**collinear_merge.inputs)
    assert out['status'] == 'nonidentifiable'
    assert out['collinear_groups']
    summary = out['groups']['all']
    assert summary['coefficients'] is None
    assert summary['intervals'] is None
    attribution = summary['group_attributions'][0]
    assert attribution['members'] == [p.key for p in collinear_merge.inputs['parents']]
    assert abs(attribution['coefficient'] - 1.1) < 1e-5
    assert attribution['interval'] is not None


def test_depth_schedule_is_recovered():
    fixture = fixture_for(depth=True)
    names = fixture.inputs['child'].names
    out = fit_merge(**fixture.inputs, groups={'early': list(names[:8]), 'late': list(names[8:])})
    assert abs(out['groups']['early']['coefficients'][0] - .2) < .01
    assert abs(out['groups']['late']['coefficients'][0] - .8) < .01
    assert out['groups']['early']['n_layers'] == 4
    assert out['groups']['late']['n_layers'] == 4
    np.testing.assert_allclose(out['groups']['early']['intervals'][0], [.2, .2], atol=1e-5)
    np.testing.assert_allclose(out['groups']['late']['intervals'][0], [.8, .8], atol=1e-5)


def test_layer_bootstrap_is_deterministic():
    fixture = fixture_for(depth=True)
    first = fit_merge(**fixture.inputs, bootstrap=40)
    second = fit_merge(**fixture.inputs, bootstrap=40)
    assert first['groups']['all']['intervals'] == second['groups']['all']['intervals']
    low, high = first['groups']['all']['intervals'][0]
    assert low < .5 < high
    assert high - low > .2  # complete layers preserve depth heterogeneity


@pytest.mark.parametrize('method', ['ties', 'dare', 'slerp'])
def test_nonlinear_recipe_is_effective_not_configured(merge_fixture, method):
    inputs = dict(merge_fixture.inputs)
    child = inputs['child']
    inputs['child'] = replace_record(child, merge_recipe={'method': method, 'coefficients': [.6, .6, -.2]})
    out = fit_merge(**inputs, bootstrap=0)
    assert out['coefficient_kind'] == 'effective'
    assert out['relative_residual'] < 1e-6
    assert out['configured_comparison']['configured'] == [.6, .6, -.2]
    assert abs(out['configured_comparison']['difference'][0] - .1) < 1e-5


def test_duplicate_base_has_both_roles_and_one_column(merge_fixture):
    inputs = dict(merge_fixture.inputs)
    inputs['parents'] = inputs['parents'] + [inputs['base'], inputs['base']]
    out = fit_merge(**inputs, bootstrap=0)
    assert out['status'] == 'identified'
    assert len(out['regressors']) == 3
    assert out['regressors'][-1]['roles'] == ['parent', 'base']
    np.testing.assert_allclose(out['groups']['all']['coefficients'], [.7, .4, -.2], atol=1e-5)


def test_fewer_than_four_layers_has_no_intervals(merge_fixture):
    names = list(merge_fixture.inputs['child'].names[:6])
    out = fit_merge(**merge_fixture.inputs, groups={'short': names})
    assert out['groups']['short']['interval_status'] == 'insufficient_layers'
    assert out['groups']['short']['intervals'] is None


def test_near_collinearity_is_grouped_even_at_full_rank(merge_fixture):
    inputs = dict(merge_fixture.inputs)
    p0, p1 = inputs['parents']
    inputs['parents'] = [p0, replace_record(p1, sketches=p0.sketches + .001 * p1.sketches)]
    out = fit_merge(**inputs, bootstrap=0)
    assert out['collinear_groups'] == [[p0.key, p1.key]]
    assert out['status'] == 'nonidentifiable'
    assert out['groups']['all']['coefficients'] is None


def test_ill_conditioned_independent_columns_abstain(merge_fixture):
    inputs = dict(merge_fixture.inputs)
    p0, p1 = inputs['parents']
    inputs['parents'] = [p0, replace_record(p1, sketches=p1.sketches * 1e-7)]
    out = fit_merge(**inputs, bootstrap=0)
    assert out['condition_number'] > 1e6
    assert out['status'] == 'nonidentifiable'
    assert out['groups']['all']['intervals'] is None


def test_missing_and_mismatched_views_are_explained(merge_fixture):
    inputs = dict(merge_fixture.inputs)
    p0, p1 = inputs['parents']
    inputs['parents'] = [replace_record(p0, names=('missing.weight',) + p0.names[1:]), p1]
    out = fit_merge(**inputs, bootstrap=0)
    assert out['n_views'] == 15
    assert out['coverage']['fraction'] == 15 / 16
    assert any('missing_view' in reason for reason in out['coverage']['reasons'])
    inputs['parents'] = [replace_record(p0, tier='t2'), p1]
    out = fit_merge(**inputs)
    assert out['status'] == 'missing_data'
    assert out['n_views'] == 0
    assert out['groups']['all']['coefficients'] is None


def test_bootstrap_keeps_group_identity_with_opposite_columns(collinear_merge):
    inputs = dict(collinear_merge.inputs)
    p0, p1 = inputs['parents']
    inputs['parents'] = [p0, replace_record(p1, sketches=-2 * p0.sketches)]
    out = fit_merge(**inputs, bootstrap=25)
    attribution = out['groups']['all']['group_attributions'][0]
    assert attribution['members'] == [p0.key, p1.key]
    assert abs(attribution['coefficient'] - 1.1) < 1e-5
    assert attribution['interval'][0] < 1.10001
    assert attribution['interval'][1] > 1.09999


def test_unknown_group_names_and_invalid_bootstrap_are_rejected(merge_fixture):
    with pytest.raises(ValueError):
        fit_merge(**merge_fixture.inputs, groups={'typo': ['absent.weight']})
    with pytest.raises(ValueError):
        fit_merge(**merge_fixture.inputs, bootstrap=-1)


def test_missing_layers_do_not_become_contiguous_blocks(merge_fixture):
    inputs = dict(merge_fixture.inputs)
    names = tuple(f'layers.{2 * (i // 2)}.view{i % 2}.weight' for i in range(16))
    inputs['child'] = replace_record(inputs['child'], names=names)
    inputs['parents'] = [replace_record(parent, names=names) for parent in inputs['parents']]
    inputs['base'] = replace_record(inputs['base'], names=names)
    out = fit_merge(**inputs)
    assert out['status'] == 'identified'
    assert out['groups']['all']['intervals'] is None
    assert out['groups']['all']['interval_status'] == 'insufficient_contiguous_layers'


def test_isolated_fitted_layers_cannot_be_omitted_from_bootstrap(merge_fixture):
    inputs = dict(merge_fixture.inputs)
    names = tuple(f'layers.{[0, 1, 3, 5][i // 4]}.view{i % 4}.weight' for i in range(16))
    inputs['child'] = replace_record(inputs['child'], names=names)
    inputs['parents'] = [replace_record(parent, names=names) for parent in inputs['parents']]
    inputs['base'] = replace_record(inputs['base'], names=names)
    out = fit_merge(**inputs)
    assert out['status'] == 'identified'
    np.testing.assert_allclose(out['groups']['all']['coefficients'], [.7, .4, -.2], atol=1e-5)
    assert out['groups']['all']['intervals'] is None
    assert out['groups']['all']['interval_status'] == 'insufficient_contiguous_layer_blocks'
