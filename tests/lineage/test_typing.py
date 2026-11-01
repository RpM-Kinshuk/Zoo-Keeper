"""Offline matched-parent delta evidence and explicit rule contracts."""

import copy
import json

import numpy as np
import pytest

from lineage.typing import pair_features, type_edge, fit_rules
from lineage.testing import SEED, make_record, replace_record


@pytest.fixture
def delta_pair():
    rng = np.random.default_rng(SEED)
    parent = make_record('lab/parent@' + 'a' * 40, rng.normal(size=(6, 64, 64)))
    sv = np.tile([16.] * 8 + [4.] * 8, (6, 1))
    delta = dict(rank=np.full(6, 8), confidence=['high'] * 6, status=['ok'] * 6,
                 frac_identical=np.zeros(6), top_sv=sv, rel_update_norm=np.full(6, .1))
    child = make_record('lab/child@' + 'b' * 40, parent.sketches + .1,
                        parent=parent.key, delta=delta)
    return child, parent


@pytest.fixture
def rules():
    return dict(adapter_max_rank=16, min_views=4, min_changed_fraction=.25, min_gap=2.,
                min_coverage=1., adapter_modules=['*.q_proj.weight', '*.v_proj.weight'],
                head_modules=['lm_head.weight'], finetune_modules=['*.mlp.*'],
                quantization_max_flatness=2., quantization_min_prior=.8,
                quantization_storage=['int8', 'int4'])


def test_below_floor_and_wrong_parent_abstain(delta_pair, rules):
    child, parent = delta_pair
    child = replace_record(child, delta_status='below_floor', delta_rank=-1,
                           delta_frac_identical=.9)
    result = type_edge(pair_features(child, parent), rules)
    assert result['edge_type'] == 'unknown'
    assert result['status'] == 'abstained'
    assert result['features']['rank'] is None
    child = replace_record(child, parent='other@' + 'a' * 40)
    assert pair_features(child, parent)['rank_status'] == 'parent_mismatch'


def test_rank_alone_does_not_label_abliteration(delta_pair, rules):
    child, parent = delta_pair
    child = replace_record(child, delta_rank=1)
    assert type_edge(pair_features(child, parent), rules)['edge_type'] != 'abliteration'


def test_identical_and_missing_abstain(delta_pair, rules):
    child, parent = delta_pair
    for altered in [replace_record(child, delta_status='identical', delta_rank=0,
                                    delta_frac_identical=1.), replace_record(child, delta={})]:
        result = type_edge(pair_features(altered, parent), rules)
        assert result['edge_type'] == 'unknown'
        assert result['reasons']


def test_module_pattern_changes_label_at_same_rank(delta_pair, rules):
    child, parent = delta_pair
    assert type_edge(pair_features(child, parent), rules)['edge_type'] == 'adapter'
    names = tuple(f'layers.{i}.mlp.up_proj.weight' for i in range(6))
    child, parent = replace_record(child, names=names), replace_record(parent, names=names)
    assert type_edge(pair_features(child, parent), rules)['edge_type'] == 'finetune'


def test_head_only_addition_is_finetune(delta_pair, rules):
    child, parent = delta_pair
    delta = copy.deepcopy(child.delta)
    for field in ('status', 'rank', 'frac_identical'):
        delta[field] = np.asarray(delta[field], dtype=object if field == 'status' else None).copy()
    delta['status'][:] = 'identical'
    delta['rank'][:] = 0
    delta['frac_identical'][:] = 1.
    child = replace_record(child, delta=delta, names=child.names[:-1] + ('lm_head.weight',))
    parent = replace_record(parent, names=parent.names[:-1], shapes=parent.shapes[:-1],
                            rows=parent.rows[:-1], geometry=parent.geometry[:-1],
                            sketches=parent.sketches[:-1])
    result = type_edge(pair_features(child, parent), rules)
    assert result['edge_type'] == 'finetune'
    assert 'head_only_addition' in result['reasons']


def test_low_confidence_and_missing_rank_never_become_zero(delta_pair, rules):
    child, parent = delta_pair
    for child in [replace_record(child, delta_rank=-1),
                  replace_record(child, delta_confidence='low')]:
        features = pair_features(child, parent)
        assert features['rank'] is None
        assert type_edge(features, rules)['edge_type'] == 'unknown'


def test_unobserved_gap_and_no_invented_flatness(delta_pair, rules):
    child, parent = delta_pair
    features = pair_features(replace_record(child, delta_rank=16), parent)
    assert features['spectral_gap'] is None
    assert 'gap_outside_observed_spectrum' in features['reasons']
    assert features['delta_flatness'] is None
    assert type_edge(features, rules)['edge_type'] == 'unknown'


def test_explicit_factors_are_stronger_but_parent_and_floor_still_gate(delta_pair, rules):
    child, parent = delta_pair
    child = replace_record(child, adapter_factors={'parent': parent.key, 'verified': True,
        'modules': list(child.names), 'ranks': [8] * 6})
    child = replace_record(child, delta_top_sv=np.zeros((6, 16)))
    assert type_edge(pair_features(child, parent), rules)['edge_type'] == 'adapter'
    child = replace_record(child, delta_status='below_floor')
    assert type_edge(pair_features(child, parent), rules)['edge_type'] == 'unknown'


def test_bf16_alone_is_not_quantization(delta_pair, rules):
    child, parent = delta_pair
    child = replace_record(child, storage_dtype='bf16')
    assert type_edge(pair_features(child, parent), rules)['edge_type'] != 'quantized'
    child = replace_record(child, storage_dtype='int8', quantization_evidence={
        'parent': parent.key, 'delta_flatness': 1.2, 'leaf_prior': .9,
        'provenance': 'development_frozen'})
    assert type_edge(pair_features(child, parent), rules)['edge_type'] == 'quantized'


def test_mismatches_missing_coverage_and_unsupported_representation(delta_pair, rules):
    child, parent = delta_pair
    for altered in [replace_record(child, row_block=128),
                    replace_record(child, representation='unsupported'),
                    replace_record(child, delta_rank=[8] * 2)]:
        result = type_edge(pair_features(altered, parent), rules)
        assert result['edge_type'] == 'unknown'
        assert result['reasons']


def test_rule_json_is_explicit_and_fit_only_development_exact_roots(delta_pair, rules):
    child, parent = delta_pair
    features = pair_features(child, parent)
    bad = dict(rules)
    del bad['min_gap']
    with pytest.raises(ValueError):
        type_edge(features, bad)
    rows = [dict(features=features, edge_type='adapter', roots=['r1'], truth_tier='lab_exact')]
    provenance = dict(complete_root_components=True, duplicates_grouped=True)
    fitted = fit_rules(rows, split={'r1': 'development', 'r2': 'test'},
                       provenance=provenance, candidate_rules=[rules, dict(rules, min_gap=8.)])
    assert fitted == fit_rules(rows, split={'r1': 'development', 'r2': 'test'},
                             provenance=provenance, candidate_rules=[rules, dict(rules, min_gap=8.)])
    assert json.loads(json.dumps(fitted))['frozen'] is True
    assert type_edge(features, fitted)['edge_type'] == 'adapter'
    for split in ['test', 'prospective', 'calibration']:
        with pytest.raises(ValueError):
            fit_rules(rows, split={'r1': split}, provenance=provenance, candidate_rules=[rules])
    with pytest.raises(ValueError):
        fit_rules([dict(rows[0], roots=['r1', 'r2'])], split={'r1': 'development', 'r2': 'test'},
                  provenance=provenance, candidate_rules=[rules])


def test_removed_module_is_not_a_head_only_addition(delta_pair, rules):
    child, parent = delta_pair
    child = replace_record(child, delta_status='identical', delta_rank=0,
                           delta_frac_identical=1., names=child.names[:-1] + ('lm_head.weight',))
    result = type_edge(pair_features(child, parent), rules)
    assert result['edge_type'] == 'unknown'
    assert 'unresolved_removed_modules' in result['reasons']


def test_wrong_parent_discards_even_strong_metadata(delta_pair, rules):
    child, parent = delta_pair
    child = replace_record(child, parent='other@' + 'f' * 40, adapter_factors={
        'parent': parent.key, 'verified': True, 'modules': list(child.names), 'ranks': [8] * 6})
    features = pair_features(child, parent)
    assert features['rank'] is None
    assert features['adapter_factors'] is None
    assert type_edge(features, rules)['edge_type'] == 'unknown'
