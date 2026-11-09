import numpy as np
import pandas as pd
import pytest

from studies.baselines.lab_spectral import hill_alpha, similarity, score_pairs


def test_invalid_hill_keeps_valid_trace(tmp_path):
    import h5py
    import json
    from studies.baselines.lab_spectral import _profiles
    path = tmp_path / 'flat.h5'
    with h5py.File(path, 'w') as h:
        h.attrs['measurement_config_json'] = json.dumps({'use_svd': True, 'spectrum_storage': 'full'})
        h.create_dataset('layers/longname', data=np.array(
            [f'model.layers.{d}.q.weight' for d in range(3)], dtype=h5py.string_dtype()))
        h.create_dataset('eigs', data=np.array([[1., 1.], [4., 4.], [2., 2.]], dtype='f4'))
    profiles, _, _, reason = _profiles(path)
    assert profiles[0] is not None
    assert profiles[1] is None
    assert 'Flat' in reason
    with h5py.File(path, 'r+') as h:
        h['eigs'][1] = [-1., 2.]
    consumed = {}
    with pytest.raises(ValueError, match='Invalid raw'):
        _profiles(path, consumed)
    assert consumed['logical_eigenvalue_bytes'] == 2 * 2 * 4


def test_saved_spectra_run_preserves_splits_and_costs(tmp_path):
    import h5py
    import json
    from studies.baselines.lab_spectral import run
    rows = []
    for key in ('a', 'b', 'c', 'd'):
        path = tmp_path / f'{key}.h5'
        with h5py.File(path, 'w') as h:
            h.attrs['measurement_config_json'] = json.dumps({'use_svd': True, 'spectrum_storage': 'full'})
            names = [f'model.layers.{depth}.q.weight' for depth in range(3)]
            h.create_dataset('layers/longname', data=np.array(names, dtype=h5py.string_dtype()))
            h.create_dataset('eigs', data=np.array([np.array([1, 2, 4, 8], dtype='f4') ** (d+1) for d in range(3)]))
        rows.append({'model_key': key, 'full_weight_spectra': path.name})
    index = tmp_path / 'models.parquet'
    pd.DataFrame(rows).to_parquet(index)
    controls = tmp_path / 'controls/t2-k64'
    for split, keys in [('development', ('a', 'b')), ('test', ('c', 'd'))]:
        (controls / split).mkdir(parents=True)
        pd.DataFrame({'query_key': [keys[0]] * 2, 'candidate_key': keys,
                      'truth_same_root': [True, False], 'eligible': [True, True]}).to_parquet(
                          controls / split / 'universe.parquet')
    (tmp_path / 'training-12').mkdir()
    pd.DataFrame({'query_key': ['a'], 'candidate_key': ['b'],
                  'truth_same_root': [True], 'eligible': [True]}).to_parquet(
                      tmp_path / 'training-12/universe.parquet')
    report = run(tmp_path, index, controls, tmp_path / 'out')
    assert all(row['answered_pairs'] == row['pairs'] == 2 for row in report['metrics'][:4])
    assert all(row['answered_pairs'] == row['pairs'] == 1 for row in report['metrics'][4:])
    assert all(row['same_family_auroc'] is None for row in report['metrics'][4:])
    assert report['logical_eigenvalue_bytes'] == 4 * 3 * 4 * 4
    assert report['new_weight_bytes'] == 0
    assert report['historical_extraction_bytes'] is None
    bindings = json.loads((tmp_path / 'out/inputs.json').read_text())
    assert 'controls/t2-k64/development/universe.parquet' in bindings
    assert 'controls/t2-k64/test/universe.parquet' in bindings
    assert 'training-12/universe.parquet' in bindings
    assert 'code' in bindings


def test_hill_formula_and_scale_invariance():
    eigs = np.array([1., 2., 4., 8.])
    expected = 1 + 2 / np.log(8.)
    assert hill_alpha(eigs) == pytest.approx(expected, abs=1e-12)
    assert hill_alpha(eigs * 17) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('eigs, expected', [
    ([0, 1, 2, 3], 1 + 2/np.log(6)),
    ([0, 0, 1, 2, 4], 1 + 2/np.log(8)),
])
def test_hill_preserves_lower_zeros_and_full_length(eigs, expected):
    assert hill_alpha(eigs) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('eigs', [[1, 1, 1, 1], [0, 0, 1, 2], [1, np.nan], [-1, 2], [1]])
def test_invalid_spectra_abstain(eigs):
    with pytest.raises(ValueError):
        hill_alpha(eigs)


def test_native_profiles_and_symmetry():
    a = {'attn.qkv': np.array([1., 3., 2.]), 'mlp.up': np.array([2., 5., 4.])}
    b = {key: values[::-1] for key, values in a.items()}
    for method in ('Trace-native', 'Spectral-Signatures-native'):
        assert similarity(a, a, method) == pytest.approx(1, abs=1e-12)
        assert similarity(a, b, method) == pytest.approx(similarity(b, a, method))
    with pytest.raises(ValueError, match='module'):
        similarity(a, {'attn.qkv': a['attn.qkv']}, 'Trace-native')
    with pytest.raises(ValueError, match='depth'):
        similarity(a, {key: np.arange(4.) for key in a}, 'Spectral-Signatures-native')
    with pytest.raises(ValueError, match='Constant'):
        similarity({'q': np.ones(3)}, {'q': np.ones(3)}, 'Trace-native')


def test_frozen_pairs_preserved_and_missing_is_not_zero():
    pairs = pd.DataFrame({'query_key': ['q', 'q'], 'candidate_key': ['a', 'missing'],
                          'truth_same_root': [True, False], 'eligible': [True, True]})
    profiles = {'q': {'q': np.array([1., 3., 2.])}, 'a': {'q': np.array([1., 3., 2.])}}
    result = score_pairs(pairs, profiles, 'Trace-native')
    assert result[['query_key', 'candidate_key']].equals(pairs[['query_key', 'candidate_key']])
    assert result.status.tolist() == ['tested', 'not_run']
    assert np.isnan(result.score.iloc[1])
    assert result.truth_same_root.equals(pairs.truth_same_root)
