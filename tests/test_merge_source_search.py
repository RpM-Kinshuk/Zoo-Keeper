import numpy as np


def test_qr_pair_fits_equal_direct_fits_with_unexplained_noise():
    from scripts.merge_source_search import rank_pairs
    rng = np.random.default_rng(20261028)
    x = rng.normal(size=(200, 5))
    y = .4*x[:, 1] + .6*x[:, 3] + .01*rng.normal(size=200)
    keys = [str(i) for i in range(5)]
    rows = rank_pairs(dict(zip(keys, x.T)), y)
    assert rows[0]['pair'] == ['1', '3']
    assert len(rows) == 10
    for row in rows:
        matrix = x[:, [int(k) for k in row['pair']]]
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
        np.testing.assert_allclose(row['coefficients'], coefficients, rtol=1e-11, atol=1e-12)
        assert row['rank'] == rank
        np.testing.assert_allclose(row['relative_residual'], np.linalg.norm(y-matrix@coefficients)/np.linalg.norm(y), rtol=1e-11)


def test_alternative_merge_siblings_remain_after_both_sources_removed():
    from scripts.merge_source_search import rank_pairs, source_pools
    a = np.array([1., 0., 0., 1.])
    b = np.array([0., 1., 1., 0.])
    vectors = dict(a=a, b=b, low=.25*a+.75*b, high=.75*a+.25*b)
    pools = source_pools(list(vectors), ['a', 'b'], {'a': 'a', 'b': 'b', 'low': 'low', 'high': 'high'})
    assert pools['remove_both'] == ['high', 'low']
    for name, keys in pools.items():
        rows = rank_pairs({k: vectors[k] for k in keys}, .5*a+.5*b)
        assert set(rows[0]['pair']) <= set(keys)
        assert rows[0]['relative_residual'] < 1e-14


def test_weight_aliases_are_removed_together_and_singular_pairs_stay_visible():
    from scripts.merge_source_search import rank_pairs, source_pools
    pools = source_pools(['a', 'alias', 'b', 'c'], ['a', 'b'], dict(a='hash-a', alias='hash-a', b='hash-b', c='hash-c'))
    assert pools['remove_first'] == ['b', 'c']
    assert pools['remove_both'] == ['c']
    y = np.array([1., 2., 3.])
    row = rank_pairs(dict(a=y, alias=y), y)[0]
    assert row['rank'] == 1
    assert row['relative_residual'] < 1e-14
    assert 'historical_source_decision' not in row


def test_qr_preserves_original_row_count_rank_tolerance():
    from scripts.merge_source_search import rank_pairs
    a = np.zeros(1000); a[0] = 1.
    b = np.zeros(1000); b[1] = 1e-14
    row = rank_pairs(dict(a=a, b=b), a+b)[0]
    assert row['rank'] == 1


def test_control_without_row_field_reuses_the_existing_hashed_row_ledger():
    from scripts.merge_source_search import control_fingerprint
    row = dict(full_weight_fingerprint='run/model.t2.fingerprint.safetensors')
    ledger = {'run/model.t1.fingerprint.safetensors': 'known-hash', 'run/other.t1.fingerprint.safetensors': 'other-hash'}
    assert control_fingerprint(row, ledger) == 'run/model.t1.fingerprint.safetensors'
    assert control_fingerprint(row, {}) is None
