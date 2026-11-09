import numpy as np
import pytest

from studies.baselines.witness import moments, anchor_scores, aggregate


def test_orthogonal_children_swap_and_distance():
    a, b, w = np.zeros((2, 2)), np.eye(2), np.array([[0., 1.], [1., 0.]])
    m = moments(a, b, w)
    sa, sb = anchor_scores(m)
    assert sa == 0
    assert sb == pytest.approx(1/np.sqrt(2))
    assert anchor_scores(moments(b, a, w)) == pytest.approx((sb, sa))
    assert m[1] < m[2]  # W is closer to A.


def test_streaming_and_triangle_geometry():
    rng = np.random.default_rng(8)
    for _ in range(20):
        a, b, w = rng.normal(size=(3, 9, 4))
        total = moments(a, b, w)
        streamed = sum((moments(a[i:i+2], b[i:i+2], w[i:i+2])
                        for i in range(0, 9, 2)), np.zeros(5))
        np.testing.assert_allclose(total, streamed, atol=1e-10)
        sa, sb = anchor_scores(total)
        assert (sa < sb) == (total[1] < total[2])


def test_roles_equal_weight_tie_and_undefined():
    assert aggregate({'q': [(0., 1.), (0., 1.)], 'down': [(1., 0.)]}) == (.5, .5)
    with pytest.raises(ValueError):
        aggregate({'q': []})
    with pytest.raises(ValueError):
        anchor_scores(np.zeros(5))
    assert anchor_scores(moments(np.zeros((2, 2)), np.ones((2, 2)),
                                np.ones((2, 2))*.5)) == (1., 1.)


@pytest.mark.parametrize('a,b,w', [
    (np.zeros((2, 2)), np.ones((3, 2)), np.ones((2, 2))),
    (np.zeros((2, 2)), np.full((2, 2), np.nan), np.ones((2, 2))),
    (np.zeros(2), np.ones(2), np.ones(2)),
])
def test_invalid_weights(a, b, w):
    with pytest.raises(ValueError):
        moments(a, b, w)
