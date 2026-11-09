import numpy as np
import pytest

from studies.baselines.residual import signature, score


def test_center_and_hidden_basis_invariance():
    rng = np.random.default_rng(3)
    up, down = rng.normal(size=(7, 4)), rng.normal(size=(4, 7))
    a, s = signature(down, up)
    perm = rng.permutation(7)
    scale = np.arange(1, 8)
    b, t = signature(down[:, perm] / scale, up[perm] * scale[:, None])
    np.testing.assert_allclose(a, b, atol=1e-10)
    assert s == pytest.approx(t, abs=1e-10)
    assert np.trace(a.reshape(4, 4)) == pytest.approx(0, abs=1e-10)


def test_assignment_and_reference_gate():
    a = np.eye(3)
    b = a[[2, 0, 1]]
    value, assignment = score((a, np.ones(3)), (b, np.ones(3)))
    assert value == pytest.approx(1)
    assert assignment.tolist() == [1, 2, 0]
    value, _ = score((a, np.ones(3)), (b, np.full(3, .2)))
    assert value == pytest.approx(.2)
    value, _ = score((a, np.ones(3)), (b, np.zeros(3)))
    assert value == 0


@pytest.mark.parametrize('down,up', [
    (np.eye(3), np.eye(3)),
    (np.zeros((3, 4)), np.zeros((4, 3))),
    (np.full((3, 4), np.nan), np.ones((4, 3))),
    (np.ones((3, 4)), np.ones((5, 3))),
])
def test_invalid_signature(down, up):
    with pytest.raises(ValueError):
        signature(down, up)


def test_incompatible_or_bad_score():
    for other in [(np.eye(2), np.ones(2)), (np.eye(3), -np.ones(3)),
                  (np.full((3, 3), np.nan), np.ones(3))]:
        with pytest.raises(ValueError):
            score((np.eye(3), np.ones(3)), other)
