import socket

import numpy as np
import pytest

from lineage.testing import SEED, make_record, make_synthetic_pool


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError('unit tests must remain offline')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    monkeypatch.setattr(socket, 'create_connection', deny)


@pytest.fixture
def synthetic_pool():
    return make_synthetic_pool()


@pytest.fixture
def aligned_pair():
    rng = np.random.default_rng(SEED)
    arrays = rng.normal(size=(6, 64, 64)).astype(np.float32)
    return make_record('lab/a@' + 'a' * 40, arrays), make_record('lab/b@' + 'b' * 40, arrays)
