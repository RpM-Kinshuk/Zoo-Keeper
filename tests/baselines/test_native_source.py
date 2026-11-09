from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import time

import pytest

from studies.baselines.native_source import PinnedSource
from studies.baselines import native_source


class Reader:
    def __init__(self):
        self.resolves = self.reads = 0

    def resolve(self, repo, filename, revision):
        self.resolves += 1
        return SimpleNamespace(size=100)

    def read(self, remote, start, end):
        self.reads += 1
        return bytes(range(start, end))


def test_native_contract_parallel_cost_and_pinning():
    reader = Reader()
    with ThreadPoolExecutor(2) as pool:
        s = PinnedSource('org/model', 'a'*40, ['weights'], reader, pool,
                         deadline=time.monotonic()+10, byte_cap=20)
        assert s.list_files() == ['weights']
        assert s.size('weights') == 100
        assert s.read_ranges('weights', [(2, 7), (9, 12)]) == [bytes(range(2, 7)), bytes(range(9, 12))]
        assert s.bytes_read == 8
        assert reader.resolves == 1


def test_bad_range_unknown_file_and_budget_do_not_read():
    reader = Reader()
    with ThreadPoolExecutor(1) as pool:
        s = PinnedSource('org/model', 'a'*40, ['weights'], reader, pool,
                         deadline=time.monotonic()+10, byte_cap=4)
        for filename, start, end in [('weights', -1, 1), ('weights', 99, 101),
                                     ('unknown', 0, 1), ('weights', 0, 5)]:
            with pytest.raises((ValueError, KeyError)):
                s.read_range(filename, start, end)
        assert reader.reads == 0
        s.deadline = time.monotonic()-1
        with pytest.raises(TimeoutError):
            s.read_range('weights', 0, 1)
        assert reader.reads == 0


def test_reject_unpinned_revision():
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(ValueError):
            PinnedSource('org/model', 'main', [], Reader(), pool,
                         deadline=time.monotonic()+10, byte_cap=4)


def test_expired_resolution_never_starts_payload_or_new_lookup(monkeypatch):
    clock = [0]
    monkeypatch.setattr(native_source, 'time', SimpleNamespace(monotonic=lambda: clock[0]))

    class SlowReader(Reader):
        def resolve(self, *args):
            result = super().resolve(*args)
            clock[0] = 20
            return result

    reader = SlowReader()
    with ThreadPoolExecutor(1) as pool:
        s = PinnedSource('org/model', 'a'*40, ['weights'], reader, pool,
                         deadline=10, byte_cap=4)
        with pytest.raises(TimeoutError):
            s.read_range('weights', 0, 1)
        assert reader.reads == 0
        other = PinnedSource('org/model', 'a'*40, ['weights'], reader, pool,
                             deadline=10, byte_cap=4)
        with pytest.raises(TimeoutError):
            other.size('weights')
        assert reader.resolves == 1


def test_source_read_failure_halts_queued_work():
    class BrokenReader(Reader):
        def read(self, *args):
            self.reads += 1
            raise OSError('transport failure')

    reader = BrokenReader()
    with ThreadPoolExecutor(1) as pool:
        s = PinnedSource('org/model', 'a'*40, ['weights'], reader, pool,
                         deadline=time.monotonic()+10, byte_cap=4)
        with pytest.raises(OSError):
            s.read_range('weights', 0, 1)
        with pytest.raises(OSError):
            s.read_range('weights', 1, 2)
        assert reader.reads == 1


def test_native_fingerprint_unchanged_by_reader_and_prefetch(tmp_path):
    import json
    import numpy as np
    from safetensors.numpy import save_file
    from evidence.hub import HubReader
    from modeldna.io.source import LocalSource
    from modeldna.fingerprint.extract import extract_fingerprint

    cfg = dict(model_type='llama', architectures=['LlamaForCausalLM'],
               num_hidden_layers=2, hidden_size=8, intermediate_size=16,
               num_attention_heads=2, num_key_value_heads=2, vocab_size=32)
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    rng = np.random.default_rng(7)
    weights = {}
    for layer in range(2):
        prefix = f'model.layers.{layer}.'
        for role, shape in [('self_attn.q_proj', (8, 8)), ('self_attn.k_proj', (8, 8)),
                            ('self_attn.v_proj', (8, 8)), ('self_attn.o_proj', (8, 8)),
                            ('mlp.up_proj', (16, 8)), ('mlp.gate_proj', (16, 8)),
                            ('mlp.down_proj', (8, 16))]:
            weights[prefix+role+'.weight'] = rng.normal(size=shape).astype('float32')
        weights[prefix+'input_layernorm.weight'] = np.ones(8, dtype='float32')
    save_file(weights, str(tmp_path/'model.safetensors'))
    reference = extract_fingerprint(LocalSource(tmp_path), mode='fast', model_id='fixture')
    with ThreadPoolExecutor(2) as pool:
        source = PinnedSource(str(tmp_path), 'a'*40,
                              ['config.json', 'model.safetensors'], HubReader(), pool,
                              deadline=time.monotonic()+30, byte_cap=1024**2)
        actual = extract_fingerprint(source, mode='fast', model_id='fixture')
    assert reference.pcs_samples
    for field in ['sigma_curves', 'vector_norms', 'vector_samples', 'pcs_samples',
                  'spectra_sketch', 'spectra_exact']:
        assert getattr(reference, field) == getattr(actual, field)
