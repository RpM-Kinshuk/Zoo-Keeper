import json
import socket

import numpy as np
import pytest

from lineage.records import compatible_views
from lineage.retrieval import retrieve
from lineage.testing import make_record, replace_record
from lineage.upstream import load_records, median_similarity


def test_retrieve_is_local_sorted_and_arch_limited(synthetic_pool):
    rows = retrieve(synthetic_pool.records[3], synthetic_pool.records, limit=3)
    assert len(rows) <= 3
    assert all(r['parent_key'] != synthetic_pool.records[3].key for r in rows)
    assert rows == sorted(rows, key=lambda r: (-r['retrieval_score'], r['parent_key']))
    assert all(r['parent_key'] != synthetic_pool.records[2].key for r in rows)


def test_settings_and_padding_are_checked(aligned_pair):
    a, b = aligned_pair
    assert len(compatible_views(a, b)) == 6
    assert compatible_views(a, replace_record(b, tier='full_weights')) == []
    assert compatible_views(a, replace_record(b, row_block=128)) == []
    assert compatible_views(a, replace_record(b, rows=b.rows + 1)) == []
    assert compatible_views(a, replace_record(b, sketch_seed=7)) == []
    assert compatible_views(a, replace_record(b, evidence_version='other')) == []


def test_vocab_growth_keeps_shared_embedding_block(aligned_pair):
    a, b = aligned_pair
    a = replace_record(a, names=('model.embed_tokens.weight',), sketches=a.sketches[:1],
                       shapes=np.array([[4096, 128]]), rows=np.array([[1920, 2176]]), geometry=('matrix',))
    b = replace_record(b, names=a.names, sketches=b.sketches[:1],
                       shapes=np.array([[4104, 128]]), rows=a.rows, geometry=a.geometry)
    assert compatible_views(a, b) == [(0, 0)]
    assert compatible_views(a, replace_record(b, shapes=np.array([[4104, 129]]))) == []


def test_small_head_excludes_zero_padding():
    x = np.ones((1, 64, 64), dtype=np.float32)
    y = x.copy()
    y[:, 2:, :] = -100
    a = replace_record(make_record('a', x), shapes=np.array([[2, 128]]), rows=np.array([[0, 2]]))
    b = replace_record(make_record('b', y), shapes=a.shapes, rows=a.rows)
    assert median_similarity(a, b) == pytest.approx(1.0)
    assert retrieve(a, [b])[0]['retrieval_score'] == pytest.approx(1.0)
    y[:, :2, :] = 0
    assert compatible_views(a, replace_record(b, sketches=y)) == []


def test_zero_view_and_nonfinite_are_explained(aligned_pair):
    a, b = aligned_pair
    bad = b.sketches.copy()
    bad[0] = 0
    bad[1, 0, 0] = np.nan
    b = replace_record(b, sketches=bad)
    out = retrieve(a, [b])[0]
    assert out['n_shared'] == 4
    assert 'zero_norm' in out['reason']
    assert 'nonfinite' in out['reason']


def test_exact_duplicate_groups_excluded_and_ties_sorted(aligned_pair):
    a, b = aligned_pair
    a = replace_record(a, duplicate_group='same')
    b = replace_record(b, duplicate_group='same')
    assert retrieve(a, [a, b]) == []
    c = replace_record(b, key='c', duplicate_group='other')
    d = replace_record(c, key='d')
    assert [r['parent_key'] for r in retrieve(a, [d, c])] == ['c', 'd']


def test_arrays_are_readonly_and_original_unchanged(aligned_pair):
    a, b = aligned_pair
    with pytest.raises(ValueError):
        a.sketches[0, 0, 0] = 0
    changed = replace_record(b, tier='full_weights')
    assert b.meta['tier'] == 'row_blocks'
    assert changed.meta['tier'] == 'full_weights'


@pytest.mark.parametrize('measurement_mode', ['row_blocks', 't1'])
def test_local_synthetic_json_roundtrip_is_offline(tmp_path, monkeypatch, measurement_mode):
    def deny(*args, **kwargs):
        raise AssertionError('network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', deny)
    fp = tmp_path / 'fingerprints'
    fp.mkdir()
    key = 'lab/tiny@local:' + 'a' * 64
    data = {'key': key, 'arch': 'cfg:synthetic', 'names': ['layer.0.weight'],
            'shapes': [[512, 128]], 'rows': [[128, 384]], 'geometry': ['matrix'],
            'sketches': np.ones((1, 64, 64)).tolist(), 'delta': {},
            'meta': {'synthetic': True, 'tier': measurement_mode, 'row_block': 256,
                     'sketch_seed': 20261028, 'evidence_version': 'synthetic-v1'}, 'coverage': []}
    (fp / 'tiny.json').write_text(json.dumps(data))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'key': key, 'fingerprint': 'fingerprints/tiny.json'}]))
    records = load_records(manifest, fp, data_root=tmp_path)
    assert records[0].key == key
    assert records[0].meta['tier'] == 'row_blocks'
    assert median_similarity(records[0], records[0]) == pytest.approx(1)
    data['key'] = 'lab/wrong@local:' + 'b' * 64
    (fp / 'tiny.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='identity'):
        load_records(manifest, fp, data_root=tmp_path)


def test_local_paths_cannot_escape_root(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{'key': 'lab/x@' + 'a' * 40, 'fingerprint': '../outside.json'}]))
    with pytest.raises(ValueError, match='relative|escape'):
        load_records(manifest, tmp_path, data_root=tmp_path)


def test_retrieval_zero_projection_is_explained(aligned_pair):
    a, b = aligned_pair
    changed = b.sketches.copy()
    changed[0, :32, :32] = 0
    b = replace_record(b, sketches=changed)
    row = retrieve(a, [b])[0]
    assert row['n_shared'] == 5
    assert 'zero_norm' in row['reason']


def test_larger_sketches_are_truncated_at_frozen_size():
    rng = np.random.default_rng(20261028)
    x = rng.normal(size=(1, 128, 128)).astype(np.float32)
    y = x.copy()
    y[:, 64:, :] *= -100
    y[:, :, 64:] *= -100
    a = make_record('a', x)
    b = make_record('b', y)
    assert median_similarity(a, b) == pytest.approx(1.0)


def test_unsupported_geometry_and_missing_settings_abstain(aligned_pair):
    a, b = aligned_pair
    assert compatible_views(a, replace_record(b, geometry=('other',) * 6)) == []
    metadata = dict(b.meta)
    metadata.pop('sketch_seed')
    assert compatible_views(a, replace_record(b, meta=metadata)) == []


def test_local_dependency_error_is_actionable(tmp_path, monkeypatch):
    import lineage.upstream as upstream
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps([{'key': 'lab/tiny@' + 'a' * 40, 'fingerprint': 'tiny.safetensors'}]))
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(upstream.importlib, 'import_module', missing)
    with pytest.raises(upstream.UpstreamDependencyError, match='owner-merged Model-Zoo'):
        load_records(path, tmp_path, data_root=tmp_path)
