"""Offline synthetic shared-direction and pinned local-block checks."""
import hashlib
import json
import socket
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from lineage.abliteration import local_block_deltas, projected_abliteration, stacked_test
from lineage.testing import make_record, replace_record


@pytest.fixture
def stacked_fixture():
    rng = np.random.default_rng(20261028)
    directions = np.linalg.qr(rng.normal(size=(32, 4)))[0]
    weights = rng.normal(size=(4, 32, 48))
    common = [-np.outer(directions[:, 0], directions[:, 0] @ w) for w in weights]
    independent = [-np.outer(directions[:, i], directions[:, i] @ w) for i, w in enumerate(weights)]
    independent = [d / np.linalg.norm(d) for d in independent]
    return SimpleNamespace(common=common, independent=independent, weights=weights, directions=directions)


def test_shared_direction_is_a_stacked_property(stacked_fixture):
    out = stacked_test(stacked_fixture.common, layer_ids=['0', '1', '2', '3'])
    neg = stacked_test(stacked_fixture.independent, layer_ids=['0', '1', '2', '3'])
    assert out['energy_fraction'] > .999
    assert neg['energy_fraction'] < .5
    assert out['space'] == 'block'
    assert stacked_test(stacked_fixture.common[:1], layer_ids=['0'])['status'] == 'insufficient_layers'
    assert stacked_test(stacked_fixture.common, layer_ids=['0'] * 4)['status'] == 'insufficient_layers'


def test_heretic_low_rank_span(stacked_fixture):
    r = stacked_fixture.directions[:, :2]
    deltas = [-r @ (r.T @ w) for w in stacked_fixture.weights]
    one = stacked_test(deltas, layer_ids=list('0123'))
    two = stacked_test(deltas, layer_ids=list('0123'), max_rank=2)
    assert one['energy_fraction'] < .8
    assert two['energy_fraction'] > .999
    assert two['rank'] == 2


def projected_pair(fixture):
    rng = np.random.default_rng(20261028)
    op = np.linalg.qr(rng.normal(size=(32, 16)))[0].T
    sketches = np.asarray([op @ w[:, :16] for w in fixture.weights], dtype=np.float32)
    updates = np.asarray([op @ d[:, :16] for d in fixture.common], dtype=np.float32)
    names = [f'layers.{i}.o_proj.weight' for i in range(4)]
    parent = make_record('p', sketches, names=names, rows=np.tile([0, 32], (4, 1)))
    child = make_record('c', sketches + updates, names=names, rows=parent.rows, parent=parent.key)
    operators = dict.fromkeys(names, 'synthetic-common-seeded-left')
    return replace_record(child, left_operators=operators), replace_record(parent, left_operators=operators)


def test_projected_direction_survives_common_left_operator(stacked_fixture):
    child, parent = projected_pair(stacked_fixture)
    out = projected_abliteration(child, parent, residual_views=list(child.names))
    assert out['energy_fraction'] > .999
    assert out['space'] == 'sketch'
    assert out['calibrated'] is False
    assert out['n_layers'] == 4


def test_mismatched_block_and_floor_abstain(stacked_fixture):
    child, parent = projected_pair(stacked_fixture)
    for bad in (replace_record(child, rows=child.rows + 1),
                replace_record(child, sketch_seed=7),
                replace_record(child, delta_status='below_floor'),
                replace_record(child, left_operators={}),
                replace_record(child, left_operators={n: n for n in child.names})):
        out = projected_abliteration(bad, parent, residual_views=list(child.names))
        assert out['status'] == 'abstain'
        assert out['reasons']
        assert out['energy_fraction'] is None
    assert projected_abliteration(child, parent, residual_views=['layers.0.q_proj.weight'])['status'] == 'abstain'
    # Missing extraction summaries do not invent raw block rank or singular vectors.
    out = projected_abliteration(child, parent, residual_views=list(child.names))
    assert out['block_rank'] is None
    assert 'missing_delta_summaries' in out['reasons']


def write_checkpoint(directory, arrays):
    directory.mkdir()
    weights = directory / 'model.safetensors'
    save_file(arrays, weights)
    manifest = {'version': 1, 'key': 'lab/synthetic@' + 'a' * 40,
                'files': [{'path': weights.name, 'sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
                           'tensors': sorted(arrays)}]}
    (directory / 'block_validation.json').write_text(json.dumps(manifest))
    return manifest


def test_local_slices_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', lambda *a: pytest.fail('network forbidden'))
    rng = np.random.default_rng(20261028)
    p = {'layers.0.o_proj.weight': rng.normal(size=(32, 48)).astype(np.float32)}
    c = {n: w + .25 for n, w in p.items()}
    child, parent = tmp_path / 'child', tmp_path / 'parent'
    write_checkpoint(child, c)
    write_checkpoint(parent, p)
    out = local_block_deltas(child, parent, tensor_names=list(p), rows=(2, 6))
    np.testing.assert_array_equal(out[0], c[next(iter(c))][2:6].astype(np.float64) - p[next(iter(p))][2:6])
    assert sorted(x.name for x in child.iterdir()) == ['block_validation.json', 'model.safetensors']
    with pytest.raises(ValueError, match='rows'):
        local_block_deltas(child, parent, tensor_names=list(p), rows=(0, 33))


@pytest.mark.parametrize('damage', ['bytes', 'missing', 'traversal', 'symlink', 'packed', 'header', 'adapter'])
def test_local_blocks_reject_unverified_or_unsupported(tmp_path, damage):
    arrays = {('lora_A.weight' if damage == 'adapter' else 'x'): np.ones((4, 4), dtype=np.float32)}
    child, parent = tmp_path / 'child', tmp_path / 'parent'
    manifest = write_checkpoint(child, arrays)
    write_checkpoint(parent, arrays)
    if damage == 'bytes':
        with (child / 'model.safetensors').open('ab') as stream:
            stream.write(b'tampered')
    elif damage == 'missing':
        (child / 'block_validation.json').unlink()
    elif damage == 'traversal':
        manifest['files'][0]['path'] = '../parent/model.safetensors'
        (child / 'block_validation.json').write_text(json.dumps(manifest))
    elif damage == 'symlink':
        (child / 'model.safetensors').unlink()
        (child / 'model.safetensors').symlink_to(parent / 'model.safetensors')
    else:
        if damage == 'adapter':
            pass
        elif damage == 'header':
            (child / 'model.safetensors').write_bytes((100 * 1024 * 1024 + 1).to_bytes(8, 'little'))
        else:
            save_file({'x': np.ones((4, 4), dtype=np.uint8)}, child / 'model.safetensors')
        manifest['files'][0]['sha256'] = hashlib.sha256((child / 'model.safetensors').read_bytes()).hexdigest()
        (child / 'block_validation.json').write_text(json.dumps(manifest))
    if damage == 'adapter':
        with pytest.raises(ValueError):
            local_block_deltas(child, parent, tensor_names=['lora_A.weight'], rows=(0, 2))
    else:
        with pytest.raises(ValueError):
            local_block_deltas(child, parent, tensor_names=['x'], rows=(0, 2))


def test_frozen_common_operator_needs_no_added_fingerprint_fields(stacked_fixture):
    child, parent = projected_pair(stacked_fixture)
    child_meta = {k: v for k, v in child.meta.items() if k != 'left_operators'}
    parent_meta = {k: v for k, v in parent.meta.items() if k != 'left_operators'}
    child_meta['evidence_version'] = parent_meta['evidence_version'] = '1'
    child = replace_record(child, meta=child_meta)
    parent = replace_record(parent, meta=parent_meta)
    assert projected_abliteration(child, parent, residual_views=list(child.names))['energy_fraction'] > .999
    bad = replace_record(child, evidence_version='unknown')
    assert projected_abliteration(bad, replace_record(parent, evidence_version='unknown'),
                                 residual_views=list(child.names))['status'] == 'abstain'
    shapes = child.shapes.copy()
    shapes[0, 0] += 1
    a, b = replace_record(child, shapes=shapes), replace_record(parent, shapes=shapes)
    assert projected_abliteration(a, b, residual_views=list(a.names))['status'] == 'abstain'


def test_plain_bfloat16_local_blocks(tmp_path):
    import torch
    from safetensors.torch import save_file as save_torch
    child, parent = tmp_path / 'child', tmp_path / 'parent'
    for directory, value in ((child, 1.5), (parent, 1.0)):
        manifest = write_checkpoint(directory, {'x': np.ones((4, 4), dtype=np.float32)})
        save_torch({'x': torch.full((4, 4), value, dtype=torch.bfloat16)}, directory / 'model.safetensors')
        manifest['files'][0]['sha256'] = hashlib.sha256((directory / 'model.safetensors').read_bytes()).hexdigest()
        (directory / 'block_validation.json').write_text(json.dumps(manifest))
    out = local_block_deltas(child, parent, tensor_names=['x'], rows=(0, 2))
    np.testing.assert_array_equal(out[0], np.full((2, 4), .5))
    assert out.verification_bytes > 0
    assert out.slice_bytes == 2 * 2 * 4 * 2


def test_floor_summary_must_match_proposed_parent(stacked_fixture):
    child, parent = projected_pair(stacked_fixture)
    other_parent_summary = replace_record(child, parent='other', delta_status='below_floor', delta_rank=-1)
    out = projected_abliteration(other_parent_summary, parent, residual_views=list(child.names))
    assert out['energy_fraction'] > .999
    assert out['floor_status'] == 'unknown'
    assert 'parent_mismatch' in out['reasons']


@pytest.mark.parametrize('extra_tensors', [[], ['y']])
def test_verification_bytes_count_every_hash_pass(tmp_path, extra_tensors):
    arrays = {name: np.ones((4, 4), dtype=np.float32) for name in ('x', 'y')}
    child, parent = tmp_path / 'child', tmp_path / 'parent'
    manifest = write_checkpoint(child, arrays)
    write_checkpoint(parent, arrays)
    # Every entry is hashed, including entries without any selected tensors and
    # repeated paths whose tensor declarations do not overlap.
    manifest['files'][0]['tensors'] = ['x']
    extra = dict(manifest['files'][0], tensors=extra_tensors)
    manifest['files'].append(extra)
    (child / 'block_validation.json').write_text(json.dumps(manifest))
    out = local_block_deltas(child, parent, tensor_names=['x'], rows=(0, 2))
    expected = 2 * (child / 'model.safetensors').stat().st_size + (parent / 'model.safetensors').stat().st_size
    assert out.verification_bytes == expected
    np.testing.assert_array_equal(out[0], np.zeros((2, 4)))
