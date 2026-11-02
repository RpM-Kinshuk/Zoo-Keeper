from types import SimpleNamespace

import numpy as np
import pytest

from lineage.lab_report import common_distances, bind_artifact, freeze_json


def fingerprint(names, values, rows=None):
    return SimpleNamespace(names=names, geometry=['dense'] * len(names),
        shape=np.array([[4, 4]] * len(names)),
        rows=np.array(rows or [[0, 4]] * len(names)),
        sketch=np.array(values, dtype=np.float32).reshape(len(names), 2, 2),
        meta=dict(tier='t1', row_block=4, sketch_seed=1, evidence_version='v1', sketch_k=2))


def test_tree_uses_one_common_coordinate_space_for_every_pair():
    a = fingerprint(['x', 'extra'], [[0, 0, 0, 0], [500, 0, 0, 0]])
    b = fingerprint(['x'], [[1, 0, 0, 0]])
    c = fingerprint(['extra', 'x'], [[-500, 0, 0, 0], [3, 0, 0, 0]])
    distance, views = common_distances([a, b, c], k=2)
    assert views == ['x']
    np.testing.assert_array_equal(distance, [[0, 1, 9], [1, 0, 4], [9, 4, 0]])


def test_common_coordinates_reject_row_changes_and_empty_intersection():
    a = fingerprint(['x'], [[0, 0, 0, 0]])
    b = fingerprint(['x'], [[1, 0, 0, 0]], rows=[[1, 5]])
    with pytest.raises(ValueError, match='common'):
        common_distances([a, b], k=2)


def test_squared_distance_retains_small_nonzero_updates():
    a = fingerprint(['x'], [[100, 100, 100, 100]])
    b = fingerprint(['x'], [[100.001, 100, 100, 100]])
    distance, _ = common_distances([a, b], k=2)
    assert distance[0, 1] > 0
    assert distance[0, 1] == pytest.approx(float((b.sketch-a.sketch).astype(np.float64).ravel()[0]**2))


def test_binding_rejects_escape_symlinks_and_mutated_artifact(tmp_path):
    artifact = tmp_path/'data'; artifact.write_bytes(b'original')
    proof = bind_artifact(tmp_path, 'data')
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        bind_artifact(tmp_path, 'data', expected=proof['sha256'])
    with pytest.raises(ValueError, match='relative'):
        bind_artifact(tmp_path, '../data')
    link = tmp_path/'link'; link.symlink_to(artifact)
    with pytest.raises(ValueError, match='regular'):
        bind_artifact(tmp_path, 'link')


def test_freeze_never_replaces_a_previous_protocol(tmp_path):
    path = tmp_path/'freeze.json'
    freeze_json(path, {'threshold': .75})
    freeze_json(path, {'threshold': .75})
    with pytest.raises(ValueError, match='immutable'):
        freeze_json(path, {'threshold': .7})


@pytest.mark.parametrize('field,value', [('tier','t2'), ('row_block',8),
                                       ('sketch_seed',2), ('evidence_version','v2')])
def test_tree_rejects_different_measurement_settings(field, value):
    a = fingerprint(['x'], [[0, 0, 0, 0]])
    b = fingerprint(['x'], [[1, 0, 0, 0]])
    b.meta[field] = value
    with pytest.raises(ValueError, match='measurement'):
        common_distances([a, b], k=2)


def test_same_key_wrong_tier_is_rejected_before_scoring():
    from lineage.lab_report import validate_measurement
    fp = fingerprint(['x'], [[1, 0, 0, 0]])
    fp.meta.update(repo_id='lab/model', revision='local:'+'a'*64, tier='t2')
    with pytest.raises(ValueError, match='measurement'):
        validate_measurement(fp, key='lab/model@local:'+'a'*64, tier='t1', row_block=4,
                             sketch_k=2, sketch_seed=1, evidence_version='v1')


def test_untyped_relatedness_rows_can_be_published(tmp_path):
    from lab.scripts.evaluate_delivery import similarity_predictions
    from lineage.io import write_edges
    a = fingerprint(['x'], [[1, 0, 0, 0]])
    keys = {'a':'lab/a@local:'+'a'*64, 'b':'lab/b@local:'+'b'*64}
    frame = similarity_predictions(['a','b'], keys, {'a':a,'b':a}, 2, 'test')
    write_edges(frame, tmp_path/'edges.parquet', provenance={'scope':'test'})
    assert set(frame.edge_type) == {'unknown'}


def test_training_input_index_excludes_unlisted_valid_artifacts():
    from lineage.lab_report import training_input_index
    key = 'lab/a@local:'+'a'*64
    progress = {'a':dict(key=key,manifest='m/a.json',fingerprint='fp/a.safetensors')}
    manifests = {'a':dict(repo_id='lab/a',revision='local:'+'a'*64,config={'hidden_size':4,'num_hidden_layers':1})}
    before = training_input_index(['a'], {'a':key}, progress, manifests)
    progress['b'] = dict(key='lab/b@local:'+'b'*64,manifest='m/b.json',fingerprint='fp/b.safetensors')
    assert training_input_index(['a'], {'a':key}, progress, manifests) == before
    assert [r['key'] for r in before] == [key]
