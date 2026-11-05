"""Delivery boundaries preserve source bytes and exact measurement contracts."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from lab.scripts import evaluate_delivery as delivery


@pytest.mark.parametrize('legacy', [True, False])
def test_delivery_mapping_accepts_both_naming_generations(legacy):
    if legacy:
        row = dict(fingerprint_t2='fp/a.t2.fingerprint.safetensors',
                   spectra_h5='fp/a.t2.spectra.h5', t1_b128='128', t1_b256='256', t1_b512='512')
    else:
        row = dict(full_weight_fingerprint='fp/a.full_weights.fingerprint.safetensors',
                   full_weight_spectra='fp/a.full_weights.spectra.h5',
                   row_blocks_128='128', row_blocks_256='256', row_blocks_512='512')
    before = dict(row)
    result = delivery.normalize_delivery_entry(row)
    assert result['row_blocks_256'] == '256'
    assert result['full_weight_fingerprint'] == row.get('fingerprint_t2', row.get('full_weight_fingerprint'))
    assert row == before


def test_delivery_rejects_conflicting_aliases_and_wrong_companion():
    row = dict(fingerprint_t2='fp/a.t2.fingerprint.safetensors', spectra_h5='fp/b.t2.spectra.h5')
    with pytest.raises(ValueError, match='sidecar'):
        delivery.normalize_delivery_entry(row)
    row['spectra_h5'] = 'fp/a.t2.spectra.h5'
    row['full_weight_fingerprint'] = 'other'
    with pytest.raises(ValueError, match='Conflicting'):
        delivery.normalize_delivery_entry(row)


@pytest.mark.parametrize('version,mode', [('1','t1'), ('2','row_blocks'), ('3','row_blocks')])
def test_frozen_source_version_is_exact_and_source_bytes_unchanged(tmp_path, version, mode):
    path = tmp_path/'a.safetensors'
    save_file({'sketch': np.ones((1,64,64),dtype=np.float32)}, path,
              metadata=dict(repo_id='a', revision='b'*40, tier=mode, row_block='256',
                            sketch_k='64', sketch_seed='20261028', evidence_version=version))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    frozen = delivery.freeze_measurement(path, key='a@'+'b'*40, measurement='original')
    assert frozen['evidence_version'] == version
    fp = SimpleNamespace(meta={**frozen, 'repo_id':'a', 'revision':'b'*40},
                         sketch=np.ones((1,64,64)))
    delivery.validate_frozen_measurement(fp, key='a@'+'b'*40, contract=frozen)
    fp.meta['evidence_version'] = '2' if version != '2' else '1'
    with pytest.raises(ValueError, match='measurement'):
        delivery.validate_frozen_measurement(fp, key='a@'+'b'*40, contract=frozen)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('field,value', [('tier','full_weights'), ('row_block','128'),
    ('sketch_k','128'), ('sketch_seed','0'), ('evidence_version','99'), ('revision','c'*40)])
def test_freeze_rejects_wrong_scientific_settings(tmp_path, field, value):
    meta = dict(repo_id='a', revision='b'*40, tier='row_blocks', row_block='256',
                sketch_k='64', sketch_seed='20261028', evidence_version='1')
    meta[field] = value
    path = tmp_path/'a.safetensors'
    save_file({'sketch':np.ones((1,64,64),dtype=np.float32)}, path, metadata=meta)
    with pytest.raises(ValueError):
        delivery.freeze_measurement(path, key='a@'+'b'*40, measurement='original')


def test_relatedness_rejects_mixed_source_versions_before_scoring():
    def fingerprint(version):
        return SimpleNamespace(names=['x'], geometry=['dense'], rows=np.array([[0,4]]),
            shape=np.array([[4,4]]), sketch=np.ones((1,64,64)),
            meta=dict(tier='row_blocks', row_block=256, sketch_seed=20261028,
                      sketch_k=64, evidence_version=version))
    fp1, fp2 = fingerprint('1'), fingerprint('3')
    with pytest.raises(ValueError, match='versions'):
        delivery.similarity_predictions(['a','b'], {'a':'a', 'b':'b'}, {'a':fp1, 'b':fp2}, 64, 'row-blocks')


@pytest.mark.parametrize('legacy', [False, True])
def test_chain_delivery_reads_and_binds_both_filename_generations(tmp_path, legacy):
    from lineage.lab_report import bind_artifact
    index = 'artifact-index3.csv' if legacy else 'artifact-index.csv'
    prefix = 't2' if legacy else 'full-weights'
    (tmp_path/index).write_text('model_id,model_key\nendpoint,lab/endpoint@local:abc\n')
    for suffix in ('freeze','progress'):
        (tmp_path/(prefix+'-'+suffix+'.json')).write_text('{}')
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    proofs = {}
    def bind(path):
        relative = path.relative_to(tmp_path).as_posix()
        proofs[relative] = bind_artifact(tmp_path, relative)
        return path
    frame = delivery.read_chain_delivery(tmp_path, bind)
    assert frame.model_id.tolist() == ['endpoint']
    assert set(proofs) == set(before)
    assert {p.name:p.read_bytes() for p in tmp_path.iterdir()} == before
