import hashlib
import json
import shutil

import pytest
import torch
from safetensors.torch import save_file

from lab.cohort_evidence import PinnedLocalReader, measure_ready, source_manifest


@pytest.fixture
def source_row(tmp_path, monkeypatch):
    monkeypatch.setenv('ZOO_DATA', str(tmp_path))
    path = tmp_path / 'lab/inputs/source'
    path.mkdir(parents=True)
    (path / 'config.json').write_text(json.dumps({'model_type':'llama', 'hidden_size':16,
        'num_hidden_layers':1, 'intermediate_size':32, 'num_attention_heads':2,
        'num_key_value_heads':1, 'vocab_size':32}))
    save_file({'model.layers.0.self_attn.q_proj.weight': torch.arange(256).reshape(16,16).float()},
              path / 'model.safetensors')
    sha = lambda name: hashlib.sha256((path / name).read_bytes()).hexdigest()
    return tmp_path, dict(model_id='source', root_id='seed1', path='lab/inputs/source',
        repo_id='publisher/source', revision='a'*40, license='apache-2.0',
        weight_sha256={'model.safetensors':sha('model.safetensors')},
        config_file_sha256=sha('config.json'), complete=True, synthetic=False, kind='base')


def test_verified_mirror_preserves_hub_identity_without_network(source_row):
    root, row = source_row
    reader = PinnedLocalReader(root)
    manifest, proof = source_manifest(row, data_root=root, reader=reader)
    assert manifest['repo_id'] == row['repo_id'] and manifest['revision'] == row['revision']
    assert proof['weight_sha256'] == row['weight_sha256']
    remote = reader.resolve(row['repo_id'], 'model.safetensors', row['revision'])
    assert reader.read(remote, 0, 8) == (root / row['path'] / 'model.safetensors').read_bytes()[:8]
    with pytest.raises(ValueError, match='unmapped'):
        reader.resolve(row['repo_id'], 'model.safetensors', 'b'*40)


def test_source_hash_mismatch_never_promotes_local_bytes_to_hub_pin(source_row):
    root, row = source_row
    row['weight_sha256']['model.safetensors'] = '0'*64
    with pytest.raises(ValueError, match='hash'):
        source_manifest(row, data_root=root, reader=PinnedLocalReader(root))


def test_incremental_measurement_resumes_matching_complete_rows(source_row):
    root, row = source_row
    run = root / 'runs/cohort'
    run.mkdir(parents=True)
    ready = run / 'ready.json'
    ready.write_text(json.dumps({'schema_version':'1', 'sources':[row], 'outputs':[]}))
    result = measure_ready(ready, data_root=root, run_root=run)
    measured = result['rows'][0]
    assert measured['key'] == 'publisher/source@' + 'a'*40
    assert measured['tensor_bytes_read'] > 0
    fp = root / measured['fingerprint']
    before = fp.stat().st_mtime_ns
    again = measure_ready(ready, data_root=root, run_root=run)
    assert again['rows'] == result['rows']
    assert fp.stat().st_mtime_ns == before
    row['root_id'] = 'changed'
    ready.write_text(json.dumps({'schema_version':'1', 'sources':[row], 'outputs':[]}))
    with pytest.raises(ValueError, match='changed'):
        measure_ready(ready, data_root=root, run_root=run)


def test_resume_rechecks_actual_source_bytes(source_row):
    root, row = source_row
    run = root/'runs/cohort'
    run.mkdir(parents=True)
    ready = run/'ready.json'
    ready.write_text(json.dumps({'schema_version':'1','sources':[row],'outputs':[]}))
    measure_ready(ready, data_root=root, run_root=run)
    weight = root/row['path']/'model.safetensors'
    data = bytearray(weight.read_bytes())
    data[-1] ^= 1
    weight.write_bytes(data)
    with pytest.raises(ValueError, match='hash'):
        measure_ready(ready, data_root=root, run_root=run)


def test_new_derived_measurement_requires_frozen_weight_hashes(source_row):
    root, source = source_row
    child_path = root/'lab/child'
    child_path.mkdir()
    for name in ('model.safetensors','config.json'):
        shutil.copyfile(root/source['path']/name, child_path/name)
    child = dict(model_id='child',root_id='seed1',path='lab/child',kind='derived',
                 parents=['source'],complete=True,synthetic=False,
                 weight_sha256={'model.safetensors':hashlib.sha256((child_path/'model.safetensors').read_bytes()).hexdigest()})
    (child_path/'lineage.json').write_text(json.dumps({'name':'lab/child',
        'bases':['publisher/source@'+'a'*40],'relation':'quantization_sim'}))
    weight=child_path/'model.safetensors';data=bytearray(weight.read_bytes());data[-1]^=1;weight.write_bytes(data)
    run=root/'runs/cohort';run.mkdir(parents=True)
    ready=run/'ready.json';ready.write_text(json.dumps({'sources':[source],'outputs':[child]}))
    with pytest.raises(ValueError,match='hash'):
        measure_ready(ready,data_root=root,run_root=run)


def test_resume_preserves_legacy_paths_versions_and_bytes(source_row):
    from safetensors import safe_open
    root, row = source_row
    run = root/'runs/cohort'; run.mkdir(parents=True)
    ready = run/'ready.json'
    ready.write_text(json.dumps({'sources':[row], 'outputs':[]}))
    measured = measure_ready(ready, data_root=root, run_root=run)['rows'][0]
    current = root/measured['fingerprint']
    with safe_open(current, framework='pt') as handle:
        meta = handle.metadata()
        tensors = {name:handle.get_tensor(name) for name in handle.keys()}
    meta.update(tier='t1', evidence_version='1')
    legacy = run/'evidence/t1'/current.name.replace('.row_blocks.', '.t1.')
    legacy.parent.mkdir()
    save_file(tensors, legacy, metadata=meta)
    current.unlink()
    manifest_path = root/measured['manifest']
    manifest = json.loads(manifest_path.read_text()); manifest['evidence_version'] = '1'
    manifest_path.write_text(json.dumps(manifest))
    measured.update(fingerprint=legacy.relative_to(root).as_posix(),
                    fingerprint_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
                    manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    progress = dict(schema_version='1', rows=[measured])
    (run/'evidence/progress.json').write_text(json.dumps(progress))
    before = {p:p.read_bytes() for p in (legacy, manifest_path, run/'evidence/progress.json')}
    assert measure_ready(ready, data_root=root, run_root=run) == progress
    assert all(p.read_bytes() == data for p,data in before.items())
    assert not current.exists()
