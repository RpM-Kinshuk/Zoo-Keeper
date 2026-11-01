"""Exercise the committed extractor with tiny local CPU tensors."""
import json

import pytest
import torch

from lineage import upstream
from lineage.typing import pair_features


@pytest.fixture
def real_inputs(tmp_path):
    from evidence.fingerprint import fingerprint_tensors
    directory = tmp_path / 'fingerprints'
    directory.mkdir()
    entries = []
    for name, revision in [('parent', 'a' * 40), ('child', 'b' * 40)]:
        identity = {'repo_id': 'lab/' + name, 'revision': revision}
        path = directory / (name + '.safetensors')
        fingerprint_tensors([
            ('model.layers.0.self_attn.q_proj.weight', torch.arange(128).reshape(16, 8).float()),
            ('model.layers.0.self_attn.v_proj.weight', torch.zeros(16, 8)),
        ], path, identity)
        manifest = tmp_path / (name + '.manifest.json')
        manifest.write_text(json.dumps(identity))
        entries.append({'key': identity['repo_id'] + '@' + revision,
                        'arch': 'cfg:explicit-test-signature',
                        'manifest': manifest.name,
                        'fingerprint': path.relative_to(tmp_path).as_posix()})
    index = tmp_path / 'inputs.json'
    index.write_text(json.dumps(entries))
    return tmp_path, directory, index, entries


def test_load_real_fingerprints_without_zoo_and_use_shared_scorer(real_inputs, monkeypatch):
    root, directory, index, _ = real_inputs
    original = upstream.importlib.import_module
    def no_zoo(name):
        assert not name.startswith('zoo'), 'extractor adapter must not need zoo modules'
        return original(name)
    monkeypatch.setattr(upstream.importlib, 'import_module', no_zoo)
    records = upstream.load_records(index, directory, data_root=root)
    a, b = records
    assert a.arch == 'cfg:explicit-test-signature'
    assert not a.meta.get('synthetic')
    assert a.delta['rank'].shape == (2,)
    assert not a.sketches.flags.writeable
    # The all-zero shared view is ineligible, not a second zero similarity.
    assert upstream.median_similarity(a, b) == pytest.approx(1.)


@pytest.mark.parametrize('field,value', [('repo_id', 'lab/wrong'), ('revision', 'c' * 40)])
def test_manifest_fingerprint_identity_must_match(real_inputs, field, value):
    root, directory, index, entries = real_inputs
    manifest = root / entries[0]['manifest']
    data = json.loads(manifest.read_text())
    data[field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='identity'):
        upstream.load_records(index, directory, data_root=root)


def test_real_input_requires_explicit_architecture(real_inputs):
    root, directory, index, entries = real_inputs
    entries[0].pop('arch')
    index.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match='architecture'):
        upstream.load_records(index, directory, data_root=root)


def test_parent_identity_is_preserved_for_delta_matching(real_inputs):
    from safetensors import safe_open
    from safetensors.numpy import load_file, save_file
    root, directory, index, entries = real_inputs
    child = directory / 'child.safetensors'
    tensors = load_file(child)
    with safe_open(child, framework='numpy') as handle:
        metadata = handle.metadata()
    metadata['parent'] = json.dumps({'repo_id': 'lab/parent', 'revision': 'a' * 40})
    save_file(tensors, child, metadata=metadata)
    parent, child = upstream.load_records(index, directory, data_root=root)
    assert child.meta['parent'] == parent.key
    assert pair_features(child, parent)['rank_status'] != 'parent_mismatch'


def test_real_manifest_path_cannot_escape_root(real_inputs):
    root, directory, index, entries = real_inputs
    entries[0]['manifest'] = '../outside.manifest.json'
    index.write_text(json.dumps(entries))
    with pytest.raises(ValueError, match='escape|relative'):
        upstream.load_records(index, directory, data_root=root)
