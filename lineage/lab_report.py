"""Bounded real-lab analysis with explicit artifact and coordinate contracts."""
import hashlib
import json
from pathlib import Path

import numpy as np

from lab.artifacts import atomic_json


def bind_artifact(root, relative, expected=None):
    root, relative = Path(root).resolve(), Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('artifact must be relative to data root')
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('artifact escapes data root')
    if not path.is_file() or any(p.is_symlink() for p in [path, *path.parents] if p != root):
        raise ValueError('artifact must be a regular non-symlink file')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    actual = digest.hexdigest()
    if expected is not None and actual != expected:
        raise ValueError('artifact hash differs from bound evidence')
    return dict(path=relative.as_posix(), sha256=actual, size_bytes=path.stat().st_size)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('immutable protocol differs; choose a new output directory')
    else:
        atomic_json(path, value)


def validate_measurement(fp, *, key, tier, row_block, sketch_k, sketch_seed, evidence_version):
    expected = dict(tier=tier, row_block=row_block, sketch_k=sketch_k,
                    sketch_seed=sketch_seed, evidence_version=evidence_version)
    if fp.meta.get('repo_id', '')+'@'+fp.meta.get('revision', '') != key:
        raise ValueError('fingerprint identity differs from frozen delivery')
    if any(fp.meta.get(field) != value for field, value in expected.items()):
        raise ValueError('fingerprint measurement settings differ from declared mode')
    if fp.sketch.ndim != 3 or fp.sketch.shape[1:] != (sketch_k, sketch_k):
        raise ValueError('fingerprint measurement size differs from metadata')


def training_input_index(names, keys, progress, manifests):
    """Select exact frozen entries; live directory additions never enter inference."""
    from zoo.schema import arch_signature
    if len(set(names)) != len(names):
        raise ValueError('duplicate training identities')
    entries = []
    for name in sorted(names):
        row, manifest = progress[name], manifests[name]
        if row['key'] != keys[name] or manifest['repo_id']+'@'+manifest['revision'] != keys[name]:
            raise ValueError('bound training identity mismatch')
        entries.append(dict(key=keys[name], arch=arch_signature(manifest.get('config', {}), []),
                            manifest=row['manifest'], fingerprint=row['fingerprint']))
    return entries


def common_distances(fingerprints, *, k=64):
    """Mean squared Frobenius distance in ONE shared coordinate space.

    Per-pair view intersections are inappropriate for the quartet diagnostic.
    Subtract in float64 before squaring to retain small updates next to large
    pretrained weights; a Gram-matrix subtraction can cancel those updates.
    """
    if not fingerprints or type(k) is not int or k < 1:
        raise ValueError('fingerprints and a positive sketch size are required')
    positions = [{name: i for i, name in enumerate(fp.names)} for fp in fingerprints]
    if any(len(p) != len(fp.names) for p, fp in zip(positions, fingerprints)):
        raise ValueError('duplicate fingerprint view names')
    if any(min(fp.sketch.shape[1:]) < k for fp in fingerprints):
        raise ValueError('requested sketch prefix is unavailable')
    for field in ('tier', 'row_block', 'sketch_seed', 'evidence_version'):
        if any(field not in fp.meta or fp.meta[field] != fingerprints[0].meta.get(field)
               for fp in fingerprints):
            raise ValueError('incompatible fingerprint measurement settings: '+field)
    if any(fp.meta.get('sketch_k') != fp.sketch.shape[1] for fp in fingerprints):
        raise ValueError('fingerprint measurement size differs from metadata')
    common = sorted(set.intersection(*(set(p) for p in positions)))
    kept = []
    for name in common:
        reference = fingerprints[0]; first = positions[0][name]
        if all(fp.geometry[p[name]] == reference.geometry[first]
               and np.array_equal(fp.shape[p[name]], reference.shape[first])
               and np.array_equal(fp.rows[p[name]], reference.rows[first])
               for fp, p in zip(fingerprints, positions)):
            kept.append(name)
    if not kept:
        raise ValueError('no common aligned coordinate views')
    vectors = [np.stack([fp.sketch[p[name], :k, :k] for name in kept]).astype(np.float64)
               for fp, p in zip(fingerprints, positions)]
    if not all(np.isfinite(v).all() for v in vectors):
        raise ValueError('nonfinite shared sketch coordinates')
    distances = np.zeros((len(vectors), len(vectors)), dtype=np.float64)
    for i, left in enumerate(vectors):
        for j in range(i):
            difference = left - vectors[j]
            distances[i, j] = distances[j, i] = float(np.square(difference).sum() / len(kept))
    return distances, kept
