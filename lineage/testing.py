"""Deterministic synthetic fixtures, never representations of real checkpoints."""

from dataclasses import fields
import copy
from types import SimpleNamespace

import numpy as np

from .records import FingerprintRecord, compatible_views, cosine_views

SEED = 20261028


def synthetic_median(a, b, *, k=64):
    pairs = compatible_views(a, b, k=k)
    return float(np.median(cosine_views(a, b, pairs, k=k))) if pairs else None


def make_record(key, sketches, *, parent=None, delta=None, arch='cfg:synthetic', tier='t1',
                row_block=256, rows=None, names=None):
    sketches = np.asarray(sketches, dtype=np.float32)
    n = len(sketches)
    return FingerprintRecord(
        key=key, arch=arch,
        names=tuple(names) if names is not None else tuple(f'layers.{i // 2}.{"q_proj" if i % 2 == 0 else "v_proj"}.weight' for i in range(n)),
        shapes=np.tile([512, 128], (n, 1)),
        rows=np.asarray(rows) if rows is not None else np.tile([128, 384], (n, 1)),
        geometry=('matrix',) * n, sketches=sketches, delta=delta or {},
        meta={'synthetic': True, 'tier': tier, 'row_block': row_block, 'sketch_seed': SEED,
              'evidence_version': 'synthetic-v1', 'parent': parent,
              'scoring_callable': synthetic_median}, coverage=[])


def replace_record(record, **changes):
    """Copy fields or metadata without sharing writable arrays or nested dictionaries."""
    data = {field.name: copy.deepcopy(getattr(record, field.name)) for field in fields(record)}
    for key, value in changes.items():
        if key in data:
            data[key] = value
        elif key.startswith('delta_'):
            data['delta'][key[6:]] = value
        else:
            data['meta'][key] = value
    return FingerprintRecord(**data)


def make_synthetic_pool():
    rng = np.random.default_rng(SEED)
    shape = (6, 64, 64)
    roots = [rng.normal(size=shape).astype(np.float32) for _ in range(3)]
    arrays = roots + [roots[0] + .1 * rng.normal(size=shape), roots[1] + .1 * rng.normal(size=shape)]
    arrays += [arrays[3] + .1 * rng.normal(size=shape), arrays[4] + .1 * rng.normal(size=shape)]
    arrays += [arrays[3] + .15 * rng.normal(size=shape), arrays[3] + .15 * rng.normal(size=shape)]
    arrays += [.6 * arrays[3] + .4 * arrays[4], arrays[3].copy(), rng.normal(size=shape)]
    keys = [f'lab/synthetic-{i}@' + f'{i + 1:040x}' for i in range(12)]
    links = [(0, 3), (1, 4), (3, 5), (4, 6), (3, 7), (3, 8), (3, 9), (4, 9)]
    records = []
    for i, array in enumerate(arrays):
        parents = [keys[p] for p, c in links if c == i]
        record = make_record(keys[i], array, parent=parents[0] if len(parents) == 1 else None,
                             arch='cfg:different' if i == 2 else 'cfg:synthetic', tier='t0' if i == 11 else 't1')
        record = replace_record(record, duplicate_group='duplicate-3' if i in (3, 10) else f'unique-{i}')
        records.append(record)
    truth = {'nodes': keys, 'edges': [{'parent_key': keys[p], 'child_key': keys[c],
              'edge_type': 'merge' if c == 9 else 'finetune', 'evidence_tier': 'lab_exact'} for p, c in links],
             'completeness': {'parent': True, 'ancestor': True, 'same_root': True},
             'duplicates': {keys[10]: keys[3]}}
    universe = [{'key': r.key, 'query_eligible': i != 11, 'candidate_eligible': i != 11,
                 'duplicate_group': r.meta['duplicate_group'], 'reason': 't0_only' if i == 11 else 'ok'}
                for i, r in enumerate(records)]
    manifests = [{'key': r.key, 'synthetic': True, 'arch': r.arch} for r in records]
    return SimpleNamespace(records=records, truth=truth, universe=universe, manifests=manifests)
