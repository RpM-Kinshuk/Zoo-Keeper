"""Aligned immutable-by-convention fingerprint records and view eligibility."""

from dataclasses import dataclass
import copy

import numpy as np


def _readonly_arrays(value):
    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    elif isinstance(value, dict):
        for item in value.values():
            _readonly_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _readonly_arrays(item)


@dataclass(frozen=True)
class FingerprintRecord:
    key: str
    arch: str
    names: tuple[str, ...]
    shapes: np.ndarray
    rows: np.ndarray
    geometry: tuple[str, ...]
    sketches: np.ndarray
    delta: dict
    meta: dict
    coverage: list[dict]

    def __post_init__(self):
        object.__setattr__(self, 'names', tuple(self.names))
        object.__setattr__(self, 'geometry', tuple(self.geometry))
        n = len(self.names)
        if len(set(self.names)) != n:
            raise ValueError('view names must be unique')
        for name in ('shapes', 'rows', 'sketches'):
            value = np.array(getattr(self, name), dtype=np.float32 if name == 'sketches' else None, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.shapes.shape != (n, 2) or self.rows.shape != (n, 2):
            raise ValueError('shapes and rows must have one matrix pair per view')
        if self.sketches.ndim != 3 or self.sketches.shape[0] != n:
            raise ValueError('sketches must have shape (views, k, k)')
        if self.sketches.shape[1] != self.sketches.shape[2]:
            raise ValueError('sketches must be square')
        if len(self.geometry) != n:
            raise ValueError('geometry must have one value per view')
        if not np.issubdtype(self.rows.dtype, np.integer) or not np.issubdtype(self.shapes.dtype, np.integer):
            raise ValueError('shapes and rows must be integers')
        if np.any(self.shapes <= 0) or np.any(self.rows[:, 0] < 0) or np.any(self.rows[:, 1] <= self.rows[:, 0]):
            raise ValueError('invalid shapes or row intervals')
        if np.any(self.rows[:, 1] > self.shapes[:, 0]):
            raise ValueError('rows exceed tensor shape')
        for name in ('delta', 'meta', 'coverage'):
            value = copy.deepcopy(getattr(self, name))
            _readonly_arrays(value)
            object.__setattr__(self, name, value)


def _vocabulary(name):
    return any(part in name.lower() for part in ('embed_tokens', 'word_embeddings', 'embedding', 'wte.', 'lm_head'))


def sketch_block(record, index, k):
    """Discard zero-padded coordinates for a block narrower than the sketch."""
    height = min(k, int(record.rows[index, 1] - record.rows[index, 0]))
    width = min(k, int(record.shapes[index, 1]))
    return record.sketches[index, :height, :width].astype(np.float64)


def view_diagnostics(a, b, *, k=64):
    """Return aligned pairs and explicit exclusions; full-shape growth stays visible."""
    if k <= 0:
        raise ValueError('k must be positive')
    for field in ('tier', 'row_block', 'sketch_seed', 'evidence_version'):
        if a.meta.get(field) is None or b.meta.get(field) is None:
            return [], [f'missing_setting:{field}']
        if a.meta[field] != b.meta[field]:
            return [], [f'setting_mismatch:{field}']
    if a.meta['tier'] != 'row_blocks':
        return [], ['unsupported_tier']
    if a.arch != b.arch:
        return [], ['architecture_mismatch']
    if min(a.sketches.shape[1], b.sketches.shape[1]) < k:
        return [], ['insufficient_sketch_size']
    indices = {name: j for j, name in enumerate(b.names)}
    pairs, reasons = [], []
    for i, name in sorted(enumerate(a.names), key=lambda item: item[1]):
        j = indices.get(name)
        reason = None
        if j is None:
            reason = 'missing_view'
        elif a.geometry[i] != b.geometry[j]:
            reason = 'geometry_mismatch'
        elif not np.array_equal(a.rows[i], b.rows[j]):
            reason = 'row_interval_mismatch'
        elif not np.array_equal(a.shapes[i], b.shapes[j]):
            if not (_vocabulary(name) and a.shapes[i, 1] == b.shapes[j, 1]):
                reason = 'shape_mismatch'
            else:
                reasons.append(f'{name}:vocabulary_shape_change')
        if reason is None:
            x, y = sketch_block(a, i, k), sketch_block(b, j, k)
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                reason = 'nonfinite'
            elif np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
                reason = 'zero_norm'
        if reason:
            reasons.append(f'{name}:{reason}')
        else:
            pairs.append((i, j))
    reasons.extend(f'{name}:missing_view' for name in sorted(set(b.names) - set(a.names)))
    return pairs, reasons


def compatible_views(a: FingerprintRecord, b: FingerprintRecord, *, k: int = 64) -> list[tuple[int, int]]:
    return view_diagnostics(a, b, k=k)[0]


def cosine_views(a, b, pairs, *, k):
    values = []
    for i, j in pairs:
        x, y = sketch_block(a, i, k).ravel(), sketch_block(b, j, k).ravel()
        values.append(float(np.clip(np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y)), -1, 1)))
    return values
