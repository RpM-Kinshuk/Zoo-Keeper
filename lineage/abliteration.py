"""Shared update spans, measured separately in sketch and exact block space.

Exact local reads require ``block_validation.json`` beside each checkpoint::

    {"version": 1, "key": "repo@<40 hex commit>", "files": [
      {"path": "model.safetensors", "sha256": "<64 hex>",
       "tensors": ["layers.0.o_proj.weight"]}]}

Local keys may instead end in ``@local:<64 hex>``. Paths are portable relative
paths; this validation sidecar does not alter the frozen fingerprint schema.
It verifies bytes against caller-supplied pins, not authenticity of the pins.
No raw blocks or singular vectors are persisted or included in diagnostics.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
import torch
from safetensors import safe_open

from .records import sketch_block, view_diagnostics

MAX_HEADER_BYTES = 100 * 1024 * 1024


def _result(status, *, space='block', n_layers=0, reasons=None):
    return dict(n_layers=n_layers, rank=None, energy_fraction=None,
                shared_direction_score=None, status=status, space=space,
                calibrated=False, reasons=list(reasons or []))


def stacked_test(deltas, *, layer_ids, max_rank=1):
    """Measure squared SVD energy of horizontally stacked row-aligned deltas.

    ``rank`` is the requested retained rank, not an inferred edge label. No
    uncalibrated energy threshold is converted into an abliteration verdict.
    """
    if not isinstance(max_rank, (int, np.integer)) or max_rank < 1:
        raise ValueError('max_rank must be a positive integer')
    if len(deltas) != len(layer_ids):
        raise ValueError('layer_ids must match deltas')
    n_layers = len(set(layer_ids))
    if n_layers < 3:
        return _result('insufficient_layers', n_layers=n_layers)
    blocks = [np.asarray(d, dtype=np.float64) for d in deltas]
    if any(d.ndim != 2 or min(d.shape) == 0 for d in blocks):
        return _result('abstain', n_layers=n_layers, reasons=['unsupported_shape'])
    if len({d.shape[0] for d in blocks}) != 1:
        return _result('abstain', n_layers=n_layers, reasons=['row_dimension_mismatch'])
    if any(not np.isfinite(d).all() for d in blocks):
        return _result('abstain', n_layers=n_layers, reasons=['nonfinite'])
    scale = max(float(np.max(np.abs(d))) for d in blocks)
    if scale == 0:
        return _result('abstain', n_layers=n_layers, reasons=['zero_update'])
    singular = np.linalg.svd(np.concatenate([d / scale for d in blocks], axis=1), compute_uv=False)
    energies = singular ** 2
    rank = min(int(max_rank), len(singular))
    fraction = float(energies[:rank].sum() / energies.sum())
    out = _result('ok', n_layers=n_layers)
    out.update(rank=rank, energy_fraction=fraction, shared_direction_score=fraction)
    return out


def projected_abliteration(child, parent, *, residual_views, max_rank=1):
    """Use signed aligned sketches under the seed/dimension/side convention.

    The explicit residual-view list restricts evidence to output-writing
    matrices. Equal block heights, row intervals, full output dimensions and
    seeds imply the same seeded rows-side projection. This only supports a
    shared span in sketch space; extraction summaries cannot supply raw deltas.
    """
    out = _result('abstain', space='sketch')
    out['block_rank'] = None
    if not residual_views or len(set(residual_views)) != len(residual_views):
        out['reasons'] = ['missing_or_duplicate_residual_views']
        return out
    k = min(child.sketches.shape[1], parent.sketches.shape[1])
    pairs, reasons = view_diagnostics(child, parent, k=k)
    by_name = {child.names[i]: (i, j) for i, j in pairs}
    chosen = []
    for name in residual_views:
        if name not in by_name:
            reasons.append(f'{name}:incompatible_or_missing_view')
        elif not any(part in name for part in ('o_proj.', 'out_proj.', 'down_proj.', 'dense_4h_to_h.', 'fc2.')):
            reasons.append(f'{name}:unsupported_residual_orientation')
        else:
            chosen.append(by_name[name])
    # Alignment exclusions outside the explicit configured views do not invalidate
    # the selected matrices, but remain visible as coverage diagnostics.
    selected_reasons = [r for r in reasons if ':' not in r or r.split(':', 1)[0] in residual_views]
    if selected_reasons or len(chosen) != len(residual_views):
        out['reasons'] = selected_reasons or ['incompatible_views']
        return out
    supported = {'synthetic-v1', '1'}
    if any((record.meta.get('evidence_version') not in supported or
            record.meta.get('sketch_convention', 'seed_dimension_side_v1') != 'seed_dimension_side_v1')
           for record in (child, parent)):
        out['reasons'] = ['unsupported_sketch_convention']
        return out
    settings = {(int(child.shapes[i, 0]), *map(int, child.rows[i]),
                 min(k, int(child.rows[i, 1] - child.rows[i, 0]))) for i, _ in chosen}
    if len(settings) != 1:
        out['reasons'] = ['noncommon_left_operator']
        return out
    for record in (child, parent):
        operators = record.meta.get('left_operators')
        if operators is not None:
            ids = [operators.get(name) for name in residual_views]
            if None in ids or len(set(ids)) != 1:
                out['reasons'] = ['noncommon_left_operator']
                return out
    if ('left_operators' in child.meta and 'left_operators' in parent.meta and
            child.meta['left_operators'] != parent.meta['left_operators']):
        out['reasons'] = ['operator_mismatch']
        return out
    matched_summary = child.meta.get('parent') == parent.key
    status = np.asarray(child.delta.get('status', []), dtype=str) if matched_summary else np.asarray([])
    ranks = np.asarray(child.delta.get('rank', [])) if matched_summary else np.asarray([])
    selected_indices = [i for i, _ in chosen]
    if status.ndim == 1 and status.size == len(child.names):
        status = status[selected_indices]
    if ranks.ndim == 1 and ranks.size == len(child.names):
        ranks = ranks[selected_indices]
    if np.any(status == 'below_floor'):
        out['reasons'] = ['below_floor']
        return out
    if np.any(ranks == -1):
        out['reasons'] = ['unresolved_delta_rank']
        return out
    layers = []
    for name in residual_views:
        match = re.search(r'(?:layers|h|blocks)\.(\d+)\.', name)
        if match is None:
            out['reasons'] = [f'{name}:missing_layer_identity']
            return out
        layers.append(match.group(1))
    measured = stacked_test([sketch_block(child, i, k) - sketch_block(parent, j, k)
                             for i, j in chosen], layer_ids=layers, max_rank=max_rank)
    measured.update(space='sketch', block_rank=None,
                    scope='supportive_projected_evidence_only',
                    floor_status='unknown' if status.size == 0 else 'reported',
                    edge_type='unknown',
                    operator={'seed': child.meta['sketch_seed'], 'side': 'rows', 'k': k},
                    coverage_reasons=reasons)
    if not matched_summary:
        measured['reasons'].append('parent_mismatch')
    if not child.delta:
        measured['reasons'].append('missing_delta_summaries')
    return measured


def _safe_path(root, relative):
    if not isinstance(relative, str) or '\\' in relative:
        raise ValueError('unsafe path')
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(p in ('..', '.') for p in path.parts):
        raise ValueError('unsafe path')
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError('symlink paths are unsupported')
    if not candidate.is_file():
        raise ValueError('missing local file')
    return candidate


def _validated_files(directory):
    root = Path(directory).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError('symlink checkpoint directory')
    manifest_path = _safe_path(root, 'block_validation.json')
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError('invalid validation manifest') from error
    if (manifest.get('version') != 1 or not re.fullmatch(
            r'[^\s]+@(?:[0-9a-f]{40}|local:[0-9a-f]{64})', str(manifest.get('key', '')))):
        raise ValueError('missing pinned manifest identity')
    index = {}
    verification_bytes = 0
    for entry in manifest.get('files', []):
        path = _safe_path(root, entry['path'])
        if path.suffix != '.safetensors' or not re.fullmatch(r'[0-9a-f]{64}', entry.get('sha256', '')):
            raise ValueError('unsupported file or missing hash')
        digest = hashlib.sha256()
        before = path.stat()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
                verification_bytes += len(chunk)
        if digest.hexdigest() != entry['sha256'] or path.stat() != before:
            raise ValueError('file hash verification failed')
        with path.open('rb') as stream:
            prefix = stream.read(8)
        header_size = int.from_bytes(prefix, 'little')
        if len(prefix) != 8 or header_size > MAX_HEADER_BYTES or header_size + 8 > before.st_size:
            raise ValueError('invalid or oversized safetensors header')
        for name in entry.get('tensors', []):
            if name in index:
                raise ValueError('duplicate tensor declaration')
            index[name] = (path, before)
    return index, verification_bytes


class _LocalBlockDeltas(list):
    """In-memory blocks with scalar verification and slice I/O accounting."""


def local_block_deltas(child_dir, parent_dir, *, tensor_names, rows):
    """Verify every manifest file, then read only requested plain matrix rows.

    Returns a list of float64 child-minus-parent blocks in caller order, with
    verification_bytes (bytes hashed, including repeated/unused file entries)
    and slice_bytes attributes recording additional payload reads. Header and filesystem overhead are excluded from slice_bytes.
    """
    if (len(rows) != 2 or any(not isinstance(v, (int, np.integer)) for v in rows)
            or rows[0] < 0 or rows[1] <= rows[0]):
        raise ValueError('invalid rows')
    if len(set(tensor_names)) != len(tensor_names):
        raise ValueError('duplicate tensor names')
    if any(any(part in name.lower() for part in ('lora_', 'adapter', 'qweight', 'qzeros'))
           for name in tensor_names):
        raise ValueError('unsupported packed or adapter representation')
    child_files, child_bytes = _validated_files(child_dir)
    parent_files, parent_bytes = _validated_files(parent_dir)
    deltas = _LocalBlockDeltas()
    deltas.verification_bytes = child_bytes + parent_bytes
    deltas.slice_bytes = 0
    for name in tensor_names:
        blocks = []
        shape = None
        for index in (child_files, parent_files):
            if name not in index:
                raise ValueError('missing tensor')
            path, verified_stat = index[name]
            if path.is_symlink() or path.stat() != verified_stat:
                raise ValueError('file changed after verification')
            with safe_open(path, framework='pt', device='cpu') as handle:
                if name not in handle.keys():
                    raise ValueError('manifest tensor missing from file')
                sliced = handle.get_slice(name)
                current_shape = sliced.get_shape()
                if len(current_shape) != 2 or sliced.get_dtype() not in ('F16', 'BF16', 'F32', 'F64'):
                    raise ValueError('unsupported packed or nonmatrix representation')
                if rows[1] > current_shape[0]:
                    raise ValueError('rows exceed tensor shape')
                if shape is not None and current_shape != shape:
                    raise ValueError('tensor shape mismatch')
                shape = current_shape
                tensor = sliced[rows[0]:rows[1], :]
                deltas.slice_bytes += tensor.numel() * tensor.element_size()
                block = tensor.to(dtype=torch.float64).numpy()
                if not np.isfinite(block).all():
                    raise ValueError('nonfinite block')
                blocks.append(block)
            if path.is_symlink() or path.stat() != verified_stat:
                raise ValueError('file changed during read')
        deltas.append(blocks[0] - blocks[1])
    return deltas
