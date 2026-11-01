"""Atomic, immutable checkpoint publication with portable provenance."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

from .schema import (HUB_PATH, LOCAL_KEY, LAB_SCHEMA_VERSION, canonical_json,
                     contained_path, validate_id)


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _portable(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError('config keys must be strings')
            _portable(item)
    elif isinstance(value, list):
        for item in value:
            _portable(item)
    elif isinstance(value, str):
        if value.startswith(('/', '~', '\\')) or '\\' in value or '..' in value.split('/'):
            raise ValueError('config contains a nonportable path')
        if len(value) > 1 and value[1] == ':':
            raise ValueError('config contains an absolute path')


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical_json(value) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _lineage(model_id, recipe):
    from .schema import EDGE_TYPES
    relation = recipe.get('edge_type')
    if relation not in EDGE_TYPES:
        raise ValueError(f'{model_id}: unknown edge_type')
    parents = recipe.get('parents', [])
    if not isinstance(parents, list) or not parents or len(set(parents)) != len(parents):
        raise ValueError(f'{model_id}: expected unique direct parents')
    reference = recipe.get('reference')
    if reference is not None and relation != 'merge':
        raise ValueError(f'{model_id}: only merges have a reference')
    direct = parents + ([reference] if reference is not None and reference not in parents else [])
    identities = recipe.get('parent_identities', {})
    if not isinstance(identities, dict) or set(identities) != set(direct):
        raise ValueError(f'{model_id}: unresolved parent identities')
    bases = []
    for parent in direct:
        validate_id(parent)
        if parent == model_id:
            raise ValueError(f'{model_id}: self parent')
        identity = identities[parent]
        if not isinstance(identity, str):
            raise ValueError(f'{model_id}: invalid identity for {parent}')
        local = LOCAL_KEY.fullmatch(identity)
        if local and local.group(1) == parent:
            bases.append('lab/' + parent)
        elif HUB_PATH.fullmatch(identity):
            bases.append(identity)
        else:
            raise ValueError(f'{model_id}: unpinned parent {parent}')
    return {'name': 'lab/' + model_id, 'bases': bases, 'relation': relation}


def write_checkpoint(model_id: str, output_root: Path, recipe: dict,
                     writer: Callable[[Path], None], *, resume: bool = False) -> Path:
    validate_id(model_id)
    _portable(recipe)
    lineage = _lineage(model_id, recipe)
    config_hash = content_hash(recipe)
    input_hash = content_hash({'parents': recipe['parent_identities'], 'inputs': recipe.get('inputs', {})})
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    target = contained_path(output_root, model_id, model_id)
    if target.is_symlink():
        raise ValueError(f'{model_id}: checkpoint symlink is not permitted')
    if target.exists():
        if not resume:
            raise ValueError(f'{model_id}: completed output already exists')
        try:
            for path in target.rglob('*'):
                contained_path(target, path.relative_to(target).as_posix(), model_id)
                if path.is_symlink():
                    raise ValueError('symlink in checkpoint')
            provenance = json.loads((target / 'recipe.json').read_text())
            actual_lineage = json.loads((target / 'lineage.json').read_text())
            safe_weights = [p for p in target.rglob('*.safetensors') if p.is_file()]
            if (provenance['config_hash'] != config_hash or provenance['input_hash'] != input_hash
                    or provenance['recipe'] != recipe or actual_lineage != lineage or not safe_weights):
                raise ValueError('mismatched or incomplete provenance')
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'{model_id}: resume mismatch or incomplete output') from exc
        return target
    staging = Path(tempfile.mkdtemp(prefix='.' + model_id + '-', dir=output_root))
    try:
        writer(staging)
        for path in staging.rglob('*'):
            contained_path(staging, path.relative_to(staging).as_posix(), model_id)
            if path.is_symlink() or path.suffix in {'.pt', '.pth', '.bin', '.pkl', '.pickle'}:
                raise ValueError(f'{model_id}: unsafe checkpoint file {path.name}')
        if not any(p.is_file() for p in staging.rglob('*.safetensors')):
            raise ValueError(f'{model_id}: writer produced no safetensors checkpoint')
        atomic_json(staging / 'recipe.json', {
            'schema_version': LAB_SCHEMA_VERSION, 'model_id': model_id,
            'recipe_hash': config_hash, 'config_hash': config_hash, 'input_hash': input_hash,
            'parent_identities': recipe['parent_identities'], 'recipe': recipe,
        })
        atomic_json(staging / 'lineage.json', lineage)
        # rename refuses an existing nonempty checkpoint; completed targets are never replaced.
        if target.exists() or target.is_symlink():
            raise ValueError(f'{model_id}: output appeared during write')
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target
