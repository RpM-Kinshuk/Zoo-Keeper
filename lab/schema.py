"""Frozen lab table contract and portable path helpers."""
import json
import re
from pathlib import Path, PurePosixPath

LAB_SCHEMA_VERSION = '1'
MODEL_COLUMNS = ['model_id', 'kind', 'path', 'base_family', 'license']
EDGE_COLUMNS = ['child', 'parent', 'edge_type', 'role', 'coefficient', 'params_json']
EDGE_TYPES = {'merge', 'lora', 'sft', 'dpo', 'abliteration', 'quantization_sim', 'laundering'}
MODEL_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')
HUB_PATH = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*@[0-9a-fA-F]{40}\Z')
LOCAL_KEY = re.compile(r'lab/([A-Za-z0-9][A-Za-z0-9._-]*)@local:[0-9a-f]{64}\Z')


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_id(model_id):
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id) or model_id in {'.', '..'}:
        raise ValueError(f'{model_id}: invalid model ID')


def validate_relative_path(value, model_id):
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or value.startswith(('~', '/')) or any(p in {'', '.', '..'} for p in value.split('/'))):
        raise ValueError(f'{model_id}: invalid relative path {value!r}')
    if PurePosixPath(value).is_absolute():
        raise ValueError(f'{model_id}: absolute path')


def contained_path(root: Path, relative: str, model_id: str) -> Path:
    validate_relative_path(relative, model_id)
    root = Path(root).resolve()
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f'{model_id}: path escapes lab root') from exc
    return candidate
