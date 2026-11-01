"""Deterministic, immutable paper protocol freezes."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical_json(value):
    """Canonical UTF-8 representation shared by identifiers and freezes."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def freeze_protocol(config: dict, inputs: dict, out: Path) -> str:
    """Create a freeze once; equal resumes succeed, unequal resumes fail.

    Paths in config and inventory must already be repository/data-root relative.
    A runtime dry-run override belongs in separate run metadata, not this freeze.
    """
    def check_paths(value):
        if isinstance(value, dict):
            for child in value.values():
                check_paths(child)
        elif isinstance(value, list):
            for child in value:
                check_paths(child)
        elif isinstance(value, (str, Path)) and Path(value).is_absolute():
            raise ValueError("frozen protocol paths must be relative")

    check_paths(config)
    check_paths(inputs)
    payload = canonical_json({"config": config, "inputs": inputs})
    digest = hashlib.sha256(payload).hexdigest()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out.parent, prefix=f".{out.name}.", delete=False) as f:
        temporary = Path(f.name)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        # Hard-link publishes without replacing a concurrent or existing freeze.
        try:
            os.link(temporary, out)
        except FileExistsError:
            if out.read_bytes() != payload:
                raise ValueError("existing frozen protocol differs") from None
    finally:
        temporary.unlink()
    return digest
