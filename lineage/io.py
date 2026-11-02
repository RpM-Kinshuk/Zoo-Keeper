"""Publish complete immutable edge/provenance bundles through one pointer."""
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from zoo.schema import LINEAGE_COLUMNS, split_key

from .engine import canonical


def write_edges(edges, out: Path, *, provenance: dict) -> None:
    if list(edges.columns) != LINEAGE_COLUMNS:
        raise ValueError('edges must use exact LINEAGE_COLUMNS order')
    for row in edges.itertuples(index=False):
        split_key(row.child_key)
        split_key(row.parent_key)
        if (row.child_key == row.parent_key or row.estimand not in {'parent', 'ancestor', 'same_root'}
                or row.evidence_tier != 'detected' or not np.isfinite(row.score)
                or row.edge_type not in {'finetune', 'adapter', 'merge', 'quantized', 'abliteration', 'unknown'}
                or not isinstance(row.calibrated, (bool, np.bool_))):
            raise ValueError('invalid detected lineage row')
        if row.calibrated and (not np.isfinite(row.p_value) or not 0 < row.p_value <= 1):
            raise ValueError('calibrated edge requires a valid p-value')
        if not row.calibrated and not (row.p_value is None or np.isnan(row.p_value)):
            raise ValueError('uncalibrated edge must have a null p-value')
        if not isinstance(json.loads(row.evidence), dict):
            raise ValueError('edge evidence must be a JSON object')
    arrays = [pa.array(edges[name].tolist(), type=pa.bool_() if name == 'calibrated' else
                       pa.float64() if name in ('score', 'p_value') else pa.string(), from_pandas=True)
              for name in LINEAGE_COLUMNS]
    table = pa.Table.from_arrays(arrays, names=LINEAGE_COLUMNS)
    # Resolve the containing directory, not an existing output symlink.
    requested = Path(out).absolute()
    requested.parent.mkdir(parents=True, exist_ok=True)
    out = requested.parent.resolve() / requested.name
    runs = out.parent / ('.' + out.name + '.runs')
    runs.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.staging-', dir=runs))
    generation = runs / uuid4().hex
    pointer = out.parent / ('.' + out.name + '.' + uuid4().hex)
    moved = False
    try:
        pq.write_table(table, stage / 'edges.parquet')
        (stage / 'provenance.json').write_text(canonical(provenance) + '\n')
        os.replace(stage, generation)
        moved = True
        pointer.symlink_to(os.path.relpath(generation / 'edges.parquet', out.parent))
        os.replace(pointer, out)
    finally:
        pointer.unlink(missing_ok=True)
        if not moved:
            shutil.rmtree(stage, ignore_errors=True)
        # Complete generations remain readable, even after an interrupted
        # pointer switch; never sweep another reader's or writer's artifacts.
