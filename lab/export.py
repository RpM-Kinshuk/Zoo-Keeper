"""Short-lived absolute-path CSVs for local extractor invocation, never release artifacts."""
import csv
import io
import os
from pathlib import Path
import tempfile

from .schema import HUB_PATH, MODEL_COLUMNS, contained_path, validate_id


def export_models(models, lab_root: Path, destination: Path) -> None:
    """Export selected completed models; the caller removes this invocation CSV.

    Portable manifests are the durable outputs. The invocation CSV is restricted
    to temporary scratch or ZOO_DATA/cache and refuses unrelated existing data.
    """
    if list(models.columns) != MODEL_COLUMNS or models.model_id.duplicated().any():
        raise ValueError('invalid or duplicate model rows for export')
    lab_root = Path(lab_root).resolve()
    destination = Path(destination).resolve()
    repository = Path(__file__).resolve().parents[1]
    if (destination.is_relative_to(repository) or not any(destination.is_relative_to(root)
            for root in (Path(tempfile.gettempdir()).resolve(), lab_root.parent / 'cache'))):
        raise ValueError('invocation CSV must be outside the repository in temporary scratch or ZOO_DATA/cache')
    rows = []
    for row in models.sort_values('model_id').itertuples(index=False):
        validate_id(row.model_id)
        if HUB_PATH.fullmatch(row.path):
            rows.append(row.path.rsplit('@', 1))
        else:
            if '@' in row.path:
                raise ValueError(f'{row.model_id}: missing full Hub commit pin')
            checkpoint = contained_path(lab_root, row.path, row.model_id)
            if not checkpoint.is_dir():
                raise ValueError(f'{row.model_id}: missing local checkpoint')
            rows.append([str(checkpoint.resolve()), ''])
    content = io.StringIO(newline='')
    writer = csv.writer(content, lineterminator='\n')
    writer.writerow(['repo_id', 'revision'])
    writer.writerows(rows)
    encoded = content.getvalue().encode('utf-8')
    if destination.exists():
        if destination.read_bytes() == encoded:
            return
        raise FileExistsError('invocation CSV already exists with different contents')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.models-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
