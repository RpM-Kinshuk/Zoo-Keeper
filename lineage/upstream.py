"""Local fingerprint input and lazy boundaries to committed Model-Zoo APIs."""

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import re

from .records import FingerprintRecord, compatible_views


class UpstreamDependencyError(ImportError):
    """The owner must merge/install the upstream packages before real use."""


def _extractor_dependencies():
    try:
        fingerprint = importlib.import_module('evidence.fingerprint')
        compare = importlib.import_module('evidence.compare')
    except ImportError as exc:
        raise UpstreamDependencyError(
            'Real fingerprints require owner-merged Model-Zoo extractor X6/X8 '
            'and its collector modules on the Python import path. '
            'No installation or network access was attempted.'
        ) from exc
    return fingerprint, compare


def _dependencies():
    fingerprint, compare = _extractor_dependencies()
    try:
        schema = importlib.import_module('zoo.schema')
        retrieval = importlib.import_module('zoo.retrieval')
    except ImportError as exc:
        raise UpstreamDependencyError(
            'Real fingerprints require the owner-merged Model-Zoo extractor X6/X8 '
            'and zoo Z4, available as evidence and zoo in this environment. '
            'No installation or network access was attempted; synthetic records work independently.'
        ) from exc
    return fingerprint, compare, schema, retrieval


def _relative(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('artifact must be relative to data_root without escape')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('artifact symlink escapes data_root')
    return resolved


def _pinned(key):
    if not isinstance(key, str) or not re.fullmatch(r'[^@\s]+@(?:[0-9a-f]{40}|local:[0-9a-f]{64})', key):
        raise ValueError('manifest identity must be a pinned model key')


def _identity(value):
    key = str(value.get('repo_id', '')) + '@' + str(value.get('revision', ''))
    _pinned(key)
    return key


def _load_real(entry, path, root):
    fingerprint, _ = _extractor_dependencies()
    arch = entry.get('arch')
    if not isinstance(arch, str) or not arch.strip():
        raise ValueError('real fingerprint requires an explicit architecture signature')
    if not isinstance(entry.get('manifest'), str):
        raise ValueError('real fingerprint requires a relative manifest path')
    manifest_path = _relative(root, entry['manifest'])
    manifest = json.loads(manifest_path.read_text())
    if _identity(manifest) != entry['key']:
        raise ValueError('index and manifest identity disagree')
    # This engine currently accepts T1 only. Reject T2 before its loader opens
    # a separate HDF5 spectrum file outside the explicit fingerprint inputs.
    from safetensors import safe_open
    with safe_open(path, framework='numpy') as handle:
        if handle.metadata().get('tier') != 't1':
            raise ValueError('lineage records currently require T1 fingerprints')
    fp = fingerprint.load_fingerprint(path)
    if _identity(fp.meta) != entry['key']:
        raise ValueError('manifest and fingerprint identity disagree')
    if fp.meta.get('evidence_version') != fingerprint.EVIDENCE_VERSION:
        raise ValueError('unsupported fingerprint evidence version')
    if manifest.get('evidence_version', fingerprint.EVIDENCE_VERSION) != fp.meta['evidence_version']:
        raise ValueError('manifest and fingerprint evidence version disagree')
    if fp.meta.get('sketch_k') != fp.sketch.shape[1]:
        raise ValueError('fingerprint sketch size disagrees with metadata')
    meta = dict(fp.meta, synthetic=False, upstream_fingerprint=fp,
                manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                fingerprint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                manifest_path=manifest_path.relative_to(root).as_posix(),
                fingerprint_path=path.relative_to(root).as_posix())
    meta['parent'] = _identity(meta['parent']) if meta.get('parent') is not None else None
    return FingerprintRecord(key=entry['key'], arch=arch, names=fp.names,
                             shapes=fp.shape, rows=fp.rows, geometry=fp.geometry,
                             sketches=fp.sketch, delta=fp.delta, meta=meta,
                             coverage=fp.coverage)


def load_records(manifests: Path, fingerprints: Path, *, data_root: Path) -> list[FingerprintRecord]:
    """Read an explicit index; real rows require key/arch/manifest/fingerprint.

    Architecture comes from the caller's authoritative index, never a guess
    based on partial views. Automatic zoo manifest discovery awaits Z1/Z4.
    JSON fingerprint payloads remain reserved for synthetic fixtures.
    """
    root = Path(data_root).resolve()
    manifest_path = Path(manifests).resolve()
    fingerprint_dir = Path(fingerprints).resolve()
    if not manifest_path.is_relative_to(root) or not fingerprint_dir.is_relative_to(root):
        raise ValueError('local input paths must remain within data_root')
    entries = json.loads(manifest_path.read_text())
    if not isinstance(entries, list):
        raise ValueError('fingerprint input index must be a JSON list')
    result, seen = [], set()
    for entry in entries:
        key = entry['key']
        _pinned(key)
        if key in seen:
            raise ValueError('duplicate manifest identity')
        seen.add(key)
        path = _relative(root, entry['fingerprint'])
        if not path.is_relative_to(fingerprint_dir):
            raise ValueError('fingerprint artifact escapes fingerprints directory')
        if path.suffix != '.json':
            result.append(_load_real(entry, path, root))
            continue
        data = json.loads(path.read_text())
        if data.get('key') != key:
            raise ValueError('manifest and fingerprint identity disagree')
        if not data.get('meta', {}).get('synthetic', False):
            raise ValueError('JSON fingerprints are reserved for synthetic fixtures')
        if entry.get('arch') is not None and entry['arch'] != data.get('arch'):
            raise ValueError('manifest and fingerprint architecture disagree')
        # Inject the specified scorer locally; never deserialize executable code.
        from .testing import synthetic_median
        data['meta']['scoring_callable'] = synthetic_median
        result.append(FingerprintRecord(**data))
    return result


def median_similarity(a, b, k=64):
    pairs = compatible_views(a, b, k=k)
    if not pairs:
        return None
    if a.meta.get('synthetic') or b.meta.get('synthetic'):
        if not (a.meta.get('synthetic') and b.meta.get('synthetic')):
            return None
        scorer = a.meta.get('scoring_callable')
        if not callable(scorer):
            raise ValueError('synthetic records require an injected median-cosine scoring callable')
        return scorer(a, b, k=k)
    _, compare = _extractor_dependencies()
    if 'upstream_fingerprint' not in a.meta or 'upstream_fingerprint' not in b.meta:
        raise UpstreamDependencyError('Real records require validated upstream fingerprints from engine Task 7.')
    # Use the shared statistic on precisely the eligible aligned views. The
    # extractor alone does not exclude zero/nonfinite views or all shape changes.
    def subset(record, indices):
        fp = record.meta['upstream_fingerprint']
        return replace(fp,
                       **{field: [getattr(fp, field)[i] for i in indices]
                          for field in ('names', 'geometry', 'component', 'sha256', 'spectra')},
                       **{field: getattr(fp, field)[indices]
                          for field in ('rows', 'shape', 'sketch')},
                       delta={field: values[indices] for field, values in fp.delta.items()})
    left, right = zip(*pairs)
    return compare.pair_similarity(subset(a, list(left)), subset(b, list(right)),
                                   sketch_k=k)['median_cosine']


def retrieve_real(query, candidates, *, limit=50):
    _, _, _, retrieval = _dependencies()
    if 'zoo_dir' not in query.meta:
        raise UpstreamDependencyError('Real retrieval requires a local upstream zoo index, integrated in engine Task 7.')
    from .retrieval import exclusion_reason
    eligible = {c.key: c for c in candidates if not exclusion_reason(query, c)
                and compatible_views(query, c)}
    # Ask for the whole index before filtering to the explicit candidate list.
    rows = retrieval.nearest(Path(query.meta['zoo_dir']), query.key, query.arch,
                             int(query.meta['zoo_index_size']))
    result = []
    for key, score in rows:
        if key in eligible:
            result.append({'parent_key': key, 'retrieval_score': float(score),
                           'n_shared': len(compatible_views(query, eligible[key])), 'reason': 'upstream_retrieval'})
    return sorted(result, key=lambda row: (-row['retrieval_score'], row['parent_key']))[:limit]
