"""Incremental local extraction with verified public-pin mirrors and read costs."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from evidence.fingerprint import fingerprint_model, load_fingerprint
from evidence.hub import HubReader
from evidence.manifest import build_manifest, write_manifest
from measurement_config import safe_filename

from .artifacts import atomic_json, content_hash
from .schema import validate_id


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('cohort path must be relative without escape')
    result = (root/path).resolve()
    if not result.is_relative_to(root):
        raise ValueError('cohort path escapes data root')
    return result


def verify_ready_weights(row, root):
    source = contained(Path(root).resolve(), row['path'])
    expected = row.get('weight_sha256', {})
    actual = {p.relative_to(source).as_posix() for p in source.rglob('*.safetensors')}
    if not expected or set(expected) != actual:
        raise ValueError('ready weight set/hash evidence mismatch')
    for name, expected_hash in expected.items():
        if sha(contained(source, name)) != expected_hash:
            raise ValueError('ready weight hash mismatch')
    return source


class PinnedLocalReader(HubReader):
    """An offline mirror; public identities are used only after byte verification."""
    def __init__(self, root):
        super().__init__()
        self.root = Path(root).resolve()
        self.mirrors = {}
        self.identities = {}
        self.tensor_bytes = Counter()
        self.metadata_bytes = Counter()
        self.lock = threading.Lock()

    def resolve(self, repo_id, filename, revision):
        key = (str(repo_id), revision)
        if key in self.mirrors:
            source = self.mirrors[key]
        elif Path(repo_id).is_dir() and Path(repo_id).resolve().is_relative_to(self.root):
            source = Path(repo_id).resolve()
        else:
            raise ValueError('unmapped source identity; network fallback forbidden')
        return super().resolve(str(source), filename, revision)

    def read(self, remote, start, end):
        data = super().read(remote, start, end)
        path = Path(remote.repo_id).resolve()
        key = self.identities.get(str(path), path.relative_to(self.root).as_posix())
        with self.lock:
            target = self.tensor_bytes if remote.filename.endswith('.safetensors') else self.metadata_bytes
            target[key] += len(data)
        return data


def source_manifest(row, *, data_root, reader):
    root = Path(data_root).resolve()
    source = contained(root, row['path'])
    marker = json.loads(contained(root, row['marker']).read_text()) if row.get('marker') else {}
    weights = row.get('weight_sha256') or marker.get('weight_sha256')
    expected_config = row.get('config_file_sha256') or marker.get('config_file_sha256')
    if not weights or set(weights) != {p.name for p in source.glob('*.safetensors')}:
        raise ValueError('source weight set/hash evidence missing')
    verified = {}
    for name, expected in weights.items():
        actual = sha(contained(source, name))
        if actual != expected:
            raise ValueError('source weight hash mismatch')
        verified[name] = actual
    if expected_config is None or sha(source/'config.json') != expected_config:
        raise ValueError('source config hash mismatch')
    key = row['repo_id'] + '@' + row['revision']
    reader.mirrors[(row['repo_id'], row['revision'])] = source
    reader.identities[str(source)] = key
    manifest = build_manifest(str(source), 'main', reader, api=None)
    manifest.update(repo_id=row['repo_id'], revision=row['revision'], license=row['license'])
    proof = dict(model_key=key, local_path=row['path'], weight_sha256=verified,
                 config_file_sha256=expected_config,
                 scope='public pin retained after complete selected-weight/config byte verification')
    return manifest, proof


def measure_ready(ready_path, *, data_root, run_root, model_ids=None, max_seconds=6600):
    root, run = Path(data_root).resolve(), Path(run_root).resolve()
    if not run.is_relative_to(root):
        raise ValueError('run must remain within data root')
    ready = json.loads(Path(ready_path).read_text())
    rows = [r for r in ready['sources'] + ready['outputs'] if r.get('complete') and not r.get('synthetic')]
    for row in rows:
        validate_id(row['model_id'])
    by_id = {r['model_id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate ready IDs')
    selected = set(model_ids or by_id)
    if not selected <= set(by_id):
        raise ValueError('selected model is not ready')
    # Include all observed parents of selected derivatives; no graph guesses.
    pending = list(selected)
    while pending:
        for parent in by_id[pending.pop()].get('parents', []):
            if parent not in by_id:
                raise ValueError('ready child has no ready parent')
            if parent not in selected:
                selected.add(parent)
                pending.append(parent)
    progress_path = run/'evidence/progress.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'schema_version':'1','rows':[]}
    completed = {row['model_id']: row for row in progress['rows']}
    for model_id in selected & set(completed):
        if completed[model_id]['ready_row_sha256'] != content_hash(by_id[model_id]):
            raise ValueError('completed ready input changed')
        for field in ('manifest','fingerprint'):
            if sha(contained(root, completed[model_id][field])) != completed[model_id][field+'_sha256']:
                raise ValueError('completed evidence artifact changed')
    reader, manifests = PinnedLocalReader(root), {}
    started = time.monotonic()
    unresolved = set(selected)
    while unresolved:
        available = sorted(k for k in unresolved if set(by_id[k].get('parents', [])) <= manifests.keys())
        if not available:
            raise ValueError('cycle in ready cohort')
        for model_id in available:
            row = by_id[model_id]
            if model_id in completed:
                manifest = json.loads(contained(root, completed[model_id]['manifest']).read_text())
                source = contained(root, row['path'])
                if row.get('repo_id'):
                    current, _ = source_manifest(row, data_root=root, reader=reader)
                else:
                    verify_ready_weights(row, root)
                    current = build_manifest(str(source), 'main', reader, api=None)
                if current['repo_id']+'@'+current['revision'] != completed[model_id]['key']:
                    raise ValueError('completed source identity changed')
                reader.identities[str(source)] = manifest['repo_id']+'@'+manifest['revision']
                manifests[model_id] = manifest
                unresolved.remove(model_id)
                continue
            if time.monotonic()-started > max_seconds:
                raise RuntimeError('bounded incremental extraction time budget reached')
            begin = time.monotonic()
            if row.get('repo_id'):
                manifest, proof = source_manifest(row, data_root=root, reader=reader)
                atomic_json(run/'evidence/sources'/f'{model_id}.json', proof)
            else:
                source = verify_ready_weights(row, root)
                manifest = build_manifest(str(source), 'main', reader, api=None)
                if manifest['repo_id'] != 'lab/'+model_id or not manifest['revision'].startswith('local:'):
                    raise ValueError('derived model has no actual portable extractor identity')
                reader.identities[str(source)] = manifest['repo_id']+'@'+manifest['revision']
            manifest_path = write_manifest(manifest, run/'evidence/manifests')
            parent_ids = row.get('parents', [])
            if len(parent_ids) > 1:
                raise ValueError('incremental pair extraction currently requires one parent')
            before = reader.tensor_bytes.copy()
            fingerprint_path = fingerprint_model(manifest, reader, 'row_blocks', run/'evidence/row_blocks',
                parent=manifests[parent_ids[0]] if parent_ids else None,
                sketch_k=64, all_experts=True, read_threads=4)
            fp = load_fingerprint(fingerprint_path)
            reads = dict(reader.tensor_bytes-before)
            measured = dict(model_id=model_id, root_id=row['root_id'], kind=row['kind'],
                key=manifest['repo_id']+'@'+manifest['revision'], parents=parent_ids,
                manifest=manifest_path.relative_to(root).as_posix(),
                fingerprint=fingerprint_path.relative_to(root).as_posix(),
                manifest_sha256=sha(manifest_path), fingerprint_sha256=sha(fingerprint_path),
                ready_row_sha256=content_hash(row), tensor_bytes_read=sum(reads.values()),
                tensor_bytes_by_source=reads, extractor_reported_bytes=fp.meta['bytes_read'],
                elapsed_seconds=time.monotonic()-begin, fingerprint_seconds=fp.meta['elapsed_seconds'],
                source_weight_bytes=sum(p.stat().st_size for p in contained(root,row['path']).glob('*.safetensors')),
                synthetic=False)
            completed[model_id] = measured
            manifests[model_id] = manifest
            unresolved.remove(model_id)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            # Separate bounded family jobs may finish together. Publish each
            # immutable completion and merge under a lock rather than replacing
            # another worker's newly written progress with an old snapshot.
            with (progress_path.parent/'progress.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                latest = json.loads(progress_path.read_text()) if progress_path.exists() else {'schema_version':'1','rows':[]}
                merged = {item['model_id']:item for item in latest['rows']}
                merged[model_id] = measured
                atomic_json(progress_path.parent/'completions'/f'{model_id}.json', measured)
                latest['rows'] = [merged[k] for k in sorted(merged)]
                atomic_json(progress_path, latest)
                progress = latest
            print(json.dumps({'phase':'fingerprinted','model_id':model_id,
                'tensor_bytes_read':measured['tensor_bytes_read'], 'seconds':measured['elapsed_seconds']}), flush=True)
    return progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=os.environ.get('ZOO_DATA'))
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--models', nargs='*')
    args = parser.parse_args(argv)
    measure_ready(args.ready, data_root=args.data_root, run_root=args.run_root, model_ids=args.models)


if __name__ == '__main__':
    main()
