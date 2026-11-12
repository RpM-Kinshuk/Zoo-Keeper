"""Prepare pinned search pools using bounded config/header reads, never weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(ROOT / 'Model-Zoo'))
from evidence.manifest import build_manifest, write_manifest
from lab.artifacts import atomic_json
from zoo.schema import arch_signature


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RateLimited(RuntimeError):
    pass


def header_reader(**kwargs):
    path = ROOT / 'runs/track-a-k8-headers-20261108/headers.py'
    spec = importlib.util.spec_from_file_location('retained_header_guard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from evidence.hub import _transport
    transport = kwargs.pop('transport', _transport)
    def admitted_transport(*args):
        status, headers, body = transport(*args)
        if status == 429:
            raise RateLimited('metadata_rate_limited')
        return status, headers, body
    return module.HeaderOnlyReader(transport=admitted_transport, **kwargs)


def inventory_identity(manifest):
    inventory = []
    for row in manifest['files']:
        if not row['filename'].endswith('.safetensors'):
            raise ValueError('non_safetensors_inventory')
        sha, size = row.get('sha256'), row.get('size')
        if not isinstance(sha, str) or re.fullmatch('[0-9a-fA-F]{64}', sha) is None:
            raise ValueError('unknown_inventory_hash')
        if type(size) is not int or size < 0:
            raise ValueError('unknown_inventory_size')
        inventory.append((sha.lower(), size))
    if not inventory:
        raise ValueError('empty_inventory')
    return hashlib.sha256(json.dumps(sorted(inventory), separators=(',', ':')).encode()).hexdigest()


def compatibility_reason(manifest, base):
    if manifest['access'] != 'granted':
        return 'header_access'
    if manifest['is_adapter']:
        return 'adapter'
    if manifest['config'].get('quantization_config'):
        return 'quantization'
    if arch_signature(manifest['config'], []) != arch_signature(base['config'], []):
        return 'config_signature'
    def shapes(value):
        return {t['name']: tuple(t['shape']) for t in value['tensors']
                if not t['name'].endswith(('embed_tokens.weight', 'lm_head.weight'))}
    if shapes(manifest) != shapes(base):
        return 'tensor_shapes'
    if not manifest.get('params_by_dtype') or not set(manifest['params_by_dtype']) <= {'BF16', 'F16', 'F32', 'F64'}:
        return 'dtype'
    if sum(manifest.get('view_counts', {}).values()) <= 0:
        return 'no_views'
    return None


def resume_outcome(path, repo_id, measure):
    if path.exists():
        value = json.loads(path.read_text())
        if value['repo_id'] != repo_id:
            raise ValueError('saved_identity_mismatch')
        return value
    value = measure()
    if value['repo_id'] != repo_id:
        raise ValueError('measured_identity_mismatch')
    atomic_json(path, value)
    return value


class PinnedInfo:
    def __init__(self, info):
        self.info = info

    def model_info(self, repo_id, revision, files_metadata=True):
        if self.info.id.lower() != repo_id.lower() or self.info.sha != revision:
            raise ValueError('api_identity_mismatch')
        return self.info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=5400)
    parser.add_argument('--reuse-run', type=Path, action='append', default=[])
    args = parser.parse_args()
    if not 1 <= args.seconds <= 5400:
        parser.error('seconds must be in1..5400')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    discovery = ROOT / 'runs/scaling-pool-discovery-20261110'
    protocol = REPO / 'docs/steering/SCALING-POOL-PROTOCOL.md'
    candidate_path = discovery / 'candidates.csv'
    pilot = json.loads((discovery / 'pilot-results.json').read_text())
    cached = {r['repo_id']: r for r in pilot['rows'] if r.get('manifest')}
    cached_errors = {}
    reused_outcome_hashes = {}
    for previous_run in args.reuse_run:
        for anchor in ['q25-05','q3-06']:
            for path in sorted((previous_run/anchor).glob('*.json')):
                value=json.loads(path.read_text())
                reused_outcome_hashes[str(path.resolve().relative_to(ROOT))]=digest(path)
                if value.get('manifest'):
                    cached[value['repo_id']]=value
                else:
                    cached_errors[value['repo_id']]=(path,value)
    bases = {}
    for anchor, repo_id in [('q25-05', 'Qwen/Qwen2.5-0.5B'), ('q3-06', 'Qwen/Qwen3-0.6B')]:
        record = pilot['rows'][next(i for i,r in enumerate(pilot['rows']) if r['repo_id']==repo_id)]
        path = ROOT / record['manifest']
        if digest(path) != record['manifest_sha256']:
            raise ValueError('base_manifest_hash')
        bases[anchor] = json.loads(path.read_text())
    frozen = dict(driver_sha256=digest(__file__), protocol_sha256=digest(protocol),
                  candidates_sha256=digest(candidate_path), pilot_results_sha256=digest(discovery / 'pilot-results.json'),
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [
                      ROOT/'runs/track-a-k8-headers-20261108/headers.py', ROOT/'Model-Zoo/evidence/manifest.py',
                      ROOT/'Model-Zoo/evidence/hub.py', ROOT/'Model-Zoo/zoo/schema.py',
                      ROOT/'Model-Zoo/evidence/views.py', ROOT/'Model-Zoo/evidence/safetensors_header.py']},
                  seed=20261028, max_unique_per_pool=2000, tensor_payload_reads=0,
                  reused_outcome_hashes=reused_outcome_hashes)
    freeze_path = out/'freeze.json'
    if freeze_path.exists():
        if json.loads(freeze_path.read_text()) != frozen:
            raise ValueError('resume_requires_identical_inputs_and_code')
    else:
        atomic_json(freeze_path, frozen)
    rows = list(csv.DictReader(candidate_path.open()))
    from huggingface_hub import HfApi
    token = next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)), None)
    if not token:
        raise RuntimeError('approved_inherited_Hub_credential_required')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    started = time.monotonic()
    stop = threading.Event()
    atomic_json(out/'runtime.json', dict(hostname=socket.gethostname(), owned_pid=os.getpid(),
                                       gpu_devices=[], invocation_seconds=args.seconds, started_unix=time.time()))
    print('metadata_start', socket.gethostname(), os.getpid(), str(out), flush=True)

    def pool(anchor):
        base = bases[anchor]
        anchor_id = base['repo_id']
        all_rows = [r for r in rows if r['anchor'] == anchor]
        excluded = Counter()
        omitted = []
        candidates = []
        for row in all_rows:
            reason = ('snapshot_format' if row['weight_format'] != 'safetensors' else
                      'snapshot_adapter' if row['is_adapter'] == 'True' else
                      'snapshot_config_conflict_unprobed' if row['config_conflicts'] != '[]' else None)
            if reason and row['repo_id'] != anchor_id:
                excluded[reason] += 1
                omitted.append(dict(repo_id=row['repo_id'],reason=reason))
            else:
                candidates.append(row)
        atomic_json(out/f'{anchor}-snapshot-omitted.json',dict(rows=omitted))
        candidates.sort(key=lambda r: (r['repo_id'] != anchor_id,
                         hashlib.sha256(f"20261028:{anchor}:{r['repo_id']}".encode()).hexdigest(),r['repo_id']))
        if not candidates or candidates[0]['repo_id'] != anchor_id:
            raise ValueError('anchor_not_in_discovery')
        directory = out/anchor
        directory.mkdir(exist_ok=True)
        representatives, selected, outcomes = {}, [], []
        api = HfApi(token=token)
        reason_stop = 'exhausted'
        for row in candidates:
            if len(selected) >= 2000:
                reason_stop = 'target_reached'
                break
            if stop.is_set() or time.monotonic()-started >= args.seconds:
                reason_stop = 'rate_limit_or_other_pool_stop' if stop.is_set() else 'time_budget'
                break
            repo_id = row['repo_id']
            path = directory/(hashlib.sha256(repo_id.encode()).hexdigest()+'.json')
            def measure():
                if repo_id in cached_errors:
                    original_path,original=cached_errors[repo_id]
                    return dict(original,api_calls=0,reader_stats={},reused_manifest=False,
                                elapsed_seconds=0.,reused_error=str(original_path.resolve().relative_to(ROOT)))
                result = dict(repo_id=repo_id, api_calls=0, reader_stats={}, reused_manifest=False)
                case_start = time.monotonic()
                reader = header_reader(token=token)
                try:
                    previous = cached.get(repo_id)
                    if previous:
                        manifest_path = ROOT/previous['manifest']
                        if digest(manifest_path) != previous['manifest_sha256']:
                            raise ValueError('cached_manifest_hash_mismatch')
                        manifest = json.loads(manifest_path.read_text())
                        result['reused_manifest'] = True
                    else:
                        result['api_calls'] += 1
                        info = api.model_info(repo_id, revision='main',files_metadata=True,timeout=15)
                        result['revision'] = info.sha
                        manifest = build_manifest(repo_id, info.sha, reader, PinnedInfo(info))
                        manifest_path = write_manifest(manifest, out/'manifests')
                    result.update(revision=manifest['revision'],manifest=str(manifest_path.relative_to(ROOT)),
                                  manifest_sha256=digest(manifest_path))
                    reason = compatibility_reason(manifest,base)
                    if reason:
                        result.update(status='rejected',reason=reason)
                    else:
                        try:
                            identity = inventory_identity(manifest)
                        except ValueError as exc:
                            result.update(status='unknown_inventory',reason=str(exc))
                        else:
                            result.update(status='header_compatible',inventory_identity=identity,
                                          selected_weight_bytes=sum(f['size'] for f in manifest['files']),
                                          n_tensors=len(manifest['tensors']))
                except Exception as exc:
                    status_code = 429 if isinstance(exc, RateLimited) else getattr(getattr(exc, 'response', None),'status_code',None)
                    result.update(status='metadata_error',error_class=type(exc).__name__,http_status=status_code)
                    if status_code == 429:
                        stop.set()
                result.update(elapsed_seconds=time.monotonic()-case_start,reader_stats=dict(reader.stats))
                return result
            value = resume_outcome(path,repo_id,measure)
            if value.get('manifest') and digest(ROOT/value['manifest']) != value['manifest_sha256']:
                raise ValueError('saved_manifest_changed')
            item = dict(value)
            if item['status'] == 'header_compatible':
                identity = item['inventory_identity']
                if identity in representatives:
                    item.update(status='duplicate_inventory',representative=representatives[identity])
                else:
                    representatives[identity] = f"{repo_id}@{item['revision']}"
                    selected.append(dict(repo_id=repo_id,revision=item['revision'],inventory_identity=identity,
                                         manifest=item['manifest'],manifest_sha256=item['manifest_sha256'],
                                         selected_weight_bytes=item['selected_weight_bytes']))
            outcomes.append(item)
            if len(outcomes)%25 == 0:
                print('metadata_progress',anchor,'attempted',len(outcomes),'unique',len(selected),flush=True)
                atomic_json(out/f'{anchor}-progress.json',dict(attempted=len(outcomes),unique=len(selected),elapsed_seconds=time.monotonic()-started))
        # This is not an extraction roster until an owner-reviewed launch adds scientific query rules.
        atomic_json(out/f'{anchor}-outcomes.json',dict(rows=outcomes,snapshot_excluded=dict(excluded),stop=reason_stop))
        pool_path = out/f'{anchor}-pool.csv'
        temp = pool_path.with_suffix('.csv.tmp')
        with temp.open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=['repo_id','revision','inventory_identity','manifest','manifest_sha256','selected_weight_bytes'])
            writer.writeheader();writer.writerows(selected)
        temp.replace(pool_path)
        counts = dict(Counter(r['status'] for r in outcomes))
        return dict(anchor=anchor,unique=len(selected),attempted=len(outcomes),status_counts=counts,
                    snapshot_excluded=dict(excluded),stop=reason_stop,candidate_universe=len(candidates),
                    api_calls=sum(r['api_calls'] for r in outcomes),
                    reader_stats={k:sum(r['reader_stats'].get(k,0) for r in outcomes)
                                  for k in ['resolves','range_reads','bytes_read','retries','refreshes']},
                    reused_pilot_manifests=sum(r['reused_manifest'] for r in outcomes),
                    recorded_case_seconds=sum(r['elapsed_seconds'] for r in outcomes),
                    selected_weight_bytes=sum(r['selected_weight_bytes'] for r in selected),pool_sha256=digest(pool_path))
    with ThreadPoolExecutor(max_workers=2) as executor:
        summaries=list(executor.map(pool,['q25-05','q3-06']))
    summary=dict(pools=summaries,elapsed_seconds=time.monotonic()-started,freeze=frozen,
                 scope='Pinned header-compatible distinct inventories only; no payload/fingerprint or ancestry/independence validation. Reader bytes exclude API/wire overhead.')
    atomic_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
