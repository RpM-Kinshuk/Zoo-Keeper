"""One fresh acquisition through an isolated catalog and existing query CLI."""
import time
STARTED = time.monotonic()
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'Zoo-Keeper'), str(ROOT/'Model-Zoo'),
               str(ROOT/'Model-Zoo/esd_experiment/src')]
OUT = ROOT/'runs/query-journey-20261112'
QUERY = 'adityabanerjee13/qwen2.5-0.5b-cpt-mix-1to1'
REVISION = '56d3eaf20ab495fa26e63e82412268b6de537d48'
BASE = 'Qwen/Qwen2.5-0.5B'
INSTRUCT = 'Qwen/Qwen2.5-0.5B-Instruct'
OLD = ROOT/'runs/track-a-expanded-r6-20261105/zoo'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    import csv
    from lab.artifacts import atomic_json
    import pyarrow.parquet as pq
    assert not (OUT/'freeze.json').exists()
    pool = ROOT/'runs/scaling-pools-20261111-v2/q25-05-pool.csv'
    with pool.open() as handle:
        row = next(r for r in csv.DictReader(handle) if r['repo_id'] == QUERY and r['revision'] == REVISION)
    manifest = ROOT/row['manifest']; assert sha(manifest) == row['manifest_sha256']
    prior = ROOT/'runs/documented-slerp-20261112/record-index.json'
    entries = json.loads(prior.read_text())
    refs = [next(e for e in entries if e['key'].split('@')[0] == repo) for repo in [BASE, INSTRUCT]]
    models = pq.read_table(OLD/'models.parquet').to_pandas()
    assert QUERY+'@'+REVISION not in set(models.model_key)
    files = [Path(__file__), OUT/'PROTOCOL.md', pool, manifest, prior]
    files += [OLD/n for n in ['zoo.json', 'models.parquet', 'scores.parquet']]
    for e in refs:
        files += [ROOT/e[k] for k in ['manifest', 'fingerprint']]
    files += list((ROOT/'Zoo-Keeper/zoo_keeper').glob('*.py'))
    files += list((ROOT/'Model-Zoo/zoo').glob('*.py'))
    files += list((ROOT/'Model-Zoo/evidence').glob('*.py'))
    files += list((ROOT/'Model-Zoo/net_esd').glob('*.py'))
    files += [ROOT/'Model-Zoo/esd_experiment/src/measurement_config.py']
    atomic_json(OUT/'freeze.json', dict(query_manifest=row['manifest'], query_manifest_sha256=row['manifest_sha256'],
        query_key=QUERY+'@'+REVISION, references=refs, old_catalog=str(OLD.relative_to(ROOT)),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        policy=dict(max_new_weight_models=1, seconds=3600, cpu_threads=1, gpu_devices=[],
                    sketch_k=64, row_block=256, seed=20261028),
        scope='One fresh-model acquisition into a3-model reference catalog;703catalog metadata comparison separately. No performance recommendation or calibrated source decision.'))
    print('prepared1newquery,2cachedreferences,no payload', flush=True)


def run():
    from lab.artifacts import atomic_json
    import pandas as pd
    import torch
    from huggingface_hub import HfApi
    from evidence.hub import HubReader
    from evidence.manifest import write_manifest
    from evidence.fingerprint import fingerprint_model, load_fingerprint
    from evidence.frame import build_frame
    from zoo.build import build_zoo
    from zoo_keeper import Zoo
    from zoo_keeper.cli import main as query_cli
    import importlib.metadata

    assert (OUT/'prospective-review.md').exists()
    assert not (OUT/'results.json').exists(), 'preserve completed journey'
    freeze = json.loads((OUT/'freeze.json').read_text())
    for name, expected in freeze['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    torch.set_num_threads(1)
    runtime = dict(hostname=socket.gethostname(), owned_pid=os.getpid(), gpu_devices=[],
        python=platform.python_version(), packages={n: importlib.metadata.version(n) for n in ['torch','numpy','duckdb','safetensors','huggingface-hub']})
    atomic_json(OUT/'runtime.json', runtime)
    token = next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)),None)
    assert token, 'inherited Hub credential required'
    api = HfApi(token=token); reader = HubReader(token=token)
    stages = {}; metadata = []; manifests = {}
    for e in freeze['references']:
        m = json.loads((ROOT/e['manifest']).read_text()); manifests[m['repo_id']] = m
    manifests[QUERY] = json.loads((ROOT/freeze['query_manifest']).read_text())
    t = time.monotonic()
    for repo in [BASE, INSTRUCT, QUERY]:
        m = manifests[repo]
        info = api.model_info(repo, revision=m['revision'], files_metadata=True, timeout=15)
        assert info.sha == m['revision']
        total = getattr(info.safetensors, 'total', None)
        parameters = getattr(info.safetensors, 'parameters', None)
        row = dict(id=repo, revision=info.sha, pipeline_tag=info.pipeline_tag,
            library_name=info.library_name, tags=info.tags, downloads=info.downloads,
            safetensors=dict(total=total, parameters=parameters),
            siblings=[dict(rfilename=f.rfilename) for f in info.siblings])
        metadata.append(row)
        atomic_json(OUT/'metadata.json', dict(rows=metadata, api_calls=len(metadata), wire_bytes='not_instrumented'))
        write_manifest(m, OUT/'manifests')
    frame = build_frame(pd.DataFrame(metadata)); frame.to_csv(OUT/'frame.csv', index=False)
    stages['metadata_seconds'] = time.monotonic()-t
    print('metadata_complete', flush=True)
    # Audit metadata filters/score provenance without rerunning the old smoke demo.
    t = time.monotonic()
    def inspect_catalog(path):
        with Zoo(path, data_root=ROOT) as z:
            constraints = dict(modality='text', license=['apache-2.0'], params_max=1e9,
                               weight_format='safetensors', access='granted', architecture='qwen2', limit=10000)
            found = z.find(**constraints)
            import pyarrow.parquet as pq
            models = pq.read_table(Path(path)/'models.parquet').to_pandas()
            expected = models.loc[(models.modality=='text') & (models.license=='apache-2.0') &
                (models.params<=1e9) & (models.weight_format=='safetensors') &
                (models.access=='granted') & (models.architecture=='qwen2'), 'model_key']
            assert set(found.model_key) == set(expected)
            qwen = models.loc[models.architecture=='qwen2']
            scores = z.info(BASE)['measured_scores']
            return dict(constraints=constraints, matches=json.loads(found.to_json(orient='records')),
                model_count=len(models), qwen2_rows=len(qwen), missing_params=int(qwen.params.isna().sum()),
                missing_modality=int(qwen.modality.isna().sum()), base_scores=scores)
    old_result = inspect_catalog(OLD)
    atomic_json(OUT/'old-catalog.json', old_result)
    stages['old_catalog_query_seconds'] = time.monotonic()-t
    t = time.monotonic()
    fp_path = fingerprint_model(manifests[QUERY], reader, 'row_blocks', OUT/'fresh',
                                sketch_k=64, compute_device='cpu', read_threads=4)
    fp = load_fingerprint(fp_path)
    assert reader.stats['bytes_read'] > 0 and fp.meta['repo_id'] == QUERY and fp.meta['revision'] == REVISION
    stages['fingerprint_seconds'] = time.monotonic()-t
    atomic_json(OUT/'progress.json', dict(stage='fresh_fingerprint_complete', stages=stages, reader_stats=reader.stats))
    print('fresh_fingerprint_complete', flush=True)
    t = time.monotonic()
    scores = pd.read_parquet(OLD/'scores.parquet')
    keys = {r+'@'+m['revision'] for r,m in manifests.items()}
    subset = scores.loc[scores.model_key.isin(keys)].copy()
    subset.to_parquet(OUT/'scores-preserved.parquet', index=False)
    catalog = build_zoo(OUT/'manifests', [fp_path]+[ROOT/e['fingerprint'] for e in freeze['references']],
                        OUT/'catalog', ROOT, frame=OUT/'frame.csv', score_tables=[subset])
    stages['catalog_build_seconds'] = time.monotonic()-t
    fresh_result = inspect_catalog(catalog)
    preserved = pd.read_parquet(catalog/'scores.parquet')
    # Compare rows without relying on column ordering or serialization order.
    canonical = lambda df: sorted(json.dumps(r,sort_keys=True) for r in json.loads(df.to_json(orient='records')))
    assert canonical(subset) == canonical(preserved)
    atomic_json(OUT/'new-catalog.json', fresh_result)
    t = time.monotonic(); common = ['--zoo',str(catalog),'--data_root',str(ROOT),'--json']
    commands = [
        ['find','--modality','text','--license','apache-2.0','--params_max','1000000000','--format','safetensors','--access','granted','--architecture','qwen2','--limit','10'],
        ['find','--skill','instruction_following','--min_score','0','--limit','10'],
        ['info',QUERY+'@'+REVISION], ['similar',QUERY+'@'+REVISION,'-k','2'],
        ['family',QUERY+'@'+REVISION], ['diff',QUERY+'@'+REVISION,BASE], ['check',QUERY+'@'+REVISION]]
    outputs = []
    for args in commands:
        stdout, stderr = io.StringIO(), io.StringIO(); before=time.monotonic()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = query_cli(args+common)
        payload = json.loads(stdout.getvalue()) if stdout.getvalue().strip() else None
        outputs.append(dict(arguments=args+['--zoo',str(catalog.relative_to(ROOT)),'--data_root','<data_root>','--json'],
            exit_code=code, seconds=time.monotonic()-before, output=payload, stderr=stderr.getvalue()))
        atomic_json(OUT/'commands.json', outputs)
    stages['cli_query_seconds'] = time.monotonic()-t
    all_ok = all(r['exit_code']==0 and not r['stderr'] and r['output'] is not None for r in outputs)
    found_keys = {row['model_key'] for row in outputs[0]['output']}
    atomic_json(OUT/'results.json', dict(completed=True, cli_ok=all_ok,
        fresh_query_in_constrained_shortlist=QUERY+'@'+REVISION in found_keys,
        old_catalog=old_result,new_catalog=fresh_result,stages=stages,
        inner_process_seconds=time.monotonic()-STARTED, reader_stats=reader.stats,
        fresh_fingerprint=str(fp_path.relative_to(ROOT)),fresh_fingerprint_sha256=sha(fp_path),
        reused_reference_fingerprints=[e['fingerprint'] for e in freeze['references']],
        metadata_api_calls=len(metadata), metadata_wire_bytes='not_instrumented',
        catalog=str(catalog.relative_to(ROOT)),runtime=runtime,freeze_sha256=sha(OUT/'freeze.json'),
        measured_score_rows_preserved=len(subset),
        interpretation=freeze['scope']))
    print('complete',all_ok,'fresh_shortlisted',QUERY+'@'+REVISION in found_keys,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    try:
        (prepare if args.prepare else run)()
    except Exception as exc:
        from lab.artifacts import atomic_json
        atomic_json(OUT/'failure.json',dict(error_class=type(exc).__name__,seconds=time.monotonic()-STARTED))
        print('journey_failed',type(exc).__name__,file=sys.stderr,flush=True)
        raise SystemExit(1) from None


if __name__=='__main__':
    main()
