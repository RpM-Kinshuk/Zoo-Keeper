"""One bounded complete-matrix comparison; reuse native scoring and decoding."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
SCOPE=ROOT/'runs/benchmark-coverage-20261113/real-comparison-inputs.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    from lab.artifacts import atomic_json
    atomic_json(path,value)


def extract_profile(record,fetch,progress=lambda n:None):
    import numpy as np
    from studies.baselines.residual import signature
    trace={role:[] for role in record['trace_roles']};vectors=[];concentrations=[]
    for layer in range(record['layers']):
        names=[f'model.layers.{layer}.{role}' for role in trace]
        matrices=fetch(names)
        if len(matrices)!=len(names):raise ValueError('incomplete layer')
        values={}
        for role,matrix in zip(trace,matrices):
            a=np.asarray(matrix,dtype=np.float64)
            if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('invalid complete matrix')
            trace[role].append(float(np.linalg.norm(a)))
            if role in ['mlp.down_proj.weight','mlp.up_proj.weight']:values[role]=a
        v,c=signature(values['mlp.down_proj.weight'],values['mlp.up_proj.weight'])
        if v.size!=record['hidden']**2:raise ValueError('incompatible hidden width')
        vectors.append(v);concentrations.append(c);progress(layer+1)
    return dict(trace=trace,vectors=np.stack(vectors),concentrations=np.asarray(concentrations))


def save_profile(path,profile):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    roles=sorted(profile['trace'])
    with tmp.open('wb') as f:
        np.savez(f,roles=np.asarray(roles),energy=np.asarray([profile['trace'][r] for r in roles]),
                 vectors=profile['vectors'],concentrations=profile['concentrations'])
    tmp.replace(path)
    return dict(sha256=sha(path),size=path.stat().st_size)


def load_profile(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        return dict(trace=dict(zip(z['roles'].tolist(),z['energy'].tolist())),vectors=z['vectors'],concentrations=z['concentrations'])


def score_panel(case,profiles):
    from studies.baselines.lab_spectral import similarity
    from studies.baselines.residual import score
    import math
    q=profiles[case['query_key']];candidates=sorted(set().union(*map(set,case['pool_settings'].values())))
    scores={'Trace':{},'TLT':{}};rows=[]
    for key in candidates:
        c=profiles[key]
        scores['Trace'][key]=similarity(q['trace'],c['trace'],'Trace-native')
        scores['TLT'][key]=score((q['vectors'],q['concentrations']),(c['vectors'],c['concentrations']))[0]
        for method in scores:
            value=float(scores[method][key]);assert math.isfinite(value)
            rows.append(dict(method=method,candidate_key=key,score=value))
    pools={}
    for name,keys in case['pool_settings'].items():
        pools[name]={}
        for method,values in scores.items():
            best=max(values[k] for k in keys)
            pools[name][method]=dict(top=sorted(k for k in keys if values[k]==best),
                ranks={k:1+sum(values[n]>values[k] for n in keys) for k in keys},source_decision='unsupported')
    return dict(query_key=case['query_key'],pairs=rows,pools=pools)


def prepare(out):
    assert not (out/'inputs.json').exists()
    scope=json.loads(SCOPE.read_text())
    for path,h in scope['source_sha256'].items():assert sha(ROOT/path)==h,path
    files=[Path(__file__),ROOT/'Zoo-Keeper/tests/test_real_comparison.py',SCOPE,
        ROOT/'Zoo-Keeper/lab/artifacts.py',ROOT/'Zoo-Keeper/lab/schema.py',
        ROOT/'Model-Zoo/evidence/fingerprint.py',ROOT/'Model-Zoo/evidence/hub.py']
    for case in scope['cases']:files.append(ROOT/case['other_checks']['source'])
    references={}
    for case in scope['cases']:
        keys=set().union(*map(set,case['pool_settings'].values()))
        repos={'Qwen/Qwen2.5-0.5B','Qwen/Qwen2.5-0.5B-Instruct'} if case['operator']=='SLERP' else ({'Qwen/Qwen3-0.6B'} if case['operator']=='FP8' else {'Qwen/Qwen2.5-0.5B-Instruct'})
        refs=sorted(k for k in keys if k.rsplit('@',1)[0] in repos);assert len(refs)==len(repos)
        references[case['query_key']]=refs
    write(out/'inputs.json',dict(scope=str(SCOPE.relative_to(ROOT)),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        protocol_sha256=hashlib.sha256((out/'RESEARCH.md').read_text().split('## Outcome')[0].encode()).hexdigest(),
        documented_reference_keys=references))
    print('prepared fixed11models/3queries/8pools; no weights read',flush=True)


def stat_token(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def guarded_transport(admit,budget):
    def transport(method,url,headers):
        admit()
        return budget.transport(method,url,headers)
    return transport


def run(out):
    import signal,socket,threading,shutil,resource
    from datetime import datetime,timezone
    from scripts.task_choice_download import AcquisitionStopped,ResponseBudget
    started=time.monotonic();epoch=time.time();deadline=started+6600
    assert not (out/'runtime.json').exists(),'never overwrite or retry this fixed campaign'
    stop_event=threading.Event();workers=None;reads=None
    def stop(*_):raise AcquisitionStopped('original campaign deadline/termination')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(6600)
    runtime=dict(status='VERIFYING',host=socket.gethostname(),host_pid=os.getpid(),gpu_devices=[],
        started_utc=datetime.now(timezone.utc).isoformat(),deadline_unix=epoch+6600,models={},local_sha_bytes=0,
        stop_command=f'kill -TERM {os.getpid()}',cpu_model_workers=2,blas_threads=8,torch_threads=1)
    budget=ResponseBudget(limit=8_000_000_000)
    def publish():
        runtime.update(elapsed_seconds=time.monotonic()-started,reserved_response_bytes=budget.reserved)
        write(out/'runtime.json',runtime)
    def admit():
        if stop_event.is_set() or time.monotonic()>=deadline:raise AcquisitionStopped('fixed deadline/cohort stopped')
    publish()
    try:
        p=json.loads((out/'inputs.json').read_text());scope=json.loads((ROOT/p['scope']).read_text())
        for paths in [p['source_sha256'],scope['source_sha256']]:
            for path,h in paths.items():admit();assert sha(ROOT/path)==h,path
        assert hashlib.sha256((out/'RESEARCH.md').read_text().split('## Outcome')[0].encode()).hexdigest()==p['protocol_sha256']
        assert len(scope['models'])==11 and scope['summary']['distinct_weight_models']==11
        mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
        assert mem>=64*1024**2 and shutil.disk_usage(out).free>=4_000_000_000
        assert len(os.sched_getaffinity(0))>=16
        runtime.update(memory_available_kib=mem,free_output_bytes=shutil.disk_usage(out).free)
        receipts={}
        for key,record in scope['models'].items():
            if not record['complete_local_files']:continue
            receipts[key]={}
            for f in record['required_files']:
                admit();path=ROOT/record['local_path']/f['filename'];assert not path.is_symlink()
                before=path.stat();assert sha(path)==f['sha256'];assert stat_token(path.stat())==stat_token(before)
                receipts[key][f['filename']]=stat_token(before);runtime['local_sha_bytes']+=before.st_size
            publish()
        write(out/'local-receipts.json',receipts)
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from evidence.hub import HubReader
        token=next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)),None);assert token
        reader=HubReader(token=token,transport=guarded_transport(admit,budget))
        runtime.update(status='PROFILING',verification_seconds=time.monotonic()-started,torch_version=torch.__version__);publish()
        reads=ThreadPoolExecutor(max_workers=4);workers=ThreadPoolExecutor(max_workers=2)
        def job(index,key,record):
            from evidence.fingerprint import _entries
            from evidence.views import plan_views,decode_view
            from evidence.safetensors_header import decode_plain
            model_started=time.monotonic();fds={};path=out/'models'/f'{index:02d}.json'
            state=dict(model_key=key,status='READING',complete_layers=0)
            def progress(n):
                admit();state.update(complete_layers=n,elapsed_seconds=time.monotonic()-model_started);write(path,state)
            try:
                admit();manifest=json.loads((ROOT/record['manifest']).read_text());assert sha(ROOT/record['manifest'])==record['manifest_sha256']
                entries=_entries(manifest);views={v.name:v for v in plan_views(entries,manifest['config'])[0]}
                required={e['name'] for e in record['physical_tensors']};seen=set();lock=threading.Lock();remotes={}
                for f in record['required_files']:
                    if record['complete_local_files']:
                        fd=os.open(ROOT/record['local_path']/f['filename'],os.O_RDONLY|os.O_NOFOLLOW)
                        fds[f['filename']]=fd;assert stat_token(os.fstat(fd))==receipts[key][f['filename']]
                    else:
                        remote=reader.resolve(manifest['repo_id'],f['filename'],manifest['revision'])
                        assert remote.size==f['size'] and remote.sha256==f['sha256'];remotes[f['filename']]=remote
                def physical(name,rows):
                    admit();assert rows is None and name in required
                    e=entries[name]
                    with lock:
                        assert name not in seen,'required tensor read twice';seen.add(name)
                    raw=os.pread(fds[e.file],e.end-e.start,e.start) if fds else reader.read(remotes[e.file],e.start,e.end)
                    return decode_plain(raw,e.dtype,tuple(e.shape))
                def matrix(name):
                    v=views[name];assert v.decode in ['plain','fp8_scaled'] and v.geometry=='matrix'
                    value=decode_view(v,physical,manifest['config'])
                    assert tuple(value.shape)==tuple(v.matrix_shape)
                    return value.numpy()
                profile=extract_profile(record,lambda names:list(reads.map(matrix,names)),progress)
                assert seen==required,'incomplete fixed matrix/scale input'
                for name,fd in fds.items():assert stat_token(os.fstat(fd))==receipts[key][name]
                relative=f'profiles/{index:02d}.npz';saved=save_profile(out/relative,profile)
                state.update(status='COMPLETE',elapsed_seconds=time.monotonic()-model_started,
                    profile_path=relative,profile_sha256=saved['sha256'],profile_bytes=saved['size'],
                    physical_tensor_bytes=record['required_tensor_bytes'],physical_tensors=len(seen),
                    input_basis='fresh complete-file SHA plus unchanged fd/stat' if fds else 'pinned revision and server file SHA; complete selected tensor ranges')
                write(path,state);return key,profile,state
            except BaseException as exc:
                state.update(status='FAILED',error_class=type(exc).__name__,reason=str(exc));write(path,state);raise
            finally:
                for fd in fds.values():os.close(fd)
        jobs={workers.submit(job,i,k,r):k for i,(k,r) in enumerate(scope['models'].items())};profiles={};work=0
        profile_started=time.monotonic()
        while jobs:
            admit();done,_=wait(jobs,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                key,profile,state=future.result();profiles[key]=profile;runtime['models'][key]=state
                work+=scope['models'][key]['required_tensor_bytes'];del jobs[future]
                if len(profiles)>=2:
                    eta=(time.monotonic()-profile_started)/work*(scope['summary']['required_tensor_bytes']-work)
                    if eta>deadline-time.monotonic():raise AcquisitionStopped('coarse completed-model throughput exceeds original deadline')
            publish()
        assert len(profiles)==11;runtime['status']='SCORING';publish()
        results=[]
        for case in scope['cases']:
            admit();result=score_panel(case,profiles);result['operator']=case['operator']
            result['documented_reference_keys']=p['documented_reference_keys'][case['query_key']]
            result['existing_native_and_operation_checks']=dict(source=case['other_checks']['source'],information_contracts=case['method_contracts'])
            results.append(result)
        runtime.update(status='COMPLETE',reader_stats=reader.stats,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);publish()
        write(out/'results.json',dict(cases=results,scope='exposed fixed relatedness-ranking diagnostic; no calibrated source decisions or population accuracy',
            inputs_sha256=sha(out/'inputs.json'),runtime_sha256=sha(out/'runtime.json')))
        signal.alarm(0);print('COMPLETE',runtime['elapsed_seconds'],flush=True)
    except BaseException as exc:
        runtime.update(status='FAILED',error_class=type(exc).__name__,reason=str(exc));publish();raise
    finally:
        stop_event.set()
        if workers is not None:workers.shutdown(wait=False,cancel_futures=True)
        if reads is not None:reads.shutdown(wait=False,cancel_futures=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.prepare:prepare(out)
    else:run(out)
