"""Fixed official-FP8 source compatibility case, not a calibrated history claim."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Zoo-Keeper'),str(ROOT/'Model-Zoo'),str(ROOT/'Model-Zoo/esd_experiment/src')]


def compare_band(candidate,codes,scales):
    """Two exact, separately reported construction checks; no fitted tolerance."""
    import torch
    from transformers.integrations.finegrained_fp8 import Fp8Quantize
    if (candidate.ndim!=2 or candidate.shape!=codes.shape or candidate.shape[0]%128
            or candidate.shape[1]%128 or scales.shape!=(candidate.shape[0]//128,candidate.shape[1]//128)
            or codes.dtype!=torch.float8_e4m3fn or not torch.isfinite(candidate).all()
            or not torch.isfinite(codes.float()).all() or not torch.isfinite(scales).all() or not (scales>0).all()):
        raise ValueError('finite complete128tiles and positive recorded scales required')
    expanded=scales.float().repeat_interleave(128,0).repeat_interleave(128,1)
    observed=(candidate.float()/expanded).clamp(-448,448).to(torch.float8_e4m3fn)
    operator=Fp8Quantize(SimpleNamespace(quantization_config={'weight_block_size':[128,128]}))
    replay=operator._quantize_one('projection.weight',candidate)
    expected_codes=replay['projection.weight'];expected_scales=replay['projection.weight_scale_inv'].to(scales.dtype)
    return dict(elements=codes.numel(),tiles=scales.numel(),
        observed_code_mismatches=int((observed.view(torch.uint8)!=codes.view(torch.uint8)).sum()),
        replay_code_mismatches=int((expected_codes.view(torch.uint8)!=codes.view(torch.uint8)).sum()),
        replay_scale_mismatches=int((expected_scales.view(torch.uint8)!=scales.contiguous().view(torch.uint8)).sum()),
        squared_error=float(((candidate.double()-codes.float().double()*expanded.double())**2).sum()),
        query_squared_norm=float(((codes.float().double()*expanded.double())**2).sum()))


def decide(matches):
    compatible=sorted(k for k,v in matches.items() if v)
    return dict(answer=compatible[0] if len(compatible)==1 else None,compatible=compatible,
                scope='Exact compatibility on measured bands under stated arithmetic; no universal historical uniqueness/calibration.')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()


def prepare(out):
    import modeldna
    import torch
    import transformers.integrations.finegrained_fp8 as published
    from modeldna.fingerprint.extract import Fingerprint,DEFAULT_SEED
    from modeldna.verdict import Thresholds
    from evidence.fingerprint import _entries
    from evidence.views import plan_views
    from lab.artifacts import atomic_json
    assert modeldna.__version__=='0.1.0' and not (out/'freeze.json').exists()
    inputs=json.loads((out/'inputs.json').read_text())
    external={r['role']:r for r in inputs['records']}
    manifests={role:json.loads((ROOT/r['manifest']).read_text()) for role,r in external.items()}
    for r in external.values():assert sha(ROOT/r['manifest'])==r['manifest_sha256']
    entries=_entries(manifests['query']);views,coverage=plan_views(entries,manifests['query']['config'])
    quantized=sorted((v for v in views if v.decode=='fp8_scaled'),key=lambda v:v.name)
    assert len(quantized)==196 and {v.name for v in quantized}=={n for n,e in entries.items() if e.dtype.startswith('F8_')}
    plan=[]
    for v in quantized:
        rows,cols=v.matrix_shape;assert rows%128==cols%128==0 and entries[v.sources[1]].dtype=='BF16'
        first=128*(int(hashlib.sha256(('20261028:'+v.name).encode()).hexdigest(),16)%(rows//128))
        plan.append(dict(name=v.name,first=first,last=first+128,cols=cols,scale_name=v.sources[1]))
    atomic_json(out/'band-plan.json',plan)
    formal_path=ROOT/'runs/track-b-pivot-unblock-a-20261105/active-roster-v2/artifact-map166.json'
    formal=json.loads(formal_path.read_text())['rows']
    candidates={'parent':dict(model_id='q3-06',local_dir='lab/inputs/cohort-v1-q3-06-c1899de289a0',manifest=external['parent']['manifest'])}
    for role,mid in [('sft','q3-06-sft-s0'),('dpo','q3-06-dpo-s0'),('adapter','q3-06-lora-dialogue-r64')]:
        r=next(x for x in formal if x['model_id']==mid);assert sha(ROOT/r['manifest'])==r['manifest_sha256']
        candidates[role]=dict(model_id=mid,local_dir='lab/'+mid,manifest=r['manifest'])
    old_native=ROOT/'runs/track-c-k2-modeldna-20261108'
    old_ledger_path=old_native/'inputs.json';old_ledger=json.loads(old_ledger_path.read_text())['run_inputs']
    paths=[Path(__file__),out/'inputs.json',out/'PROTOCOL.md',out/'band-plan.json',formal_path,old_ledger_path,
        ROOT/'Zoo-Keeper/studies/baselines/fp8_input.py',ROOT/'Zoo-Keeper/studies/baselines/native_source.py',
        ROOT/'Zoo-Keeper/scripts/task_choice_download.py',ROOT/'Zoo-Keeper/tests/test_fp8_input.py',ROOT/'Zoo-Keeper/tests/test_documented_fp8.py',Path(published.__file__)]
    for role,r in candidates.items():
        m=json.loads((ROOT/r['manifest']).read_text());r['weights']={f['filename']:f['sha256'] for f in m['files'] if f['filename'].endswith('.safetensors')}
        for name,h in r['weights'].items():assert old_ledger[r['local_dir']+'/'+name]==h
        r['native']=str((old_native/(r['model_id']+'.json')).relative_to(ROOT))
        native=Fingerprint.from_dict(json.loads((ROOT/r['native']).read_text()))
        assert native.model_id==r['model_id'] and native.mode=='fast' and native.seed==DEFAULT_SEED
        paths+=[ROOT/r['manifest'],ROOT/r['native'],ROOT/r['local_dir']/'config.json']
        if role!='parent':paths += [ROOT/r['local_dir']/'recipe.json',ROOT/r['local_dir']/'lineage.json']
    for r in external.values():paths+=[ROOT/r['manifest']]+[ROOT/x['path'] for x in r['sources']]
    paths+=list((ROOT/'Model-Zoo/evidence').glob('*.py'))
    package=Path(modeldna.__file__).parent
    installed_path=ROOT/'runs/track-c-k2-remaining-20261108/modeldna/installed-package.json'
    installed=json.loads(installed_path.read_text())
    for name,h in installed['source_file_sha256'].items():assert sha(package.parent/name)==h
    paths+=list(package.rglob('*.py'))+[installed_path]
    atomic_json(out/'freeze.json',dict(source_sha256={str(p.resolve()):sha(p) for p in paths},external=external,candidates=candidates,
        pools=dict(present=sorted(candidates),removed=sorted(set(candidates)-{'parent'})),native_version=modeldna.__version__,native_seed=DEFAULT_SEED,
        native_thresholds=vars(Thresholds()),band_count=len(plan),coverage=coverage,query_config=manifests['query']['config'],
        budget=dict(seconds=6300,outer_seconds=6600,response_bytes=1_600_000_000,weight_identities=5,cpu_threads=1),
        exposure='One new official FP8 query in an exposed Qwen root with exposed lab/native candidates; no untouched-root or calibrated-C3 claim.'))
    print('prepared196bands,4cachedcandidates,sourcepresent/removed,noquerypayload',flush=True)


def run(out):
    import os,socket,time,signal,platform
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack
    from datetime import datetime,timezone
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from modeldna.fingerprint.extract import Fingerprint,DEFAULT_SEED
    from modeldna.compare import compare_fingerprints
    from modeldna.verdict import judge
    from evidence.hub import HubReader
    from evidence.fingerprint import _entries
    from evidence.views import decode_view,plan_views
    from lab.artifacts import atomic_json
    from scripts.task_choice_download import ResponseBudget,AcquisitionStopped
    from studies.baselines.native_source import PinnedSource
    from studies.baselines.fp8_input import decoded_extractor
    torch.set_num_threads(1)
    def stop(signum,frame):raise AcquisitionStopped('fixed FP8 case deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(6300)
    started=time.monotonic();freeze=json.loads((out/'freeze.json').read_text())
    assert (out/'prospective-review.md').exists() and not (out/'runtime.json').exists()
    runtime=dict(status='VERIFYING',host=socket.gethostname(),host_pid=os.getpid(),gpu_devices=[],started_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),torch=torch.__version__,local_hash_scan_bytes=0,local_band_bytes=0,stages=[],stop_command=f'kill -TERM {os.getpid()}')
    def publish():
        runtime['elapsed_seconds']=time.monotonic()-started
        atomic_json(out/'runtime.json',runtime)
    publish();print(json.dumps(dict(host=runtime['host'],host_pid=os.getpid(),gpu_devices=[])),flush=True)
    budget=ResponseBudget(limit=freeze['budget']['response_bytes'])
    token=next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)),None);assert token
    reader=HubReader(token=token,transport=budget.transport)
    try:
        for name,h in freeze['source_sha256'].items():assert sha(Path(name))==h,name
        plan=json.loads((out/'band-plan.json').read_text());query=freeze['external']['query']
        manifest=json.loads((ROOT/query['manifest']).read_text());entries=_entries(manifest)
        views={v.name:v for v in plan_views(entries,manifest['config'])[0] if v.decode=='fp8_scaled'}
        with ExitStack() as stack:
            handles={};natives={}
            for role,r in freeze['candidates'].items():
                loc={}
                for name,h in r['weights'].items():
                    p=ROOT/r['local_dir']/name;assert sha(p)==h,'local frozen weight changed'
                    runtime['local_hash_scan_bytes']+=p.stat().st_size
                    handle=stack.enter_context(safe_open(str(p),framework='pt',device='cpu'))
                    for key in handle.keys():
                        assert key not in loc;loc[key]=handle
                handles[role]=loc
                natives[role]=Fingerprint.from_dict(json.loads((ROOT/r['native']).read_text()))
                publish();print('verified_local',role,flush=True)
            runtime['verification_seconds']=time.monotonic()-started;runtime['status']='READING_BANDS';publish()
            remotes={}
            def read_query(name,rows):
                from evidence.safetensors_header import decode_plain
                e=entries[name];shape=tuple(e.shape);start=e.start;end=e.end
                if rows is not None:
                    a,b,step=rows.indices(shape[0]);assert step==1
                    stride=(e.end-e.start)//shape[0];start+=a*stride;end=e.start+b*stride;shape=(b-a,*shape[1:])
                if e.file not in remotes:
                    remote=reader.resolve(query['repo_id'],e.file,query['revision'])
                    expected=next(f for f in manifest['files'] if f['filename']==e.file)
                    assert remote.size==expected['size'] and remote.sha256==expected['sha256'];remotes[e.file]=remote
                return decode_plain(reader.read(remotes[e.file],start,end),e.dtype,shape)
            measurements=[];artifacts={};begin=time.monotonic()
            for i,p in enumerate(plan):
                name=p['name'];row=slice(p['first'],p['last']);v=views[name]
                codes=read_query(name,row);scales=read_query(p['scale_name'],slice(p['first']//128,p['last']//128))
                # Independently exercise the existing decoder for the same physical band.
                qvalues=decode_view(v,lambda n,r: codes if n==name else scales,manifest['config'],row_slice=row)
                assert qvalues.shape==codes.shape
                band={'query_codes':codes,'query_scales':scales};comparison={}
                for role,loc in handles.items():
                    assert name in loc and tuple(loc[name].get_slice(name).get_shape())==tuple(entries[name].shape)
                    candidate=loc[name].get_slice(name)[p['first']:p['last'],:];runtime['local_band_bytes']+=candidate.numel()*candidate.element_size()
                    band[role]=candidate;comparison[role]=compare_band(candidate,codes,scales)
                    assert comparison[role]['query_squared_norm']>=0
                path=out/'bands'/f'{i:03d}.safetensors';path.parent.mkdir(exist_ok=True)
                save_file(band,str(path),metadata=dict(tensor=name,row_start=str(p['first']),scope='private reproduction input; not for publication'))
                artifacts[str(path.relative_to(ROOT))]=sha(path)
                measurements.append(dict(plan=p,path=str(path.relative_to(ROOT)),sha256=artifacts[str(path.relative_to(ROOT))],candidates=comparison))
                if (i+1)%28==0:print('bands',i+1,'of196',flush=True);publish()
            atomic_json(out/'bands.json',measurements)
            runtime['band_seconds']=time.monotonic()-begin;runtime['status']='NATIVE_DECODED_INPUT';publish()
        with ThreadPoolExecutor(4) as pool:
            files=[f['filename'] for f in manifest['files']]+['config.json','tokenizer.json']
            source=PinnedSource(query['repo_id'],query['revision'],files,reader,pool,deadline=started+6200,byte_cap=1_500_000_000)
            begin=time.monotonic();native=decoded_extractor(source,entries,manifest['config'],mode='fast',seed=DEFAULT_SEED,
                model_id=query['repo_id']+'@'+query['revision'],revision=query['revision']).run()
            native_path=out/'native-query-decoded.json';atomic_json(native_path,native.to_dict());artifacts[str(native_path.relative_to(ROOT))]=sha(native_path)
            runtime['native_seconds']=time.monotonic()-begin;runtime['native_source_bytes']=source.bytes_read;publish()
        pairs={role:compare_fingerprints(native,fp) for role,fp in natives.items()}
        pools={}
        for pool,roles in freeze['pools'].items():
            checks={}
            for rule,fields in [('observed_scale',['observed_code_mismatches']),('published_replay',['replay_code_mismatches','replay_scale_mismatches'])]:
                matches={role:all(all(row['candidates'][role][field]==0 for field in fields) for row in measurements) for role in roles}
                checks[rule]=decide(matches)
            errors={role:sum(row['candidates'][role]['squared_error'] for row in measurements) for role in roles}
            norm=sum(row['candidates'][roles[0]]['query_squared_norm'] for row in measurements)
            ranking=sorted(roles,key=lambda role:(errors[role],role))
            verdict=judge([(pairs[role],'qwen3') for role in roles],suspect_dtype=native.arch.torch_dtype,
                candidate_dtypes={natives[role].model_id:natives[role].arch.torch_dtype for role in roles})
            pools[pool]=dict(candidates=roles,checks=checks,relative_l2={r:(errors[r]/norm)**.5 for r in roles},similarity_ranking=ranking,
                modeldna_decoded_input_verdict=verdict.to_dict(),historical_source_probabilities='not_calibrated_for_FP8')
        totals={role:{field:sum(row['candidates'][role][field] for row in measurements) for field in measurements[0]['candidates'][role]} for role in freeze['candidates']}
        runtime.update(status='COMPLETE',reserved_response_bytes=budget.reserved,reader_stats=reader.stats);publish()
        result=dict(pools=pools,totals=totals,native_pairs={r:e.to_dict() for r,e in pairs.items()},runtime=runtime,
            artifacts=artifacts,freeze_sha256=sha(out/'freeze.json'),native_raw_fp8_support='INVALID_uint8_codes_without_scales; numerical result uses explicit dense-input adapter',
            scope='One documented vendor query, sampled complete tiles, exposed root/relatives; exact arithmetic compatibility only, no general source guarantee or approved claim.')
        print('COMPLETE',json.dumps({p:{k:v['answer'] for k,v in r['checks'].items()} for p,r in pools.items()}),flush=True)
    except BaseException as exc:
        runtime.update(status='FAILED',error_class=type(exc).__name__);publish();raise
    finally:
        runtime.update(reserved_response_bytes=budget.reserved,reader_stats=reader.stats,finished_utc=datetime.now(timezone.utc).isoformat());publish()
        if runtime['status']=='COMPLETE':
            result['runtime']=runtime
            atomic_json(out/'results.json',result)
        signal.alarm(0)


if __name__=='__main__':
    import argparse,os
    os.environ['CUDA_VISIBLE_DEVICES']=''
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--out',type=Path,default=ROOT/'runs/documented-fp8-20261112')
    args=parser.parse_args()
    prepare(args.out) if args.prepare else run(args.out)
