"""One fixed real-recipe comparison using existing Zoo and native modelDNA fits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,socket,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Zoo-Keeper'),str(ROOT/'Model-Zoo')]
from lab.artifacts import atomic_json


def candidate_pools(keys,sources):
    a,b=sources
    return dict(present=sorted(keys),remove_base=sorted(k for k in keys if k!=a),
                remove_instruct=sorted(k for k in keys if k!=b),
                remove_both=sorted(k for k in keys if k not in sources))


def zoo_inputs(records,keys):
    if not keys:raise ValueError('empty_pool')
    return dict(parents=[records[k] for k in keys],base=records[keys[0]])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();out=ROOT/'runs/documented-slerp-20261112'
    from modeldna import __version__
    from modeldna.fingerprint.extract import extract_fingerprint,Fingerprint,DEFAULT_SEED
    from modeldna.merge import decompose_fingerprints,MATERIAL_ALPHA,MERGE_GAIN,EXPLAINED_COS
    from evidence.hub import HubReader
    from evidence.manifest import build_manifest,write_manifest
    from evidence.fingerprint import fingerprint_model,load_fingerprint
    from lineage.upstream import load_records
    from lineage.merge import fit_merge
    from studies.baselines.native_source import PinnedSource
    from zoo.schema import arch_signature
    import modeldna.merge,modeldna.fingerprint.extract,torch
    assert __version__=='0.1.0'
    inp=json.loads((out/'inputs.json').read_text());external={r['role']:r for r in inp['records']}
    formal_path=ROOT/'runs/track-b-pivot-unblock-a-20261105/active-roster-v2/artifact-map166.json'
    formal=json.loads(formal_path.read_text())['rows']
    controls={}
    for role,mid in [('sft','q25-05-sft-s0'),('adapter','q25-05-lora-dialogue-r8')]:
        original=next(r for r in formal if r['model_id']==mid)
        controls[role]=dict(model_id=mid,manifest=original['manifest'],manifest_sha256=original['manifest_sha256'],
          native=f'runs/track-c-k2-modeldna-20261108/{mid}.json',local_dir=f'lab/{mid}')
    source_files=[Path(__file__),out/'PROTOCOL.md',out/'inputs.json',formal_path,
       ROOT/'Zoo-Keeper/lineage/merge.py',ROOT/'Zoo-Keeper/lineage/records.py',ROOT/'Zoo-Keeper/lineage/upstream.py',
       ROOT/'Zoo-Keeper/studies/baselines/native_source.py',Path(modeldna.merge.__file__),Path(modeldna.fingerprint.extract.__file__)]
    installed_path=ROOT/'runs/track-c-k2-remaining-20261108/modeldna/installed-package.json'
    installed=json.loads(installed_path.read_text())
    package_root=Path(modeldna.merge.__file__).parent.parent
    for name,h in installed['source_file_sha256'].items():
        path=package_root/name
        assert sha(path)==h,'native package source changed since cached controls'
        source_files.append(path)
    source_files.append(installed_path)
    source_files+=list((ROOT/'Model-Zoo/evidence').glob('*.py'))
    source_files+=list((ROOT/'Model-Zoo/net_esd').glob('*.py'))
    source_files += [ROOT/'Model-Zoo/esd_experiment/src/measurement_config.py',ROOT/'Model-Zoo/zoo/schema.py']
    for r in external.values():
        assert sha(ROOT/r['manifest'])==r['manifest_sha256']
        source_files.append(ROOT/r['manifest'])
        for p in r['sources']:
            assert sha(ROOT/p['path'])==p['sha256'];source_files.append(ROOT/p['path'])
    ledger_path=ROOT/'runs/track-c-k2-modeldna-20261108/inputs.json'
    source_files.append(ledger_path);ledger=json.loads(ledger_path.read_text())['run_inputs']
    for r in controls.values():
        assert sha(ROOT/r['manifest'])==r['manifest_sha256']
        source_files += [ROOT/r['manifest'],ROOT/r['native']]
        cached_native=Fingerprint.from_dict(json.loads((ROOT/r['native']).read_text()))
        assert cached_native.model_id==r['model_id'] and cached_native.revision=='local-pinned'
        assert cached_native.mode=='fast' and cached_native.seed==DEFAULT_SEED
        old=json.loads((ROOT/r['manifest']).read_text())
        r['expected_weights']={f['filename']:f['sha256'] for f in old['files'] if f['filename'].endswith('.safetensors')}
        for name,h in r['expected_weights'].items():assert ledger[r['local_dir']+'/'+name]==h
    pools=candidate_pools(['base','instruct','sft','adapter'],['base','instruct'])
    freeze=dict(source_sha256={str(p):sha(p) for p in source_files},controls=controls,external=external,
       candidate_roles=pools,native_version=__version__,native_seed=DEFAULT_SEED,native_mode='fast',native_thresholds=dict(material=MATERIAL_ALPHA,gain=MERGE_GAIN,explained_cos=EXPLAINED_COS),
       policy=dict(max_weight_models=5,seconds=3600,cpu_threads=1,native_source_cap=512*1024**2,sketch_k=64,row_block=256,seed=20261028),
       exposure='Prospective real query within exposed Qwen root; constructed controls already exposed; no untouched-root confirmation.')
    if args.prepare:
        assert not (out/'freeze.json').exists();atomic_json(out/'freeze.json',freeze)
        print('prepared5identities,4fixedpools,no payloads',flush=True);return
    assert json.loads((out/'freeze.json').read_text())==freeze,'frozen source/input changed'
    assert (out/'prospective-review.md').exists(),'independent prospective review required'
    assert not (out/'results.json').exists(),'no overwrite of completed comparison'
    torch.set_num_threads(1);started=time.monotonic();deadline=started+3500
    reader=HubReader(token=next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)),None))
    native={};row_fps={};manifests={};cost=[]
    atomic_json(out/'runtime.json',dict(hostname=socket.gethostname(),owned_pid=os.getpid(),gpu_devices=[],started_unix=time.time()))
    print('start',socket.gethostname(),os.getpid(),flush=True)
    for role,r in controls.items():
        native[role]=Fingerprint.from_dict(json.loads((ROOT/r['native']).read_text()))
        t=time.monotonic();manifest=build_manifest(str(ROOT/r['local_dir']),'',reader,None)
        actual={f['filename']:f['sha256'] for f in manifest['files'] if f['filename'].endswith('.safetensors')}
        assert actual==r['expected_weights'],'control weight changed'
        manifests[role]=manifest;mp=write_manifest(manifest,out/'current-manifests')
        fp=fingerprint_model(manifest,reader,'row_blocks',out/'row_blocks',sketch_k=64,compute_device='cpu',read_threads=4)
        row_fps[role]=fp
        cost.append(dict(role=role,status='acquired_current_row,reused_native',seconds=time.monotonic()-t,
                        local_verification_scan_bytes=sum(f['size'] for f in manifest['files']),fingerprint=str(fp.relative_to(ROOT))))
        atomic_json(out/'progress.json',dict(cost=cost,reader_stats=reader.stats))
    with ThreadPoolExecutor(4) as executor:
        for role in ['base','instruct','query']:
            assert time.monotonic()<deadline
            r=external[role];manifest=json.loads((ROOT/r['manifest']).read_text());manifests[role]=manifest
            t=time.monotonic();fp=fingerprint_model(manifest,reader,'row_blocks',out/'row_blocks',sketch_k=64,compute_device='cpu',read_threads=4)
            row_fps[role]=fp;row_seconds=time.monotonic()-t
            source=PinnedSource(r['repo_id'],r['revision'],[f['filename'] for f in manifest['files']]+['config.json'],reader,executor,deadline=deadline,byte_cap=512*1024**2)
            t=time.monotonic();native_fp=extract_fingerprint(source,mode='fast',model_id=r['repo_id']+'@'+r['revision'],revision=r['revision'])
            native[role]=native_fp;atomic_json(out/f'native-{role}.json',native_fp.to_dict())
            cost.append(dict(role=role,status='acquired',row_seconds=row_seconds,native_seconds=time.monotonic()-t,native_source_bytes=source.bytes_read,fingerprint=str(fp.relative_to(ROOT))))
            atomic_json(out/'progress.json',dict(cost=cost,reader_stats=reader.stats))
            print('acquired',role,flush=True)
    index=[]
    for role in ['base','instruct','sft','adapter','query']:
        m=manifests[role]
        mp=write_manifest(m,out/'current-manifests')
        index.append(dict(key=m['repo_id']+'@'+m['revision'],arch=arch_signature(m['config'],[]),manifest=str(mp.relative_to(ROOT)),fingerprint=str(row_fps[role].relative_to(ROOT))))
    atomic_json(out/'record-index.json',index)
    records=load_records(out/'record-index.json',out/'row_blocks',data_root=ROOT)
    zoo={role:next(r for r in records if r.key==index[i]['key']) for i,role in enumerate(['base','instruct','sft','adapter','query'])}
    from dataclasses import replace
    zoo['query']=replace(zoo['query'],meta=dict(zoo['query'].meta,merge_recipe={'method':'slerp'}))
    pool_results=[]
    for name,roles in pools.items():
        z=fit_merge(zoo['query'],**zoo_inputs(zoo,roles),bootstrap=0,groups={})
        n=decompose_fingerprints(native['query'],[native[k] for k in roles],base=None).to_dict()
        role_by_id={native[k].model_id:k for k in roles}
        nominations=sorted(role_by_id[k] for k,a in n['alphas'].items() if a>=MATERIAL_ALPHA)
        pool_results.append(dict(pool=name,candidates=roles,zoo=z,modeldna=n,
           native_material_ingredient_roles=nominations,recipe_relative_extra_nominations=sorted(set(nominations)-{'base','instruct'}),
           missing_documented_ingredients=sorted({'base','instruct'}-set(roles)),
           historical_source_decision='not_calibrated; native compatibility nominations are not historical source probabilities'))
    oracle=fit_merge(zoo['query'],parents=[zoo['instruct'],zoo['base']],base=zoo['instruct'],bootstrap=0,groups={})
    def finite(value):
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [finite(v) for v in value]
        if isinstance(value,float) and not __import__('math').isfinite(value):return None
        return value
    result=dict(pools=pool_results,oracle_supplied_parent_fit=oracle,cost=cost,reader_stats=reader.stats,elapsed_seconds=time.monotonic()-started,
        source_freeze_sha256=sha(out/'freeze.json'),artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in list((out/'row_blocks').glob('*.safetensors'))+list(out.glob('native-*.json'))},
        interpretation='One documented real SLERP query; effective coefficients only, no configured t accuracy or calibrated source precision. Lab controls are related, not independent real negatives.')
    atomic_json(out/'results.json',finite(result))
    print('complete',[(r['pool'],r['zoo']['status'],r['modeldna']['summary'],r['native_material_ingredient_roles']) for r in pool_results],flush=True)


if __name__=='__main__':main()
