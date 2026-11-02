"""Prepare immutable GPU variants and run one explicitly authorized training job.

Uses the existing frozen preflight and checkpoint writers. The CPU launch
bundle and all dataset selections remain unchanged. Set an external 7200s
timeout when invoking ``run``; throughput also rejects over-budget jobs.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from lab.launches import file_hash


def write_json(path, value, *, immutable=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,sort_keys=True,indent=2)+'\n'
    if immutable and path.exists():
        if path.read_text()!=data: raise FileExistsError(str(path))
        return
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(data); temporary.replace(path)


def prepare_bundle(source, destination):
    source,destination=Path(source),Path(destination)
    configs=[]
    paths=sorted((source/'configs').glob('*.json'))
    if not paths: raise ValueError('missing source launch configs')
    id_map={json.loads(p.read_text())['recipe']['model_id']:
            json.loads(p.read_text())['recipe']['model_id'].replace('-launch-','-gpu-v1-') for p in paths}
    if any(a==b for a,b in id_map.items()) or len(set(id_map.values()))!=len(id_map):
        raise ValueError('source IDs cannot be mapped uniquely to GPU IDs')
    manifest_path=destination/'manifest.json'
    for path in paths:
        config=copy.deepcopy(json.loads(path.read_text())); recipe=config['recipe']
        old_id=recipe['model_id']; recipe.update(model_id=id_map[old_id],device='cuda:0')
        recipe['parents']=[id_map.get(p,p) for p in recipe['parents']]
        recipe['parent_identities']={id_map.get(k,k):v for k,v in recipe['parent_identities'].items()}
        old_model_path=config['inputs']['model_path']
        for old,new in id_map.items():
            if old_model_path=='lab/'+old: config['inputs']['model_path']='lab/'+new
        relative='configs/'+recipe['model_id']+'.json'
        # Check a previously frozen manifest before touching any variant.
        if manifest_path.exists():
            old_manifest=json.loads(manifest_path.read_text())
            old_record=next((r for r in old_manifest['configs'] if r['source_path']==str(path.resolve())),None)
            if old_record is None or old_record['source_sha256']!=file_hash(path):
                raise FileExistsError('source bundle differs from frozen GPU manifest')
        write_json(destination/relative,config,immutable=True)
        configs.append(dict(model_id=recipe['model_id'],trainer=config['trainer'],
            source_path=str(path.resolve()),source_sha256=file_hash(path),
            config_path=relative,config_sha256=file_hash(destination/relative)))
    manifest=dict(schema_version=1,source_bundle=str(source.resolve()),
        source_bundle_index_sha256=file_hash(source/'bundle-index.json') if (source/'bundle-index.json').exists() else None,
        device='cuda:0',physical_device='CUDA_VISIBLE_DEVICES supplied per launch',
        gpu_selection=dict(utilization_percent_below=80,memory_fraction_below=.8,
            minimum_free_mib=24*1024,consecutive_checks=2,check_interval_seconds=5),configs=configs)
    write_json(manifest_path,manifest,immutable=True)
    return manifest


def eligible_gpu(row):
    try:
        utilization=float(row['utilization_percent']); used=float(row['memory_used_mib']); total=float(row['memory_total_mib'])
        return (all(math.isfinite(x) for x in (utilization,used,total)) and total>0
                and 0<=utilization<80 and 0<=used/total<.8 and total-used>=24*1024)
    except (KeyError,TypeError,ValueError):
        return False


def gpu_snapshot(uuid):
    result=subprocess.run(['nvidia-smi','--id='+uuid,
        '--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=15)
    rows=result.stdout.strip().splitlines()
    if len(rows)!=1: raise ValueError('physical UUID must select exactly one GPU')
    actual,name,total,used,util=[x.strip() for x in rows[0].split(',')]
    if actual!=uuid: raise ValueError('GPU UUID differs from requested device')
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),uuid=actual,name=name,
        memory_total_mib=float(total),memory_used_mib=float(used),utilization_percent=float(util))


def run_job(config_path, root, uuid, profile_path, expected_hash, *, resolve_parent=False):
    if file_hash(config_path)!=expected_hash: raise ValueError('frozen GPU config hash mismatch')
    config=json.loads(Path(config_path).read_text())
    if config['recipe']['device']!='cuda:0': raise ValueError('expected logical cuda:0 recipe')
    if not uuid.startswith('GPU-') or any(c not in 'GPU-0123456789abcdef' for c in uuid):
        raise ValueError('explicit physical GPU UUID required')
    if 'torch' in __import__('sys').modules: raise RuntimeError('CUDA visibility must be set before torch import')
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    root=Path(root); profile_path=Path(profile_path)
    if profile_path.exists(): raise FileExistsError('use a new profile path for each launch attempt')
    lock_path=root/'runs/track-b-training-gpu-v1/locks'/(uuid+'.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        profile=dict(status='CHECKING_GPU',model_id=config['recipe']['model_id'],config_path=str(Path(config_path).resolve()),
            config_sha256=expected_hash,host=socket.gethostname(),pid=os.getpid(),gpu_uuid=uuid,
            started_utc=datetime.now(timezone.utc).isoformat(),steps_completed=0,gpu_checks=[])
        started=time.monotonic(); torch=None; original_step=None; step_times=[]
        def publish():
            profile['elapsed_seconds']=time.monotonic()-started
            if torch is not None and torch.cuda.is_initialized():
                profile.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if len(step_times)>1:
                profile['seconds_per_step']=(step_times[-1]-step_times[0])/(len(step_times)-1)
                profile['projected_training_seconds']=profile['seconds_per_step']*config['recipe']['steps']
            write_json(profile_path,profile)
        try:
            for check in range(2):
                if check: time.sleep(5)
                snapshot=gpu_snapshot(uuid); profile['gpu_checks'].append(snapshot); publish()
                if not eligible_gpu(snapshot): raise RuntimeError('GPU does not meet both <80% limits and 24GiB headroom')
            import torch as torch_module
            torch=torch_module
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
            torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
            profile.update(status='RUNNING',gpu_name=torch.cuda.get_device_name(0),torch_version=torch.__version__)
            from lab import launches
            if resolve_parent:
                if config['trainer']!='dpo': raise ValueError('only DPO needs parent resolution')
                config=launches.resolve_sft_parent(config,root)
                resolved=profile_path.parent/'resolved-configs'/(config['recipe']['model_id']+'.json')
                write_json(resolved,config,immutable=True)
                config_path=resolved; expected_hash=file_hash(resolved)
                profile.update(resolved_config_path=str(resolved),resolved_config_sha256=expected_hash,
                    actual_parent_key=config['inputs']['local_parent']['model_key'])
            original_step=torch.optim.AdamW.step
            def measured_step(optimizer,*args,**kwargs):
                result=original_step(optimizer,*args,**kwargs)
                torch.cuda.synchronize()
                step_times.append(time.monotonic()); profile['steps_completed']=len(step_times)
                if len(step_times)==5 or len(step_times)%16==0:
                    publish(); print(json.dumps(profile,sort_keys=True),flush=True)
                    if profile.get('projected_training_seconds',0)>7200:
                        raise RuntimeError('measured projection exceeds two-hour launch budget')
                return result
            torch.optim.AdamW.step=measured_step
            publish()
            launches.main(['--config',str(config_path),'--data-root',str(root),
                '--expected-config-sha256',expected_hash,'--execute','--resume'])
            torch.cuda.synchronize()
            training=json.loads((root/'lab'/config['recipe']['model_id']/'training.json').read_text())
            if training['steps_completed']!=config['recipe']['steps']: raise RuntimeError('incomplete output training provenance')
            profile.update(status='COMPLETE',output=str(root/'lab'/config['recipe']['model_id']),
                output_steps_completed=training['steps_completed'],
                training_json_sha256=file_hash(root/'lab'/config['recipe']['model_id']/'training.json'))
        except BaseException as error:
            profile.update(status='FAILED',error_type=type(error).__name__,error=str(error))
            raise
        finally:
            if original_step is not None: torch.optim.AdamW.step=original_step
            publish(); print(json.dumps(profile,sort_keys=True),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    prepare=subs.add_parser('prepare'); prepare.add_argument('--source',type=Path,required=True); prepare.add_argument('--out',type=Path,required=True)
    run=subs.add_parser('run'); run.add_argument('--config',type=Path,required=True)
    run.add_argument('--data-root',type=Path,required=True); run.add_argument('--gpu-uuid',required=True)
    run.add_argument('--profile',type=Path,required=True); run.add_argument('--expected-config-sha256',required=True)
    run.add_argument('--resolve-parent',action='store_true')
    args=parser.parse_args(argv)
    if args.command=='prepare': print(json.dumps(prepare_bundle(args.source,args.out),sort_keys=True))
    else: run_job(args.config,args.data_root,args.gpu_uuid,args.profile,args.expected_config_sha256,resolve_parent=args.resolve_parent)


if __name__=='__main__': main()
