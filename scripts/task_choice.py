"""Small prospective task probes; choices never receive validation outcomes."""
import hashlib
import math
import re

import numpy as np

SEED = 20261028


class EvaluationStopped(BaseException):
    """Deadline/budget stop that generic network/scheduler retries cannot swallow."""



def parse_label(text, n_labels, *, hit_limit=False):
    """Only a single requested ASCII letter is a valid generated answer."""
    if hit_limit:
        return None
    value = text.strip()
    if len(value) == 1 and value in 'ABCD'[:n_labels]:
        return ord(value) - ord('A')
    return None


def split_rows(rows, n_probe=32, n_validation=256):
    """Pick one record per normalized text/passage group, then hash split."""
    groups = {}
    for row in sorted(rows, key=lambda r: r['id']):
        group = ' '.join(row.get('group', row['text']).casefold().split())
        groups.setdefault(group, row)
    def order(item):
        group, _ = item
        return hashlib.sha256(f'{SEED}:{group}'.encode()).hexdigest()
    selected = [r for _, r in sorted(groups.items(), key=order)]
    if n_probe < 1 or n_validation < 1 or len(selected) < n_probe + n_validation:
        raise ValueError('insufficient distinct groups')
    return dict(probe=selected[:n_probe], validation=selected[n_probe:n_probe+n_validation])


def choose(probe_correct, metadata):
    """Use only complete probe vectors and complete fixed baseline metadata."""
    keys = sorted(metadata)
    if set(keys) != set(probe_correct) or len(keys) < 2:
        raise ValueError('aligned candidate evidence required')
    lengths = {len(probe_correct[k]) for k in keys}
    if len(lengths) != 1 or next(iter(lengths)) < 1:
        raise ValueError('aligned nonempty probe rows required')
    if any(type(v) is not bool for k in keys for v in probe_correct[k]):
        raise ValueError('boolean probe correctness required')
    fields = ['params_b', 'downloads', 'historical_mean']
    if any(not math.isfinite(metadata[k][f]) for k in keys for f in fields):
        raise ValueError('complete finite baseline metadata required')
    result = {'probe': min(keys, key=lambda k: (-sum(probe_correct[k]), metadata[k]['params_b'], k))}
    for policy, field in [('largest', 'params_b'), ('downloads', 'downloads'), ('historical_mean', 'historical_mean')]:
        result[policy] = min(keys, key=lambda k: (-metadata[k][field], k))
    return result


def score_choices(correct, choices):
    """Paired finite-panel report, with a fixed per-item bootstrap interval."""
    keys = sorted(correct)
    a = np.asarray([correct[k] for k in keys], dtype=float)
    if a.ndim != 2 or a.shape[1] < 2 or not np.isin(a, [0, 1]).all():
        raise ValueError('aligned binary validation rows required')
    if any(k not in keys for k in choices.values()):
        raise ValueError('choice absent from validation panel')
    values = a.mean(axis=1)
    accuracy = dict(zip(keys, values.tolist()))
    best = float(values.max())
    probe, baseline = a[keys.index(choices['probe'])], a[keys.index(choices['largest'])]
    d = probe - baseline
    rng = np.random.default_rng(SEED)
    boot = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(5000)])
    return dict(n_validation=a.shape[1], candidate_accuracy=accuracy,
        finite_panel_best_accuracy=best, uniform_expected_accuracy=float(values.mean()),
        policies={name: dict(selected=key, accuracy=accuracy[key], finite_panel_regret=best-accuracy[key])
                  for name, key in choices.items()},
        probe_vs_largest=dict(gain=float(d.mean()), probe_only_correct=int((d == 1).sum()),
            largest_only_correct=int((d == -1).sum()), both_correct=int(((probe == 1) & (baseline == 1)).sum()),
            both_wrong=int(((probe == 0) & (baseline == 0)).sum()),
            bootstrap_95_interval=np.quantile(boot, [.025, .975]).tolist(), bootstrap_replicates=5000,
            scope='Item-resampling interval conditional on frozen probes/candidates; not task/model population evidence.'))


def validate_cached_file(path, expected_sha256, receipt):
    """Validate a fresh full-SHA receipt under the frozen immutable-cache assumption."""
    if receipt['expected_sha256'] != expected_sha256 or receipt['observed_sha256'] != expected_sha256:
        raise ValueError('cached inventory digest mismatch')
    if str(path.resolve()) != receipt['absolute_path'] or path.is_symlink():
        raise ValueError('cached file path changed')
    stat = path.stat()
    actual = dict(dev=stat.st_dev,ino=stat.st_ino,size=stat.st_size,
                  mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)
    if any(actual[k] != receipt[k] for k in actual):
        raise ValueError('cached file changed after full verification')


def run_panel(*, out_name='task-choice-20261112'):
    """Run one frozen panel, with validation generated only after choice freeze."""
    import fcntl
    import gc
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import platform
    import signal
    import socket
    import subprocess
    import sys
    import time
    from datetime import datetime, timezone

    started = time.monotonic()
    attempt_start_epoch = time.time()
    def deadline(signum, frame):
        raise EvaluationStopped('bounded task evaluation stopped')
    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(6600)
    root = Path(__file__).resolve().parents[2]
    if re.fullmatch(r'[a-z0-9-]+',out_name) is None:
        raise ValueError('invalid fixed-run output name')
    out = root/'runs'/out_name
    sys.path[:0] = [str(root/'Zoo-Keeper'), str(root/'Model-Zoo'),
                    str(root/'Model-Zoo/esd_experiment/src'), str(root/'Model-Zoo/esd_experiment')]
    from lab.artifacts import atomic_json
    from lab.launches import file_hash
    from scripts.task_choice_inputs import prompt
    from evidence.hub import HubReader

    freeze_path = out/'freeze.json'
    freeze = json.loads(freeze_path.read_text())
    original_start = freeze.get('original_start_utc',datetime.now(timezone.utc).isoformat())
    elapsed_offset = max(0.,attempt_start_epoch-datetime.fromisoformat(original_start).timestamp())
    if elapsed_offset >= 6600:
        raise EvaluationStopped('original deadline exhausted')
    signal.alarm(max(1,int(6600-(time.time()-datetime.fromisoformat(original_start).timestamp()))))
    assert (out/'prospective-review-final.md').exists()
    assert not (out/'runtime.json').exists(), 'preserve every model-job attempt'
    for name, expected in freeze['sha256'].items():
        assert file_hash(root/name) == expected, name
    prepared = json.loads((out/'prepared.json').read_text())
    models = sorted(prepared['models'], key=lambda m: m['repo_id'])
    datasets = prepared['datasets']
    rows = {d['task']: json.loads((root/d['selected_path']).read_text()) for d in datasets}
    metadata = {m['repo_id']+'@'+m['revision']: dict(params_b=m['historical_evidence']['reported_params_b'],
        downloads=m['downloads'], historical_mean=sum(m['historical_evidence']['source_scores'].values())/6)
        for m in models}
    runtime = dict(status='ACQUIRING', host=socket.gethostname(), host_pid=os.getpid(),
        started_utc=original_start, attempt_started_utc=datetime.now(timezone.utc).isoformat(), gpu_uuid=None, device=None,
        freeze_sha256=file_hash(freeze_path), distinct_weight_models=len(models), model_load_passes=0,
        weight_payload_bytes=0, download_hash_scan_bytes=0, acquisition_seconds=0.,
        stages=[], python=platform.python_version(),
        packages={n: importlib.metadata.version(n) for n in ['torch','transformers','safetensors','huggingface-hub','numpy']},
        stop_command=f'kill -TERM {os.getpid()}', resume='No automatic resume; retain partial attempt and diagnose under unchanged scientific rules.')
    def publish():
        runtime['attempt_elapsed_seconds'] = time.monotonic()-started
        runtime['elapsed_seconds'] = elapsed_offset+runtime['attempt_elapsed_seconds']
        atomic_json(out/'runtime.json', runtime)
    publish()
    print(json.dumps(dict(host=runtime['host'], host_pid=os.getpid(), status='ACQUIRING', stop_command=runtime['stop_command'])), flush=True)
    token = next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)), None)
    assert token
    from evidence.hub import _transport
    reserved_response_bytes = freeze.get('inherited_reserved_response_bytes',0)
    runtime['download_response_byte_upper_bound'] = reserved_response_bytes
    runtime['inherited_response_body_bytes'] = freeze.get('inherited_response_body_bytes',0)
    publish()
    def limited_transport(method, url, headers):
        nonlocal reserved_response_bytes
        if method != 'HEAD':
            start, end = map(int, headers['Range'][6:].split('-'))
            # _transport reads at most requested bytes plus one overflow byte.
            if reserved_response_bytes + end-start+2 > 30_000_000_000:
                raise EvaluationStopped('download response-body byte limit')
            reserved_response_bytes += end-start+2
            runtime['download_response_byte_upper_bound'] = reserved_response_bytes
        return _transport(method, url, headers)
    reader = HubReader(token=token, transport=limited_transport)
    lock = None
    dispatcher = None
    torch = None
    try:
        begin = time.monotonic()
        for m in models:
            folder = root/m['local_path']
            files = [f for f in m['files'] if f['name'].endswith('.safetensors')]
            assert files and all('/' not in f['name'] and re.fullmatch('[0-9a-f]{64}',f['sha256'] or '') for f in files)
            for f in files:
                path = folder/f['name']
                if path.exists():
                    receipt = freeze.get('verified_cached_inputs',{}).get(str(path.relative_to(root)))
                    if receipt is not None:
                        assert freeze.get('local_only') and path.stat().st_size == f['size']
                        validate_cached_file(path,f['sha256'],receipt)
                        runtime['receipt_checks'] = runtime.get('receipt_checks',0)+1
                    else:
                        assert path.stat().st_size == f['size'] and file_hash(path) == f['sha256']
                        runtime['download_hash_scan_bytes'] += f['size']
                else:
                    if freeze.get('local_only'):
                        raise EvaluationStopped('local checkpoint missing; no further acquisition allowed')
                    remote = reader.resolve(m['repo_id'],f['name'],m['revision'])
                    assert remote.size == f['size'] and remote.sha256 == f['sha256']
                    assert runtime['weight_payload_bytes']+remote.size <= 30_000_000_000
                    temporary = path.with_suffix('.partial')
                    with temporary.open('wb') as stream:
                        for start in range(0,remote.size,64*1024**2):
                            body = reader.read(remote,start,min(start+64*1024**2,remote.size))
                            stream.write(body)
                            runtime['weight_payload_bytes'] += len(body)
                            runtime['reader_stats'] = dict(reader.stats)
                            publish()
                    assert file_hash(temporary) == f['sha256']
                    runtime['download_hash_scan_bytes'] += f['size']
                    temporary.replace(path)
                print(json.dumps(dict(status='weight_file_verified', repo=m['repo_id'], file=f['name'], bytes=f['size'])),flush=True)
        runtime['acquisition_seconds'] = time.monotonic()-begin
        runtime['status'] = 'ADMITTING_GPU'
        publish()
        # Reuse the existing collector's live process/device admission, then its
        # lab companion's shared per-UUID lock and two explicit live checks.
        from gputracker.gputracker import GPUDispatcher
        from lab.scripts.gpu_training import gpu_snapshot, eligible_gpu
        dispatcher = GPUDispatcher(str(out/'gpu-config.json'))
        devices = dispatcher.get_free_gpus(1)
        assert devices is not None and len(devices) == 1
        index = devices[0]
        query = subprocess.run(['nvidia-smi',f'--id={index}','--query-gpu=uuid','--format=csv,noheader'],
            check=True,capture_output=True,text=True,timeout=15)
        uuid = query.stdout.strip()
        assert re.fullmatch(r'GPU-[a-f0-9-]+',uuid)
        lock_path = root/'runs/track-b-training-gpu-v1/locks'/(uuid+'.lock')
        lock_path.parent.mkdir(parents=True,exist_ok=True)
        lock = lock_path.open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime.update(gpu_uuid=uuid,physical_index=index,lock_path=str(lock_path.relative_to(root)),gpu_checks=[])
        for check in range(2):
            if check:
                time.sleep(5)
            snapshot = gpu_snapshot(uuid)
            runtime['gpu_checks'].append(snapshot)
            assert eligible_gpu(snapshot) and snapshot['memory_used_mib']<500
            active = subprocess.run(['nvidia-smi',f'--id={uuid}',
                '--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True,timeout=15)
            assert not active.stdout.strip(), 'GPU has a live process'
            publish()
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        import torch as torch_module
        torch = torch_module
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        assert torch.cuda.is_available()
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        runtime.update(status='PROBE',device=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
            stop_command=f'kill -TERM {os.getpid()}')
        publish()
        selections = None
        all_probe = {d['task']: {} for d in datasets}
        for phase in ['probe','validation']:
            if phase == 'validation':
                selections = {task: choose(correct,metadata) for task,correct in all_probe.items()}
                proof = {str(p.relative_to(root)):file_hash(p) for p in sorted((out/'traces/probe').glob('*.json'))}
                atomic_json(out/'choices.json',dict(choices=selections,probe_correct=all_probe,metadata=metadata,
                    probe_sha256=proof,freeze_sha256=file_hash(freeze_path),frozen_utc=datetime.now(timezone.utc).isoformat()))
                runtime['choices_sha256'] = file_hash(out/'choices.json')
                runtime['status'] = 'VALIDATION'
                publish()
                print('both_task_choices_frozen_before_validation',flush=True)
            for m in models:
                begin = time.monotonic()
                folder = root/m['local_path']
                if freeze.get('verified_cached_inputs'):
                    for weight in m['files']:
                        if weight['name'].endswith('.safetensors'):
                            cached_path=folder/weight['name']
                            receipt=freeze['verified_cached_inputs'][str(cached_path.relative_to(root))]
                            validate_cached_file(cached_path,weight['sha256'],receipt)
                            runtime['receipt_checks']=runtime.get('receipt_checks',0)+1
                tokenizer = AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
                tokenizer.padding_side = 'left'
                tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,use_safetensors=True,
                    trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').to('cuda:0').eval()
                runtime['model_load_passes'] += 1
                stage = dict(phase=phase,repo_id=m['repo_id'],load_seconds=time.monotonic()-begin,tasks=[])
                key = m['repo_id']+'@'+m['revision']
                for d in datasets:
                    task = d['task']
                    trace = []
                    task_started = time.monotonic()
                    for offset in range(0,len(rows[task][phase]),8):
                        batch = rows[task][phase][offset:offset+8]
                        messages = [tokenizer.apply_chat_template([dict(role='user',content=prompt(task,r))],
                            tokenize=False,add_generation_prompt=True) for r in batch]
                        inputs = tokenizer(messages,add_special_tokens=False,padding=True,truncation=False,return_tensors='pt').to('cuda:0')
                        assert inputs['input_ids'].shape[1] <= 2048
                        config = GenerationConfig(max_new_tokens=8,do_sample=False,num_beams=1,
                            eos_token_id=model.generation_config.eos_token_id,pad_token_id=tokenizer.pad_token_id,
                            use_cache=True)
                        with torch.inference_mode():
                            generated = model.generate(**inputs,generation_config=config)
                        continuation = generated[:,inputs['input_ids'].shape[1]:].cpu().tolist()
                        for r,text,ids,n in zip(batch,messages,continuation,inputs['attention_mask'].sum(dim=1).cpu().tolist()):
                            answer = tokenizer.decode(ids,skip_special_tokens=True)
                            eos = config.eos_token_id
                            eos = set(eos if isinstance(eos,list) else [eos])
                            terminated = any(i in eos for i in ids)
                            hit_limit = len(ids)==8 and not terminated
                            parsed = parse_label(answer,2,hit_limit=hit_limit)
                            trace.append(dict(id=r['id'],gold=r['label'],prediction=parsed,correct=parsed==r['label'],
                                output=answer,generated_token_ids=ids,prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                prompt_tokens=n,hit_generation_limit=hit_limit,eos_seen=terminated))
                        target = out/'traces'/phase/(m['repo_id'].replace('/','--')+'--'+task+'.json')
                        atomic_json(target,dict(model_key=key,task=task,phase=phase,rows=trace))
                        publish()
                    torch.cuda.synchronize()
                    stage['tasks'].append(dict(task=task,seconds=time.monotonic()-task_started,items=len(trace)))
                    if phase == 'probe':
                        all_probe[task][key] = [r['correct'] for r in trace]
                    # Validation stdout deliberately omits predictions/metrics.
                    print(json.dumps(dict(phase=phase,repo=m['repo_id'],task=task,items=len(trace))),flush=True)
                runtime['stages'].append(stage)
                del generated,inputs,model,tokenizer
                gc.collect()
                torch.cuda.empty_cache()
                runtime['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                runtime['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                publish()
        runtime['status'] = 'COMPLETE_UNSCORED'
    except BaseException as error:
        runtime.update(status='FAILED',error_class=type(error).__name__)
        raise
    finally:
        runtime['reader_stats'] = dict(reader.stats)
        runtime['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if lock is not None:
            fcntl.flock(lock,fcntl.LOCK_UN)
            lock.close()
        if dispatcher is not None:
            dispatcher.occupied_gpus.clear()
        runtime['reservation_released'] = True
        publish()
        signal.alarm(0)
    print(json.dumps(dict(status=runtime['status'],elapsed_seconds=runtime['elapsed_seconds'])),flush=True)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-name',default='task-choice-20261112')
    run_panel(out_name=parser.parse_args().out_name)
