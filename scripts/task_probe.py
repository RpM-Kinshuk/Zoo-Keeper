"""One fixed parallel task-probe campaign; reuse cached checkpoints and readers."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import unicodedata

ROOT=Path(__file__).resolve().parents[2]
SEED=20261028
HINTS={'ag_news':{'ag_news','ag-news','news-classification','topic-classification','text-classification'},
       'emotion':{'emotion','emotions','sentiment-analysis','text-classification','dataset:dair-ai/emotion'}}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def write(path,value):
    from lab.artifacts import atomic_json
    atomic_json(path,value)


def loading_record(loading):
    """Transformers returns key sets; progress records require JSON arrays."""
    return {k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()}


def balanced_split(rows,k):
    from scripts.task_choice import split_rows
    groups={}
    for row in sorted(rows,key=lambda r:r['id']):
        group=' '.join(unicodedata.normalize('NFKC',row['text']).casefold().split())
        groups.setdefault(group,[]).append(dict(row,group=group))
    conflicts={g for g,v in groups.items() if len({r['label'] for r in v})!=1}
    eligible=[v[0] for g,v in groups.items() if g not in conflicts]
    def quotas(n,phase):
        order=sorted(range(k),key=lambda c:hashlib.sha256(f'{SEED}:{phase}:{c}'.encode()).hexdigest())
        return {c:n//k+int(c in order[:n%k]) for c in range(k)}
    qp,qv=quotas(32,'probe'),quotas(128,'validation');result={'probe':[],'validation':[]}
    for c in range(k):
        part=split_rows([r for r in eligible if r['label']==c],qp[c],qv[c])
        for phase in result:result[phase]+=part[phase]
    for phase in result:
        result[phase].sort(key=lambda r:hashlib.sha256(f'{SEED}:{r["group"]}'.encode()).hexdigest())
    return result,dict(source_rows=len(rows),duplicate_rows=sum(len(v)-1 for v in groups.values()),
                       conflicting_groups=len(conflicts),eligible_groups=len(eligible))


def aggregate_rotations(scores,shifts):
    k=len(scores[0])
    values=[sum(row[(c-shift)%k] for row,shift in zip(scores,shifts))/len(scores) for c in range(k)]
    if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite label scores')
    return max(range(k),key=lambda c:(values[c],-c))


def label_tokens(tokenizer,prefix,k):
    before=tokenizer.encode(prefix,add_special_tokens=False);tokens=[]
    for letter in 'ABCDEF'[:k]:
        full=tokenizer.encode(prefix+' '+letter,add_special_tokens=False)
        if full[:len(before)]!=before or len(full)!=len(before)+1:
            raise ValueError('literal label does not extend the exact prefix by one token')
        tokens.append(full[-1])
    if len(set(tokens))!=k:raise ValueError('label tokens are not distinct')
    return before,tokens


def choose(correct,metadata):
    keys=sorted(metadata)
    if set(correct)!=set(keys) or len(keys)<2 or any(len(correct[k])!=32 for k in keys):
        raise ValueError('complete aligned three-model probe required')
    if any(type(v) is not bool for k in keys for v in correct[k]):raise ValueError('boolean probe rows required')
    popularity=lambda k:(-(metadata[k]['downloads'] if metadata[k]['downloads'] is not None else -math.inf),k)
    matches=[k for k in keys if metadata[k]['task_match']]
    status='matched'
    if not matches:matches=keys;status='no_matches_downloads_fallback'
    elif len(matches)==len(keys):status='all_match_downloads_fallback'
    return dict(choices=dict(probe=min(keys,key=lambda k:(-sum(correct[k]),metadata[k]['params_b'],k)),
        largest=min(keys,key=lambda k:(-metadata[k]['params_b'],k)),downloads=min(keys,key=popularity),
        task_labels=min(matches,key=popularity)),task_label_status=status,matching_models=[k for k in keys if metadata[k]['task_match']])


def prepare(out):
    import pandas as pd
    from transformers import AutoTokenizer
    started=time.monotonic();assert not (out/'prepared.json').exists()
    meta=json.loads((out/'metadata.json').read_text());data=json.loads((out/'data-inputs.json').read_text())
    prior=json.loads((ROOT/'runs/task-choice-20261112/prepared.json').read_text())
    receipts=json.loads((ROOT/'runs/task-choice-available-20261112/freeze.json').read_text())['verified_cached_inputs']
    models=[]
    for row in [r for r in meta['records'] if r['kind']=='model']:
        old=next((m for m in prior['models'] if m['repo_id']==row['repo_id']),None)
        q3=row['repo_id']=='Qwen/Qwen3-0.6B'
        name='q3' if q3 else ('q15' if row['repo_id'].startswith('Qwen/') else 'phi')
        folder='lab/inputs/cohort-v1-q3-06-c1899de289a0' if q3 else old['local_path']
        weights=[f for f in row['files'] if '/' not in f['name'] and f['name'].endswith('.safetensors')]
        assert weights and all((ROOT/folder/f['name']).is_file() and (ROOT/folder/f['name']).stat().st_size==f['size'] for f in weights)
        tok=AutoTokenizer.from_pretrained(ROOT/folder,local_files_only=True,trust_remote_code=False)
        tags={t.casefold() for t in row['tags']}|{(row['pipeline_tag'] or '').casefold()}
        models.append(dict(name=name,key=row['repo_id']+'@'+row['revision'],repo_id=row['repo_id'],revision=row['revision'],
            local_path=folder,weights=weights,params_b=.6 if q3 else (1.5 if name=='q15' else 3.8),downloads=row['downloads'],
            task_matches={task:bool(tags&HINTS[task]) for task in HINTS},pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            chat_template_sha256=digest(tok.chat_template),thinking_disabled=q3,
            prior_receipts={str((Path(folder)/f['name'])):receipts[str(Path(folder)/f['name'])] for f in weights if str(Path(folder)/f['name']) in receipts}))
    selected={};task_info={}
    for row in data['datasets']:
        task='ag_news' if row['repo_id']=='fancyzhx/ag_news' else 'emotion'
        frame=pd.read_parquet(ROOT/row['local_parquet']);rows=[]
        for i,r in frame.iterrows():
            text=str(r['text'])
            if len(text)<=2400:rows.append(dict(id=f'{row["repo_id"]}@{row["revision"]}:test:{i}',text=text,label=int(r['label'])))
        split,stats=balanced_split(rows,len(row['label_names']));selected[task]=split
        task_info[task]=dict(repo_id=row['repo_id'],revision=row['revision'],labels=row['label_names'],
            length_excluded=len(frame)-len(rows),**stats)
    write(out/'selected.json',selected)
    for model in models:
        tok=AutoTokenizer.from_pretrained(ROOT/model['local_path'],local_files_only=True,trust_remote_code=False);packets={}
        for task,split in selected.items():
            labels=task_info[task]['labels'];k=len(labels);packets[task]={}
            for phase,rows in split.items():
                packets[task][phase]=[]
                for row in rows:
                    shift=int(hashlib.sha256(f'{SEED}:{task}:{row["id"]}'.encode()).hexdigest(),16)%k
                    versions=[]
                    for s in [shift,(shift+k//2)%k]:
                        options='\n'.join(f'{chr(65+j)}: {labels[(j+s)%k]}' for j in range(k))
                        instruction=('Classify the news topic.' if task=='ag_news' else 'Classify the emotion expressed in the text.')
                        message=instruction+'\n'+options+'\nReply with only the label letter.\n\nText:\n'+row['text']
                        extra={'enable_thinking':False} if model['thinking_disabled'] else {}
                        prefix=tok.apply_chat_template([dict(role='user',content=message)],tokenize=False,add_generation_prompt=True,**extra)+'Answer:'
                        ids,choices=label_tokens(tok,prefix,k);assert len(ids)<=2048
                        versions.append(dict(shift=s,input_ids=ids,label_tokens=choices,prompt_sha256=hashlib.sha256(prefix.encode()).hexdigest()))
                    packets[task][phase].append(dict(id=row['id'],gold=row['label'],rotations=versions))
        path=out/'packets'/(model['name']+'.json');write(path,packets);model['packets']=str(path.relative_to(ROOT));model['packets_sha256']=sha(path)
    files=[Path(__file__),ROOT/'Zoo-Keeper/tests/test_task_probe.py',ROOT/'Zoo-Keeper/scripts/task_choice.py',
        out/'metadata.json',out/'data-inputs.json',out/'selected.json']
    files += [ROOT/r['path'] for r in data['records']]
    for model in models:
        files += [f for f in (ROOT/model['local_path']).iterdir() if f.is_file() and f.suffix in ['.json','.model','.txt']]
    prepared=dict(models=models,tasks=task_info,input_sha256={str(f.relative_to(ROOT)):sha(f) for f in files},
        protocol_sha256=hashlib.sha256((out/'RESEARCH.md').read_text().split('## Outcome')[0].encode()).hexdigest(),
        preparation_seconds=time.monotonic()-started,model_loads=0,backend='cpu',intra_threads=8,inter_threads=1)
    write(out/'prepared.json',prepared)
    print('prepared3cached models,2fresh tasks,32probe/128validation,2fixed rotations; no scoring',flush=True)


def score_packets(model,versions,pad_id):
    import torch
    width=max(len(v['input_ids']) for v in versions)
    ids=torch.full((len(versions),width),pad_id,dtype=torch.long);mask=torch.zeros_like(ids)
    for i,v in enumerate(versions):
        n=len(v['input_ids']);ids[i,-n:]=torch.tensor(v['input_ids']);mask[i,-n:]=1
    positions=(mask.cumsum(-1)-1).masked_fill(mask==0,0)
    with torch.inference_mode():
        logits=model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,logits_to_keep=1).logits[:,-1].float()
        tokens=torch.tensor([v['label_tokens'] for v in versions])
        scores=torch.log_softmax(logits,-1).gather(1,tokens)
    if not torch.isfinite(scores).all():raise ValueError('nonfinite continuation score')
    return scores.tolist(),len(versions)*width


def worker(out,name):
    import signal,resource
    from scripts.task_choice import EvaluationStopped,validate_cached_file
    p=json.loads((out/'prepared.json').read_text());campaign=json.loads((out/'campaign.json').read_text())
    m=next(m for m in p['models'] if m['name']==name);deadline=campaign['deadline_unix'];started=time.monotonic()
    def stop(*_):raise EvaluationStopped('original campaign deadline/termination')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop)
    signal.alarm(max(1,int(deadline-time.time())))
    profile=dict(host_pid=os.getpid(),model_key=m['key'],prepared_sha256=sha(out/'prepared.json'),status='IMPORTING',phases={})
    def publish():
        profile['elapsed_seconds']=time.monotonic()-started;write(out/'workers'/(name+'.json'),profile)
    publish()
    try:
        import torch
        torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(SEED)
        from transformers import AutoModelForCausalLM
        for receipt in campaign['receipts'][name].values():
            validate_cached_file(Path(receipt['absolute_path']),receipt['expected_sha256'],receipt)
        t=time.monotonic()
        model,loading=AutoModelForCausalLM.from_pretrained(ROOT/m['local_path'],local_files_only=True,use_safetensors=True,
            trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager',output_loading_info=True)
        assert not any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),loading
        model.eval()
        profile.update(status='LOADED',load_seconds=time.monotonic()-t,torch_version=torch.__version__,dtype='bfloat16',loading_info=loading_record(loading))
        packets=json.loads((ROOT/m['packets']).read_text());assert sha(ROOT/m['packets'])==m['packets_sha256']
        total_work=sum(len(items[i:i+4])*2*max(len(v['input_ids']) for item in items[i:i+4] for v in item['rotations'])
            for task in packets.values() for items in task.values() for i in range(0,len(items),4))
        used=0;score_seconds=0.;batches=0
        for phase,count in [('probe',32),('validation',128)]:
            if phase=='validation':
                profile['status']='WAITING_FOR_CHOICES';publish()
                while not (out/'release.json').exists():
                    if time.time()>=deadline:raise EvaluationStopped('barrier deadline')
                    time.sleep(.2)
                release=json.loads((out/'release.json').read_text());assert release['choices_sha256']==sha(out/'choices.json')
                choices=json.loads((out/'choices.json').read_text())
                assert choices['prepared_sha256']==profile['prepared_sha256']
                assert choices['probe_sha256'][name]==sha(out/'traces/probe'/(name+'.json'))
            profile['status']=phase.upper();t=time.monotonic();trace=dict(model_key=m['key'],prepared_sha256=profile['prepared_sha256'],phase=phase,complete=False,tasks={task:[] for task in packets})
            path=out/'traces'/phase/(name+'.json')
            for task in packets:
                items=packets[task][phase]
                for offset in range(0,len(items),4):
                    chunk=items[offset:offset+4];versions=[v for item in chunk for v in item['rotations']]
                    began=time.monotonic();scores,work=score_packets(model,versions,m['pad_token_id']);score_seconds+=time.monotonic()-began
                    used+=work;batches+=1
                    for i,item in enumerate(chunk):
                        values=scores[2*i:2*i+2];shifts=[v['shift'] for v in item['rotations']]
                        pred=aggregate_rotations(values,shifts)
                        trace['tasks'][task].append(dict(id=item['id'],gold=item['gold'],prediction=pred,correct=pred==item['gold'],
                            rotation_logprob=values,shifts=shifts,prompt_sha256=[v['prompt_sha256'] for v in item['rotations']]))
                    write(path,trace);profile['counts']={k:len(v) for k,v in trace['tasks'].items()};publish()
                    if batches>=4 and score_seconds/max(1,used)*(total_work-used)>deadline-time.time():
                        raise EvaluationStopped('measured throughput projects past original deadline')
            assert all(len(v)==count for v in trace['tasks'].values())
            trace['complete']=True;write(path,trace)
            profile['phases'][phase]=dict(seconds=time.monotonic()-t,trace_sha256=sha(path),items=sum(len(v) for v in trace['tasks'].values()))
            profile['status']='COMPLETE_PROBE' if phase=='probe' else 'COMPLETE';publish()
        for receipt in campaign['receipts'][name].values():validate_cached_file(Path(receipt['absolute_path']),receipt['expected_sha256'],receipt)
        profile.update(score_seconds=score_seconds,padded_token_positions=used,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,status='COMPLETE');publish()
    except BaseException as exc:
        profile.update(status='FAILED',error_class=type(exc).__name__,reason=str(exc));publish();raise


def summarize(out):
    p=json.loads((out/'prepared.json').read_text());frozen=json.loads((out/'choices.json').read_text())
    selected=json.loads((out/'selected.json').read_text())
    traces={m['key']:json.loads((out/'traces/validation'/(m['name']+'.json')).read_text()) for m in p['models']}
    result={}
    for task in p['tasks']:
        keys=sorted(traces);rows={k:traces[k]['tasks'][task] for k in keys};ids=[r['id'] for r in rows[keys[0]]]
        expected=[(r['id'],r['label']) for r in selected[task]['validation']]
        assert len(ids)==128 and all(traces[k]['complete'] and [(r['id'],r['gold']) for r in rows[k]]==expected for k in keys)
        acc={k:sum(r['correct'] for r in rows[k])/128 for k in keys};best=max(acc.values());choices=frozen['tasks'][task]['choices']
        probe=[r['correct'] for r in rows[choices['probe']]];comparisons={}
        for policy,key in choices.items():
            other=[r['correct'] for r in rows[key]]
            comparisons[policy]=dict(selected=key,correct=sum(other),accuracy=acc[key],finite_panel_regret=best-acc[key],
                probe_minus_policy=(sum(probe)-sum(other))/128,probe_only_correct=sum(a and not b for a,b in zip(probe,other)),
                policy_only_correct=sum(b and not a for a,b in zip(probe,other)))
        result[task]=dict(candidate_accuracy=acc,policies=comparisons,uniform_expected_accuracy=sum(acc.values())/len(acc),
            task_label_status=frozen['tasks'][task]['task_label_status'],n_validation=128)
    return result


def run(out,epoch=None):
    import signal,socket,subprocess,sys
    from datetime import datetime,timezone
    from scripts.task_choice import EvaluationStopped,validate_cached_file
    epoch=time.time() if epoch is None else epoch
    started=time.monotonic()-(time.time()-epoch);children=[];p=json.loads((out/'prepared.json').read_text())
    assert not any((out/n).exists() for n in ['campaign.json','choices.json','release.json','results.json']),'preserve previous attempts/barriers'
    def stop(*_):raise EvaluationStopped('campaign deadline/termination')
    assert 0<=time.time()-epoch<6600,'original campaign budget exhausted or future clock'
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(max(1,int(epoch+6600-time.time())))
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
    assert memory['MemAvailable']>=64*1024**2
    campaign=dict(status='VERIFYING',host=socket.gethostname(),host_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
        deadline_unix=epoch+6600,prepared_sha256=sha(out/'prepared.json'),gpu_devices=[],distinct_weight_models=3,
        memory_available_kib=memory['MemAvailable'],cpu_workers=3,threads_per_worker=8,receipts={},fresh_weight_sha_bytes=0,workers={})
    def publish():campaign['elapsed_seconds']=time.monotonic()-started;write(out/'campaign.json',campaign)
    publish()
    try:
        for name,expected in p['input_sha256'].items():assert sha(ROOT/name)==expected,name
        assert hashlib.sha256((out/'RESEARCH.md').read_text().split('## Outcome')[0].encode()).hexdigest()==p['protocol_sha256']
        for m in p['models']:
            campaign['receipts'][m['name']]={}
            for f in m['weights']:
                path=ROOT/m['local_path']/f['name'];assert path.is_file() and not path.is_symlink()
                inherited=m['prior_receipts'].get(str(path.relative_to(ROOT)))
                if inherited is not None:validate_cached_file(path,f['sha256'],inherited)
                else:
                    before=path.stat()
                    assert sha(path)==f['sha256']
                    after=path.stat()
                    assert all(getattr(before,k)==getattr(after,k) for k in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']),'weight changed during hashing'
                    campaign['fresh_weight_sha_bytes']+=after.st_size
                stat=path.stat();receipt=dict(expected_sha256=f['sha256'],observed_sha256=f['sha256'],absolute_path=str(path.resolve()),
                    dev=stat.st_dev,ino=stat.st_ino,size=stat.st_size,mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns,
                    basis='inherited full-SHA receipt plus unchanged inode/stat' if inherited else 'fresh full SHA')
                campaign['receipts'][m['name']][f['name']]=receipt
        publish()
        for m in p['models']:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',OMP_MAX_ACTIVE_LEVELS='1',MKL_DYNAMIC='FALSE')
            log=(out/(m['name']+'.log')).open('w');seconds=max(1,int(campaign['deadline_unix']-time.time()))
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=20s',str(seconds),sys.executable,'-m','scripts.task_probe','--out',str(out),'--worker',m['name']],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();children.append(child)
            campaign['workers'][m['name']]=dict(process_group=child.pid,stop=f'kill -TERM -- -{child.pid}')
        campaign['status']='PROBES';publish()
        released=False
        while True:
            states={m['name']:json.loads((out/'workers'/(m['name']+'.json')).read_text()) for m in p['models'] if (out/'workers'/(m['name']+'.json')).exists()}
            if any(s['status']=='FAILED' for s in states.values()) or any(c.poll() not in (None,0) for c in children):raise EvaluationStopped('worker failure; fixed cohort stopped')
            if not released and len(states)==3 and all(s['status'] in ['COMPLETE_PROBE','WAITING_FOR_CHOICES'] for s in states.values()):
                probe={m['name']:json.loads((out/'traces/probe'/(m['name']+'.json')).read_text()) for m in p['models']}
                tasks={}
                for task in p['tasks']:
                    correct={};metadata={};ids=None
                    for m in p['models']:
                        trace=probe[m['name']];assert trace['complete'] and trace['model_key']==m['key'] and trace['prepared_sha256']==campaign['prepared_sha256']
                        rows=trace['tasks'][task];current=[r['id'] for r in rows]
                        selected=json.loads((out/'selected.json').read_text())[task]['probe']
                        assert len(rows)==32 and [(r['id'],r['gold']) for r in rows]==[(r['id'],r['label']) for r in selected]
                        assert ids is None or ids==current;ids=current
                        correct[m['key']]=[r['correct'] for r in rows];metadata[m['key']]=dict(params_b=m['params_b'],downloads=m['downloads'],task_match=m['task_matches'][task])
                    tasks[task]=choose(correct,metadata)
                write(out/'choices.json',dict(tasks=tasks,prepared_sha256=campaign['prepared_sha256'],probe_sha256={m['name']:sha(out/'traces/probe'/(m['name']+'.json')) for m in p['models']}))
                write(out/'release.json',dict(choices_sha256=sha(out/'choices.json'),released_utc=datetime.now(timezone.utc).isoformat()))
                released=True;campaign.update(status='VALIDATION',selection_seconds=time.monotonic()-started);publish();print('both choices frozen; validation released',flush=True)
            if all(c.poll() is not None for c in children):
                assert released and all(c.returncode==0 for c in children);break
            time.sleep(1)
        result=summarize(out);campaign['status']='COMPLETE';publish()
        write(out/'results.json',dict(tasks=result,scope='One fixed three-model/two-task constrained-classification diagnostic; no population/C6 claim',
            campaign_sha256=sha(out/'campaign.json'),choices_sha256=sha(out/'choices.json'),prepared_sha256=campaign['prepared_sha256']))
        signal.alarm(0);print('complete',campaign['elapsed_seconds'],flush=True)
    except BaseException as exc:
        campaign.update(status='FAILED',error_class=type(exc).__name__,reason=str(exc));publish();raise
    finally:
        for child in children:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker')
    parser.add_argument('--started-unix',type=float,help='preserve original clock after a pre-worker setup failure')
    args=parser.parse_args();out=args.out.resolve()
    if args.prepare:prepare(out)
    elif args.worker:worker(out,args.worker)
    else:run(out,args.started_unix)
