"""Materialize bounded licensed launch inputs from locally acquired pinned JSONL.

No network, model-weight downloads, optimizers, or training occur here. Raw source
facts/cards/selected rows must already exist in bundle/data/{task}.
"""
import argparse
import json
import time
from pathlib import Path

from lab.artifacts import content_hash
from lab.launches import causal_row, preference_row, file_hash, preflight, split_ids
from lab.registry import expand_inventory

PARTITIONS={'training':(0,4096),'direction':(4096,4352),'validation':(4352,4608),'scoring':(4608,4864)}
TOKEN_FILES=('tokenizer.json','tokenizer_config.json','special_tokens_map.json')


def prepare(root, bundle, model_id='smol-135'):
    import torch
    from safetensors.torch import save_file, load_file
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    if (bundle/'manifest.json').exists():
        raise FileExistsError('immutable bundle already has a manifest; choose a new version')
    pin=json.loads((Path(__file__).parents[1]/'pins.json').read_text())[model_id]
    tokenizer_dir=root/'lab/inputs/evidence'/model_id
    token_hashes={n:file_hash(tokenizer_dir/n) for n in TOKEN_FILES if (tokenizer_dir/n).exists()}
    if content_hash(token_hashes)!=pin['tokenizer_sha256']:raise ValueError('tokenizer identity mismatch')
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,trust_remote_code=False)
    source=f'lab/inputs/cohort-v1-{model_id}-{pin["revision"][:12]}'
    relative=lambda path:path.relative_to(root).as_posix()
    evidence={relative(tokenizer_dir/n):h for n,h in token_hashes.items()}
    evidence[pin['license_path']]=pin['license_sha256']
    jobs=[];datasets={};token_specs={}
    for task,label in [('instruction','instruction'),('dialogue','instruction'),('preference','preference'),('decision','decision')]:
        begin=time.monotonic(); data_dir=bundle/'data'/label
        acquisition=json.loads((data_dir/'acquisition.json').read_text())
        publisher=json.loads((data_dir/'publisher-metadata.json').read_text())
        license=publisher['cardData']['license']
        if isinstance(license,list) and len(license)==1:license=license[0]
        if publisher['sha']!=acquisition['revision']:raise ValueError('publisher revision mismatch')
        for name in ('README.md','acquisition.json','publisher-metadata.json','selected-rows.jsonl'):
            evidence[relative(data_dir/name)]=file_hash(data_dir/name)
        raw=[json.loads(line) for line in (data_dir/'selected-rows.jsonl').read_text().splitlines()]
        ids={};parts={};filter_counts={}
        for split,(lo,hi) in PARTITIONS.items():
            rows=[];selected=[]
            for entry in raw[lo:hi]:
                row=entry['row']
                if task in {'instruction','dialogue'}:
                    messages=row['messages'][:2] if task=='instruction' else row['messages']
                    result=causal_row(tokenizer,messages,512)
                    value={'input_ids':result} if result is not None else None
                elif task=='preference':
                    # Raw chosen/rejected contain complete conversations; render the
                    # same prompt and only the final assistant completion separately.
                    chosen,rejected=row['chosen'],row['rejected']
                    if chosen[:-1]!=rejected[:-1] or not chosen[:-1]:continue
                    prompt=''.join(m['role']+': '+m['content']+'\n' for m in chosen[:-1])+'assistant: '
                    result=preference_row(tokenizer,prompt,chosen[-1]['content'],rejected[-1]['content'],512)
                    value=dict(zip(('input_ids','rejected_input_ids','chosen_mask','rejected_mask'),result)) if result else None
                else:
                    tokens=tokenizer.encode(row['title']+'\n'+row['content'],add_special_tokens=False)[:128]
                    if not tokens or 0 in tokens:continue
                    value=dict(input_ids=tokens+[0]*(128-len(tokens)),attention_mask=[1]*len(tokens)+[0]*(128-len(tokens)),labels=int(row['label']))
                if value is None:continue
                selected.append(acquisition['repo_id']+'@'+acquisition['revision']+':'+acquisition['source_file']['rfilename']+':'+str(entry['row_index']))
                rows.append(value)
            if not rows:raise ValueError(task+': empty '+split+' after no-padding filter')
            ids[split]=selected;filter_counts[split]={'considered':hi-lo,'accepted':len(rows),'rejected':hi-lo-len(rows)}
            tensors={k:torch.tensor([r[k] for r in rows],dtype=torch.long) for k in rows[0]}
            path=bundle/'tokens'/model_id/task/(split+'.safetensors');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                previous=load_file(str(path))
                if previous.keys()!=tensors.keys() or any(not torch.equal(previous[k],v) for k,v in tensors.items()):raise ValueError('interrupted preparation token mismatch')
            else:
                save_file(tensors,str(path))
            parts[split]={'path':relative(path),'sha256':file_hash(path),'rows':len(rows),'bytes':path.stat().st_size}
            evidence[relative(path)]=parts[split]['sha256']
        split_ids(ids)
        preprocessing=(dict(format='classification_tokens_v1',padding='right',label_names=['negative','positive'],pad_token_id=0) if task=='decision' else
                       dict(format='preference_tokens_v1',loss='completion_only',padding='none',template='role-colon-newline-v1',prompt_truncation=256,completion_truncation='suffix-budget-prefix') if task=='preference' else
                       dict(format='causal_tokens_v1',loss='all_tokens',padding='none',template='role-colon-newline-v1',turns='first-user-assistant' if task=='instruction' else 'all',truncation='first-512',short_rows='excluded'))
        data_pin=dict(model_id=label,kind='dataset',status='pinned',repo_id=acquisition['repo_id'],revision=acquisition['revision'],license=license,
            license_path=relative(data_dir/'README.md'),license_sha256=file_hash(data_dir/'README.md'),license_evidence_type='pinned_readme_declaration',
            split={'instruction':'train_sft','preference':'train_prefs','decision':'train'}[label],selected_index_sha256=content_hash(ids['training']),preprocessing=preprocessing,
            source_file=acquisition['source_file']['rfilename'],source_file_sha256=acquisition['source_file']['lfs']['sha256'],selected_rows_sha256=file_hash(data_dir/'selected-rows.jsonl'))
        inputs=dict(model_path=source,tokens_path=parts['training']['path'],model_pin=pin,data_pin=data_pin,sample_ids=ids['training'],heldout_ids=sum((ids[n] for n in ('direction','validation','scoring')),[]),tokens_sha256=parts['training']['sha256'])
        token_specs[task]=dict(inputs=inputs,selections=ids)
        datasets[task]=dict(pin=data_pin,partitions=parts,filter_counts=filter_counts,tokenizer_sha256=pin['tokenizer_sha256'],tokenizer_files=token_hashes,
            preprocessing_seconds=time.monotonic()-begin,acquisition=acquisition)
    # Extra small-base validation jobs retain scientific optimizer settings, but
    # cannot satisfy the four-anchor G1 quotas.
    for task in ('instruction','dialogue'):
        for rank in (8,64,256):
            jobs.append(('lora',task,dict(model_id=f'{model_id}-launch-lora-{task}-r{rank}',rank=rank,alpha=rank,steps=128,effective_batch=16,learning_rate=2e-4)))
    for index in (0,1):
        jobs.append(('sft','instruction',dict(model_id=f'{model_id}-launch-sft-s{index}',steps=256,effective_batch=16,learning_rate=2e-5,training_seed=20261028+index)))
        jobs.append(('dpo','preference',dict(model_id=f'{model_id}-launch-dpo-s{index}',steps=128,effective_batch=8,learning_rate=5e-6,beta=0.1,parents=[f'{model_id}-launch-sft-s{index}'])))
    jobs.append(('decision','decision',dict(model_id=f'{model_id}-launch-decision-s0',steps=32,effective_batch=16,learning_rate=2e-5,label_names=['negative','positive'],pad_token_id=0,embedded_adapter=False)))
    configs=[]
    configdir=bundle/'configs';configdir.mkdir(exist_ok=True)
    for edge,task,overrides in jobs:
        recipe=dict(edge_type='sft' if edge=='decision' else edge,parents=[model_id],parent_identities={model_id:pin['repo_id']+'@'+pin['revision']},seed=20261028,
            sequence_length=128 if edge=='decision' else 512,optimizer='adamw',scheduler='constant',dropout=0,rslora=False,device='cpu',torch_dtype='bfloat16')
        recipe.update(overrides)
        if edge=='dpo':recipe['parent_identities']={} # Missing local key is explicit until actual SFT completes.
        if edge=='lora':recipe['target_modules']=['q_proj','v_proj']
        value=dict(schema_version='1',trainer=edge,recipe=recipe,**token_specs[task],evidence_hashes=evidence,quota_credit=False)
        target=configdir/(recipe['model_id']+'.json');target.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
        try:result=preflight(value,root)
        except (ValueError,OSError,KeyError) as error:result={'status':'BLOCKED','reason':str(error)}
        configs.append(dict(model_id=recipe['model_id'],trainer=edge,config=relative(target),config_sha256=file_hash(target),preflight=result,output='lab/'+recipe['model_id']))
    official=[]
    inventory=expand_inventory({})
    for edge in ('lora','sft','dpo','decision'):
        for recipe in inventory[edge]:
            official.append(dict(trainer=edge,recipe=recipe,status='BLOCKED',missing=['local pinned anchor safetensors','anchor-specific frozen token artifacts']+(['completed matching SFT output and actual local key'] if edge=='dpo' else [])))
    manifest=dict(schema_version='1',status='PREPARED_NO_TRAINING',quota_credit=False,model_pin=pin,datasets=datasets,launches=configs,official_inventory=official,
        replacements=[dict(original='trl-lib/ultrafeedback_binarized@47124cb5778f5d50de1c7676a412828f3ea7c555',reason='publisher license absent',replacement=datasets['preference']['pin']['repo_id']),dict(original='stanfordnlp/sst2@8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb',reason='publisher license unknown',replacement=datasets['decision']['pin']['repo_id'])],
        training_time=dict(measured=False,note='No forward/backward optimizer timing performed; scientific plan full batches provisional LoRA40-80GPUh/SFT+DPO70-130GPUh/decision0.3-1.5GPUh; these are not measured Smol135 predictions.'),
        resource_notes=dict(source_weight_bytes=sum((root/source/f).stat().st_size for f in pin['weight_sha256']) if (root/source).exists() else None,
            planned_output_weight_bytes_each=2*134515008,token_bytes=sum(p.stat().st_size for p in (bundle/'tokens').rglob('*.safetensors')),
            peak_memory='UNMEASURED: optimizer fp32 full-tuning parameters, gradients and Adam moments approximately16bytes/parameter plus activations/logits; DPO adds frozen fp32 reference and multiple logits tensors. No GPU allocated.',
            storage='11 extra outputs about3GB primary weights plus LoRA factors, staging, safe loading headroom; exact output bytes measured only after human jobs.'),
        evidence_hashes=evidence)
    (bundle/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    (bundle/'manifest.sha256').write_text(file_hash(bundle/'manifest.json')+'  manifest.json\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args()
    manifest=prepare(args.data_root.resolve(),args.bundle.resolve())
    print(json.dumps({'status':manifest['status'],'configs':len(manifest['launches']),'official_blocked':len(manifest['official_inventory']),'counts':{k:{s:v['rows'] for s,v in d['partitions'].items()} for k,d in manifest['datasets'].items()}},sort_keys=True))

if __name__=='__main__':main()
