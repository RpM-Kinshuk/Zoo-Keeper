"""Three real scoped base→SFT→DPO→quantized chains; no formal quota credit.

Existing source/training outputs and A fingerprints are immutable inputs. New
terminal checkpoints use the reviewed quantization writer. Inherited T1 files
are reused, never recomputed; only three new terminals need extraction.
"""
import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import time

from lab.artifacts import atomic_json, content_hash, write_checkpoint
from lab.chains import chain_paths
from lab.launches import check_files, file_hash
from lab.quantization import checked_source, run_quantization
from lab.registry import verify_tables
from lab.schema import MODEL_COLUMNS, EDGE_COLUMNS, canonical_json, contained_path

RUN='runs/track-b-chain-cohort-v1'
TRAINING='runs/track-b-training-gpu-v1'
DELIVERY='runs/track-a-l12-20261102.mnbqfy_y'
INHERITED={'smol-135',*[f'smol-135-gpu-v1-{edge}-s{seed}' for edge in ('sft','dpo') for seed in (0,1)]}
TERMINALS=[(0,4),(0,8),(1,4)]


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('immutable chain evidence differs: '+path.name)
    else:atomic_json(path,value)


def scoped_tables(ready):
    import pandas as pd
    models=[];edges=[]
    for row in ready['sources']+ready['outputs']:
        source=bool(row.get('repo_id'))
        models.append(dict(model_id=row['model_id'],kind='base' if source else 'derived',
            path=row['repo_id']+'@'+row['revision'] if source else row['path'].removeprefix('lab/'),
            base_family='llama',license=row['license']))
        for parent in row['parents']:
            edges.append(dict(child=row['model_id'],parent=parent,edge_type=row['edge_type'],role='parent',
                coefficient='',params_json=canonical_json(dict(scope='extra_depth_three_chain',formal_quota_credit=False))))
    return pd.DataFrame(models,columns=MODEL_COLUMNS),pd.DataFrame(edges,columns=EDGE_COLUMNS)


def verify_chain_ready(ready):
    rows=ready['sources']+ready['outputs'];by_id={r['model_id']:r for r in rows}
    if len(by_id)!=len(rows):raise ValueError('duplicate chain IDs')
    if any(not r.get('complete') or any(p not in by_id or not by_id[p].get('complete') for p in r['parents']) for r in rows):
        raise ValueError('missing or incomplete chain intermediate')
    models,edges=scoped_tables(ready);paths=chain_paths(models,edges)
    expected={f'chain-v1-smol-135-dpo-s{seed}-q{bits}' for seed,bits in TERMINALS}
    if len(paths)!=3 or {p[-1] for p in paths}!=expected or any(len(p)!=4 for p in paths):
        raise ValueError('three complete depth-three scoped paths required')
    return dict(status='SCOPED_CHAINS_COMPLETE',chain_paths=paths,scoped_chain_count=3,
                formal_chain_credit=0,formal_minimum_status='PENDING_OFFICIAL_ANCHORS')


def prepare(root):
    root=Path(root).resolve();run=root/RUN
    training=json.loads((root/TRAINING/'ready.json').read_text())
    inherited={r['model_id']:copy.deepcopy(r) for r in training['sources']+training['outputs'] if r['model_id'] in INHERITED}
    if set(inherited)!=INHERITED or any(not r.get('complete') for r in inherited.values()):
        raise ValueError('completed inherited base/SFT/DPO inputs required')
    index_path=root/DELIVERY/'artifact-index43.csv';index={r['model_id']:r for r in csv.DictReader(index_path.open())}
    recipes=[];evidence={str((root/TRAINING/'ready.json').relative_to(root)):file_hash(root/TRAINING/'ready.json'),
        index_path.relative_to(root).as_posix():file_hash(index_path)}
    for seed,bits in TERMINALS:
        parent=f'smol-135-gpu-v1-dpo-s{seed}';row=inherited[parent];entry=index[parent]
        manifest_path=contained_path(root,entry['manifest'],'manifest');manifest=json.loads(manifest_path.read_text())
        identity=manifest['repo_id']+'@'+manifest['revision']
        if identity!=entry['model_key'] or manifest['repo_id']!='lab/'+parent:raise ValueError('A parent identity differs')
        path=contained_path(root,row['path'],parent);weights={f['filename']:f['sha256'] for f in manifest['files'] if f['filename'].endswith('.safetensors')}
        if weights!=row['weight_sha256']:raise ValueError('parent manifest/ready weight hashes disagree')
        hashes={name:file_hash(path/name) for name in ('config.json','recipe.json','lineage.json','training.json')}
        evidence[entry['manifest']]=file_hash(manifest_path)
        evidence.update({(path/name).relative_to(root).as_posix():h for name,h in hashes.items()})
        provenance=json.loads((path/'recipe.json').read_text());training_meta=json.loads((path/'training.json').read_text())
        if (provenance['recipe']['edge_type']!='dpo' or provenance['model_id']!=parent
                or provenance['config_hash']!=content_hash(provenance['recipe'])
                or training_meta['steps_completed']!=provenance['recipe']['steps']
                or training_meta['reference_unchanged'] is not True):raise ValueError('parent DPO provenance is incomplete')
        pin=dict(path=row['path'].removeprefix('lab/'),model_key=identity,
            config_file_sha256=hashes['config.json'],config_sha256=content_hash(json.loads((path/'config.json').read_text())),
            weight_sha256=weights,manifest=dict(repo_id=manifest['repo_id'],revision=manifest['revision'],
                files=[dict(filename=n,sha256=h) for n,h in sorted(weights.items())]),
            provenance_sha256=hashes,extractor_manifest_sha256=file_hash(manifest_path))
        recipe=dict(model_id=f'chain-v1-smol-135-dpo-s{seed}-q{bits}',edge_type='quantization_sim',bits=bits,
            parents=[parent],parent_identities={parent:identity},inputs={parent:pin},
            cohort='track-b-chain-cohort-v1',synthetic=False,formal_quota_credit=False)
        checked_source(recipe,{'root':root/'lab'});recipes.append(recipe)
    code_paths=['lab/scripts/chain_cohort.py','lab/quantization.py','lab/artifacts.py','lab/chains.py']
    plan=dict(schema_version='1',recipes=recipes,inherited_ready=dict(sources=[inherited['smol-135']],
        outputs=[inherited[k] for k in sorted(INHERITED-{'smol-135'})]),evidence_hashes=evidence,
        generator_sha256={p:file_hash(Path(__file__).parents[2]/p) for p in code_paths},
        limits=dict(weight_models=3,wall_seconds=6600,gpu_hours=0,torch_threads=4),
        formal_quota_credit=False,scope='three extra Smol135 paths; no original L10/G1 credit')
    immutable_json(run/'frozen.json',plan)
    immutable_json(run/'runner-snapshot.json',dict(runner_sha256=file_hash(__file__)))
    return plan


def generate(root):
    import torch
    from transformers import AutoModelForCausalLM
    root=Path(root).resolve();run=root/RUN;plan=prepare(root);check_files(root,plan['evidence_hashes'])
    torch.set_num_threads(4);begin=time.monotonic()
    ready_path=run/'ready.json'
    ready=json.loads(ready_path.read_text()) if ready_path.exists() else dict(schema_version='1',**plan['inherited_ready'],
        formal_quota_credit=False,scope=plan['scope'])
    inherited_ids=INHERITED
    for recipe in plan['recipes']:
        existing=next((r for r in ready['outputs'] if r['model_id']==recipe['model_id']),None)
        if existing:
            check_files(root,{existing['path']+'/'+name:h for name,h in existing['weight_sha256'].items()})
            if json.loads((root/existing['path']/'recipe.json').read_text())['recipe']!=recipe:raise ValueError('ready chain recipe differs')
            continue
        if time.monotonic()-begin>plan['limits']['wall_seconds']:raise RuntimeError('bounded chain generation budget reached')
        start=time.monotonic();target=write_checkpoint(recipe['model_id'],root/'lab',recipe,
            lambda stage:run_quantization(recipe,{'root':root/'lab'},stage),resume=True)
        model=AutoModelForCausalLM.from_pretrained(target,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,dtype=torch.bfloat16)
        parameters=sum(p.numel() for p in model.parameters())
        if parameters!=134515008 or any(p.is_floating_point() and p.dtype!=torch.bfloat16 for p in model.parameters()):
            raise ValueError('safe BF16 chain reload differs from Smol135 contract')
        del model;gc.collect()
        row=dict(model_id=recipe['model_id'],path=target.relative_to(root).as_posix(),root_id='smol-135',
            parents=recipe['parents'],kind='derived',edge_type='quantization_sim',license='apache-2.0',
            repo_id=None,revision=None,split='development',complete=True,synthetic=False,bits=recipe['bits'],
            weight_sha256={p.name:file_hash(p) for p in target.glob('*.safetensors')},
            config_file_sha256=file_hash(target/'config.json'),parameters_reloaded=parameters,
            weight_bytes=sum(p.stat().st_size for p in target.glob('*.safetensors')),seconds=time.monotonic()-start)
        ready['outputs'].append(row);atomic_json(ready_path,ready)
        print(json.dumps(dict(phase='chain_terminal_complete',model_id=row['model_id'],seconds=row['seconds'])),flush=True)
    summary=verify_chain_ready(ready);models,edges=scoped_tables(ready);verification=verify_tables(models,edges,root/'lab')
    if verification['status']!='PASS':raise ValueError('scoped chain verification failed: '+str(verification))
    scoped=run/'scoped';scoped.mkdir(exist_ok=True);models.to_csv(scoped/'models.csv',index=False);edges.to_csv(scoped/'edges.csv',index=False)
    atomic_json(run/'verification.json',verification)
    atomic_json(run/'summary.json',dict(**summary,ready_sha256=file_hash(ready_path),
        elapsed_seconds=time.monotonic()-begin,output_weight_bytes=sum(r['weight_bytes'] for r in ready['outputs'] if r['model_id'] not in inherited_ids),gpu_hours=0))
    return ready


def fingerprint(root):
    from lab.cohort_evidence import measure_ready,sha
    root=Path(root).resolve();run=root/RUN;ready=json.loads((run/'ready.json').read_text());verify_chain_ready(ready)
    inherited={r['model_id']:r for r in ready['sources']+ready['outputs'] if r['model_id'] in INHERITED}
    old=json.loads((root/TRAINING/'evidence/progress.json').read_text());rows=[]
    for row in old['rows']:
        if row['model_id'] not in inherited:continue
        if row['ready_row_sha256']!=content_hash(inherited[row['model_id']]):raise ValueError('inherited ready evidence changed')
        for key in ('manifest','fingerprint'):
            if sha(root/row[key])!=row[key+'_sha256']:raise ValueError('inherited evidence bytes changed')
        rows.append(row)
    if {r['model_id'] for r in rows}!=INHERITED:raise ValueError('five inherited T1 rows required')
    progress_path=run/'evidence/progress.json'
    if not progress_path.exists():atomic_json(progress_path,dict(schema_version='1',rows=rows))
    immutable_json(run/'reused-evidence.json',dict(model_ids=sorted(INHERITED),
        scope='existing fingerprints copied as progress pointers; their tensor bytes are prior measurements, not new reads',
        manifest_fingerprint_sha256={r['model_id']:{k:r[k] for k in ('manifest_sha256','fingerprint_sha256')} for r in rows}))
    return measure_ready(run/'ready.json',data_root=root,run_root=run,
        model_ids=[r['model_id'] for r in ready['outputs'] if r['model_id'] not in INHERITED])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','generate','fingerprint'])
    parser.add_argument('--data-root',type=Path,required=True);args=parser.parse_args(argv)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    result={'prepare':prepare,'generate':generate,'fingerprint':fingerprint}[args.command](args.data_root)
    print(json.dumps(dict(command=args.command,status='COMPLETE',run=RUN)),flush=True)


if __name__=='__main__':main()
