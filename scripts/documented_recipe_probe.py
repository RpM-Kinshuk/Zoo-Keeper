"""Bounded metadata-only recipe admission; malformed checkpoints stay explicit."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'Zoo-Keeper'),str(ROOT/'Model-Zoo')]
from huggingface_hub import HfApi
from evidence.hub import HubReader
from lab.artifacts import atomic_json
from scripts.scaling_pools import header_reader,PinnedInfo
from evidence.manifest import build_manifest,write_manifest
out=ROOT/'runs/documented-recipe-admission-20261112';out.mkdir(exist_ok=True)
assert not (out/'results.json').exists()
token=next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)),None);assert token
api=HfApi(token=token);reader=HubReader(token=token);headers=header_reader(token=token);rows=[]
# Scope choices depend on metadata/recipe readability, never observed detector outcomes.
for repo in ['pravdin/Qwen2.5-0.5B-linear-merge','MergekitCloud/mergekit-125','Sakalti/SJT-0.5B']:
 row=dict(repo_id=repo)
 try:
  info=api.model_info(repo,files_metadata=True,timeout=15);row['revision']=info.sha
  for name in ['README.md','mergekit_config.yml','config.json']:
   if name not in {s.rfilename for s in info.siblings}:continue
   remote=reader.resolve(repo,name,info.sha);assert remote.size<=1024**2
   data=reader.read(remote,0,remote.size);p=out/'sources'/repo.replace('/','--')/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
   row[name]=dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest())
  try:
   manifest=build_manifest(repo,info.sha,headers,PinnedInfo(info));p=write_manifest(manifest,out/'manifests')
   row['manifest']=str(p.relative_to(ROOT));row['header_status']='readable'
  except Exception as exc:row.update(header_status='error',header_error_class=type(exc).__name__)
 except Exception as exc:row.update(metadata_error=type(exc).__name__)
 rows.append(row);atomic_json(out/'progress.json',dict(rows=rows))
atomic_json(out/'results.json',dict(rows=rows,reader_stats=reader.stats,header_stats=headers.stats,weight_payload_reads=0))
print(json.dumps([{k:v for k,v in r.items() if k not in ['README.md','config.json','mergekit_config.yml']} for r in rows],indent=2))
