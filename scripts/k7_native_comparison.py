"""Bounded native modelDNA comparison of two pinned real-model neighborhoods."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,time
from pathlib import Path
from studies.baselines.paths import workspace_root
from huggingface_hub import HfApi
from modeldna import __version__
from modeldna.fingerprint.extract import extract_fingerprint,Fingerprint
from modeldna.compare import compare_fingerprints
from modeldna.verdict import judge,Thresholds
from evidence.hub import HubReader
from lab.artifacts import atomic_json
from studies.baselines.native_source import PinnedSource

root=workspace_root(Path(__file__).resolve().parents[1]);repo=Path(__file__).resolve().parents[1]
out=root/'runs/track-c-k7-native-20261108-v2';out.mkdir(exist_ok=True)
args=argparse.ArgumentParser();args.add_argument('--admit-only',action='store_true');args=args.parse_args()
assert __version__=='0.1.0'
assert os.environ.get('HF_TOKEN'), 'Inherited credential unavailable'
source=root/'runs/track-c-k7-handoff-20261108/candidate-pairs.json'
pairs=json.loads(source.read_text());selected=[pairs[i] for i in [0,1,2,3,4,5,6,8,9]]
assert all(r['truth_tier']=='declared' and r['construction_parent_revision'] is None for r in selected)
groups={};families={}
for row in selected:
 family=('olmo2' if row['child_key'].startswith('allenai/OLMo') else 'tulu3' if row['child_key'].startswith('allenai/Llama') else 'qwen2')
 groups.setdefault(family,set()).update([row['child_key'],row['parent_snapshot_key']])
 assert row['parent_snapshot_key']
for family,keys in groups.items():
 assert len(keys)==4
 for key in keys:families[key]=family
keys=sorted(families);assert len(keys)==12
hashes={}
for p in [source,Path(__file__),repo/'studies/baselines/native_source.py',repo/'docs/research/2026-11-08-k7-native-comparison-protocol.md',root/'runs/track-c-k2-remaining-20261108/modeldna/installed-package.json']:
 hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
if not (out/'protocol.json').exists():
 atomic_json(out/'protocol.json',{'text':(repo/'docs/research/2026-11-08-k7-native-comparison-protocol.md').read_text(),'models':keys,'queries':selected,'groups':{k:sorted(v) for k,v in groups.items()},'thresholds':vars(Thresholds()),'input_sha256':hashes,'owned_pid':os.getpid()})

def safe_error(exc):
 status=getattr(getattr(exc,'response',None),'status_code',None)
 return {'category':type(exc).__name__,'http_status':status if isinstance(status,int) else None}

metadata_path=out/'metadata.json'
if not metadata_path.exists():
 previous_metadata=json.loads((root/'runs/track-c-k7-native-20261108/metadata.json').read_text())
 prior_records={r['key']:r for r in previous_metadata['records']}
 api=HfApi(endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co'),token=os.environ['HF_TOKEN']);records=[];started=time.monotonic()
 for key in keys:
  model,revision=key.rsplit('@',1);t=time.monotonic()
  if key in prior_records:
   records.append(prior_records[key]);continue
  try:
   info=api.model_info(model,revision=revision,files_metadata=True,timeout=30);assert info.sha==revision
   files=[]
   for f in info.siblings or []:
    lfs=getattr(f,'lfs',None);sha=getattr(lfs,'sha256',None) if lfs is not None else None
    files.append({'name':f.rfilename,'size':f.size,'lfs_sha256':sha})
   record={'key':key,'status':'admitted','files':files,'metadata_seconds':time.monotonic()-t}
  except Exception as exc:record={'key':key,'status':'no_access_or_metadata_failure',**safe_error(exc),'metadata_seconds':time.monotonic()-t}
  records.append(record);atomic_json(out/'metadata-progress.json',{'records':records});print('metadata',key,record['status'],flush=True)
 atomic_json(metadata_path,{'records':records,'elapsed_seconds':time.monotonic()-started,'note':'Retained metadata is not a measurement of HTTP wire bytes'})
admission_path=out/'neighborhood-admission.json'
if not admission_path.exists():
 metadata=json.loads(metadata_path.read_text());by_key={r['key']:r for r in metadata['records']};checker=HubReader();admission=[]
 for family,members in groups.items():
  member_status=[]
  for key in sorted(members):
   record=by_key[key];files=record.get('files',[]);shards=sorted(f['name'] for f in files if f['name'].endswith('.safetensors') and '/' not in f['name'])
   if record['status']!='admitted' or not shards:
    member_status.append({'key':key,'status':'no_safetensors' if not shards else 'metadata_failure'});continue
   model,revision=key.rsplit('@',1)
   try:
    remote=checker.resolve(model,shards[0],revision);member_status.append({'key':key,'status':'head_access_confirmed','size':remote.size})
   except Exception as exc:member_status.append({'key':key,'status':'no_access_or_head_failure',**safe_error(exc)})
  admission.append({'family':family,'members':member_status,'complete_neighborhood_admitted':all(r['status']=='head_access_confirmed' for r in member_status)})
 atomic_json(admission_path,{'groups':admission,'reader_stats':checker.stats})
 print('neighborhood admission',[(r['family'],r['complete_neighborhood_admitted']) for r in admission],flush=True)
if args.admit_only:raise SystemExit(0)
assert not (out/'results.json').exists(), 'Do not overwrite completed results'
metadata=json.loads(metadata_path.read_text());hashes[str(metadata_path.relative_to(root))]=hashlib.sha256(metadata_path.read_bytes()).hexdigest()
admission=json.loads(admission_path.read_text());blocked={r['family'] for r in admission['groups'] if not r['complete_neighborhood_admitted']}
hashes[str(admission_path.relative_to(root))]=hashlib.sha256(admission_path.read_bytes()).hexdigest()
started=time.monotonic();deadline=started+6600;fps={};extractions=[];reader=HubReader();stop=False
with ThreadPoolExecutor(8) as pool:
 for record in metadata['records']:
  key=record['key'];fp_path=out/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.json');model,revision=key.rsplit('@',1)
  if families[key] in blocked:
   extractions.append({'key':key,'status':'not_run','reason':'incomplete_supported_accessible_neighborhood'});continue
  if record['status']!='admitted' or stop:
   extractions.append({'key':key,'status':'not_run','reason':'metadata_not_admitted' if not stop else 'prior_transport_or_budget_failure'});continue
  previous_fp=root/'runs/track-c-k7-native-20261108'/fp_path.name
  if previous_fp.exists() and not fp_path.exists():atomic_json(fp_path,json.loads(previous_fp.read_text()))
  if fp_path.exists():
   fp=Fingerprint.from_dict(json.loads(fp_path.read_text()));assert fp.model_id==key and fp.revision==revision
   fps[key]=fp;extractions.append({'key':key,'status':'reused','bytes_read':0,'fingerprint':fp_path.name});continue
  src=PinnedSource(model,revision,[f['name'] for f in record['files']],reader,pool,deadline=deadline,byte_cap=1024**3);t=time.monotonic()
  try:
   fp=extract_fingerprint(src,mode='fast',model_id=key,revision=revision)
   advertised={f['name']:f for f in record['files']}
   for name,remote in src.remotes.items():
    meta=advertised[name]
    if meta['size'] is not None:assert remote.size==meta['size']
    if remote.sha256 and meta['lfs_sha256']:assert remote.sha256==meta['lfs_sha256']
   atomic_json(fp_path,fp.to_dict());fps[key]=fp
   item={'key':key,'status':'tested','fingerprint':fp_path.name,'bytes_read':src.bytes_read,'seconds':time.monotonic()-t,'files_read':[{'name':name,'size':remote.size,'advertised_sha256':remote.sha256} for name,remote in sorted(src.remotes.items())]}
  except Exception as exc:
   item={'key':key,'status':'not_run','bytes_read':src.bytes_read,'seconds':time.monotonic()-t,**safe_error(exc)}
   # Do not send another neighborhood's payloads after a transport/cap failure.
   stop=True
  extractions.append(item);atomic_json(out/'extraction-progress.json',{'extractions':extractions,'reader_stats':reader.stats});print('extract',key,item['status'],item['bytes_read'],round(item.get('seconds',0),2),flush=True)
rows=[];verdicts=[]
for row in selected:
 key=row['child_key'];parent=row['parent_snapshot_key'];family=families[key];candidates=sorted(groups[family]-{key});evs=[]
 for candidate in candidates:
  if key not in fps or candidate not in fps:
   rows.append({'query':key,'candidate':candidate,'status':'not_run','reason':'native_fingerprint_missing','truth_tier':'declared'});continue
  ev=compare_fingerprints(fps[key],fps[candidate]);evs.append((ev,family));rows.append({'query':key,'candidate':candidate,'status':'tested','declared_parent':candidate==parent,'truth_tier':'declared','construction_input_revision':None,'evidence':ev.to_dict()})
 for present in (True,False):
  intended=[c for c in candidates if present or c!=parent]
  if key not in fps or any(c not in fps for c in intended):
   verdicts.append({'query':key,'parent_present':present,'status':'not_run','reason':'incomplete_fixed_neighborhood'});continue
  chosen=[(ev,fam) for ev,fam in evs if present or ev.candidate_id!=parent]
  verdict=judge(chosen,suspect_dtype=fps[key].arch.torch_dtype,candidate_dtypes={c:fps[c].arch.torch_dtype for c in intended})
  verdicts.append({'query':key,'declared_parent':parent,'parent_present':present,'status':'tested','truth_tier':'declared','verdict':verdict.to_dict()})
summary={'fingerprints':len(fps),'planned_models':12,'pairs_tested':sum(r['status']=='tested' for r in rows),'planned_pairs':27,'verdicts_tested':sum(v['status']=='tested' for v in verdicts),'planned_verdicts':18,'complete':len(fps)==12 and all(r['status']=='tested' for r in rows) and all(v['status']=='tested' for v in verdicts),'construction_edges_verified':0}
atomic_json(out/'results.json',{'summary':summary,'rows':rows,'verdicts':verdicts,'extractions':extractions,'input_sha256':hashes,'reader_stats':reader.stats,'elapsed_seconds':time.monotonic()-started,'metadata_seconds':metadata['elapsed_seconds'],'source_bytes_new':sum(x.get('bytes_read',0) for x in extractions),'source_bytes_fingerprints_cumulative':sum(fp.bytes_read for fp in fps.values()),'package_version':__version__})
print(json.dumps(summary),flush=True)
