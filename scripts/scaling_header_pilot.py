"""Pinned metadata-only feasibility pilot reusing the guarded manifest path."""
import csv,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
root=Path('/jumbo/yaoqingyang/kinshuk/zoo'); sys.path[:0]=[str(root/'Model-Zoo')]
from evidence.manifest import build_manifest,write_manifest
from zoo.schema import arch_signature
from huggingface_hub import HfApi
out=root/'runs/scaling-pool-discovery-20261110'
source=root/'runs/track-a-k8-headers-20261108/headers.py'
spec=importlib.util.spec_from_file_location('existing_header_guard',source); guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
rows=list(csv.DictReader((out/'candidates.csv').open())); selected=[]
base_repos={}
for anchor in ['q25-05','q3-06']:
 recipe=json.loads((root/'lab-restored-archive/models'/f'{anchor}-abliteration-heretic/recipe.json').read_text())
 identity=recipe['parent_identities'][anchor]; repo,pin=identity.rsplit('@',1);base_repos[anchor]=repo
 selected.append(dict(anchor=anchor,role='pinned_anchor',repo_id=repo,revision=pin))
for anchor in ['q25-05','q3-06']:
 candidates=[r for r in rows if r['anchor']==anchor and r['repo_id'] not in base_repos.values() and r['weight_format']=='safetensors' and r['is_adapter']=='False']
 used=set()
 for role,predicate,n in [('direct',lambda r:r['direct_declared_base']=='True' and r['config_conflicts']=='[]',2),('name_only',lambda r:r['direct_declared_base']=='False' and r['name_match']=='True' and r['config_conflicts']=='[]',2),('known_config_conflict',lambda r:r['config_conflicts']!='[]',1)]:
  ordered=sorted((r for r in candidates if predicate(r) and r['repo_id'] not in used),key=lambda r:(-int(r['downloads']),r['repo_id']))[:n]
  for r in ordered:
   used.add(r['repo_id']);selected.append(dict(anchor=anchor,role=role,repo_id=r['repo_id'],revision=''))
requested=out/'pilot-requested.csv'
assert not requested.exists(), 'new pilot requires new output directory'
with requested.open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['anchor','role','repo_id','revision']);writer.writeheader();writer.writerows(selected)
assert len(selected)<=12 and len({r['repo_id'] for r in selected})==len(selected)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
freeze=dict(requested_sha256=digest(requested),driver_sha256=digest(Path(__file__)),protocol_sha256=digest(out/'HEADER-PILOT.md'),guard_sha256=digest(source),manifest_source_sha256=digest(root/'Model-Zoo/evidence/manifest.py'),hub_source_sha256=digest(root/'Model-Zoo/evidence/hub.py'),network_model_requests=0,weight_payload_reads=0)
(out/'pilot-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
token=next((os.environ[k] for k in ['HF_TOKEN','HF_UPLOAD','HF_UPLOAD_TOKEN'] if os.environ.get(k)),None);assert token
api=HfApi(token=token); reader=guard.HeaderOnlyReader(token=token); results=[]; manifests={}; start=time.monotonic()
class PinnedInfo:
 def __init__(self,info): self.info=info
 def model_info(self,repo_id,revision,files_metadata=True):
  assert self.info.id.lower()==repo_id.lower() and self.info.sha==revision
  return self.info
for row in selected:
 case_start=time.monotonic(); result=dict(row)
 try:
  info=api.model_info(row['repo_id'],revision=row['revision'] or 'main',files_metadata=True,timeout=15)
  freeze['network_model_requests']+=1
  result['revision']=info.sha
  manifest=build_manifest(row['repo_id'],info.sha,reader,PinnedInfo(info))
  path=write_manifest(manifest,out/'pilot-manifests'); result['manifest']=str(path.relative_to(root));result['manifest_sha256']=digest(path)
  result.update(access=manifest['access'],is_adapter=manifest['is_adapter'],view_counts=manifest.get('view_counts'),params_by_dtype=manifest.get('params_by_dtype'),quantization_config=manifest['config'].get('quantization_config'))
  config=manifest['config']; expected=json.loads((root/'lab-restored-archive/models'/f"{row['anchor']}-abliteration-heretic/config.json").read_text())
  result['config_signature_match']=arch_signature(config,[])==arch_signature(expected,[])
  def shape_map(m):
   return {t['name']:t['shape'] for t in m['tensors'] if not t['name'].endswith(('embed_tokens.weight','lm_head.weight'))}
  if row['role']=='pinned_anchor': manifests[row['anchor']]=manifest
  anchor_manifest=manifests.get(row['anchor'])
  result['nonvocab_tensor_shapes_match']=shape_map(manifest)==shape_map(anchor_manifest) if anchor_manifest else None
  result['plain_dtypes']=bool(manifest.get('params_by_dtype')) and set(manifest['params_by_dtype'])<=set(['BF16','F16','F32','F64'])
  result['header_compatible']=bool(result['config_signature_match'] and result['nonvocab_tensor_shapes_match'] and result['plain_dtypes'] and manifest['access']=='granted' and not manifest['is_adapter'] and sum(manifest.get('view_counts',{}).values())>0 and not result['quantization_config'])
  result['status']='measured_header_only'
 except Exception as exc:
  result['status']='error';result['error_class']=type(exc).__name__
 result['elapsed_seconds']=time.monotonic()-case_start;results.append(result)
 record=dict(freeze=freeze,rows=results,reader_stats=reader.stats,elapsed_seconds=time.monotonic()-start,complete=len(results)==len(selected),scope='Metadata/config/headers only; no tensor payload or scientific pool freeze; API bodies excluded from reader bytes.')
 temp=out/'pilot-results.json.tmp';temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(out/'pilot-results.json')
 print(row['anchor'],row['role'],row['repo_id'],result['status'],'compatible',result.get('header_compatible'),flush=True)
print('pilot_complete',len(results),'cases',reader.stats,'seconds',time.monotonic()-start,flush=True)
