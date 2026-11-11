"""One offline discovery scan; no same-architecture assertion or final pool freeze."""
import csv,hashlib,json,os,re,sys,time
from pathlib import Path
import duckdb
import pandas as pd
root=Path('/jumbo/yaoqingyang/kinshuk/zoo')
sys.path[:0]=[str(root/'Model-Zoo')]
from evidence.frame import build_frame, _items, _record
from zoo.schema import _CONFIG_FIELDS
out=root/'runs/scaling-pool-discovery-20261110'
source=root/'cache/hub-stats/f34bdc1fd7b92f98eff5ef5650ff7a2111bc46ba/models.parquet'
anchors={
 'q25-05':('Qwen/Qwen2.5-0.5B',r'qwen2[._-]?5[-_]0[._-]?5b'),
 'q3-06':('Qwen/Qwen3-0.6B',r'qwen3[-_]0[._-]?6b'),
 'smol-360':('HuggingFaceTB/SmolLM2-360M',r'smollm2[-_]360m'),
 'olmo-1b':('allenai/OLMo-2-0425-1B',r'olmo[-_]2[-_]0425[-_]1b')}
start=time.monotonic()
con=duckdb.connect()
con.execute("SET threads=2")
con.execute("SET memory_limit='2GB'")
# SQL values are bound, including the path and every untrusted identifier/pattern.
conditions=[];params=[str(source)]
for repo,pattern in anchors.values():
 conditions.append('(regexp_matches(lower(id), ?) OR list_contains(list_transform(baseModels.models, b -> lower(b.id)), ?))')
 params.extend([pattern,repo.lower()])
query='SELECT id,pipeline_tag,library_name,tags,downloads,safetensors,gguf,siblings,baseModels,createdAt,cardData,config FROM read_parquet(?) WHERE '+ ' OR '.join(conditions)
if '--reuse-saved-records' in sys.argv:
 rows=json.loads((out/'source-candidates.json').read_text())
else:
 rows=con.execute(query,params).fetchall(); columns=[c[0] for c in con.description]
 rows=[dict(zip(columns,r)) for r in rows]
if '--reuse-saved-records' not in sys.argv:
 (out/'source-candidates.json').write_text(json.dumps(rows,default=str)+'\n')
frame=build_frame(pd.DataFrame(rows)); by_id={r['repo_id']:r for r in frame.to_dict('records')}
selected=[];counts={}
for anchor,(repo,pattern) in anchors.items():
 local=[]
 for raw in rows:
  bases=raw['baseModels'] or {}; bases=bases.get('models') or []
  named=bool(re.search(pattern,raw['id'].lower()))
  declared=repo.lower() in [b['id'].lower() for b in bases if b.get('id')]
  if not (named or declared): continue
  f=by_id.get(raw['id']); config=raw.get('config') or '{}'
  if isinstance(config,str): config=json.loads(config)
  expected=json.loads((root/'lab-restored-archive/models'/f'{anchor}-abliteration-heretic/config.json').read_text())
  present={k:config[k] for k in _CONFIG_FIELDS if k in config}
  fields=list(_CONFIG_FIELDS)
  complete=all(k in config for k in fields if k in expected)
  conflicts=[k for k in fields if k in config and config[k]!=expected.get(k)]
  tags=[str(t) for t in _items(raw.get('tags'))]
  base_info=_record(raw.get('baseModels'))
  adapter=base_info.get('relation')=='adapter' or raw.get('library_name')=='peft' or any(t in ('peft','lora','kohya') or t.startswith('base_model:adapter:') for t in tags)
  compressed=any(t.lower() in ('gptq','awq','bitsandbytes','mlx') or t.lower().startswith(('quantization:gptq','quantization:awq','quantization:bitsandbytes')) for t in tags) or raw.get('library_name')=='mlx'
  local.append(dict(anchor=anchor,repo_id=raw['id'],name_match=named,direct_declared_base=declared,
    weight_format=f['weight_format'] if f else 'no_weights',is_adapter=adapter,compressed_format_hint=compressed,
    params=f['params'] if f else None,downloads=raw['downloads'] or 0,
    config_signature_fields_complete=complete,config_conflicts=json.dumps(conflicts),
    metadata_model_type=config.get('model_type'),metadata_config_fields=json.dumps(present,sort_keys=True),
    revision='',shape_status='unverified'))
 unique={r['repo_id']:r for r in local}; assert len(unique)==len(local)
 selected.extend(local)
 full=[r for r in local if r['weight_format']=='safetensors' and not r['is_adapter']]
 counts[anchor]=dict(anchor_repo=repo,discovered=len(local),direct_declared=sum(r['direct_declared_base'] for r in local),name_match=sum(r['name_match'] for r in local),
   safetensors_nonadapter_discovery=len(full),compressed_format_hints=sum(r['compressed_format_hint'] for r in full),adapters=sum(r['is_adapter'] for r in local),
   formats={k:sum(r['weight_format']==k for r in local) for k in sorted({r['weight_format'] for r in local})},
   complete_config=sum(r['config_signature_fields_complete'] for r in full),config_conflicts=sum(r['config_conflicts']!='[]' for r in full),
   verified_same_shape=0,verified_readable=0)
selected.sort(key=lambda r:(r['anchor'],r['repo_id']))
with (out/'candidates.csv').open('w') as handle:
 writer=csv.DictWriter(handle,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
provisional=sorted(counts,key=lambda k:(-counts[k]['safetensors_nonadapter_discovery'],k))[:2]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as handle:
  for block in iter(lambda:handle.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
summary=dict(snapshot_revision=source.parent.name,snapshot_rows=con.execute('SELECT count(*) FROM read_parquet(?)',[str(source)]).fetchone()[0],
 snapshot_sha256=(json.loads((out/'summary-initial.json').read_text())['snapshot_sha256'] if '--reuse-saved-records' in sys.argv else digest(source)),snapshot_hash_reused_from_initial='--reuse-saved-records' in sys.argv,source_candidates_sha256=digest(out/'source-candidates.json'),anchors=counts,provisional_discovery_leaders=provisional,
 exact_shape_pool_choice_frozen=False,weight_models_read=0,network_requests=0,
 source_hashes={str(p.relative_to(root)):digest(p) for p in [Path(__file__),out/'PROTOCOL.md',root/'Model-Zoo/evidence/frame.py',root/'Model-Zoo/zoo/schema.py']},
 elapsed_seconds=time.monotonic()-start,
 limits='Snapshot name/structured-baseModels-direct-declaration discovery only; base_model tag-only/name-free candidates omitted. Renamed indirect/undeclared and post-snapshot candidates omitted. Config metadata and file formats do not certify safe readability, shapes, independence or source truth.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
