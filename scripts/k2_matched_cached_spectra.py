"""Existing native spectral methods on B's exact fixed candidate pools."""
import hashlib,json,time
from pathlib import Path
from studies.baselines.paths import workspace_root
import numpy as np
from safetensors import safe_open
from lab.artifacts import atomic_json
from studies.baselines.lab_spectral import _profiles,similarity,METHODS
from spectral_research.study import reassessment_lineage
root=workspace_root(Path(__file__).resolve().parents[1]);repo=Path(__file__).resolve().parents[1]
out=root/'runs/track-c-k2-matched-spectra-20261108';out.mkdir(exist_ok=True);assert not (out/'results.json').exists()
prior=root/'runs/track-b-k5-parent-out-20261108';p=json.loads((prior/'protocol.json').read_text());expected=json.loads((prior/'results.json').read_text())['input_sha256']
catalog_path=root/'runs/track-a-k8-protocol-20261108/catalog/queries.json';catalog={r['model_id']:r for r in json.loads(catalog_path.read_text())}
basepath={'olmo-1b':'inputs/pinned-base-olmo-1b-a1847dff3500','q25-05':'inputs/cohort-v1-q25-05-060db6499f32','q3-06':'inputs/cohort-v1-q3-06-c1899de289a0','smol-360':'inputs/cohort-v1-smol-360-f8027fd0eaee'}
hashes={};cost={'hash_scan_bytes':0,'logical_eigenvalue_bytes':0};started=time.monotonic()
def bind(path):
 key=str(path.relative_to(root))
 if key not in hashes:
  h=hashlib.sha256()
  with path.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b);cost['hash_scan_bytes']+=len(b)
  hashes[key]=h.hexdigest()
 return hashes[key]
for path in [catalog_path,prior/'protocol.json',prior/'results.json',Path(__file__),repo/'studies/baselines/lab_spectral.py',Path(reassessment_lineage.__file__),repo/'docs/research/2026-11-08-k2-matched-spectra-protocol.md']:bind(path)
atomic_json(out/'protocol.json',{'text':(repo/'docs/research/2026-11-08-k2-matched-spectra-protocol.md').read_text(),'queries':p['queries'],'pools':p['pools'],'input_sha256':hashes.copy()})
names=sorted(set(p['queries']).union(*(p['pools'][q] for q in p['queries'])));assert len(names)==46
profiles={method:{} for method in METHODS};audit=[]
for mid in names:
 row={'model_id':mid};payload={}
 try:
  item=catalog[mid];mp=root/item['manifest'];bind(mp);manifest=json.loads(mp.read_text())
  assert item['model_key']==manifest['repo_id']+'@'+manifest['revision']
  for f in manifest['files']:
   physical='lab/'+basepath.get(mid,mid)+'/'+f['filename'];assert f['sha256']==expected[physical], 'weight identity mismatch'
  fp=root/item['full_weight']['path'];spectra=root/item['full_weight_spectra']
  assert str(spectra)==str(fp).replace('.fingerprint.safetensors','.spectra.h5')
  with safe_open(str(fp),framework='np') as handle:header=handle.metadata()
  assert header['repo_id']+'@'+header['revision']==item['model_key']
  bind(spectra);values,logical,skipped,hill_error=_profiles(spectra,payload)
  cost['logical_eigenvalue_bytes']+=logical
  for method,value in zip(METHODS,values):
   if value is not None:profiles[method][mid]=value
  row.update(status='read',spectra=item['full_weight_spectra'],logical_eigenvalue_bytes=logical,skipped=skipped,hill_error=hill_error)
 except (KeyError,ValueError,AssertionError,OSError,TypeError) as exc:
  row.update(status='not_run',reason=str(exc));cost['logical_eigenvalue_bytes']+=payload.get('logical_eigenvalue_bytes',0)
 audit.append(row)
rows=[];rankings=[]
for query in p['queries']:
 recipe=root/'lab'/query/'recipe.json';bind(recipe);r=json.loads(recipe.read_text());r=r.get('recipe',r);parent=r['parents'][0]
 for method in METHODS:
  scores={}
  for candidate in p['pools'][query]:
   row={'query':query,'candidate':candidate,'method':method,'true_parent':candidate==parent}
   try:
    s=similarity(profiles[method][query],profiles[method][candidate],method);assert np.isfinite(s);scores[candidate]=s;row.update(status='tested',score=s)
   except (KeyError,ValueError,AssertionError) as exc:row.update(status='not_run',reason=str(exc))
   rows.append(row)
  complete=len(scores)==len(p['pools'][query]);top=[n for n in scores if scores[n]==max(scores.values())] if scores else []
  absent={n:s for n,s in scores.items() if n!=parent}
  rankings.append({'query':query,'method':method,'complete_pool':complete,'top_candidates':top,'unique_correct_top':complete and top==[parent],'parent_rank':1+sum(s>scores[parent] for s in scores.values()) if parent in scores else None,'source_absent_top':[n for n in absent if absent[n]==max(absent.values())] if absent else [],'source_absent_pool_complete':len(absent)==len(p['pools'][query])-1,'parent_decision':'unsupported'})
summary={method:{'planned_pairs':126,'tested_pairs':sum(r['status']=='tested' for r in rows if r['method']==method),'complete_pools':sum(r['complete_pool'] for r in rankings if r['method']==method),'unique_correct_top':sum(r['unique_correct_top'] for r in rankings if r['method']==method)} for method in METHODS}
atomic_json(out/'results.json',{'summary':summary,'rows':rows,'rankings':rankings,'profile_audit':audit,'input_sha256':hashes,'cost':cost,'elapsed_seconds':time.monotonic()-started,'new_weight_bytes':0,'historical_extraction_cost':'not_measured'})
print(json.dumps(summary));print('profile failures',[r for r in audit if r['status']!='read'])
