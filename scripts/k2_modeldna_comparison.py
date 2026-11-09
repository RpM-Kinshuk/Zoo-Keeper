"""Native published-package run; see predeclared modelDNA protocol."""
from pathlib import Path
from studies.baselines.paths import workspace_root
import hashlib,json,time
from collections import Counter
from modeldna import __version__
from modeldna.io.source import open_source
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.compare import compare_fingerprints
from modeldna.verdict import judge,Thresholds
from lab.artifacts import atomic_json
assert __version__=='0.1.0'
root=workspace_root(Path(__file__).resolve().parents[1]); repo=Path(__file__).resolve().parents[1]
out=root/'runs/track-c-k2-modeldna-20261108';out.mkdir(exist_ok=True)
assert not (out/'results.json').exists()
prior=root/'runs/track-b-k5-parent-out-20261108'
p=json.loads((prior/'protocol.json').read_text());expected=json.loads((prior/'results.json').read_text())['input_sha256']
basepath={'olmo-1b':'inputs/pinned-base-olmo-1b-a1847dff3500','q25-05':'inputs/cohort-v1-q25-05-060db6499f32','q3-06':'inputs/cohort-v1-q3-06-c1899de289a0','smol-360':'inputs/cohort-v1-smol-360-f8027fd0eaee'}
hashes={};cost={'hash_scan_bytes':0,'native_source_bytes':0};start=time.monotonic()
def bind(path,verify=False):
 key=str(path.relative_to(root));h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b);cost['hash_scan_bytes']+=len(b)
 hashes[key]=h.hexdigest()
 if verify:assert hashes[key]==expected[key],key
for path in [Path(__file__),repo/'docs/research/2026-11-08-k2-modeldna-protocol.md',prior/'protocol.json',prior/'results.json']:bind(path)
atomic_json(out/'protocol.json',{'text':(repo/'docs/research/2026-11-08-k2-modeldna-protocol.md').read_text(),'queries':p['queries'],'pools':p['pools'],'input_sha256':hashes.copy(),'thresholds':vars(Thresholds())})
names=sorted(set(p['queries']).union(*(p['pools'][q] for q in p['queries'])))
fps={};extractions=[]
for mid in names:
 path=root/'lab'/basepath.get(mid,mid)
 for shard in sorted(path.glob('*.safetensors')):bind(shard,True)
 for filename in ['config.json','tokenizer.json','tokenizer_config.json']:
  if (path/filename).exists():bind(path/filename)
 t=time.monotonic();source=open_source(str(path));fp=extract_fingerprint(source,mode='fast',model_id=mid,revision='local-pinned')
 cost['native_source_bytes']+=source.bytes_read
 atomic_json(out/(mid+'.json'),fp.to_dict());fps[mid]=fp
 extractions.append({'model_id':mid,'seconds':time.monotonic()-t,'bytes_read':source.bytes_read})
 print('extracted',mid,round(time.monotonic()-t,2),flush=True)
 assert time.monotonic()-start<6600
rows=[];verdicts=[];scoring=time.monotonic()
for q in p['queries']:
 recipe=root/'lab'/q/'recipe.json';bind(recipe,True);r=json.loads(recipe.read_text());r=r.get('recipe',r);parent=r['parents'][0];family=r['base_family'];evidences=[]
 for c in p['pools'][q]:
  ev=compare_fingerprints(fps[q],fps[c]);evidences.append((ev,family));rows.append({'query':q,'candidate':c,'true_parent':c==parent,'evidence':ev.to_dict()})
 for present in (True,False):
  pool=[x for x in evidences if present or x[0].candidate_id!=parent]
  v=judge(pool,suspect_dtype=fps[q].arch.torch_dtype,candidate_dtypes={ev.candidate_id:fps[ev.candidate_id].arch.torch_dtype for ev,_ in pool})
  verdicts.append({'query':q,'parent':parent,'parent_present':present,'verdict':v.to_dict()})
summary={'models':len(fps),'comparisons':len(rows),'verdicts':len(verdicts),'complete':len(fps)==46 and len(rows)==126 and len(verdicts)==24,
 'present_classes':dict(Counter(v['verdict']['verdict'] for v in verdicts if v['parent_present'])),
 'absent_classes':dict(Counter(v['verdict']['verdict'] for v in verdicts if not v['parent_present']))}
atomic_json(out/'results.json',{'summary':summary,'rows':rows,'verdicts':verdicts,'extractions':extractions,'input_sha256':hashes,'cost':cost,'elapsed_seconds':time.monotonic()-start,'scoring_seconds':time.monotonic()-scoring,'package_version':__version__})
print(json.dumps(summary),flush=True)
