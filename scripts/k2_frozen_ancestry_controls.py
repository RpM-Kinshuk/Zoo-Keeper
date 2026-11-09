"""Missing ancestry baselines on the unchanged inherited L12 control universes."""
from collections import Counter
from contextlib import ExitStack
import hashlib,json,time
from pathlib import Path
from studies.baselines.paths import workspace_root
import numpy as np
import pandas as pd
from safetensors import safe_open
from sklearn.metrics import roc_auc_score
from importlib.metadata import distribution
import modeldna
from modeldna.io.source import LocalSource
from modeldna.fingerprint.extract import extract_fingerprint
from modeldna.compare import compare_fingerprints
from modeldna.calibration import DEFAULT_CALIBRATOR
from modeldna.verdict import judge,Thresholds
from lab.artifacts import atomic_json
from studies.baselines.residual import signature,score

root=workspace_root(Path(__file__).resolve().parents[1]);repo=Path(__file__).resolve().parents[1]
out=root/'runs/track-c-k2-ancestry-controls-20261109';out.mkdir(exist_ok=True);assert not (out/'results.json').exists()
assert modeldna.__version__=='0.1.0'
start=time.monotonic();cost={'hash_scan_bytes':0,'residual_logical_weight_bytes':0,'modeldna_source_bytes':0};hashes={}
def bind(path):
 key=str(path.relative_to(root))
 if key not in hashes:
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):h.update(block);cost['hash_scan_bytes']+=len(block)
  hashes[key]=h.hexdigest()
 return hashes[key]
control=root/'runs/track-b-l12-evaluation-v2/controls/t2-k64';universes={}
for split in ['development','test']:
 path=control/split/'universe.parquet';bind(path);universes[split]=pd.read_parquet(path)
 assert len(universes[split])==30
catalog_path=root/'runs/track-a-k8-protocol-20261108/catalog/queries.json';bind(catalog_path);catalog={r['model_key']:r for r in json.loads(catalog_path.read_text())}
keys=sorted(set().union(*(set(u.query_key)|set(u.candidate_key) for u in universes.values())));assert len(keys)==12
paths={};manifests={}
for key in keys:
 if key.startswith('lab/'):
  mid=key.split('@')[0][4:];paths[key]=root/'lab'/mid;recipe=paths[key]/'recipe.json';bind(recipe);r=json.loads(recipe.read_text());r=r.get('recipe',r)
  for parent,pin in r['inputs'].items():
   if pin['model_key'] in keys:
    pp=root/'lab'/pin['path'];paths[pin['model_key']]=pp
    for name,h in pin['weight_sha256'].items():assert bind(pp/name)==h
 for field in ['manifest']:
  path=root/catalog[key][field];bind(path);m=json.loads(path.read_text());assert m['repo_id']+'@'+m['revision']==key;manifests[key]=m
assert set(paths)==set(keys)
for key,path in paths.items():
 bind(path/'config.json')
 if (path/'lineage.json').exists():bind(path/'lineage.json')
 for f in manifests[key]['files']:assert bind(path/f['filename'])==f['sha256']
for path in [Path(__file__),repo/'studies/baselines/residual.py',repo/'docs/research/2026-11-09-k2-ancestry-controls-protocol.md']:bind(path)
d=distribution('modeldna');package_hashes={str(p):hashlib.sha256(d.locate_file(p).read_bytes()).hexdigest() for p in d.files if str(p).startswith('modeldna/') and str(p).endswith('.py')}
atomic_json(out/'protocol.json',{'text':(repo/'docs/research/2026-11-09-k2-ancestry-controls-protocol.md').read_text(),'models':keys,'input_sha256':hashes.copy(),'native_thresholds':vars(Thresholds()),'native_calibrator':vars(DEFAULT_CALIBRATOR),'package_source_sha256':package_hashes})
fps={};profiles={};extractions=[]
for key in keys:
 path=paths[key];cfg=json.loads((path/'config.json').read_text());assert cfg['model_type']=='gpt_neox'
 record={'key':key,'layers_expected':cfg['num_hidden_layers'],'width':cfg['hidden_size']};t=time.monotonic()
 try:
  source=LocalSource(path);fp=extract_fingerprint(source,mode='fast',model_id=key,revision=key.rsplit('@',1)[1]);fps[key]=fp
  cost['modeldna_source_bytes']+=source.bytes_read;record.update(modeldna_status='tested',modeldna_source_bytes=source.bytes_read)
  atomic_json(out/(hashlib.sha256(key.encode()).hexdigest()[:16]+'.json'),fp.to_dict())
 except Exception as exc:record.update(modeldna_status='not_run',modeldna_reason=type(exc).__name__)
 try:
  vectors=[];concentrations=[]
  with ExitStack() as stack:
   loc={}
   for shard in sorted(path.glob('*.safetensors')):
    handle=stack.enter_context(safe_open(str(shard),framework='pt',device='cpu'))
    for name in handle.keys():assert name not in loc;loc[name]=handle
   for layer in range(cfg['num_hidden_layers']):
    names=[f'gpt_neox.layers.{layer}.mlp.{role}.weight' for role in ['dense_4h_to_h','dense_h_to_4h']]
    down,up=[loc[name].get_tensor(name) for name in names];assert down.shape[0]==cfg['hidden_size'] and up.shape[1]==cfg['hidden_size']
    cost['residual_logical_weight_bytes']+=sum(x.numel()*x.element_size() for x in (down,up))
    vector,concentration=signature(down.double().numpy(),up.double().numpy());vectors.append(vector);concentrations.append(concentration)
   record['excluded_tensors']=[{'tensor':name,'reason':'outside_complete_canonical_MLP_path'} for name in sorted(loc) if not any(name==f'gpt_neox.layers.{layer}.mlp.{role}.weight' for layer in range(cfg['num_hidden_layers']) for role in ['dense_4h_to_h','dense_h_to_4h'])]
  profiles[key]=(np.stack(vectors),np.array(concentrations));record.update(residual_status='tested',layers_measured=len(vectors))
 except Exception as exc:record.update(residual_status='not_run',residual_reason=type(exc).__name__)
 record['seconds']=time.monotonic()-t;extractions.append(record);atomic_json(out/'progress.json',{'extractions':extractions,'cost':cost});print(key,record['modeldna_status'],record['residual_status'],flush=True)
 assert time.monotonic()-start<6600
rows=[];th=Thresholds()
for split,universe in universes.items():
 for position,original in enumerate(universe.to_dict('records')):
  for method in ['modelDNA-native','Training-Leaves-Traces-canonical-MLP']:
   row=dict(original);row.update(split=split,original_position=position,method=method)
   try:
    assert original['eligible'],'Frozen ineligible pair'
    a,b=original['query_key'],original['candidate_key']
    if method=='modelDNA-native':
     ev=compare_fingerprints(fps[a],fps[b]);assert ev.shape_compatible and ev.layers_match,'incompatible native shape'
     probability=DEFAULT_CALIBRATOR.predict(ev.features());verdict=judge([(ev,'pythia14m')],suspect_dtype=fps[a].arch.torch_dtype,candidate_dtypes={b:fps[b].arch.torch_dtype})
     row.update(score=probability,evidence=ev.to_dict(),native_verdict=verdict.to_dict(),native_band='positive' if probability>=th.positive else 'abstention' if probability>=th.abstain else 'below_abstention')
    else:
     value,assignment=score(profiles[a],profiles[b]);row.update(score=value,aligned_blocks=len(assignment),identity_alignment=int(np.sum(assignment==np.arange(len(assignment)))))
    assert np.isfinite(row['score']);row['status']='tested'
   except Exception as exc:row.update(status='not_run',reason=type(exc).__name__)
   rows.append(row)
metrics=[]
for split,universe in universes.items():
 for method in ['modelDNA-native','Training-Leaves-Traces-canonical-MLP']:
  selected=[r for r in rows if r['split']==split and r['method']==method];valid=[r for r in selected if r['status']=='tested'];positive=[r['score'] for r in valid if r['truth_same_root']];negative=[r['score'] for r in valid if not r['truth_same_root']]
  metric={'split':split,'method':method,'planned_pairs':30,'tested_pairs':len(valid),'same_root_pairs':len(positive),'different_root_pairs':len(negative),'same_root_auroc':float(roc_auc_score([r['truth_same_root'] for r in valid],[r['score'] for r in valid])) if len(valid)==30 and positive and negative else None,'positive_range':[min(positive),max(positive)] if positive else None,'negative_range':[min(negative),max(negative)] if negative else None,'immediate_parent':'unsupported','direction':'unsupported'}
  if method=='modelDNA-native':
   metric['native_classes_by_truth']={str(label):dict(Counter(r['native_verdict']['verdict'] for r in valid if r['truth_same_root']==label)) for label in [True,False]}
   metric['native_bands_by_truth']={str(label):dict(Counter(r['native_band'] for r in valid if r['truth_same_root']==label)) for label in [True,False]}
  metrics.append(metric)
summary={'models':len(keys),'native_fingerprints':len(fps),'residual_profiles':len(profiles),'scored_pairs_per_method':{m:sum(r['status']=='tested' for r in rows if r['method']==m) for m in ['modelDNA-native','Training-Leaves-Traces-canonical-MLP']},'all_rows_accounted':len(rows)==120}
atomic_json(out/'results.json',{'summary':summary,'metrics':metrics,'rows':rows,'extractions':extractions,'input_sha256':hashes,'package_source_sha256':package_hashes,'native_thresholds':vars(th),'native_calibrator':vars(DEFAULT_CALIBRATOR),'cost':cost,'elapsed_seconds':time.monotonic()-start})
print(json.dumps(summary),flush=True);print(json.dumps(metrics),flush=True)
