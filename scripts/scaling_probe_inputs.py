"""Freeze two deterministic cost-probe inputs; this command never reads weights."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Zoo-Keeper'),str(ROOT/'Model-Zoo')]
from scripts.scaling_pools import inventory_identity,compatibility_reason
from lab.artifacts import atomic_json

out=ROOT/'runs/scaling-cost-probe-20261111'
assert not (out/'freeze.json').exists(),'do not overwrite frozen probe'
pre=ROOT/'runs/scaling-pools-20261111-v2'
discovery=ROOT/'runs/scaling-pool-discovery-20261110'
source=ROOT/'runs/track-a-k8-headers-20261108/preflight.py'
spec=importlib.util.spec_from_file_location('retained_budget_estimator',source)
preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(preflight)
rows=list(csv.DictReader((discovery/'candidates.csv').open()))
pilot=json.loads((discovery/'pilot-results.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selected=[];pins={}
for anchor,base_repo in [('q25-05','Qwen/Qwen2.5-0.5B'),('q3-06','Qwen/Qwen3-0.6B')]:
 base_record=next(r for r in pilot['rows'] if r['repo_id']==base_repo)
 base=json.loads((ROOT/base_record['manifest']).read_text());seen={inventory_identity(base)}
 candidates=[r for r in rows if r['anchor']==anchor and r['weight_format']=='safetensors' and r['is_adapter']=='False' and r['config_conflicts']=='[]' and r['repo_id']!=base_repo]
 candidates.sort(key=lambda r:(hashlib.sha256(f"20261028:{anchor}:{r['repo_id']}".encode()).hexdigest(),r['repo_id']))
 for candidate in candidates:
  receipt=pre/anchor/(hashlib.sha256(candidate['repo_id'].encode()).hexdigest()+'.json')
  assert receipt.exists(),'earlier deterministic candidate not yet terminal'
  record=json.loads(receipt.read_text());pins[str(receipt.relative_to(ROOT))]=sha(receipt)
  if record['status']!='header_compatible':continue
  manifest_path=ROOT/record['manifest'];assert sha(manifest_path)==record['manifest_sha256']
  manifest=json.loads(manifest_path.read_text());assert compatibility_reason(manifest,base) is None
  identity=inventory_identity(manifest)
  if identity in seen:continue
  plan=preflight.estimate(manifest);assert plan['status']=='PASS',plan['reasons']
  selected.append(dict(anchor=anchor,repo_id=manifest['repo_id'],revision=manifest['revision'],manifest=record['manifest'],manifest_sha256=record['manifest_sha256'],plan=plan))
  break
 else:raise ValueError('no deterministic eligible nonanchor yet')
assert len(selected)==2 and len({r['repo_id'] for r in selected})==2
(out/'manifests').mkdir(exist_ok=True)
for row in selected:shutil.copy2(ROOT/row['manifest'],out/'manifests'/Path(row['manifest']).name)
with (out/'models.csv').open('w') as handle:
 writer=csv.DictWriter(handle,fieldnames=['repo_id','revision']);writer.writeheader();writer.writerows({k:r[k] for k in ['repo_id','revision']} for r in selected)
files=[Path(__file__),out/'PROTOCOL.md',out/'models.csv',source,pre/'freeze.json',discovery/'candidates.csv']
files+=list((ROOT/'Model-Zoo/evidence').glob('*.py'))
files += [ROOT/'Model-Zoo/esd_experiment/src/measurement_config.py']
freeze=dict(selected=selected,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},selection_outcome_sha256=pins,
 policy=dict(weight_models=2,device='cpu',workers=1,numerical_threads=1,sketch_k=64,row_block=256,with_parent=False,parent_cache=False,seconds=600),concurrent_job='scaling-pools-20261111-v2 metadata only')
atomic_json(out/'freeze.json',freeze)
print(json.dumps([dict(anchor=r['anchor'],repo_id=r['repo_id'],revision=r['revision'],planned_bytes=r['plan']['planned_requested_payload_bytes'],views=r['plan']['nviews']) for r in selected],indent=2))
