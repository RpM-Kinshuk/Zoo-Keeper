"""Prepare review-only scaling inputs/budget. Never launches a weight job."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'Zoo-Keeper'),str(ROOT/'Model-Zoo')]
from lab.artifacts import atomic_json
from scripts.scaling_pools import inventory_identity

source=ROOT/'runs/track-a-k8-headers-20261108/preflight.py'
spec=importlib.util.spec_from_file_location('retained_preflight',source)
preflight=importlib.util.module_from_spec(spec);spec.loader.exec_module(preflight)
pre=ROOT/'runs/scaling-pools-20261111-v2';out=ROOT/'runs/scaling-launch-20261111'
out.mkdir(exist_ok=True);(out/'manifests').mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if '--reuse-structural-budget' in sys.argv:
    previous=json.loads((out/'launch-plan.json').read_text())
    summary=previous['pools'];pins=previous['input_sha256']
else:
    models=[];plans=[];pins={}
    for anchor in ['q25-05','q3-06']:
     pool_path=pre/f'{anchor}-pool.csv';pins[str(pool_path.relative_to(ROOT))]=sha(pool_path)
     rows=list(csv.DictReader(pool_path.open()));assert len(rows)==2000
     for row in rows:
      path=ROOT/row['manifest'];assert sha(path)==row['manifest_sha256']
      manifest=json.loads(path.read_text());assert manifest['revision']==row['revision']
      assert inventory_identity(manifest)==row['inventory_identity']
      plan=preflight.estimate(manifest)
      plans.append(dict(anchor=anchor,repo_id=row['repo_id'],revision=row['revision'],**plan))
      models.append(dict(repo_id=row['repo_id'],revision=row['revision']))
      pins[str(path.relative_to(ROOT))]=row['manifest_sha256']
      target=out/'manifests'/path.name
      if target.exists():assert sha(target)==row['manifest_sha256']
      else:target.symlink_to(path.resolve())
    assert len(models)==4000 and len({(r['repo_id'],r['revision']) for r in models})==4000
    with (out/'models.csv').open('w') as handle:
     writer=csv.DictWriter(handle,fieldnames=['repo_id','revision']);writer.writeheader();writer.writerows(models)
    atomic_json(out/'per-model-plans.json',dict(models=plans))
    summary={}
    for anchor in ['q25-05','q3-06']:
     rows=[r for r in plans if r['anchor']==anchor]
     summary[anchor]=dict(models=len(rows),operational_pass=sum(r['status']=='PASS' for r in rows),
      requested_payload_bytes=sum(r['planned_requested_payload_bytes'] for r in rows),
      planned_view_reads=sum(r['nviews'] for r in rows),
      expected_fp_storage_bytes_ESTIMATE=sum(r['expected_fp_storage_bytes'] for r in rows),
      reserve_fp_storage_bytes_ESTIMATE=sum(r['reserve_fp_storage_bytes'] for r in rows),
      largest_decoded_f64_bytes=max(r['largest_decoded_f64_source_or_block_bytes'] for r in rows))
queries=ROOT/'runs/track-b-pivot-unblock-a-20261105/active-roster-v2/models.csv'
roster=list(csv.DictReader(queries.open()))
query_counts={a:sum(r['kind']=='derived' and r['model_id'].startswith(a+'-') for r in roster) for a in summary}
pins[str(queries.relative_to(ROOT))]=sha(queries)
files=[Path(__file__),source,pre/'summary.json',out/'models.csv',out/'per-model-plans.json',ROOT/'Zoo-Keeper/docs/steering/SCALING-POOL-PROTOCOL.md']
files+=list((ROOT/'Model-Zoo/evidence').glob('*.py'))
files+=list((ROOT/'Model-Zoo/net_esd').glob('*.py'))
files += [ROOT/'Model-Zoo/esd_experiment/src/measurement_config.py',ROOT/'Model-Zoo/requirements.txt']
pins.update({str(p.relative_to(ROOT)):sha(p) for p in files})
policy=dict(measurement_mode='row_blocks',sketch_k=64,compute_device='cpu',workers=4,numerical_threads_per_worker=1,with_parent=False,parent_cache=False,timeout_seconds=28800)
probe_summary=ROOT/'runs/scaling-cost-probe-20261111/row_blocks/summary.json'
probe_freeze=ROOT/'runs/scaling-cost-probe-20261111/freeze.json'
forecast={}
if probe_summary.exists():
 result=json.loads(probe_summary.read_text());freeze=json.loads(probe_freeze.read_text())
 for row in freeze['selected']:
  actual=next((r for r in result.get('models',[]) if r['repo_id']==row['repo_id'] and r['revision']==row['revision'] and r['status']=='complete'),None)
  if actual:
   assert actual['bytes_read']==row['plan']['planned_requested_payload_bytes']
   actual_bytes=actual['bytes_read'];seconds=actual['elapsed_seconds'];totals=summary[row['anchor']]
   forecast[row['anchor']]=dict(probe_bytes=actual_bytes,probe_seconds=seconds,
     byte_scaled_serial_hours_ESTIMATE=totals['requested_payload_bytes']/actual_bytes*seconds/3600,
     view_scaled_serial_hours_ESTIMATE=totals['planned_view_reads']/row['plan']['nviews']*seconds/3600,
     limits='One F32 example; alternative byte/view extrapolations, no BF16/parallel/tail-latency validation or guaranteed budget.')
 pins[str(probe_summary.relative_to(ROOT))]=sha(probe_summary);pins[str(probe_freeze.relative_to(ROOT))]=sha(probe_freeze)
plan=dict(status='DRAFT_SCIENTIFIC_SCORING_NOT_READY',ready_for_owner_launch=False,weight_jobs_started=False,
 pools=summary,policy=policy,formal_derived_query_counts=query_counts,unique_formal_query_ceiling=sum(query_counts.values()),
 existing_query_target=200,query_shortfall_at_least=200-sum(query_counts.values()),
 scientific_blockers=['Exact query/truth/exposure/scoring/source-removed contract not frozen.',
  'At most60 constructed queries from two exposed roots;≥200target unmet.',
  'Related-model pools do not establish false-family rates without documented independent same-shape controls.',
  'Extra fingerprints do not repair an uninformative relative/source decision rule.'],
 free_bytes_observed=shutil.disk_usage(ROOT).free,input_sha256=pins,forecast=forecast,
 requested_payload_bytes=sum(r['requested_payload_bytes'] for r in summary.values()),
 expected_fp_storage_bytes_ESTIMATE=sum(r['expected_fp_storage_bytes_ESTIMATE'] for r in summary.values()),
 reserve_fp_storage_bytes_ESTIMATE=sum(r['reserve_fp_storage_bytes_ESTIMATE'] for r in summary.values()),
 limits='Header compatibility is not payload access. Resharded/transformed duplicates may remain; inventories are not independent roots. Operational caps do not bound full SVD process RAM. CPU chosen; no GPUs reserved.8howner timeout is a resumable cap, not predicted completion.')
atomic_json(out/'launch-plan.json',plan)
command='''# REVIEW-ONLY DRAFT; scientific readiness is not established. OWNER launches large jobs.
export ZOO_DATA=/jumbo/yaoqingyang/kinshuk/zoo
export PYTHONPATH="$ZOO_DATA/Model-Zoo:$ZOO_DATA/Model-Research:$ZOO_DATA/Model-Zoo/esd_experiment/src${PYTHONPATH:+:$PYTHONPATH}"
# Use an already inherited approved HF_TOKEN; never source/print credentials.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
conda run -n esd_ind timeout --signal=TERM 28800s python -m evidence.cli batch \\
  --models "$ZOO_DATA/runs/scaling-launch-20261111/models.csv" \\
  --manifests "$ZOO_DATA/runs/scaling-launch-20261111/manifests" \\
  --out "$ZOO_DATA/runs/scaling-launch-20261111/row_blocks" \\
  --measurement_mode row_blocks --sketch_k 64 --compute_device cpu --workers 4 \\
  --no-all_experts --no-with_parent --no-parent_cache --also_row_block_sizes none
# Resume the exact same command/inputs; existing completed artifacts are skipped.
'''
(out/'COMMAND.txt').write_text(command)
print(json.dumps({k:v for k,v in plan.items() if k not in ['input_sha256']},indent=2))
