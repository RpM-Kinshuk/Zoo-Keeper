"""Audit restored recipe truth and saved scalars without reading any weights."""
import hashlib,json,math
from pathlib import Path
root=Path('/jumbo/yaoqingyang/kinshuk/zoo'); out=root/'runs/heretic-companion-recovery-20261110'
inventory_path=root/'runs/track-a-k0-20261107/inventory.json'
inv={r['path']:r for r in json.loads(inventory_path.read_text())['files']}
paths=[inventory_path]; recipes={}; source_checks=[]
active_path=root/'runs/track-b-k5-active-writers-20261108/results.json'
active=json.loads(active_path.read_text()); paths.append(active_path)
for anchor,run in [('olmo-1b','olmo-1b-attempt2'),('q25-05','q25-05'),('q3-06','q3-06'),('smol-360','smol-360-attempt2')]:
 mid=anchor+'-abliteration-heretic'; directory=root/'lab-restored-archive/models'/mid
 companions=[r['path'] for r in json.loads((out/'verification.json').read_text())['files'] if '/'+mid+'/' in r['path']]
 assert len(companions)==6
 for name in sorted(companions):
  p=root/name
  assert p.name!='model.safetensors'
  key='models/'+str(p.relative_to(directory.parent)); data=p.read_bytes()
  assert len(data)==inv[key]['size'] and hashlib.sha256(data).hexdigest()==inv[key]['sha256']
  paths.append(p)
 record=json.loads((directory/'abliteration.json').read_text())
 recipe=json.loads((directory/'recipe.json').read_text())
 assert record['config']['row_normalization']=='full'
 rank=record['config']['full_normalization_lora_rank']; assert rank==3
 selected=[r for r in active['rows'] if r['model_id']==mid]; assert len(selected)==1
 layers=[b['layer'] for b in selected[0]['blocks']]
 assert all(record['per_layer_settings'][str(layer)]['mlp.down_proj']['weight']!=0 for layer in layers)
 recipes[anchor]=dict(model_id=mid,configured_rank=rank,row_normalization='full',active_layers=layers,
  parent_identities=recipe['parent_identities'],selected_objective_metrics=record['selected_objective_metrics'],
  validation=record['validation'],behavior_interpretation='No paired heldout baseline; no refusal-removal success inferred.')
 frozen=root/'runs/track-b-remaining-20261105/evidence-snapshot/runs/heretic'/run/'frozen.json'
 paths.append(frozen)
 source_checks.append(dict(anchor=anchor,frozen_manifest=str(frozen.relative_to(root)),historical_source_pins=json.loads(frozen.read_text()).get('installed_source_sha256')))
rows=[]
for case in active['rows']:
 rank=3 # frozen configured rank in all four positives; same diagnostic on controls
 for block in case['blocks']:
  norm=block['delta_frobenius_norm']; s=block['top_singular_values']
  finite=len(s)>=rank and all(v is not None and math.isfinite(v) for v in s[:rank])
  tail=norm*norm-sum(v*v for v in s[:rank]) if finite else None
  residual=math.sqrt(tail) if tail is not None and tail>=0 else None
  rows.append(dict(model_id=case['model_id'],anchor=case['anchor'],positive=case['positive'],tensor=block['tensor'],
   relative_update=block['rel_update_norm'],above_existing_update_floor=block['rel_update_norm']>=.003,
   old_status=block['status'],old_confidence=block['rank_confidence'],old_rank=block['rank'],
   rank3_tail_approx=residual,rank3_tail_fraction_approx=residual/norm if residual is not None and norm else None,
   status='approximate_saved_scalar_diagnostic_only' if residual is not None else 'unavailable_saved_singular_values',
   new_compatibility_flag=None,reason='No validated Heretic product/cast/add/export allowance; FP32 singular values are not certified bounds.'))
assert len(rows)==132 and sum(r['positive'] for r in rows)==12
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths.append(Path(__file__))
summary=dict(restored_companions=24,recipe_cases=recipes,source_pin_records=source_checks,
 saved_positive_blocks=12,saved_control_blocks=120,old_resolved_positives=active['summary']['positive_resolved'],
 approximate_tail_available_positive=sum(r['positive'] and r['rank3_tail_approx'] is not None for r in rows),
 approximate_tail_available_controls=sum(not r['positive'] and r['rank3_tail_approx'] is not None for r in rows),
 above_floor_positive=sum(r['positive'] and r['above_existing_update_floor'] for r in rows),
 new_method_flags=0,weight_models_read=0,exposure='Existing exposed development panel; no new detector or final evaluation.',
 inputs_sha256={str(p.relative_to(root)):h(p) for p in paths})
(out/'saved-rank3-diagnostics.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['inputs_sha256','source_pin_records','recipe_cases']},indent=2))
for anchor in recipes:
 a=[r for r in rows if r['positive'] and r['anchor']==anchor]
 print(anchor,'approximate rank3 tail fractions',[r['rank3_tail_fraction_approx'] for r in a])
