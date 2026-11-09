"""Full-tensor cosine adaptation; see predeclared witness protocol."""
from collections import Counter
from contextlib import ExitStack
import hashlib,json,os,re,time
from pathlib import Path
from studies.baselines.paths import workspace_root
import numpy as np
import torch
from safetensors import safe_open
from lab.artifacts import atomic_json
from studies.baselines.witness import moments,anchor_scores,aggregate

root=workspace_root(Path(__file__).resolve().parents[1]);repo=Path(__file__).resolve().parents[1]
audit=root/'runs/track-c-k2-remaining-20261108/witness'
out=root/'runs/track-c-k2-witness-20261108-v3';out.mkdir(exist_ok=True)
assert not (out/'results.json').exists(), 'Do not overwrite a completed result'
triples=json.loads((audit/'proposed-triples.json').read_text());assert len(triples)==16
prior=root/'runs/track-b-k5-parent-out-20261108/results.json'
expected=json.loads(prior.read_text())['input_sha256']
basepath={'olmo-1b':'inputs/pinned-base-olmo-1b-a1847dff3500','q25-05':'inputs/cohort-v1-q25-05-060db6499f32','q3-06':'inputs/cohort-v1-q3-06-c1899de289a0','smol-360':'inputs/cohort-v1-smol-360-f8027fd0eaee'}
roles=['self_attn.'+r+'_proj' for r in ['q','k','v','o']]+['mlp.'+r+'_proj' for r in ['up','down','gate']]
started=time.monotonic();cost={'hash_scan_bytes':0,'logical_weight_bytes':0};hashes={};rows=[];inventories=[]
def bind(p,verify=False):
 key=str(p.relative_to(root))
 if key not in hashes:
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b);cost['hash_scan_bytes']+=len(b)
  hashes[key]=h.hexdigest()
 if verify:assert hashes[key]==expected[key],key
 return hashes[key]
def checkpoint(mid):return root/'lab'/basepath.get(mid,mid)
for p in [Path(__file__),repo/'studies/baselines/witness.py',repo/'docs/research/2026-11-08-k2-witness-protocol.md',audit/'proposed-triples.json',audit/'paper-v2.html',prior]:bind(p)
atomic_json(out/'protocol.json',{'text':(repo/'docs/research/2026-11-08-k2-witness-protocol.md').read_text(),'triples':triples,'input_sha256':hashes.copy(),'owned_pid':os.getpid(),'backend':'CPU float64,8threads'})
for family in basepath:
 cases=[t for t in triples if t['child'].startswith(family+'-')]
 names=sorted({t[k] for t in cases for k in ['parent','child','witness']});assert len(names)==5
 stats={t['child']:{} for t in cases};caststats={t['child']:{} for t in cases};errors={}
 with ExitStack() as stack:
  sources={};configs={}
  for mid in names:
   path=checkpoint(mid);bind(path/'config.json');configs[mid]=json.loads((path/'config.json').read_text());loc={}
   for shard in sorted(path.glob('*.safetensors')):
    bind(shard,True);handle=stack.enter_context(safe_open(str(shard),framework='pt',device='cpu'))
    for key in handle.keys():assert key not in loc;loc[key]=handle
   assert loc;sources[mid]=loc
  cfg=configs[names[0]];depth=cfg['num_hidden_layers'];width=cfg['hidden_size']
  assert all(c['num_hidden_layers']==depth and c['hidden_size']==width for c in configs.values())
  keys=sorted(f'model.layers.{layer}.{role}.weight' for layer in range(depth) for role in roles)
  for mid in names:
   selected={k for k in sources[mid] if re.fullmatch(r'model\.layers\.\d+\.(?:self_attn|mlp)\.(?:q|k|v|o|up|down|gate)_proj\.weight',k)}
   assert selected==set(keys), 'Unexpected or missing projection'
   inventories.append({'model_id':mid,'selected':len(keys),'excluded':[{'tensor':k,'reason':'outside_dense_projection_protocol'} for k in sorted(set(sources[mid])-set(keys))]})
  for t in cases:
   recipe=checkpoint(t['child'])/'recipe.json';assert bind(recipe)==t['edge_recipe_sha256']
   raw=json.loads(recipe.read_text());raw=raw.get('recipe',raw);assert raw['parents']==[t['parent']]
   assert bind(root/t['witness_recipe_path'])==t['witness_recipe_sha256']
   parent_pin=raw['inputs'].get('local_parent',raw['inputs']['model_pin'])
   for filename,h in parent_pin['weight_sha256'].items():assert bind(checkpoint(t['parent'])/filename,True)==h
  for key in keys:
   shapes=[sources[n][key].get_slice(key).get_shape() for n in names]
   dtypes=[sources[n][key].get_slice(key).get_dtype() for n in names]
   assert len(shapes[0])==2 and all(s==shapes[0] for s in shapes) and all(d in ['F32','BF16','F16','F64'] for d in dtypes)
   totals={t['child']:np.zeros(5) for t in cases};casttotals={t['child']:np.zeros(5) for t in cases}
   for start in range(0,shapes[0][0],128):
    blocks={};castblocks={}
    for mid in names:
     block=sources[mid][key].get_slice(key)[start:start+128,:]
     cost['logical_weight_bytes']+=block.numel()*block.element_size();blocks[mid]=block.double().numpy();castblocks[mid]=block.to(torch.bfloat16).double().numpy()
    for t in cases:
     try:
      totals[t['child']]+=moments(*(blocks[t[k]] for k in ['parent','child','witness']))
      casttotals[t['child']]+=moments(*(castblocks[t[k]] for k in ['parent','child','witness']))
     except ValueError as exc:errors[t['child']]=str(exc)
    assert time.monotonic()-started<6600
   for t in cases:
    stats[t['child']][key]=totals[t['child']]
    caststats[t['child']][key]=casttotals[t['child']]
  for mode,selected_stats in [('stored',stats),('bf16_view',caststats)]:
   for t in cases:
    role_scores={r:[] for r in roles};tensor_rows=[];distances=np.zeros(2)
    for key,value in selected_stats[t['child']].items():
     role=key.split('.',3)[3].removesuffix('.weight')
     try:
      sa,sb=anchor_scores(value);role_scores[role].append((sa,sb));distances+=value[1:3]
      tensor_rows.append({'tensor':key,'status':'tested','parent_anchor':sa,'child_anchor':sb,'norm2_delta':float(value[0]),'norm2_witness_parent':float(value[1]),'norm2_witness_child':float(value[2])})
     except ValueError as exc:tensor_rows.append({'tensor':key,'status':'undefined','reason':str(exc)})
    row={k:t[k] for k in ['parent','child','witness']};row.update(mode=mode,family=family,edit='dpo' if '-dpo-' in t['child'] else 'sft',tensors=tensor_rows,expected_tensors=len(keys),eligible_tensors=sum(x['status']=='tested' for x in tensor_rows),evidence_tier='construction_records')
    if t['child'] in errors:row.update(status='not_run',reason=errors[t['child']])
    else:
     try:
      sa,sb=aggregate(role_scores);prediction=None if sa==sb else (t['parent'] if sa<sb else t['child'])
      nearest=None if distances[0]==distances[1] else (t['parent'] if distances[0]<distances[1] else t['child'])
      row.update(status='tested',parent_anchor=sa,child_anchor=sb,margin=sb-sa,predicted_parent=prediction,correct=prediction==t['parent'],distance_control_parent=nearest,distance_control_correct=nearest==t['parent'],eligible_norm2_witness_parent=float(distances[0]),eligible_norm2_witness_child=float(distances[1]))
     except ValueError as exc:row.update(status='abstain',reason=str(exc))
    rows.append(row);print(t['child'],row['status'],row.get('correct'),row.get('margin'),flush=True)
 atomic_json(out/'progress.json',{'rows':rows,'inventories':inventories,'cost':cost,'input_sha256':hashes})
primary=[r for r in rows if r['mode']=='stored']
control=[r for r in rows if r['mode']=='bf16_view']
allrows=rows;rows=primary
summary={'cases':len(rows),'models':len(inventories),'answered':sum(r.get('predicted_parent') is not None for r in rows),'correct':sum(r.get('correct',False) for r in rows),'wrong':sum(r.get('predicted_parent') is not None and not r['correct'] for r in rows),'ties':sum(r['status']=='tested' and r['predicted_parent'] is None for r in rows),'abstain':sum(r['status']=='abstain' for r in rows),'not_run':sum(r['status']=='not_run' for r in rows),'complete':len(rows)==16,'distance_correct':sum(r.get('distance_control_correct',False) for r in rows)}
atomic_json(out/'results.json',{'summary':summary,'rows':allrows,'control_summary':dict(Counter(('correct' if r.get('correct') else 'wrong') if r.get('predicted_parent') is not None else r['status'] for r in control)),'inventories':inventories,'cost':cost,'input_sha256':hashes,'elapsed_seconds':time.monotonic()-started})
print(json.dumps(summary),flush=True)
