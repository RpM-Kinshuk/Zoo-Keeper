"""Check existing scalar marks and construction-active writer coverage."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from lab.artifacts import atomic_json

root = Path(os.environ['ZOO_DATA']).resolve()
repo = Path(__file__).resolve().parents[1]
out = root / 'runs/track-b-k5-control-audit-20261108'
out.mkdir(exist_ok=False)
hashes = {}


def read(path, expected=None):
    value = path.read_bytes()
    digest = hashlib.sha256(value).hexdigest()
    if expected is not None:
        assert digest == expected
    # Temporary source locations stay portable in the artifact.
    key = str(path.relative_to(root)) if path.is_relative_to(root) else 'temporary-lab/'+path.parent.name+'/'+path.name
    hashes[key] = digest
    return json.loads(value)


old = read(root / 'runs/track-b-k5-joined-20261108-v2/results.json')
protocol = repo / 'docs/steering/K5-control-audit-protocol.md'
atomic_json(out / 'protocol.json', dict(text=protocol.read_text(),
    sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(), new_checkpoint_matrix_reads=False))
release = read(root / 'runs/track-a-k0-20261107/release-manifest.json')
pins = {r['path']:r['sha256'] for r in release['repositories'][0]['inventory']}
positives = [r for r in old['rows'] if r['group']=='ties_dare']
controls = [r for r in old['rows'] if r['group'] in ('adapter','sft_dpo')]
refusal = [r for r in old['rows'] if r['group']=='refusal']
assert len(positives)==12 and len(controls)==40 and len(refusal)==8
shares = []
for anchor in [None]+sorted({r['anchor'] for r in positives}):
    p = [r['target_identical_fraction'] for r in positives if anchor is None or r['anchor']==anchor]
    c = [r['target_identical_fraction'] for r in controls if anchor is None or r['anchor']==anchor]
    shares.append(dict(anchor=anchor or 'pooled', positives=len(p), controls=len(c),
        min_positive=min(p), max_control=max(c), margin=min(p)-max(c),
        all_positive_zero_false_higher_share_cutoff_exists=min(p)>max(c)))


def gates(blocks):
    return dict(blocks=len(blocks), status=dict(Counter(x['status'] for x in blocks)),
        below_relative_update_floor=sum(x['rel_update_norm']<.003 for x in blocks),
        low_confidence=sum(x['rank_confidence']!='high' for x in blocks),
        interpretable=sum(x['interpretable'] for x in blocks),
        interpretable_rank_one=sum(x['interpretable'] and x['rank']==1 for x in blocks))


writers = []
for r in refusal:
    mid = r['model_id']
    candidates = [root/'lab'/mid/'abliteration.json',Path(os.environ['TEMPORARY_LAB_ROOT'])/mid/'abliteration.json']
    path = next(p for p in candidates if p.exists())
    meta = read(path, pins['models/'+mid+'/abliteration.json'])
    classic = meta['method']=='classic'
    per = meta.get('per_layer_settings',{})
    selected = []
    for x in r['rank_diagnostics']:
        layer = re.search(r'layers\.(\d+)\.',x['tensor']).group(1)
        if classic:
            assert x['tensor'] in meta['writers']
            weight = 1.
        else:
            weight = per[layer]['mlp.down_proj']['weight']
        selected.append(dict(tensor=x['tensor'],layer=int(layer),construction_weight=weight,
                             active=weight!=0,gate=x))
    active_layers = sorted(int(k) for k,v in per.items() if v['mlp.down_proj']['weight']!=0) if not classic else sorted({int(re.search(r'layers\.(\d+)\.',n).group(1)) for n in meta['writers'] if n.endswith('down_proj.weight')})
    writers.append(dict(model_id=mid,anchor=r['anchor'],method=meta['method'],
        algorithm=meta['algorithm'],classic_shared_direction=meta.get('shared_direction_all_writers'),
        heretic_direction_index=meta.get('direction_index'),selected=selected,
        active_down_layers=active_layers,proposed_first_three_active_down_layers=active_layers[:3]))
atomic_json(out/'results.json', dict(share_feasibility=shares,
    positive_writer_gates=gates([x for r in refusal for x in r['rank_diagnostics']]),
    control_writer_gates=gates([x for r in controls for x in r['writer_control_diagnostics']]),
    writer_construction_coverage=writers,input_sha256=hashes,
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print(json.dumps(dict(shares=shares, positive_gates=gates([x for r in refusal for x in r['rank_diagnostics']]),
    control_gates=gates([x for r in controls for x in r['writer_control_diagnostics']]),
    heretic_selected_active=sum(x['active'] for r in writers if r['method']=='heretic' for x in r['selected']),
    heretic_selected_total=sum(len(r['selected']) for r in writers if r['method']=='heretic'),
    classic_shared_count=sum(r['classic_shared_direction'] is True for r in writers)),sort_keys=True))
