"""Existing scorer on existing catalog groups; no weights or classifier fitting."""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from evidence.fingerprint import load_fingerprint
from evidence.pilot_analysis import _score, _atomic
from zoo.tables import duplicate_groups
from lab.artifacts import atomic_json
from lineage.engine import _direction

root = Path(os.environ['ZOO_DATA']).resolve()
repo = Path(__file__).resolve().parents[1]
out = root / 'runs/track-b-existing-artifacts-20261108'
out.mkdir(exist_ok=False)
started = time.monotonic()
hashes = {}


def bind(p, expected=None):
    rel = str(p.relative_to(root))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if expected is not None:
        assert h == expected, 'input digest mismatch: ' + rel
    hashes[rel] = h
    return h


def read(rel):
    p = root / rel
    bind(p)
    return json.loads(p.read_text())


catalog = read('runs/track-a-k8-protocol-20261108/catalog/queries.json')
queries = {q['model_key']: q for q in catalog if q['eligible_proposed_query']}
assert len(queries) == 216
calibration = read('runs/track-a-roster-v2-calibration-20261105/calibration.json')
reference = read('runs/track-b-pivot-unblock-a-20261105/active-roster-v2/reference_pairs.json')
method = read('runs/track-b-real-cohort-v1/inference-method.json')
prior = read('runs/track-b-k5-parent-out-20261108/results.json')
assert method['direction_method_hash'] is None and method['typing_rules'] is None
threshold = calibration['related_threshold']
null_max = max(calibration['null_quantiles'])
negative_pairs = {frozenset((p['a'], p['b'])) for p in reference['pairs'] if p['label'] == 'independent'}
protocol = repo / 'docs/steering/K8-existing-artifacts-protocol.md'
bind(protocol)
bind(Path(__file__).resolve())
for module in ['fingerprint.py', 'pilot_analysis.py', 'compare.py', 'sketch.py', 'compatibility.py']:
    bind(root / 'Model-Zoo/evidence' / module)
bind(root / 'Model-Zoo/zoo/tables.py')
bind(repo / 'lineage/engine.py')
atomic_json(out / 'protocol.json', dict(text=protocol.read_text(), keys=sorted(queries),
    threshold=threshold, null_max=null_max, k=64, cpu_threads=1,
    candidate_universe='eligible existing catalog only', scientific_status='exposed_descriptive',
    no_new_model_weights=True, no_gpu=True, input_sha256=hashes.copy()))

fps, views, pin_counts = {}, [], Counter()
for key, q in sorted(queries.items()):
    pointer = q['row_block']
    p = root / pointer['path']
    bind(p, pointer['sha256'])
    pin_counts['existing_digest' if pointer['sha256'] else 'fresh_digest_no_prior'] += 1
    fp = load_fingerprint(p)
    assert fp.meta['repo_id'] + '@' + fp.meta['revision'] == key
    assert fp.meta['tier'] == 'row_blocks' and fp.meta['row_block'] == 256
    assert fp.meta['sketch_seed'] == 20261028 and fp.meta['sketch_k'] >= 64
    # Copy only the already-declared nested prefix into analysis memory.
    # No source file or source metadata is rewritten.
    fps[key] = replace(fp, sketch=fp.sketch[:, :64, :64].copy())
    for i, name in enumerate(fp.names):
        views.append(dict(model_key=key, tier='row_blocks', name=name, geometry=fp.geometry[i],
            n_rows=int(fp.shape[i, 0]), n_cols=int(fp.shape[i, 1]),
            r0=int(fp.rows[i, 0]), r1=int(fp.rows[i, 1]), block_sha256=fp.sha256[i]))
    direction, reason = _direction(SimpleNamespace(meta=fp.meta), SimpleNamespace(key='unused'),
                                   {'adapter_factors': None}, method)
    assert direction is None and reason['reason'] == 'no_asymmetric_direction_evidence'

groups = duplicate_groups(pd.DataFrame(views))
members = {}
for key, rep in sorted(groups.items()):
    members.setdefault(rep, []).append(key)
assert set(groups) == set(queries)
atomic_json(out / 'groups.json', dict(groups=groups, members=members,
    scope='identical compatible measured row blocks, not full weight identity'))

scores = {}
pair_rows = []
keys = sorted(queries)
for i, a in enumerate(keys):
    assert time.monotonic() - started < 6600
    for b in keys[i+1:]:
        value = _score(fps[a], fps[b], k=64)
        scores[a, b] = value
        pair_rows.append(dict(a=a, b=b, score=value))
    if i % 25 == 0:
        atomic_json(out / 'progress.json', dict(query_rows=i+1, pairs=len(pair_rows),
                                              elapsed_seconds=time.monotonic()-started))
        print('scored', i+1, 'of', len(keys), 'query rows', flush=True)
atomic_json(out / 'pairs.json', pair_rows)


def score(a, b):
    return scores[tuple(sorted((a, b)))]


def label(a, b):
    if set(queries[a]['root_keys']) & set(queries[b]['root_keys']):
        return 'positive'
    if frozenset((a, b)) in negative_pairs:
        return 'negative'
    return 'unknown'


def evaluate(key, variant, removed, present_eligible):
    q = queries[key]
    candidates = sorted(rep for rep in members if rep not in removed)
    comparable = [c for c in candidates if score(key, c) is not None]
    hits = [c for c in comparable if score(key, c) >= threshold and score(key, c) > null_max]
    hit_labels = Counter(label(key, c) for c in hits)
    positive_candidates = [c for c in comparable if label(key, c) == 'positive']
    best = max((score(key, c) for c in comparable), default=None)
    top = [c for c in comparable if score(key, c) == best]
    exact_parents = set(q['direct_parent_keys'])
    return dict(query=key, model_id=q['model_id'], cohort=q['cohort'],
        inherited_split=q['inherited_split'], root_R5_heldout_restriction=q['root_R5_heldout_restriction'],
        variant=variant, paired_present_eligible=present_eligible,
        parent_keys=sorted(exact_parents), removed_groups=sorted(removed),
        candidate_groups=len(candidates), candidate_identities=sum(len(members[c]) for c in candidates),
        comparable_groups=len(comparable), known_positive_comparable_groups=len(positive_candidates),
        known_positive_candidate_groups=sum(label(key,c)=='positive' for c in candidates),
        useful_family_evidence=bool(hit_labels['positive']),
        known_false_hits=hit_labels['negative'], unknown_hits=hit_labels['unknown'],
        positive_hits=hit_labels['positive'], n_hits=len(hits), hits=hits,
        false_hit_bounds=[hit_labels['negative'], hit_labels['negative']+hit_labels['unknown']],
        family_abstention=bool(comparable) and not hits, unavailable=not comparable,
        unknown_only_answer=bool(hits) and not hit_labels['positive'] and not hit_labels['negative'],
        top_score=best, top_keys=top, top_exact_parent_identity=bool(set(top)&exact_parents),
        unique_top_exact_parent_identity=len(top)==1 and top[0] in exact_parents,
        top_group_contains_parent=any(set(members[c])&exact_parents for c in top),
        generic_parent_status='no_asymmetric_direction_evidence_not_a_parent_answer')


rows = []
for key, q in sorted(queries.items()):
    self_group = groups[key]
    parents = set(q['direct_parent_keys'])
    parent_missing = parents - set(queries)
    parent_self = {p for p in parents if p in groups and groups[p] == self_group}
    present_ok = bool(parents) and not parent_missing and not parent_self
    row = evaluate(key, 'present_catalog', {self_group}, present_ok)
    row.update(source_present_status=('no_parent_label' if not parents else 'missing_exact_parent' if parent_missing
        else 'source_in_removed_query_group' if parent_self else 'all_exact_parents_available'),
        source_missing_keys=sorted(parent_missing), source_in_query_group=sorted(parent_self))
    rows.append(row)
    if parents:
        removed = {self_group} | {groups[p] for p in parents if p in groups}
        absent = evaluate(key, 'source_removed', removed, present_ok)
        absent['source_present_status'] = row['source_present_status']
        rows.append(absent)


def summarize(subset):
    hits = sum(r['n_hits'] for r in subset)
    false = sum(r['known_false_hits'] for r in subset)
    unknown = sum(r['unknown_hits'] for r in subset)
    return dict(queries=len(subset), roots=len({k for r in subset for k in queries[r['query']]['root_keys']}),
        scoreable=sum(not r['unavailable'] for r in subset), unavailable=sum(r['unavailable'] for r in subset),
        known_positive_scoreable=sum(r['known_positive_comparable_groups']>0 for r in subset),
        useful_family_queries=sum(r['useful_family_evidence'] for r in subset),
        missed_known_positive_queries=sum(r['known_positive_comparable_groups']>0 and not r['useful_family_evidence'] for r in subset),
        family_abstentions=sum(r['family_abstention'] for r in subset),
        known_false_query_answers=sum(r['known_false_hits']>0 for r in subset),
        unknown_query_answers=sum(r['unknown_hits']>0 for r in subset),
        unknown_only_answers=sum(r['unknown_only_answer'] for r in subset),
        returned_hits=hits, known_false_hits=false, unknown_hits=unknown,
        false_hit_identification_bounds=[false,false+unknown],
        false_hit_fraction_bounds=[false/hits,(false+unknown)/hits] if hits else None,
        unique_top_exact_parent_identity=sum(r['unique_top_exact_parent_identity'] for r in subset),
        top_group_contains_parent=sum(r['top_group_contains_parent'] for r in subset))


present = [r for r in rows if r['variant']=='present_catalog']
paired_present = [r for r in present if r['paired_present_eligible']]
paired_absent = [r for r in rows if r['variant']=='source_removed' and r['paired_present_eligible']]
assert {r['query'] for r in paired_present} == {r['query'] for r in paired_absent}
summary = dict(catalog_queries=len(queries), groups=len(members), pair_comparisons=len(pair_rows),
    digest_counts=dict(pin_counts), all_existing_catalog=summarize(present),
    paired_source_present=summarize(paired_present), paired_source_removed=summarize(paired_absent),
    present_source_status=dict(Counter(r['source_present_status'] for r in present)),
    closed_quantization_original_pool_summary=prior['summary'],
    by_cohort={c:summarize([r for r in present if r['cohort']==c]) for c in sorted({r['cohort'] for r in present})},
    threshold=threshold, elapsed_seconds=time.monotonic()-started, no_new_collection=True,
    scientific_scope='descriptive exposed-case family evidence, not parent predictions or new calibration')
atomic_json(out / 'results.json', dict(summary=summary, rows=rows, input_sha256=hashes))
atomic_json(out / 'inputs.json', hashes)
print(json.dumps(summary, sort_keys=True), flush=True)
