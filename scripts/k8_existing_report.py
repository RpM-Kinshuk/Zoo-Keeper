"""Derive decision-relevant tables from saved scores, without rescoring."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

from lab.artifacts import atomic_json
from evidence.pilot_analysis import _atomic

root = Path(os.environ['ZOO_DATA']).resolve()
repo = Path(__file__).resolve().parents[1]
out = root / 'runs/track-b-existing-artifacts-20261108'
assert not (out / 'REPORT.md').exists()
hashes = {}


def read(rel):
    p = root / rel
    hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


data = read('runs/track-b-existing-artifacts-20261108/results.json')
pair_rows = read('runs/track-b-existing-artifacts-20261108/pairs.json')
catalog = read('runs/track-a-k8-protocol-20261108/catalog/queries.json')
queries = {q['model_key']:q for q in catalog if q['eligible_proposed_query']}
refs = read('runs/track-b-pivot-unblock-a-20261105/active-roster-v2/reference_pairs.json')
negative = {frozenset((p['a'],p['b'])) for p in refs['pairs'] if p['label']=='independent'}
edge_path = root / 'runs/track-b-pivot-unblock-a-20261105/active-roster-v2/edges.csv'
hashes[str(edge_path.relative_to(root))] = hashlib.sha256(edge_path.read_bytes()).hexdigest()
edges = {r['child']:json.loads(r['params_json']) for r in csv.DictReader(edge_path.open())}


def stratum(q):
    mid = q['model_id']
    if '-decision-' in mid:
        return 'decision head'
    if mid in edges:
        r = edges[mid]
    else:
        recipes = [p for p in q['source_paths'] if p.endswith('/recipe.json')]
        if not recipes:
            return 'BERT scratch controls' if q['cohort']=='K6_BERT30' else 'bases / pretraining controls'
        r = read(recipes[0])
        r = r.get('recipe',r)
    if r['edge_type']=='merge':
        return 'merge '+r['method']
    return {'lora':'adapter','quantization_sim':'quantization', 'sft':'SFT/DPO',
            'dpo':'SFT/DPO','abliteration':'refusal construction','laundering':'disguise'}.get(r['edge_type'],r['edge_type'])


strata = {k:stratum(q) for k,q in queries.items()}
denoms = Counter()
for r in pair_rows:
    a,b = r['a'],r['b']
    label = 'positive' if set(queries[a]['root_keys'])&set(queries[b]['root_keys']) else 'negative' if frozenset((a,b)) in negative else 'unknown'
    denoms[label+'_total'] += 1
    if r['score'] is not None:
        denoms[label+'_finite'] += 1
        if r['score'] >= data['summary']['threshold']:
            denoms[label+'_hits'] += 1
assert denoms['negative_finite']==180 and denoms['negative_hits']==0
assert denoms['unknown_hits']==0
present = [r for r in data['rows'] if r['variant']=='present_catalog']
misses = [r for r in present if r['known_positive_comparable_groups'] and not r['useful_family_evidence']]
assert len(misses)==15 and all(strata[r['query']]=='quantization' for r in misses)
absent = [r for r in data['rows'] if r['variant']=='source_removed' and r['paired_present_eligible']]
family_absent = [r for r in absent if r['known_positive_candidate_groups']==0]
assert len(family_absent)==1 and family_absent[0]['model_id']=='q25-15-decision-s0'
assert family_absent[0]['unavailable'] and not family_absent[0]['family_abstention']

table = []
for name in sorted(set(strata.values())):
    p = [r for r in present if strata[r['query']]==name]
    a = [r for r in absent if strata[r['query']]==name]
    table.append(dict(edit=name, queries=len(p), known_positive_available=sum(r['known_positive_comparable_groups']>0 for r in p),
        useful_present=sum(r['useful_family_evidence'] for r in p), abstain_present=sum(r['family_abstention'] for r in p),
        paired_source_removed=len(a), useful_removed=sum(r['useful_family_evidence'] for r in a),
        abstain_removed=sum(r['family_abstention'] for r in a), unavailable_removed=sum(r['unavailable'] for r in a)))
assert sum(r['queries'] for r in table)==216
derived = dict(pair_label_denominators=dict(denoms), edit_table=table,
    missed_known_positive_query_keys=[r['query'] for r in misses],
    paired_source_absent_family_absent=[r['query'] for r in family_absent],
    collection_recommendation='none: existing misses need method/noise/label work, not an arbitrary larger candidate pool',
    input_sha256=hashes)
atomic_json(out / 'report-tables.json', derived)
summary = data['summary']
lines = ['# Existing fingerprints: family matches with source present and removed', '',
    'The useful question is whether current evidence identifies family or immediate source,',
    'and whether adding fingerprints would resolve the observed failures. No weights, new',
    'fingerprints or network reads were needed. The 5k roster is not a collection target.', '',
    '| Existing catalog / paired variant | Queries | Known-positive scoreable | Useful family answers | Known-wrong family answers | Family abstentions | Unavailable |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for label, field in [('All existing catalog','all_existing_catalog'),('True source present','paired_source_present'),('True source removed','paired_source_removed')]:
    r = summary[field]
    lines.append(f"| {label} | {r['queries']} | {r['known_positive_scoreable']} | {r['useful_family_queries']} | {r['known_false_query_answers']} | {r['family_abstentions']} | {r['unavailable']} |")
lines += ['', 'Useful means at least one documented same-root hit under the unchanged R5 rule.',
    'It does not mean immediate-parent recovery. All returned hits in this finite catalog',
    'have known-positive root labels; no unknown hits were returned. Pair-level unknowns',
    f"remain explicit: {denoms['unknown_finite']} finite unknown relationships were evaluated.",
    f"Certified independent controls: {denoms['negative_hits']} hits / {denoms['negative_finite']} finite undirected pairs.",
    'Zero observed false hits is not a confidence guarantee; the pairs share roots and',
    'most queries are exposed. Only the two original exact keys retain inherited heldout',
    'labels; this new candidate-pool analysis is descriptive, not fresh heldout validation.', '',
    'All 15 present-source known-positive misses are quantization cases. Adding more',
    'unrelated fingerprints cannot make a missed existing source cross a fixed threshold.',
    'The closed exact-quantizer checks answer a different question and are not reopened.', '',
    '| Construction stratum (truth labels, not detected edit type) | Queries | Known-positive available | Useful present | Abstain present | Paired removed | Useful removed | Abstain removed | Unavailable removed |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in table:
    lines.append('| '+' | '.join(str(v) for v in r.values())+' |')
lines += ['', 'The removed-source decision-head case q25-15-decision-s0 has no remaining known',
    'family candidate and no comparable score: family absent and unavailable, not an',
    'ordinary abstention. The other 133 paired removed-source queries retain scoreable',
    'known relatives. Source absence and family absence must stay separate.', '',
    '## What source checks actually establish', '',
    'The generic engine config has no fitted typing rules or frozen asymmetric direction',
    'evidence, so family scores provide no immediate-parent answer. Source identity and',
    'order accuracy are not evaluated from those scores. Only 20/134 source-present',
    'queries have a unique top-scoring exact source identity; this is a rank diagnostic,',
    'not a classifier result. Those 20 are single-source cases. Ties/group membership',
    'and multi-source cases are retained in the per-query rows.', '',
    'Closed exact quantizer replay remains 12 correct present-source answers and 12',
    'absent-source abstentions in its ORIGINAL pools, not this 216-key universe.',
    'Do not report family matches after removal as false parent answers: no parent was named.', '',
    '## Next evidence justified by these outcomes', '',
    'Collect no additional fingerprints now. First address the method gap: fixed family',
    'similarity misses known quantizations, while unchanged tensors and high similarity',
    'do not distinguish parents from relatives. Use the existing exact-mark outputs and',
    'matched noise controls to propose a scoped decision rule before final testing.',
    'Better source labels/controls and untouched roots are needed for validation; a',
    'larger unlabeled pool would not supply those. C real-case source revisions remain',
    'a provenance dependency, not a storage dependency. No new threshold/split is applied.',
    'Reconsider collection only for a named failure that missing comparable, labeled',
    'evidence could resolve. Specify that minimum case set before costs or capacity.', '',
    '## Reproduction and integrity', '',
    f"216 keys → {summary['groups']} observed row-block groups; {summary['pair_comparisons']} unordered score pairs.",
    'Self groups and true-source groups removed exactly; relatives retained. Duplicate',
    'groups describe measured blocks, not full-weight identity or inherited truth labels.',
    f"CPU one numerical thread; {summary['elapsed_seconds']:.3f}s analysis elapsed. Zero GPU/weight/network reads.",
    'Commands from Track B checkout with esd_ind active and ZOO_DATA set explicitly:',
    '`PYTHONPATH=$PWD:$ZOO_DATA/Model-Zoo python scripts/k8_existing_artifacts.py`',
    'then the same environment with `python scripts/k8_existing_report.py`.',
    'Outputs are immutable; use a new run name for a new execution.',
    'protocol.json predates scoring; inputs.json binds every fingerprint/source; pairs,',
    'groups and per-query rows retain all outcomes. One independent review required',
    'negative denominators, edit strata and explicit family-absent accounting; included.',
    'No source files, scientific thresholds, splits or truth edited. Deadlines retained.']
_atomic(out / 'REPORT.md', '\n'.join(lines)+'\n')
print(json.dumps(denoms,sort_keys=True))
