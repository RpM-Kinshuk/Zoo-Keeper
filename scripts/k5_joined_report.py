"""Join existing scientific outputs; no weight reads or detector fitting."""
import csv
import hashlib
import json
import os
from pathlib import Path
from collections import Counter
import io

from lab.artifacts import atomic_json

root = Path(os.environ['ZOO_DATA'])
repo = Path(__file__).resolve().parents[1]
run = 'runs/track-b-k5-joined-20261108-v2'
out = root / run
assert not (out / 'summary.json').exists()
inputs = {}
inputs[str(Path(__file__).resolve().relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def read(relative):
    p = root / relative
    inputs[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


data = read(run + '/results.json')
assert len(data['rows']) == 64 and len({r['model_id'] for r in data['rows']}) == 64
signatures = read(run + '/tensor-signatures.json')
prior = read('runs/track-b-k5-parent-out-20261108/results.json')
linear = read('runs/track-b-k3-deltas-20261107/summary.json')
slerp = read('runs/track-b-k3-slerp-trig-20261107-v2/results.json')
real = read('runs/track-c-k7-handoff-20261108/candidate-pairs.json')
writers = read('runs/track-c-k7-writer-diagnostic-20261108-v2/report.json')
comparisons = read('runs/track-c-k2-lab-spectral-20261107-v2/report.json')
modeldna = read('runs/track-c-k2-modeldna-20261108/results.json')
traces = read('runs/track-c-k2-residual-20261108/results.json')
edges_path = repo / 'lab/edges.csv'
inputs[str(edges_path.relative_to(root))] = hashlib.sha256(edges_path.read_bytes()).hexdigest()
edges = {r['child']: json.loads(r['params_json']) for r in csv.DictReader(edges_path.open())}
assert len(edges) == 112


def group(mid, r):
    if '-decision-' in mid:
        return 'Decision heads'
    return {'lora': 'Adapters', 'abliteration': 'Refusal removal',
            'sft': 'SFT/DPO', 'dpo': 'SFT/DPO', 'quantization_sim': 'Quantization',
            'laundering': 'Disguise'}.get(r['edge_type'],
            {'linear': 'Linear merge', 'slerp': 'SLERP merge',
             'ties': 'TIES merge', 'dare_ties': 'DARE/TIES merge'}.get(r.get('method')))


eligible = Counter(group(k, v) for k, v in edges.items())
assert None not in eligible and sum(eligible.values()) == 112
measured = Counter(group(r['model_id'], edges[r['model_id']]) for r in data['rows'] if r['group'] != 'base')
measured.update({'Quantization': 12, 'Linear merge': 9, 'SLERP merge': 9})
table = []
for g in ['Adapters', 'Refusal removal', 'TIES merge', 'DARE/TIES merge',
          'SFT/DPO', 'Quantization', 'Linear merge', 'SLERP merge', 'Decision heads', 'Disguise']:
    n = eligible[g]
    quant = g == 'Quantization'
    row = dict(group=g, eligible=n, measurement_tested=measured[g],
               measurement_not_run=n-measured[g], parent_present_correct=12 if quant else None,
               parent_present_wrong=0 if quant else None, parent_present_abstain=0 if quant else None,
               parent_present_not_run=0 if quant else n, parent_absent_wrong=0 if quant else None,
               parent_absent_abstain=12 if quant else None, parent_absent_not_run=0 if quant else n)
    row['scope'] = {
        'Adapters': 'untouched tensor invariant and one sampled target rank; support set not parent decision',
        'Refusal removal': 'three sampled writers; full/shared/absent-parent detector not tested',
        'TIES merge': 'target-entry shares only; no approved classification cutoff',
        'DARE/TIES merge': 'target-entry shares only; no approved classification cutoff',
        'SFT/DPO': 'controls; ordinary order exploratory, not exact-mark parent recovery',
        'Quantization': 'closed lab exact replay subset; empirical small-pool result, not general guarantee',
        'Linear merge': 'closed known-source coefficient test; unknown-source/order not tested',
        'SLERP merge': 'closed known-source coefficient test; unknown-source/order not tested',
        'Decision heads': 'not tested in K5 joined run',
        'Disguise': 'bounded K14 separate; no new disguise evaluation',
    }[g]
    table.append(row)

adapter = [r for r in data['rows'] if r['group'] == 'adapter']
assert len(adapter) == 24
assert all(len(r.get('writer_control_diagnostics', [])) == 3 for r in data['rows'] if r['group'] in ('adapter', 'sft_dpo'))
assert all(len(r['rank_diagnostics']) == 3 for r in data['rows'] if r['group'] == 'refusal')
summary = dict(eligible_children=112, measurement_tested=sum(r['measurement_tested'] for r in table),
    measurement_not_run=sum(r['measurement_not_run'] for r in table), new_weight_identities=64,
    adapter_native_untouched=sum(r['native_true_parent_untouched'] for r in adapter),
    adapter_bf16_untouched=sum(r['bf16_true_parent_untouched'] for r in adapter),
    adapter_ambiguous_present=sum(len(r['native_support_present']) > 1 for r in adapter),
    adapter_nonparent_support_absent=sum(bool(r['native_support_absent']) for r in adapter),
    adapter_resolved_sampled_ranks=sum(r['rank_diagnostics'][0]['interpretable'] for r in adapter),
    adapter_sampled_rank_within_target=sum(r['rank_diagnostics'][0]['interpretable'] and
        abs(r['rank_diagnostics'][0]['rank']-r['configured_rank']) <= .25*r['configured_rank'] for r in adapter),
    refusal_resolved_writer_blocks=sum(x['interpretable'] for r in data['rows'] if r['group']=='refusal' for x in r['rank_diagnostics']),
    real_declared_edges=len(real), real_verified_edges=sum(r['truth_tier']=='verified' for r in real),
    real_informative_writer_blocks=sum(r['relative_update'] > .003 and r['cast_noise_bound'] < r['delta_norm'] for r in writers['records']),
    cpu_seconds=data['elapsed_seconds'], parent_decision_tested=12, parent_decision_not_run=100)
assert prior['summary']['present_answers'] == 12 and prior['summary']['present_errors'] == 0
assert prior['summary']['absent_answers'] == 0 and prior['summary']['absent_errors'] == 0
assert linear['linear']['passes'] == 9 and sum(r['coefficient_pass'] for r in slerp['rows']) == 9
atomic_json(out / 'summary.json', dict(summary=summary, lab_table=table, input_sha256=inputs,
    real_case_tiers=dict(Counter((r['expected_edit'] + ':' + r['truth_tier']) for r in real)),
    comparison_scope=comparisons['scope'], matched_comparisons={
        'modeldna': modeldna['summary'], 'training_leaves_traces': traces['summary']}))
buffer = io.StringIO()
writer = csv.DictWriter(buffer, fieldnames=list(table[0]))
writer.writeheader()
writer.writerows(table)


def atomic_text(path, text):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text)
    temp.replace(path)


atomic_text(out / 'shared-evidence.csv', buffer.getvalue())
lines = ['# Shared edit-group evidence for the Nov14 review', '',
    'Exploratory lab measurements, not a final detector evaluation. Native/cast equality and',
    'rank abstentions retain the prewritten rules. No claim promoted; no thresholds fitted.', '',
    '| Group | Eligible | Measurement tested | Useful/wrong present-parent answers | Absent-parent abstentions | Parent trials not run (each variant) | Scope |',
    '| --- | ---: | ---: | --- | --- | ---: | --- |']
for r in table:
    q = r['group'] == 'Quantization'
    lines.append(f"| {r['group']} | {r['eligible']} | {r['measurement_tested']} | {'12 / 0' if q else 'NOT TESTED'} | {'12/12' if q else 'NOT TESTED'} | {r['parent_present_not_run']} | {r['scope']} |")
lines += ['', 'Measurement tested means the scoped measurement ran, not that the entire C2 test passed.',
    'Native untouched-tensor pass count: ' + str(summary['adapter_native_untouched']) + '/24.',
    'Separate bf16-cast equivalence count: ' + str(summary['adapter_bf16_untouched']) + '/24.',
    'Adapter support ambiguous with parent present: ' + str(summary['adapter_ambiguous_present']) + '/24;',
    'support survives true-parent removal: ' + str(summary['adapter_nonparent_support_absent']) + '/24.',
    'These are support sets, not wrong answers or successful parent decisions.',
    'Resolved sampled adapter ranks: ' + str(summary['adapter_resolved_sampled_ranks']) + '/24; within ±25% target: ' + str(summary['adapter_sampled_rank_within_target']) + '.',
    'Other ranks abstain under unchanged floor/confidence rules; no full-rank case verdict.', '',
    '## Exact-share measurements', '',
    '| Group | Cases | Target-entry identical-to-anchor share min–max | Detector verdict |',
    '| --- | ---: | --- | --- |']
for label, g in [('TIES/DARE','ties_dare'),('LoRA controls','adapter'),('SFT/DPO controls','sft_dpo')]:
    values = [r['target_identical_fraction'] for r in data['rows'] if r['group'] == g]
    lines.append(f'| {label} | {len(values)} | {min(values):.12g}–{max(values):.12g} | no approved share cutoff; NOT TESTED |')
lines += ['', '## Real cases, comparisons and uncertainty', '',
    f"C handoff: {len(real)} declared candidate edges; zero verified construction edges. "
    f"Its four supplied Qwen writer blocks yield {summary['real_informative_writer_blocks']}/4 informative rank allowances; INCONCLUSIVE.",
    'No new remote weights were read. Same update-floor and nonvacuity requirements applied;',
    'full/shared-direction and matched real controls remain absent. Source-tiered cases:', '',
    '| Expected edit / evidence tier | Candidate edges | Complete exact-mark tests |',
    '| --- | ---: | ---: |']
for key, n in sorted(Counter(r['expected_edit'] + ' / ' + r['truth_tier'] for r in real).items()):
    lines.append(f'| {key} | {n} | 0 |')
lines += ['', 'Trace and Spectral Signatures retain their existing small old-cohort result.',
    'They are family/symmetric relatedness comparisons, not immediate-parent or edit-type answers;',
    'they have not run on the fixed quantization parent-out pools. C has supplied matched',
    'quantization comparisons for two additional baselines (saved results imported, no rerun):', '',
    '| Method | Matched pairs | Parent present | Parent absent |',
    '| --- | ---: | --- | --- |',
    f"| Training Leaves Traces | {traces['summary']['comparisons']}/126 | true parent uniquely first {traces['summary']['unique_correct_top']}/12, descriptive rank | rankings only; no absent-source rule |",
    f"| modelDNA | {modeldna['summary']['comparisons']}/126 | 12/12 same-family unresolved; no parent answer | 12/12 same-family unresolved; no absent-source detection claim |",
    'No independent ancestry negatives in those pools, so no fair ancestry AUROC/FPR.',
    'Remaining edit-group comparisons and Witness Overlap remain C dependencies.', '',
    'All newly measured constructions are exposed lab evidence with few dependent roots.',
    'No iid pair interval or broad calibration claim is supported. Untested decisions are not',
    'counted as successful abstentions. Quantization absent-parent conditional precision is',
    'undefined (zero answers). Its 12/12 present coverage is not pooled full-cohort coverage.', '',
    'K4 numerical/extraction and K14 behavior/permutation diagnostics stay bounded and separate;',
    'no new tree/disguise claim. D6 allows these open tests in K12, separately from approved conclusions.', '',
    '## Reproduction and costs', '',
    'Use conda esd_ind; set ZOO_DATA explicitly without reading .env. From Track B checkout:',
    '`PYTHONPATH=$PWD:$ZOO_DATA/Model-Zoo python scripts/k5_joined_measurements.py --temporary-lab-root <temporary-lab-root>`',
    'then `python scripts/k5_joined_report.py` with the same PYTHONPATH. Outputs are immutable;',
    'choose a new run directory for a new execution. GPU unavailable; CPU, four threads.',
    f"Completed measurement elapsed {data['elapsed_seconds']:.3f}s. No full end-to-end acquisition cost claim.",
    'Source/weight pins and file hashes are in protocol.json/results.json; report source hashes',
    'are in summary.json. The interrupted v1 retains partial data and is superseded.',
    'Native checkpoints and existing passed runs preserved. No commits/merges/pushes.', '',
    'Remaining work: approved discriminating detector/parent rules, matched calibration,',
    'full wide-writer and adapter-rank coverage, decision heads, source-tiered complete Hub',
    'cases and matched published comparisons. Nov21 and subsequent test deadlines retained.']
atomic_text(out / 'REPORT.md', '\n'.join(lines) + '\n')
print(json.dumps(summary, sort_keys=True))
