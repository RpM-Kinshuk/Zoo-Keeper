# Ancestry discrimination versus immediate-source ambiguity
2026-11-09 · Track C · existing frozen controls

Practical finding: modelDNA and centered residual signatures separate same-root
quantized descendants from different-seed-root checkpoints on this small panel,
although earlier tests did not identify a unique source among relatives. Useful
ancestry discrimination and exact-source identification are different questions.
No new model collection, training, split, truth or threshold was needed.

| Method | Development scored / AUROC | Inherited-test scored / AUROC |
| --- | --- | --- |
| modelDNA0.1.0 native fast, new |30/30 ·1.00|30/30 ·1.00|
| Training Leaves Traces canonical MLP, new |30/30 ·1.00|30/30 ·1.00|
| Trace-native, archived unchanged |30/30 ·1.00|30/30 ·1.00|
| Spectral-Signatures-native, archived unchanged |30/30 ·1.00|30/30 ·1.00|

Each partition has12 same-root and18 different-root ordered pairs from just
TWO root labels. Those are dependent/reversed checkpoint comparisons, not18
independent negative experiments. There are four Pythia14m seed roots overall,
with q4/q8 constructed descendants. Root independence/provenance and truth-tier
strings are inherited from the existing frozen benchmark; no new training
history is certified. Labels, eligibility, every original row and order preserved.
This already exposed panel is not a new prospective or blind test. No broad
false-positive bound, decoder-scale reliability, type/order result, calibrated
immediate-parent result or claim promotion follows. Current large real-model
neighborhoods still have no matched independent ancestry negatives.

## Native score bands explain an important naming ambiguity
ModelDNA's published positive threshold remains0.9, abstention threshold0.5,
with its unchanged synthetic-bootstrap coefficients. No threshold fitted here.
All12 same-root pairs in EACH partition are in the positive band. None of the
18 different-root pairs in EITHER partition reaches that positive band.
But the18 DEVELOPMENT different-root pairs all receive SAME_FAMILY_UNRESOLVED
in the abstention band; the18 inherited-test different-root pairs receive
NO_MATCH below the abstention band. The class string alone is therefore not a
confirmed-family decision. This is the method's abstention behavior, not18
confident false ancestry answers and not evidence of a guaranteed calibrated FPR.
Native type strings on same-root pairs are retained, not scored as edit truth.

Existing real Qwen/Tulu results were checked WITHOUT rerunning: all12 unresolved
verdicts there have probabilities in the native positive band; the ambiguity is
competition among multiple relatives. Their same class label has a different
score/notes context than the development independent-seed abstentions. High
family evidence still does not identify a unique historical source or establish
verified construction. See label-semantics-audit.json; historical results unchanged.

Exact score ranges and all per-pair features/verdicts remain in results.json.
The four native signal aggregates are present on all60 new modelDNA pairs.
Residual signatures use complete GPTNeoX dense4h→h times denseh→4h MLP factors,
all layers, the existing centering/reference-concentration gate/Hungarian matcher.
No test-null calibration, no new parent or missing-source rule, no SVD-variant
sweep. The single-class132-pair training panel was deliberately not rerun because
it cannot answer this independent-control question. Archived spectral methods
were not rerun; their universe SHA256s match the new run exactly.

## Input and leakage checks
All12 complete local safetensors identities match exact catalog manifests and
construction parent-file pins; configs/recipes/lineage bound in inputs.json.
No name-only joins. Exact keys, root keys and full weight-file hash sets are
disjoint between partitions. Frozen same-root labels agree with construction-root
mapping. Every original field/order was checked against all method rows.
These checks address accidental identity leakage, not generalization from two
roots or uncertainty about public seed training histories. No tuning occurred.

## Cost and status
Operational completeness PASS:12 native fingerprints,12 full-MLP profiles,
60/60 scored pairs for each new method. No not_run case or silently dropped pair.
Native local-reader logical bytes 52,784,484; complete-MLP
logical bytes 18,874,368; counted input SHA256
scan bytes 641,462,368. These are distinct read counters, not
a physical disk-I/O sum (OS caching/metadata amplification not instrumented).
Elapsed extraction/scoring 62.156887s, CPU8threads, no GPU or model
network reads. Native fingerprint samples remain local; no raw blocks/deltas saved.
Historical spectral extraction cost is unknown, so cached and fresh read timings
are not compared as method speed. esd_ind lacked modeldna at session start;
published0.1.0 wheel restored --no-deps, no existing package/version changed.
Installation and tests excluded from run elapsed time. No commits/merges/pushes.

Stopping condition met: the fixed controls show useful ancestry discrimination
within their scope, not a reason to add seeds, rerun quantization, or promote C1.
Next shared outcome: report family score/band and source resolution separately,
retain small-root/real-control gaps, and let B's agreed validation proposal set
any further discriminating experiment. Witness contrast remains deferred by B.
