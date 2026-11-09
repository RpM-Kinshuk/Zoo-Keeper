# Training Leaves Traces: matched quantization comparison
Declared before weight inspection or scores, 2026-11-08.

Use the exact twelve queries and 126 candidates in
`runs/track-b-k5-parent-out-20261108/protocol.json`, forty-six distinct local models.
Do not change any pool, split, identity, label, scientific threshold or truth table.
Construction records determine immediate-parent labels. All these close relatives
share ancestry: controls for a quantization mark are NOT negatives for ancestry.

Implement equations 3–6 and Appendix B.1 of
https://arxiv.org/html/2608.14929v1 on the complete canonical MLP path down×up
for every block; exclude gate×down and attention paths explicitly. OLMo uses
its equivalent feed-forward projections. Decode stored values to float64,
subtract tr(M)/d I, normalize the flattened remainder. Compute trace-concentration
gate with tau_s=min reference-query concentration, then Hungarian maximum matching.
This is the published reference-oriented gate; do not silently symmetrize it.
No independent compatible roots in these pools: calibrated Related/Unrelated,
AUROC, false-parent rate, edge type and direction are unsupported, never zero.
Compute scores, per-query unique top candidate and true-parent rank for descriptive
comparison only; ties use exact equality, no chosen epsilon. Remove the parent
and report the ranking, not an invented abstention rule. No calibration fitted.

Execution completeness PASS iff all126 scores finite, all12 queries covered,
and all expected layers read, pins match, no skipped matrices or unreported case.
Any missing/invalid input is not_run; operational failures do not validate science.
No discrimination pass target is set: rankings are exploratory, not calibrated
parent decisions. Algebra tests PASS at float64 tolerance1e-10: centered identity
removed, hidden permutation/rescaling invariance, Hungarian recovery, gate behavior;
invalid/zero/nonfinite/incompatible inputs rejected. Tolerance is roundoff only.
Write tests first. Record backend, read/hash bytes and wall time separately.

CPU only, eight threads; no GPU availability established on current host.
At most46 models, expected <1hour; hard stop110minutes, no model downloads,
no raw blocks/signatures retained, only scalar comparisons and input hashes saved.
Re-use safetensors.safe_open, scipy assignment and lab.artifacts.atomic_json;
no remote code execution or new benchmark framework. Source audit for modelDNA
runs independently; Witness Overlap follows. No claim promotion.
