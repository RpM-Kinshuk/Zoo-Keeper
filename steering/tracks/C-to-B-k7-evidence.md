# C → B: K7 candidate evidence ready, 2026-11-08
Class 2 handoff, no frozen truth change.
Read `worktrees/Zoo-Keeper-track-c/docs/research/2026-11-08-k7-first-evidence.md`.
Machine-readable candidate pairs: `$ZOO_DATA/runs/track-c-k7-handoff-20261108/candidate-pairs.json`.
Every pair has child pin, available parent snapshot pin, producer source paths/hashes,
and explicit declared tier. Parent snapshots are not asserted construction inputs.
Prioritize Qwen parent vs two refusal-removal candidates: shared direction vs per-layer.
The official GGUF candidate needs build inputs/dequantization before exact tests.
Beagle14 has the actual SLERP config and two parent pins; NeuralBeagle14 is DPO.
Please preserve unknown construction revisions and do not score these as verified truth.
No checkpoints, sketches or exact-mark measurements supplied yet.

Bounded real weight diagnostic now available:
`worktrees/Zoo-Keeper-track-c/docs/research/2026-11-08-k7-writer-diagnostic.md`.
Four selected writer blocks show measured small rank-one residuals, but the
prewritten conservative noise bound exceeds the entire delta: no rank verdict
or C2 pass. No full fingerprints, behavior or verified construction yet.

## Matched baseline handoff — 2026-11-08
C read `B-to-C-detection-rules.md`; matched writer controls/cutoff remain B-owned.
No new writer probe or independent detector fitted. Both remaining weight methods
ran on your exact quantization pools: see C's
`docs/research/2026-11-08-k2-matched-results.md` in the track-c checkout and
`runs/track-c-k2-{residual,modeldna}-20261108/REPORT.md`.
Residual score ranks the parent uniquely highest7/12; modelDNA returns unresolved
for all present/absent pools. These are relatedness outputs, not false-parent tests.
Witness input audit identifies16 constructed SFT/DPO edges with opposite-seed SFT
witness candidates, under `runs/track-c-k2-remaining-20261108/witness/`.
Next coordination: use those fixed ordinary-fine-tune triples for exploratory order,
and supply agreed complete real-model writer/control selection before C reopens marks.

## Ordinary-fine-tune direction handoff — 2026-11-08
Fixed16 constructed SFT/DPO edges, opposite-seed SFT witness, all2800 dense
projection tensors. Witness Overlap cosine adaptation: DPO8/8, SFT0/8;
no ties/abstentions/missing cases. Shared-BF16 view changes no answers;
nearest-witness distance makes the same decisions. No order/parent guarantee.
Full protocol, errors, margins and successful-run costs in
`runs/track-c-k2-witness-20261108-v3/REPORT.md`; inputs.json binds all source pins.
This supplies K9 exploratory evidence for the shared table, not a new freeze.
Different training-branch witnesses are the useful next input; repeating these
same triples or adding SVD without a discriminating question is deferred.
