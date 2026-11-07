# Track B handoff

## Current checkpoint — 2026-11-05, scoped E9 baseline verified

Use `esd_ind`. Reuse existing code/helpers/entry points wherever practical,
useful and possible (D7.10). Current work uses existing lineage CLI and evaluation
harness only; no new production code, fitting, calibration or C blind reads.
Current report: `runs/track-b-e9-public166-20261105/REPORT.md`;
`verification.json` verifies879 source bindings and five evaluation freezes.
All owned evaluation commands exited. This analysis/documentation batch is
uncommitted; no push or worktree deletion.

**E9 baseline:** approved166public identities,27,390ordered pairs;2,058emitted
uncalibrated same-root claims. Same-architecture AUROC0.689385, TPR0.539623 at
inherited0.75,0false matches. Parent/ancestor have zero scored claims and zero
coverage; direction accuracy null.1,298compatible-architecture pairs fail the
unchanged source-version guard. Refreshed only8T0manifests via existing builder
at exact unchanged checkpoint keys; original manifests/fingerprints preserved.
No individual architecture in formal166 has both same-root label classes, so
pooled AUROC is insufficient within-architecture validation.

**Separate controlled174:** reuse8already measured public Pythia derivatives;
formal roster unchanged, no new generation/extraction/fit. Original6held-out
queries,17same-architecture identities/16eligible candidates per query:
relatedness AUROC1.00,12/12positive recall and0/84negative false matches at0.75.
Parent/ancestor still fully abstain, including4held-out parent positives.
All2,058formal claims remain exactly identical. This is a descriptive extension
of already observed controls, not independent replication or a174effective-pool
claim. Source hashes and inherited root assignments are frozen. Full E9 remains
open: large pool, actual costs, baselines and directed evidence unavailable.

**A has consumed the handoff:** current `TRACK-A.md` reports approved166calibrated
R6 and expanded703-model R6 complete, all6CLIqueries pass. Do not repeat those
runs. `runs/track-a-expanded-r6-20261105/` records development G0 diagnostics:
immediate-parent8.276786false matches/query atTPR0.904459 in an incomplete small
pool; spectral alpha gap0.427726>0.1 and empirical-null width diagnostic fail.
Retain full spectra/empirical calibration; no5000-candidate or G0/G1 decision.

**Generation/roster status:** L7generation8/8(classic4+stock Heretic4), L14heads6/6.
All8drift-v2 functional-equivalence outcomes remain FAILED. The owner subsequently
approved the exact8-slot construction-counting/roster amendment, so active
`formal166-public-constructions-v2` has166/166mapped; original defaults,158map and
failed results remain immutable. The earlier generation-only restriction was
superseded for this explicit amendment. Read
`runs/track-b-pivot-unblock-a-20261105/A-unblock-request-active.json` and approval.
The embedded-PEFT view-name repair is merged in Model-Zoo at7c3c356; same-key A
remeasurement is already consumed. Historical verification470PASS/1skip and
658Zoo-Keeper PASS is retained; no code changed in the current E9 batch.

**Approved E6 pivot:**48nodes/44edges, all8quartet diagnostics flag81.8–96.2%
violations; nRFnull. Report how tree-like and focus on supervised edge scoring.
Existing E7 diagnostic uses18development/6held-out queries; no validated
asymmetric direction. Original freezes/results/thresholds are retained.

**Next:** separately reviewed asymmetric-direction protocol using existing
modules/development controls; reviewed source-version compatibility path or
matching-version inputs; public baselines/measured costs and owner-launched
large pool. Do not fit a new model or relax a guard under this baseline scope.
C's blind boundary remains untouched.13raw checkpoints (23.8GB) remain at
`/tmp/zoo-track-b-remaining-20261105`; durable copy still needs primary capacity.
All numerical evidence is primary. Historical entries below are superseded by
this current checkpoint where approvals/progress differ.

## Historical checkpoint — 2026-11-05, esd_ind GPU continuation

Continue in `worktrees/Zoo-Keeper-track-b`, branch `track-b` at `1d728a7`.
Use `conda activate esd_ind`; GPU execution needs access outside the GPU-isolating
sandbox. CUDA and shared-lock admission were verified. No B jobs remain running.
Earlier source edits and compatibility repair remain uncommitted; this continuation
changed no production code, installed no packages and created no commits.

**16 new checkpoints ready for A** under
`runs/track-b-gpu-continuation-20261105/` (read `REPORT.md`):

- **L9:**8/16 public transforms pass:4 rescalings+4 bounded perturbations.
  All8 permutation/rotation cases fail unchanged equivalence checks; audits retained,
  no output/credit or tolerance relaxation. C's blind subset remains separate.
- **L7:**4/4 classic shared-direction projections complete. Frozen native
  Amazon Polarity direction labels and fixed middle layer; sentiment-conditioned
  mathematical examples, no refusal-quality claim. One direction is shared across
  residual writers; use the frozen drivers, not the old per-layer-direction runner.
  BF16 rounding is explicit. Heretic0/4 still blocked by the installed unsafe loader.
- **L14:**4 new heads verified: Qwen2.5 full,Qwen3 embedded rank8 PEFT,ModernBERT
  seeds0/1;32 steps each, actual safe reload/finite logits, Qwen3 backbone preserved.
  With existing Smol360,5/6 complete.1.5B head remains storage-blocked; its config/
  native inputs are frozen, but its reclaimed local source must be restaged at the
  exact pin before launch. Shared download/cache retained.

**A handoff:** `delivery/A-measurement-request.json`, `ready.json`,
`artifact-index.csv`, `manifests/` and `verification.json`.16 exact content keys,
parents and file hashes; T0 metadata only. A owns new numerical measurements.
Reuse A's verified137 parent/base map and preserve the old six/two queue evidence.

Current readiness: **137 measured /137 full-weight,16 new generated awaiting A,
13 ungenerated** (4Heretic+8failed transforms+1head). See
`delivery/formal-coverage166.csv` and `coverage-overlay.json`; the older global
"current" proof is a historical snapshot. No calibration/G0 or full-lab claim.

Next: A measures the16 new checkpoints. Retain failed-transform evidence for the
person's protocol decision; do not silently relax checks. Finish the1.5B head after
capacity is available and resolve Heretic through a reviewed safe implementation.
Preserve all original recipes/held-outs and C's blind boundary. No commit/merge/push
unless requested. Prior compatibility verification:627 tests PASS; real GPU and
artifact verification for this continuation are recorded in its report.

## Historical checkpoint — 2026-11-04

Numbered mode labels, source revisions and test totals below are historical.

Continue in `worktrees/Zoo-Keeper-track-b`, branch `track-b`. Preserve its
uncommitted source edits (merge compatibility, four-rank training, tests/docs).
Last full suite: **599 passed**; core reviews and real four-GPU runs passed.
No B jobs remain running.

Completed:24 LoRAs,30 merge recipes,12 quantizations,8 SFT,8 DPO,3 prescribed
chains. Current coverage: **137 measured /129 fullT2;29 ungenerated**.

**Next:** reconcile A's latest progress, then work through L7 classic
abliterations, L9 public transforms and remaining L14 heads. Heretic has a
safe-loading blocker; do not count classic as Heretic. A has eight ready T2
upgrades in the separate six- and two-output queues.

Starting points:

- Main `handoff/CHECKLIST.md` and the current `STATUS.md` snapshot.
- Queue pointers: `handoff/track-b-incremental-l12.md`.
- Latest evidence: `runs/track-b-olmo-dpo-finish-20261104/REPORT.md`.
- Current coverage: `runs/track-b-reconciliation-20261103/global-coverage-proof-current.json`.

Use existing plans, source and pinned inputs; reuse completed work. Preserve
held-outs and C's blind boundary. Calibration/G0 and the full lab remain open.
Use eligible GPUs with fresh usage/memory checks and shared locks. Work
autonomously; keep questions, skills and verification lean. No commit/merge/push
unless requested. Historical detail need only be consulted when useful.
