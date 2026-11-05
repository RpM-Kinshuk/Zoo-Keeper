# Track B handoff

## Current continuation requirements — 2026-11-05

See [the shared compatibility handoff](MODEL-ZOO-COMPATIBILITY.md).

- Reconcile B's old worktree with committed compatibility readers while preserving
  its local training/merge edits. Main's query/lineage support needs no duplicate fix.
- Fix the remaining producer/evaluator contract together:
  `lab/cohort_evidence.py` still requests a removed mode;
  `lab/scripts/evaluate_delivery.py` still assumes old labels/suffixes and today's
  writer version. Use descriptive modes and compatibility-aware source mappings,
  preserving exact frozen measurement/version/hash checks.
- Produce new delivery/evaluation mappings for A, reuse completed evidence, and
  keep frozen reports and study protocols intact. Existing tiny resume test
  reproduces the producer failure; shared reader checks pass (24 tests).
- Continue L7/L9/L14 after this focused compatibility work. Both pending A queues
  remain distinct, and original training recipes/root splits are unchanged.

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
