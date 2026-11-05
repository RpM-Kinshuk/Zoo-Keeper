# Track A handoff

## Current continuation requirements — 2026-11-05

The compatibility audit at Model-Zoo `de2aabd` / Zoo-Keeper `99cc6c2`
found work remaining; see [the shared handoff](MODEL-ZOO-COMPATIBILITY.md).

- Update A's old collection/query worktrees to the committed interfaces while
  preserving local edits; verify the actual imported source.
- Derive new, hash-bound launches for both pending six- and two-output queues
  using `--measurement_mode full_weights` and `--also_row_block_sizes`.
  Preserve original queues and completed measurements.
- Reconcile coverage additively, then reuse compatible existing artifacts for
  build/query. Package defaults do not automatically include historical runs or
  conventions. No remeasurement is needed merely to change names.
- Latest recorded baseline is 137 measured / 129 full-weight / 29 ungenerated;
  eight upgrades are ready. Complete those upgrades before resuming R5/calibrated
  R6 when their actual prerequisites are met. This audit ran no model jobs.

## Historical checkpoint — 2026-11-04

The numbered mode names and source revisions below describe that checkpoint.
The requirements above govern new launches.

Continue in the existing `track-a` worktrees under `worktrees/`. Extractor and zoo/query source are implemented and merged. Model-Zoo main: `5262c90`; the approved reference-source amendment is on Zoo-Keeper main at `7a0d786`.

Current coverage: **129/166 full T2; 135 measured overall**. Six new B outputs are ready for T2; **31 checkpoints remain ungenerated**. All 47 references are complete. Full spectra and original source provenance are retained. No Track A jobs are running.

**Next:** inspect B's six-output handoff and complete only the new T2 upgrades:
`runs/track-b-l6-remaining-20261104/execution6/incremental-A-l12-launch.json`.
Reuse completed bases and fingerprints. Continue toward R5 calibration and the calibrated R6 build as the remaining inputs become ready.

Useful starting points:

- Current coverage/proof: `runs/track-b-reconciliation-20261103/global-coverage-proof-current.json`.
- Existing T2 paths: `runs/track-a-r5-readiness-20261104T092928Z/formal129-artifact-map.json`.
- Cross-track updates: `handoff/track-b-incremental-l12.md` and the current `STATUS.md` snapshot.

Preserve held-out splits; calibration/G0 is still incomplete. Optional Hub retries are closed. GPUs are authorized when both utilization and occupied memory are below 80%; follow the existing admission/locking policy.

Work autonomously and keep checks and documentation lean. Cleanup/checkpoint evidence: `runs/track-a-handoff-20261104T135402Z/`.
