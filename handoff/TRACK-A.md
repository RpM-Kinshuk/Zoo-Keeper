# Track A handoff

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
