# Track A handoff

## Current checkpoint — 2026-11-05, approved roster v2 active

**A's roster/pair-input blockers are cleared.** Owner explicitly approved the
8slot amendment. Active roster `formal166-public-constructions-v2` has166/166
measured identities,0missing, and the existing scorer produces all180independent
and157parent scores. Original166table/158map and all failed audits are retained.
Public construction count16/16 under the approved policy; all8drift-v2
functional outcomes remain FAILED. No G0/G1 decision or blind access.

**Consume:** `runs/track-b-pivot-unblock-a-20261105/A-unblock-request-active.json`.
It seals `active-roster-v2/activation.json`, `approval.json`, `readiness.json`,
models/edges/pairs, artifact-map166 and row-block-map166. Input-only hardlink
directories `row_blocks/` and `full_weights/` provide existing APIs a complete
cohort without copying checkpoints. Use a new output directory and preserve
existing fit/evaluation partitions:180available independent pairs do not imply
permission to fit on held-outs. Existing A calibration/R6 is authorized; no
calibration has been written by B.

**One measurement repaired:** full numerical readiness initially found336/337
pairs, with q3-06-decision-s0 blocked by56embedded LoRA/dense view-name collisions.
Reviewed Task3 naming fix merged into Model-Zoo main at `7c3c356`:
physical sources unchanged, canonical base names and separate[lora]products,
no guard relaxation. The same checkpoint was remeasured; existing parent scorer
now succeeds. Proof/source hashes/new5file bundle:
`runs/track-a-embedded-peft-repair-20261105/`. Active map points to this corrected
bundle; original158map/files unchanged. Activation's complete337pair preflight
used the existing merged reader/scorer, so A can consume these corrected
artifacts through the existing reader/scorer. Future extraction uses the
merged canonical-name fix; historical source freezes remain unchanged.
Focused28tests pass and core review found
no Critical/Important findings; extractor/store package suite470PASS/1skip
(69.31s), with corrected single-thread fake-network test environment. Source fix is now locally merged; no push.

Raw23.8GB checkpoint durability still needs capacity; existing primary numerical
outputs/manifests are sufficient for this calibration/R6 pipeline. No extra
weight extraction or old queue replay is needed.

## Historical checkpoint — 2026-11-05, B delivery measured

All29 newly handed-off checkpoints measured and verified:21 formal +8 auxiliary.
**Formal coverage158/166 full-weight**;8 original failed-transform slots remain.
Eight drift-v2 auxiliaries have separate measurements and zero formal credit.
No calibration, threshold, held-out or roster changes. No jobs/workers/locks.

Current evidence under `runs/track-a-delivery29-20261105/`:
- `REPORT.md`, `verification.json`:29 bundles/full spectra/three companions and
  explicit ModernBERT build/find/info/diff PASS;709.81s in esd_ind on GPUs3/4.
- `formal-artifact-map158.json`, `formal-coverage166.csv`:additive formal mapping;
  original137 measurements and B's generation fields retained.
- `auxiliary-artifact-map8.json`:no formal quota credit.
- `calibration-readiness.json`:8 exact unresolved original transform IDs.

**Next:** resolve original permutation/rotation equivalence failures or obtain an
explicit owner-approved roster/protocol amendment. Do not substitute failed
auxiliaries or write a partial-cohort calibration. R5/calibrated R6 remain open.
B's13 temporary checkpoint inputs still need durable storage (23.8GB versus
about3.3GiB primary free); source root binding retained, numerical outputs and
manifests are preserved on primary storage. No blind outputs inspected.

Historical checkpoints follow; do not repeat completed queues.

## Verified continuation checkpoint — 2026-11-05

All eight pending upgrades COMPLETE in existing esd_ind on GPUs3/4, with
paired admission/shared GPU/output locks. GPU access requires execution outside
the sandbox; the driver works. Both A worktrees current at Model-Zoo `de2aabd`
/ Zoo-Keeper `1d728a7`. Frozen imports/source hashes and original queue hashes
retained under `runs/track-a-compatibility-20261105/`.

**Current:137 measured /137 full-weight /29 ungenerated.** All eight bundles
(full spectra and128/256/512 companions) verified; additive exact-key mapping:
`runs/track-a-compatibility-20261105/full-weight-artifact-map137.json`.
Upgraded explicit build/find/info/diff PASS; no workers and all owned locks
released. See `REPORT.md` and `upgrade-verification.json` in that run.

**Next:** R5 remains blocked on29 ungenerated checkpoints; current readiness is
`runs/track-a-compatibility-20261105/calibration-readiness.json`. Reuse these
137 mappings when new eligible outputs arrive. Preserve held-outs and full
spectra; no partial-cohort calibration or calibrated R6 claim. Earlier queues
and counts below are historical and must not trigger duplicate extraction.

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
