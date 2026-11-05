# Model-Zoo compatibility: 2026-11-05

Audit baseline: Model-Zoo main `de2aabd`, Zoo-Keeper main `99cc6c2`.
This is a forward migration handoff, not a claim that all consumers are fixed.
The current contract below supersedes older interface examples in this folder;
older scientific protocols, measurements and recorded commands remain historical.

## Current contract and components already compatible

- Measurement modes: `metadata_only`, `row_blocks`, `full_weights`.
  Extraction uses `--measurement_mode row_blocks|full_weights` and
  `--also_row_block_sizes`; oversized-model controls are
  `--oversized_models_evidence` and `--oversized_models_device`.
- New artifact suffixes are `.row_blocks.fingerprint.safetensors`,
  `.row_blocks_128.fingerprint.safetensors`,
  `.row_blocks_512.fingerprint.safetensors`,
  `.full_weights.fingerprint.safetensors`, and `.full_weights.spectra.h5`.
- New model columns are `row_block_fingerprint`, `full_weight_fingerprint`,
  `full_weight_spectra`; the database also includes the `layers` table.
  Evidence writes version 3 and reads 1/2/3; zoo writes version 2 and reads 0/1/2.
  These serialization versions do not indicate changed numerical algorithms.
- Main-checkout query and lineage loaders already normalize historical names.
  Existing calibration-config mode values are normalized too: no rewrite needed.
  Keep identity, row geometry, sketch size/seed, exact frozen source version,
  artifact hashes and path-containment checks. A readable version alone does not
  establish that two scientific measurement contracts can be mixed.
- Reuse `evidence.compatibility` at input boundaries, including `canonical_mode`,
  `companion_spectra_path`, and table normalization. Never relabel old source
  files as version 3 or rewrite their checksums merely to satisfy validation.

## Track A: required before continuation

1. Bring collection/query worktrees onto the current committed interfaces while
   preserving local changes and following D10. Main's compatibility changes are
   absent from the old A/B worktrees; a recorded revision in a launch JSON does
   not pin Python imports. Bind the actual source revision and local source hashes.
2. Derive new execution manifests for **both** pending queues below, preserving
   the originals and binding their hashes. Translate removed CLI flags, preserve
   model keys/input hashes/parents/scientific settings, and record new output paths
   and descriptive result fields. Update forward queue generation as needed;
   never replay frozen commands against unpinned current imports.
3. Reconcile coverage additively from verified artifact mappings. Latest recorded
   coverage is **137 measured / 129 full-weight / 29 ungenerated**, with eight
   upgrades ready (six plus two); this audit did not recalculate coverage.
   Existing one-time refresh scripts assume older counts and are not reusable
   reconcilers. Preserve approved reference mappings and held-out assignments.
4. Build/query existing evidence through explicit artifact inputs or explicitly
   selected historical package conventions. Default builds scan packages, not
   arbitrary run trees; changed convention IDs can exclude old packages. Backfill
   only eligible completed runs. Standalone lab fingerprints can use explicit
   build inputs and need no invented ESD files. Do not remeasure solely for naming.
   Keep stored-view spectra distinct from packaged `esd/metrics.h5` and its layer
   indices; retain `/layers/spectrum_source` and explicit missing-parent/delta status.

Pending queue paths, relative to the workspace data root:

- `runs/track-b-l6-remaining-20261104/execution6/incremental-A-l12-launch.json`
- `runs/track-b-l6-remaining-20261104/world4/future-capacity/execution2/incremental-A-l12-launch.json`

Both contain historical `--tier t2` and `--also_t1_blocks`; use
`--measurement_mode full_weights` and `--also_row_block_sizes` in derived launches.
The corresponding historical generators are `finalize_six.py` and
`finalize_future_two.py` under `runs/track-b-l6-remaining-20261104/`; adapt future generation in new
source/output records rather than editing recorded run evidence in place.

## Track B: required before new extraction/evaluation

1. Reconcile the existing B worktree with current committed Zoo-Keeper code,
   retaining its uncommitted training/merge changes. Do not reimplement query or
   lineage compatibility that already exists on main.
2. Update `lab/cohort_evidence.py` to request `row_blocks`; its removed numbered
   mode currently blocks new extraction, including chain-cohort measurements.
   Keep resume/input-hash checks and preserve existing progress paths as legacy
   inputs. Use descriptive names for newly introduced controls and outputs.
3. Update `lab/scripts/evaluate_delivery.py` together with the producer: use
   canonical modes, support historical/current artifact mappings and companion
   paths, and validate each source's frozen readable version rather than today's
   writer version. Its literal legacy expectations fail even after the current
   loader normalizes historical files. Preserve strict scientific comparisons;
   do not silently mix versions, broaden a frozen cohort, or retune thresholds.
4. Publish a new hash-bound delivery/evaluation mapping for A and reuse completed
   measurements. Leave old delivery columns as explicit legacy input adapters;
   use descriptive names in new schemas. Training recipes, lab truth, root splits
   and generation priorities are unaffected.

## Studies and historical runs: no blanket migration

Inspected nonblind study/reconciliation code consumes legacy tables/HDF5 and
freezes its implementation/input hashes. No current extractor-interface break
was identified there. Leave frozen protocols, scripts captured with results,
checkpoints, run JSON/CSV files and reports unchanged. A new evidence-backed study
must use a new version/output directory and record its source contract. Track C
owns those studies and its blind boundary; no blind outputs were inspected.

## Focused verification and completion criteria

- Query schema/version compatibility plus lineage upstream/report checks:
  **24 passed in 2.03s** (existing tests; CPU/offline).
- Existing tiny cohort resume test: **1 failed in 0.15s**, reproducing the
  unsupported mode at `lab/cohort_evidence.py:181` →
  `evidence/fingerprint.py:650` (`Invalid tier, sketch size, or read thread count`).
  This remains an assigned Track B fix, not a passing integration claim.
- Both pending queue argument arrays reject the removed controls; translated
  arrays parse successfully against current CLI. Parsing only; no launch.
- A acceptance: derived queue arguments parse, pinned inputs/source and completed
  artifact reuse are verified, and a small build/query smoke preserves coverage.
  B acceptance: existing tiny producer/resume check passes; delivery checks cover
  both naming generations, exact historical/current source versions, mismatch
  rejection and unchanged frozen input bytes. No broad benchmark is needed.

No model/GPU jobs, downloads, installs, commits or worktree updates were made.
