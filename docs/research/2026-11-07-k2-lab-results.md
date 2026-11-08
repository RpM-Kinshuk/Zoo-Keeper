# K2 frozen L12 lab comparison, 2026-11-07

Both native-module adaptations scored all existing development/test pairs and
all training pairs. Held-out same-root AUROC is 1.00 for each method. This is
only the small Pythia control cohort, not a new scale or exact-parent result.

| Split | Trace-native answered | Spectral-Signatures-native answered | Same-root AUROC (both) |
| --- | --- | --- | --- |
| development | 30/30 | 30/30 | 1.00 |
| held-out test | 30/30 | 30/30 | 1.00 |
| training | 132/132 | 132/132 | N/A: single-class same-root truth |

Immediate-parent decisions, direction and edge typing are **not applicable**
for these symmetric similarity adaptations. No threshold is fitted; no
threshold-based precision or calibrated error-control claim is made.

Methods: Trace uses exact Frobenius norms from saved full eigenvalues and the
existing Model-Research interpolation/Pearson helper. Spectral Signatures uses
fixed k=floor(n/2) Hill exponents, then mean per-module Spearman correlation.
Native GPTNeoX modules include fused QKV; no fictitious separate Q/K/V spectra
are constructed. SmolLM native projections are retained individually. This is
an explicit architecture adaptation, not a full reproduction of the papers.
The fine-grained Trace subspace component was **not run**.

Inputs are the unchanged L12 t2-k64 development/test universes and training-12
universe, plus the expanded-r6 model index pointing to full-SVD companions.
Every frozen pair, label and eligibility flag was checked against output.
The 166-model final benchmark comparison and verified Hub tier remain pending.

Shared feature read for both methods: 4,847,616 logical eig-array
bytes. File scans for SHA256: 14,961,020 bytes, including model/universe
files; HDF5 metadata/cache/OS read amplification is not instrumented. These are
distinct counters, not an end-to-end physical-I/O total. New model-weight reads:
0. Historical full-spectrum extraction bytes/time: unknown (N/A),
so no per-query cold-read or byte-efficiency conclusion is supported.
Elapsed cached feature/scoring run: 1.997327 seconds; excludes original
extraction and unit tests. Reference amortization is not inferred.

Protocol: `2026-11-07-k2-lab-protocol.md`, written before runs. Formula tests
use 1e-12 float64 tolerance; dataset metrics are descriptive. Ten task tests
pass. One fresh core review found Hill/Trace availability coupling, missing
optional training inputs, and lost rejected-read counters; regressions cover
all three and fixes were applied. v1 is superseded by v2, not deleted.

Run: `$ZOO_DATA/runs/track-c-k2-lab-spectral-20261107-v2/`.
Input manifest SHA256: `fb826cde945ac1b60e341fac58db029bf6b3e8f3701dae8af1c8f36ae0883c3d`.
The manifest binds all data and adapter/Trace-helper source hashes.
Artifacts use paths relative to `$ZOO_DATA`; no weights or raw row blocks copied.

Reproduce from the workspace root (choose a fresh output directory):
```bash
set -a; . ./.env; set +a
conda activate esd_ind
PYTHONPATH=Model-Research:worktrees/Zoo-Keeper-track-c python -m studies.baselines.lab_spectral \
  --data-root "$ZOO_DATA" \
  --model-index "$ZOO_DATA/runs/track-a-expanded-r6-20261105/zoo/models.parquet" \
  --control-root "$ZOO_DATA/runs/track-b-l12-evaluation-v2/controls/t2-k64" \
  --out "$ZOO_DATA/runs/track-c-k2-lab-spectral-new"
```
No installs, commits, shared-interface changes or claim promotions.

Validation handoff: task tests **10 passed**. Full repository suite **444 passed,
2 failed** (107.01s); both failures also reproduce when only
`tests/lineage/test_upstream.py` runs (2 failed, 4 passed), without importing
the new baseline module. The unchanged track-c `lineage/upstream.py:81`
requires tier `t1`; the current Model-Zoo extractor writes `row_blocks`.
`git diff --exit-code HEAD -- lineage/upstream.py` is clean. Failures:
- `test_load_real_fingerprints_without_zoo_and_use_shared_scorer`
- `test_parent_identity_is_preserved_for_delta_matching`
Both raise `ValueError: lineage records currently require T1 fingerprints`.
This inherited cross-checkout compatibility issue remains for integration;
no full-suite pass or merge readiness is claimed. No running processes remain.
Suggested commit message for overseer: `studies: compare native spectral baselines on frozen L12 pairs`.
