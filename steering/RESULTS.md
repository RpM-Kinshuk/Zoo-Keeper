# Results ledger

Every number that may appear on the board, in a paper, or in a README comes
from this table. Tracks add one row per new number; the steward curates.
Paths are relative to `$ZOO_DATA` unless they start with a repository name.

**Status:**
- **tested:** a real run on real data;
- **exploratory:** real data, but weak provenance or a small sample;
- **built:** code only, never run on real data;
- **broken test:** the test itself needs fixing before it means anything.

Seeded on 2026-11-07 from the agents' own records. These numbers have not
been re-run.

## Current outcome summary (Nov 8 scientific-necessity review)

These are existing runs, not a new experiment or broad claim promotion.
Use the dated rows for targets/provenance; old partial inventories and repaired
failures are history, not reasons to repeat completed work.

| Question | Current result against target | Scientific limit / next decision | Source |
| --- | --- | --- | --- |
| Quantization detection/source replay | 12/12 detected/replayed; 0/30 controls flagged, matching written targets | Narrow constructed-quantizer test; no general Hub/order proof | `runs/track-b-k5-quantization-20261108/REPORT.md` |
| Missing source | 12/12 present answers correct; all 12 absent cases abstain, versus ≥50% present coverage and ≤0.05 wrong per answer targets | Only 12/112 lab immediate-source queries tested; absent-case precision undefined with no answers | `runs/track-b-k5-parent-out-20261108/REPORT.md`; joined report |
| Known-source merge proportions | 18/18 within 0.05 coefficient error; 0 misses versus ≤4 allowed | Unknown-source/order/Hub questions open | `runs/track-b-k3-slerp-trig-20261107-v2/REPORT.md` |
| Broader edit/source discrimination | Scoped measurements on 90/112 children; 24/24 adapter support sets ambiguous; sampled ranks resolve 8/24 adapters and no refusal writers; merge shares overlap controls | Measurement is not detector success. D7 flags isolated-mark sufficiency; develop only decision-changing rules/controls | `runs/track-b-k5-joined-20261108-v2/REPORT.md` |
| Published comparisons | Residual and modelDNA cover 126/126 quantization pairs; source uniquely top 7/12 for residual; no definite modelDNA source answers | Same-family relatives are not independent negatives. Remaining edits/real cases/witness direction open | Track C `docs/research/2026-11-08-k2-matched-results.md` |
| Same-design unrelated controls | 30 trained BERT models versus ≥30 count target, 0/435 observed false matches versus 0 target | Initialization, positive recall and decoder transfer unresolved; diagnostic stays closed in scope | `runs/track-a-k6-bert-20261108/REPORT.md`; `REVIEW.md` |
| Is new collection necessary? | Selected roster has 0/216 documented exact-positive query matches and 0/134 known parents; saved anchors yield 226/226 finite comparisons covering 134 queries before dedup | Missing exact keys are not negative labels. Evaluate saved artifacts first; collection/storage deferred until a necessary information gap exists | `runs/track-a-k8-readiness-20261108/REPORT.md` |
| Practical workflow | Saved info/similar/family/check demo retained with explicit missing-base abstention | Artifact-only timing is not full fresh-model cost; no rerun needed | `runs/track-a-workflow-20261108/REPORT.md` |

No new verified real-model construction edge or complete Hub detector verdict.
K0 archive and K10 hash audit remain closed in recorded scope. Four GPUs are a
shared ceiling; current B/C evidence used CPU because driver access failed.
The previous large-pool byte/storage forecasts are historical for an unapproved
scenario, not requirements for the existing-artifact evaluation.

## Dated measurements

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-02 | B | Same family or not, on held-out families (only 2 families, 6 queries) | AUROC 1.00, partial and full reads | ≥0.98 | pass, but tiny sample | tested | `Zoo-Keeper/docs/l12-evaluation.md` |
| 11-02 | B | Do partial reads rank related pairs like full reads? | Spearman 0.94 (24 pairs) | ≥0.9 | pass | tested | same |
| 11-05 | A | Wrong "immediate parent" answers per query (pilot + lab; labels incomplete) | 8.28 | ≤0.05 | fail | tested | `runs/track-a-expanded-r6-20261105/G0-development-report.md` |
| 11-02 | B | Which model came first (direction) | 0% coverage: never attempted | — | no result | built | `Zoo-Keeper/docs/l12-evaluation.md` |
| 11-05 | A | Partial-read spectra vs full-read spectra (alpha) | median gap 0.43 | ≤0.1 | fail: spectra need full reads | tested | `runs/track-a-expanded-r6-20261105/REPORT.md` |
| 11-05 | B | Tree test: share of model quartets breaking the tree rule | 82–96% | pivot if >20% | test used a near-exact tolerance with no noise model | broken test | `runs/track-b-e6-expanded-20261105/` |
| 11-03 | B | Merge recipe fits (30 lab merges) | 30/30 refused to answer | ±0.05 coefficient error | design fitted raw weights, whose columns are about 99.9% identical | broken test | `runs/track-b-formal-qwen-merges-v1/REPORT-v4.md` |
| 11-05 | B | Disguise transforms that keep the model's behavior (permutation, rotation) | 0/8 kept behavior | — | could not be built | tested | `Zoo-Keeper/docs/plans/2026-11-05-public-transform-equivalence-v2.md` |
| 10-31 | B | Edge typing (LoRA vs full vs quantization vs merge vs ablation) and the ablation detector | — | — | never run on the lab | built | `Zoo-Keeper/lineage/typing.py` |
| 10-31 | A | Bytes read by a partial read, SmolLM2-135M | 27% | ≤10% median per stratum | fail on a tiny model; per-stratum numbers not yet reported | tested | handoff STATUS, X6 smoke |
| 11-05 | A | Calibration check on held-out independent pairs | 0 false alarms in 15 pairs | — | too small to conclude | tested | `runs/track-a-roster-v2-calibration-20261105/R5-report.md` |
| 11-05 | A | Leaderboard scores whose commit matches the measured commit | 24/150 | — | low (fix: PLAN K10) | tested | `runs/track-a-expanded-r6-20261105/REPORT.md` |
| 11-07 | C | Published lineage baselines run on the lab | none | all four priority baselines | not started | — | — |
| 10-31 | C | Heritability, strict protocol | 0 estimates (0/3,288 scores verified, no sampling variance) | — | blocked by over-strict rules (PLAN K11) | tested | `Zoo-Keeper/docs/papers/heritability-workshop.md` |
| 11-01 | C | Benchmark questions that weight features are worth as a score predictor (old, unpinned data) | about 3–39 questions; within a family, weights did worse on 4/4 yes/no tasks | — | sobering, useful | exploratory | `studies/heritability/legacy-first-pass-v1/report.md` |
| 11-05 | A | Coverage | lab 166/166 measured; pilot 284/287 full reads, 97/97 partial; zoo snapshot 703 models | — | infrastructure works | tested | `runs/track-a-expanded-r6-20261105/` |

| 11-07 | B | K4 olmo-1b/row_blocks_k64: Numerical repeat maximum distance difference | 0 | 0 | repeat matches | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/row_blocks_k64: Numerical-only quartet tolerance | 8.34470651e-14 | 4e + 64 eps × max(1,max distance) | diagnostic only | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/row_blocks_k64: Four-point absolute gap median | 6.22520206e-05 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/row_blocks_k64: Four-point absolute gap p95 | 0.000324223133 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/row_blocks_k64: Four-point absolute gap maximum | 0.000602258806 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/row_blocks_k64: Real quartets above numerical-only floor | 100% | >20% diagnostic flag | flag; extraction noise unmeasured | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/row_blocks_k64: Matched exact-tree quartets above numerical-only floor | 0% | 0% | control passes | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/full_weights_k128: Numerical repeat maximum distance difference | 0 | 0 | repeat matches | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/full_weights_k128: Numerical-only quartet tolerance | 3.6524611e-13 | 4e + 64 eps × max(1,max distance) | diagnostic only | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/full_weights_k128: Four-point absolute gap median | 0.000209159285 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/full_weights_k128: Four-point absolute gap p95 | 0.00111840172 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/full_weights_k128: Four-point absolute gap maximum | 0.00217738069 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/full_weights_k128: Real quartets above numerical-only floor | 100% | >20% diagnostic flag | flag; extraction noise unmeasured | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 olmo-1b/full_weights_k128: Matched exact-tree quartets above numerical-only floor | 0% | 0% | control passes | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/row_blocks_k64: Numerical repeat maximum distance difference | 0 | 0 | repeat matches | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/row_blocks_k64: Numerical-only quartet tolerance | 2.30569918e-14 | 4e + 64 eps × max(1,max distance) | diagnostic only | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/row_blocks_k64: Four-point absolute gap median | 3.01220558e-05 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/row_blocks_k64: Four-point absolute gap p95 | 0.000355323576 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/row_blocks_k64: Four-point absolute gap maximum | 0.000900891916 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/row_blocks_k64: Real quartets above numerical-only floor | 100% | >20% diagnostic flag | flag; extraction noise unmeasured | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/row_blocks_k64: Matched exact-tree quartets above numerical-only floor | 0% | 0% | control passes | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/full_weights_k128: Numerical repeat maximum distance difference | 0 | 0 | repeat matches | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/full_weights_k128: Numerical-only quartet tolerance | 9.44469572e-14 | 4e + 64 eps × max(1,max distance) | diagnostic only | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/full_weights_k128: Four-point absolute gap median | 0.000136137685 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/full_weights_k128: Four-point absolute gap p95 | 0.000845259435 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/full_weights_k128: Four-point absolute gap maximum | 0.0029598633 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/full_weights_k128: Real quartets above numerical-only floor | 100% | >20% diagnostic flag | flag; extraction noise unmeasured | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q25-05/full_weights_k128: Matched exact-tree quartets above numerical-only floor | 0% | 0% | control passes | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/row_blocks_k64: Numerical repeat maximum distance difference | 0 | 0 | repeat matches | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/row_blocks_k64: Numerical-only quartet tolerance | 3.74546056e-14 | 4e + 64 eps × max(1,max distance) | diagnostic only | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/row_blocks_k64: Four-point absolute gap median | 2.88163166e-05 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/row_blocks_k64: Four-point absolute gap p95 | 0.000353257632 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/row_blocks_k64: Four-point absolute gap maximum | 0.00079155701 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/row_blocks_k64: Real quartets above numerical-only floor | 100% | >20% diagnostic flag | flag; extraction noise unmeasured | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/row_blocks_k64: Matched exact-tree quartets above numerical-only floor | 0% | 0% | control passes | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/full_weights_k128: Numerical repeat maximum distance difference | 0 | 0 | repeat matches | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/full_weights_k128: Numerical-only quartet tolerance | 1.46632487e-13 | 4e + 64 eps × max(1,max distance) | diagnostic only | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/full_weights_k128: Four-point absolute gap median | 8.71666124e-05 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/full_weights_k128: Four-point absolute gap p95 | 0.000983833151 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/full_weights_k128: Four-point absolute gap maximum | 0.00291651193 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/full_weights_k128: Real quartets above numerical-only floor | 100% | >20% diagnostic flag | flag; extraction noise unmeasured | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 q3-06/full_weights_k128: Matched exact-tree quartets above numerical-only floor | 0% | 0% | control passes | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/row_blocks_k64: Numerical repeat maximum distance difference | 0 | 0 | repeat matches | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/row_blocks_k64: Numerical-only quartet tolerance | 1.47471701e-12 | 4e + 64 eps × max(1,max distance) | diagnostic only | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/row_blocks_k64: Four-point absolute gap median | 0.000209605485 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/row_blocks_k64: Four-point absolute gap p95 | 0.00555908301 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/row_blocks_k64: Four-point absolute gap maximum | 0.00983539207 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/row_blocks_k64: Real quartets above numerical-only floor | 100% | >20% diagnostic flag | flag; extraction noise unmeasured | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/row_blocks_k64: Matched exact-tree quartets above numerical-only floor | 0% | 0% | control passes | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/full_weights_k128: Numerical repeat maximum distance difference | 0 | 0 | repeat matches | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/full_weights_k128: Numerical-only quartet tolerance | 5.91748206e-12 | 4e + 64 eps × max(1,max distance) | diagnostic only | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/full_weights_k128: Four-point absolute gap median | 0.000692173173 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/full_weights_k128: Four-point absolute gap p95 | 0.0302232889 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/full_weights_k128: Four-point absolute gap maximum | 0.0515036648 | compare numerical tolerance | descriptive | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/full_weights_k128: Real quartets above numerical-only floor | 100% | >20% diagnostic flag | flag; extraction noise unmeasured | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |
| 11-07 | B | K4 smol-360/full_weights_k128: Matched exact-tree quartets above numerical-only floor | 0% | 0% | control passes | exploratory | `runs/track-b-k4-numerical-repeat-20261107/results.json` |

| 11-07 | B | K3 linear coefficient cases | 9/9 | error ≤0.05 | pass | exploratory | `runs/track-b-k3-deltas-20261107/REPORT.md` |
| 11-07 | B | K3 maximum linear coefficient error | 1.819184380913441e-08 | ≤0.05 | pass | exploratory | `runs/track-b-k3-deltas-20261107/REPORT.md` |
| 11-07 | B | K3 approximate SLERP interpolation cases | 9/9 | error ≤0.05 | passes approximation; exact angle/fallback pending | exploratory | `runs/track-b-k3-deltas-20261107/REPORT.md` |
| 11-07 | B | K3 maximum approximate SLERP interpolation error | 0.00019421971458799536 | ≤0.05 | passes approximation only | exploratory | `runs/track-b-k3-deltas-20261107/REPORT.md` |
| 11-07 | B | K3 combined linear/approximate SLERP misses | 0 | ≤4 | diagnostic passes; exact SLERP pending | exploratory | `runs/track-b-k3-deltas-20261107/REPORT.md` |
| 11-07 | B | K3 numerical full-rank delta fits | 30/30 | report rank; exact duplicate abstention only | all numerically fit | exploratory | `runs/track-b-k3-deltas-20261107/REPORT.md` |
| 11-07 | B | K14 olmo-1b-laundering-mlp_permutation-drift-v2 source cosine | 0.99999856319036351 | strict highest among compatible roots | only compatible root; no competitive retrieval evidence | exploratory | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 olmo-1b-laundering-mlp_permutation-drift-v2 fp32 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 olmo-1b-laundering-mlp_permutation-drift-v2 bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 olmo-1b-laundering-mlp_permutation-drift-v2 reloaded_bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 olmo-1b-laundering-value_rotation-drift-v2 source cosine | 0.99999858976694778 | strict highest among compatible roots | only compatible root; no competitive retrieval evidence | exploratory | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 olmo-1b-laundering-value_rotation-drift-v2 fp32 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 olmo-1b-laundering-value_rotation-drift-v2 bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 olmo-1b-laundering-value_rotation-drift-v2 reloaded_bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q25-05-laundering-mlp_permutation-drift-v2 source cosine | 0.99999999999999989 | strict highest among compatible roots | only compatible root; no competitive retrieval evidence | exploratory | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q25-05-laundering-mlp_permutation-drift-v2 fp32 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q25-05-laundering-mlp_permutation-drift-v2 bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q25-05-laundering-mlp_permutation-drift-v2 reloaded_bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q25-05-laundering-value_rotation-drift-v2 source cosine | 1 | strict highest among compatible roots | only compatible root; no competitive retrieval evidence | exploratory | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q25-05-laundering-value_rotation-drift-v2 fp32 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q25-05-laundering-value_rotation-drift-v2 bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q25-05-laundering-value_rotation-drift-v2 reloaded_bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q3-06-laundering-mlp_permutation-drift-v2 source cosine | 0.99999999999999989 | strict highest among compatible roots | only compatible root; no competitive retrieval evidence | exploratory | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q3-06-laundering-mlp_permutation-drift-v2 fp32 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q3-06-laundering-mlp_permutation-drift-v2 bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q3-06-laundering-mlp_permutation-drift-v2 reloaded_bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q3-06-laundering-value_rotation-drift-v2 source cosine | 1 | strict highest among compatible roots | only compatible root; no competitive retrieval evidence | exploratory | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q3-06-laundering-value_rotation-drift-v2 fp32 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q3-06-laundering-value_rotation-drift-v2 bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 q3-06-laundering-value_rotation-drift-v2 reloaded_bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 smol-360-laundering-mlp_permutation-drift-v2 source cosine | 0.99999999999999989 | strict highest among compatible roots | only compatible root; no competitive retrieval evidence | exploratory | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 smol-360-laundering-mlp_permutation-drift-v2 fp32 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 smol-360-laundering-mlp_permutation-drift-v2 bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 smol-360-laundering-mlp_permutation-drift-v2 reloaded_bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 smol-360-laundering-value_rotation-drift-v2 source cosine | 1 | strict highest among compatible roots | only compatible root; no competitive retrieval evidence | exploratory | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 smol-360-laundering-value_rotation-drift-v2 fp32 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 smol-360-laundering-value_rotation-drift-v2 bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 smol-360-laundering-value_rotation-drift-v2 reloaded_bfloat16 behavior check | FAIL | original frozen allclose | failure unchanged; rounding-only explanation unestablished | tested | `runs/track-b-k14-20261107/results.json` |
| 11-07 | B | K14 exact MLP reordering: inverse indexing | exact | bit equality | pass: algebraic construction only | tested | `runs/track-b-k14-20261107/single-layer.json` |
| 11-07 | B | K14 reordered down-projection relative sketch drift | 1.413877034409836 | 3.814697265625e-06 | invariance fails; tensor only | tested | `runs/track-b-k14-20261107/single-layer.json` |
| 11-07 | B | K14 reordered down-projection sketch cosine | -0.0026966164884539372 | descriptive | tensor only; full-network match unmeasured | tested | `runs/track-b-k14-20261107/single-layer.json` |

| 11-07 | B | K3 SLERP children passing uniform linear endpoint reproduction | 6/9 | all tensors within declared float32 allowance | linear assumption fails for Qwen2.5; preserved negative result | tested | `runs/track-b-k3-slerp-endpoints-20261107/results.json` |
| 11-07 | B | K3 SLERP tensors failing linear endpoint reproduction | 9 | 0 | fail; test exact trigonometric relation next | tested | `runs/track-b-k3-slerp-endpoints-20261107/results.json` |
| 11-07 | B | K3 exact SLERP coefficient discrepancy smol-360-merge-slerp-025 | 9.966322989107823e-09 | ≤0.05 | pass on original fitted views | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 exact SLERP coefficient discrepancy smol-360-merge-slerp-05 | 7.1502811449519754e-09 | ≤0.05 | pass on original fitted views | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 exact SLERP coefficient discrepancy smol-360-merge-slerp-layerwise | 1.4937319342545052e-06 | ≤0.05 | pass on original fitted views | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 exact SLERP coefficient discrepancy q25-05-merge-slerp-025 | 0.00013777845570406289 | ≤0.05 | pass on original fitted views | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 exact SLERP coefficient discrepancy q25-05-merge-slerp-05 | 0.00016671507095078475 | ≤0.05 | pass on original fitted views | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 exact SLERP coefficient discrepancy q25-05-merge-slerp-layerwise | 0.00019421971458799536 | ≤0.05 | pass on original fitted views | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 exact SLERP coefficient discrepancy q3-06-merge-slerp-025 | 1.2593303733154926e-08 | ≤0.05 | pass on original fitted views | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 exact SLERP coefficient discrepancy q3-06-merge-slerp-05 | 1.81924631470487e-08 | ≤0.05 | pass on original fitted views | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 exact SLERP coefficient discrepancy q3-06-merge-slerp-layerwise | 5.965521517325989e-07 | ≤0.05 | pass on original fitted views | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 exact SLERP cases meeting coefficient target | 9/9 | ≤0.05 per case | pass | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 previously failed tensors reproduced by trigonometric SLERP | 9/9 | same float32 allowance | pass; historical source bytes not verified | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K3 combined linear and exact SLERP coefficient misses | 0/18 | ≤4 misses | pass on lab sketches; no Hub or claim promotion | tested | `runs/track-b-k3-slerp-trig-20261107-v2/results.json` |
| 11-07 | B | K4 fresh-read Smol360 models | 14 | 14 existing tree nodes | scope only | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |
| 11-07 | B | K4 fresh-read common views | 225 | same aligned views | scope only | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |
| 11-07 | B | K4 fresh-read same-backend distance noise | 0.0 | measure before tolerance | exact repeat | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |
| 11-07 | B | K4 original extraction vs CPU distance sensitivity | 0.0 | measure before tolerance | exact agreement | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |
| 11-07 | B | K4 fresh-read absolute quartet tolerance | 1.474717005503261e-12 | 4e plus float64 allowance | predeclared formula | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |
| 11-07 | B | K4 fresh-read absolute gap median | 0.00020960548466231188 | compare measured allowance | descriptive | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |
| 11-07 | B | K4 fresh-read absolute gap p95 | 0.005559083008691346 | compare measured allowance | descriptive | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |
| 11-07 | B | K4 fresh-read absolute gap maximum | 0.009835392073625826 | compare measured allowance | descriptive | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |
| 11-07 | B | K4 fresh-read Smol360 real quartet violations | 1001/1001 | >20% diagnostic flag | flag; not an ancestry conclusion | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |
| 11-07 | B | K4 fresh-read matched exact-tree quartet violations | 0/1001 | 0 | control passes at measured zero noise; nonzero-noise power untested | tested | `runs/track-b-k4-extraction-repeat-20261107/results.json` |

| 11-08 | B | K5/C2 quantization value-grid detections | 12/12 | 12/12 | PASS; lab only | tested | `runs/track-b-k5-quantization-20261108/REPORT.md` |
| 11-08 | B | K5/C2 quantization grid flags on SFT/DPO/LoRA controls | 0/30 | 0/30 | PASS; lab only | tested | `runs/track-b-k5-quantization-20261108/REPORT.md` |
| 11-08 | B | K5/C2 known parent replay bit-exact | 12/12 | 12/12 | PASS; lab only | tested | `runs/track-b-k5-quantization-20261108/REPORT.md` |
| 11-08 | B | K5/C2 incomplete quantization/control cases | 0 | 0 | none; lab only | tested | `runs/track-b-k5-quantization-20261108/REPORT.md` |
| 11-08 | B | K5/C2 complete child/parent tensor-name and shape coverage | 12/12 | 12/12 | PASS; original child weight pins also match; lab only | tested | `runs/track-b-k5-quantization-20261108/REPORT.md` |

| 11-08 | B | Existing quantizer unit tests after K5 replay audit | 29 passed | no failures | PASS; one NVML warning | tested | `runs/track-b-k5-quantization-20261108/verification.json` |

| 11-08 | B | K5 quantization parent-present answers | 12/12 | ≥50% coverage | PASS subset | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |
| 11-08 | B | K5 quantization parent-present wrong answers | 0/12 answers | ≤0.05 per answer | PASS subset | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |
| 11-08 | B | K5 quantization parent-absent answers | 0/12 queries | abstain when source absent | all abstain; precision undefined | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |
| 11-08 | B | K5 quantization parent-absent false named parents | 0 | no false parent | no answered queries | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |
| 11-08 | B | K5 quantization pooled false parent rate | 0/12 answers | ≤0.05 | PASS empirical subset only | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |
| 11-08 | B | K5 quantization parent-present coverage | 100% | ≥50% | PASS subset | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |
| 11-08 | B | K5 quantization incomplete parent-out queries | 0 | 0 | complete | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |
| 11-08 | B | K5 parent-out effective tensor coverage | 126/126 candidate pairs | all complete | PASS incl tied aliases | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |
| 11-08 | B | K5 parent-out distinct frozen identities | 46 | ≤100 weight models | within small-run limit | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |
| 11-08 | B | K5 parent-out weight pins matching original delivery | 50 files | all selected files | PASS | tested | `runs/track-b-k5-parent-out-20261108/REPORT.md` |

| 11-08 | B | Initial main integration test passes | 648 | all pass | 20 environment failures | tested | `runs/track-b-k5-parent-out-20261108/verification.json` |
| 11-08 | B | Initial main integration test failures | 20 | 0 | research import path, metadata, Gloo environment | tested | `runs/track-b-k5-parent-out-20261108/verification.json` |
| 11-08 | B | Environment-corrected main integration test passes | 664 | all pass | four socket-restricted cases remain | tested | `runs/track-b-k5-parent-out-20261108/verification.json` |
| 11-08 | B | Environment-corrected main integration test failures | 4 | 0 | sandbox Gloo restriction; verified on separate rerun | tested | `runs/track-b-k5-parent-out-20261108/verification.json` |
| 11-08 | B | Distributed-training file outside socket sandbox | 19 passed | 0 failures | PASS; covers all four preceding failures | tested | `runs/track-b-k5-parent-out-20261108/verification.json` |

## Steward verification: 2026-11-07

No new track result was supplied. Fresh skeptic spot check: **matched** the
11-02 related-pair Spearman row, 0.9370629370629371 (rounds to 0.94), against
its pre-existing target ≥0.9. There were 24 related ordered pairs among 60
joined pairs. This rechecks stored predictions, not extraction or a claim's
full test; no ladder promotion.

Source: `/jumbo/yaoqingyang/kinshuk/zoo/runs/track-b-l12-evaluation-v2/controls/`, modes `t1-b256-k64` and
`t2-k64`, development and test predictions plus each T1 `universe.parquet`.
The independent checker joined query/candidate keys, selected same-root
truth, and recomputed Spearman under `esd_ind`; no model run was launched.
Aggregate input SHA256: `71cc36dcfed36f800333cdd2c56f3d3ce7c3114b9ed4d75165c6db0ad10f8be7`.
Hash construction: development then test; within each, T1 predictions, T2
predictions, T1 universe; concatenate relative path, NUL, binary file SHA256,
then hash the concatenation.
Skeptic: agrees with unchanged claims; this checks arithmetic only. Keep
lead acknowledgements pending and do not infer historical frozen-file
adherence from today's snapshot.

## Track A — K0, 2026-11-07

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-07 | A | At-risk temporary checkpoints with a verified gated remote copy | 13/13 | 13/13 | pass | tested | `runs/track-a-k0-20261107/remote-verification.json` |
| 11-07 | A | At-risk checkpoint files uploaded and verified by remote list and size | 81/81 | 81/81 | pass | tested | same; `RpM-Kinshuk/zoo-lab@61b01d947a27a197cc8e63627a6285558d66ecde` |
| 11-07 | A | Actual temporary checkpoint storage uploaded | 23,803,007,860 bytes | replace 23.8 GB estimate | measured | tested | `runs/track-a-k0-20261107/inventory.json` |
| 11-07 | A | Uploaded safetensors whose remote SHA256 equals the local hash | 17/17 | 17/17 | pass | tested | `runs/track-a-k0-20261107/remote-verification.json` |
| 11-07 | A | Authorized storage repositories public with manual approval gating | 3/3 | 3/3 | pass after applying required gating | tested | `runs/track-a-k0-20261107/authorization.md`; remote metadata checked before upload |

## Track C primary-source check, 2026-11-07 (literature, not reproduced)

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-07 | C | MPS candidate-pool sizes in published coverage/set-size curves | 5, 10, 20, 50, 80, 100 | locate full range | primary-source check pass; curve novelty already overlaps | literature; author-reported | `Zoo-Keeper` track-c `docs/research/2026-11-07-k2-primary-check.md`; MPS Fig. 2/App. F; input manifest bound there |
| 11-07 | C | MPS benchmark population with declared Hub lineage | 455 models | check generic benchmark-size comparison | inherited size comparison needs correction; exact-truth distinction unresolved | literature; author-reported | same report; MPS Table 1/App. D; same input manifest |
| 11-07 | C | GGUF recorded-input independent verifiability | 0/198 repositories | distinguish exact re-derivation from similarity | missing build inputs prevent this protocol; not proof of absent parentage | literature; author-reported | same report; GGUF §IV; same input manifest |

## Track A — K6 and K8, 2026-11-07

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-07 | A | K6 independent models with pinned existing partial fingerprints | 45 | — | stratified evidence only | tested | `runs/track-a-k6-20261107/result.json` |
| 11-07 | A | K6 largest group sharing the stored architecture/config | 9 | ≥30 | incomplete strict single-config target | tested | `same` |
| 11-07 | A | K6 false family matches on existing independent pairs (165 fit pairs included) | 0/180 | 0 | pass diagnostic; reused fitting evidence | tested | `same` |
| 11-07 | A | K6 false family matches on held-out independent pairs | 0/15 | 0 | pass small held-out diagnostic; no claim promotion | tested | `same` |
| 11-07 | A | K8 completed standalone partial-read pilot measurements | 97/129 | all eligible requests accounted for | pass accounting; 32 exclusions retained | tested | `runs/track-a-k8-20261107/REPORT.md` |
| 11-07 | A | K8 total requested tensor payload in completed pilot | 50,693,801,632 bytes | — | measured extraction payload, not network traffic | tested | `same` |
| 11-07 | A | K8 header-verified 7.5–8.5B standalone measurements | 9 | — | existing examples, mixed runtime settings | tested | `same` |
| 11-07 | A | K8 extraction payload for header-verified 8B-range models, median | 641,728,512 bytes | — | measured; search latency excluded | tested | `same` |
| 11-07 | A | K8 extraction seconds for header-verified 8B-range models, median | 849.98 seconds | — | measured historically, mixed hardware | tested | `same` |
| 11-07 | A | K8 extraction seconds for header-verified 8B-range models, p90 | 936.10 seconds | — | measured historically | tested | `same` |
| 11-07 | A | K8 comparable standalone 70B extraction cost | NOT MEASURED | new 70B model cost | incomplete | tested | `same` |
| 11-07 | A | K8 <1B: completed models | 32 models | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 <1B: median requested tensor payload | 87,297,024.00 bytes | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 <1B: median extraction time | 127.53 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 <1B: p90 extraction time | 322.32 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 1-3B: completed models | 22 models | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 1-3B: median requested tensor payload | 261,357,568.00 bytes | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 1-3B: median extraction time | 310.76 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 1-3B: p90 extraction time | 447.91 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 3-9B: completed models | 34 models | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 3-9B: median requested tensor payload | 621,170,688.00 bytes | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 3-9B: median extraction time | 512.57 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 3-9B: p90 extraction time | 923.32 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 9-35B: completed models | 8 models | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 9-35B: median requested tensor payload | 959,508,480.00 bytes | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 9-35B: median extraction time | 192.19 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 9-35B: p90 extraction time | 585.05 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 35-75B: completed models | 0 models | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 >=75B: completed models | 0 models | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 unknown: completed models | 1 models | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 unknown: median requested tensor payload | 103638400 bytes | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 unknown: median extraction time | 211.66 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 unknown: p90 extraction time | 211.66 seconds | descriptive cost table | measured historical pilot; mixed runtimes | tested | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 PREDICTION: 5k with completed-pilot mixture, requested payload | 2,613,082,558,350.52 bytes | no launch budget established | conditional prediction; excludes gaps/retries/search/storage | exploratory | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K8 PREDICTION: 5k with completed-pilot mixture, serial extraction | 492.27 model-hours | no launch budget established | conditional prediction; excludes gaps/retries/search/storage | exploratory | `runs/track-a-k8-20261107/cost-report.json` |
| 11-07 | A | K0 prepared primary training/adapter batch: models | 45 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-irreplaceable.json` |
| 11-07 | A | K0 prepared primary training/adapter batch: files | 338 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-irreplaceable.json` |
| 11-07 | A | K0 prepared primary training/adapter batch: bytes | 66,935,465,778 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-irreplaceable.json` |
| 11-07 | A | K0 prepared other formal constructed outputs: models | 54 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-rebuildable.json` |
| 11-07 | A | K0 prepared other formal constructed outputs: files | 345 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-rebuildable.json` |
| 11-07 | A | K0 prepared other formal constructed outputs: bytes | 95,996,397,722 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-rebuildable.json` |
| 11-07 | A | K0 prepared auxiliary nonformal outputs: models | 27 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-nonformal.json` |
| 11-07 | A | K0 prepared auxiliary nonformal outputs: files | 196 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-nonformal.json` |
| 11-07 | A | K0 prepared auxiliary nonformal outputs: bytes | 5,983,139,358 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-nonformal.json` |
| 11-07 | A | K0 prepared permitted lab evidence archive: models | 112 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-evidence.json` |
| 11-07 | A | K0 prepared permitted lab evidence archive: files | 448 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-evidence.json` |
| 11-07 | A | K0 prepared permitted lab evidence archive: bytes | 2,848,505,527 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-evidence.json` |
| 11-07 | A | K0 prepared filtered lab database snapshot: models | 112 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-db.json` |
| 11-07 | A | K0 prepared filtered lab database snapshot: files | 18 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-db.json` |
| 11-07 | A | K0 prepared filtered lab database snapshot: bytes | 8,654,697 | inventory before upload | locally verified; NOT remotely durable | tested | `runs/track-a-k0-20261107/inventory-db.json` |
| 11-07 | A | K10 score joins verified by full weight-file hashes | 144/150 (96%) | verify exact inventories at both commits | pass for 144; 6 unknown | tested | `runs/track-a-k10-20261107/summary.json`; `validation.json` |
| 11-07 | A | K10 newly verified score joins beyond exact commit matches | 120 | — | measured improvement | tested | `runs/track-a-k10-20261107/summary.json`; `validation.json` |
| 11-07 | A | K10 scored models with verified weight identity | 21/22 | all source revisions available | one model unknown: missing leaderboard revision | tested | `runs/track-a-k10-20261107/summary.json`; `validation.json` |
| 11-07 | A | K10 score rows with missing source revision | 6 | 0 unknown | incomplete coverage, never marked verified | tested | `runs/track-a-k10-20261107/summary.json`; `validation.json` |
| 11-07 | A | K10 weight payload downloaded | 0 bytes | 0 for metadata-only verification | pass | tested | `runs/track-a-k10-20261107/summary.json`; `validation.json` |
| 11-07 | A | K0 total constructed checkpoint storage inventoried | 192,718,010,718 bytes | replace estimated checkpoint footprint | measured; only temporary batch remote | tested | `runs/track-a-k0-20261107/inventory*.json`; `REPORT.md` |
| 11-07 | A | K0 remaining constructed checkpoint models | 126 | all remaining local constructed outputs inventoried | prepared, not uploaded | tested | `runs/track-a-k0-20261107/inventory*.json`; `REPORT.md` |
| 11-07 | A | K0 remaining constructed checkpoint storage | 168,915,002,858 bytes | inventory before upload | prepared, not uploaded | tested | `runs/track-a-k0-20261107/inventory*.json`; `REPORT.md` |
| 11-07 | A | K0 filtered database models covered by retrieval indexes | 111/112 | no unpermitted identities | singleton q25-15 architecture intentionally omitted by existing builder | tested | `runs/track-a-k0-20261107/inventory*.json`; `REPORT.md` |

## Track A — authorized K0 continuation, 2026-11-07

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-07 | A | K0 Primary trained/adapter models verified remotely | 45/45 | 45/45 | pass | tested | `runs/track-a-k0-20261107/remote-verification-irreplaceable.json`; `RpM-Kinshuk/zoo-lab@af09467e7749661df781ea5b180a860f861f6143` |
| 11-07 | A | K0 Primary trained/adapter files verified by remote list and size | 338/338 | 338/338 | pass | tested | `runs/track-a-k0-20261107/remote-verification-irreplaceable.json`; `RpM-Kinshuk/zoo-lab@af09467e7749661df781ea5b180a860f861f6143` |
| 11-07 | A | K0 Primary trained/adapter storage uploaded | 66,935,465,778 bytes | inventory match | pass | tested | `runs/track-a-k0-20261107/remote-verification-irreplaceable.json`; `RpM-Kinshuk/zoo-lab@af09467e7749661df781ea5b180a860f861f6143` |
| 11-07 | A | K0 Primary trained/adapter safetensors hashes verified remotely | 69/69 | 69/69 | pass | tested | `runs/track-a-k0-20261107/remote-verification-irreplaceable.json`; `RpM-Kinshuk/zoo-lab@af09467e7749661df781ea5b180a860f861f6143` |
| 11-07 | A | K0 Other formal constructed models verified remotely | 54/54 | 54/54 | pass | tested | `runs/track-a-k0-20261107/remote-verification-rebuildable.json`; `RpM-Kinshuk/zoo-lab@4b22aab3a8b972d89bb32bc3af314074730c67b0` |
| 11-07 | A | K0 Other formal constructed files verified by remote list and size | 345/345 | 345/345 | pass | tested | `runs/track-a-k0-20261107/remote-verification-rebuildable.json`; `RpM-Kinshuk/zoo-lab@4b22aab3a8b972d89bb32bc3af314074730c67b0` |
| 11-07 | A | K0 Other formal constructed storage uploaded | 95,996,397,722 bytes | inventory match | pass | tested | `runs/track-a-k0-20261107/remote-verification-rebuildable.json`; `RpM-Kinshuk/zoo-lab@4b22aab3a8b972d89bb32bc3af314074730c67b0` |
| 11-07 | A | K0 Other formal constructed safetensors hashes verified remotely | 61/61 | 61/61 | pass | tested | `runs/track-a-k0-20261107/remote-verification-rebuildable.json`; `RpM-Kinshuk/zoo-lab@4b22aab3a8b972d89bb32bc3af314074730c67b0` |

## Track C K2 frozen L12 comparison (native-module adaptations)

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-07 | C | Trace-native development same-root AUROC | 1.00 | descriptive; no new gate | measured; small control cohort | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | Spectral-Signatures-native development same-root AUROC | 1.00 | descriptive; no new gate | measured; small control cohort | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | Trace-native test same-root AUROC | 1.00 | descriptive; no new gate | measured; small control cohort | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | Spectral-Signatures-native test same-root AUROC | 1.00 | descriptive; no new gate | measured; small control cohort | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | Frozen development pairs scored by each native spectral adaptation | 30/30 | preserve all frozen pairs | pass | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | Frozen test pairs scored by each native spectral adaptation | 30/30 | preserve all frozen pairs | pass | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | Frozen training pairs scored by each native spectral adaptation | 132/132 | preserve all frozen pairs | pass | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | Shared logical eig-array bytes read | 4847616 | report shared payload | measured; physical IO/extraction costs unavailable | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | Data-file bytes scanned for SHA256 | 14961020 | report separately from payload | measured; physical IO/extraction costs unavailable | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | New model-weight bytes read | 0 | reuse existing full spectra | measured; physical IO/extraction costs unavailable | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | C | Cached two-method feature/scoring elapsed seconds | 1.9973272180068307 | descriptive; extraction excluded | measured; physical IO/extraction costs unavailable | tested | `runs/track-c-k2-lab-spectral-20261107-v2/report.json`; bound inputs in `inputs.json`; Track C `docs/research/2026-11-07-k2-lab-results.md` |
| 11-07 | A | K0 Auxiliary nonformal outputs: verified model identities | 27 | inventory match | pass | tested | `runs/track-a-k0-20261107/remote-verification-nonformal.json`; `RpM-Kinshuk/zoo-lab@e2039a33e8fed85264fc4de85bc068bd805af226` |
| 11-07 | A | K0 Auxiliary nonformal outputs: verified files | 196 | file-list and size match | pass | tested | `runs/track-a-k0-20261107/remote-verification-nonformal.json`; `RpM-Kinshuk/zoo-lab@e2039a33e8fed85264fc4de85bc068bd805af226` |
| 11-07 | A | K0 Auxiliary nonformal outputs: verified bytes | 5,983,139,358 | inventory match | pass | tested | `runs/track-a-k0-20261107/remote-verification-nonformal.json`; `RpM-Kinshuk/zoo-lab@e2039a33e8fed85264fc4de85bc068bd805af226` |
| 11-07 | A | K0 Auxiliary nonformal outputs: verified safetensors hashes | 33 | all safetensors match | pass | tested | `runs/track-a-k0-20261107/remote-verification-nonformal.json`; `RpM-Kinshuk/zoo-lab@e2039a33e8fed85264fc4de85bc068bd805af226` |
| 11-07 | A | K0 Lab README/tables metadata: verified files | 4 | file-list and size match | pass | tested | `runs/track-a-k0-20261107/remote-verification-metadata.json`; `RpM-Kinshuk/zoo-lab@a69530b2930700dba1aa551ddd0fafcc050711f8` |
| 11-07 | A | K0 Lab README/tables metadata: verified bytes | 446,259 | inventory match | pass | tested | `runs/track-a-k0-20261107/remote-verification-metadata.json`; `RpM-Kinshuk/zoo-lab@a69530b2930700dba1aa551ddd0fafcc050711f8` |
| 11-07 | A | K0 Lab README/tables metadata: verified safetensors hashes | 0 | all safetensors match | pass | tested | `runs/track-a-k0-20261107/remote-verification-metadata.json`; `RpM-Kinshuk/zoo-lab@a69530b2930700dba1aa551ddd0fafcc050711f8` |
| 11-07 | A | K0 Permitted lab evidence: verified model identities | 112 | inventory match | pass | tested | `runs/track-a-k0-20261107/remote-verification-evidence.json`; `RpM-Kinshuk/zoo-evidence@498853257da38151b473beea9fc05acda612aaf6` |
| 11-07 | A | K0 Permitted lab evidence: verified files | 448 | file-list and size match | pass | tested | `runs/track-a-k0-20261107/remote-verification-evidence.json`; `RpM-Kinshuk/zoo-evidence@498853257da38151b473beea9fc05acda612aaf6` |
| 11-07 | A | K0 Permitted lab evidence: verified bytes | 2,848,505,527 | inventory match | pass | tested | `runs/track-a-k0-20261107/remote-verification-evidence.json`; `RpM-Kinshuk/zoo-evidence@498853257da38151b473beea9fc05acda612aaf6` |
| 11-07 | A | K0 Permitted lab evidence: verified safetensors hashes | 224 | all safetensors match | pass | tested | `runs/track-a-k0-20261107/remote-verification-evidence.json`; `RpM-Kinshuk/zoo-evidence@498853257da38151b473beea9fc05acda612aaf6` |
| 11-07 | A | K0 Filtered lab database: verified model identities | 112 | inventory match | pass | tested | `runs/track-a-k0-20261107/remote-verification-db.json`; `RpM-Kinshuk/zoo-db@a40c3da43710587ed27f22fcad58b05189cb723a` |
| 11-07 | A | K0 Filtered lab database: verified files | 18 | file-list and size match | pass | tested | `runs/track-a-k0-20261107/remote-verification-db.json`; `RpM-Kinshuk/zoo-db@a40c3da43710587ed27f22fcad58b05189cb723a` |
| 11-07 | A | K0 Filtered lab database: verified bytes | 8,654,697 | inventory match | pass | tested | `runs/track-a-k0-20261107/remote-verification-db.json`; `RpM-Kinshuk/zoo-db@a40c3da43710587ed27f22fcad58b05189cb723a` |
| 11-07 | A | K0 Filtered lab database: verified safetensors hashes | 0 | all safetensors match | pass | tested | `runs/track-a-k0-20261107/remote-verification-db.json`; `RpM-Kinshuk/zoo-db@a40c3da43710587ed27f22fcad58b05189cb723a` |
| 11-07 | A | K0 Total constructed checkpoints now durable | 139/139 | all prepared constructed checkpoints | pass | tested | `runs/track-a-k0-20261107/final-remote-check.json`; `release-manifest.json` |
| 11-07 | A | K0 Final uploaded artifact files | 1,430/1,430 | combined inventory exactly matches remote lists | pass | tested | `runs/track-a-k0-20261107/final-remote-check.json`; `release-manifest.json` |
| 11-07 | A | K0 Final uploaded artifact bytes | 195,575,617,201 bytes | combined inventory size match | pass | tested | `runs/track-a-k0-20261107/final-remote-check.json`; `release-manifest.json` |
| 11-07 | A | K0 Final safetensors hashes verified remotely | 404/404 | all checkpoint and permitted fingerprint files | pass | tested | `runs/track-a-k0-20261107/final-remote-check.json`; `release-manifest.json` |
| 11-07 | A | K0 Final authorized repositories manually gated | 3/3 | public with manual approval | pass | tested | `runs/track-a-k0-20261107/final-remote-check.json`; `release-manifest.json` |
| 11-07 | A | K0 zoo-lab complete archive file count | 964 | final inventory match | pass | tested | `runs/track-a-k0-20261107/final-remote-check.json` |
| 11-07 | A | K0 zoo-lab complete archive size | 192,718,456,977 bytes | final inventory match | pass | tested | `runs/track-a-k0-20261107/final-remote-check.json` |
| 11-07 | A | K0 Local source files still present with original inventory sizes | 1,430/1,430 | final inventory match | pass | tested | `runs/track-a-k0-20261107/final-remote-check.json` |

## Steward evidence review: 2026-11-08

Reviewed A's K0 final remote check and K6/K10/K8 reports, B's K3 v2,
K4 extraction-repeat and K14 reports, and C's primary check and K2 v2 note.
No new experiment or duplicate measurement was added. Plan status reflects
these reports, not a fresh remote audit. Weekly arithmetic check already done
Nov 7; it is not repeated merely because the board changed.
Fresh skeptic agrees with K3's narrow lab coefficient-relation pilot (18
passes, allowance ≤4 misses); selected views/related parents/three families
cannot establish unknown-parent recovery, order or real Hub transfer.
Fresh skeptic also confirms D5's documentary correction: MPS reports 455
declared-base models and pool curves through 100. No full claim promotion.

## Track C K7 first metadata evidence, 2026-11-08

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-08 | C | Selected repository names pinned | 18/20 | pin selected candidates | partial; two not found | tested (metadata only) | `runs/track-c-k7-handoff-20261108/summary.json`; bound source manifests in `inputs.json`; Track C `docs/research/2026-11-08-k7-first-evidence.md` |
| 11-08 | C | Declared edges with current parent snapshot pins | 13/14 | parent evidence on selected edges | partial; construction revisions unknown | tested (metadata only) | `runs/track-c-k7-handoff-20261108/summary.json`; bound source manifests in `inputs.json`; Track C `docs/research/2026-11-08-k7-first-evidence.md` |
| 11-08 | C | Verified construction edges | 0 | first verified set Dec 1 | pending; metadata does not pass construction rule | tested (metadata only) | `runs/track-c-k7-handoff-20261108/summary.json`; bound source manifests in `inputs.json`; Track C `docs/research/2026-11-08-k7-first-evidence.md` |
| 11-08 | C | Merge recipes collected | 1 | at least 20 by Dec 5 | partial | tested (metadata only) | `runs/track-c-k7-handoff-20261108/summary.json`; bound source manifests in `inputs.json`; Track C `docs/research/2026-11-08-k7-first-evidence.md` |
| 11-08 | C | Small producer source files archived | 40 | bind producer evidence | hash checks pass | tested (metadata only) | `runs/track-c-k7-handoff-20261108/summary.json`; bound source manifests in `inputs.json`; Track C `docs/research/2026-11-08-k7-first-evidence.md` |
| 11-08 | C | Small producer source payload bytes | 234927 | record collection cost | measured; network overhead excluded | tested (metadata only) | `runs/track-c-k7-handoff-20261108/summary.json`; bound source manifests in `inputs.json`; Track C `docs/research/2026-11-08-k7-first-evidence.md` |
| 11-08 | C | New model-weight bytes read | 0 | metadata-only first pass | pass; weight tests not run | tested (metadata only) | `runs/track-c-k7-handoff-20261108/summary.json`; bound source manifests in `inputs.json`; Track C `docs/research/2026-11-08-k7-first-evidence.md` |

## Track A — K6 feasibility, 2026-11-08

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-08 | A | K6 Selected candidate configurations and source metadata pinned | 32 | audit before weight collection | tested | tested | `runs/track-a-k6-feasibility-20261108/REPORT.md`; `input-freeze.json` |
| 11-08 | A | K6 Distinct scratch-training candidates with matching normalized BERT-base backbone configs and safe artifacts | 30 | ≥30 in one config | metadata feasibility pass; fingerprints not yet tested | tested | `runs/track-a-k6-feasibility-20261108/REPORT.md`; `input-freeze.json` |
| 11-08 | A | K6 Existing official safetensors-bot refs with matching source-bin inventories | 29/29 | unchanged source inventory and matching config | pass | tested | `runs/track-a-k6-feasibility-20261108/REPORT.md`; `input-freeze.json` |
| 11-08 | A | K6 Selected safe-file content hashes distinct | 30/30 | no byte-identical safe files | pass metadata; measured-block dedup still required | tested | `runs/track-a-k6-feasibility-20261108/REPORT.md`; `input-freeze.json` |
| 11-08 | A | K6 Selected safe-file sizes, full-file conservative ceiling | 13,213,602,148 bytes | bounded30-model collection | metadata sum; partial reads expected smaller | tested | `runs/track-a-k6-feasibility-20261108/REPORT.md`; `input-freeze.json` |
| 11-08 | A | K6 Weight/header payload collected during feasibility | 0 bytes | 0 before feasibility report | pass | tested | `runs/track-a-k6-feasibility-20261108/REPORT.md`; `input-freeze.json` |

## Track A — K6 BERT controls, 2026-11-08

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-08 | A | K6 Publisher-provenance scratch-trained BERT models, matching normalized backbone config, with partial fingerprints | 30/30 | ≥30, one config | pass count/config; distinct seeds not certified | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |
| 11-08 | A | K6 Matching measured encoder matrix shapes per model | 72 | all encoder matrices | pass | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |
| 11-08 | A | K6 Distinct measured encoder-block signatures | 30/30 | 30/30 | pass exact observed-block dedup only | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |
| 11-08 | A | K6 Independent-control pairs with finite scores | 435/435 | 435/435 | pass | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |
| 11-08 | A | K6 False same-family decisions at unchanged threshold | 0/435 (0%) | 0 | pass narrow diagnostic; no recall or C1 promotion | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |
| 11-08 | A | K6 Compatible shared extractor views per scored pair | 78 | ≥72 | pass; all shared views used | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |
| 11-08 | A | K6 Maximum independent-control score | 0.5309124134194921 | fixed decision threshold0.9999585892781632 | below threshold; initialization relationship unresolved | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |
| 11-08 | A | K6 Identical measured encoder blocks in the high-score pair | 0/72 | investigate potential exact-copy leakage | no observed exact block copies; does not prove seed independence | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |
| 11-08 | A | K6 Requested tensor bytes for the complete K6 cohort | 2,642,780,160 bytes | bounded partial reads | measured; no full-weight download | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |
| 11-08 | A | K6 Summed extraction time for the complete K6 cohort | 806.7643289239495 model-seconds | expected action under2hours | pass; this is not parallel wall time | tested | `runs/track-a-k6-bert-20261108/REPORT.md`; `result.json`; `REVIEW.md` |

## Track C K7 real Hub selected writer blocks, 2026-11-08

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-08 | C | model.layers.16.self_attn.o_proj.weight relative update | 0.021599033944084765 | >0.003 update floor | above floor; not verified parent | tested diagnostic | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | model.layers.16.self_attn.o_proj.weight rank-one residual / delta norm | 0.07264633683662697 | bound ratio 1.4468154412131187 | consistency check passes but bound exceeds whole delta: uninformative rank test | tested diagnostic; no C2 pass | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | model.layers.16.mlp.down_proj.weight relative update | 0.020521958201355633 | >0.003 update floor | above floor; not verified parent | tested diagnostic | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | model.layers.16.mlp.down_proj.weight rank-one residual / delta norm | 0.07645241678852022 | bound ratio 1.5226872448733881 | consistency check passes but bound exceeds whole delta: uninformative rank test | tested diagnostic; no C2 pass | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | model.layers.17.self_attn.o_proj.weight relative update | 0.020318074418206003 | >0.003 update floor | above floor; not verified parent | tested diagnostic | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | model.layers.17.self_attn.o_proj.weight rank-one residual / delta norm | 0.07713000357560641 | bound ratio 1.5379971969321773 | consistency check passes but bound exceeds whole delta: uninformative rank test | tested diagnostic; no C2 pass | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | model.layers.17.mlp.down_proj.weight relative update | 0.01928199909615549 | >0.003 update floor | above floor; not verified parent | tested diagnostic | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | model.layers.17.mlp.down_proj.weight rank-one residual / delta norm | 0.08121047260365137 | bound ratio 1.6206004127035278 | consistency check passes but bound exceeds whole delta: uninformative rank test | tested diagnostic; no C2 pass | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | v2 selected weight payload bytes | 46137344 | bounded two-model pilot | measured; invalid v1 repeated reads excluded | tested diagnostic | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | v2 metadata/header payload bytes | 107782 | bounded two-model pilot | measured; invalid v1 repeated reads excluded | tested diagnostic | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |
| 11-08 | C | v2 diagnostic wall seconds | 4.80540579702938 | bounded two-model pilot | measured; invalid v1 repeated reads excluded | tested diagnostic | `runs/track-c-k7-writer-diagnostic-20261108-v2/report.json`; source hashes `inputs.json`; track-c `docs/research/2026-11-08-k7-writer-diagnostic.md` |

## Track A — K8 bounded cost and query preparation, 2026-11-08

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-08 | A | K8 Dense Llama3.1 8B extraction payload | 641,728,512 bytes | nonzero standalone partial-read cost | pass | tested | `runs/track-a-k8-cost-20261108/cost-results.json` |
| 11-08 | A | K8 Dense Llama3.1 8B extraction runtime | 104.8430084810243 seconds | bounded same-policy cost | pass | tested | `same` |
| 11-08 | A | K8 Dense Llama3.1 70B extraction payload | 3,196,059,648 bytes | nonzero standalone partial-read cost | pass | tested | `same` |
| 11-08 | A | K8 Dense Llama3.1 70B extraction runtime | 357.92708002997097 seconds | bounded same-policy cost | pass | tested | `same` |
| 11-08 | A | K8 Completed 8B artifact resume, new reader calls | 0 | 0 with network forbidden | pass | tested | `runs/track-a-k8-cost-20261108/resume-verification-8b.json` |
| 11-08 | A | K8 Proposed eligible pinned query keys | 216 | ≥200 catalog identities | pass catalog; not200heldout positives | tested | `runs/track-a-k8-protocol-20261108/catalog/summary.json` |
| 11-08 | A | K8 Catalog queries with another known same-root query | 145 | report positive availability separately | candidate-pool presence not yet verified | tested | `same` |
| 11-08 | A | K8 Known-positive roots in query catalog | 11 | report root diversity | no generalization claim | tested | `same` |
| 11-08 | A | K8 Exact inherited R5 heldout query keys | 2 | preserve existing split | pass; no new assignments | tested | `same` |
| 11-08 | A | K8 Header-only pilot complete extractor manifests | 100/100 | 100/100 | pass T0 metadata only | tested | `runs/track-a-k8-headers-20261108/pilot100-summary.json` |
| 11-08 | A | K8 Header-only pilot metadata payload | 6,482,301 bytes | no tensor data reads | pass guarded header/JSON reads only | tested | `same` |
| 11-08 | A | K8 Header-only pilot summed model runtime | 255.912868699932 model-seconds | forecast metadata action bound | measured; not parallel wall time | tested | `same` |
| 11-08 | A | Existing saved-evidence info/similar/family/check workflow incl database opening | 0.809020432 seconds | finite local query cost, explicit missing-base abstention | pass demonstration; not fresh-model end-to-end cost or accuracy | tested | `runs/track-a-workflow-20261108/demo.json` |
| 11-08 | A | K8 Fixed candidate frame identities selected before API calls | 6,000 | fixed bounded reserve | pass fixed selection | tested | `runs/track-a-k8-inputs-20261108/SUMMARY.json` |
| 11-08 | A | K8 Exact distinct candidate repository/revision pins resolved | 5,992 | ≥5,000 before eligibility | pass metadata only;8 not found | tested | `runs/track-a-k8-inputs-20261108/SUMMARY.json` |
| 11-08 | A | K8 Metadata-qualified candidate pins | 5,400 | report conservative screen | pass metadata only | tested | `runs/track-a-k8-inputs-20261108/SUMMARY.json` |
| 11-08 | A | K8 Distinct metadata root-safe-file inventory groups | 5,224 | ≥5,000 before header/resource checks | pass provisional reserve | tested | `runs/track-a-k8-inputs-20261108/SUMMARY.json` |
| 11-08 | A | K8 Metadata root-safe-file duplicate excess removed | 176 | remove exact inventory repeats | pass metadata grouping;not tensor equivalence | tested | `runs/track-a-k8-inputs-20261108/SUMMARY.json` |

## Track A — final bounded K8 preparation, 2026-11-08

| Date | Track | What was measured (plain words) | Number | Target | Verdict | Status | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11-08 | A | K8 Fixed reserve terminal outcomes | 5,224 | 5224 | pass accounting | tested | `runs/track-a-k8-headers-20261108/header-outcomes.json` |
| 11-08 | A | K8 Published exact-pinned extractor header manifests | 5,222 | all usable attempts explicit | pass metadata scope | tested | `runs/track-a-k8-headers-20261108/header-outcomes.json` |
| 11-08 | A | K8 Terminal parser failures preserved | 2 | no hidden failures | excluded;invalid size/duplicate names | tested | `runs/track-a-k8-headers-20261108/header-outcomes.json` |
| 11-08 | A | K8 Header/resource-eligible distinct inventories | 5,154 | ≥5,000 | pass operational preparation | tested | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Selected resource-eligible candidates | 5,000 | 5,000 | pass headers only;not measured scores | tested | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Passing fixed reserve remaining | 154 | report reserve | available before score-dependent checks | tested | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Total terminal/resource exclusions | 70 | all failures explicit | retained;rules unchanged | tested | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Recorded header/JSON requested metadata | 426,670,883 bytes | bounded metadata only | no candidate tensor collection | tested | `runs/track-a-k8-headers-20261108/header-outcomes.json` |
| 11-08 | A | K8 Recorded summed metadata model time | 16,297.081669 summed model-seconds | report actual counters | not wall/CPU time;aborted sandbox start excluded | tested | `runs/track-a-k8-headers-20261108/header-outcomes.json` |
| 11-08 | A | K8 Selected planned row256 tensor payload | 2,566,948,468,040 bytes | bounded supported-layout plan | not actual network traffic or approved launch | planned | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Fingerprint storage point estimate | 19,701,678,952 bytes | capacity planning | empirical estimate;not guaranteed format bound | predicted | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Fingerprint storage planning reserve | 40,663,816,944 bytes | capacity planning | empirical estimate;not guaranteed format bound | predicted | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Free-space gate including proposed margin | 52,977,254,828 bytes | reserve×1.25+2GiB | unmet;capacity decision required | planned | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Observed persistent filesystem free space | 26,254,245,888 bytes | ≥52,977,254,828 | FAIL storage;availability not reservation | tested | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Storage shortfall against proposed gate | 26,723,008,940 bytes | 0 | FAIL;retain local copies | planned | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Dense-anchor scaled serial elapsed forecast | 112.441281 hours | planning only | not CPU time or wall guarantee | predicted | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Dense-anchor scaled serial elapsed forecast ideal four-worker wall division | 28.110320 hours | planning only | not a measured concurrency speedup | predicted | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Historical mixed-host bin serial elapsed forecast | 505.905900 hours | planning only | not CPU time or wall guarantee | predicted | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Historical mixed-host bin serial elapsed forecast ideal four-worker wall division | 126.476475 hours | planning only | not a measured concurrency speedup | predicted | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Models omitted by missing historical runtime bins | 91 | report missingness | historical forecast incomplete | tested | `runs/track-a-k8-headers-20261108/budget.json` |
| 11-08 | A | K8 Minimum planned compatible candidates/query | 0 | report per-query effective scope | structural estimate only | tested | `runs/track-a-k8-protocol-20261108/geometry-coverage.json` |
| 11-08 | A | K8 Maximum planned compatible candidates/query | 252 | report per-query effective scope | not final numeric eligibility/bound | tested | `runs/track-a-k8-protocol-20261108/geometry-coverage.json` |
| 11-08 | A | K8 Queries with zero planned compatible candidates | 12 | 0 | 12 scoped coverage gaps | tested | `runs/track-a-k8-protocol-20261108/geometry-coverage.json` |
| 11-08 | A | K8 Queries with at least100 planned compatible candidates | 64 | report scope | not labeled hard-negative count | tested | `runs/track-a-k8-protocol-20261108/geometry-coverage.json` |
| 11-08 | A | K8 Queries with5000 planned compatible candidates | 0 | report against global5000 target | none;global roster is not per-query5000 | tested | `runs/track-a-k8-protocol-20261108/geometry-coverage.json` |
| 11-08 | A | K8 Production structural-guard differential checks | 15040 | all agree | pass structural logic only | tested | `runs/track-a-k8-protocol-20261108/geometry-coverage.json` |
| 11-08 | A | K8 Planned compatible candidates for each K6 BERT query | 2 | report cohort scope | header geometry estimate;not retained numeric score count | tested | `runs/track-a-k8-protocol-20261108/geometry-coverage.csv` |
| 11-08 | A | K8 Queries with documented positive peer/root exact key in fixed selected roster | 0 / 216 | report known-positive presence | exact-key join only; other revisions/unlabeled relations unknown | tested | `runs/track-a-k8-readiness-20261108/summary.json` |
| 11-08 | A | K8 Queries with known direct-parent keys | 134 | report eligible positive-case scope | catalog parent labels retained; not independent roots | tested | `runs/track-a-k8-readiness-20261108/summary.json` |
| 11-08 | A | K8 Known-parent queries with any parent exact key in fixed selected roster | 0 / 134 | report true-source presence | separate anchors needed for known-present evaluation | tested | `runs/track-a-k8-readiness-20261108/summary.json` |
| 11-08 | A | K8 Known-parent queries with all parent keys in eligible catalog | 134 | report anchor availability | pointers retained; no pool addition or recall claim | tested | `runs/track-a-k8-readiness-20261108/summary.json` |
| 11-08 | A | K8 Query self identities in fixed selected roster | 0 | exclude self before evaluation | exact pins only; group dedup remains required | tested | `runs/track-a-k8-readiness-20261108/summary.json` |
| 11-08 | A | K8 Inherited heldout queries with documented positive exact key in fixed roster | 0 / 2 | report inherited-case positive presence | not proof of absent relatives or fresh blinded population | tested | `runs/track-a-k8-readiness-20261108/summary.json` |
| 11-08 | A | K8 Proposed root/parent anchor pairs checked | 226 | account for every requested anchor | existing fingerprints only; before group removal | tested | `runs/track-a-k8-readiness-20261108/anchor-feasibility.json` |
| 11-08 | A | K8 Proposed root/parent anchor pairs numerically comparable | 226 | finite existing strict k64 scores | comparability only; no threshold or recall test | tested | `runs/track-a-k8-readiness-20261108/anchor-feasibility.json` |
| 11-08 | A | K8 Proposed root/parent anchor pairs unavailable | 0 | report exclusions explicitly | identities/settings/hashes pass; no final pool readiness | tested | `runs/track-a-k8-readiness-20261108/anchor-feasibility.json` |
| 11-08 | A | K8 Queries with a comparable proposed anchor | 134 | report known-positive feasible cases | before duplicate/group removal; not unseen independent roots | tested | `runs/track-a-k8-readiness-20261108/anchor-feasibility.json` |
| 11-08 | A | K8 Queries with a comparable proposed direct-parent anchor | 134 | report parent-present feasibility | no parent decision/coverage/recall test | tested | `runs/track-a-k8-readiness-20261108/anchor-feasibility.json` |
| 11-08 | A | K8 Persistent available bytes at readiness snapshot | 19444793344 bytes | >=52977254828 bytes | FAIL current space; snapshot not quota/reservation | tested | `runs/track-a-k8-readiness-20261108/summary.json` |
| 11-08 | A | K8 Persistent shortfall at readiness snapshot | 33532461484 bytes | 0 bytes | FAIL; existing storage draft to steward, default defer | tested | `runs/track-a-k8-readiness-20261108/summary.json` |

## Track C — matched published comparisons, 2026-11-08

| Measurement | Observed | Target/scope | Meaning | Source |
| --- | ---: | --- | --- | --- |
| Matched baseline models | 46 | 46 fixed distinct models | PASS completeness | `runs/track-c-k2-residual-20261108/REPORT.md` |
| Matched baseline anchor families | 4 | four inherited anchors; not four independent test roots | descriptive coverage | `runs/track-c-k2-residual-20261108/REPORT.md` |
| Training Leaves Traces matched scores | 126 | 126 B pairs | PASS completeness | `runs/track-c-k2-residual-20261108/REPORT.md` |
| Training Leaves Traces unique true-parent top ranks | 7/12 | no scientific rank threshold; ancestry method | descriptive; not parent decisions | `runs/track-c-k2-residual-20261108/REPORT.md` |
| Training Leaves Traces logical weight bytes | 28,003,270,656 | record full MLP payload | measured CPU read cost | `runs/track-c-k2-residual-20261108/REPORT.md` |
| Training Leaves Traces verification scan bytes | 76,403,189,056 | record separately from payload | measured cost | `runs/track-c-k2-residual-20261108/REPORT.md` |
| Training Leaves Traces elapsed seconds | 195.695743 | expected <3600s; stop6600s | within bounded-run limit | `runs/track-c-k2-residual-20261108/REPORT.md` |
| modelDNA matched scores | 126 | 126 B pairs | PASS completeness | `runs/track-c-k2-modeldna-20261108/REPORT.md` |
| modelDNA complete aggregate-signal pairs | 126/126 | all four aggregates present | no aggregate signal gap | `runs/track-c-k2-modeldna-20261108/REPORT.md` |
| modelDNA source-present unresolved verdicts | 12/12 | no newly chosen accuracy target | no definite parent answer | `runs/track-c-k2-modeldna-20261108/REPORT.md` |
| modelDNA source-absent unresolved verdicts | 12/12 | report unchanged native verdict | unresolved in both conditions | `runs/track-c-k2-modeldna-20261108/REPORT.md` |
| modelDNA definite parent answers | 0/12 | descriptive coverage, not an error rate | error rate undefined with no answers | `runs/track-c-k2-modeldna-20261108/REPORT.md` |
| modelDNA native source bytes | 8,680,734,831 | record native reader cost | measured CPU read cost | `runs/track-c-k2-modeldna-20261108/REPORT.md` |
| modelDNA verification scan bytes | 76,442,069,366 | record separately from payload | measured cost | `runs/track-c-k2-modeldna-20261108/REPORT.md` |
| modelDNA elapsed seconds | 124.358342 | expected <3600s; stop6600s | within bounded-run limit | `runs/track-c-k2-modeldna-20261108/REPORT.md` |
| 11-08 | B | K5 joined formal constructed children | 112 | all112 visible | entire roster; not a full detector pass | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 joined scoped measurements across formal edit groups | 90 | 112 visible | partial measurements plus closed prior subsets;22 not run | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 joined children with no joined measurement | 22 | report every gap | six heads and16 disguise; bounded K14 separate | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Adapters eligible children | 24 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Adapters scoped measurement-tested children | 24 | 24 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Refusal removal eligible children | 8 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Refusal removal scoped measurement-tested children | 8 | 8 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 TIES merge eligible children | 6 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 TIES merge scoped measurement-tested children | 6 | 6 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 DARE/TIES merge eligible children | 6 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 DARE/TIES merge scoped measurement-tested children | 6 | 6 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 SFT/DPO eligible children | 16 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 SFT/DPO scoped measurement-tested children | 16 | 16 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Quantization eligible children | 12 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Quantization scoped measurement-tested children | 12 | 12 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Linear merge eligible children | 9 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Linear merge scoped measurement-tested children | 9 | 9 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 SLERP merge eligible children | 9 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 SLERP merge scoped measurement-tested children | 9 | 9 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Decision heads eligible children | 6 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Decision heads scoped measurement-tested children | 0 | 6 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Disguise eligible children | 16 | full formal group | fixed roster; no split change | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 Disguise scoped measurement-tested children | 0 | 16 | measurement coverage; not complete detection/parent decision | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 immediate-parent queries with decisions tested | 12 | 112 | closed quantization subset only; remaining100 NOT TESTED | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 immediate-parent queries not tested in each variant | 100 | report missingness | not counted as successful abstentions | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 adapter strict native untouched-tensor preservation | 12/24 | 24/24 in unchanged native layout | raw layout invariant incomplete; casts/aliases explicit | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 adapter separately measured bf16-cast untouched equivalence | 18/24 | report separately | not substituted for native byte equality | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 adapter native failures explained by source-storage cast | 6 | report gaps | all shared untouched tensors agree after explicit bf16 cast | tested | `runs/track-b-k5-joined-20261108-v2/verification.json` |
| 11-08 | B | K5 adapter native failures explained by tied-head layout | 6 | report gaps | omitted parent lm_head equals parent embedding; shared untouched tensors exact | tested | `runs/track-b-k5-joined-20261108-v2/verification.json` |
| 11-08 | B | K5 adapter ambiguous support with true source present | 24/24 | unique immediate source would be needed | untouched invariant alone does not identify parent | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 adapter nonparent support after true-source removal | 24/24 | reject unsupported source claims | descriptive support survives; not scored as parent answers | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 adapter resolved sampled target ranks | 8/24 | report resolved coverage | existing floor/confidence; sixteen unresolved | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 adapter resolved sampled ranks within configured ±25% | 8/8 | all resolved sampled blocks | not full-matrix rank or full-case C2 pass | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 refusal resolved sampled writer ranks | 0/24 | report unresolved coverage | all sampled writers unresolved under unchanged rules | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 matched nonremoved writer control cases | 40 | same selection as refusal positives | SFT/DPO and LoRA controls; no detector cutoff | tested | `runs/track-b-k5-joined-20261108-v2/verification.json` |
| 11-08 | B | K5 matched nonremoved writer control blocks | 120 | three writers per control | diagnostics available; no calibrated classification | tested | `runs/track-b-k5-joined-20261108-v2/verification.json` |
| 11-08 | B | K5 TIES/DARE minimum target-entry equal-to-anchor share | 0.015401840209960938 | descriptive; no cutoff | zero-tolerance stored values; overlap visible | tested | `runs/track-b-k5-joined-20261108-v2/results.json` |
| 11-08 | B | K5 TIES/DARE maximum target-entry equal-to-anchor share | 0.3108605448404948 | descriptive; no cutoff | not full untouched-module share or classified detection | tested | `runs/track-b-k5-joined-20261108-v2/results.json` |
| 11-08 | B | K5 LoRA controls minimum target-entry equal-to-anchor share | 1.1473894119262695e-06 | descriptive; no cutoff | zero-tolerance stored values; overlap visible | tested | `runs/track-b-k5-joined-20261108-v2/results.json` |
| 11-08 | B | K5 LoRA controls maximum target-entry equal-to-anchor share | 0.4226670837402344 | descriptive; no cutoff | not full untouched-module share or classified detection | tested | `runs/track-b-k5-joined-20261108-v2/results.json` |
| 11-08 | B | K5 SFT/DPO controls minimum target-entry equal-to-anchor share | 2.4065375328063965e-06 | descriptive; no cutoff | zero-tolerance stored values; overlap visible | tested | `runs/track-b-k5-joined-20261108-v2/results.json` |
| 11-08 | B | K5 SFT/DPO controls maximum target-entry equal-to-anchor share | 0.45564292907714843 | descriptive; no cutoff | not full untouched-module share or classified detection | tested | `runs/track-b-k5-joined-20261108-v2/results.json` |
| 11-08 | B | K5 C supplied real writer blocks with informative noise allowance | 0/4 | allowance smaller than whole delta | inconclusive; source tier stays declared | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 shared Hub candidate edges dpo:declared | 3 | source-tier accounting | zero complete exact-mark tests; no verified truth added | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 shared Hub candidate edges merge:declared | 2 | source-tier accounting | zero complete exact-mark tests; no verified truth added | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 shared Hub candidate edges merge_parent:declared | 1 | source-tier accounting | zero complete exact-mark tests; no verified truth added | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 shared Hub candidate edges posttraining:declared | 1 | source-tier accounting | zero complete exact-mark tests; no verified truth added | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 shared Hub candidate edges posttraining_unspecified:declared | 1 | source-tier accounting | zero complete exact-mark tests; no verified truth added | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 shared Hub candidate edges quantization:declared | 1 | source-tier accounting | zero complete exact-mark tests; no verified truth added | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 shared Hub candidate edges refusal_removal:declared | 2 | source-tier accounting | zero complete exact-mark tests; no verified truth added | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 shared Hub candidate edges rlvr:declared | 1 | source-tier accounting | zero complete exact-mark tests; no verified truth added | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 shared Hub candidate edges sft:declared | 2 | source-tier accounting | zero complete exact-mark tests; no verified truth added | tested | `runs/track-b-k5-joined-20261108-v2/summary.json` |
| 11-08 | B | K5 joined CPU measurement elapsed | 394.34426659799647 | under7200s; agent limit | four threads; GPU unavailable; not isolated end-to-end cost | tested | `runs/track-b-k5-joined-20261108-v2/results.json` |

## Track C — Witness Overlap ordinary fine-tunes, 2026-11-08

| Measurement | Observed | Target/scope | Meaning | Source |
| --- | ---: | --- | --- | --- |
| Witness construction-record cases | 16/16 | all frozen proposed triples | PASS operational coverage | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness distinct models | 20 | ≤100 models; fixed existing checkpoints | bounded local run | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness correct orientations | 8/16 | exploratory; no accuracy gate | not a general direction success | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness wrong orientations | 8/16 | report every error | all SFT edges reversed | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness SFT correct | 0/8 | fixed opposite-seed SFT witnesses | witness construction fails here | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness DPO correct | 8/8 | same fixed witness rule | conditional lab result only | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness ties | 0/16 | exact ties abstain | no ties | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness abstentions | 0/16 | no eligible role means abstain | all roles usable | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness not-run cases | 0/16 | account for every frozen case | none missing | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness each-family correct | 2/4 per family | four lab anchors, not independent pair trials | same SFT/DPO pattern | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness BF16-view correct | 8/16 | predeclared storage control | same pattern | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness BF16-view changed answers | 0/16 | compare frozen primary/control | storage equalization did not reverse answers | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness nearest-witness distance correct | 8/16 | cheap same-input interpretation control | no causal order proof | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness distance-control agreement | 16/16 | same eligible projections | all decisions agree | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness full projection tensors | 2800/2800 | all required complete dense projections | no undefined tensor scores | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness logical weight bytes | 24,012,914,688 | full tensors, reused for both modes | successful v3 read cost | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness verification/input bytes | 32,653,895,960 | separate from payload | successful v3 hash-scan cost | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| Witness successful-run seconds | 146.637193 | expected<1800s; stop6600s | within run limit; failed setup IO excluded | `runs/track-c-k2-witness-20261108-v3/REPORT.md` |
| 11-08 | B | K8 existing-artifact eligible catalog keys | 216 | all eligible existing keys | no new collection; seven original exclusions retained | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 existing-artifact observed row-block groups | 212 | self/group removal | not full-weight identity | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 existing-artifact scored unordered pairs | 23220 | account all finite/unavailable pairs | existing fixed k64 scorer; no weights | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog queries | 216 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog roots | 82 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog scoreable | 216 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog known_positive_scoreable | 145 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog useful_family_queries | 130 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog family_abstentions | 86 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog unavailable | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog known_false_query_answers | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog unknown_query_answers | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog known_false_hits | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog unknown_hits | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 catalog returned_hits | 1887 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present queries | 134 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present roots | 11 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present scoreable | 134 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present known_positive_scoreable | 134 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present useful_family_queries | 119 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present family_abstentions | 15 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present unavailable | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present known_false_query_answers | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present unknown_query_answers | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present known_false_hits | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present unknown_hits | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source present returned_hits | 1786 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed queries | 134 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed roots | 11 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed scoreable | 133 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed known_positive_scoreable | 133 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed useful_family_queries | 112 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed family_abstentions | 21 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed unavailable | 1 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed known_false_query_answers | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed unknown_query_answers | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed known_false_hits | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed unknown_hits | 0 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired source removed returned_hits | 1628 | descriptive; no new pass target | family evidence only; not source inference | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 present-source known-positive misses | 15 | report failures | all are quantization cases; exact quantizer verdict unchanged | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 exact-source unique top-score identity diagnostic | 20/134 | rank only, not classifier accuracy | all20 single-source; ties and multi-source retained | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired single-source queries | 104 | report source cardinality | no joint-parent recovery inferred | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 paired multi-source queries | 30 | report source cardinality | family support is not complete parent-set recovery | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
| 11-08 | B | K8 raw pair labels positive_total | 1706 | preserve explicit label coverage | unknown relationships never negatives | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 raw pair labels positive_finite | 1706 | preserve explicit label coverage | unknown relationships never negatives | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 raw pair labels positive_hits | 994 | preserve explicit label coverage | unknown relationships never negatives | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 raw pair labels negative_total | 180 | preserve explicit label coverage | unknown relationships never negatives | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 raw pair labels negative_finite | 180 | preserve explicit label coverage | unknown relationships never negatives | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 raw pair labels unknown_total | 21334 | preserve explicit label coverage | unknown relationships never negatives | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 raw pair labels unknown_finite | 523 | preserve explicit label coverage | unknown relationships never negatives | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 certified-independent support hits | 0/180 | original fixed rule | dependent same-architecture controls; no iid guarantee | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing BERT scratch controls abstain_present | 30 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing BERT scratch controls abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing BERT scratch controls known_positive_available | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing BERT scratch controls paired_source_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing BERT scratch controls queries | 30 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing BERT scratch controls unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing BERT scratch controls useful_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing BERT scratch controls useful_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing SFT/DPO abstain_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing SFT/DPO abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing SFT/DPO known_positive_available | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing SFT/DPO paired_source_removed | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing SFT/DPO queries | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing SFT/DPO unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing SFT/DPO useful_present | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing SFT/DPO useful_removed | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing adapter abstain_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing adapter abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing adapter known_positive_available | 30 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing adapter paired_source_removed | 30 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing adapter queries | 30 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing adapter unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing adapter useful_present | 30 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing adapter useful_removed | 30 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing bases / pretraining controls abstain_present | 41 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing bases / pretraining controls abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing bases / pretraining controls known_positive_available | 11 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing bases / pretraining controls paired_source_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing bases / pretraining controls queries | 52 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing bases / pretraining controls unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing bases / pretraining controls useful_present | 11 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing bases / pretraining controls useful_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing decision head abstain_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing decision head abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing decision head known_positive_available | 7 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing decision head paired_source_removed | 7 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing decision head queries | 7 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing decision head unavailable_removed | 1 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing decision head useful_present | 7 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing decision head useful_removed | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing disguise abstain_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing disguise abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing disguise known_positive_available | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing disguise paired_source_removed | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing disguise queries | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing disguise unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing disguise useful_present | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing disguise useful_removed | 18 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge dare_ties abstain_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge dare_ties abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge dare_ties known_positive_available | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge dare_ties paired_source_removed | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge dare_ties queries | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge dare_ties unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge dare_ties useful_present | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge dare_ties useful_removed | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge linear abstain_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge linear abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge linear known_positive_available | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge linear paired_source_removed | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge linear queries | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge linear unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge linear useful_present | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge linear useful_removed | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge slerp abstain_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge slerp abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge slerp known_positive_available | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge slerp paired_source_removed | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge slerp queries | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge slerp unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge slerp useful_present | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge slerp useful_removed | 9 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge ties abstain_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge ties abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge ties known_positive_available | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge ties paired_source_removed | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge ties queries | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge ties unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge ties useful_present | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing merge ties useful_removed | 6 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing quantization abstain_present | 15 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing quantization abstain_removed | 21 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing quantization known_positive_available | 23 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing quantization paired_source_removed | 23 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing quantization queries | 23 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing quantization unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing quantization useful_present | 8 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing quantization useful_removed | 2 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing refusal construction abstain_present | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing refusal construction abstain_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing refusal construction known_positive_available | 8 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing refusal construction paired_source_removed | 8 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing refusal construction queries | 8 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing refusal construction unavailable_removed | 0 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing refusal construction useful_present | 8 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing refusal construction useful_removed | 8 | fixed construction stratum | truth stratum, not detected edit; removed variant paired only | tested | `runs/track-b-existing-artifacts-20261108/report-tables.json` |
| 11-08 | B | K8 existing-artifact CPU elapsed | 13.527163679013029 | under7200s agent limit | one numerical thread; no acquisition or GPU timing | tested | `runs/track-b-existing-artifacts-20261108/results.json` |
