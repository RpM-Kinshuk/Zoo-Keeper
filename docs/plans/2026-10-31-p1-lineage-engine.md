# P1 Lineage Engine Implementation Plan

> **For agentic workers:** Use the lean per-task loop in `handoff/PROMPTS.md`: test first, one review for core logic, and leave changes uncommitted. Skills may support that loop; do not add agents or checks beyond it.

**Goal:** Infer typed, calibrated model lineage from local fingerprints and measure false parents per query as the candidate pool grows.

**Architecture:** A CPU-first `lineage/` package consumes immutable local manifests and fingerprints. Exact retrieval proposes candidates; aligned sketches support merge regression and projected update tests; extraction-time delta summaries support edge typing. Calibration and evaluation are separate stages with frozen root-held-out splits. A thin adapter imports the merged Model-Zoo packages only when real fingerprints are used, so synthetic development can start independently.

**Tech Stack:** Python in `esd_ind`, numpy, pandas, scipy, safetensors, pyarrow, pytest; networkx for Chu–Liu/Edmonds. Reuse installed versions; pin every imported dependency to the installed version in `requirements.txt` when implementation begins. No installs are part of this planning task.

**Spec:** `docs/research/2026-10-27-merged-research-plan.md` §9 ideas 7–11, §10, §11 P1 row, §12; `handoff/GUIDANCE.md` §§4.2–4.4 and 5.2; checklist E1–E9. Authoritative upstream interfaces are Model-Zoo's extractor Tasks 6, 8–9 and zoo Tasks 1, 4–6. The companion lab plan defines the exact-truth scorer.

**State:** E1 approved for implementation by the owner on 2026-10-31. No implementation or runs are authorized by this document alone. Suggested plan commit: `docs: plan P1 lineage engine and evaluation`.

## Global Constraints

- Work in `track-b`'s worktree after plan review. Keep new plans, `CHECKLIST.md`, and `STATUS.md` in the main checkout. Never commit, merge, rebase, or push.
- Local mode only. Inputs are explicit paths; never resolve a Hub ID, download weights, import frozen Model-Research code, or execute checkpoint code. Unit tests deny network access.
- Never open, enumerate, hash, or inspect `$ZOO_DATA/lab_blind` before the frozen evaluation step. Track C owns it and returns evaluation results; training and calibration receive public data only.
- Model keys are `<repo_id>@<40-character SHA>` or `<repo_id>@local:<sha256>`. Stored paths are relative to `$ZOO_DATA`. All output writes use a temporary sibling and rename; resume validates input/config hashes.
- Reuse `load_fingerprint`, `pair_similarity`, `zoo.schema.arch_signature`, `zoo.retrieval` and `LINEAGE_COLUMNS` after their code is committed and merged into Model-Zoo's `main`. Do not copy production implementations into Zoo-Keeper.
- Seeds: `20261028`. Retrieval uses T1 only, matching architecture signatures, view names, row intervals, sketch seed and evidence version. The calibrated statistic is `pair_similarity(..., sketch_k=64)["median_cosine"]`, with `tier="t1"`, `row_block=256`; larger stored sketches are truncated to 64. Retrieval similarity is never assigned this null.
- Fingerprints contain signed weight sketches and delta summaries, **not signed raw block deltas or singular vectors**. Use aligned child-minus-parent sketches for projected tests; never reconstruct missing raw deltas from singular values. Extraction-time rank applies only to `meta["parent"]` matching the proposed parent.
- `below_floor`, missing data, insufficient shared views, collinear regressors, unsupported representations, and unresolved direction cause explicit abstention. Never turn `rank=-1` into rank zero.
- Distinguish three estimands (`same_root`, `ancestor`, `parent`) and three truth tiers (lab exact, verified Hub, declared audited). A same-series sibling is a negative for ancestor/parent, usually a positive for same-root. No undeclared or merely different-root model is automatically a negative.
- Every inferred row is `evidence_tier="detected"`; observed shared bytes or declarations are separate evidence. Scores, coverage, and `calibrated` are explicit. No predictions of capability, adversarial robustness, or causal direction from symmetric cosine alone.
- Preserve `LINEAGE_COLUMNS` exactly: `child_key,parent_key,estimand,edge_type,evidence_tier,evidence,score,p_value,calibrated,method_version`. `method_version` already exists. Edge vocabulary: `finetune,adapter,merge,quantized,abliteration,unknown`. Put recipe intervals and diagnostics in `evidence` JSON and sidecars; do not change upstream schemas.
- Group byte duplicates, near-duplicates and reuploads before splitting. Hold out complete root components, including every parent of a merge; quarantine cross-split merges. Fit rules, cutoffs, nulls and conformal sets on development/calibration only. Freeze protocol before final/prospective runs.
- Model reading jobs above 100 models or two hours, and all paid/inference batches above D4's limit, are `[HUMAN] launches`. Existing-fingerprint analysis is not a weight-reading job, but jobs expected above two hours are still queued. No benchmark downloads or large runs in unit tests.

## Review Focus

1. A T2 sketch, differently placed row block, vocabulary extension, or small padded head must not silently enter a mismatched T1 comparison (Task 1).
2. Near-collinear merge parents and nonlinear mergekit transforms need effective-coefficient diagnostics rather than invented configured-weight recovery (Task 2).
3. Missing parent delta metadata, bf16 flooring, and projected stacked directions must not become confident low-rank edge labels (Tasks 3–4).
4. Candidate retrieval, dependent views/candidates, thin null tails, and unseen families must not create unsupported error-control guarantees (Task 6).
5. Missing predictions, incomplete Hub truth, duplicate leakage, an interrupted write, and timestamp-only direction must remain visible in the denominators and outputs (Tasks 5, 7–8).

## Shared fixtures and execution rules

Task 1 creates `tests/lineage/conftest.py` and `lineage/testing.py`. All fixture arrays use `np.random.default_rng(20261028)` and float32 storage, float64 fitting. `synthetic_pool` returns `SimpleNamespace(records, truth, universe, manifests)` with 12 models: two independent roots of one architecture, an independent different architecture, each root's child and grandchild, two siblings of the first child, one two-parent merge, one byte duplicate, one T0-only model. Six aligned views across three layers have shape `(512,128)`, rows `(128,384)`, and 64×64 sketches; metadata carries the frozen settings. `truth` is a DAG with explicit completeness flags and `universe` lists query/candidate eligibility. Hub pins use deterministic marker SHAs; local pins use 64 hex digits. Fixtures never claim to be real checkpoints.

`make_record(key, sketches, *, parent=None, delta=None, arch="cfg:synthetic", tier="t1", row_block=256, rows=None, names=None)` returns the record defined in Task 1. Fixtures required in later tasks are described with their owning tests. Assertions below are normative; implementers add imports and precisely the fixture construction described, never weaken thresholds to pass.

For each task: write its tests, run its named file and see the expected missing module/function failure, implement, rerun only that file, then run `python -m pytest tests/lineage -q` once at task completion. The lead runs `python -m pytest tests -q` once at a batch handoff. Core tasks receive one fresh review; no commit steps execute. Commands below run from the Zoo-Keeper worktree in `esd_ind`.

---

### Task 1 (E2): Local records, synthetic retrieval, and upstream adapter

**Files:** Create `lineage/__init__.py`, `lineage/records.py`, `lineage/retrieval.py`, `lineage/upstream.py`, `lineage/testing.py`, `tests/lineage/conftest.py`, `tests/lineage/test_records.py`; update `requirements.txt` when first imports require it.

**Interfaces:**
- `FingerprintRecord(key: str, arch: str, names: tuple[str,...], shapes: np.ndarray, rows: np.ndarray, geometry: tuple[str,...], sketches: np.ndarray, delta: dict, meta: dict, coverage: list[dict])` is immutable by convention; arrays are read-only on construction.
- `compatible_views(a: FingerprintRecord, b: FingerprintRecord, *, k: int=64) -> list[tuple[int,int]]` matches names and geometry plus row intervals, compatible non-vocabulary dimensions, seed, evidence version and tier; checks finite nonzero sketch norms. Shape changes are recorded separately; a common embedding block may compare despite full vocabulary growth.
- `retrieve(query: FingerprintRecord, candidates: list[FingerprintRecord], *, limit: int=50) -> list[dict]` returns `parent_key,retrieval_score,n_shared,reason`, sorted descending score then key. Self and exact duplicate-group alternatives are excluded and documented. Synthetic implementation uses normalized 32×32 blocks and exact mean cosine over up to 32 fixed common views; production adapter calls upstream retrieval rather than duplicating it.
- `load_records(manifests: Path, fingerprints: Path, *, data_root: Path) -> list[FingerprintRecord]` in `upstream.py` lazily imports merged `evidence`/`zoo`; manifest identity, relative artifacts and fingerprint identity must agree. Missing imports give an actionable dependency error, not automatic installation. `median_similarity(a,b,k=64)` calls upstream `pair_similarity` for real records; synthetic records use an injected scoring callable with the identical stated median-cosine definition.

- [ ] **Step 1: Write the failing tests.**

```python
def test_retrieve_is_local_sorted_and_arch_limited(synthetic_pool):
    rows = retrieve(synthetic_pool.records[3], synthetic_pool.records, limit=3)
    assert len(rows) <= 3
    assert all(r["parent_key"] != synthetic_pool.records[3].key for r in rows)
    assert rows == sorted(rows, key=lambda r: (-r["retrieval_score"], r["parent_key"]))
    assert all(r["parent_key"] != synthetic_pool.records[2].key for r in rows)

def test_settings_and_padding_are_checked(aligned_pair):
    a, b = aligned_pair
    assert len(compatible_views(a, b)) == 6
    assert compatible_views(a, replace_record(b, tier="t2")) == []
    assert compatible_views(a, replace_record(b, row_block=128)) == []
    assert compatible_views(a, replace_record(b, rows=b.rows + 1)) == []
    assert compatible_views(a, replace_record(b, sketch_seed=7)) == []
```

`aligned_pair` is two records with identical six-view structure. `replace_record` copies a record and changes fields or metadata, leaving original immutable. Add `test_vocab_growth_keeps_shared_embedding_block` (4096→4104 rows, same 1920:2176 block accepted), `test_small_head_excludes_zero_padding` (2×128 head, no cosine mass from padding), `test_zero_view_and_nonfinite_are_explained` and `test_load_records_never_uses_network` (socket connect raises; local fixture loader succeeds when upstream available). Keep the latter in `tests/lineage/test_upstream.py` and run it only once upstream is merged; E2's synthetic task does not skip its own tests to await A.

- [ ] **Step 2:** Run `python -m pytest tests/lineage/test_records.py -q`; expected missing `lineage` failure.
- [ ] **Step 3:** Implement the interfaces and shared fixtures. Do not import `evidence` or `zoo` at package import time. E2 may close on synthetic code; upstream integration is completed in Task 7 after A merges X6/X8/Z4.
- [ ] **Step 4:** Run the named tests, then package tests once; expected PASS. A JSON fixture generated in `tmp_path` is the tiny local dry run; no data artifacts committed.
- [ ] **Step 5:** Suggested commit: `lineage: local synthetic records and exact retrieval`.

### Task 2 (E3): Unconstrained merge regression and layer-block intervals

**Files:** Create `lineage/merge.py`, `tests/lineage/test_merge.py`.

**Interfaces:** `fit_merge(child: FingerprintRecord, parents: list[FingerprintRecord], base: FingerprintRecord, *, groups: dict[str,list[str]]|None=None, bootstrap: int=200, block_layers: int=2, seed: int=20261028) -> dict`. Output contains per-group `coefficients`, `coefficient_kind`, `intervals`, `condition_number`, `rank`, `collinear_groups`, `relative_residual`, `n_views`, and `status`.

- [ ] **Step 1: Write tests.** Build three independent parent/base 64×64 sketch arrays over eight layers, two views each, then child `0.7*parent0 + 0.4*parent1 - 0.2*base`. No additive noise in exact recovery test. A depth fixture changes parent coefficient from 0.2 to 0.8 in the second four-layer group. `groups` maps `early`/`late` to explicit names; absent groups means one group per tensor, plus an all-layer summary.

```python
def test_unconstrained_coefficients_and_depth(merge_fixture):
    out = fit_merge(**merge_fixture.inputs)
    assert np.max(np.abs(np.asarray(out["groups"]["all"]["coefficients"]) - [0.7, 0.4, -0.2])) < 1e-5
    assert abs(sum(out["groups"]["all"]["coefficients"]) - 0.9) < 1e-5
    assert out["status"] == "identified"

def test_collinear_parents_are_not_attributed(collinear_merge):
    out = fit_merge(**collinear_merge.inputs)
    assert out["status"] == "nonidentifiable"
    assert out["collinear_groups"]
    assert out["groups"]["all"]["intervals"] is None
```

Add `test_depth_schedule_is_recovered` (0.2/0.8 within 0.01), `test_layer_bootstrap_is_deterministic` (same seed gives identical intervals), and `test_nonlinear_recipe_is_effective_not_configured` (TIES/DARE metadata marks effective; residual and configured comparison are both retained). Duplicate base also present as a parent is one regressor with both roles recorded, not two identical columns.

- [ ] **Step 2:** Run `python -m pytest tests/lineage/test_merge.py -q`; expected missing function/module failure.
- [ ] **Step 3:** Align common sketches, vectorize and solve float64 `np.linalg.lstsq` without coefficient bounds or sum constraint. Use contiguous two-layer moving blocks for 200 bootstrap resamples; fewer than four layers gives `insufficient_layers` for intervals. Rank deficiency or condition number >1e6 gives nonidentifiable; group columns with absolute normalized correlation ≥0.999 and report group attribution only. Missing views reduce coverage explicitly. Bootstrap group identities must remain stable. SLERP/TIES/DARE are nonlinear; compare effective fits and layer patterns to configs without promising every coefficient equals configured weight. On actual lab merges, measure ±0.05 target and report PASS/FAIL, never loosen tests to meet it.
- [ ] **Step 4:** Run test file then package tests once, expected PASS.
- [ ] **Step 5:** Suggested commit: `lineage: unconstrained merge fits and layer bootstrap`.

### Task 3 (E4): Delta features and edge typing with abstention

**Files:** Create `lineage/typing.py`, `tests/lineage/test_typing.py`.

**Interfaces:** `pair_features(child: FingerprintRecord, parent: FingerprintRecord) -> dict`; `type_edge(features: dict, rules: dict) -> dict` returns `edge_type,status,reasons,features`. Rules are learned/frozen on development roots only; rule JSON defines numerical cutoffs, eligible modules and required coverage rather than hiding defaults in code.

- [ ] **Step 1: Write tests.** `delta_pair` has six view summaries with ranks 8, confidence `high`, status `ok` as emitted by the extractor, known module names and matching `meta.parent`. Use `ok`, `identical`, `below_floor` exactly, with confidence `high`/`low`; fingerprint arrays expose confidence under `delta["confidence"]`. Rules fixture has `adapter_max_rank=16,min_views=4,min_changed_fraction=0.25,min_gap=2.0`, with fixture gaps 4.0. `below_floor` fixture uses rank -1 and identical fraction 0.9.

```python
def test_below_floor_and_wrong_parent_abstain(delta_pair, rules):
    child, parent = delta_pair
    child = replace_record(child, delta_status="below_floor", delta_rank=-1)
    assert type_edge(pair_features(child, parent), rules)["edge_type"] == "unknown"
    child = replace_record(child, parent="other@" + "a"*40)
    assert pair_features(child, parent)["rank_status"] == "parent_mismatch"

def test_rank_alone_does_not_label_abliteration(delta_pair, rules):
    child, parent = delta_pair
    child = replace_record(child, delta_rank=1)
    assert type_edge(pair_features(child, parent), rules)["edge_type"] != "abliteration"
```

Add tests for identical/no update, no summaries, head-only addition as `finetune`, and two fixtures sharing rank but different changed-module pattern. Explicit adapter factors provide stronger structural evidence; a low numerical rank alone is insufficient. Quantization simulation uses delta flatness, storage facts and learned leaf prior; bf16 storage by itself does not imply quantization.

- [ ] **Step 2:** Run `python -m pytest tests/lineage/test_typing.py -q`; expect missing module/function.
- [ ] **Step 3:** Produce rank, spectral-gap, fraction-identical and module-pattern features only from matched-parent summaries. Interpret successful status as `ok`, and confidence as `high`/`low`, following upstream X4/X6. Retain absent features as null with reasons. Learn a shallow explicit rule tree on development exact truth, include `unknown` as an outcome, then freeze its JSON. Sketched numerical rank is capped at k/4 reliability and never substitutes for extraction-time block rank.
- [ ] **Step 4:** Run named tests then package tests once, expected PASS.
- [ ] **Step 5:** Suggested commit: `lineage: evidence-aware edge typing and floor abstention`.

### Task 4 (E5): Stacked abliteration in aligned projected and local block space

**Files:** Create `lineage/abliteration.py`, `tests/lineage/test_abliteration.py`.

**Interfaces:** `stacked_test(deltas: list[np.ndarray], *, layer_ids: list[str], max_rank: int=1) -> dict` returns `n_layers,rank,energy_fraction,shared_direction_score,status`; `projected_abliteration(child, parent, *, residual_views: list[str], max_rank: int=1) -> dict`; `local_block_deltas(child_dir: Path,parent_dir: Path,*,tensor_names: list[str],rows: tuple[int,int]) -> list[np.ndarray]` reads only plain local safetensors matrix slices after manifest hash verification, no raw output is written. Exact local block verification is optional shortlist evidence, not a new fingerprint field.

- [ ] **Step 1: Write tests.** Build four parent matrices (32,48); common unit vector r and Δ_l = -r(rᵀW_l). Negative fixture has four mutually orthogonal left directions with equal update energy. Both use identity projection for the exact test; a second fixture projects all blocks with the same seeded 16×32 orthonormal operator.

```python
def test_shared_direction_is_a_stacked_property(stacked_fixture):
    out = stacked_test(stacked_fixture.common, layer_ids=["0","1","2","3"])
    neg = stacked_test(stacked_fixture.independent, layer_ids=["0","1","2","3"])
    assert out["energy_fraction"] > 0.999
    assert neg["energy_fraction"] < 0.5
    assert stacked_test(stacked_fixture.common[:1], layer_ids=["0"])["status"] == "insufficient_layers"
```

Add `test_projected_direction_survives_common_left_operator`, `test_heretic_low_rank_span` (two common directions needs `max_rank=2`), `test_mismatched_block_and_floor_abstain`, and `test_local_slices_offline` (generated safetensors, exact child-parent slices, socket connect forbidden). Projected test must return `space="sketch"`; block test returns `space="block"`. Per-layer unrelated directions are not a shared low-rank result even when every layer has rank one.

- [ ] **Step 2:** Run `python -m pytest tests/lineage/test_abliteration.py -q`; expected missing module.
- [ ] **Step 3:** Concatenate row-aligned deltas horizontally and compute SVD; require ≥3 distinct layers and common output-side projection. Score explained squared singular-value energy at rank 1 and small multi-direction ranks; cutoffs are calibrated against lab non-abliterations. `residual_views` is an explicit config list of output-oriented residual-writing matrices; reject shapes/orientation that cannot share the row direction. Projected deltas equal signed child-minus-parent sketches only under identical operators and alignment. A sketch result is supportive detected evidence, not proof of full-space shared rank. For exact shortlist verification read local safetensors slices with `safe_open`, never pickle/full model load, verify pins and record additional bytes. Unsupported packed/adapter representations abstain rather than silently switching decode conventions. Keep raw deltas in memory only. If a bounded resume cache is needed, raw parent blocks may live only in explicit `--parent_cache` under `$ZOO_DATA/cache`; sidecars contain scalar statistics, hashes and pinned candidate keys, never raw blocks or directions.
- [ ] **Step 4:** Run test file then package tests once, expected PASS.
- [ ] **Step 5:** Suggested commit: `lineage: stacked abliteration with projected and block diagnostics`.

### Task 5 (E6): Tree-likeness and observed-ancestor assembly

**Files:** Create `lineage/tree.py`, `tests/lineage/test_tree.py`, `configs/lineage/tree.json`.

**Interfaces:** `tree_diagnostics(distances: np.ndarray, labels: list[str], truth_edges: list[tuple[str,str]], *, predicted_edges: list[tuple[str,str]]|None=None, tolerance: float=1e-6) -> dict`; `assemble_tree(nodes: list[str], edges: pd.DataFrame, *, roots: list[str], times: dict[str,str]|None=None, leaf_keys: set[str]|None=None) -> pd.DataFrame`; `pivot_flags(*, four_point: float, normalized_rf: float) -> list[str]`. `is_acyclic(frame)` is a test helper using a topological traversal; `pivot_flags` only reports thresholds. Directed edge frame has `parent_key,child_key,cost,direction_status`.

- [ ] **Step 1: Write tests.** `tree_fixture` uses six observed nodes on a root→two children→three grandchildren graph, edge lengths 1,2,3,4,5 and shortest-path tree distances. The fixture passes `predicted_edges` equal to its known tree solely to test the metric; production predictions come from assembly, never truth. Without predicted edges nRF is null with `not_assembled`. Explicit quartet fixture is all unit distances except d(a,b)=3, violating the four-point condition. Assembly fixture includes a directed cycle with costs chosen to require contraction.

```python
def test_tree_metric_and_cycle_contraction(tree_fixture):
    out = tree_diagnostics(**tree_fixture.inputs)
    assert out["four_point_violation_fraction"] == 0
    assert out["normalized_rf"] == 0
    graph = assemble_tree(**tree_fixture.cyclic_inputs)
    assert len(graph) == len(tree_fixture.cyclic_inputs["nodes"]) - 1
    assert is_acyclic(graph)

def test_pivot_thresholds_are_reported_not_decided():
    assert pivot_flags(four_point=0.21, normalized_rf=0.1) == ["four_point_gt_0.20"]
    assert pivot_flags(four_point=0.1, normalized_rf=0.21) == ["normalized_rf_gt_0.20"]
```

Add `test_non_tree_quartet`, `test_sibling_cosine` (d1=d2=1,d12=2 gives zero; zero-length edges give null/reason), `test_reticulation_removed_before_tree_fit`, and `test_symmetric_similarity_cannot_direct` (unknown timestamps and no asymmetric evidence give unresolved direction, not a fabricated edge).

- [ ] **Step 2:** Run `python -m pytest tests/lineage/test_tree.py -q`; expected missing module.
- [ ] **Step 3:** Report all four-point quartets for small families; otherwise seeded sample of at most 10,000 with numerator/denominator. Sort three quartet sums; violation means largest minus second-largest > tolerance × max(1,largest). nRF compares unrooted nontrivial split sets on identical observed leaves: symmetric difference divided by sum of split counts (zero when both empty); compute against the assembled predicted tree, not a tree reconstructed from truth distances. Report missing leaf coverage. Compute sibling update cosine `(d1+d2-d12)/(2*sqrt(d1*d2))` from squared sketch distances only when d1,d2 positive; include raw value and no guaranteed concentration claim. Exclude recovered merges and descendants with multi-root ambiguity for tree-only diagnostics; separately report reticulate fraction.

Use NetworkX's minimum spanning arborescence (Chu–Liu/Edmonds) on permitted directed edges, with a synthetic super-root connected only to allowed roots; deterministic key-order tie breaking. Candidate cost is negative log frozen parent plausibility plus configurable upload-time/quantized-leaf priors. These priors are labeled assumptions; timestamps alone never certify direction. Unknown orientations remain in a sidecar and reduce direction coverage. No neighbor joining is needed for observed-ancestor assembly; latent-ancestor reconstruction is parked until observed-node results justify it. Run the public lab diagnostic and record threshold verdicts on Nov 27; the person decides the pivot.

- [ ] **Step 4:** Run test file then package tests once, expected PASS. Dry-run six-node cycle fixture requires no model weights.
- [ ] **Step 5:** Suggested commit: `lineage: tree diagnostics and directed arborescence assembly`.

### Task 6 (E7): Empirical null, tail model, per-query control, and open sets

**Files:** Create `lineage/calibration.py`, `lineage/conformal.py`, `tests/lineage/test_calibration.py`, `tests/lineage/test_conformal.py`, `configs/lineage/calibration.json`.

**Interfaces:**
- `fit_null(scores: pd.DataFrame, *, estimand: str, settings: dict, tail_quantile: float=0.9) -> dict`. Rows: `query_key,candidate_key,query_family,candidate_family,negative_kind,score,split`; only certified negatives from calibration split enter. Artifact contains empirical samples by exchangeability stratum, GPD fit/diagnostics, statistic settings, source hashes, root groups, method version.
- `exact_shortlist_permutation(observed: float, null_scores: np.ndarray) -> dict` returns `p_value,n_permutations,validity`. This enumerates all assignments of the candidate label among the observed and exchangeable null scores; use conservative ties. The result is `(1 + count(null >= observed))/(n+1)`, an exact label-permutation rank test under the recorded exchangeability assumption, not coordinate shuffling.
- `query_control(p_values: np.ndarray, *, alpha: float=0.05, pool_size: int, validity: str="unvalidated") -> np.ndarray` uses e-BH on `e=0.5/sqrt(p)`; missing/non-shortlisted candidates have e=0 and count in full `pool_size` m. Reject top r when `e_(r) >= m/(alpha*r)`. Require p in (0,1] and `validity="exchangeable"` from validated empirical/permutation mode; raise `ValueError` for `unvalidated`, `approximate_gpd`, or unsupported modes. The caller must preserve the source calibration validity, never upgrade it based on score magnitude. No guaranteed control from an unchecked GPD fit.
- `fit_conformal(calibration: pd.DataFrame, *, alpha: float=0.1) -> dict`; `prediction_set(candidate_probabilities: dict[str,float], *, family: str, calibration: dict, scorer_hash: str|None=None, candidate_universe_hash: str|None=None) -> dict`. Class probabilities include `not_in_reference`, generated by a frozen development model. Fit inputs require frozen scorer/candidate-universe SHA256 provenance; application must match it before calibrated status. The complete fitted artifact (including thresholds) is content-hashed and checked at use; missing or mismatched application provenance returns an uncalibrated full set. Null artifacts additionally bind each supported family to its architecture, which caller settings must match. Calibration rows have `family,true_label,nonconformity` where nonconformity is the maximum of `1-probability[label]` over all true parent/ancestor/root candidates for that query, or `1-probability[not_in_reference]` when none is in the pool; missing-reference examples are real held-out roots. Calibration therefore targets inclusion of the whole true set for an exchangeable query, not only a conveniently chosen parent. A fixed development scorer forms candidate logits from median cosine divided by a positive scalar temperature, plus a `not_in_reference` intercept; fit the temperature/intercept by development-only average negative log probability over true labels for the stated estimand and fixed candidate-universe construction, record them in method JSON, then freeze before conformal calibration. Report both whole-true-set inclusion and marginal candidate coverage; the max-true-label nonconformity handles multiple parents without converting them to mutually exclusive ground truth. Output contains `labels,coverage_scope,status`.

- [ ] **Step 1: Write tests.** Use explicit null arrays rather than randomly hoping for small p-values. Calibration fixtures contain disjoint root IDs and same-architecture certified independents, siblings, and grandparents tagged by estimand. Conformal fixture has ten calibration nonconformities `[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]` in one family.

```python
def test_exact_permutation_uses_ties_and_finite_floor():
    out = exact_shortlist_permutation(0.9, np.asarray([0.1,0.9,0.2]))
    assert out["p_value"] == 0.5
    assert exact_shortlist_permutation(1.0, np.arange(9)/10)["p_value"] == 0.1

def test_full_pool_multiplicity_is_not_shortlist_size():
    p = np.asarray([1e-8, 0.8])
    assert query_control(p, alpha=0.05, pool_size=2, validity="exchangeable").tolist() == [True,False]
    assert query_control(p, alpha=0.05, pool_size=5000, validity="exchangeable").tolist() == [False,False]

def test_settings_and_family_transfer_do_not_fake_calibration(calibration_fixture):
    assert calibration_fixture.apply(tier="t2")["status"] == "uncalibrated"
    assert calibration_fixture.apply(row_block=128)["status"] == "uncalibrated"
    assert calibration_fixture.apply(sketch_k=32)["status"] == "uncalibrated"
    assert calibration_fixture.apply(family="unseen")["guarantee"] == "transfer_unvalidated"
```

Add `test_unvalidated_or_gpd_p_values_rejected` (both validity modes raise `ValueError`), `test_sibling_not_same_root_negative`, `test_gpd_thin_tail_falls_back` (<50 exceedances gives `insufficient_tail`), `test_stouffer_never_used` (duplicate views do not multiply n_pairs), `test_unknown_truth_excluded`, `test_retrieval_uses_full_pool_denominator`, and `test_conformal_unseen_family_abstains` (returns all candidates plus `not_in_reference`, `status="uncalibrated_family"`). For the ten-score fixture, α=.1 gives the tenth score 0.5 as finite-sample quantile; labels with probability ≥0.5 enter, including `not_in_reference` when its probability qualifies. `ceil((n+1)*(1-alpha))>n` means threshold infinity, not capped confidence.

- [ ] **Step 2:** Run `python -m pytest tests/lineage/test_calibration.py tests/lineage/test_conformal.py -q`; expected missing modules.
- [ ] **Step 3:** Calibrate median cosine at k64 directly, never combine tensor p-values. Null strata: same-root uses known-independent pretrains; ancestor adds siblings; parent adds siblings and non-immediate ancestors. Estimate these separately; for a composite negative null use the maximum stratum p-value, report which strata exist, and abstain when a required stratum has no valid calibration. Samples sharing a query/seed are not independent draws: report cluster counts, fit uncertainty by root/seed blocks, and require a justified conditional exchangeability design for exact p-values. Pair count alone never certifies that design.

GPD: fit exceedances above the empirical 90th percentile with loc fixed zero; require ≥50 exceedances, finite parameters/support, and record held-out tail QQ/coverage diagnostics and bootstrap intervals. Retain the empirical estimate and p-value floor alongside model-based extrapolation. Fitted-tail p-values are labeled approximate; do not convert them to supposedly valid e-values merely because a diagnostic passed. Use exact shortlist label permutations as the conservative fallback when exchangeability can be justified; otherwise report uncalibrated. This may lack power at pool size 5k; that is a measured result, not permission to claim the target was reached.

For valid superuniform p-values, `e=0.5*p**(-0.5)` is a fixed p-to-e calibrator. e-BH tolerates dependence across candidates but still requires marginally valid e-values; record that assumption. Retrieval selection can only zero e-values, never reduce m to the shortlist. Test family transfer explicitly: train/freeze on one family, calibrate on another partition, evaluate on a held-out family; report empirical coverage/FDR and distinguish it from finite-sample exchangeable guarantees. Mondrian sets use the `(n+1)` quantile within supported families. New families have no distribution-free guarantee; return full/abstaining sets until labeled calibration supports them. Weighted conformal is deferred because the checklist permits Mondrian **or** weighted sets. The upstream X9 JSON remains unchanged; the engine artifact is separate and references its hash/settings. Export a compatible X9 display calibration only through its existing writer.

- [ ] **Step 4:** Run both files then package tests once, expected PASS. A tiny dry run exhausts the four candidate-label permutations, not GPU work.
- [ ] **Step 5:** Suggested commit: `lineage: empirical calibration query control and open sets`.

### Task 7 (E8): Frozen-schema edge output and local CLI

**Files:** Create `lineage/engine.py`, `lineage/io.py`, `lineage/cli.py`, `tests/lineage/test_engine.py`, `tests/lineage/test_upstream.py`, `configs/lineage/method.json`, `docs/operations/lineage.md`.

**Interfaces:** `infer(records: list[FingerprintRecord], *, config: dict, calibration: dict|None=None) -> tuple[pd.DataFrame,dict]` returns exact LINEAGE_COLUMNS and diagnostics; `write_edges(edges: pd.DataFrame, out: Path, *, provenance: dict) -> None` atomically writes Parquet and a JSON provenance sidecar; `main(argv: list[str]|None=None) -> int` implements `python -m lineage.cli infer --manifests DIR --fingerprints DIR --data-root ROOT --config FILE --calibration FILE --out FILE` and `dry-run --out DIR`.

- [ ] **Step 1: Write tests.** `fail_next_replace(monkeypatch)` is a test helper that patches the first output-publication rename to raise `OSError`, retaining other tempfile operations. `valid_edges` is one exact-schema detected parent row from the synthetic fixture; `method_config` is the default development JSON with no null and score threshold 0.0. The synthetic fixture produces a frozen child-parent inference candidate and no fitted calibration. A patched atomic rename raises before replacing a preexisting valid Parquet file.

```python
def test_inferred_rows_keep_frozen_schema(synthetic_pool, method_config):
    edges, diagnostics = infer(synthetic_pool.records, config=method_config)
    assert list(edges.columns) == LINEAGE_COLUMNS
    assert set(edges.evidence_tier) <= {"detected"}
    assert not edges.calibrated.any()
    assert set(edges.edge_type) <= {"finetune","adapter","merge","quantized","abliteration","unknown"}
    assert edges.method_version.eq("p1-0").all()

def test_interrupted_output_retains_previous(valid_edges, tmp_path, monkeypatch):
    out = tmp_path / "edges.parquet"
    write_edges(valid_edges, out, provenance={"input_hash":"a"})
    before = out.read_bytes()
    fail_next_replace(monkeypatch)
    with pytest.raises(OSError):
        write_edges(valid_edges, out, provenance={"input_hash":"b"})
    assert out.read_bytes() == before
```

Add `test_no_common_views_is_a_diagnostic_not_an_edge`, `test_direction_requires_asymmetric_evidence`, `test_lineage_into_zoo_build` (tiny upstream universe plus `--lineage_edges`; check rows preserved), `test_resume_rejects_changed_input`, and the local loader tests deferred from Task 1. Integration requires X6/X8/Z4/Z6 committed and merged by the person; synthetic code must not depend on another track's uncommitted files.

- [ ] **Step 2:** Run `python -m pytest tests/lineage/test_engine.py tests/lineage/test_upstream.py -q`; expect missing module/functions before implementation.
- [ ] **Step 3:** Orchestrate retrieval→median similarity→merge/typing/stacked evidence→estimand-specific calibration→optional directed assembly. Separate symmetric same-root scores from directional ancestor/parent scores; unresolved directions stay diagnostic and reduce coverage. `score` always increases with evidence for the stated estimand and need not be probability. No claim nesting is inferred by equating all three scores: closure can propagate supported parent→ancestor→same-root claims with evidence references and conservative calibration status. Null p-values stay null, calibrated false when no valid matching null. Export detected edges only at the configured frozen reporting threshold; write every abstention in diagnostics with query and reason.

Use exact upstream schema order and Parquet types, including empty tables; `evidence` is canonical JSON with `space`, `calibration_mode`, `reason`, merge/typing fields and provenance references. `method_version="p1-0"` is bumped whenever the method changes; a hash also captures the full config. Atomic sidecar bundle uses a temporary directory renamed into a run directory; the authoritative edge file is published only once the bundle is complete. Document an exact tiny-scale local command and how to resume safely.

- [ ] **Step 4:** Run named tests then package tests once, expected PASS. Run `python -m lineage.cli dry-run --out "$ZOO_DATA/runs/lineage-smoke"`; no weights/network, expected valid edges plus diagnostics and hashes.
- [ ] **Step 5:** Suggested commit: `lineage: local inference CLI and zoo-compatible edge artifacts`.

### Task 8 (E9): Evaluation, evidence/byte curves, freeze, and prospective protocol

**Files:** Create `lineage/evaluate.py`, `lineage/baseline.py`, `tests/lineage/test_evaluate.py`, `configs/lineage/evaluation.json`; extend `docs/operations/lineage.md`.

**Interfaces:**
- `Baseline.run(self, query_keys: list[str], candidate_keys: list[str], *, budget_bytes: int, seed: int) -> tuple[pd.DataFrame,pd.DataFrame]` returns exact-schema prediction rows and per-query costs `query_key,bytes_read,seconds,status,reason`; adapters live with C's baselines, not new implementations here. Cost includes reference construction amortized and unamortized, additional verification bytes, and query bytes separately.
- `evaluate(predictions: pd.DataFrame, truth: pd.DataFrame, universe: pd.DataFrame, costs: pd.DataFrame, *, config: dict) -> dict`. Truth columns `query_key,candidate_key,estimand,label,truth_tier,root_group,split`; label is 0/1/null (unknown). Universe requires `query_key,candidate_key,eligible`; optional nullable `truth_parent,truth_ancestor,truth_same_root,truth_tier` carry explicit label sidecars for scorer interchange and do not modify frozen lab tables. Incomplete Hub truth remains null rather than inferred negative. Complete lab same-root truth means nonempty intersection of the roots reachable from each model; a merge inherits all source roots, so same-root is not assumed transitive. Ancestor is directed transitive closure and parent is a direct parent-role edge; merge reference-only base roles are not parent-role edges unless independently present. Public lab metrics call `lab.score.score_lab(models,edges,predictions,manifests,universe,threshold=...)`; Hub metrics use only explicit labeled positives/negatives and report unknown coverage.
- CLI `evaluate --predictions FILE --truth FILE --universe FILE --costs FILE --config FILE --out DIR`, and `freeze --config FILE --inputs FILE --out DIR` creates immutable canonical config/input hashes/method hash before evaluation. Artifacts: `metrics.json`, `curves.parquet`, `coverage.parquet`, `protocol.json`, `input_hashes.json`; figures can be generated from these in the paper phase.

- [ ] **Step 1: Write tests.** `evaluation_fixture` has two queries, each one positive and one certified negative; scores .9/.8 and .7/.1 respectively. At threshold .75, one false parent and one true parent across two queries gives .5 false parents/query and TPR .5. One missing prediction is an abstention with score -inf for ranking, not an implicit true negative. `truth_tier` is explicit for every row.

```python
def test_false_parents_denominators_and_missing_rows(evaluation_fixture):
    out = evaluate(**evaluation_fixture.inputs)
    assert out["at_frozen_threshold"]["false_matches_per_query"] == 0.5
    assert out["at_frozen_threshold"]["tpr"] == 0.5
    assert out["at_frozen_threshold"]["n_queries"] == 2
    assert out["unknown_truth_pairs"] == 0

def test_unknown_hub_truth_is_not_a_negative(evaluation_fixture):
    inp = evaluation_fixture.with_unknown_negative()
    out = evaluate(**inp)
    assert out["unknown_truth_pairs"] == 1
    assert out["negative_pairs"] == 1
```

Add `test_duplicate_groups_and_merge_roots_do_not_cross_split`, `test_root_holdout_includes_all_merge_parents`, `test_pool_curves_include_unretrieved_candidates`, `test_test_tpr_threshold_does_not_tune_deployment` and `test_freeze_is_immutable`. Diagnostic false matches at TPR .9 uses the **highest** score threshold reaching recall ≥.9 (ties included) on the labeled evaluation ROC as a descriptive benchmark; frozen deployment-threshold performance is reported separately and no threshold feeds back into the method. AUROC is undefined/null with explicit status when either class is absent; direction accuracy excludes symmetric same-root pairs and reports the number resolved/eligible. Coverage and abstention include all eligible queries, not only returned edges.

- [ ] **Step 2:** Run `python -m pytest tests/lineage/test_evaluate.py -q`; expect missing module.
- [ ] **Step 3:** Produce every metric by estimand and truth tier: AUROC, false parents/query at TPR .9, deployed error/TPR, direction accuracy, candidate and query coverage, abstention, bytes/answer. Candidate pool sizes: 100, 500, 1000, 5000, 10000, 20000 up to the available pool; fixed nested sampled pools keep positives and use the same queries across methods. Show extrapolation separately with calibration-model uncertainty, never as measured results. Small fixture curves use 2/4/8 candidates and disclose toy scale.

Evidence ladder uses identical queries: metadata→T0→T1→T2→black-box (200 labeled pairs, owned by C). Include Trace, Spectral Signatures, modelDNA at matched bytes, Witness Overlap direction, 2602.00772, MoTHer and metadata imputation. Missing baseline artifacts are reported `not_run`, not zero error. Split families before fitting calibration; report each transfer direction. Pre-G0 development uses public lab only. At G0 freeze method/config and hash prospective predictions for models created after 2026-12-04; pin Hub timestamps/commits from A's metadata and never tune on prospective outcomes.

For blind evaluation, C applies the frozen method/scorer in an isolated evaluation step and returns aggregate metrics; B does not need to inspect private checkpoint paths or truth. Do not pre-open the blind directory to plan this.

Report numerical threshold verdicts exactly: G0 parent-v-independent AUROC ≥.98 and T1 no more than .01 below T2; related cosine Spearman ≥.9; decision agreement ≥.9; false matches/query ≤.05 at TPR .9 for a 5k pool; parent-v-sibling and parent-v-grandparent AUROCs. E3 coefficient target ±.05 on eligible lab merges includes denominator/nonidentifiable count. Nov27 tree thresholds >.20 violations or >.2 nRF flag a person decision. Dec18 P1 needs ≥2× fewer false matches/query at TPR .9 **or** ≥10× fewer bytes at equal AUROC versus the best baseline. Absent pool/baseline/metric yields `NOT_EVALUABLE`, not PASS. A person decides pivots and gates.

- [ ] **Step 4:** Run test file then package tests once, expected PASS. Tiny dry run evaluates generated fixtures offline. Lead runs the full Zoo-Keeper suite once for the batch.
- [ ] **Step 5:** Suggested commit: `lineage: frozen evaluation protocol and scaling curves`.

## Runs, estimates, and launch boundary

These are planning estimates, not measurements. Fingerprints with V views and k64 consume about `4*V*64²` bytes/model before metadata; 5k models ×200 views is about 16.4 GB, and k128 is four times that. Stream queries; do not require the entire collection's expanded float64 sketches in RAM. Exact retrieval's compact fp16 index is about 65,536 bytes/model (32 views ×1024 entries ×2), or 328 MB for 5k; exact rescoring and merge fitting dominate CPU time. Benchmark tiny local fixtures first and project measured throughput. No weight training, paid APIs, or GPUs are required by the core engine.

| Work | Command after implementation | Inputs / outputs | Estimated cost and launcher |
| --- | --- | --- | --- |
| Synthetic smoke | `python -m lineage.cli dry-run --out "$ZOO_DATA/runs/lineage-smoke"` | generated inputs; relative diagnostic artifacts | <10 MB, <1 min CPU, <10 MB output, 0 GPU-hours; agent |
| Public lab inference | `python -m lineage.cli infer --manifests "$ZOO_DATA/runs/lab-v0/manifests" --fingerprints "$ZOO_DATA/runs/lab-v0/t1" --data-root "$ZOO_DATA" --config configs/lineage/method.json --calibration "$ZOO_DATA/runs/p1-calibration/engine.json" --out "$ZOO_DATA/runs/p1-lab/edges.parquet"` | pinned existing T1 fingerprints; method/null hashes in sidecar | ~0.5–3 GB read, ~5–60 min CPU estimate, <1 GB output, 0 GPU-hours; agent only after measured projection <2 h |
| Full P1 scaling | same `evaluate` CLI with frozen prediction/truth/universe/cost Parquet paths recorded by E9 | 2–5k queries, 5–20k references; immutable protocol/output bundle | ~10–100 GB existing fingerprints across methods, ~2–12 h CPU estimate, ~1–20 GB outputs, 0 GPU-hours core engine; **[HUMAN] launches** if >2 h projection |
| Exact local shortlist blocks | inference config enables `block_verification` on explicit pinned local checkpoint pairs | plain matrices only; aggregate diagnostics, never raw blocks | read bytes = sum over pairs/layers `2*b*columns*dtype_bytes`; enforce ≤100 distinct weight models and <2 h, otherwise **[HUMAN] launches** |
| T2/black-box evidence | A/C's frozen commands in their launch queue | existing A/C outputs consumed by E9 | no duplicate launch here; the person launches their large jobs |

Before any large run is queued, its implementer replaces the throughput range with a measured projection, records the exact argv and pinned input hash manifest in `STATUS.md`, and records verification/resume (`protocol.json` hash plus complete expected rows; rerun same immutable config, skipping only outputs whose hashes match). This draft does not pretend nonexistent real pins or output files are ready to launch.

## After this plan

A person reviews L1/E1 before code begins (D10.4). E2 can then start on synthetic fingerprints while A develops upstream packages; E3–E7 proceed on synthetic/public lab data. E8 real imports and zoo integration wait for A's merged interfaces, and the public lab depends on L1/L12. The blind set remains C's responsibility through evaluation. No frozen interface change is proposed.

Parked: capability priors, deployment identity, adversarial guarantees, up-scaling DTW/pruning detectors beyond reporting unsupported gaps, latent-ancestor neighbor joining, weighted conformal, ingestion, hosted service, MCP, and new papers. These are not needed for E1's committed engine. The main scientific risk remains power and transfer: valid small-sample nulls may not support 5k-pool error targets, and merge coefficients may be nonidentifiable. Report those results rather than claiming guarantees or weakening targets.

**Review decisions:** approve this concrete implementation plan; defaults above (e-BH, Mondrian sets, conservative empirical/permutation mode, projected plus optional exact block tests) need no separate choices. Actual gate/pivot decisions and launches remain with the person under the existing handoff. No other inputs are needed to start E2 after plan review.
