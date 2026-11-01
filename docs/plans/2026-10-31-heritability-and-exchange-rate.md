# Heritability and First Exchange-Rate Pass Implementation Plan

> **For agentic workers:** Execute one task at a time using the lean loop in `handoff/PROMPTS.md`. The person reviews this plan before implementation. Agents never stage, commit, merge, or push; commit messages below are suggestions for the person.

**Status:** S1 approved by the owner on 2026-10-31. Code and draft committed and merged locally at `0a457b9` on the owner's request (combined suite416 tests pass); empirical S3/S4 estimates remain unavailable in the frozen partial audit because evidence is missing.

**Goal:** Measure per-skill familial inheritance with measurement error accounted for, then price the audited September predictions in equivalent evaluation items for estimation and selection. Produce a reproducible first exchange-rate table and an ICLR 2027 workshop draft, including unavailable and negative results.

**Architecture:** A small offline package, `studies/heritability/`, assembles versioned score and graph tables, constructs family/network covariance, fits REML and observed-parent regressions, audits existing predictions, and writes reports. Heritability uses score/graph inputs independently of historical run outputs; only the exchange-rate pass depends on September predictions. Code and protocol live in Zoo-Keeper, immutable small source snapshots under `data/heritability/`, and all generated artifacts under `$ZOO_DATA/studies/heritability/<run>/`.

**Tech Stack:** Python 3.10 in `esd_ind`, numpy, scipy, pandas, pytest, and pyarrow only if Parquet output is chosen in a later integration. Use scipy's optimizer and Cholesky solves for the small REML fit; no new mixed-model framework. Pin imported dependencies to their installed versions in `requirements.txt`; install only missing packages without changing existing versions, and log any install under D7.5.

**Spec:** `docs/research/2026-10-27-merged-research-plan.md` §§5.1, 5.4, 9 ideas 2–3, 10, 11 (P2/workshop row), and 12; `handoff/CHECKLIST.md` S1–S4 and S7; `handoff/DIRECTIVES.md` D2, D4, D6–D11; `handoff/GUIDANCE.md` §§3.7, 4.2–4.5. Paths below are relative to Zoo-Keeper unless explicitly prefixed by another repository.

## Source inventory and boundaries

Inspected at planning time; presence is not evidence of revision verification:

- `Model-Research/data/sampled_leaderboard.csv` (~1.2 MB) has `Model sha`, `Precision`, `eval_name`, and `IFEval Raw`, `BBH Raw`, `MATH Lvl 5 Raw`, `GPQA Raw`, `MUSR Raw`, `MMLU-PRO Raw`. Raw values inspected are on [0, 1]. Its `Generation` field is not sufficient evidence that a row belongs to leaderboard v1.
- `Model-Research/data/sampled_metadata.csv` (~0.4 MB) has `model_id`, `source_model`, `base_model_relation`.
- `Model-Research/data/sampled_model_subgraph.json` (~9 MB) has `nodes` and `edges`; edges use `source` for parent and `target` for child. Model Atlas relationships lack commit evidence and remain **declared**.
- `Model-Research/data/curated/phase2_v3/model_zoo_phase2_leaderboard_raw.csv` (~1.3 MB) is a potential supplementary score snapshot; avoid counting overlapping records twice.
- `Model-Research/configs/htsr_combined_study_v4.json` freezes historical seed 17, 2,000 bootstrap replicates, and five outer folds. Copy as provenance, not as permission to refit old predictors.
- `Model-Research/spectral_research/study/performance.py` writes `actual`, `prediction`, `feature_block`, and `outer_fold`; its modern targets use normalized task columns and percentage-point semantics. Dividing an old prediction by 100 does **not** undo floor-clipped normalization. Do not relabel these as Raw predictions.
- A v1 per-task source, verified benchmark denominators/variance metadata, and September run outputs have not been established. S2 discovers and audits them; no invented file names, item counts, commits, or historical metrics.

This plan covers S2, S3, S4 and S7. S5 gets a proposed adapter contract below, but its port and empirical evaluation await Track A artifacts and agreement with the reviewed E1 harness. S6 blind transforms, S8 queries, S9 paid/card audit and S10–S11 baselines remain their own checklist work. T1 weight-distance/edge-type extensions wait for their committed dependencies and never block the score-only first pass.

## Global Constraints

- Never read or print `.env` or token values. Load the workspace environment only by sourcing it; activate `esd_ind`. All tests are offline, generate fixtures in `tmp_path`, and neither access Hub nor read real runs.
- Port or copy needed source material from frozen Model-Research; never import from it or edit it. Only the four explicitly selected small source tables/graph and historical config above may be copied after hashes are recorded. Do not copy checkpoints, spectra, run outputs, or predictions into git.
- Model keys remain `<repo_id>@<40-character SHA>` or `lab/<model_id>@local:<sha256>`. Never invent a revision for a bare Atlas ID. Unpinned source records remain in an audit table rather than receiving a fabricated model key.
- Preserve deployment identity: `model_key`, `benchmark`, `task`, `harness`, `precision`, and `quantization` identify a score. Casefold repository matching only; preserve original IDs. Missing/ambiguous deployment evidence yields `unverified`, not an assumed bf16 match. Retain precision variants as different observations and avoid treating them as independent model replicates.
- `revision_verified` is true only when two independent records of the measured checkpoint match: the score's SHA and the pinned model/input manifest SHA. A score's own SHA alone is `recorded`, not a verified join. Report numerators and denominators separately for score-to-model and prediction-to-score joins.
- Use per-task Raw accuracy on [0, 1], never normalized/floor-clipped scores, and never a six-task mean as a binomial observation. v1 and v2 tasks/harness versions stay separate. Out-of-range predictions remain visible; out-of-range observations are invalid.
- Measurement variance needs evidence. For an ordinary binary accuracy with documented item count n, use `v = p*(1-p)/n`. Composite, macro-averaged or correlated measures require documented sampling variance/item-level covariance; otherwise emit `variance_unavailable` and exclude from measurement-corrected fits. Do not guess an effective n for IFEval/BBH/MUSR. Retain descriptive summaries for unsupported tasks.
- Primary scale is raw accuracy. Logit is a prespecified sensitivity: use `p_tilde=(successes+0.5)/(n+1)` only for documented binomial counts; recompute corrected-binomial variance `v_tilde=p_tilde*(1-p_tilde)/(n+1)` and then delta-method logit variance `v_tilde/[p_tilde*(1-p_tilde)]**2`. Never carry a raw endpoint zero variance into the logit fit; label the correction. Do not logit-transform unavailable composite variances. Endpoint sensitivity reports raw results alongside logit results.
- Group duplicate/re-upload evidence before splits; hold whole roots out for cross-family analyses. Multi-parent links union roots for splitting; never assign a merge to an arbitrary parent to obtain a cleaner holdout. Report split coverage/exclusions and historical leakage that cannot be repaired without refitting.
- Freeze one paper protocol with input sha256, schema mappings, seeds, splits, model choices, optimizer tolerances and bootstrap count before final analysis. Historical predictions retain their original frozen protocol and are reused descriptively when provenance is incomplete; do not tune on held-out/prospective data or silently rerun historical predictors.
- Every write uses a temporary sibling then rename. Stored paths are relative to `$ZOO_DATA` or the documented repository source root; never store machine-specific paths. Source snapshots are immutable, sorted outputs deterministic, timestamps only in run metadata. Resume verifies hashes before reusing a stage.
- Declared, verified and detected lineage stay distinct; detected evidence remains `uncalibrated` until a matching calibration exists. No changes to fingerprint layout, lab tables, `LINEAGE_COLUMNS`, or zoo score/prediction tables.
- Analysis reads existing tables, not model weights. Tiny dry runs use no GPUs and finish in minutes. Any full job projected beyond two hours is **[HUMAN] launches**, as is paid work; queue exact command, pinned inputs, bytes/time/storage/GPU-hours, verification and resume instructions under `handoff/STATUS.md`. No paid API, model evaluation or publication is part of this first pass.

## Review Focus

1. Same repo at two commits/precisions, missing SHA, or only self-attested SHA: no false verified join (Tasks 1, 4).
2. Composite scores, p=0/1, unknown n, and zero-byte files: explicit coverage and unavailable variance; no numerical fiction (Tasks 1–3).
3. Multi-parent graph with missing coefficients, cycles and crossing roots: valid covariance or documented abstention, never a silently simplified tree (Task 2).
4. Historical normalized predictions and leaking folds: no Raw conversion or out-of-sample claim without evidence (Tasks 4–5).
5. Correlated pair errors, a constant target and zero prediction error: correct selection numerator, undefined statistics labeled, stable deterministic output (Tasks 5–6).

## Data contracts

Use pandas tables, CSV for the first pass, and JSON for audits; no dependency on the unimplemented zoo build.

- `scores.csv`: `record_id,model_key,repo_id,revision,benchmark,task,harness,precision,quantization,accuracy,n_items,measurement_variance,variance_source,revision_status,revision_verified,source_hash,source_row`. Empty identity/variance fields are legal only with an audit reason. One row per deployment/task observation; identical records across source snapshots are deduplicated before assigning a deterministic unique `record_id` from the canonical deployment/task key. Keep sorted source-hash/row provenance for every collapsed copy. Conflicting values at that same identity are all excluded with a reason; different precision/revision observations remain separate.
- `edges.csv`: `parent_id,child_id,parent_key,child_key,relation,evidence_tier,coefficient,coefficient_source,branch_length,branch_length_source,source_hash`. Bare IDs survive; keys remain empty if unpinned. Coefficients must be finite, nonnegative and sum to one **for this Brownian inheritance model**. This is not a restriction on P1 unconstrained merge regression; signed/non-normalized recipes are unsupported by the primary Brownian model and reported separately.
- `groups.csv`: `model_key,duplicate_group,root_group,split_group,group_evidence`. `split_group` is the connected union of shared root ancestry and duplicate groups; related deployments of a checkpoint stay together.
- `predictions.csv`: `model_key,benchmark,task,harness,precision,quantization,method,setting,fold,actual,prediction,prediction_scale,prediction_semantics,split_group,revision_verified,provenance_status,source_hash`. `setting` is `within_family` or `root_holdout` only when the old split audit supports it; otherwise `historical_unverified`. Method includes paired no-weight predictors on identical held-out rows.
- `exclusions.csv`: `stage,record_id,reason,detail`; `coverage.json` has counts by task/source/reason, verified fractions and denominator definitions.

## Task 1: Source assembly, score identity and variance audit (S2)

**Files:** Create `studies/__init__.py`, `studies/heritability/__init__.py`, `studies/heritability/data.py`, `studies/heritability/protocol.py`, `configs/heritability_v1.json`, `configs/heritability_tasks.json`, `tests/heritability/conftest.py`, `tests/heritability/test_data.py`, `requirements.txt`; copy selected immutable sources to `data/heritability/` only during implementation.

**Interfaces:**
- `assemble_scores(sources: list[dict], pinned_models: pd.DataFrame, task_registry: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]`: returns scores, exclusions, coverage.
- `inventory_inputs(paths: list[Path], *, data_root: Path) -> dict`: sizes/hashes, zero-byte missing reasons; never opens `.env` or traverses arbitrary home/cache folders.
- Task registry: source-column mapping, benchmark/task/harness semantics, scale, binary/composite kind, denominator/variance evidence path+hash, and deployment mapping. Missing registry evidence remains an explicit unavailable entry.
- `freeze_protocol(config: dict, inputs: dict, out: Path) -> str`: canonical sorted JSON, returns sha256; refuse existing unequal freeze. Draft config uses seed `20261028`, 2,000 cluster bootstrap replicates and a tiny `--dry-run` override recorded separately. Historical seed stays 17 in historical provenance.

- [x] **Step 1: Write failing tests.** `score_sources(tmp_path)` writes three tiny CSVs: raw v2 with `Model sha`, `Precision` and normalized distractors; a second source with identical/contradicting rows; a v1 source with explicit task metadata. `pins` supplies independent manifest SHA/deployment records. `registry` declares a binary n=100 task and one unsupported composite. Assert:

```python
def test_identity_raw_and_variance(score_sources, pins, registry):
    scores, excluded, audit = assemble_scores(score_sources, pins, registry)
    a = scores.query("record_id == 'a'").iloc[0]
    assert a.accuracy == 0.2                 # normalized distractor was 0.0
    assert a.measurement_variance == pytest.approx(0.0016)
    assert a.revision_verified
    assert not scores.query("revision_status == 'recorded'").revision_verified.any()
    assert set(scores.precision) >= {'bf16', 'fp16'}
    assert 'conflicting_score' in set(excluded.reason)
    assert 'variance_unavailable' in set(excluded.reason)
    assert audit['verified_joins'] <= audit['eligible_joins']

def test_zero_bytes_and_protocol(tmp_path):
    p = tmp_path / 'empty.csv'; p.write_bytes(b'')
    inv = inventory_inputs([p], data_root=tmp_path)
    assert inv['files'][0]['status'] == 'missing_zero_bytes'
    out = tmp_path / 'frozen.json'
    assert freeze_protocol({'seed': 20261028}, inv, out) == freeze_protocol({'seed': 20261028}, inv, out)
    with pytest.raises(ValueError, match='frozen'):
        freeze_protocol({'seed': 1}, inv, out)
```

Also assert invalid observed range excluded, unknown item count not filled, n=0 rejected, v1/v2 kept separate, missing precision unverified, missing pins do not self-verify, and conflicting revisions never collapse. Fixtures must specify record IDs rather than use real model data.

- [x] **Step 2: Run** `python -m pytest tests/heritability/test_data.py -q`; expect import failure for absent `studies.heritability`.
- [x] **Step 3: Implement** the contracts. Copy only the selected ~12 MB sources atomically and record original relative paths/hashes. Inventory v1 candidates from explicitly supplied source directories; do not infer v1 from `Generation`. Allow a caller-supplied v1 source with an explicit column/semantics map. Audit missing denominators and independent pin evidence in `coverage.json`; useful v2 descriptive work proceeds even if v1 is absent. Copy the historical config as provenance only.
- [x] **Step 4: Run** the task file, then `python -m pytest tests/heritability -q` once. No broad suite yet. Suggested message: `feat(study): assemble raw scores with identity and variance audits`.

## Task 2: Declared ancestry, split groups and network covariance (S3)

**Files:** Create `studies/heritability/ancestry.py`, `tests/heritability/test_ancestry.py`.

**Interfaces:**
- `assemble_edges(atlas: dict, metadata: pd.DataFrame, pins: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]`.
- `build_groups(models: pd.DataFrame, edges: pd.DataFrame, duplicates: pd.DataFrame) -> pd.DataFrame`.
- `network_covariance(nodes: list[str], edges: pd.DataFrame) -> np.ndarray`: rows/columns exactly `nodes` order.

For a topological node order define inheritance row `a_i = sum_j w_ij*a_j + e_i*sqrt(t_i)`; roots have their own independent unit innovation, a single-parent edge has w=1. `K=A@A.T`. Unit edge increments are a prespecified initial model and labeled `unit`, not measured evolutionary time. Known merge coefficients are used only with provenance and the convex constraints above. Unknown merge coefficients exclude that node **and affected descendants** from the primary network fit, with counts; a separately labeled equal-weight sensitivity may include them. Never exclude them from raw descriptive tables. Normalize K by its mean diagonal for reporting variance fractions; keep observed ancestors in the covariance.

- [x] **Step 1: Write failing tests.** A root r, children a/b, and merge m=(a+b)/2 have innovation lengths 1. Add a declared no-coefficient merge u and a descendant v in separate fixtures. Test:

```python
def test_network_covariance():
    edges = pd.DataFrame([
        {'parent_id':'r','child_id':'a','coefficient':1.,'branch_length':1.},
        {'parent_id':'r','child_id':'b','coefficient':1.,'branch_length':1.},
        {'parent_id':'a','child_id':'m','coefficient':.5,'branch_length':1.},
        {'parent_id':'b','child_id':'m','coefficient':.5,'branch_length':1.},
    ])
    k = network_covariance(['r','a','b','m'], edges)
    assert np.allclose(k, [[1,1,1,1],[1,2,1,1.5],[1,1,2,1.5],[1,1.5,1.5,2.5]])
    assert np.linalg.eigvalsh(k).min() >= -1e-12
```

Assert source→target direction, observed roots retained, cycles rejected, missing coefficients reported for u/v, negative coefficients rejected, known shared roots unite split groups, and a duplicate across two roots unites both groups before any split.

- [x] **Step 2: Run** `python -m pytest tests/heritability/test_ancestry.py -q`; expect absent ancestry module.
- [x] **Step 3: Implement** the recurrence and grouping. Atlas edges stay declared even when endpoint score revisions verify; known commits do not verify the relationship. No edge type inferred from a repository name. Multiple deployment rows map to the same latent checkpoint; use an explicit primary deployment from the frozen config, with others in sensitivity reports.
- [x] **Step 4: Run** this task file, then the study package once. Suggested message: `feat(study): add ancestry groups and Brownian merge covariance`.

## Task 3: REML, observed-parent regression and per-skill summaries (S3)

**Files:** Create `studies/heritability/statistics.py`, `tests/heritability/test_statistics.py`.

**Interfaces:**
- `reml_nll(y: np.ndarray, x: np.ndarray, v: np.ndarray) -> float`.
- `fit_reml(y, x, kernel, measurement_variance) -> dict` with variance components, fixed effects, converged/boundary flags, n/families and `familial_fraction`.
- `parent_offspring(pairs: pd.DataFrame) -> dict` with naive and corrected slopes, intercept, variance denominator, coverage and status.
- `skill_summary(actual: np.ndarray, predicted: np.ndarray) -> dict` with MAE, RMSE, MAE/sd and R²; constant targets produce unavailable scale-free metrics with a reason.

Primary family model: `V=tau2*(Z@Z.T)+sigma2*I+D`, `X=[intercept, centered log(parameter_count)]` when parameter count is documented for all retained rows; otherwise intercept-only, labeled and with the same selection across compared models. Per-edge-type effects wait for usable relation evidence. Separately fit `V=tau2*K+sigma2*I+D` on the supported network subset. D is known score sampling variance, not absorbed into an estimated residual. Minimize the restricted Gaussian objective using Cholesky solves:

`0.5*(logdet(V)+logdet(X.T@inv(V)@X)+y.T@P@y+(n-p)*log(2*pi))`.

Estimate nonnegative components, including explicit boundary solutions; report optimizer failure as unavailable, never substitute a point estimate. Familial fraction is the kernel-normalized latent fraction `tau2/(tau2+sigma2)`, with covariance model named, not a causal heritability claim.

Observed-parent regression uses only matching task/harness and verified compatible deployment pairs, gives the observed parent score rather than an imputed family average, and first reports ordinary slope. The measurement-corrected moment slope is `cov(parent,child)/(var(parent)-mean(parent_variance))`, using population moments consistently; report unavailable when denominator ≤0. Repeated parents and related errors require cluster bootstrap by split_group. Correction assumes independent parent/child measurement errors; if paired item-level error covariance is available subtract its average from the numerator, otherwise label this assumption and include naive sensitivity. Do not treat merges as single-parent pairs.

- [x] **Step 1: Write failing tests.** Compare `reml_nll` on a hand-built positive-definite 4×4 V to the expression above computed directly with numpy, absolute tolerance 1e-10. Use 40 families × 5 offspring with RNG seed 20261028, independent normal family effects of variance 0.02, residual variance 0.01 and known D=0.002*I; X is an intercept, kernel is Z@Z.T. Assert familial fraction within 0.15 of 2/3, convergence and nonnegative components. A zero-family-variance fixture must select/report a boundary rather than fail. Test exact moment correction:

```python
def test_parent_measurement_correction():
    pairs = pd.DataFrame({'parent_accuracy':[.2,.4,.6,.8],
                          'child_accuracy':[.3,.4,.5,.6],
                          'parent_variance':[.01]*4,
                          'error_covariance':[0.]*4})
    result = parent_offspring(pairs)
    assert result['naive_slope'] == pytest.approx(.5)
    assert result['corrected_slope'] == pytest.approx(.625)
```

Additional tests: singular X rejected, n≤p unavailable, D shape/negative values rejected, insufficient corrected denominator unavailable, missing variances excluded/countable, binomial endpoint logit finite under the explicit correction, composite variance not fabricated, constant-target scaled metrics unavailable, negative R² preserved.

- [x] **Step 2: Run** `python -m pytest tests/heritability/test_statistics.py -q`; expect absent statistics module.
- [x] **Step 3: Implement** the contracts and deterministic cluster bootstrap (2,000 final replicates, 20 tiny dry-run replicates). Fit/report per task, benchmark, relation tier and scale; no pooled normalized score. Bootstrap complete split groups, retain failed replicate counts, and show percentile 95% intervals only when at least 90% of replicates succeed (analysis rule, not a project pivot threshold). Describe smaller support without inventing precision. Report measurement variance fraction/noise ceiling separately from latent familial fraction. Use the same target set for scale-free skill comparisons and report excluded support.
- [x] **Step 4: Run** the task file, then the study package once. Suggested message: `feat(study): estimate familial variance with known measurement noise`.

## Task 4: Locate and audit September predictions without refitting (S2/S4)

**Files:** Create `studies/heritability/historical.py`, `tests/heritability/test_historical.py`.

**Interfaces:**
- `discover_runs(search_roots: list[Path]) -> pd.DataFrame`: bounded search of supplied roots for `research_runs/` and manifest/config/prediction artifacts, returning candidate path, size, hashes and status; paths relative to supplied roots.
- `audit_predictions(run: Path, scores: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]`.

- [x] **Step 1: Write failing tests.** Tiny run fixture contains a hashed frozen config, explicit train/test groups, a Raw per-task prediction CSV, a matching score source and paired no-weight predictions. Also write a normalized modern-target artifact, a zero-byte run, a missing train-membership artifact and a second deployment at the same SHA. Assert eligible Raw predictions join exactly by SHA/deployment/task, normalized predictions receive `normalized_target_not_raw`, zero-byte files receive `missing_zero_bytes`, crossing root/duplicate groups receive `historical_leakage`, missing training membership receives `split_unverifiable`, and no-weight/weight comparison requires identical held-out rows and folds. A missing directory returns `not_found` without preventing score assembly/statistics.
- [x] **Step 2: Run** `python -m pytest tests/heritability/test_historical.py -q`; expect absent historical module.
- [x] **Step 3: Implement** discovery and schema adapters, with explicit mappings for `actual/prediction/outer_fold/feature_block`. Start with the source repo and the old cluster copy path supplied by the person or visible workspace layout; never recursive-scan the entire filesystem. Do not load historical .pt/.bin or execute source scripts. Hash relevant tables/configs only. Inspect target semantics before import; if an old normalized target is the only result, preserve a labeled descriptive table and mark the Raw exchange-rate cell unavailable. Neither inverse clipping nor observed-target replacement recovers valid Raw errors.

If predictions cannot be located, ask the person for the `research_runs/` directory and record a blocker **for S4 only**. S2's raw-score inventory and S3 can finish. If only aggregate predictions exist, report the missing per-skill outputs rather than treating the aggregate as six individual skills. September outcomes remain historical/unpinned when original weights lack SHA evidence.
- [x] **Step 4: Run** the task file, then the study package once. Suggested message: `feat(study): audit historical prediction provenance and held-out splits`.

## Task 5: Estimation and selection exchange rates (S4)

**Files:** Create `studies/heritability/exchange.py`, `tests/heritability/test_exchange.py`.

**Interfaces:**
- `exchange_rate(actual: np.ndarray, predicted: np.ndarray, item_variance: np.ndarray) -> dict`.
- `selection_pairs(predictions: pd.DataFrame) -> pd.DataFrame`.
- `exchange_table(predictions: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame`.

For estimation `s²=mean((prediction-actual)²)`, `q=mean(p*(1-p))`, `k=q/s²`. Report MAE, actual RMSE, strict upper bound `q/MAE²` and universal `0.25/MAE²`. An optional historical approximation `0.25/(1.25*MAE)**2` is labeled an approximation, never a strict bound. No observed RMSE is replaced by 1.25×MAE.

For selection use pair residual `(pred_i-pred_j)-(actual_i-actual_j)` on the same task, method and held-out fold. Remove pairs from the same duplicate group; label within-family vs across-family by audited root ancestry, separately from the prediction-training setting. Construct each unordered pair exactly once, sort keys, and bootstrap source split groups rather than pretending pairs are independent. Limit a large pair table with a prespecified seeded uniform sample of at most 100,000 pairs per task/stratum, record sampling counts, and reuse identical pairs for no-weight and weight methods.

If each candidate receives n independent binary items, the difference has variance `(p_i*(1-p_i)+p_j*(1-p_j))/n`. Use that numerator with pairwise residual MSE; k means **items per candidate**, with total items `2*k`. If both candidates answer the same items, subtract `2*cov(correct_i,correct_j)` only when item responses establish it; otherwise label the independent-item approximation. Composite tasks need their documented per-item variance design or receive `item_equivalence_unavailable`. In particular, `0.25/s_delta²` is not a universal pairwise upper bound; independent binary pairs have a numerator bounded by 0.5.

For both estimands report incremental `k_weight-k_no_weight` on matched support, including negative values. Do not claim that subtracting two separate upper bounds bounds this increment. Report raw pairwise error, ranking accuracy and regret as descriptive selection outcomes, not an end-to-end `recommend` claim. Random-item equivalence is the primary first pass; adaptive scenarios divide k once by specified factors 2 and 4, labeled sensitivity, not measured IRT savings. No new tinyBenchmarks/BenchPress evaluation is launched here; these remain matched-cost extensions in the selection paper.

- [x] **Step 1: Write failing tests.**

```python
def test_strict_and_measured_exchange():
    r = exchange_rate(np.array([.5,.5]), np.array([.6,.4]), np.array([.25,.25]))
    assert r['rmse'] == pytest.approx(.1)
    assert r['k'] == pytest.approx(25.)
    assert r['strict_mae_upper'] == pytest.approx(25.)
    assert r['k'] <= r['strict_mae_upper'] + 1e-10

def test_selection_cancels_shared_error(pair_predictions):
    # Actual [.4,.6], predicted [.5,.7], same family, distinct duplicate groups.
    pairs = selection_pairs(pair_predictions)
    assert len(pairs) == 1
    assert pairs.iloc[0].residual_difference == pytest.approx(0.)
    assert pairs.iloc[0].independent_item_variance == pytest.approx(.48)
```

Test pair differences with residual .1 yield k=48 for numerator .48, total=96; no identical-duplicate pair; negative incremental value remains; mismatched baseline rows excluded with reason; heavy-tailed error gives RMSE>MAE and the strict bound still holds; zero MSE emits `perfect_on_sample` and null numeric k (no finite certainty claim); zero MAE bound flagged unbounded; unsupported composite item equivalence unavailable; predictions outside [0,1] retained and counted.
- [x] **Step 2: Run** `python -m pytest tests/heritability/test_exchange.py -q`; expect absent exchange module.
- [x] **Step 3: Implement** the contracts and interval calculation using the frozen grouping; export all strata including missing/unverified coverage cells. Separate historical descriptive results from revision-verified/protocol-eligible results.
- [x] **Step 4: Run** the task file, then the study package once. Suggested message: `feat(study): price prediction errors for estimation and selection`.

## Task 6: Frozen run CLI, tiny dry run and result tables (S2–S4)

**Files:** Create `studies/heritability/cli.py`, `studies/heritability/report.py`, `tests/heritability/test_cli.py`, `docs/operations/heritability.md`.

**Interfaces:** `main(argv: list[str] | None = None) -> int` with subcommands:

- `assemble --config <draft.json> --source-root <Model-Research> --out <run>`: Task 1 sources plus graph, audit and groups.
- `historical --run-root <research_runs> --assembled <run> --out <audit-run>`: Task 4 only.
- `freeze --config <draft.json> --assembled <run> --out <final-run>`: copy canonical hashed protocol/input inventory and selected inputs; refuse mutations.
- `analyze --run <final-run> [--resume]`: verify hashes, execute eligible REML/parent/exchange analyses, write coverage, fit diagnostics, tables and `report.md`.
- `dry-run --out <temporary-path>`: generate tiny synthetic sources/graph/Raw historical predictions, then run the same pipeline with recorded 20-bootstrap override; no real network/files/GPU access.

- [x] **Step 1: Write failing tests.** CLI dry run finishes with deterministic statistical tables and no absolute stored paths. Freeze rejects modified input hashes; resume accepts identical inputs and refuses changed config. Simulate a report-stage failure and assert the previous completed report remains intact. Missing historical data still writes heritability and an exchange table with reason `historical_predictions_missing`. Reports expose convergence failures, observed sample counts, verified join fractions and missing task rows. Test `main` returns nonzero on corrupted freeze and zero for an honest partial report labeled `PARTIAL`.
- [x] **Step 2: Run** `python -m pytest tests/heritability/test_cli.py -q`; expect absent CLI module.
- [x] **Step 3: Implement** orchestration with no duplicated statistical logic. Outputs: `protocol.json`, `inputs.json`, score/edge/group snapshots, `exclusions.csv`, `coverage.json`, `heritability.csv`, `parent_offspring.csv`, `exchange.csv`, `fit_diagnostics.json`, and `report.md`. A task with no eligible observations gets an explicit unavailable row rather than disappearing.

Commands run from the track-c worktree after the reviewed code is implemented:

```bash
python -m studies.heritability.cli dry-run --out "$ZOO_DATA/studies/heritability/dry-run"
python -m studies.heritability.cli assemble --config configs/heritability_v1.json --source-root ../../Model-Research --out "$ZOO_DATA/studies/heritability/assembly-v1"
# Only after S2 identifies and audits the original run directory:
python -m studies.heritability.cli historical --run-root "$SEPTEMBER_RUNS" --assembled "$ZOO_DATA/studies/heritability/assembly-v1" --out "$ZOO_DATA/studies/heritability/historical-v1"
python -m studies.heritability.cli freeze --config configs/heritability_v1.json --assembled "$ZOO_DATA/studies/heritability/assembly-v1" --out "$ZOO_DATA/studies/heritability/workshop-v1"
python -m studies.heritability.cli analyze --run "$ZOO_DATA/studies/heritability/workshop-v1" --resume
```

`SEPTEMBER_RUNS` is a shell argument convenience supplied by the person, not a new mandatory `.env` key. Before final freeze, the config references the audited historical-v1 tables if available; otherwise exchange is explicitly unavailable and heritability proceeds. Do not freeze an absolute `SEPTEMBER_RUNS` path into artifacts.

Expected first-pass input reads ~12 MB of selected sources plus audited prediction tables (measure during inventory), no weights/downloads, 0 GPU-hours; assembled outputs estimated <100 MB, report outputs <20 MB. Time estimates: assembly <5 minutes, tiny run <2 minutes, final 2,000-bootstrap fits initially 0.5–4 CPU-hours depending on support. Time 20 replicates first and extrapolate. A projected >2-hour final command is **[HUMAN] launches**, queued with measured byte count, hashes, wall time estimate, storage, 0 GPU-hours, exact command above, verification (`report.md` and coverage/fit status plus matching hashes), and the same `--resume` command. Do not silently lower bootstrap count to fit the agent limit.

- [x] **Step 4: Run** task file, then `python -m pytest tests/heritability -q` once. At the batch handoff run `python -m pytest tests -q` once; report counts/skips. Suggested message: `feat(study): add frozen analysis runner and audited result reports`.

## Task 7: Workshop draft and downstream handoffs (S7)

**Files:** Create `docs/papers/heritability-workshop.md`, update `docs/operations/heritability.md`. No statistical implementation or new tests in this documentation task.

**Consumes:** Frozen protocol hash, audited tables and coverage from Task 6. **Produces:** A workshop draft with every quantitative claim linked to a table row and exact frozen run; a compact exchange-rate table for later selection work.

- [x] **Step 1:** Write the methods/results/limitations draft, using named prior work from merged-plan §3.3 and §9 idea 2 (PhyloLM, Tamura et al., Laufer et al.; network Brownian model and prior effective sample-size antecedents). Present familial association, not causal inheritance or novelty of the formula. Explain unknown revisions, declared Atlas labels, composite variance gaps, unit branch lengths and independent-item assumptions.
- [x] **Step 2:** Include per-skill/per-supported-edge-type raw and logit fits, parent slopes, scale-free errors, measurement noise, confidence intervals, revision-verified fractions and exclusions, historical exchange estimates vs strict bounds, within/across-family selection results, and incremental no-weight comparison. If a requested result is unavailable, say why in its table cell; do not replace it with a normalized result or an unsupported type classification.
- [x] **Step 3:** Record the ICLR 2027 workshop as the intended target; a person verifies venue/deadline later under F5 and submits. No website lookup/publication is needed to finish this draft. Documentation check only: every numeric claim traces to existing frozen output; no additional full test suite for prose.
- [x] **Step 4:** Handoff schema proposal for S5/E1: `rank_candidates(query_key: str, candidate_keys: list[str], artifacts: dict) -> pd.DataFrame` with `query_key,candidate_key,method,score,bytes_read,status,reason,method_version`, score descending and candidate key tie-breaks. Artifact refs use frozen model keys and data-relative paths. Missing T2 spectra/fingerprint coverage causes abstention, never an empty success. Trace and Spectral Signatures port from `Model-Research/spectral_research/features.py` and `spectral_research/study/lineage.py`; port only the actual baseline math, with its provenance, rather than the old study runner. Agree this non-frozen adapter with the reviewed engine plan before implementing S5; convert to E1's approved interface if different. Existing `LINEAGE_COLUMNS` and lab scoring inputs remain unchanged.

Suggested message: `docs(study): draft workshop findings and baseline handoffs`.

## After this plan

1. The person reviews S1, then implementation starts in track-c. The lead updates only the main checkout's `handoff/CHECKLIST.md` and `handoff/STATUS.md`; all code stays uncommitted in the track-c worktree. New plan remains in the main checkout under D10.4.
2. S2 can finish the available raw-score/graph audit while missing v1 inputs or September outputs remain separately queued. S3 does not wait for predictions; measurement-corrected task fits require defensible variance metadata, with descriptive work retained elsewhere.
3. S4 uses eligible existing predictions only. No refitting, new benchmark evaluation or repair of old leaky splits is authorized by this plan. If only normalized/aggregate predictions survive, report that limitation and ask the person whether to scope a later Raw predictor rerun as new work.
4. T1 distance and typed-edge extensions, the full matched-cost selection study and tinyBenchmarks/BenchPress/scaling-law comparisons wait for later artifacts and the selection-paper plan. No phylogenetic GP, daily ingestion, hosted service, paid judging or full-model training is added.
5. S5 ports/evaluates baselines once the extractor and engine contracts are reviewed and dependencies committed into main. S6–S11 follow the checklist; do not access/share the blind subset through this study pipeline.

## Input evidence and approved modeling choices

- A validated v1 Raw source, benchmark item/variance evidence and an independent pinned deployment index remain unavailable in this frozen audit. The code supports explicit future mappings; unsupported tasks remain reported gaps.
- September `research_runs/` outputs were located and audited read-only. The frozen first-pass audit contains original artifacts and records 290,496 historical rows without claiming eligible Raw semantics or verified joins. Modern normalized targets cannot be relabeled Raw; legacy semantics remain distinct and unvalidated.
- The owner approved the stated first-pass modeling calls and routine implementation autonomy: raw primary/logit sensitivity; unit network increments; unknown merge weights excluded from the primary network fit with equal-weight sensitivity; source-root cluster bootstrap; independent-item selection equivalence when paired responses are absent. These are draft choices for this implementation, not changes to a frozen project interface.
