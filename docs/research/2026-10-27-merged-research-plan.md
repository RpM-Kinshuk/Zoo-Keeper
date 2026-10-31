# Model Zoo: a measured, queryable database of open models (merged research plan)

**Status:** draft v3 for team review, 2026-10-31.
- v2 (2026-10-29) was revised after a review from five independent perspectives
  (section 0).
- v3 re-centers the plan on the zoo and its query framework (section 0.1).
- v3 also adds section 6.8 (2026-10-31): System One decision models and other
  trending model types, checked against the extractor's rules.
- Moved on 2026-11-01 from Model-Research (now frozen) to Zoo-Keeper
  `docs/research/`, next to Plans A and B. Model-Zoo collects and builds the
  database; Zoo-Keeper holds the queries, the research, and the handoff.

It merges two plans:

- **Plan A:** the census and query plan
  (`2026-10-23-census-and-query-plan.md`);
- **Plan B:** the frontier research plan
  (`model_zoo_frontier_research_plan_2026-10-23.md`).

Both plans were treated as hypotheses. They were checked against the current
`Model-Research` and `Model-Zoo` checkouts (both updated 2026-10-24) and against
the literature on 2026-10-24.

Markers used throughout:

- *(verified)*: checked against code, data, or a primary source on the date
  given.
- *(est.)*: a back-of-envelope number to calibrate in the pilot.
- *(call)*: a decision made in this draft that the team should check.
- *(hypothesis)*: a claim that the plan tests, not a finding.

Team answers (2026-10-27):

- **People:** plan for 2–3.
- **Compute:** GPU and CPU hours are available, CPU more easily.
- **Bandwidth:** download bandwidth is available.
- **Public service:** not scoped. The deliverable is a framework and a way to do
  this; whether it becomes a service depends on how effective it proves.
- **Authorization:** large runs need no sign-off. The operations-guide note only
  stops agents from launching them automatically, so people launch them.
- **Aim:** push the research frontier and earn the attention of top labs,
  researchers, and companies with something genuinely good and useful. This is
  not corporate tooling.

Later team answers (2026-10-28 and 2026-10-29):

- gated access through the owner's personal HF account;
- at most about $20 a day of Claude API calls for evaluation;
- support every model type with one solution;
- larger sketches if they help;
- simple yet useful over the most advanced.

Section 13 is written per unit and per stratum, to be sized after the pilot.

## 0. Review of 2026-10-29: what five perspectives found, and what changed

Five independent reviewers read v1 of this plan and the extractor's
implementation plan. Each was given the documents and the team's goals, not the
author's conclusions. The perspectives were:

- **two from the goals:** a top-venue area chair, and a frontier-lab or Hub
  practitioner;
- **two from the directives:** a feasibility lead for 2–3 people, and the
  "every model type, simple" audit, which read live safetensors headers from
  18 Hub repositories;
- **one technical:** a statistics and correctness reviewer, who ran
  simulations.

### Findings that changed the plan

| Perspective | Finding | Change |
| --- | --- | --- |
| Area chair | "A few percent of the bytes" is not novel: modelDNA reads 100–300 MB, about 1–2% of a 7B model. P1 claimed 10⁴+ models but planned 2–5k. Calibration with abstention and coverage sets are already published. | P1's headline becomes **"pairwise lineage accuracy does not scale"**: false parents per query against candidate-pool size, from 10² up to the reached pool, for every baseline, plus an accuracy-versus-bytes curve with modelDNA at matched bytes (section 11) |
| Area chair | Missing baselines: black-box provenance testing (2502.00706), MoTHer (ICLR 2025), metadata imputation (2606.21787). Section 5.2 overclaimed that behavior cannot reveal descent. | Added to P1's core evidence; section 5.2 rewritten |
| Area chair, correctness | Hard negatives defined as "a different declared root" are the easy case. Lab-only calibration does not transfer to the Hub. Stouffer combination over dependent tensors is anti-conservative. | Three nested estimands (same root, ancestor, immediate parent); a verified-Hub truth tier; a blind laundering set; cross-family calibration; empirical calibration of the combined statistic with a generalized Pareto tail and per-query FDR (sections 9 idea 11 and 10) |
| Correctness | The bf16 noise floor is 10–30× optimistic. Raw Gavish–Donoho over-counts rank below ρ ≈ 3e-3 and on heteroscedastic weights. Stored-sketch rank capacity is about k/4, not 0.75k. Unscaled Gaussian operators are ill-conditioned. The FP8 block size cannot be inferred from shapes. | Abstain below the floor; biwhitened rank with a gap-based confidence; orthonormal, side-specific nested operators; block size read from config; capacity claims corrected (section 6.3) |
| Correctness | The tree-recovery "guarantee" (Atteson plus JL) is additive against multiplicative error, vacuous for star families, and the wrong tool when ancestors are observed. | Demoted to a remark. The finding becomes **"how tree-like is model evolution?"** Minimum-cost arborescence is used when ancestors are observed (idea 10) |
| Correctness | Merge recovery under sum-to-one is misspecified: SLERP brings in a base term, mergekit often varies t by layer, and TIES/DARE recover effective weights. | Unconstrained regression on [parents, base] per tensor or layer group, with a block bootstrap (idea 8) |
| Correctness, area chair | Pagel's λ and Blomberg's K do not fit observed ancestors, polytomies, and merges. The folk-wisdom finding has prior work (PhyloLM, Tamura et al., Laufer et al.). | REML mixed model and parent–offspring regression with binomial measurement error, on raw or logit scales; novelty narrowed (idea 2) |
| Universality | The view table mis-measures real Hub storage in several ways: Whisper and linear-attention conv1d tensors are sliced into up to about 246k junk views; NVFP4 and GPTQ scale tensors are measured as weights; SDXL, Mistral, and gpt-oss store weights 2–5 times; kohya LoRA is never paired; expert orientation varies within one family. | Decode × geometry table, a never-a-matrix suffix list, module poisoning, file-set selection, all LoRA families with α scaling, an expert subset, and orientation from config (section 6.6) |
| Universality | Fine-grained MoE breaks the cost model: T1 reads 20–25% of bytes, and fingerprints reach 0.3–0.7 GB. | 8 seeded experts per layer at T1, `--all_experts` for the lab and anchors (section 5.3) |
| Practitioner | No usable tool until February. Private checkpoints are not Hub safetensors. HDF5 is unfriendly to share. "Predicted skill changes" in a diff would destroy trust. | Internal `fingerprint`/`diff`/`check` CLI at G0 with JSON and exit codes; `fingerprint_tensors()` for private formats; fingerprints as `.safetensors`; diff reports measured facts only (sections 6.2 and 9 idea 19) |
| Practitioner | Real Hub merges ship `mergekit_config.yml`, which is free ground truth. The genealogy lab is the most citable artifact if released. | Verified-Hub truth tier; the lab is released as a benchmark with a scorer (sections 10 and 11) |
| Feasibility | M0 is about 13 person-weeks against 7–10 available by 11-21. The KDD February cycle accepts only resubmissions. The lab has no size target. GRPO and full 7B training need GPU reservations the site lacks. The CPU T2 estimate is untested (likely 10–30 node-minutes per 7B, not 1–5). P3's scope by May exceeds bytes and GPUs. | G0 moves to **2026-12-04**, with measurable thresholds; the fallback is the benchmark framing at ICML, or COLM 2027; a minimum lab of about 120 models with GRPO cut; T2 timed in week 1; P3 rescoped; a week-by-week plan (section 12) |
| Area chair, practitioner, feasibility | P6, checkpoint triage on pretraining trajectories, is the wrong phase and too late: validation loss already predicts static evaluations there. | **P6 is cut as a paper.** Checkpoint diffs become a tool feature (a training callback in zoo v1). P2 and the exchange rate become results inside the selection paper (section 11, revised 2026-10-31) |

### Conflicts between perspectives, and how they were resolved

- **Release timing.** The practitioner wanted a public tool v0.1 at G0, GGUF in
  M1, and a watch script in M1. Feasibility wanted all of it deferred.
  - **Resolved:** the CLI is built inside the extractor (Task 8) and used
    internally from G0.
  - The public release, GGUF dequantization, the watch script, and the training
    callback ship as zoo v1 after the ICML submission (February–March).
- **Candidate-pool scale.** The area chair wanted 50–100k candidates in P1.
  Feasibility capped the Hub set at 2–5k before the deadline.
  - **Resolved:** the candidate pool is whatever T1 completes in the
    December 7–18 window (target 5–20k, minimum 5k).
  - False matches per query are reported at each pool size reached and
    extrapolated with the calibrated null. 2–5k labeled queries are kept.
- **The zero-byte tensor-sharing preprint.** The area chair proposed it for
  December–January. It does not fit 2–3 people before ICML.
  - **Resolved:** it moves to February–March as the first, cheap component of
    P3.
- **The MoE fingerprint size.** The practitioner proposed k = 16 for experts.
  - **Resolved:** the expert subset is used instead (universality review),
    because it also fixes the byte cost.

### Verified as sound

These claims survived checking:

- the exchange-rate arithmetic;
- the abliteration row-block argument;
- the row block commuting with linear, SLERP, DARE, and TIES merges;
- nested operators (bitwise exact);
- the sketch null standard deviation of about 1.12/k;
- the Gavish–Donoho polynomial;
- fp32 sketch storage;
- that CDN range reads are not rate-limited.

### 0.1 Re-centered on 2026-10-31: the zoo and its query framework are the core

The team confirmed the original aim of both source plans: **a database (zoo) of
models plus a framework that helps people query it.** After the review, v2 had
drifted toward lineage as the main product, with model selection parked. v3
puts the zoo first.

- Lineage, `diff`, and `check` become **query types** the zoo answers.
- The papers follow the zoo:
  1. lineage, which organizes it;
  2. the zoo release itself;
  3. selection, the query framework's hardest question.
- A minimal queryable zoo exists from the first gate (section 6.7).
- Two clarifications were also agreed:
  - "exposing wrong metadata" is a side effect, not a goal;
  - `diff` matters mainly between *different* models, not between revisions of
    one repository, because released weights rarely change (section 9,
    idea 19).

## 1. In one paragraph

**We build the Model Zoo: a measured, queryable database of open models, with a
framework anyone can use to ask it questions.** Every model in it carries:

- pinned identity and metadata;
- a fingerprint, a small summary of its weights read from a few percent of the
  file;
- its measured lineage;
- measured or predicted scores, with uncertainty;
- explicit coverage of what was and was not measured.

The framework answers practical questions:

| Question | Query | How it is answered |
| --- | --- | --- |
| Which model should I use for my task under my constraints? | `recommend` | Filter by constraints, rank by measured scores where they exist and predicted scores with intervals where they do not, then optionally run a quick test on the user's own examples |
| What models are like this one, or a cheaper replacement? | `similar` | Nearest fingerprints, collapsing near-duplicates |
| What is this model, really? Its family, base, edge type, license chain | `info`, `family` | Lineage from weights, with evidence tiers |
| What changed between these two models? | `diff` | Measured, per-layer facts |
| Is this model's declared base right? | `check` | Fingerprint comparison against the declared base |
| Give me models matching these filters | `find` | Metadata and measured facts in DuckDB |

Two principles guide the zoo:

- **Weights answer what only weights can show:** lineage, edge type, merge
  recipes, duplicates.
- **Measurement answers what testing shows more cheaply:** capability where
  evaluation is cheap. Weights serve as a priced prior only where evaluation is
  expensive (section 5.1).

Everything is open and works for every model type: the zoo dataset, the query
library and CLI, the fingerprint tool, and the genealogy lab as a benchmark. A
hosted service follows only if the framework proves effective.

## 1.1 Focus: what we commit to, and what waits *(call, revised 2026-10-29)*

The rest of this document is a portfolio. This section is the plan a team of
2–3 people actually executes. A new idea enters only by displacing one below.

**Committed, in order** (revised 2026-10-31 to put the zoo first):

1. **Extractor, zoo store, and lab, in parallel from week 1** (section 12).
   - The extractor writes fingerprints.
   - The zoo store (section 6.7) loads manifests, fingerprints, scores, and
     lineage into Parquet and DuckDB.
   - The P1 engine starts in week 2 on lab models in local mode.
2. **Zoo v0 at G0 (2026-12-04), internal.** The pilot and lab models are
   queryable through `find`, `info`, `similar`, `family`, `diff`, and `check`,
   from a Python library and a CLI.
3. **P1, lineage, the flagship (ICML 2027):** "pairwise lineage accuracy does
   not scale; typed, calibrated lineage does". This is what organizes the zoo
   into families, duplicates, and license chains. It uses:
   - exact lab truth;
   - a verified-Hub truth tier;
   - a T1 candidate pool of 5–20k;
   - a prospective set frozen at G0.
4. **One cheap study on existing data, by person C from week 1:** capability
   heritability with the corrected method, plus a first exchange-rate pass
   (ideas 2 and 3). It targets an ICLR 2027 workshop paper. Both feed the
   zoo's score predictions and the selection paper.
5. **Zoo v1, public (February–March, after submission):** the query library
   and CLI, GGUF lineage, a demo Space, and the zero-byte tensor-sharing note.
6. **Zoo v2 and P3, the zoo release (NeurIPS 2027 E&D):**
   - T0 Hub-wide;
   - T1 on 50–100k models, sharded across hosts;
   - T2 on a 2–5k probability sample;
   - T3-lite on at most 2k models;
   - two-phase error correction.

   Idea 5's reliability study ships with it.
7. **The selection paper (ICLR 2028):** `recommend`, evaluated end to end.
   - It reports regret and total cost against strong non-weight baselines.
   - The exchange rate and heritability are results inside it.
   - Gate G2 decides how much weight evidence the recommender uses, not
     whether the recommender exists.

**Cut:** P6 as a paper (checkpoint diffs survive as a tool feature), GRPO in the
lab, and full-parameter 7B training.

**Parked until the committed work lands:**
- the phylogenetic Gaussian-process prior (idea 12);
- heterogeneous evidence acquisition beyond "quick test on your examples"
  (idea 15);
- deployment identity beyond recording format and quantization (idea 16);
- the MCP wrapper (thin, after zoo v1);
- GPTQ/AWQ/bitsandbytes dequantization;
- daily ingestion;
- the VLDB systems paper.

**Pivot rules, each with a measurable trigger and a date:**

| Trigger (decided on) | Then |
| --- | --- |
| T1 lineage AUROC is more than 0.01 below T2's, or decision agreement < 0.9 on the lab (G0, 12-04) | Read full tensors for a depth subset (Plan A's tiered option) instead of row blocks |
| On independent pairs, the null's \|mean\| exceeds 1 null sd, or its spread falls outside 0.7–1.5× the theoretical 1.12/k (G0, 12-04) | Use exact permutation tests on shortlists only, and drop closed-form error claims |
| Four-point violations exceed 20%, or normalized Robinson–Foulds distance exceeds 0.2 on lab trees (11-27) | Report "how tree-like" as the finding and use supervised edge scoring; no tree-recovery claim |
| P1 has neither ≥2× fewer false matches per query at TPR 0.9 nor ≥10× fewer bytes at equal AUROC than the best baseline (12-18) | Reframe P1 as a benchmark paper at ICML: the lab, the protocol, and the measured failure modes of every method. Never shrink the lab to save time; shrink the Hub set |
| The exchange rate is small | That becomes a result of the selection paper, and `recommend` leans on measured scores, not a reason to push capability prediction harder |

**How the team works:**
- **Roles** (A, B, C):
  - A owns infrastructure: the extractor, the pilot, and the Hub runs.
  - B owns the lab and the lineage engine.
  - C owns the existing-data study, the baselines, and the label audit.
  - With two people, C's work splits between A (frame, negatives, label audit)
    and B (baselines limited to Trace, modelDNA, and Witness Overlap), and the
    workshop paper is deferred.
- **Interfaces first.** The fingerprint format freezes at the end of week 1,
  so B builds the engine on synthetic fingerprints while A finishes the reader.
- **Accounts.** One token, the owner's `HF_TOKEN` (revised 2026-11-01: one
  person works with agent sessions).
- **Weekly.** A one-page result log covering what was measured, what changed in
  the plan, and what was killed.
- **Reproducibility.** Every reported number traces to a frozen config and
  pinned inputs, as the current study already does.
- **Claims.** Novelty claims are limited to the openings in section 3.3.

## 2. What changed since Plans A and B *(verified 2026-10-27)*

The `Model-Zoo` scale branch was merged on 2026-10-24. Several Plan A
prerequisites now exist for **new** collections. None of them repairs the
**historical** 12,553 analyses behind the HTSR study.

| Plan A prerequisite | Current collector | Historical 12,553 analyses |
| --- | --- | --- |
| Revision identity | Done. Preparation requires 40-character model and base commit SHAs, and records config labels at the pinned revision. | Not recorded. Score joins stay unverifiable. |
| Full spectra | Done. `--save_eigs` defaults to on; HDF5 `/eigs` holds full computed spectra aligned to canonical names. | Spectra exist for 17 models only. |
| Small-eigenvalue accuracy | Done. SVD (CUDA `gesvd`) is the default in fp32; the Gram path is opt-in; fp64 is available for reference checks. | fp32 Gram path. Values near the 1e-5 cutoff are at the noise floor (example below). |
| Matrix coverage | Done. Every supported matrix is kept by default (selection 4). The aspect-ratio ≥ 8 skip is opt-in. | Aspect-ratio skip was active. Key and value projections are missing wherever hidden size ÷ key-value width ≥ 8 (example below). |
| Architecture labels | Documented: 8,467 truncated cells; the exporter is absent from the checkout. New pins record config labels separately. | Truncated labels unusable for stratification. |
| Credential in launch script | Local `run_script.sh` requires `HF_TOKEN` from the environment; no token appears in its git history. | Plan B reports a hardcoded token in the cluster copy; not checked here. |
| Stream, do not load | **Not done.** Checkpoint mode reads downloaded safetensors with `safe_open`; there are no HTTP range reads. | Not applicable. |
| Per-tensor content hashes | **Not done.** Only config and adapter hashes are computed. | Not applicable. |
| Linear sketches | **Not done.** | Not applicable. |
| Packed quantization (GPTQ, AWQ, GGUF, NF4, FP8) | **Not measured** (explicit gap; packing is never inferred). | Not applicable. |
| Scale | **Unproven.** The catalog has 52 rows: 46 successes across 44 repositories, 4 strict-load failures (BERT pooler keys), and 2 held. All are at most 3B parameters; median download is 0.44 GB, maximum 1.26 GB. The defaults are a 6 GiB weight budget and one worker. The operations guide's "50k run not authorized" note stops agents from launching one automatically; people launch large runs (team, 2026-10-27). | Not applicable. |

**Consequence.** The collector now does the measurement correctly; it has not yet
shown it can do so at scale. The binding gaps are:

- bytes: streaming and range reads;
- models above about 3B parameters, which the 6 GiB budget excludes in bf16;
- packed quantization;
- tensor hashes and sketches, the coordinate information that lineage needs.

These gaps define Milestone 0.

## 3. Verification ledger

### 3.1 Claims about our own code and data

| Claim (source) | Status | Evidence |
| --- | --- | --- |
| 8,487 one-character `Architecture` cells (A) | **Corrected** | 8,487 one-character cells, of which 8,467 are truncations (`u` 3,848, `L` 1,977, `M` 1,380, `Q` 1,018, …) and 20 are `?`. `docs/operations/curation.md` reports 8,467. |
| Only 17 of 12,553 models have eigenvalues (A) | **Verified**, with cause | The old collector had `--save_eigs` off by default. Fixed for new runs. |
| fp32 Gram is wrong at the bottom of the spectrum (A) | **Verified** in historical data | `svd_results/01-ai--Yi-1.5-9B-Chat.csv`, layer-0 `q_proj`: 3,565 eigenvalues retained, `matrix_rank` 3,075, `xmin` = 1.0005e-5 (the cutoff itself). Features using small eigenvalues, such as entropy, rank, and log-spectrum distances, are unreliable historically. Hill on the top half of the spectrum is unaffected. |
| The Qwen2.5-3B pilot omitted 72 tensors (B) | **Verified, cause identified** | They are `k_proj` and `v_proj` in all 36 layers: hidden size 2048 ÷ key-value width 256 = 8. The local `svd_results` copy has 57 models missing exactly key and value (hidden sizes 2048, 4096, 8192). Newer architectures increasingly have such ratios, so the historical "complete seven-role" set is biased against them. |
| Local `svd_results` is complete | **Not true** | 1,606 of 3,817 files are zero bytes, including 91% of the files for models above 40B. This may be a partial copy of the cluster data. Reconcile before reuse, and count empty files as failures. |
| 46 successful records across 44 repositories (B) | **Verified** | `Model-Zoo/analysis_runs/phase2/catalog.csv`. |
| HTSR numbers quoted by both plans | **Verified** | ARI 0.924; base match 82.30% vs 85.19% (top three 99.29%); root-availability AUROC 0.676 (score) and 0.442 (margin); MAE values as quoted. |

### 3.2 Claims about the literature (checked 2026-10-24)

Almost every cited ID exists and is characterized correctly. Corrections and
additions:

| Item | Correction or addition |
| --- | --- |
| 2608.07786 | One paper. Title: "Who Built This Model? Tracing LLM Lineage via Spectral Fingerprints in Weight Space" (Chen, Shang, Liu; COLM 2026). "LLM Biometrics" (B) is a phrase from its abstract. Its "Trace" is per-layer spectral energy tr(WᵀW) correlated across depth, not layer norm (A). Our study implements it correctly. |
| HF "State of Open Models: Summer 2026" | "1.5% of repositories account for 99.2% of downloads" is verbatim. "Coding agents are the Hub's largest client" (A) is **not** in the post. Claude Code led *agent-tagged* traffic, with 44.4%. |
| DHS DeepSeek order (A) | Signed in July 2026 and published in the Federal Register on 2026-10-20 (91 FR 66346). Whether it covers fine-tuned derivatives is a matter of interpretation. |
| MatrixFormer 2610.06751 | A general matrix-completion model. LLM score completion is one application among several. |
| 2610.13123 "Why Heavy-Tailed Weight Spectra Predict Model Quality" (NeurIPS 2026) | Final-layer Bayes free energy (BFE-W). LLM evidence is two families with six sizes each. It cannot be compared across activation dimensions. A within-series baseline (section 9, idea 18), not a Hub-wide one. |
| FARMS 2506.06280 | Co-authored by this team. It is the principled replacement for the aspect-ratio skip, and its submatrix sampling fits the row-block reads of section 6.3. |
| Spectral Signatures 2607.03377 (KDD 2026) | From the same research group. Our foundation, not a competitor. |
| **Missing from both plans** | 2602.00772 "Provable Model Provenance Set for LLMs": provenance sets with provable coverage; the closest prior work to calibrated provenance. 2610.30626: weight-only spectral fingerprinting, posted 2026-10-22. 2606.21787: metadata-based lineage imputation over 317,133 Hub models. 2608.08139 TokenPrint: calibrated, but needs forward passes. 2606.24020 BenchPress: predicts scores from other benchmarks, a strong non-weight baseline. W2T 2603.15990: LoRA performance from adapter weights. |
| modelDNA 2607.10617 | Closer than Plan A implies: a public dataset, a live HF scanner, merge decomposition by sum-to-one least squares, and calibrated verdicts with abstention. Small evaluation: 15 models. |
| Witness Overlap 2609.31784 (NeurIPS 2026) | 95.3% correct parent-to-child direction over 176 checkpoints. Our direction claims must beat or combine with it. |

### 3.3 Novelty after the check

- **Still open:**
  - A public weight-derived dataset for 10k+ real LLM checkpoints. The largest
    published one is 499 models; our unpublished historical corpus is 12.5k.
  - Lineage *and edge typing* evaluated at 10^4+ models with exact lab truth and
    query-level error control.
  - Design-based (inverse-probability-weighted) ecosystem prevalence estimates.
  - Capability heritability on the model tree.
  - Cost-aware model selection that prices weight evidence against evaluation.
- **Already taken (do not claim):**
  - calibration with abstention on a few models (modelDNA);
  - provenance sets with coverage (2602.00772);
  - catalog-scale retrieval (ModelLakeFishing, 3M models);
  - recommendation from evaluation records (ModelLens);
  - weight-only performance prediction as such (Spectral Signatures kNN, W2T,
    BFE-W).

  Plan A's claim 3, "nobody predicts capability for never-evaluated models", must
  narrow to *calibrated, per-skill, cross-family* prediction.
- **Also taken or partly taken** (added 2026-10-29):
  - reading a few percent of the bytes (modelDNA reads 100–300 MB, about 1–2%
    of a 7B model);
  - tree and direction recovery from weights (MoTHer, ICLR 2025; Witness
    Overlap);
  - behavioral provenance tests (2502.00706, TokenPrint);
  - lineage-aware score prediction (PhyloLM, ICLR 2025; Tamura et al.,
    COLM 2025, 2,934 models);
  - trait heritability on Hub family trees (Laufer et al.);
  - the exchange-rate formula itself, which is the prior effective sample size
    (Morita, Thall, and Müller, 2008).
- **What remains ours to claim** is the combination at scale:
  - typed lineage (edge type, merge recipe, adapter rank) with per-query error
    control, measured against candidate-pool size from 10² to 10⁴+;
  - exact and verified truth, including mergekit configs on real Hub merges;
  - one pipeline for every model type;
  - error-corrected, design-based prevalence;
  - heritability per skill and per edge type, corrected for measurement error;
  - the exchange rate for selection (pairwise differences), not just
    estimation.

## 4. Reconciliation: where the plans agree, conflict, and how this plan resolves it

**Agreed by both and kept:**

- pin revisions;
- save full spectra;
- alpha must earn its place, with non-alpha spectral trees as the simple
  baseline;
- per-skill targets instead of retired leaderboard averages;
- lineage needs query-level calibration and controlled ground truth;
- spend evaluation where it changes a decision;
- the LoRA shortcut: singular values of R_B R_Aᵀ;
- freeze splits; hold out roots; run prospective tests;
- a local SDK for private checkpoints.

| Topic | Plan A | Plan B | Merged resolution |
| --- | --- | --- | --- |
| Core objective | Census, then lineage, capability, risk, and query engines | Model selection under incomplete evidence and an evaluation budget | Both, ordered by where weights are uniquely informative: lineage and modification first, selection second, and the census as the shared substrate (section 5). |
| Scale and pacing | 25–50k models by January, 100k by April | 1k → 5–10k → 50k → 100k, each step gated by demonstrated value | Separate cheap tiers from expensive ones. T0 goes Hub-wide at once and T1 (row blocks) reaches about 100k; T2 (full tensors) and T3 (behavior) grow behind value gates (section 12). |
| Metadata | Mostly weights, with metadata for constraints | Keep the weights-only science track and add a hybrid | Two tracks bridged by B's evidence ladder, **with costs attached** (section 10). |
| Headline metric | "A weight prior is worth k items" | Selection regret and total decision cost | Both. k is the exchange rate (section 5.1); regret against total cost is the decision metric. |
| Unit of recommendation | Checkpoint | Deployment identity: quantization, tokenizer, template, runtime | Deployment identity (idea 16). |
| Sampling | Strata with oversampling | Mixture plus a probability sample with recorded inclusion probabilities | Plan A's strata **plus** recorded inclusion probabilities, so prevalence claims are valid (idea 14). |
| Coordinate information | Two-sided 32×32 sketches, CountSketch, SimHash | "Directional sketches", unspecified | One two-sided Gaussian sketch, resized and stored in fp32 (idea 7). SimHash comes from its signs; CountSketch is dropped as redundant. |
| Index | FAISS | Exact search first (a 100k × 512 float32 matrix is about 205 MB) | Exact search until a measurement shows it is too slow. |
| Capability model | Two-sided amortized IRT with a lineage prior | Bilinear ŷ(m,q) = a(q) + u(q)ᵀz_global + v(q)ᵀz_update | The same family. B's linear form is the first baseline; A's hierarchical IRT is the extension, with a phylogenetic kernel (idea 12). |
| Timeline | Calendar to May 2027 | Calendar should follow measured throughput | Venue dates are fixed; the scope of each paper flexes at stated gates (section 12). |
| First experiment | 500-model pilot | Do spectral and update descriptors improve task shortlists over metadata and historical scores? | Both, plus three studies that run on data we already have (ideas 2–4). |
| Novelty claims | "First census", "nobody has…" | "Promising hypothesis, not demonstrated novelty" | B's caution, using the specific openings in section 3.3. |

## 5. Judgment: four findings that reshape both plans

### 5.1 The exchange rate: when weight evidence beats evaluation

Let a prior predict a model's accuracy with root-mean-square error s on a 0–1
scale. Each benchmark item answered at accuracy p carries Fisher information
1/(p(1−p)). The prior is therefore worth about

  **k ≈ p(1 − p) / s²** randomly sampled items,

and at most 0.25/s². Adaptive item selection (IRT) makes each item roughly 2–4×
more informative, which divides k by the same factor.

Applying this to the September results, taking RMSE ≈ 1.25 × MAE and p = 0.5 as
the upper bound:

| Setting (modern / legacy six-task average) | MAE | Prior worth at most |
| --- | ---: | ---: |
| Near-match groups kept together, non-alpha trees | 3.21 / 3.07 | about 155 / 170 items |
| Whole weight cluster held out, non-alpha ridge | 6.56 / 6.87 | about 37 / 34 items |
| Whole cluster held out, training median (no weights) | 9.09 / 13.51 | about 19 / 9 items |

Reading the table:

- **"At most" is not strict.** RMSE = 1.25 × MAE holds for Gaussian errors, and
  heavier tails push RMSE higher. The strict bound uses RMSE ≥ MAE, giving about
  243/265, 58/53, and 30/14 items. The order of magnitude stands: about 10²
  items in-family, a few dozen across families.
- **The value of weights is incremental over the no-weight prior.** Across
  families that is about 37 − 19 ≈ 18 items (modern) and 34 − 9 ≈ 25 items
  (legacy).
- **Per skill it is smaller:** IFEval's MAE of 11.4 is worth at most about 12
  items.
- **Normalization.** Undoing the leaderboard's score normalization changes
  these numbers for the v2 tasks only; binary BBH and MuSR subtasks differ by up
  to 4×.
- **Adaptive item selection.** IRT's 2–4× gain comes partly from population
  item data, which is itself a prior; do not count it twice.
- **Selection needs a different quantity.** Choosing between siblings depends
  on the error in pairwise *differences*. Within a family that error can be
  much smaller than the absolute RMSE, because shared base effects cancel. The
  exchange rate is therefore reported separately for estimation and for
  selection (within-family against cross-family).
- **Cost comparison.** Once a model is loaded, 10² short items cost seconds to
  a couple of GPU-minutes. Reading the weights costs the same download plus
  1–30 compute-minutes. The T2 CPU estimate is untested and is timed in week 1.

**Implications:**

1. **Static benchmarks.** Whenever a census model is already downloaded, measuring
   its static skills directly is cheaper than predicting them. The census
   therefore routes its T3 models through a "download once" GPU path (section
   6.2). Those models get T2 and a small adaptive item bank from a single
   download (T3-lite, section 6) *(call)*. Models on the CPU streaming path are
   never fully resident on a GPU node, so they get no T3-lite unless they are
   downloaded a second time.
2. **Where weights pay as a prior:**
   - evaluation is expensive per item: agentic tasks, long-context tasks, judge-
     or human-scored tasks, and safety red-teaming (about $1–10 per item);
   - target-task labels are scarce: a user's 5–50 examples;
   - the weights are already local: a lab's own training checkpoints, where bytes
     cost nothing;
   - bytes are avoided: T0 and T1 screening reads 0–10% of a model, while
     evaluation needs all of it.
3. **Publish the exchange rate itself** per skill, per setting, and per
   evaluation type, for estimation and for selection. It ships as one table in
   the docs, not as a decision engine, and it is the core of the capability
   paper (section 11).

### 5.2 What weights uniquely reveal

**Behavior can show descent.** Black-box provenance tests (2502.00706,
TokenPrint) infer it from outputs, given inference access.

**What behavior cannot cheaply show:**
- merge proportions;
- the edge type (adapter rank, full fine-tune, quantization, layer
  duplication);
- which tensors are frozen copies;
- per-query error control across a large candidate pool without running every
  candidate.

Weights show these, often from a small fraction of the bytes and without
running anything. The science is cleanest here, and the demand broadest:
- open-model maintainers tracking derivatives;
- labs auditing merges and checkpoints;
- safety researchers finding refusal-ablated models;
- practitioners asking "what is this model, really?".

Competitors are still small-scale. **Lineage and modification come first**
(P1), and P1 compares against a black-box baseline on a 200-pair subset.

### 5.3 Bytes, not FLOPs, are the cost

Plan A estimates 1.73 PB for 100k models under the full option. With CPU
abundant, compute is cheap: fp32 SVD, sketches, and hashes all run on CPU nodes.
Bytes are the cost. Three levers:

1. **Zero-byte tier (T0).** API metadata, safetensors headers, sha256, and Xet
   chunk-hash lists together with header offsets map *which tensors* are shared
   between repositories without downloading any weights. That covers re-uploads,
   frozen embeddings, and passthrough layers. *(hypothesis: Xet reconstruction
   metadata is available for enough public repositories; check in week 1.)*
2. **Row-block tier (T1).** Read one contiguous block of rows per tensor (one
   HTTP range request), about 2–10% of the bytes. **Every linear functional of W
   preserves the sketch algebra**, so merge recipes, adapter rank (rank of a row
   block of BA ≤ r, resolvable for r below the block height b), and abliteration
   (left factor r restricted to the block is still shared across layers) all
   survive. FARMS-style subsampled spectra come from the same blocks.
   *(hypothesis; validate T1 against T2 in the pilot and on the lab. The
   historical scalar data cannot test this; see idea 4.)*
3. **Adaptive escalation.** Fetch T2 (full tensors) only when T1 evidence is
   ambiguous for a query or the model belongs to a stratum that needs full
   spectra: anchors of at most 75B parameters, labeled models, the lab, and the
   probability sample.

**Fine-grained MoE exception** *(measured 2026-10-29)*.
- Expert matrices have only 512–1024 rows, so a 256-row block per expert reads
  20–25% of model bytes, and fingerprints reach 18k–45k views.
- T1 therefore reads **8 seeded experts per layer**, plus shared experts and
  routers. Every expert is read for the lab and anchors.
- Pruned or renumbered experts are still caught at T0 by sha256 and Xet hashes.

If T1 holds up, 100k models need about 75–150 TB of reads instead of about
1.5 PB *(est.)*.

### 5.4 What the existing data already says about capability

Individual-task errors in the September study are 11.4 MAE for IFEval and 1.9
for GPQA. The raw gap is only suggestive: normalized GPQA scores bunch near the
floor, while IFEval spans roughly 0–90. Comparing the two requires a scale-free
measure (MAE divided by the target's standard deviation, or R²). If the gap
survives that, it is what one would expect if knowledge and reasoning are
inherited from the base model while instruction following is set by
post-training. That hypothesis is close to folk wisdom and has prior work
(PhyloLM, Tamura et al., Laufer et al.). Its value here is to be *measured
correctly*:
- per skill;
- per edge type;
- with binomial measurement error removed (near-floor GPQA otherwise pulls
  heritability toward zero);
- against weight distance.

Idea 2 does this cheaply, and the answer predicts where lineage priors will and
will not work.

## 6. System design

### 6.1 Evidence tiers

| Tier | Contents | Bytes read *(est.)* | Compute | Unlocks |
| --- | --- | --- | --- | --- |
| **T0 manifest** | `model_info` at the pinned commit (declared base and relation, license, gated flag, card); file sha256; Xet chunk lists; safetensors or GGUF headers (names, shapes, dtypes, offsets, quantization config) | About 100 KB per shard | Negligible | Exact duplicates; re-shards; tensor-level sharing; architecture signature; declared graph; quantization method |
| **T1 row-block signature** | For every matrix view (section 6.6): one contiguous row block (b = 256; start `min((rows−256)//2, 1920)`); an orthonormal two-sided sketch; block hash; block spectrum. When the parent is known, the exact block delta (section 6.3). MoE: 8 seeded experts per layer. | About 2–10% of model bytes | CPU, seconds per model | Fine-tune neighbors; merge recipes; edge typing above the noise floor; abliteration (stacked test); up-scaling maps; approximate spectral features |
| **T2 full measurement** | Full tensors: fp32 SVD spectra (current collector conventions), full two-sided sketches, per-tensor sha256; T1 blocks at b = 128, 256, and 512 come free from the same pass | 100% | CPU: 10–30 node-minutes per 7B model *(est.; timed in week 1)*, about 10× that for 35–75B | Every alpha estimator; full-spectrum distances; invariant-product spectra for anchors; robust verification |
| **T3 behavior** | T3-lite: about 100–150 adaptively selected items across skills. T3-full: 300–600 items plus task panels. A refusal probe for flagged models and for the phase-2 sample. | Model fully loaded | GPU: about 5 minutes (lite) to 0.1–0.2 hours (full) per model ≤8B *(est.)* | Measured skills; labels for every capability experiment; the exchange rate |

### 6.2 Building on the current collector *(call)*

- **A self-contained `evidence` package inside Model-Zoo** (implementation plan:
  `Model-Zoo/docs/superpowers/plans/2026-10-28-weight-evidence-extractor.md`).
  It streams one tensor at a time over HTTP range requests, and memory stays
  proportional to the largest tensor, so there is no 6 GiB budget. It reuses
  the collector's numerics for T2 spectra. Runner integration waits for G0.
- **Output formats.**
  - Fingerprints are `.safetensors` files: sketches as tensors, identity and
    coverage as JSON metadata. Anyone can share or load one with standard
    tools.
  - Full spectra stay in the collector's HDF5 layout.
  - Census tables are Parquet.
- **Inputs beyond the Hub.**
  - Local directories, pinned by content.
  - `fingerprint_tensors(iterable)` for private formats (torch DCP, FSDP,
    Megatron) and training loops.
  - The reader honors `HF_ENDPOINT`, proxies, and `HF_HUB_OFFLINE`, and local
    mode makes zero network calls.
- **One tool.** `fingerprint`, `diff`, and `check` all emit JSON and use exit
  codes, so they work in CI. They are used internally from G0 and released
  publicly as zoo v1 after the ICML submission.
- **There are two extraction paths.**
  - **CPU streaming path.** Covers T1 for everything, and T2 for models that
    need no T3. It runs on CPU nodes.
  - **GPU "download once" path.** Covers models selected for T3. Their files go
    to local scratch on a GPU node, T2 and T3 run there, and the files are then
    deleted.

  So the T2 set contains the T3 set, and no model is downloaded twice.
- **Network access.** If CPU partitions have no outbound HTTPS, readers run on
  transfer nodes or behind an HTTP proxy. A pinned range read does not care
  which host makes it.
- **Request budget.** Each file is resolved once, and its redirected CDN URL is
  then range-read (section 15.2). In the worst case, where every one of the
  roughly 250 per-model range reads counts as a resolve, a PRO account still
  manages about 14k models a day. If the pilot shows requests binding anyway,
  the reader merges adjacent small tensors into larger ranges, trading bytes for
  fewer requests.
- **Packed formats.**
  - FP8, MXFP4, and NVFP4 decode exactly in the extractor (section 6.6).
  - GGUF (Q8_0, Q4_K, Q5_K, Q6_K, via `gguf-py`) is dequantized for **lineage
    only** in zoo v1. GGUF-only repositories are 16% of the top 1,000 and
    carry 8% of downloads (measured 2026-10-29).
  - GPTQ, AWQ, and bitsandbytes stay parked.
  - All of these are recorded as "reconstructed effective weights", a distinct
    kind of evidence. Spectral science on packed storage stays out of scope.
  - For P1, the lab's quantizations are simulated: quantized, dequantized, and
    saved as bf16. Real packed-format lineage is claimed only after zoo v1.
- **Tokens.** One token, the owner's `HF_TOKEN`, kept in the workspace `.env`
  (revised 2026-11-01).

### 6.3 Sketch specification *(call)*

Revised 2026-10-29 after the correctness review's simulations.

- **Operators: orthonormal, nested, side-specific.**
  - For each dimension d and side (rows or columns), draw a 256×d Gaussian
    from the seed `[20261028, d, side]` and orthonormalize its rows by QR with a
    sign fix. The size-k operator is the first k rows.
  - Nesting is exact: a 64×64 sketch is the top-left block of the 128×128 one.
  - Orthonormal rows keep sketch-domain noise white. Unscaled Gaussians had
    condition number 2.9 at k = 64 and 5.6 at k = 128.
  - Side-specific seeds remove a trace and transpose leak for square matrices.
- **Sizes:**
  - **64 for the census:** about 16 KB per view;
  - **128 for the lab and anchors:** about 64 KB;
  - **up to 256** where an experiment shows it helps. At k = b = 256 the
    row operator is invertible, so larger is pointless.
- **What a stored sketch can do.**
  - Retrieval, relatedness, and merge regression all work on the common block.
  - Rank estimated *from a stored sketch* is reliable only to about **k/4**:
    about 16 at k = 64 and about 48 at k = 128. Gavish–Donoho's median rule
    breaks above k/2, and the sketched factor shrinks as (1 − √(r/k))².
  - Higher-rank questions use the extraction-time block delta below.
- **Extraction-time block delta.**
  - When the parent is known (declared, or retrieved), the extractor compares
    the child's row block with the parent's identical rows while both are in
    memory.
  - Parent blocks are cached internally and never published.
  - The delta is computed whenever both share `(r0, r1, cols)`. Because the
    row start is capped at 1920, vocabulary growth no longer shifts the
    embedding block.
- **Rounding noise and the floor (corrected).**
  - The delta of two bf16 checkpoints is the update quantized to the bf16 grid.
  - Below half an ulp the "noise" is exactly −Δ, so its variance tracks |Δ|.
    It is heavy-tailed and heteroscedastic, not white.
  - RMS rounding error for Gaussian weights is about 1.66e-3 of |w|.
  - Simulated on a 256×4096 block: raw Gavish–Donoho **over-counts** below a
    relative update ρ ≈ 1.5–3e-3 (for example r = 8 at ρ = 1e-3 gives 18–20)
    and on heteroscedastic weights. The earlier floor of 3e-5·√r was 10–30×
    optimistic.
  - **Procedure:**
    1. If the fraction of bit-identical elements exceeds 0.5 (about ρ < 3e-3),
       report `below_floor` and no rank.
    2. Otherwise, biwhiten the delta with the parent's known ulp²/12 variance
       profile (Sinkhorn scaling).
    3. Threshold at the known-noise Marchenko–Pastur edge with a 10% margin.
    4. Report a gap-based confidence.
  - Simulation shows biwhitening recovers r = 8 and r = 64 on heteroscedastic
    parents.
  - **Never type a rank-1 edit from a single tensor.** Abliteration is detected
    only by the stacked test across layers (idea 9).
- **What rank can and cannot separate.** Both LoRA and full fine-tuning have
  decaying spectra. A rank count alone does not separate LoRA rank 64 from a
  full fine-tune: at decay 0.7ⁱ, r = 64 reads as 9–12. Edge typing therefore
  combines:
  - the rank;
  - the spectral gap;
  - the per-module change pattern (which modules changed);
  - the fraction of bit-identical elements.

  The lab measures how well that combination works. It is not assumed.
- **Sketch values stored in fp32.** Rounding them to bf16 would add noise the
  size of a typical delta.
- **Size.** At k = 64, about 16 KB per view, so about 0.4 TB for 100k dense
  models. MoE fingerprints stay comparable in size because of the expert subset
  (section 5.3); a k = 128 census would be about 1.6 TB per 100k models.
  Retrieval slices the top-left 32×32 block at query time.

### 6.4 Data model

This is Plan B's record set with Plan A's fields added.

| Record | Essential fields |
| --- | --- |
| Checkpoint artifact | Repository, resolved commit, file inventory with sha256 and Xet IDs, adapter and base dependencies, selection stratum, **inclusion probability** |
| Measurement | Artifact, tier, collector and convention versions, numerics, canonical tensor name, row-block indices, coverage, spectrum and sketch locations |
| Evaluation | Artifact *and deployment identity*, task and item IDs, harness, prompt, decoding, runtime, source (ours, `.eval_results/`, Every Eval Ever, leaderboard) |
| Relationship | Declared or detected, edge type, evidence tier (verified, likely, declared), p-value or e-value, direction evidence |
| Recommendation experiment | Query, eligible set, evidence purchased with costs, final choice, regret |

Storage is per-model fingerprint `.safetensors` files and the collector's HDF5
spectra, plus Parquet tables and DuckDB for analysis.
Counts are kept separately for repositories, revisions, unique tensor contents,
roots, derivatives, adapters, quantized variants, and successful measurements
(Plan B).

### 6.5 Which models *(call)*

Plan A's strata, in fill order:

1. anchors;
2. head (most downloaded);
3. labeled (public evaluation records);
4. long tail, stratified;
5. lab;
6. prospective set.

Additions:

- T0 covers the whole candidate pool, across **all modalities** (section 6.6):
  every repository with safetensors, GGUF, or `.bin` files.
  - `.bin`-only repositories are 13% of the top 1,000 and carry 14% of
    downloads, mostly vision and audio.
  - They form a T0-only stratum, because pickles are not range-read.
- Strata are modality × size. The papers keep their depth in LLMs and
  vision-language models. Vision, audio, image generation (including
  diffusion LoRAs), and embedding models get their own census strata at modest
  sample sizes. The code path is the same, so the cost is mainly bytes.
- A **probability sample** of 2–5k models goes to T2. It is drawn with known
  inclusion probabilities from the T0 frame restricted to **models of at most
  75B parameters**, so prevalence claims are design-unbiased for that stated
  population.
- A random **phase-2 subsample** of it gets the refusal probe and a card audit,
  which measures the detectors' sensitivity and specificity (idea 14).
- Models above 75B parameters, including large MoE models, are few. They are
  **enumerated, not sampled**: every one gets T0 and T1, and they are reported
  as a separate census stratum. Anchors in this class get T1 only, because full
  T2 would cost 20–40 TB *(est., Plan A)*.
- Size strata explicitly include 7–75B models, which the current 6 GiB default
  excludes.

### 6.6 Every model type, one simple abstraction *(team directive, 2026-10-29)*

Researchers want one tool that works on whatever model they have, and the field
keeps moving toward mixture-of-experts, multimodal, merged, FP8, and
adapter-based releases. The design therefore works on **stored tensors, not
architectures**.

Revised 2026-10-29 after the universality review read live headers from 18 Hub
repositories (gpt-oss, Qwen3.6 MoE, Qwen3-VL FP8, the NVFP4 derivative, DeepSeek
V3.1, Mixtral, SDXL, a kohya LoRA, PEFT, GPTQ/AWQ, timm, Whisper, and others).
Two independent choices are made per stored tensor:

**How the numbers are decoded:**

| Decode | Rule |
| --- | --- |
| plain | F64, F32, F16, BF16, FP8 without a scale |
| FP8 scaled | The scale always multiplies. It may be F32 or BF16, and per-tensor, per-channel, block, or one 3D grid per expert. **The block size comes from `quantization_config.weight_block_size`**: shapes cannot settle it (DeepSeek's 576-row tensor fits blocks from 116 to 143). |
| MXFP4, NVFP4 | Exact: a 16-entry FP4 lookup × block scale (× global scale) |
| GPTQ, AWQ, bitsandbytes, GGUF | Recorded gap; GGUF arrives in zoo v1 for lineage only |

**How the decoded tensor becomes matrices:**

| Geometry | Rule | Covers |
| --- | --- | --- |
| matrix | 2D, after squeezing leading 1-dims | dense layers, embeddings, ViT projections, per-expert weights, merges |
| expert slice | 3D **only if** the name contains `experts` or dim 0 equals the configured expert count. Oriented (out, in) using `hidden_size`, because Qwen3-VL and Qwen3.6 store experts in opposite orientations. Fused `gate_up_proj` is split into gate and up, except gpt-oss, which interleaves them. 8 seeded experts per layer at T1. | fused and unfused MoE |
| conv flattened | Any other 3D, 4D, or 5D tensor: output channels × the rest. T2 spectra keep the collector's convolution convention. | CNNs, patch embeddings, Whisper and audio front-ends, diffusion UNets, linear-attention conv1d |
| LoRA product | Pairs of `lora_A`/`lora_B`, `lora_down`/`lora_up`, `lora.down`/`lora.up`, and `lora_embedding_A`/`lora_embedding_B`. Scale α/r (rslora: α/√r; per-module patterns honored; kohya α from `.alpha`, defaulting to r). The T1 sketch uses the same row block as a merged child, so "base plus this public LoRA" can be tested against a merged model. | PEFT, kohya, and diffusers adapters for LLMs and diffusion |

**Simple rules that prevent mismeasurement:**

- **Never a matrix:**
  - scale and quantization side tensors (`_scale`, `_scale_inv`, `_scale_2`,
    `scales`, `qzeros`, `g_idx`, `absmax`, `quant_map`, `SCB`, `input_scale`);
  - `_bias`, `.alpha`;
  - integer buffers (`position_ids`, `num_batches_tracked`).
- **Module poisoning.** If any tensor under a module is packed with no decode
  rule, its sibling tensors are not measured as if they were weights.
- **Vector-like:** a matrix with fewer than 8 columns is skipped. Without this,
  Qwen3.6's `conv1d` `[8192, 1, 4]` would become about 246k junk views. A
  matrix with fewer than 8 rows but at least 8 columns is a decision or
  classification head and is measured as a `head` view (revised 2026-10-31;
  section 6.8).
- **File-set selection.** An index file defines its shard set. Diffusers repos
  keep one non-variant file per component. Root single-files, `.fp16.`,
  `consolidated*`, and `original/` copies are dropped when a canonical set
  exists. This matters because SDXL ships 35 GB for a 7 GB model, Mistral two
  full copies, and gpt-oss a second copy under different names.
- **Duplicates.** Identical block hashes, such as DeepSeek's MTP embedding and
  head copies, are recorded once as `duplicate_of`.
- **Fused labels.** `qkv`, `query_key_value`, `w13`, and `in_proj` views are
  marked `fused`, so analyses can stratify on them.
- **Component.** The label is the subfolder plus the name prefix up to the first
  number (for example `model.visual.blocks` or `unet/down_blocks`). It is used
  for reports only, never as a measurement input.
- **Unsupported adapters.** LoHa, LoKr, and DoRA magnitudes are recorded as
  `unsupported_adapter_type`.
- **Coverage.** Every stored tensor appears in coverage with its decode, its
  geometry, and its views or a skip reason.

**Simplicity rule.** One reader, one decode × geometry table, one fingerprint
format. A new model type means a new table row, never a per-architecture code
path. Nothing more advanced is adopted unless an experiment shows the simple
version fails.

Rows added on 2026-10-31, after checking System One and other trending
repositories against this table, are listed in section 6.8.

### 6.7 The zoo and its query framework *(added 2026-10-31; the core product)*

**Store.** Simple and rebuildable. It is built by Model-Zoo's `zoo` package
and queried through Zoo-Keeper's `zoo_keeper` package (2026-11-01):

- Parquet tables queried with DuckDB;
- fingerprint `.safetensors` files and HDF5 spectra, referenced by path;
- no database server.

It is rebuilt from the extractor's outputs and score sources, so it can always
be regenerated.

| Table | One row per | Key fields |
| --- | --- | --- |
| `models` | pinned revision | repo, revision, modality, size, format, license, gated, declared base and relation, downloads, stratum and inclusion probability, coverage summary |
| `views` | measured matrix | model, view name, geometry, component, block hash, fingerprint path and index |
| `lineage` | candidate edge | child, parent, estimand (same root, ancestor, or parent), edge type, merge coefficients, evidence tier (verified, detected, or declared), score or p-value, method version |
| `scores` | model × benchmark × skill | value, source (ours, `.eval_results/`, Every Eval Ever, leaderboard), whether the revision is verified, harness, item count |
| `predictions` | model × skill | predicted value, interval, method, inputs used (parent scores, fingerprint neighbors) |
| `coverage` | stored tensor | decode, geometry, status, skip reason |

**Query framework.** A Python library and a CLI. A thin MCP wrapper comes after
zoo v1, because coding agents are a large share of Hub traffic.

- `find`: filters over metadata and measured facts (modality, size, license,
  format, verified-score thresholds).
- `info MODEL`: one card with identity, metadata, measured facts, scores
  (measured kept separate from predicted), lineage with evidence tiers, and
  coverage.
- `similar MODEL`: nearest fingerprints by common-block cosine, exact search
  (about 205 MB for 100k models at 512 dimensions), near-duplicates collapsed.
- `family MODEL`: ancestors and descendants, edge types, and license chain.
- `diff A B` and `check MODEL`: from the fingerprint tool (idea 19).
- `recommend --task ... --constraints ... [--examples FILE]`. This is Plan B's
  core flow:
  1. filter by hard constraints;
  2. rank by measured scores where they exist, and by predicted scores with
     intervals elsewhere;
  3. collapse near-duplicates;
  4. optionally run a quick test on the user's 5–50 examples (successive
     halving over the shortlist).

  It returns a short list with evidence and uncertainty. **Predicted scores
  are always labeled as predictions.**

  **Predictions start simple:**
  - the parent's measured score plus a per-skill shift learned from documented
    parent–child pairs;
  - fingerprint nearest neighbors;
  - abstention when the interval is too wide.

  The heritability study (idea 2) tells us, per skill, how far a parent's score
  can be trusted for its children.

**Releases:**

- **Zoo v0 (G0, 2026-12-04, internal):** the pilot and lab models; `find`,
  `info`, `similar`, `family`, `diff`, and `check`.
- **Zoo v1 (February–March, public):** the census subset collected by then;
  the library and CLI on pip; a demo Space; `recommend` without the quick test.
- **Zoo v2 (May, with P3):** 50–100k models at T1, with the dataset snapshot.
- **The selection paper (September):** `recommend` with the quick test,
  evaluated end to end.

### 6.8 System One models and other trending types *(checked 2026-10-31)*

The team asked that System One models, and other trending architectures that
are promising and useful, be included and queryable. Not everything needs
support.

**What System One models are.**
- *Decision models*: an encoder, or a decoder, with a decision head. They
  return a typed choice, a score, or a yes/no with calibrated confidence,
  instead of generated text.
- **Ecosystem:** a registry (systemonemodels.ai) and a benchmark,
  System1-Arena (arXiv 2610.21244): about 92k questions over 105 tasks, with
  10 System One models against 20 LLMs. Its findings:
  - the best System One models match fast LLMs on static decisions, with
    better calibration at about a quarter of the latency;
  - they rarely succeed in agentic loops.
- **Scale (measured on the Hub API):**
  - 382 repositories tagged `system-one` or `decision-model` within about six
    weeks;
  - the top family has millions of downloads;
  - most are Qwen3.5 decoders (9B–35B) with decision heads or answer-token
    readout; others are ModernBERT and DeBERTa encoders and GLiNER;
  - 21 of the top 45 are GGUF-only;
  - none of the top 45 ship the registry's `systemone.yaml` manifest.

**What was checked.**
- Live safetensors headers and configs of 28 repositories:
  - 14 System One models;
  - one or two from each family on the Hub's trending list of 2026-10-31:
    hybrid linear-attention and convolution LLMs, a diffusion MoE language
    model, omni models, video diffusion, TTS, NeMo ASR, robotics, image-to-3D,
    a multimodal embedding model, DeepSeek-V4.1, a 1T FP8 MoE, and GGUF
    i-quants.

**Already handled** by the decode × geometry table:
- hybrid and linear-attention models;
- the diffusion language model;
- omni models (through the nested `text_config`);
- video diffusers;
- TTS and NeMo models (which ship safetensors);
- the embedding model;
- the 1T FP8 MoE;
- NVFP4 System One models;
- robotics checkpoints.

**Rows and rules added** (extractor plan, Tasks 1, 4b, 5, 7, and 8):

| Gap found | Change |
| --- | --- |
| Decision heads (`classifier [2, 384]`, `score [1, 2560]`, laya's `act_head [2, 256]`) were skipped as vector-like | New geometry `head`: fewer than 8 rows and at least 8 columns, read whole. Depthwise convolutions (few columns) are still skipped. |
| System One models could not be recognized | A T0 `decision` block from tags, `id2label`/`num_labels`/`problem_type`, and classification architectures. The frame gets `system_one` and `is_decision`, and the pilot a `decision` stratum (20 models, pilot now 320). |
| PEFT LoRA factors stored inside full checkpoints with no adapter config (most System One decoders); MLX LoRA (`lora_a`/`lora_b`, transposed) | Both are LoRA rows. An unknown scale is recorded as `lora_scale_unknown`. |
| Integer buffers (`relative_position_index`, NeMo scalars, bool flags) blocked whole layers | Only packed weight tensors poison a module |
| DeepSeek-V4.x: FP4 experts packed in int8 with E8M0 block-32 scales, and FP8 attention with E8M0 scales, left nearly the whole model unmeasured | E8M0 scales accepted; FP4-in-int8 decoded as MXFP4 when `expert_dtype == "fp4"`; verified against DeepSeek's reference |
| ComfyUI-style single-file repositories ship one component in several precisions | One variant read; the others recorded as `precision_variant`. Bundled LoRAs become `bundled_adapter`. |
| ONNX-only repositories; pickles next to safetensors | ONNX-only is T0-only; pickles are recorded as `pickle_not_read` |
| GGUF is far more common than assumed | Zoo v0 takes GGUF architecture, parameter count, and quantization types from the Hub API at T0, so GGUF models are findable. Zoo v1's GGUF lineage covers every type `gguf-py` dequantizes, including IQ types, and MLX quantization. |

**Queries** (zoo plan):
- `find --decision`, `--system_one`, and `--architecture`;
- `info` shows the decision type, labels, and head views;
- `diff` against the base lists the added head, which is usually the whole
  difference between a decision model and its base encoder or LLM.

**Research.**
- **The lab** adds at least 6 decision-head fine-tunes (section 12), so
  System One lineage has exact truth.
- **The selection paper** considers a decision-task family: System One
  models against LLMs on System1-Arena-style tasks. It is cheap to evaluate,
  and latency and calibration count in the cost.
- **Zoo v1** loads System1-Arena-style scores (accuracy, calibration error,
  latency) through the generic score adapter.

**Not adopted:** GPTQ, AWQ, bitsandbytes, and "comfy int8-convrot"
dequantization (recorded gaps), and per-architecture code for any family.

## 7. Research questions

- **RQ1. Uniquely weights.** How do false parents per query grow with
  candidate-pool size for pairwise methods? At what byte cost can same-root,
  ancestor, and immediate-parent relations, edge type, and merge recipe be
  recovered with per-query error control at 10⁴+ candidates?
- **RQ2. Ecosystem.** What does the open-weight ecosystem look like when it is
  *measured rather than declared*? Covers duplicates, undeclared derivatives,
  license drift, and safety-modified prevalence, all with design-based
  confidence intervals.
- **RQ3. Heritability.** Which capabilities are inherited along lineage, which
  does post-training set, and can weight updates predict the change?
- **RQ4. Exchange rate.** How many evaluation items is weight evidence worth,
  for estimation and for selection, within and across families? In which
  regimes does it beat matched-cost baselines?
- **RQ5 (tool, not paper).** Do simple per-layer change statistics between
  training checkpoints help decide when to re-run expensive evaluations? This
  is answered by tool users, not by a paper.
- **RQ6. The zoo's central question** (added 2026-10-31). Given a task,
  constraints, and a small evaluation budget, how much faster and more reliably
  does a measured zoo get someone to a good model than popularity, model cards,
  leaderboards, or bandit search without a prior? Measured as regret against
  total cost.

## 8. Goals, non-goals, anti-goals

**Goals**

1. A revision-pinned census:
   - T0 Hub-wide, including `.bin`-only and GGUF-only repositories;
   - T1 for 50–100k models across all modalities;
   - T2 for a 2–5k probability sample, with phase-2 detector validation;
   - T3-lite for at most 2k models.
2. A genealogy lab with exact truth, about 120 models, released as a benchmark
   with a scorer.
3. A lineage and modification engine with calibrated, inspectable evidence and
   per-query error control.
4. Published exchange rates, for estimation and for selection: capability is
   measured where cheap and predicted only where evaluation is expensive.
5. **The zoo's query framework** (section 6.7): `find`, `info`, `similar`,
   `family`, `diff`, `check`, and `recommend`, as a Python library and a CLI.
   It uses the open fingerprint tool for every model type, with JSON output,
   exit codes, offline mode, and an in-memory API. Zoo v1 is public in
   February–March. A thin MCP wrapper follows. A hosted service comes only if
   the framework proves effective.
6. Papers in the sequence of section 11.

**Non-goals**

- Distillation into freshly initialized students, or cross-architecture lineage.
  Weights cannot show either; say so explicitly.
- Closed frontier models.
- Replacing serialization scanners; link to their results instead.
- A new general-purpose benchmark.
- Spectral science on packed quantized storage.

**Anti-goals: failure even if the system works**

- A lineage or capability claim without calibrated uncertainty and inspectable
  evidence.
- Leakage through near-duplicates, re-uploads, or merges crossing splits.
- A dataset dump without claims.
- Reconstructive weight information for restrictively licensed models.
- Naming organizations without verification and prior notice.
- **Claiming that weights predict capability where a cheap direct measurement
  would have been better.** The exchange rate must be reported.

## 9. Ideas portfolio, from simple to advanced

Each idea gives a testable claim, its practical use, its cost, and where it
lands. Ideas 2–5 need **no new collection**.

### Level 1: simple and immediate

**1. Hub-wide T0 census.**
- Claim: report the share of repositories that are exact copies or re-shards,
  and the share of tensors shared byte-for-byte across repositories.
- Use: deduplication before any download, which saves bytes; spam and re-upload
  detection; crediting the original uploader.
- Cost: API calls and headers only.
- Lands in: P3, and powers everything else.

**2. Capability heritability, measured correctly** *(hypothesis: knowledge and
reasoning have high heritability; instruction following low)*. Method revised
2026-10-29.
- Data: per-task leaderboard v1 and v2 item-level or score data and Model Atlas
  edges. No weights are needed for v1. The data needs no cluster outputs, so it
  is not gated on the historical copy.
- Method. Pagel's λ and Blomberg's K are not used: they assume a tree with
  branch lengths and traits only at the tips, while the Hub graph has observed
  ancestors, polytomies, and merges. Instead:
  - a REML mixed model with a family random effect and each score's known
    binomial measurement variance;
  - parent–offspring regression on observed base scores;
  - network Brownian motion (Bastide et al., 2018) for merges;
  - raw or logit accuracy, never the floor-clipped normalized scores.
- Weight distance as branch length. Once T1 exists, cheap on about 5k
  leaderboard models: does capability change scale with ‖Δ‖, per skill?
- Heritability per edge type comes once P1 can type edges.
- Prior work: PhyloLM, Tamura et al. (COLM 2025), and Laufer et al. The
  contribution is measurement-error-corrected, per-skill, per-edge-type
  estimates, not the hypothesis itself.
- Use: tells practitioners which skills fine-tunes actually move, and tells us
  where lineage priors can work.
- Cost: CPU-hours.
- Lands in: an ICLR 2027 workshop paper, then the zoo's score predictions and the selection paper.

**3. Exchange-rate study.**
- Method:
  - Start from section 5.1, the prior effective sample size (Morita, Thall,
    and Müller, 2008).
  - Measure k per skill and per held-out setting, for **estimation** (absolute
    error) and for **selection** (error in pairwise differences, within and
    across families).
  - Compare random against adaptive items, in GPU-hours per evaluation type.
- Matched-cost baselines: tinyBenchmarks, BenchPress, and observational scaling
  laws.
- First pass: person C, on existing September predictions, in week 1–3.
- Use: one published table anyone can read.
- Cost: analysis.
- Lands in: the selection paper.

**4. Depth-subset curves from existing data.**
- Method: simulate reading only some layers or roles using the historical
  per-matrix statistics (complete-role subset). Plot grouping ARI, base-match
  accuracy, and score MAE against the fraction of bytes.
- Use: answers Plan A's *depth-subset* question (which layers and roles to read)
  before collecting anything. It **cannot** test row-block reads or sketch-based
  lineage, because the historical data has scalar statistics only and no
  coordinate information. Those are tested in the pilot (T1 against T2) and on
  the lab.
- Data check first: confirm the cluster copy of the historical outputs is
  complete. The local copy has 1,606 zero-byte files. If the cluster copy shares
  those gaps, shrink this study and the idea 5 sample.
- Cost: CPU-hours.
- Lands in: P1 and P3 methods.

**5. How much can unpinned weight studies be trusted? (Release the historical
corpus with this claim.)**
- Decision: no standalone dump; refine the research first (team, 2026-10-28).
- Method:
  - Re-measure a stratified sample of about 500–1,000 historical models with the
    pinned collector.
  - Quantify how often the Hub revision changed since the historical
    measurement.
  - Quantify how much the Gram-versus-SVD numerics and the missing key and value
    matrices shift each feature.
  - Check whether the September grouping, base-matching, and score-prediction
    conclusions survive on the re-measured set.
- Use: the result applies to every published weight study that did not pin
  revisions, including ours. It turns the historical 12.5k per-matrix statistics
  into a documented baseline worth releasing: the largest real-weight LLM corpus,
  25× the published maximum, with its error bars measured.
- Release with P3, or with the February–March zero-byte tensor-sharing note if
  the result is striking.
- Cost: about 1k pinned measurements, mostly on the CPU path, plus packaging and
  a license-terms check.

**6. Robust tail statistics on new spectra.**
- Method: Hill at k = n/2 with closed-form standard error (α − 1)/√k; FARMS
  correction; Marchenko–Pastur-normalized spectra; a matched comparison with the
  reference paper's estimator.
- Use: settles whether alpha adds anything beyond the non-alpha baseline.
- Lands in: the P3 appendix and the selection paper.

### Level 2: linear-algebraic fingerprinting (T1 and T2)

**7. Orthonormal nested sketches and extraction-time block deltas** (section
6.3).
- Claim, on lab truth:
  - stored row-block sketches separate parents from siblings and grandparents;
  - extraction-time block deltas, above the noise floor, give update rank and
    the per-module change pattern.
- Whether those separate LoRA rank 8 and 64 from full fine-tuning is measured on
  the lab, not assumed.
- **The null.** Shared operators center the sketch null at the *population*
  weight cosine of unrelated models, not at zero. Shared signed structure (means,
  same-seed initializations, embeddings) shifts it. The null is therefore always
  estimated empirically from known-independent models, for example the
  PolyPythias seed variants. The theoretical spread, about 1.12/k, is only a
  sanity check.

**8. Merge-recipe recovery.**
Method revised 2026-10-29.
- Model: regress each child sketch on **[parent sketches, base sketch]**, per
  tensor or per layer group, **without a sum-to-one constraint**.
  - Coefficient intervals come from a bootstrap that resamples blocks of
    layers, because tensors within a model are dependent.
  - Coefficients that vary with depth recover mergekit's per-layer `t`
    schedules.
- Theory, corrected:
  - SLERP's coefficients sum to 1 + t(1−t)Ω²/2, so the base enters as a
    regressor.
  - DARE stays consistent in least squares, but its residual variance is
    p/(1−p) times the signal (9× at p = 0.9). That is a strong signature, not a
    small one.
  - TIES and DARE-TIES yield *effective* weights (disjoint mean and trimming),
    not the configured ones, so they are reported as effective coefficients.
  - Task arithmetic has Σc ≠ 1.
  - Identifiability needs full column rank. Report the condition number and
    group near-collinear parents.
- Ground truth:
  - the lab's merges;
  - **real Hub merges that publish `mergekit_config.yml`**: thousands exist,
    and they are free ground truth for the verified-Hub tier.
- Claim: effective coefficients within ±0.05 on lab merges and agreement with
  published configs *(est. target)*.
- Use: merge genealogy, attribution, and license chains through merges.
- Comparison: modelDNA, whose sum-to-one least squares is the baseline.

**9. Edge typing.**
- Adapter or full fine-tune: the biwhitened rank above the floor, the spectral
  gap, and the per-module change pattern (section 6.3). The result is
  `below_floor` when the update is too small to type.
- Abliteration, the **stacked test** (verified): for residual-writing matrices
  `[hidden, *]`, ΔWₗ = −r(rᵀWₗ) puts r on the row side. A row block keeps P_B r,
  which is shared across layers.
  - Stack the layers' block deltas and test for one shared left direction.
  - The block captures about b/hidden of r's energy, so the test pools over
    layers.
  - It is never applied per tensor.
  - For multi-direction or per-layer variants such as Heretic, test for low
    rank instead.
- Quantization: a flat delta spectrum, header dtypes, and the leaf prior.
- Pruning: a sparsity pattern.
- Up-scaling: dynamic time warping over per-layer sketches.

**10. How tree-like is model evolution?** *(hypothesis; revised 2026-10-29)*
- Model: if successive updates are nearly orthogonal, squared distances add along
  lineage paths, so d = ‖Δ‖² is close to a tree metric.
- **The finding** is how far reality departs from that. Measure:
  - four-point-condition violations on the lab and on verified Hub families;
  - sibling update cosine, cos(Δ₁, Δ₂) = (d₁ + d₂ − d₁₂)/(2√(d₁d₂)), as a
    signal of shared training data;
  - how much of the graph is reticulate (merges).
- Assembly:
  - recover merges first, by recipe regression (idea 8);
  - on the remaining tree, use a **minimum-cost arborescence** (Chu–Liu/Edmonds)
    when ancestors are observed, with the upload-time and leaf priors;
  - use neighbor joining only for latent ancestors.
- **Theory, demoted to a remark.** Atteson's neighbor-joining bound needs
  *additive* error below half the shortest internal edge. Johnson–Lindenstrauss
  gives *multiplicative* error. The resulting sketch size grows as
  (d_max/μ)² log N. That is vacuous next to long pendant edges (quantizations,
  full fine-tunes) and for star families (μ = 0).
- Two-sided sketches behave as a Kronecker JL with relative sd
  √(4/(k·r_eff) + 2/k²), so low-rank deltas concentrate poorly.
- Quantization noise is nearly orthogonal to everything, so a quantized model
  behaves as a pendant edge of predictable length.

**11. Calibrated open-set provenance with query-level error control.**
Method revised 2026-10-29.
- Problem: at 100k candidates, a per-comparison false-positive rate of 1e-3
  yields about 100 false matches (Plan B). This is P1's headline: **false
  parents per query, plotted against candidate-pool size for every method.**
- Method:
  - **No Stouffer combination.** Tensors within a model are strongly
    dependent, so combined p-values would be anti-conservative.
  - **Calibrate the combined statistic** (the median sketch cosine over shared
    views) directly against an empirical null of siblings and known-independent
    same-architecture models. Fit a generalized Pareto tail for small p-values.
  - Control false discoveries per query across candidates (BH or e-BH).
  - Run exact permutation tests in the style of Zhu et al. (ICML 2025) on the
    shortlist only.
  - Use Mondrian or weighted conformal sets with a "not in reference" label,
    calibrated **across families**: calibrate on one base family, test on
    another, because lab-to-Hub exchangeability fails.
- **Scope.** T1 assumes no adversary. Function-preserving hidden-unit
  permutation and targeted perturbations defeat a public-seed coordinate
  sketch. Adversarial claims need AWM-style or permutation-invariant
  verification on shortlists, and the paper reports what evasion costs.
- Report:
  - false matches per query at fixed TPR, for each nested estimand;
  - coverage, abstention, and direction accuracy;
  - bytes per answer.
- Must compare against 2602.00772, Witness Overlap, and modelDNA.

### Level 3: statistical models of the ecosystem

**12. Phylogenetic Gaussian-process prior for capabilities.**
- Model: Cov(θₘ, θₘ′) = σ² · t(LCA(m, m′)), with branch lengths t measured as
  squared weight distances (idea 10). Add a kernel on spectral features for
  roots, and use an IRT likelihood over items.
- Prediction: predictive variance grows with update size, which gives a natural
  abstention rule. Testable: is |θ_child − θ_parent| scaled by ‖Δ‖, per skill and
  per layer group?
- Baselines: lineage-regularized matrix factorization, ModelLens-style, Spectral
  Signatures kNN, BenchPress, tinyBenchmarks and ATLAS at matched items.

**13. Update descriptors and merge additivity.**
- Descriptors: delta spectra, which layers changed, intruder dimensions, and LoRA
  R_B R_Aᵀ.
- Claim (Plan B): among same-base derivatives, update descriptors separate useful
  specialization from regressions better than lineage alone.
- Add: is θ_merge ≈ Σ cᵢ θᵢ? Measured per skill, merge super- or sub-additivity
  is itself a finding.

**14. Design-based prevalence, corrected for detector error.** Revised
2026-10-29.
- Horvitz–Thompson estimates prevalence of *detections*, not of the truth. The
  design is therefore two-phase:
  - **Phase 1:** run the detectors on the 2–5k probability sample.
  - **Phase 2:** draw a random subsample from it. Apply T2, the refusal probe,
    and the card audit to measure sensitivity and specificity.
  - **Correct** the prevalence estimates by double sampling (or Rogan–Gladen).
- **Weighting.** Report repository-weighted *and* download-weighted estimates,
  because 1.5% of repositories carry 99.2% of downloads.
- **Nonresponse.** Adjust for gated and failed models, which are reported as
  coverage gaps.
- **Names say what is measured.** "Refusal-direction-ablated (stacked test,
  probe-confirmed)", not "safety-stripped": uncensoring by fine-tuning has no
  rank signature.
- Targets: undeclared derivatives, laundering, license-breaking chains,
  duplication, and refusal ablation.

### Level 4: decisions

**15. Cost-aware selection with heterogeneous evidence acquisition** (Plan B's
idea 3, generalized).
- Actions: T1 read, T2 read, k items of task t, a latency measurement, or one
  user label.
- Policy: knowledge gradient or expected improvement under the idea 12
  posterior.
- Metric: regret against total cost, including bytes. Shortlist recall is
  reported separately.
- Evaluated where evaluation is expensive: at least one agentic or judge-scored
  family, in a later stage. The default is self-hosted: a local open judge model, and agentic tasks
  graded by execution. Cost is therefore measured in GPU-hours. Paid API judges
  or human raters are added only if a cash budget exists *(call)*.

**16. Deployment identity.**
- Recommend (checkpoint, quantization, runtime, template). The capability of a
  quantized variant is modeled as the parent's θ plus a degradation term by
  method and bit width, learned from measured T3 pairs.
- Use: Plan B's "cheaper replacement for a known model" query.

**17. Prospective, pre-registered predictions.**
- Freeze the lineage method at G0 (2026-12-04).
- Every model uploaded after that date is a prospective query. P1 then has
  about two months of prospective queries at submission. Publish a hash of the
  frozen predictions, and score them later for P3 too.
- Use: credibility and attention.

### Level 5: for frontier labs and everyone else

**18. Checkpoint diffs in training (a tool feature, not a paper).** Revised
2026-10-29.
- **Why P6 was cut.** On pretraining trajectories, validation loss (free to
  labs) already predicts static evaluations. And "the weights are local" also
  means "the model is local", so 100 items cost GPU-minutes. The valuable
  regime is RL or post-training checkpoints with expensive agentic
  evaluations, and that needs lab data the team does not have.
- **What ships instead,** in zoo v1: a training callback of about 50 lines
  for Trainer, Lightning, or torchtitan, built on `fingerprint_tensors()`.
  - It logs per-layer update norm, the fraction of bit-identical elements, and
    biwhitened rank against both the previous save and the base, to W&B or
    TensorBoard.
  - It comes with one documented rule: re-run expensive evaluations only when
    weight change since the last evaluated checkpoint exceeds τ.
- If a lab partner offers RL trajectories with expensive evaluations, this
  becomes a study.

**19. A diff for model weights: measured facts only.**
- **Where it matters** (clarified 2026-10-31): between *different* models:
  - a base and its fine-tune;
  - two fine-tunes of the same base;
  - a merge and its parents;
  - a quantized copy and its original;
  - a lab's own training checkpoints, which change a lot and are mostly
    private.
- **Revisions of one Hub repository** usually keep the same weights; new
  commits are mostly READMEs, configs, chat templates, or re-sharding. A diff
  there mostly confirms "identical", which is cheap verification but not the
  main use. Idea 5 measures how often released weights really change.
- Given two checkpoints, or one and its declared base, report per view:
  - whether it is identical, changed, added, removed, or reshaped (for example,
    vocabulary rows added);
  - cosine;
  - the fraction of bit-identical elements;
  - the relative update norm;
  - rank with its confidence, or `below_floor`.
- Then a per-component summary and merge coefficients when `--parents` is given.
- **No predicted skill changes.** The first wrong prediction would cost the tool
  its credibility, and section 5.1 says to measure skills directly.
- "Related" verdicts appear only with a calibration file; until then the output
  reads "uncalibrated, cosine X against null Y".
- Works offline on private models, emits JSON, and uses exit codes for CI.

**20. Give the measurements back to the ecosystem.**
- Suggested `base_model` corrections offered to HF, each with its evidence tier.
- **A derivative-watch script** (zoo v1). Given a lab's models, it:
  - filters the daily hub-stats dump by header-shape signature;
  - runs T1 on new matching uploads;
  - writes a Markdown/CSV report.

  It runs as cron or a GitHub Action, with no hosted service.
- **Seeds and adversaries.** The public seed suits honest-provenance questions.
  Adversarial monitoring should use a private seed (`--seed`), because a known
  seed can be targeted.
- Standard exports (CycloneDX ML-BOM and SPDX AI) as file formats, not as a
  product.

## 10. Evaluation protocol

This keeps the current culture: frozen protocols, input hashes, audits, and
visible failures.

- **Identity.** Join scores only on matching commits and deployment identity.
  Report the fraction of joins that verify.
- **Leakage.** Group near-duplicates by tensor hash, sketch similarity, and
  re-uploads *before* splitting. Track multi-parent edges that cross partitions.
- **Splits.** Near-duplicate groups kept together; root held out; task held out;
  collection transfer reported per direction; prospective; naturally
  undocumented models (Plan B). Artificial masking is an extra test, not a
  replacement.
- **Lineage** (revised 2026-10-29):
  - **Three nested estimands:** same root, ancestor, and immediate parent. Each
    has its own error rate. Scoring only against declared labels would penalize
    correctly detected misdeclarations.
  - **Three truth tiers:**
    1. exact lab truth;
    2. verified Hub truth: byte-identical tensors from T0, vendor pipelines that
       release their intermediates (OLMo-2 and Tülu 3 SFT → DPO → RLVR,
       SmolLM2, prune-and-distill releases), and published
       `mergekit_config.yml` files;
    3. declared labels, audited for noise by the Claude card audit.
  - **Hard negatives** are known-independent same-architecture models (for
    example, the PolyPythias seed variants) and same-series siblings, *not*
    "different declared roots". Undeclared derivatives contaminate that easy
    negative set.
  - **A blind laundering set** (permutation, rescaling, rotation, and targeted
    perturbation) is built by a teammate who does not work on the lineage
    methods.
  - **Cross-family calibration:** calibrate on one base family and test on
    another.
  - **Report** false matches per query at fixed TPR against candidate-pool
    size, abstention coverage, direction accuracy, and bytes per answer.
  - **Evidence ladder on the same queries:** metadata → T0 → T1 → T2 → a
    black-box test (200-pair subset).
- **Capability.** Per-skill errors; noise ceiling; cost-matched item baselines;
  out-of-range predictions kept visible; **exchange rate reported**.
- **Selection.** Regret, total cost, constraint violations, shortlist recall,
  calibration, abstention, and coverage (Plan B).
- **Evidence ladder with costs.** Metadata → historical evaluations → T0 → T1 →
  T2 → update descriptors → T3 probes. Report a cost-effectiveness frontier, not
  just accuracy.
- **Prevalence.** Design-weighted estimates with confidence intervals; manual
  audits of Hub lineage labels.
- **Protocol freeze.** One frozen config per paper before its final run, as in
  `configs/htsr_combined_study_v4.json`.

## 11. Papers and venues *(dates est. from recent years unless marked)*

With 2–3 people the papers run **in sequence**, one main submission at a time,
each reusing the artifacts of the one before *(call)*.

Revised 2026-10-29.

| Order | # | Working claim | Core evidence | Target |
| --- | --- | --- | --- | --- |
| 1 | P1 | **Pairwise lineage accuracy does not scale; typed, calibrated lineage does.** At 10⁴+ candidates, published methods return many false parents per query. This method gives typed lineage (edge type, merge recipe) with per-query error control, from a laptop, using one pipeline for LLMs, MoE, and diffusion LoRAs. | The genealogy lab (about 120 models, exact truth, released with a scorer); the verified-Hub tier including mergekit configs; a T1 candidate pool of 5–20k with 2–5k labeled queries; prospective queries since G0; false parents per query against pool size and accuracy against bytes for Trace, Spectral Signatures, modelDNA (matched bytes), Witness Overlap (direction), 2602.00772, MoTHer, metadata imputation, and black-box testing (200-pair subset) | ICML 2027 (abstract about Jan 22, paper about Jan 28; est.). Fallback: the benchmark framing at ICML, then COLM 2027 (about late March; est.). KDD's February cycle accepts only resubmissions. |
| 1b | P2 | Which skills post-training actually moves: heritability per skill, corrected for measurement error | Idea 2 on existing data (person C, from week 1) | ICLR 2027 workshop (about February); folded into the selection paper |
| 2 | — | Zero-byte note: the Hub's tensor-sharing graph and how many repositories silently changed their weights | T0 headers and Xet hashes Hub-wide; idea 5 | arXiv with zoo v1, February–March |
| 3 | P3 | **The Model Zoo: the Hub measured, not declared** (revised 2026-10-31: this is the zoo release) | The zoo itself: 50–100k models with fingerprints, lineage, scores, and coverage, plus the query framework; design-unbiased, error-corrected prevalence (two-phase) of duplicates, undeclared derivatives, merges, license chains, and refusal-direction ablation, repository- and download-weighted; benchmark tasks and splits; prospective scoring | NeurIPS 2027 Evaluations & Datasets (about early May; est.) |
| 4 | Selection | **Choosing a model from a measured zoo of 100k with little evaluation** (revised 2026-10-31) | `recommend` evaluated end to end on dense panels in 2–3 task families (one self-hosted expensive-evaluation family): regret and total cost against strong non-weight baselines (popularity, metadata, historical scores, BenchPress, ModelLens-style ranking, bandit search without a prior). The exchange rate for estimation and selection, and corrected heritability per edge type, are results inside it. | ICLR 2028 (about late September 2027; est.) |

**Cut:** P6, checkpoint triage, as a paper (idea 18 explains why). It survives
as a tool feature. The zoo could also become a model-lake systems paper (VLDB
takes monthly submissions until 2027-03-01, *verified by Plan A*). That is
optional and not scheduled.

### 11.1 Why this would earn attention

Each item below is something a top lab, researcher, or practitioner would want
to use or cite.

- **The zoo itself.** A free, queryable database of tens of thousands of open
  models across every modality. Each has measured weights, lineage, and scores,
  and an honest coverage label, all through one library and CLI. Nobody else
  offers measured answers to "what is this model, what is it like, and what
  should I use?"
- **A scaling failure everyone can see.** One plot: false parents per query
  against candidate-pool size, for every published lineage method. It is the
  plot reviewers and practitioners remember.
- **The Hub's family tree, measured.** The largest real-weight LLM dataset by at
  least an order of magnitude. Its measured genealogy exposes undeclared
  derivatives, merge recipes, and refusal-ablated models that model cards do
  not show. Every prevalence figure is error-corrected.
- **A benchmark others will use.** The genealogy lab, with exact edges and a
  scorer. Earlier work evaluated on 15–176 models.
- **A crisp scientific finding.** How tree-like model evolution really is, and
  which capabilities post-training actually moves, measured correctly.
- **A useful sobering number.** The exchange rate tells everyone working on
  weight-based prediction where it pays and where it does not.
- **Prospective, pre-registered predictions** for models not yet uploaded.
- **A tool people install the same day** (zoo v1, after submission):
  - paste a Hub ID, or two, and get within about a minute, from a few percent
    of the bytes, a per-layer map of what changed;
  - the effective update rank;
  - vocabulary rows added;
  - whether the declared base is consistent with the weights;
  - the recovered merge recipe next to the published mergekit config.
  - **Launch-week gallery:** diffs of recent headline open releases against
    their bases.
  - **Afterwards:** a diff within hours of each major release. That recurring
    "here's what they actually changed" image is what people share.

## 12. Milestones and gates *(call)*

Revised 2026-10-29 from the feasibility review.

**Capacity.** From 2026-10-29 to the ICML deadline there are about 31
person-weeks with three people and about 21 with two, after holidays and 10%
overhead. The P1 path costs about 27 person-weeks after the cuts below, so
three people fit and two need the two-person variant in section 1.1.

**Minimum genealogy lab** (about 120 models; 150–300 GPU-hours), on
permissive bases:

- **Bases:** at least 6, from 3 families: Qwen2.5-0.5B/1.5B/7B,
  Qwen3-0.6B/1.7B, SmolLM2, OLMo-2-1B. *Avoid Qwen2.5-3B*, which is under the
  Qwen Research License.
- **Merges:** at least 30 (linear, SLERP, TIES, DARE; 2–4 parents; some with
  per-layer t).
- **LoRA:** at least 24 runs at ranks 8, 64, and 256.
- **Full fine-tunes:** at least 8 SFT and 8 DPO runs at ≤1.7B.
- **Abliterations:** at least 8, classic and Heretic.
- **Simulated quantizations:** at least 12 (quantized, dequantized, saved as
  bf16).
- **Laundering transforms:** at least 20 (permutation, rescaling, rotation,
  targeted perturbation). Part of the set is built blind by person C.
- **Chains:** at least 3 of depth ≥3.
- **Decision heads** *(added 2026-10-31, section 6.8)*: at least 6
  sequence-classification fine-tunes. Four go on lab decoder bases and two on
  ModernBERT-base, and one is stored as PEFT factors inside a full checkpoint,
  as many System One models ship. Each costs minutes of GPU time.
- **Hard negatives,** free: PolyPythias (5 sizes × 9 seeds, same data, different
  seeds) and same-series siblings.
- **Cut:** GRPO and full-parameter 7B training, which need GPU reservations the
  site does not offer.

### Week-by-week to ICML

A owns infrastructure, B the lab and methods, C the studies and baselines.

| Week | A | B | C |
| --- | --- | --- | --- |
| Oct 29–Nov 6 | Day-one checks (internet from CPU nodes, CPU SVD timing, scratch, Xet); Tasks 1–3; **freeze the fingerprint format** | Lab plan and edge list; merges and laundering; queue LoRA | Housekeeping; heritability and exchange-rate data (gated requests already filed) |
| Nov 9–13 | Tasks 4b, 2, 5 | Task 4; abliteration; simulated quantization; SFT/DPO; engine on synthetic fingerprints | Heritability statistics; Trace and Spectral Signatures reimplementations |
| Nov 16–20 | Task 6; lab v0 fingerprinted | Merge regression; rank and edge typing | Hard negatives (PolyPythias); Task 7; zoo `scores` table from leaderboard and Every Eval Ever (verified joins only) |
| Nov 23–27 | Tasks 8–9; launch the pilot; zoo store (tables from manifests and fingerprints) | Stacked abliteration test; tree-likeness (**pivot check 11-27**) | Workshop draft |
| Nov 30–Dec 4 | Pilot report; **zoo v0 queries** (`find`, `info`, `similar`, `family`); **G0 (12-04)**; freeze the method and start the prospective set | Calibration (empirical null, generalized Pareto tail, FDR); lineage edges into the zoo | Hub query set; Claude label audit |
| Dec 7–11 | Hub T1 candidate pool (target 5–20k) | Lab complete; **G1** | modelDNA, Witness Overlap, MoTHer |
| Dec 14–18 | Byte curves; T2 subset | Full evaluation; **pivot check 12-18** | Baselines on the Hub; black-box subset; finish the workshop paper |
| Dec 21–Jan 1 | Rerun buffer | Theory remarks | Related work |
| Jan 4–8 | **Freeze** | Results | Methods |
| Jan 11–15 | Figures; anonymize code | Write | Write |
| Jan 18–22 | **Abstract** | Polish | Supplement |
| Jan 25–29 | **Paper** | | |

### Gate G0 (2026-12-04): thresholds

**Cost:**
- T1 median bytes ≤10% of the model per stratum;
- p90 T1 time ≤10 minutes per model ≤9B;
- failures ≤5%, excluding no-access and packed formats;
- no unhandled 429s;
- a 5k-model T1 run projects to ≤4 days.

**T1 for lineage** (measured on the lab plus known-independent pretrains; Hub
pairs are only a sanity check):
- parent-against-independent AUROC ≥0.98, and no more than 0.01 below T2;
- within-related Spearman between T1 and T2 cosines ≥0.9;
- decision agreement ≥0.9;
- false matches per query ≤0.05 at TPR 0.9 against a 5k pool;
- parent-against-sibling and parent-against-grandparent AUROCs reported.

**T1 for spectra:** median |α_T1 − α_T2| ≤0.1 and per-matrix Spearman ≥0.8.
Otherwise spectra need T2, which affects P3 only.

**Pilot design:**
- T2 runs on the lab, the independents, the Hub related pairs, and 10 models
  per stratum above 9B (about 4 TB), with T1 blocks at b = 128, 256, and 512
  coming free from the same pass.
- T1 runs on the rest of the 320-model Hub pilot (290 until 2026-10-31,
  when video and decision strata were added).

**G1 (12-11).** The minimum lab is complete. If it is not, the Hub set shrinks,
never the lab.

The zoo store and v0 queries add about 1.5 person-weeks for A. That is offset
by the deferred Hub-wide T0, and C's leaderboard work fills the `scores` table.

### After ICML

- **February–March: zoo v1, public, and the zero-byte note.**
  - The query library and CLI on pip.
  - `recommend` without the quick test.
  - A demo Space.
  - GGUF dequantization for lineage.
  - The derivative-watch script.
  - The training callback.
  - The Hub-wide tensor-sharing note with idea 5.
- **February–April: zoo v2 census, sharded across hosts.**
  - T0 Hub-wide.
  - T1 on 50–100k models.
  - T2 on a 2–5k probability sample.
  - Phase-2 detector validation on a random subsample.
  - T3-lite on at most 2k models.
  - Analyses repository- and download-weighted.
  - Rebuttal and reviewing duties for ICML fall here, so budget for them.
- **May:** submit P3, the zoo release, with the dataset snapshot and the query
  framework, after giving notice to any named party.
- **June–September: the selection paper.**
  - `recommend` with the quick test on users' examples, evaluated end to end on
    dense panels in 2–3 task families, one of them a self-hosted
    expensive-evaluation family. A decision-task family (System One models
    against LLMs, section 6.8) is a candidate.
  - Strong non-weight baselines: popularity, metadata, historical scores,
    BenchPress, ModelLens-style ranking, and bandit search without a prior.
  - Exchange rate for estimation and selection, and corrected heritability per
    edge type, as results.
  - **Gate G2** decides how much weight evidence `recommend` uses: weights
    stay in the ranking only where they improve regret per unit cost over the
    best non-weight baseline.
  - Score the prospective set.
  - Hosted services or ingestion are decided only after evidence of
    effectiveness.

## 13. Budget *(est., per unit; size after G0)*

| Item | Per unit | At plan scale |
| --- | --- | --- |
| T0 | About 3 API calls plus about 100 KB per shard | Whole candidate pool. The frame comes from the daily hub-stats dump, and API calls go only to selected candidates; the free tier allows about 288k API calls a day (section 15.2). |
| T1 reads | About 2–10% of model bytes (MoE with the expert subset) | 50–100k models: about 40–150 TB |
| T2 reads | 100% | P1: the lab plus about 4 TB of pilot. P3: a 2–5k probability sample, about 30–75 TB |
| T2 compute | 10–30 CPU node-minutes per 7B model *(est.; timed in week 1)*, about 10× that for 35–75B | About 0.5–2.5k node-hours for 5k models |
| T3-lite | About 5 GPU-minutes per model ≤8B, reusing the T2 download | About 170 GPU-hours for ≤2k models |
| T3-full | 0.1–0.2 H100-hours per model ≤8B | Capability paper only; sized then |
| Dense panels | Depends on the family; agentic families cost the most | One self-hosted family; calibrate after choosing it |
| Genealogy lab | — | 150–300 GPU-hours for the minimum lab |
| Storage | — | Lab checkpoints 1–2 TB; parent-block cache and T2 scratch with a bounded size; spectra about 0.2 TB; fingerprints about 0.4 TB per 100k models at k = 64; headers about 4 GB |
| Claude API | — | Evaluation: at most about $20 a day, and only once judge-scored evaluation starts in a later stage (local judges first). Coding agents: ample, not infinite (section 15.3). |

Plan A's 1.73 PB figure applies only if every model gets T2. Plan B's GPU table
assumed GPU spectral computation for every model. On CPU, with row blocks, both
binding constraints relax.

## 14. Risks

| Risk | Mitigation |
| --- | --- |
| T1 row blocks miss information that full tensors carry | Pilot and lab validation per use; adaptive escalation to T2; T2 for anchors of at most 75B, labeled models, the lab, and the probability sample |
| Scooped on lineage (modelDNA, Witness Overlap, 2602.00772, 2610.30626, the Atlas group) | Differentiate on scale, exact lab truth, edge typing, query-level error control, and byte cost; release the historical corpus with idea 5's reliability claim as soon as that result is solid |
| The exchange rate makes capability-from-weights look small | Built in: it becomes a result of the selection paper, and `recommend` relies on measured scores where testing is cheap |
| The sketch null is not centered (expected: shared operators center it at the population weight cosine) | Always use an empirical null from known-independent models; exact permutation tests on shortlists |
| Model evolution is not tree-like (shared update directions, merges) | That measurement is itself the finding; recover merges first, then use a minimum-cost arborescence or supervised edge scoring |
| Rank typing is wrong near the bf16 floor | Abstain below the floor (fraction identical > 0.5); biwhitened rank with a gap confidence; never type rank-1 per tensor |
| Lab calibration does not transfer to the Hub | Cross-family calibration, the verified-Hub tier, and a blind laundering set |
| An adversary evades public-seed coordinate sketches | State the no-adversary scope; private seeds for monitoring; permutation-invariant or AWM verification on shortlists; report what evasion costs |
| CPU nodes lack outbound internet | Day-one check; readers on transfer nodes or behind a proxy |
| Storage grows without bound (lab checkpoints, parent cache, T2 scratch) | A storage line in section 13; bounded caches; delete T2 scratch per model |
| ICML rebuttal and reviewing collide with the P3 census | Budget rebuttal weeks in February–March; shard the census across hosts |
| The owner's account is the only gated-license holder | The owner's token is used for gated repositories only; members use their own tokens for ungated reads |
| HF rate limits or terms of service at bulk scale | Measured limits do not bind even in the worst case (section 15.2); always authenticate; never automate license acceptance or multiply accounts; deduplicate before download |
| Legal exposure (gated models, licenses) | Derived and non-reconstructive statistics only; fp32 sketches are 4,096 numbers per tensor; a takedown path; a read of each family's terms before release |
| Misuse (detector evasion; harassment through the audit) | Aggregate reporting; notify parties before naming them; document laundering limits; robust mode on shortlists |
| A team of 2–3 people spread too thin | One main submission at a time (P1 → P3, the zoo → selection paper); the workshop paper reuses existing data; P6 cut; former P5 folded into P3; a two-person variant defined in section 1.1 |

## 15. Decisions

### 15.1 Taken (team, 2026-10-27)

- **Team size.** Plan for 2–3 people. The papers run in sequence (section 11),
  M0 is core-only, and P1 is lab-centric from the start.
- **Hosted service.** Not scoped. The deliverable is an open framework and a way
  to do this. A service, or ongoing ingestion, is decided after G2.
- **Authorization.** Large runs need no sign-off; people launch them, not agents.
- **Framing.** The aim is the research frontier and genuinely useful open tools,
  not corporate tooling.

### 15.2 Bulk access at scale *(call; measured 2026-10-28)*

The recipe below works the same on a personal or a research account. The only
differences are who holds the licenses and whose quota is used.

**What the problem actually is (measured on the HF API, 2026-10-28).**

| Top text-generation repositories by downloads | Ungated | Gated, automatic approval | Gated, manual approval |
| --- | ---: | ---: | ---: |
| Ranks 1–1,000 | 966 (96.6%) | 4 | 30 |
| Ranks 1,001–3,000 | 1,954 (97.7%) | 27 | 19 |
| Ranks 3,001–10,000 | 6,944 (99.2%) | 33 | 23 |

- Across the top 10k, only **136 repositories (1.4%) are gated**, held by **43
  owners**.
- Google (33), Meta (31), and Cohere (11) account for 55% of them.
- Gated repositories carry 3.9% of downloads in the top 1,000.

So gated access is a short, one-time list, not a scaling problem. Rate limits
and bytes are what scale.

**Hub rate limits** (HF documentation, current main branch; 5-minute fixed
windows; for organizations, each member gets the full limit separately):

| Plan | Hub API | Resolvers (file downloads) |
| --- | ---: | ---: |
| Free user | 1,000 | 5,000 |
| PRO user | 2,500 | 12,000 |
| Academia Hub organization (per member) | 3,000 | 20,000 |
| Enterprise organization (per member) | 6,000 | 50,000 |

**The recipe.**

1. **One account that downloads.** Gated access is granted per user, and every
   token inherits its user's access. We use the owner's personal account
   (below).
   - If resolver limits bind, PRO costs a few dollars a month and raises them
     about 2.4×.
   - Do not create extra accounts to multiply quota. Team members using their
     own accounts for their own runs is fine, because organization limits are
     per member by design.
2. **Build the frame without API pressure.** The public daily dump
   `cfahlgren1/hub-stats` (`models.parquet`; last updated 2026-10-27) lists the
   Hub's models without per-model API calls. Call `model_info` only for selected
   candidates, to pin commits and read `gated`, the declared base, and the
   license. At 1,000 API calls per 5 minutes on the free tier, that is about
   288k calls a day, enough for the whole candidate pool in days.
3. **Classify access in T0:** ungated, gated-auto, or gated-manual.
   - **Ungated** (98–99%): read directly.
   - **Gated:** accept the terms once per owner or family in the web UI. About 40
     owners cover the top 10k; it is an afternoon's work. Some families share one
     acceptance across repositories.
   - **Gated-manual:** request access for anchors and lab bases only. Log
     pending requests.
   - **Never automate the acceptance flow**: that risks the terms of service.
   - **Unaccessed gated repositories** keep their T0 metadata and are reported
     as a measured coverage gap of about 1%.
4. **Download efficiently.**
   - Always authenticate.
   - Use `huggingface_hub` 1.2 or later, which waits out 429 responses using the
     `RateLimit` header.
   - Deduplicate by sha256 and Xet chunks before fetching.
   - Resolve each file once and range-read the redirected CDN URL. Only URLs
     containing `/resolve/` count against the resolver bucket.
     *(hypothesis: CDN range reads after a redirect are not counted; confirm
     with the `RateLimit` headers in the pilot.)*
5. **Why limits will not bind (worst case).** Even if every T1 range read
   counted as a resolve, at about 250 tensors per model:
   - a free account manages about 5.8k models a day;
   - PRO about 14k;
   - an Academia Hub member about 23k.

   That puts T1 for 100k models at under three weeks on PRO, and less if CDN
   reads are free. Header reads and T2 downloads cost about 5 resolves per
   model. **Bandwidth binds, not requests.**
6. **Ask HF only for what plans cannot give.** A short note to HF's research
   contacts covering:
   - what the census is;
   - one research account with raised limits;
   - Xet reconstruction metadata at scale (for the zero-byte tier);
   - their view on publishing derived statistics;
   - a channel for suggested `base_model` corrections.

   This is useful, but **no longer a gate**.

**Which account holds the licenses: the owner's personal account** (team
decision, 2026-10-28; no capable organization or academic account is available,
and the owner accepts being the license holder).
- Upgrade it to PRO if resolver limits bind in the pilot.
- Use fine-grained, read-only tokens, one per run environment, held only in
  environment variables or the cluster secret store, as the collector already
  requires. Personal use of the account shares the same quota, so keep heavy
  personal downloads off it during collection runs.
- Record the license version and acceptance date in T0.
- Gated-access requests were filed and handled by the team (2026-10-31).

**Publishing gated-family derivatives.**
- Publish scalar statistics and spectra by default.
- Hold back 64×64 sketches until someone has read that family's terms on derived
  statistics.
- Raw row blocks and cached parent blocks are never published.

### 15.3 Settled since the first draft

1. **Hugging Face access:** the recipe in section 15.2. An agreement is optional.
2. **Cash for evaluation.** This means money spent on evaluation beyond our own
   GPU-hours:
   - paid API calls to a strong model acting as judge;
   - paid human raters or annotators;
   - paid hosted benchmarks or agent sandboxes.

   **Budget: at most about $20 a day in Claude API calls** (team, 2026-10-28),
   roughly $600 a month. Everything else runs on our own GPUs. Spend it only
   where a strong model's judgment is the measurement, in this order:
   1. **Judge calibration for the selection paper.** The judge-scored task family is graded by an
      open judge model on our GPUs. Claude grades a stratified sample only, a
      few hundred to a couple of thousand responses per task, to measure the
      open judge's agreement. The paper reports that agreement. Claude never
      grades the full panel.
   2. **Lineage-label audit for P1 and P3.** Claude reads model cards for a
      sample of Hub models and extracts the declared base, relation, and
      training notes. Those extractions are compared with the Hub's
      `base_model` fields and our detected lineage to measure label noise. A
      smaller model suffices for this.
   3. **Small demos.** These cost almost nothing.

   Rules:
   - A hard daily cap enforced in code.
   - Batch processing and prompt caching wherever the API allows.
   - Spend is logged per experiment.
   - Convert the budget into call counts using current prices once each prompt
     is fixed.
   - Agentic tasks stay graded by running them; no paid judge is needed there.
   - **Settled (team, 2026-10-31):**
     - **LLM-as-judge and human judging are deferred to a later stage.** They
       are needed only for the selection paper's judge-scored task family. The
       first option to try is a local open model as the judge, run on our own
       GPUs.
     - The $20-a-day cap applies only if paid evaluation calls are used.
     - The coding budget for implementation agents is ample but not infinite,
       so use it sensibly without over-optimizing.
     - The card audit for lineage labels is text extraction, not judging. It
       may use a small Claude model or a local model.
3. **v0 release: deferred** (team, 2026-10-28). Research is refined first. The
   historical corpus is released only together with a claim that gives it
   substance (idea 5).

## 16. Calls made in this draft

- **The zoo and its query framework are the core product** (team, 2026-10-31).
  Lineage, `diff`, and `check` are query types. The papers are lineage, then
  the zoo release, then selection with `recommend`. A minimal queryable zoo
  exists from G0.
- Lineage and modification (P1) come first among the papers, because they
  organize the zoo and weights are uniquely informative there (section 5.2).
  Exposing wrong metadata is a side effect, not a goal.
- P1's headline is "pairwise lineage accuracy does not scale": false parents
  per query against candidate-pool size, with three nested estimands and three
  truth tiers.
- Papers run in sequence for 2–3 people:
  1. P1 (ICML 2027; fallback the benchmark framing, then COLM);
  2. the zero-byte note (February–March);
  3. P3 (NeurIPS 2027 Evaluations & Datasets);
  4. the selection paper, `recommend` evaluated end to end (ICLR 2028).

  The workshop paper reuses existing data. P6 is cut and survives as a tool
  feature. The refusal-ablation audit (former P5) is folded into P3.
- G0 is on 2026-12-04 with measurable thresholds (section 12). The P1 engine
  starts on lab models in week 2, so it never waits on G0.
- The minimum lab is about 120 models on permissive bases. GRPO and
  full-parameter 7B training are cut. Hard negatives come from PolyPythias.
- T3-lite in P3 is capped at 2k models. Direct measurement still beats
  prediction for static skills (section 5.1).
- Row-block T1 is the default census tier, with adaptive escalation to T2. T2
  passes emit T1 blocks at several heights for free.
- Sketch operators are orthonormal (QR), nested, and side-specific. Sizes are 64
  for the census and 128 for the lab and anchors, up to 256. Stored-sketch rank
  is trusted only to about k/4.
- Every model type goes through one decode × geometry table (section 6.6), with
  never-a-matrix suffixes, module poisoning, vector-like skips, file-set
  selection, all LoRA families, an expert subset, and orientation from config.
- The simplest design that works is preferred everywhere. Anything more advanced
  must first be shown necessary by an experiment.
- Block deltas abstain below the bf16 floor and use biwhitened rank with a gap
  confidence. Parent blocks stay internal.
- Inclusion probabilities are recorded. A probability sample of 2–5k from the
  ≤75B frame goes to T2, with a phase-2 subsample to measure detector error.
  Models above 75B are enumerated as a separate stratum. `.bin`-only
  repositories are T0-only.
- One self-contained `evidence` package with a CPU streaming path, and a GPU
  "download once" path only where T3 runs. If CPU nodes lack outbound HTTPS,
  readers run on transfer nodes or behind a proxy.
- FP8, MXFP4, and NVFP4 are decoded exactly. GGUF arrives in zoo v1 for
  lineage only. GPTQ, AWQ, and bitsandbytes stay parked. The lab's
  quantizations are simulated for P1.
- Fingerprints are `.safetensors` files. The `fingerprint`/`diff`/`check` CLI
  is internal from G0 and public in zoo v1. `diff` reports measured facts,
  never predicted skills.
- The genealogy lab defaults to permissively licensed bases. Gated bases are
  added once access is granted on the owner's account.
- Extraction proceeds; publication of gated-family sketches waits for a read of
  that family's terms.
- The deliverable is an open framework, not a hosted service.
- Bulk access follows the recipe in section 15.2:
  - one authenticated account (the owner's), upgraded to PRO if limits bind;
  - the frame comes from the hub-stats dump;
  - gated terms are accepted once per owner, by hand;
  - each file is resolved once and then range-read.

  An HF agreement is optional.
- Gated licenses are held on the owner's personal account.
- Expensive evaluation is self-hosted (open judges, execution-graded agentic
  tasks). About $20 a day of Claude calls goes to judge calibration on samples
  and a lineage-label audit, under a hard cap.
- Exact search is used until a measurement says otherwise.
- The historical 12.5k corpus stays frozen as a baseline and is re-measured
  inside the census. It is released only together with idea 5's reliability
  study, not as a standalone dump.
- Heritability uses a REML mixed model and parent–offspring regression with
  binomial measurement error. Merges are handled by network Brownian motion.
- One token, the owner's `HF_TOKEN` (revised 2026-11-01).
- The item bank, the exact dense-panel family, and code placement beyond the
  extractor are left to the builder.

## 17. References added or corrected

The full reference lists in Plans A and B stand, except for the corrections in
section 3.2. Added:

- Provable Model Provenance Set for LLMs: arXiv 2602.00772.
- Model-intrinsic spectral fingerprinting for LLMs (IEEE WIFS 2026): arXiv
  2610.30626.
- Imputation of pretrained language model metadata (317k models): arXiv
  2606.21787.
- TokenPrint: arXiv 2608.08139.
- BenchPress, "You Don't Need to Run Every Eval": arXiv 2606.24020.
- W2T, "LoRA Weights Already Know What They Can Do": arXiv 2603.15990.
- ModelLakeFishing: arXiv 2610.04904.
- WeightCLIP (ICML 2026): arXiv 2607.03551.
- SELECT-LLM: arXiv 2510.09418.
- Why Heavy-Tailed Weight Spectra Predict Model Quality (NeurIPS 2026): arXiv
  2610.13123.
- Witness Overlap (NeurIPS 2026): arXiv 2609.31784.
- Atteson, "The performance of neighbor-joining methods of phylogenetic
  reconstruction", Algorithmica, 1999.
- Pagel, "Inferring the historical patterns of biological evolution", Nature,
  1999. Blomberg, Garland and Ives, Evolution, 2003. (Considered and replaced;
  see idea 2.)
- Horvitz and Thompson, JASA, 1952.

Added 2026-10-29, from the review:

- Model Provenance Testing (black-box), Nikolić et al., NeurIPS 2025: arXiv
  2502.00706.
- MoTHer, model tree heritage recovery, ICLR 2025: arXiv 2405.18432.
- PhyloLM, ICLR 2025; Tamura et al., COLM 2025 (lineage-aware score prediction,
  2,934 models).
- Morita, Thall and Müller, "Determining the effective sample size of a
  parametric prior", Biometrics, 2008.
- Gavish and Donoho, "The optimal hard threshold for singular values is 4/√3",
  IEEE Transactions on Information Theory, 2014.
- Bastide et al., "Phylogenetic comparative methods on phylogenetic networks
  with reticulations", Systematic Biology, 2018.
- Rogan and Gladen, "Estimating prevalence from the results of a screening
  test", American Journal of Epidemiology, 1978.
- Chu–Liu/Edmonds minimum-cost arborescence (Chu and Liu, 1965; Edmonds, 1967).
- PolyPythias (multiple-seed Pythia suite), EleutherAI.
