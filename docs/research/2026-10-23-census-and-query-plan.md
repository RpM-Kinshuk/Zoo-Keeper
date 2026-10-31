# Model Census and Zoo Query: scale-up research plan

**Status:** draft for team review, 2026-10-23. Built from the current HTSR study,
the Model-Zoo pipeline, the reference paper, and a literature and web survey run
the same day. Names are placeholders. *(call)* marks a decision I made that the
team should check. *(est.)* marks a back-of-envelope number to calibrate in the
pilot. Survey numbers carry their source; anything a survey agent measured on
one or two models is labeled as such.

## 1. In one paragraph

Build the first weight-derived, revision-pinned census of the open model
ecosystem. That means about 100k public checkpoints, each with full eigenvalue
spectra, small linear weight sketches, tensor hashes, and verified lineage. On
top of it, build a query engine that answers "which model should I use for X
under constraints Y?". Each answer carries predicted capability with calibrated
uncertainty, lineage evidence, the effective license, and risk flags. Weights
give a cheap prior for every model; a handful of targeted evaluations confirm the
shortlist. The census, the lineage method, and the search method are each
publishable. Together they fill the gap left by two things: the retired Open LLM
Leaderboard and self-reported model cards.

## 2. Starting point

### What we have

- About 12.5k analyzed models (8,892 Phase 2, 3,661 ablation) with per-matrix
  scalar statistics.
- Family grouping: 10,946 models in 17 groups, with adjusted Rand index 0.924
  against 2,981 broad-family labels.
- Supplied-base matching: 82–85% first choice and about 99% top three, among 29
  bases.
- Score prediction: MAE about 3.1–3.3 with near-match groups kept together, but
  7–9.5 with a whole cluster held out.
- A GPU orchestrator (Model-Zoo) with supervision, format-aware loading, and
  resume.
- An evaluation culture of frozen protocols, input hashes, near-duplicate
  grouping, transfer tests, and audits. Keep it. Reviewers will not find this
  elsewhere, and the field's weakest papers lack exactly this.

### What must change before scaling

1. **Revision identity.** Record for every analysis:
   - the requested and effective repository;
   - the resolved commit;
   - the file sha256 and Xet hash;
   - the adapter base commit.

   Without these, no score join can be verified. This is the reassessment's main
   open limitation.
2. **Full spectra.** Only 17 of 12,553 models have saved eigenvalues. Save
   unfiltered eigenvalues for every matrix. Every alpha estimator (Hill at n/2,
   DKS, Clauset–Shalizi–Newman (CSN), FARMS-corrected) then becomes an offline
   computation. For small eigenvalues, use `svdvals` or an fp64 Gram matrix: on
   one ill-conditioned `q_proj`, a survey test found the fp32 Gram matrix off by
   about 120× at the bottom of the spectrum.

   The current 1e-5 eigenvalue filter rarely matters: in a 400-model sample,
   0.7% of 95,866 matrices lost more than 10% of their eigenvalues. Still,
   unfiltered spectra remove the question.
3. **Some coordinate information.** Spectra are rotation-invariant, so they
   cannot do two things:
   - separate sibling fine-tunes of one base (Chen et al., COLM 2026);
   - recover merges, because spectra are nonlinear in the weights.

   Add small linear sketches (section 5.2).
4. **Stream, do not load.** Read each tensor from safetensors over HTTP range
   requests. That removes full model loads, local disk, and per-model GPU-memory
   limits. A survey agent tested this live against the Hugging Face (HF) CDN.
5. **Fix curation data.** 8,487 of 15,000 rows in
   `Model-Zoo/data/curated/model_zoo_phase2.csv` have a one-character
   `Architecture` ('u', 'L', 'M', 'Q', and so on), which looks like the first
   character of a string. This is harmless for weight-only analysis but wrong for
   any metadata baseline.
6. **Retarget scores.** The Open LLM Leaderboard retired on 2025-03-13. Its v1
   and v2 averages are frozen, and reviewers will ask why they matter. Predict
   per-skill abilities and evaluate decisions instead (section 5.5).

## 3. Where the field is (October 2026)

| Work | What it does | Scale | What it lacks for our goal |
| --- | --- | --- | --- |
| Model Atlas (Horwitz et al. 2025; NeurIPS 2025 position track) | Metadata graph. DAG recovery from about 100 sampled neurons plus time priors. | About 400k documented edges; 63k-model demo | No capability or calibration; mostly metadata |
| Anatomy of an ML ecosystem (Laufer et al. 2025) | Family trees and license drift across 1.86M models | Metadata only | No weights |
| Spectral Signatures (Zhang … Yang, KDD 2026) | Alpha-Hill depth profiles; lineage, clustering, kNN performance | 499 models; random splits | Scale, family hold-out, siblings, calibration |
| Who Built This Model? (Chen et al., COLM 2026) | Layer-norm "Trace" plus top-256 singular-subspace alignment | About 110 pairs | Scale; per-pair cost |
| AWM, HuRef, centered residuals, Independence Tests | Pairwise verification with strong invariance or exact p-values | 10–200 pairs | O(pairs) with full weights; not a search index |
| modelDNA (Adil et al. 2026) | Ranged reads (100–300 MB). Calibrated verdicts with abstention. Merge weights by least squares. | 15 models; 55-model graph | Scale and evaluation |
| Cisco Model Provenance Kit (Apr 2026) | Fingerprints from weights, tokenizer, and metadata against about 150 bases | Product | Open benchmark; capability |
| ModelLens (Cai et al. 2026) | Latent space from 1.62M leaderboard records | 47k models | Needs evaluation records (cold start) |
| HugSelect (2026) | Explainable multi-criteria selection | 71k-model knowledge base | Metadata only |
| Discovering Hidden Gems (Kahana et al. 2026) | Bandit model search. Rarely downloaded fine-tunes often beat popular ones. | About 2k evaluated | No prior; about 50 queries per candidate |
| EmbedLLM, LOCUS, UniRoute, LLM DNA | Model embeddings from responses | 100–300 models | Inference needed on every new model |
| metabench, tinyBenchmarks, ATLAS, Fluid Benchmarking | IRT and adaptive testing: tens to hundreds of items suffice | 5k+ models (leaderboard v1) | No weight-side prior |

**What nobody has published, as far as the survey found:**

1. **A public weight-feature dataset for 10k+ real LLM checkpoints**, joined to
   lineage and evaluations. The largest real-weight LLM study has about 500
   models. Weight zoos with 100k+ models contain only tiny lab-trained networks.
2. **Weight-based lineage evaluated beyond about 10^2–10^3 models**, with held-out
   families, merges, quantization, and calibrated open-set answers. No conformal
   provenance work was found.
3. **Capability prediction for never-evaluated models.** Every router and
   recommender needs responses or evaluation records for each new model.
4. **One query system** that combines lineage, capability, risk, and deployment
   constraints.

## 4. Goals, non-goals, anti-goals

### Goals

- **G1 Census.** About 100k revision-pinned checkpoints with full spectra,
  sketches, hashes, and metadata. About 10k of them also get item-level
  evaluations. Plus a controlled genealogy set with exact ground truth.
- **G2 Lineage engine.** Parent, edge type, direction, and a calibrated "no
  parent in the database" answer at Hub scale, with evidence a user can inspect.
- **G3 Capability engine.** Per-skill predictions with calibrated intervals for
  every census model, and a weight prior that measurably cuts evaluation cost.
- **G4 Query engine.** A natural-language or structured request returns ranked,
  deduplicated candidates. Each comes with predicted capability, lineage,
  effective license, risk flags, and optional on-demand verification.
- **G5 Living resource.** An open dataset, library, API, MCP server, and HF
  Space, with daily ingestion of new popular models.
- **G6 Papers** (section 9).

### Non-goals

- Detecting distillation into freshly initialized students, or cross-architecture
  lineage. Weights cannot show either; state this scope explicitly.
- Ranking closed frontier models, which have no public weights.
- Replacing serialization scanners (Protect AI, JFrog, ClamAV). Link to their
  results instead.
- A new general-purpose benchmark.

### Anti-goals: failures even if the system "works"

- Confident wrong answers: any lineage or capability claim without calibrated
  uncertainty and inspectable evidence.
- Leakage: near-duplicates, re-uploads, or merges of test models inside training
  splits.
- A dataset dump without claims. Evaluation and dataset reviewers now explicitly
  reject "datasets as endpoints".
- Publishing reconstructive weight information for restrictively licensed models.
- Naming organizations or people in a launch audit without verification and prior
  notice.

## 5. System design

```text
Hugging Face Hub (~3M repos)
  -> T0 manifest for all candidates -> selection (~100k)
  -> streaming extractor (T1 signatures, T2 spectra)
  -> census store (Parquet + DuckDB, FAISS)
  -> engines: lineage | capability | risk
  -> Zoo Query (library, REST, MCP, HF Space, CLI)

Also feeding the census store:
  - T3 evaluation campaign (~10k models)
  - genealogy lab (exact ground truth)
```

### 5.1 Which 100k models *(call)*

**Candidate pool.** Text-generation and image-text-to-text repositories with
safetensors or GGUF files. Among the top 100k text-generation repositories:

- 60% have safetensors;
- 28% are GGUF-only;
- 12% have neither (survey crawl of the HF API).

**Strata, filled in this order:**

1. **Anchors (about 3–5k).** Every declared base, official organization releases,
   and every ancestor of anything selected.
2. **Head (about 30–40k).** The most-downloaded repositories. 1.5% of
   repositories account for 99.2% of downloads (HF Summer 2026 report), so this
   covers almost all real use.
3. **Labeled.** Every model with public evaluation records: about 13k on the Open
   LLM Leaderboard and about 22k in Every Eval Ever. These overlap heavily with
   the head.
4. **Long tail (about 40k).** Stratified by family, relation (finetune, adapter,
   merge, quantized, unknown), size, and upload quarter. Oversample repositories
   with undeclared lineage.
5. **Lab (about 1–2k).** The controlled genealogy set (section 5.3) plus hard
   negatives.
6. **Prospective set.** Models uploaded after a freeze date. Process them, freeze
   predictions, and score them later as a pre-registered test of the future.

**Other formats.** GGUF-only repositories get T0 for all of them, and T1 from
dequantized sampled tensors (via `gguf-py`) inside strata 2 and 4. Pickle-only
repositories come last and load with `weights_only=True`.

### 5.2 Fingerprint tiers

| Tier | Contents | Unlocks | Cost per model *(est.)* |
| --- | --- | --- | --- |
| **T0 manifest** | `model_info` (commit sha, parameters by dtype, declared bases and relation, license, gated flag, card); file sha256 and Xet hash; safetensors or GGUF headers (names, shapes, dtypes, quantization config) | Architecture signature; quantization method; exact duplicates (sha256); re-shards (Xet chunk overlap); declared graph | About 3 API calls plus 100 KB per shard |
| **T1 signature** | Per-tensor content hash; two-sided linear sketch per tensor (see below); CountSketch-256 plus SimHash per tensor; spectra for a fixed depth subset | Exactly shared tensors (frozen embeddings, passthrough merges); fine-tune neighbors; merge recipes; adapter and abliteration rank tests | All bytes of each sketched tensor |
| **T2 full spectra** | Unfiltered eigenvalues for every matrix; for anchors, spectra of invariant products | Every alpha estimator; full-spectrum distances; laundering-robust comparison | About 10–60 GPU-seconds per 7B model, if streamed |
| **T3 behavior** | About 300–600 IRT-selected items across skills, run while the weights are local; a small refusal probe for flagged models | Capability labels; calibration of the weight prior | About 0.1–0.2 H100-hours per model of 8B or less |

More detail on three of these:

- **The two-sided sketch** is S_L W S_Rᵀ, 32×32, with shared seeded sketch
  matrices per dimension so sketches are comparable across models.
- **CountSketch** discriminates in the survey's test on two matrices: cosine
  0.99986 between base and Instruct, against about ±0.08 for a row-permuted
  control.
- **The invariant products** are W_Q W_Kᵀ and W_V W_O per head, and W_down W_up.

Sketching needs every byte of a tensor. That forces a choice between two options
*(decision for the team)*:

- **Full option:** stream every tensor of every model, giving T1 and T2 for all
  of them.
- **Tiered option:** for most models, stream a depth subset (for example 8 layers
  plus embeddings). Stream everything for anchors, labeled models, and the lab.

**Why linear sketches.** A sketch is a linear map of the weights, and that buys
four things:

- **Merges.** If child = Σ c_i parent_i, then sketch(child) = Σ c_i
  sketch(parent_i). The recipe follows from non-negative least squares on a few
  hundred thousand numbers. modelDNA showed least-squares merge recovery on full
  tensors for 15 models; our addition is doing it in sketch space at Hub scale.
- **Adapters.** If child = parent + BA, then rank(S_L ΔW S_Rᵀ) ≤ r. That detects
  an adapter, and estimates its rank, without the adapter file.
- **Abliteration.** Orthogonalizing residual-writing matrices against a refusal
  direction r gives ΔW = −r(rᵀW): a rank-1 change per matrix, with left factor
  S_L r shared across layers. Refusal is not always a single direction
  (arXiv 2502.17420, 2602.02132), so test for low rank, not exactly rank 1.
- **Unrelated same-architecture models.** Sketch cosine should concentrate near
  zero, with variance about 1/k, which gives a closed-form null. Real weights
  share initialization statistics and massive-activation dimensions, so check
  this against hard negatives in the pilot.

**Limits of sketches:**

- They are not reconstructive: 1,024 numbers summarize a 16M-entry matrix. Legal
  review is still needed.
- They are not invariant to permutation or rotation. That is why spectra and
  invariant-product spectra stay in the design for adversarial cases.

**Storage for 100k models *(est.)*:**

| Item | Size |
| --- | --- |
| Full spectra, compressed | About 0.2 TB |
| Two-sided sketches | About 45 GB |
| CountSketches | About 15 GB |
| Headers | About 4 GB |

### 5.3 Genealogy lab: exact ground truth

**Why it is needed:**

- Hub `base_model` fields are self-reported.
- The Model Atlas reports more than 60% of the ecosystem as undocumented.
- Reviewers expect exact truth plus hard negatives.

**Contents *(call)*.** Start from about 10–15 bases between 0.5B and 8B
parameters: Qwen2.5 and Qwen3, Llama-3.2, Gemma-2 and Gemma-3, SmolLM, and OLMo-2.
OLMo and Pythia publish intermediate checkpoints, which give pretraining chains.
From those bases, create:

- full SFT; LoRA SFT at ranks 8, 64, and 256; DPO; short GRPO runs;
- chains of 2–4 steps (A→B→C) for direction and additivity tests;
- merges with known weights via mergekit: linear, SLERP, TIES, DARE, passthrough;
- quantizations: GPTQ, AWQ, GGUF Q4_K_M and Q8_0, bitsandbytes NF4, FP8;
- abliteration (the classic single-direction method and Heretic), unstructured
  and structured pruning, depth up-scaling, continued pretraining;
- laundering stress tests: neuron permutation, per-neuron reciprocal rescaling,
  orthogonal residual rotation;
- hard negatives: independently trained same-architecture models (OpenLLaMA
  against Llama, Amber, and those used by Thakur and Khoury), and same-series
  siblings.

**Cost *(est.)*:** about 300–800 GPU-hours, almost all from SFT, DPO, and GRPO.
Merges, quantization, and abliteration are nearly free.

### 5.4 Lineage engine

A cascade from cheap to expensive; each stage runs only on the survivors of the
one before.

1. **Exact.** File sha256, Xet chunk overlap, and per-tensor hashes find
   re-uploads, re-shards, untouched tensors, and passthrough layer maps. The
   survey found 87.3% of Xet chunks shared between a re-sharded copy of
   Qwen2.5-7B-Instruct and the original. Base and Instruct, by contrast, shared
   0%.
2. **Retrieve.** Approximate nearest-neighbor search over invariant
   full-spectrum embeddings and SimHash returns the top candidate relatives
   across the whole census.
3. **Verify.** Sketch cosine against a closed-form or empirical null gives a
   p-value for "related". For anchors that have singular vectors computed, a
   principal-angle subspace test (Chen et al.) separates siblings.
4. **Type the edge.** Each edge type has its own signature:
   - finetune: a broad-rank change;
   - adapter: a low-rank change, with an estimated rank;
   - merge: a non-negative least-squares recipe and its residual;
   - quantized: dtype and structured noise, plus the leaf prior;
   - abliterated: a low-rank change in residual-writing matrices;
   - pruned: a sparsity pattern;
   - up-scaled: a layer-duplication map from assignment or dynamic time warping.
5. **Assemble the DAG.** Build a constrained minimum-cost arborescence
   (Chu–Liu/Edmonds) over verified edges, using two priors:
   - the parent is uploaded first, which holds for 99.73% of documented edges;
   - quantized models are leaves, which holds for 99.41% (both from the Model
     Atlas).

   Merges get multiple parents. For pairs uploaded at the same time, use
   witness-overlap tests (Li et al., NeurIPS 2026) and the additivity test in
   section 5.4.1.
6. **Calibrate "parent not in the database".** Use conformal or null-calibrated
   tests on leave-one-base-out folds, with Holm or Benjamini–Hochberg correction
   across candidates. Report the false-positive rate at a fixed true-positive
   rate, coverage, and abstention.

   Today's best-versus-second margin separates available from unavailable roots
   with AUROC 0.442. The similarity score itself does better, but only 0.676.
7. **Robust mode,** for adversarial claims and IP disputes. Run invariant-product
   spectra and AWM-style pairwise verification on the final 1–3 candidates only.

#### 5.4.1 Hypothesis worth testing: lineage distances form a tree metric

Fine-tuning steps change millions of weights independently. If successive steps
are nearly orthogonal in high dimension, squared distances add along a path:
‖Δ_AC‖² ≈ ‖Δ_AB‖² + ‖Δ_BC‖². If that holds, classical phylogenetics recovers the
topology with known guarantees (the four-point condition, neighbor joining), and
it identifies the middle model of a chain. Test this on lab chains first. It can
fail when steps share directions, for example after similar instruction data.

### 5.5 Capability engine

**Target.** Per-skill abilities rather than retired leaderboard averages: math,
code, instruction following, knowledge, reasoning, multilingual, long context,
and refusal behavior. Keep the v1 and v2 averages only for continuity with the
existing study.

**Model *(call)*:** two-sided amortized item response theory (IRT) with a
weight-and-lineage prior.

```text
P(model m answers item i correctly) = sigmoid(a_i . theta_m - b_i)
theta_m ~ Normal(mu_m, Sigma)
  roots:       mu_m = g(weight features of m)
  derivatives: mu_m = theta_parent + h(change in weight features)
  merges:      recipe-weighted parents
(a_i, b_i): predicted from item text (Truong et al., ICML 2025), refined by responses
```

**Data sources:**

- T0–T2 features for every model.
- Item-level responses where they exist: leaderboard v1 details for about 7k
  models, v2 details for about 4.5k (gated datasets), and Every Eval Ever.
- Our own T3 campaign.

**Why lineage.** Most Hub models descend from a few bases (Qwen alone has more
than 150k declared derivatives). Predicting a model's change from a known parent
is easier than predicting from scratch. The current whole-cluster MAE of 7–9.5 is
exactly the failure this targets.

**Uncertainty.** Hierarchical or Mondrian conformal prediction, with lineage
clusters as groups. Weighted conformal for shift to new families. When the
interval is too wide, abstain and schedule an evaluation.

**Choosing which models to evaluate.** Pick the census models that get T3
evaluation by their expected reduction in decision regret over likely queries:
probabilistic matrix factorization with an acquisition function, in the style of
Fusi et al. Do not sample uniformly.

**Headline measure: "a weight prior is worth k items".** k is the number of
adaptively chosen items an evaluation needs without the prior to match the
prior-informed error. Report it in-family and new-family separately.

**Required baselines:**

- training median;
- Spectral Signatures kNN;
- our current non-alpha spectral trees;
- metadata-graph kNN;
- Sloth and observational scaling from parameters and tokens;
- lineage-regularized matrix factorization;
- random sample plus regression (Zhang et al., NeurIPS 2025);
- tinyBenchmarks and ATLAS at matched item budgets;
- EmbedLLM and LOCUS given a few responses;
- MatrixFormer zero-shot.

**Controls:**

- residualize on log-parameters and architecture;
- a matrix-shape-only control;
- the benchmark noise ceiling.

### 5.6 Risk engine

Every flag carries an evidence level: verified, likely, or declared only.

- **Declared versus detected base.** Catches undeclared derivatives and
  contradicted "from scratch" claims.
- **Effective license.** The most restrictive terms along the detected ancestry.
  Flag downstream licenses that are more permissive than upstream ones (Laufer et
  al. document this drift), plus naming-clause problems.
- **Safety-modified.** The abliteration signature from section 5.2, plus a
  50–100-prompt refusal probe for flagged or high-download derivatives.
  Uncensoring by fine-tuning has no rank signature, so it needs the probe.

  Prior art is close: "Auditing Abliteration" (2026) treats abliteration as
  tampering, and the abliteration.org wiki discusses reverse-engineering it from
  weights. Our addition is sketch-based, calibrated detection plus a prevalence
  measurement at Hub scale. About 8.7k repositories contain "abliterated" in
  their name (survey count, 2026-10-23).
- **Re-upload or spam.** Exact duplicates and re-shards; credit the earliest
  upload.
- **Serialization.** Link to the Hub's scanner results rather than rebuilding
  them.

### 5.7 Query engine (Zoo Query)

**How a request is answered:**

1. **Parse** the request into a schema (an LLM does this) and show the parsed
   constraints back for confirmation.
   - Hard constraints: license class over the effective license; memory given
     format and quantization; context length; languages; runtime support;
     excluding safety-modified lineage.
   - Soft goal: a task description and/or 5–50 user examples with a scoring
     rule.
   - Risk attitude.
2. **Filter** with DuckDB over the census Parquet files.
3. **Score.** Map the description or examples to a task vector, i.e. weights
   over skills in item space. Combine it with each model's predicted abilities
   and their uncertainty. Collapse near-duplicate clusters to one representative.
4. **Rank** by expected utility, or by a lower or upper confidence bound,
   depending on the stated risk attitude.
5. **Verify (optional).** Run best-arm identification (successive halving or
   LUCB) over the top k on the user's examples. The prior means fewer queries;
   Hidden Gems needs about 50 per candidate without one.
6. **Answer** with one card per model:
   - predicted versus measured skills, with intervals;
   - lineage (declared versus detected, with evidence tier);
   - effective license and risk flags;
   - deployment facts and near-duplicates.

**Interfaces:**

- a Python library;
- a REST API;
- an MCP server (coding agents are now the Hub's largest client; HF Summer 2026
  report);
- an HF Space;
- a CLI, for example "is repo X derived from Y?", answered with a p-value.

**ModelSearchBench, to evaluate it *(call)*.**

- **Requests:** 200–500 realistic ones, drawn from user personas, public forum
  questions, and internal Booking.com use cases. Each has constraints and a
  held-out task set.
- **Ground truth:** evaluate every constraint-satisfying candidate in a large
  pool.
- **Metrics:** regret@1, probability that the best model is in the top five,
  NDCG, and GPU-minutes per answer.
- **Baselines:**
  - downloads, likes, or trending rank;
  - HF search;
  - an LLM agent reading model cards;
  - metadata kNN;
  - an evaluation-only latent space in the style of ModelLens;
  - bandit search without a prior.

## 6. Ideas, from simple to advanced

Each idea ends in a claim we can test.

1. **Hashes and headers.** How much of the Hub is exact duplicates or re-shards?
   A free and striking ecosystem statistic, available from T0 alone.
2. **Robust tail statistics.**
   - Main signature: Hill at k = n/2 with closed-form standard error
     (α − 1)/√k. A survey test found that this matches the bootstrap.
   - Report CSN only together with the KS distance D, tail size, and a bootstrap.
   - Add the FARMS correction for aspect ratio.
   - The survey found Hill(n/2) nearly unchanged between Qwen2.5-7B base and
     Instruct on two matrices; test this at scale.
3. **Full-spectrum distances.**
   - Use Wasserstein-1 between per-role log-eigenvalue distributions; it has a
     closed form via quantile functions.
   - Align depth with dynamic time warping instead of interpolation. That handles
     up-scaled and frankenmerged models with repeated layers.
4. **A learned encoder over spectral tokens.** Tokens are (role, depth,
   statistics), and the encoder is trained contrastively on verified lineage.
   This is a candidate for the first whole-LLM embedding from weights at Hub
   scale: the survey found no published method for 7B+ models.
5. **Linear sketch algebra:** merges, adapter rank, and abliteration
   (section 5.2).
6. **The tree-metric hypothesis** and phylogenetic reconstruction
   (section 5.4.1).
7. **Calibrated provenance:** conformal and null-calibrated open-set answers
   (section 5.4).
8. **Phylogenetic comparative methods.**
   - Measure how much capability variance is inherited, using Pagel's λ or
     Blomberg's K on the lineage tree.
   - Use Brownian-motion or Ornstein–Uhlenbeck priors along edges.
   - PhyloLM builds phylogenies from outputs. We did not find these
     comparative statistics applied to capabilities.
9. **Two-sided amortized IRT with a lineage prior** (section 5.5).
10. **Value of information.** Choose which models to evaluate in the campaign by
    expected value. At query time, use prior-informed best-arm identification.

## 7. Evaluation rules: keep the current ones and extend them

- **Revisions.** Pin every revision. Join scores only on matching commits, and
  report what fraction of joins verify.
- **Near-duplicates.** Group them (by sketch similarity, exact tensors, and
  re-uploads) before any split.
- **Four test designs:** near-duplicate groups kept together, whole family held
  out, collection transfer, and prospective (models uploaded after the freeze).
- **Lineage tests:**
  - held-out bases;
  - hard negatives: independent same-architecture models and same-series
    siblings;
  - laundering stress tests;
  - false-positive rate at a fixed true-positive rate, and abstention coverage.
- **Capability tests:** per-skill and decision metrics, against the noise
  ceiling and cost-matched item baselines. Keep out-of-range predictions visible.
- **Frozen protocols.** Freeze each paper's protocol in a config before its final
  run, as `configs/htsr_combined_study_v4.json` does today.
- **Label audits.** Audit a manual sample of Hub lineage labels, and report the
  label noise found.

## 8. Practical deliverables and who uses them

### Deliverables

- **An HF dataset:**
  - Parquet shards of about 1k models each;
  - Croissant and Responsible AI metadata;
  - a license column;
  - a takedown process;
  - versioned snapshots.
- **A fingerprinting library.** It fingerprints one model by range reads on a
  laptop or CPU node. The same code powers the census.
- **Zoo Query** as an API, an MCP server, and an HF Space.
  - Exports to CycloneDX ML-BOM and SPDX AI.
  - Suggested `base_model` corrections offered to HF; staff currently add these
    by hand.
- **Daily ingestion** of new repositories above a download threshold. The Hub
  adds about 500 text-generation repositories a day (survey count).
- **Internal use at Booking.com:**
  - governance for models pulled by teams (lineage, license chain, risk);
  - selection for travel use cases;
  - the same engine run on private checkpoints.

### Who uses it

- **Practitioners:** choose a model with predicted skills and its real license
  chain.
- **Open-weight labs:**
  - monitor derivatives for undeclared use, license violations, and
    safety-stripped variants;
  - choose starting checkpoints for post-training.
- **Security, governance, and compliance teams:**
  - lineage evidence for ML-BOMs;
  - questions about undeclared derivatives, such as those raised by the October
    2026 DHS order on DeepSeek-derived products;
  - EU AI Act downstream documentation.
- **Safety institutes:** ranked lists of high-download, safety-modified
  derivatives for red-teaming.
- **Researchers:** the largest real-weight dataset for weight-space learning and
  HTSR.

## 9. Papers and venues

ICML 2027 and NeurIPS 2027 dates were not yet published when checked; the dates
below follow recent years. VLDB 2027 takes monthly submissions until
2027-03-01 (checked).

| # | Working claim | Core evidence | Target |
| --- | --- | --- | --- |
| A | Lineage at Hub scale with calibrated evidence | Cascade, sketch algebra, tree-metric test, conformal open-set answers. Evaluated on lab truth plus 25–50k Hub models. Compared with Spectral Signatures, Trace, AWM on shortlists, and Atlas-style recovery on parent accuracy, false-positive rate, and cost. | ICML 2027 (deadline usually late January) |
| B | A census of the open-weight ecosystem | 100k revision-pinned models with spectra, sketches, verified lineage, and about 10k item-level evaluations. Benchmark tasks and protocols. Headline audit findings: undeclared derivatives, license drift, prevalence of safety-modified models, duplicates. | NeurIPS 2027 Evaluations & Datasets track (usually May) |
| C | Weight priors for model search | Two-sided IRT with a lineage prior, conformal intervals, "worth k items", and regret on ModelSearchBench | NeurIPS 2027 main track or ICLR 2028 |
| D | How post-training reshapes spectra | Causal attribution from the lab, prevalence from the Hub. Within-lineage change in spectrum versus change in capability. Phylogenetic signal. | ICML or NeurIPS 2027, or JMLR |
| E | Safety-modified derivatives at Hub scale | Abliteration signature, refusal probe, prevalence, responsible disclosure | ICLR 2027 workshops (about February), then main track |

Optionally, Zoo Query could also go to VLDB as a "model lake" system paper.

## 10. Budget for 100k models *(est.; calibrate in the pilot)*

| Item | Full option | Tiered option |
| --- | --- | --- |
| Download | About 1.73 PB, or about 1.4–1.5 PB after deduplication. That is 6.4 days at a sustained 25 Gbps or 16 days at 10 Gbps; plan for 1.5–2× with failures. | About 0.4–0.7 PB |
| Spectra (GPU) | 0.4–1.8k GPU-hours by roofline estimate. Up to about 15k at the reference paper's measured 337 s per 7B model. Calibrate against our own 12.5k-model throughput. | Lower |
| Evaluation campaign (10k models) | About 1–2k H100-hours (about $3–6k at cloud rates), run while the weights are already downloaded | Same |
| Genealogy lab | 300–800 GPU-hours | Same |
| Storage | About 0.2 TB of spectra, 45 GB of two-sided sketches, 15 GB of CountSketches, 4 GB of headers | Lower |
| HF API | About 300k API calls and 400k resolves, about 25 hours of free-tier quota | Lower |

Network bandwidth, not GPU time, is the binding constraint. The byte total comes
from parameter counts of the first 100k safetensors text-generation models of
75B parameters or fewer (survey crawl). By bytes, those models break down as:

| Size | Share of models | Share of bytes |
| --- | --- | --- |
| 3–9B | 38% | 31% |
| 15–35B | 8% | 27% |
| 35–75B | 3.6% | 25% |

## 11. Timeline *(call; staffing to be sized)*

### Phase 0: foundations (November 2026, 3–4 weeks)

- **Manifest and data fixes.** Crawl the T0 manifest for all candidates, write
  the selection policy, and fix the `Architecture` bug.
- **Streaming extractor.**
  - Revision pinning, full spectra, sketches, and hashes.
  - Format adapters: safetensors first, then GGUF, GPTQ, AWQ, NF4, FP8, and LoRA.
  - LoRA uses the r×r shortcut: the singular values of R_B R_Aᵀ.
- **500-model pilot.** Measure throughput, bandwidth, and failure rate, then
  choose the full or tiered option.
- **Legal review.** Publishing derived statistics, including for gated models,
  and whether sketches are non-reconstructive.
- **Outreach.** Start the genealogy lab. Contact HF about bulk access and
  collaboration.

**Exit:** pilot numbers, frozen schema v2, and a legal answer.

### Phase 1: lineage (December 2026 – January 2027)

- Process 25–50k census models plus the lab.
- Build the lineage cascade and calibration, and run the tree-metric test.
- Write paper A for ICML 2027.

**Exit:** parent accuracy and false-positive rate, on the lab and on the Hub,
against the baselines.

### Phase 2: census and capability (February – April 2027)

- Scale to 100k models.
- Run the T3 campaign on about 10k actively chosen models.
- Build the capability and risk engines.
- Build ModelSearchBench and Zoo Query v1 (API, MCP, Space).
- Submit the workshop version of paper E (February).

**Exit:** prediction intervals that meet their stated coverage, and regret
against the baselines.

### Phase 3: release (May 2027)

- Papers B and C.
- Public dataset, library, Space, and MCP server.
- A launch audit with responsible disclosure.

### Phase 4: living resource and science (June 2027 onward)

- Daily ingestion.
- Scoring of the prospective set.
- Paper D.

## 12. Risks

| Risk | Mitigation |
| --- | --- |
| Bandwidth or cost overrun | Tiered option, deduplication, strata priority, HF collaboration |
| Being scooped. The Model Atlas group, the LLM-Biometrics group, modelDNA, Cisco, and ModelLens are all active. | Release the census early (arXiv plus HF) to become the reference. Differentiate on scale, calibration, and cold-start capability. |
| Weight features plateau for capability prediction | Built into the design: weights are the prior and items the confirmation. Publish the cliff honestly. |
| Noisy Hub lineage labels | Lab truth, a verified-edge subset, and a manual audit |
| Legal exposure (gated models, licenses) | Derived statistics only; no reconstructive data for restrictive licenses; a takedown path; sign-off before release |
| Misuse (detector evasion, harassment through the audit) | Aggregate reporting; notify organizations before naming them; document laundering limits; keep the robust mode |
| The sketch null is not centered | Empirical nulls from hard negatives; check in the pilot |

## 13. Decisions for the team (open)

These change what gets built. Phase 0 cannot finish without them.

1. **Resources.**
   - How many people, for how long?
   - Where are the GPUs, and how many hours are approved?
   - What is the sustained internet download rate at that site?
   - Is a Hugging Face collaboration real or only hoped for?

   These decide whether the full option is possible at all. At 1–2 Gbps it would
   take months. They also decide whether Phase 1 (25–50k models, the lab, and
   paper A by late January) is realistic.
2. **Priorities.**
   - Is ICML 2027 a hard target, or is NeurIPS 2027 the first deadline that must
     be met?
   - Is an internal Booking.com team waiting on governance features by a date?
     If so, the license chain and risk flags move ahead of capability
     prediction.
   - Who owns a public API with daily ingestion after launch? That needs
     production engineering that the timeline does not yet show.
3. **Access and release.**
   - Under whose accounts are gated licenses (Llama, Gemma) accepted at scale?
   - Will Booking.com and the university partners approve:
     - publishing derived statistics, code, and a public service?
     - a launch audit that names organizations?
     - internal use cases in ModelSearchBench?

   If public release is blocked, paper B and goal G5 change. If gated models are
   excluded, the anchors, lab bases, and hard negatives all change.

Whether legal review blocks extraction or runs alongside it depends on the
answer to item 3.

## 14. Calls made in this draft

- The core is text LLMs and the language towers of vision-language models.
  GGUF-only repositories get T0 everywhere and T1 only from sampled tensors.
  Diffusion models and their LoRAs are out of scope for now.
- Two-sided 32×32 sketches plus CountSketch-256 per tensor. Singular vectors only
  for anchors, and never published for restrictive licenses.
- Hill(n/2) with closed-form standard error is the primary tail statistic. DKS
  and CSN are kept as diagnostics.
- Per-skill IRT abilities replace leaderboard averages as the prediction target.
- About 10k models get item-level evaluation, chosen actively.
- Paper A (lineage) comes first, because it needs only the census plus the lab,
  and the competition there is moving fastest.
- Models above 75B parameters, including large MoE models (DeepSeek-V3/R1
  family, Llama-3.1-405B, Qwen3-235B), enter as **anchors only**.
  - About 30–50 such bases get T0 plus the depth-subset T1 *(est. about
    20–40 TB)*, with expert tensors handled by batched eigensolves.
  - Their derivatives get T0 plus sampled tensors.
  - This keeps governance questions such as "is this a DeepSeek derivative?"
    answerable without hundreds of extra terabytes.
- The existing 12.5k analyses are **re-run inside the census**, because they
  lack commits and full spectra. The old outputs stay frozen as the comparison
  baseline, as the reassessment did with the earlier run.
- The pilot's thresholds for choosing the full or tiered option, the item bank,
  and code placement are left to the builder.
- This plan lives in `docs/research/scaleup/`, following that directory's README.

## 15. References

Survey agents checked every arXiv ID against the arXiv API on 2026-10-23. Venues
marked † were not independently confirmed.

**Population and lineage**

- Model Atlas, Horwitz et al. — arXiv 2503.10633. v2 is the NeurIPS 2025
  position paper "We Should Chart an Atlas of All the World's Models".
- Laufer, Oderinwale, Kleinberg, "Anatomy of a Machine Learning Ecosystem" —
  arXiv 2508.06811.
- Spectral Signatures of LLMs, Zhang … Yang, KDD 2026 — arXiv 2607.03377.
- Chen, Shang, Liu, "Who Built This Model?", COLM 2026 — arXiv 2608.07786.
- Thakur, Khoury, centered residual signatures — arXiv 2608.14929.
- AWM, Zeng et al., ICLR 2026 — arXiv 2510.06738.
- HuRef, NeurIPS 2024 — arXiv 2312.04828.
- Independence Tests, Zhu et al., ICML 2025 — arXiv 2502.12292.
- Model Provenance Testing, Nikolić et al., NeurIPS 2025 — arXiv 2502.00706.
- MoTHer, ICLR 2025 — arXiv 2405.18432.
- modelDNA, Adil et al. — arXiv 2607.10617.
- Witness Overlap, Li et al., NeurIPS 2026 — arXiv 2609.31784.
- GhostSpec, AAAI 2026 — arXiv 2511.06390.
- LoRA vs full fine-tuning (intruder dimensions) — arXiv 2410.21228.
- Spectral DeTuning, ICML 2024 — arXiv 2402.10208.

**Weight-space learning**

- Model Zoos, NeurIPS 2022 Datasets & Benchmarks — arXiv 2209.14764.
- Survey of weight-space learning — arXiv 2603.10090.
- Learning model representations from public hubs — arXiv 2510.02096.
- Graph metanetworks for transformer LMs — arXiv 2610.18471.
- Universal Weight Encoder — arXiv 2610.21995.
- ProbeX, CVPR 2025 — arXiv 2410.13569.
- Spectral geometry of LoRA adapters — arXiv 2604.08844.
- W2T — arXiv 2603.15990.

**Search, prediction, and evaluation**

- ModelLens — arXiv 2605.07075.
- HugSelect — arXiv 2608.08069.
- Discovering Hidden Gems — arXiv 2601.22157.
- EmbedLLM, ICLR 2025 — arXiv 2410.02223.
- LOCUS — arXiv 2601.21082.
- UniRoute, ICLR 2026† — arXiv 2502.08773.
- LLM DNA, ICLR 2026 — arXiv 2509.24496.
- Lineage-regularized MF — arXiv 2504.19811.
- Observational scaling laws, NeurIPS 2024 — arXiv 2405.10938.
- Sloth, NeurIPS 2025 — arXiv 2412.06540.
- tinyBenchmarks, ICML 2024 — arXiv 2402.14992.
- metabench, ICLR 2025 — arXiv 2407.12844.
- Amortized model-based evaluation, ICML 2025 — arXiv 2503.13335.
- How benchmark prediction from fewer data misses the mark, NeurIPS 2025 —
  arXiv 2506.07673.
- ATLAS, ICML 2026 — arXiv 2511.04689.
- Fluid Benchmarking, COLM 2025 — arXiv 2509.11106.
- MatrixFormer — arXiv 2610.06751.
- Every Eval Ever — arXiv 2606.14516.
- Probabilistic matrix factorization for AutoML — arXiv 1705.05355.
- Hierarchical conformal, Dunn et al. — arXiv 1809.07441.
- Weighted conformal under shift — arXiv 1904.06019.

**HTSR and tail estimation**

- Martin & Mahoney — arXiv 1810.01075 and 2002.06716.
- FARMS, ICML 2025 — arXiv 2506.06280.
- Clauset, Shalizi, Newman — arXiv 0706.1062.
- Halko, Martinsson, Tropp — arXiv 0909.4061.

**Safety and ecosystem**

- Refusal geometry — arXiv 2502.17420 and 2602.02132.
- "Auditing Abliteration" (2026) — scholarship.miami.edu.
- Heretic — github.com/p-e-w/heretic.
- HF "State of Open Models: Summer 2026" — huggingface.co/blog.
- Open LLM Leaderboard retirement — huggingface.co/spaces/open-llm-leaderboard,
  discussion 1135.
- ZipLLM (Hub-wide deduplication) — arXiv 2505.06252.
- EU AI Act GPAI guidelines — digital-strategy.ec.europa.eu.
