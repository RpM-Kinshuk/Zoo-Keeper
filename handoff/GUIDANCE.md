# Guidance

**Current interface amendment, 2026-11-05:**
[Model-Zoo compatibility](MODEL-ZOO-COMPATIBILITY.md) defines the current
measurement names, artifact/schema versions, package inputs and remaining A/B
work. It supersedes older interface examples below; scientific decisions and
frozen historical records retain their original meaning.

This document covers what we are building, why, and the few ideas you need to
make good decisions on any task. It summarizes the merged research plan
(`Zoo-Keeper/docs/research/2026-10-27-merged-research-plan.md`),
whose section numbers are given as "§". Where this summary and a source
disagree, the source wins (`README.md`, "When documents disagree").

## 1. The product

**The Model Zoo is a measured, queryable database of open models, with a
library and CLI anyone can use to ask it questions.** Every model in it carries:

- its pinned identity: the repository and its exact 40-character commit;
- a fingerprint: a small summary of its weights;
- its measured lineage, with evidence tiers;
- measured scores, kept separate from predicted scores;
- explicit coverage of what was and was not measured.

| Question someone asks | Query | Available in |
| --- | --- | --- |
| Give me models matching these filters | `find` | zoo v0 |
| What is this model? | `info` | zoo v0 |
| What models are like this one? | `similar` | zoo v0 |
| Where did it come from, and under which licenses? | `family` | zoo v0 |
| What changed between these two models? | `diff` | zoo v0 (extractor CLI and zoo library) |
| Is the declared base consistent with the weights? | `check` | zoo v0 |
| Which model should I use for my task? | `recommend` | zoo v1 without the quick test; the selection paper with it |

Decision ("System One") models are first-class members (merged plan §6.8):
- `find --decision` and `--system_one` list them;
- `info` shows their decision type, labels, and heads;
- `diff` against the base shows the head they added.

**Who it is for:**
- practitioners choosing a model;
- open-model maintainers tracking derivatives;
- labs auditing merges and checkpoints;
- safety researchers;
- researchers who need a real-weight dataset.

**What it is not:** a hosted service (not yet), corporate tooling, a new
benchmark, or a security scanner. Exposing wrong metadata is a side effect,
never the goal.

## 2. Two principles

1. **Weights answer what only weights can show:** lineage, edge type (LoRA,
   full fine-tune, merge, quantization), merge recipes, and duplicates.
2. **Measurement answers what testing shows more cheaply.** Weight evidence is
   a priced prior only where evaluation is expensive. Its price is the
   *exchange rate*: a prior with RMSE s at accuracy p is worth about
   k ≈ p(1−p)/s² evaluation items (§5.1). In-family that is about 10² items;
   across families, a few dozen.

## 3. How it works

### 3.1 Evidence tiers (§6.1)

| Tier | What is read | Cost | What it unlocks |
| --- | --- | --- | --- |
| **T0 manifest** | API metadata at the pinned commit, file hashes, safetensors headers, config | About 100 KB per shard | Duplicates, re-shards, shared tensors, architecture, declared lineage, format |
| **T1 signature** | One 256-row block per matrix | About 2–10% of model bytes | Neighbors, merge recipes, edge typing above the noise floor |
| **T2 full** | Full tensors, giving full spectra (the existing collector's numerics, version 7) | 100% | Every spectral estimator. A T2 pass also emits T1 blocks at heights 128, 256, and 512 at no extra cost. |
| **T3 behavior** | Model run on benchmark items | GPU | Measured skills. Deferred until zoo v2. |

### 3.2 Fingerprints

For each matrix *view*, the extractor reads one contiguous block of rows, `b =
256` rows starting at `min((rows−256)//2, 1920)`. The cap of 1920 keeps the
embedding block aligned when a fine-tune adds vocabulary. From that block it
computes:

- a two-sided **sketch** `S_rows · W_block · S_colsᵀ` of size k × k. The
  operators are orthonormal (QR with a sign fix), nested, and side-specific,
  seeded by `[20261028, d, side]`. k = 64 for the census and 128 for the lab and
  anchors, stored in fp32.
- a **block hash** (sha256) and the **block spectrum**.
- when the parent is known, the exact **block delta** against the parent's
  same rows (§3.4 below).

Two sketches compare by cosine on their common top-left block. A fingerprint is
one `.safetensors` file, with sketches as tensors and identity plus coverage as
JSON metadata, so anyone can load it.

Rank estimated *from a stored sketch* is reliable only to about k/4. Questions
about higher rank use the block delta computed at extraction time.

### 3.3 One table for every model type (§6.6)

The extractor works on **stored tensors, not architectures.** Each stored
tensor gets two independent choices:

- **decode:**
  - `plain`;
  - `fp8_scaled`: block size from `quantization_config.weight_block_size`;
    F32, BF16, or E8M0 scales;
  - `mxfp4`: including DeepSeek-V4.x's FP4 packed in int8;
  - `nvfp4`;
  - recorded gaps: GPTQ, AWQ, bitsandbytes, MLX quantization, and GGUF. GGUF
    and MLX arrive in zoo v1, for lineage only. ONNX-only repositories are
    T0-only.
- **geometry:**
  - `matrix`;
  - `expert_slice`: oriented with `hidden_size`; fused `gate_up` is split,
    except in gpt-oss; 8 seeded experts per layer at T1;
  - `conv_flat`;
  - `head`: fewer than 8 rows and at least 8 columns, the decision or
    classification head of a System One model;
  - `lora_product`: PEFT (also when its factors sit inside a full
    checkpoint), kohya, diffusers, and MLX, with α/r scaling (α/√r for
    rslora).

**Simple rules prevent mismeasurement:**
- never-a-matrix suffixes (scales, `qzeros`, `g_idx`, `.alpha`, and so on);
- module poisoning;
- vector-like skips (fewer than 8 columns), while a head (fewer than 8 rows)
  is kept;
- only packed weights block a module, never integer buffers;
- file-set de-duplication;
- `duplicate_of` for identical blocks.

Every stored tensor appears in coverage with a status or a skip reason. **A new
model type means a new table row, never a per-architecture code path.**

### 3.4 Block delta and the noise floor (§6.3)

The delta between two bf16 checkpoints is the update quantized to the bf16
grid. Below a relative update ρ ≈ 3e-3 it is mostly rounding, and a raw rank
estimate is wrong. The procedure:

1. All elements bit-identical: status `identical`.
2. More than half bit-identical: status `below_floor`, and no rank is reported.
3. Otherwise: biwhiten the delta with the parent's ulp²/12 profile (Sinkhorn),
   threshold at `1.1·(√rows + √cols)`, and report the rank with a gap-based
   confidence.

Never type a rank-1 edit from one tensor. Abliteration is detected only by the
*stacked* test across layers.

### 3.5 The zoo store and queries (§6.7; zoo plan)

- Six Parquet tables (`models`, `views`, `coverage`, `lineage`, `scores`,
  `predictions`) and one retrieval matrix per architecture signature, all in
  one directory with no database server.
- Rebuilt atomically and deterministically from the extractor's outputs.
- Queried through `Zoo(path)` over in-memory DuckDB.
- Weight similarity compares only models with the same architecture signature,
  which ignores vocabulary size.
- v0 lineage edges:
  - declared (from metadata);
  - verified (shared identical tensors);
  - detected (sketch similarity), labeled "uncalibrated" until the G0
    calibration file exists.

### 3.6 What P1 measures (§9 ideas 7–11, §10)

**The headline: "pairwise lineage accuracy does not scale."** False parents per
query are plotted against candidate-pool size, from 10² up to the pool reached,
for every method.

**Three nested estimands:** same root, ancestor, and immediate parent.

**Three truth tiers:**
1. the exact lab truth;
2. verified Hub truth: byte-identical tensors, vendor pipelines that release
   intermediates, and published `mergekit_config.yml` files;
3. declared labels, audited for noise.

**Methods:**
- retrieval;
- merge regression on [parents, base] without sum-to-one;
- edge typing: rank, gap, module pattern, and fraction identical;
- the stacked abliteration test;
- an empirical null with a generalized Pareto tail;
- per-query BH or e-BH false-discovery control;
- cross-family calibration;
- minimum-cost arborescence assembly.

**Baselines:** Trace, Spectral Signatures, modelDNA (at matched bytes), Witness
Overlap, 2602.00772, MoTHer, metadata imputation, and black-box testing on a
200-pair subset.

### 3.7 Capability, heritability, and the exchange rate (§5.1, §5.4, ideas 2–3)

**Heritability:** which skills post-training actually moves.
- Fit a REML mixed model with a family random effect and each score's binomial
  measurement variance.
- Add parent–offspring regression on observed base scores.
- Handle merges with network Brownian motion.
- Use raw or logit accuracy, never normalized scores.

**The exchange rate** is reported per skill, both for estimation (absolute
error) and for selection (error in pairwise differences, within and across
families).

These feed the zoo's first score predictions (zoo v1) and the selection paper.

## 4. Components and interfaces

### 4.1 Component map

| Component | Location | Plan | Track | Consumes | Produces |
| --- | --- | --- | --- | --- | --- |
| Extractor | `Model-Zoo/evidence/` | extractor plan (exists) | A | Hub or local checkpoints, the hub-stats dump | manifests, T1/T2 fingerprints, T2 spectra, `diff`/`check`, the G0 report, the calibration file |
| Zoo database build | `Model-Zoo/zoo/` | zoo plan, Tasks 1–6 and `build` (exists) | A | manifests, fingerprints, the sampled `models.csv`, scores, lineage edges, calibration | the zoo directory: Parquet tables and retrieval matrices |
| Zoo queries | `Zoo-Keeper/zoo_keeper/` | zoo plan, Tasks 7–8 (exists) | A | a zoo directory, plus `evidence` and `zoo` from Model-Zoo | `find`, `info`, `similar`, `family`, `diff`, `check`; later `recommend` |
| Genealogy lab | `Zoo-Keeper/lab/` *(suggested)* | **to write** (L1) | B, with C for the blind set | permissive bases, mergekit, PEFT, TRL | about 120 local checkpoints and the exact edge list (section 4.4) |
| P1 lineage engine | `Zoo-Keeper/lineage/` *(suggested)* | **to write** (E1) | B | fingerprints, lab truth, the calibration null | lineage edges in Parquet with the zoo's `LINEAGE_COLUMNS`; P1 results |
| Heritability and exchange-rate study | `Zoo-Keeper/studies/heritability/` *(suggested)* | **to write** (S1) | C | leaderboard v1/v2 per-task scores, Model Atlas edges, September predictions | estimates per skill and edge type; the workshop paper; inputs for zoo v1 predictions |
| Baselines and label audit | `Zoo-Keeper/baselines/` *(suggested)* | part of S1/E1 | C | fingerprints, cards | baseline results on the same queries |

**How the repositories relate:**
- **Model-Zoo** collects: the existing collector, the extractor, the pilot
  and its analysis, and the code that builds the database.
- **Zoo-Keeper** builds on the collection: queries, the lab, the lineage
  engine, studies, baselines, the research plan, and this handoff. It imports
  Model-Zoo's packages through `pip install -e Model-Zoo` (in `esd_ind`, once
  per server), and its tests run with `python -m pytest tests -q`.
- **Model-Research** is frozen: nothing new goes there. Port what you need,
  such as Trace from `spectral_research/` or the leaderboard CSVs from
  `data/`, into Zoo-Keeper rather than importing it.
- The September study outputs (`research_runs/`) are not in the checkout; they
  are on the old cluster copy. S2 locates them, or asks the person.

### 4.2 Frozen interfaces

Changing any of these needs a person's approval, a version bump, and an entry
in the `STATUS.md` change log (D7.12).

| Interface | Defined in | Frozen from |
| --- | --- | --- |
| Model key `"<repo_id>@<revision>"`, where the revision is a 40-character SHA or `local:<sha256>` | zoo plan, Global Constraints | now |
| Artifact file names, built from `safe_filename(f"{repo_id}@{revision}")` | extractor plan, Global Constraints | now |
| Manifest JSON keys | extractor plan, Task 5 | Nov 13 |
| Fingerprint file layout and the `Fingerprint` fields | extractor plan, Task 6 (dtype-aware block hashes, v1) | **2026-10-31: XF approved** |
| Coverage vocabulary (statuses `measured`, `consumed`, `skipped`, and `duplicate`, plus the closed list of reasons) | extractor plan, Task 4b | Nov 12 |
| Calibration file JSON (`statistic`, `sketch_k`, `row_block`, `tier`, `n_pairs`, `null_quantiles`, `related_threshold`), and the one statistic it describes (`pair_similarity`'s median cosine) | extractor plan, Tasks 8–9 | Nov 27 |
| `diff`/`check` JSON (`relatedness.verdict`; the top-level `verdict` for `check` via `check_verdict`) | extractor plan, Task 8 | Nov 27 |
| Lab tables (section 4.4), versioned by `LAB_SCHEMA_VERSION` in the lab validator | this document, then the lab plan | Nov 9 |
| `LINEAGE_COLUMNS` (including its `edge_type` vocabulary) and the other table columns | zoo plan, Task 1 | now: the P1 engine and the lab scorer build on it from week 2 |

### 4.3 Data flow

```
hub-stats dump ─► frame.csv ─► models.csv (stratified draw, inclusion probabilities)
                                   │
Hub repo or local dir ─► manifest (T0) ─► fingerprint T1 / T2 (+ spectra .h5)
                                   │                     │
lab: edges.csv + models.csv ───────┼─────────────────────┼─► pilot analysis ─► G0 report
                                   │                     │                  ─► calibration.json
                                   ▼                     ▼
leaderboard (Raw, verified joins) ─► zoo build: Parquet tables + retrieval matrices
P1 engine lineage edges (Parquet) ─►      │
                                          ▼
                       Zoo(path): find · info · similar · family · diff · check
```

### 4.4 Lab tables (minimum contract; the lab plan may only add columns)

All local lab checkpoints live under `$ZOO_DATA/lab` (section 4.5).
Paths in these tables are relative to it.

`lab/models.csv` has one row per lab model, including bases and independents:

| Column | Meaning |
| --- | --- |
| `model_id` | A stable lab ID |
| `kind` | `base`, `derived`, or `independent` (a known-independent hard negative, such as a PolyPythias seed) |
| `path` | Checkpoint directory relative to `$ZOO_DATA/lab`, or `repo_id@<40-char sha>` for a Hub model |
| `base_family` | For example `qwen2.5`, `qwen3`, `smollm2`, `olmo2`, `pythia` |
| `license` | The SPDX-like license of the base |

`lab/edges.csv` has one row per directed edge of exact truth:

| Column | Meaning |
| --- | --- |
| `child`, `parent` | Lab model IDs |
| `edge_type` | `merge`, `lora`, `sft`, `dpo`, `abliteration`, `quantization_sim`, or `laundering` |
| `role` | `parent`, or `base` for a merge's base or reference model |
| `coefficient` | The configured merge weight; empty for other edge types |
| `params_json` | Method details: merge method and per-layer t; LoRA rank, α, rslora, and targets; quantization method and bits; laundering transform; seed; data |

**Each local checkpoint directory holds a `lineage.json`,** written by the lab
script that made it:

```json
{"name": "lab/<model_id>", "bases": ["lab/<parent_id>", "Qwen/Qwen2.5-0.5B@<sha>"], "relation": "<edge_type>"}
```

The extractor uses `name` as the manifest's `repo_id` (extractor Task 5), so a
lab model's key is `lab/<model_id>@local:<sha256>` on every host. `bases` gives
`--with_parent` and `check` their parents in local mode.

**Running the extractor on the lab.** A small helper in `lab/` writes the CSV
that `evidence.cli manifest --models` reads:
- one row per model;
- `repo_id` is the absolute checkpoint path, or the Hub `repo_id`;
- `revision` is empty for local directories, which are pinned by content, or
  the pinned SHA for Hub models.

**Edge types in the zoo.** The lab's `edge_type` maps to the zoo's
`LINEAGE_COLUMNS` vocabulary:

| Lab | Zoo |
| --- | --- |
| `lora` | `adapter` |
| `sft`, `dpo` | `finetune` |
| `merge` | `merge` |
| `quantization_sim` | `quantized` |
| `abliteration` | `abliteration` |

`laundering` edges are scored on whether the edge is found, not on its type.
The scorer (L13) maps lab IDs to zoo keys through the manifests.

**Versioning.** The lab validator defines `LAB_SCHEMA_VERSION = "1"`. Any change
to these tables bumps it (D7.12).

**The blind laundering subset** uses the same tables and `lineage.json` files,
kept outside git under `$ZOO_DATA/lab_blind`.
- Track C builds it. Track B never reads it before evaluation (D8.8).
- Its models count toward the lab's 20 laundering transforms at G1.

### 4.5 Where outputs go

- **Code:** `evidence/` and `zoo/` in Model-Zoo. In Zoo-Keeper:
  `zoo_keeper/`, `lab/` (scripts, configs, the validator, and the two CSVs
  above, but never the blind set), `lineage/`, `studies/`, and `baselines/`.
- **Data** lives outside the repositories, under `$ZOO_DATA`. `<workspace>/.env`
  sets only `ZOO_DATA` and `HF_HOME`; everything else is a fixed subfolder,
  and code creates folders as it needs them:

  | Path | Holds | Keep? |
  | --- | --- | --- |
  | `$ZOO_DATA/runs/<run>/` | extractor runs: manifests, `t1/`, `t2/`, reports | durable (re-reading the Hub costs bandwidth) |
  | `$ZOO_DATA/builds/<name>/` | zoo builds; `builds/current` is the zoo CLI's default | durable but rebuildable |
  | `$ZOO_DATA/lab/` | lab checkpoints, 1–2 TB | **durable and precious:** 150–300 GPU-hours to rebuild, and released as a benchmark in May |
  | `$ZOO_DATA/lab_blind/` | the blind laundering subset, for track C only | durable and precious |
  | `$ZOO_DATA/cache/` | T2 download scratch and the parent-block cache (the HF cache sits at `$HF_HOME`) | disposable; keep bounded |
- Run outputs are never committed to git (D6.3).

## 5. The plan in time

### 5.1 Committed order (§1.1)

1. Extractor, zoo store, and lab, in parallel from week 1. The P1 engine
   starts in week 2 on synthetic fingerprints and lab models.
2. **Zoo v0 at G0 (Dec 4), internal.**
3. **P1 at ICML 2027.**
4. The heritability and exchange-rate study (C, from week 1), targeting an
   ICLR 2027 workshop.
5. **Zoo v1, public (Feb–Mar):**
   - the library and CLI on pip, and a demo Space;
   - `recommend` without the quick test;
   - GGUF lineage;
   - a derivative-watch script and a training callback;
   - the zero-byte note.
6. **Zoo v2 and P3 (NeurIPS 2027 E&D, May):**
   - T0 Hub-wide;
   - T1 on 50–100k models;
   - T2 on a 2–5k probability sample;
   - T3-lite on at most 2k models;
   - two-phase error correction.
7. **The selection paper (ICLR 2028):** `recommend` evaluated end to end.

### 5.2 Pivot rules (§1.1). A person decides them on these thresholds.

| Trigger | Decided on | Then |
| --- | --- | --- |
| T1 lineage AUROC more than 0.01 below T2, or decision agreement < 0.9 on the lab | G0, Dec 4 | Read full tensors for a depth subset instead of row blocks |
| On independent pairs, the null's \|mean\| exceeds 1 null sd, or its spread falls outside 0.7–1.5× the theoretical 1.12/k | G0, Dec 4 | Exact permutation tests on shortlists only; drop closed-form error claims |
| Four-point violations > 20%, or normalized Robinson–Foulds > 0.2 on lab trees | Nov 27 | Report "how tree-like" as the finding; supervised edge scoring; no tree-recovery claim |
| P1 has neither ≥2× fewer false matches per query at TPR 0.9 nor ≥10× fewer bytes at equal AUROC than the best baseline | Dec 18 | Reframe P1 as a benchmark paper at ICML. Shrink the Hub set, never the lab. |
| The exchange rate is small | — | It becomes a result of the selection paper; `recommend` leans on measured scores |

**Owner decision, 2026-11-05:** apply the tree-likeness pivot now. The expanded
public E6 diagnostic flags all eight family/mode comparisons (81.8–96.2%
quartet violations at the unchanged1e-6 tolerance). Report “how tree-like” and
focus the next P1 work on supervised edge scoring; make no tree-recovery claim.
The owner instructed “go for it, and unblock A as well.” Decision/evidence:
`runs/track-b-pivot-unblock-a-20261105/pivot-decision.json`. This approves the
E6 direction change; lab roster/counting and A calibration remain separate.

### 5.3 Tracks

One person, the owner, works with agent sessions. The work is split into three
tracks so that sessions can run in parallel, each on its own branch and
worktree.
- The person's initiating prompt says which session runs which track; one
  session may run several in turn.
- The person reviews and commits everything (D10.1).

| Track | Owns | Repository and branch |
| --- | --- | --- |
| **A**, infrastructure | the extractor (all of `evidence/`), the database build, the frame and pilot, Hub runs, and the queries | `Model-Zoo` and `Zoo-Keeper`, `track-a` in each |
| **B**, lab and methods | the lab, including its hard negatives, and the lineage engine | `Zoo-Keeper`, `track-b` |
| **C**, studies | the existing-data study, the baselines, the label audit, and the blind set | `Zoo-Keeper`, `track-c` |

The merged plan's roles A, B, and C, and its two-person variant, were written
for people; they map onto these tracks. The blind laundering set stays blind
between sessions: a track B session never reads it (D8.8).

## 6. Settled decisions: do not reopen

Each line is a team decision or a reviewed call. Changing one needs a person.

1. **The zoo and its query framework are the core product.** Lineage, `diff`,
   and `check` are query types (2026-10-31).
2. **Papers run in sequence:** P1 (ICML 2027) → the zero-byte note
   (Feb–Mar) → P3, the zoo release (NeurIPS 2027 E&D) → selection
   (ICLR 2028). The heritability workshop paper reuses existing data. **P6 is
   cut.**
3. **P1's headline** is "pairwise lineage accuracy does not scale", with three
   nested estimands and three truth tiers. Its fallback is the benchmark
   framing at ICML, then COLM 2027. (KDD's February cycle accepts only
   resubmissions.)
4. **The team** is one person, the owner, working with agent sessions on
   tracks A, B, and C (2026-10-31; the plan first assumed 2–3 people). The deliverable is an open
   framework, not a hosted service. The aim is the research frontier and
   genuinely useful open tools, not corporate tooling.
5. **Large runs need no sign-off,** but the person launches them, not
   agents. Agents never commit; they merge into `main` only when asked
   (2026-10-31).
6. **Access.**
   - The owner's personal HF account holds the gated licenses.
   - One token, the owner's `HF_TOKEN`, kept in `<workspace>/.env`.
   - Upgrade to PRO only if limits bind.
   - Gated requests were filed on 2026-10-31.
   - Never automate acceptance; never multiply accounts.
7. **Budget.** Paid evaluation (LLM or human judging) is deferred to a later
   stage, with a local judge first. Claude spend is at most about $20 a day
   when used, under a code-enforced cap. The coding budget is ample but not
   infinite (2026-10-31).
8. **The historical 12.5k corpus** is never released as a standalone dump; it
   is released only with idea 5's reliability study.
9. **T1 row blocks** are the default census tier, with escalation to T2. T2
   emits T1 blocks for free.
10. **The sketch specification** is as in section 3.2 (k up to 256; fp32
    storage).
11. **Block deltas** abstain below the floor and use biwhitened rank. Parent
    blocks stay internal.
12. **Formats.**
    - One decode × geometry table.
    - FP8, MXFP4, and NVFP4 are decoded exactly.
    - GGUF arrives in zoo v1, for lineage only.
    - GPTQ, AWQ, and bitsandbytes stay parked.
    - `.bin` repositories are T0-only.
    - The lab's quantizations are simulated for P1.
13. **MoE:** 8 seeded experts per layer at T1; every expert for the lab and
    anchors.
14. **Storage.** Fingerprints are `.safetensors`, spectra HDF5, and tables
    Parquet with DuckDB. Search is exact until a measurement shows it is too
    slow.
15. **`diff` and `check`** report measured facts only, and are "uncalibrated"
    until G0. They are internal from G0 and public in zoo v1.
16. **Zoo releases:** v0 is internal at G0; v1 is public in Feb–Mar; v2 comes
    in May with P3.
17. **The minimum lab** is about 120 models on permissive bases. GRPO and
    full-parameter 7B training are cut. PolyPythias provides the hard
    negatives.
18. **Sampling.**
    - A 2–5k probability sample is drawn from the ≤75B frame with recorded
      inclusion probabilities.
    - Models above 75B are enumerated as a separate stratum.
    - A random phase-2 subsample measures detector error.
19. **The heritability method** is REML with parent–offspring regression and
    binomial measurement error, using network Brownian motion for merges.
20. **The lineage method freezes at G0.** Models uploaded after Dec 4 form the
    prospective set, and the frozen predictions are hashed.
21. **The simplest design that works,** everywhere.
22. **System One models and promising trending types are supported through
    table rows,** not per-architecture code (2026-10-31, merged plan §6.8).
    - Heads, the extra LoRA layouts, DeepSeek-V4.x's FP4, single-file
      variants, and ONNX T0 all land before the Nov 6 format freeze.
    - Decision metadata and GGUF metadata are in zoo v0.
    - Decision scores, GGUF and MLX dequantization, and the decision-task
      family come in zoo v1 and the selection paper.

## 7. Known traps

Each of these was found by checking, simulating, or reading real Hub files.
Avoid them.

### Measurement

- **The bf16 floor.** Below ρ ≈ 3e-3, deltas are rounding noise. Raw
  Gavish–Donoho over-counts rank there (for example, r = 8 reads as 18–20).
- **Rank does not type edges.** A rank count alone cannot separate LoRA rank 64
  from a full fine-tune: at decay 0.7ⁱ, r = 64 reads as 9–12. Combine rank,
  spectral gap, the module change pattern, and the fraction identical, and let
  the lab measure how well that works.
- **Scale and quantization side tensors** (`weight_scale`, `scales`, `qzeros`,
  `g_idx`, `absmax`, `_scale_inv`) are never weight matrices.
- **Not every 3D tensor is an expert stack.** Whisper `conv1 [768, 80, 3]` and
  linear-attention `conv1d [8192, 1, 4]` are not; sliced naively they become
  about 246k junk views.
- **Expert orientation differs between families**, even within Qwen. Orient
  with `hidden_size`. gpt-oss interleaves gate and up.
- **The FP8 block size** must come from config. DeepSeek's 576-row tensor fits
  blocks from 116 to 143.
- **Repositories store weights several times:** SDXL (35 GB for 7 GB),
  Mistral `consolidated`, gpt-oss `original/`, and `.fp16.` variants. Read
  them once.
- **Fine-grained MoE** at T1 would read 20–25% of the bytes without the expert
  subset.
- **Decision heads are tiny,** for example `[2, 384]`. A generic "skip small
  matrices" rule drops exactly the part that makes a System One model what it
  is.
- **Integer tensors are not all packed weights.** A Swin
  `relative_position_index` or a NeMo integer scalar must not block the layer
  around it.

### Statistics

- **The sketch null is not centered at zero.** Shared operators center it at
  the population weight cosine. Estimate it empirically from known-independent
  models.
- **Tensors within a model are dependent.** Do not use Stouffer's method.
  Calibrate the combined statistic (median cosine over shared views) directly,
  with a generalized Pareto tail and per-query false-discovery control.
- **A calibration applies only to the statistic it was built for.** A null for
  the median cosine at k = 64 says nothing about a mean cosine over 32×32
  retrieval blocks.
- **Lab calibration does not transfer to the Hub automatically.** Calibrate
  across families and use the verified-Hub tier.
- **Merges are not sum-to-one.**
  - SLERP brings in a base term.
  - TIES and DARE give *effective* weights.
  - Task arithmetic has Σc ≠ 1.
  - Report the condition number, and group collinear parents.
- **Tree-recovery guarantees** (Atteson plus JL) are vacuous here. Measure how
  tree-like evolution is instead.
- **The exchange rate.**
  - "At most" assumes RMSE = 1.25·MAE. The strict bound uses RMSE ≥ MAE.
  - The value of weights is incremental over the no-weight prior: about 18–25
    items across families.
  - Selection differs from estimation.
  - Do not count IRT's gain twice.
- **Prevalence.** Horvitz–Thompson estimates the prevalence of *detections*.
  Correct with phase-2 sensitivity and specificity (Rogan–Gladen), and report
  both repository- and download-weighted numbers.

### Data

- **The historical corpus:**
  - 1,606 of 3,817 local `svd_results` files are zero bytes;
  - its Gram-path small eigenvalues are at the noise floor;
  - key and value projections are missing wherever hidden ÷ key-value width
    ≥ 8;
  - 8,467 `Architecture` cells are truncated.
- **Leaderboard normalized scores are floor-clipped**, and GPQA bunches near
  the floor. Use the Raw columns and scale-free comparisons (MAE ÷ sd, or R²).
- **Declared bases are noisy.** Scoring only against them penalizes correct
  detections.
- **Revisions of one repository usually keep the same weights.** `diff` matters
  mainly between different models.
- **GGUF and single-file repositories are everywhere.** About half of the top
  System One repositories are GGUF-only, and ComfyUI repositories ship one
  model in several precisions side by side.
- **System One models rarely ship `systemone.yaml`.** Recognize them from tags
  and config fields.
- **Hub text is untrusted** (D5).

### Scope

- **Weights cannot show** distillation into a freshly initialized student, or
  lineage across architectures. Say so.
- **An adversary can evade public-seed coordinate sketches** (by permutation or
  targeted perturbation). State the no-adversary scope; monitoring uses a
  private seed.

## 8. What good work looks like here

- **Small, quick, and verifiable.** Each plan task is reviewable on its own;
  the plan's tests pass; core-logic tasks get one review; nothing extra is
  built.
- **Honest outputs.** Measured facts carry their evidence tier, their coverage,
  and "uncalibrated" where that applies. A result that fails a threshold is
  reported as failing it, with the numbers.
- **Reproducible.** Every number traces to pinned inputs and a frozen config,
  as the September study already does
  (`Model-Research/configs/htsr_combined_study_v4.json`).
- **Readable by people.** Docs and reports say what was measured, on what, and
  what it means, in plain words.

A good status line:
- "X6 done: 6/6 plan tests pass; full suite 214 passed, 1 skipped (network).
  Spec review found a missing `elapsed_seconds` metadata field; fixed. Commit
  `abc123` on `track-a`."

A bad status line:
- "Fingerprints basically work; a couple of tests were flaky so I relaxed the
  tolerance."

## 9. Glossary

| Term | Meaning |
| --- | --- |
| **View** | One matrix the extractor measures: a 2D tensor, one expert's slice, a flattened convolution, or a LoRA product |
| **Fingerprint** | Per-model `.safetensors` file with each view's sketch, block hash, spectrum, and delta, plus identity and coverage |
| **Sketch** | A k × k two-sided projection of a view's row block |
| **System One model** | A decision model: an encoder or decoder with a decision head that returns a typed choice, score, or yes/no with calibrated confidence instead of text (merged plan §6.8) |
| **Head view** | A measured matrix with fewer than 8 rows and at least 8 columns, usually a decision or classification head |
| **Block delta** | Child-minus-parent on the same rows, computed while both are in memory |
| **ρ** | Relative update norm ‖Δ‖/‖W‖ |
| **`frac_identical`** | The fraction of bit-identical elements |
| **`below_floor`** | Too small a change to type |
| **Biwhitening** | Rescaling rows and columns so the noise variance is uniform before thresholding singular values |
| **Coverage** | One record per stored tensor: its decode, geometry, and status or skip reason |
| **Architecture signature** | `cfg:` hash of the core config fields (vocabulary excluded), or a `shape:` hash. Weight similarity compares only within a signature. |
| **`dup_group`** | Models whose views are all byte-identical |
| **Evidence tier** | `verified` (shared bytes or published configs), `detected` (statistical), or `declared` (metadata) |
| **Estimand** | Same root, ancestor, or immediate parent |
| **Truth tier** | Lab exact, verified Hub, or declared (audited) |
| **Hard negative** | A known-independent same-architecture model (PolyPythias seeds), or a same-series sibling |
| **Laundering** | A transform meant to hide lineage: permutation, rescaling, rotation, or targeted perturbation |
| **Abliteration** | Removing a refusal direction r from residual-writing matrices; detected by the stacked test across layers |
| **Calibration file** | JSON with the null quantiles of the median sketch cosine at a stated k. It turns "uncalibrated" into related, unrelated, or undetermined with a p-value. |
| **e-BH** | The e-value version of Benjamini–Hochberg false-discovery control |
| **GPD** | Generalized Pareto distribution, fit to the null's tail to get small p-values |
| **Exchange rate** | How many evaluation items weight evidence is worth: k ≈ p(1−p)/s² |
| **Heritability** | How much of a child's score is explained by its parent's, per skill |
| **Inclusion probability** | The chance a model entered the sample. It makes prevalence estimates design-unbiased. |
| **Prospective set** | Models uploaded after the G0 method freeze, scored later |
| **G0, G1, G2** | Gates: G0 is the T1 decision and zoo v0 (Dec 4); G1 is the lab complete (Dec 11); G2 is how much weight evidence `recommend` uses (Sep 2027) |
| **Trace, Spectral Signatures, modelDNA, Witness Overlap, MoTHer** | Published lineage methods used as baselines (§3.2, §17) |
