# Checklist

All the work, in order, from now to the selection paper. Near-term items are
detailed; later ones are coarse, and each gets its own plan first.

## How to use this file

**Markers:**

| Marker | Meaning |
| --- | --- |
| `[ ]` | open |
| `[~]` | in progress; the agent and branch are named in `STATUS.md` |
| `[r]` | implemented and reviewed by agents; uncommitted, waiting for the person to review and commit |
| `[x]` | done: the person reviewed and committed it |
| `[!]` | blocked; the reason is in `STATUS.md` |
| `[?]` | written and awaiting a person's review (plans) |
| `[HUMAN]` | a person does it; agents prepare and queue it in `STATUS.md` |

**Owners:** A, B, and C are the tracks in `GUIDANCE.md`, section 5.3. Agent
sessions run them, as the person's initiating prompt assigns.

**Cross-track code:** a task that needs code from another track waits until the
person has committed and merged it into `main` (D10.1). The lead asks as soon
as it is blocked.

**Due dates** are targets from the merged plan's week table (§12). A slip is
logged in `STATUS.md`, never hidden. IDs are stable; use them in `STATUS.md`.

**This file and `STATUS.md` have one copy,** in the main `Zoo-Keeper` checkout,
uncommitted until the person commits them (D10.3).

**Definition of done for every code item** (X, Z, and the code parts of L, E,
and S):

1. The plan's tests were written first and seen failing for the expected
   reason.
2. The plan's tests pass, and so do that package's tests. The full suite
   runs once per batch handed to the person, not per task.
3. Core-logic tasks passed their one review (`PROMPTS.md`).
4. It is left uncommitted on the track's branch. `STATUS.md` records the
   changed files and the plan's commit message as the suggested message.
5. The box is marked `[r]`, and `STATUS.md` has a line with the test counts.
   The person reviews, commits, and marks `[x]`.

In Zoo-Keeper, the full suite is `python -m pytest tests -q`.

---

## P0. Before agents start

Tokens and roles need no item: the tokens are in place, and one person
works with agent sessions.

- [x] **P0-3 [HUMAN] Commit the starting documents.**
  - In `Model-Zoo`: the two plans (`docs/superpowers/plans/`).
  - In `Zoo-Keeper`: `handoff/`, `docs/research/` (the merged plan and Plans A
    and B), `scripts/` (the access scripts), and the README.
  - In `Model-Research`: the removal of the moved plans, and the pointer edits
    to its READMEs.
  - Branches are not created here; each track creates its own (AS, BS, CS).
- [x] **P0-4 Environment and baseline** (A).
  - Use the existing `esd_ind` conda environment on the work machine
    (`conda activate esd_ind`). More packages may be installed into it as
    needed (D7.5).
  - Record in `STATUS.md` its Python version and the installed versions of
    torch, transformers, safetensors, huggingface-hub, numpy, scipy, pandas,
    h5py, pyarrow, and duckdb. The plans assume Python 3.10; note any
    difference.
  - Run `python esd_experiment/tests/test_setup.py` and
    `python -m pytest esd_experiment/tests -q`.
  - Done when: the versions and the baseline pass and skip counts are in
    `STATUS.md`.

---

## Phase 1: to gate G0 (Oct 31 – Dec 4)

### Track A: extractor (`Model-Zoo/evidence/`; extractor plan)

- [x] **AS Set up track A** (A, first). Needs P0-3.
  - Create branch `track-a` and a worktree from the latest `main` in both
    repositories (paths relative to the workspace root):
    - `git -C Model-Zoo worktree add ../worktrees/Model-Zoo-track-a -b track-a main`
    - `git -C Zoo-Keeper worktree add ../worktrees/Zoo-Keeper-track-a -b track-a main`
  - Record the branches and worktree paths in `STATUS.md`.

- [!] **A0 Day-one checks** (A, due Nov 4).
  - Write `evidence/scripts/day_one_checks.py` exactly as extractor Task 10
    specifies. It has no dependencies on other tasks, so write it now.
  - It is a diagnostic script with no unit test, so item 1 of the definition
    of done does not apply.
  - Run it on a login node and on a CPU and a GPU compute node (through the
    scheduler if there is one; this fits D4.1).
  - Done when: `STATUS.md` ("Server facts") lists, per node type, outbound
    HTTPS, the CPU `svdvals` time for 3584×18944, free space under
    `$ZOO_DATA`, and whether
    Xet metadata is reachable. The CPU timing replaces the estimate in the
    merged plan's §13; note it there and log the change in `STATUS.md`.
- [x] **X1 Task 1: safetensors headers and plain decoding** (A, due Nov 3).
  Includes F8_E8M0 decoding for block scales.
- [x] **X3 Task 3: orthonormal nested sketches and row blocks** (A, due Nov 5).
  It can run in parallel with X1.
- [x] **X2 Task 2: Hub reader** (A, due Nov 10). Needs X1.
- [x] **XF [HUMAN] Freeze the fingerprint format** (approved **Oct 31**; originally due Nov 6).
  - Owner approved after a reasoned technical check. Initial v1 hash definition
    includes the effective storage dtype; the existing file/field layout is
    retained. Approval is effective; these documentation edits await commit.
  - A person confirms that extractor Task 6's file layout and `Fingerprint`
    fields are frozen as written.
  - The same person approves or rejects every 2026-10-31 row of the
    `STATUS.md` change log, because several touch frozen interfaces in both
    plans, and marks each row.
  - From then on, track B builds on that layout. Before X6 lands, B writes
    synthetic fingerprints with a small writer that follows it.
- [x] **X4 Task 4: block delta and rank with abstention** (A, due Nov 13).
  Needs X1.
  - If the heteroscedastic test fails, report the measured ranks. Never loosen
    the test (D7.3).
- [x] **X4b Task 4b: decode × geometry view table** (A, due Nov 12). Needs X1.
  - It includes the rows added on 2026-10-31 for System One and trending
    models (merged plan §6.8):
    - the `head` geometry;
    - MLX LoRA, and PEFT factors stored inside full checkpoints;
    - the rule that only packed weights poison a module;
    - DeepSeek-V4.x's FP4 and E8M0 decodes.
  - Before writing the MXFP4 and NVFP4 fixtures, verify nibble order and scale
    layout against the reference dequantizers. Read one real tensor each from
    `openai/gpt-oss-20b` and `nvidia/Qwen3.6-35B-A3B-NVFP4` by range read,
    which is under the large-run threshold. Do the same for one FP4 expert
    tensor from `deepseek-ai/DeepSeek-V4.1-Flash-1017`. This needs `HF_TOKEN`
    (D2.3).
  - Use transformers' MXFP4 code as the reference. If NVIDIA ModelOpt is
    needed for NVFP4, install it into `esd_ind` only if that changes no
    installed version; otherwise ask (D7.5).
- [x] **X5 Task 5: T0 manifest with file-set selection** (A, due Nov 13). Needs
  X2 and X4b. The manifest keys freeze when it is done.
  - It includes the `decision` block, GGUF metadata from the Hub API,
    ONNX-only as T0-only, and the `precision_variant`, `bundled_adapter`, and
    `pickle_not_read` file reasons.
- [x] **X6 Task 6: fingerprints, `fingerprint_tensors`, `load_fingerprint`** (A,
  due Nov 20). Needs X1–X5.
- [x] **X7 Task 7: selection frame and pilot draw** (A, due Nov 20). Needs
  X1.
  - The modalities include `video`, `3d`, and `robotics`. System One models
    form a `decision` stratum, and the pilot is 320 models.
  - Download `cfahlgren1/hub-stats` `models.parquet` and record its size in
    `STATUS.md`.
  - Pin `pyarrow`.
- [x] **X8 Task 8: CLI, `diff_fingerprints`, `pair_similarity`, resumable
  batch** (A, due Nov 25). Needs X5–X7.
- [x] **X9 Task 9: pilot analysis, G0 report, calibration file** (A, due
  Nov 27). Needs X6, X8, and L2 (the lab tables, `GUIDANCE.md` 4.4).
- [x] **X10 Task 10: documentation, runbook, full suite, network smoke test**
  (A, due Nov 27). Needs X1–X9.
  - The network smoke test needs `HF_TOKEN` set in the environment. If it is
    not set, mark the item blocked and ask (D2.3).

### Track A: zoo v0 (build in `Model-Zoo/zoo/`, queries in `Zoo-Keeper/zoo_keeper/`; zoo plan)

- [x] **Z1 Task 1: schema, model keys, architecture signatures, `universe`
  fixture** (A, due Nov 23). Needs X6, because the fixture uses
  `fingerprint_tensors`.
- [x] **Z2 Task 2: score sources with revision-verified joins** (A, due
  Nov 24). Needs Z1.
- [x] **Z3 Task 3: in-memory tables** (A, due Nov 25). Needs Z1, Z2, X5, X6,
  and X7.
- [x] **Z4 Task 4: retrieval matrices and exact similarity** (A, due Nov 26).
  Needs Z3.
- [x] **Z5 Task 5: v0 lineage edges** (A, due Nov 27). Needs Z4, plus X8's
  `pair_similarity` and calibration format.
- [x] **Z6 Task 6: atomic deterministic build** (A, due Dec 1). Needs Z3–Z5.
- [ ] **Z7 Task 7: query API (`Zoo`), in Zoo-Keeper** (A, due Dec 2). Needs Z6
  and X8, committed and merged into Model-Zoo's `main` by the person, then
  `pip install -e Model-Zoo` (D10.1).
- [ ] **Z8 Task 8: CLIs and docs** (A, due Dec 4): `build` in Model-Zoo, and
  the query commands in Zoo-Keeper. Step 5 (the real-data smoke build) is R6.

### Track B: genealogy lab (the plan is written first)

- [x] **BS Set up track B** (B, first). Needs P0-3.
  - Create branch `track-b` and its worktree in `Zoo-Keeper` from the latest
    `main`:
    `git -C Zoo-Keeper worktree add ../worktrees/Zoo-Keeper-track-b -b track-b main`.
  - Record the branch and the worktree path in `STATUS.md`.

- [x] **L1 Write the lab implementation plan** (B, due Nov 4; a person reviews
  it by Nov 6).
  - Use the existing plans' format and save it to `Zoo-Keeper/docs/plans/`.
  - Sources: §12 (minimum lab), §10 (truth tiers and the blind set), and §6.5.
  - It must cover:
    - the bases, with license checks;
    - one script and one config per edge type, each dry-run at tiny scale (for
      example on SmolLM2-135M);
    - the edge list (`GUIDANCE.md` 4.4);
    - GPU-hour and storage estimates;
    - which jobs a person launches;
    - a scorer (L13);
    - how the lab is later released as a benchmark.
- [x] **L2 Lab tables and validator** (B, due Nov 9). Needs L1 to be
  reviewed.
  - `lab/models.csv` and `lab/edges.csv` use the columns in `GUIDANCE.md` 4.4.
  - A validator checks types, IDs, and acyclicity.
  - The validator defines `LAB_SCHEMA_VERSION`, and the tables cover every
    planned model except the blind subset, which C validates separately.
  - Lab scripts write each checkpoint's `lineage.json`.
- [x] **L3 Bases pinned** (B, due Nov 9).
  - At least 6 bases from at least 3 families, chosen from Qwen2.5-0.5B, 1.5B,
    and 7B; Qwen3-0.6B and 1.7B; SmolLM2; and OLMo-2-1B.
  - **Never Qwen2.5-3B.**
  - Record each base's commit and license.
- [~] **L4 Merges: at least 30** (B; configs due Nov 11; done by Nov 13).
  Agents may run the merges if they fit D4.1's limits; otherwise a person
  launches them.
  - Linear, SLERP, TIES, and DARE, with 2–4 parents each.
  - Some with per-layer `t`.
  - Use mergekit configs and keep each config's hash.
- [~] **L5 LoRA: at least 24 runs at ranks 8, 64, and 256** (B; queued in week
  1; [HUMAN] launches; done by Nov 20).
  - 2026-11-02:6 extra Smol135 GPU runs complete (instruction/dialogue, ranks8/64/256), exact scoped verification PASS. No formal quota credit; their fingerprints await A nested-adapter selector fix.
- [~] **L6 Full fine-tunes: at least 8 SFT and 8 DPO at ≤1.7B** (B; [HUMAN]
  launches; done by Nov 20).
  - 2026-11-02:2 extra Smol135 SFT and2 matching DPO GPU jobs complete, full frozen steps and unchanged DPO references; scoped verification/T1/scoring complete. No formal quota credit.
- [~] **L7 Abliterations: at least 8, classic and Heretic** (B, due Nov 20).
- [~] **L8 Simulated quantizations: at least 12** (B, due Nov 20). Each model is
  quantized, dequantized, and saved as bf16.
  - Three exact planned OLMo outputs are complete. Scoped Smol/Pythia extras
    are recorded separately and do not silently replace the minimum inventory.
- [~] **L9 Laundering transforms: at least 20** (B, due Nov 13).
  - Real scoped Smol rescaling/perturbation complete; rejected MLP permutation
    is excluded. Whole-lab and blind counts remain open.
  - Function-preserving permutation, rescaling, rotation, and targeted
    perturbation.
  - S6 (C) contributes the blind part. G1 counts L9 and S6 together toward
    the 20.
- [~] **L10 Chains: at least 3 of depth ≥3** (B, due Nov 20).
- [~] **L14 Decision heads: at least 6** (B, due Nov 20).
  - 2026-11-02:1 extra Smol135/Amazon Polarity classifier complete; held-out scoring183/256=71.48%, majority137/256=53.52%; verification/T1 complete. No formal quota credit.
  - Sequence-classification fine-tunes: 4 on lab decoder bases and 2 on
    ModernBERT-base (apache-2.0).
  - One is stored as PEFT factors inside a full checkpoint, as many System One
    models ship.
  - Each costs minutes of GPU time, so agents may run the batch (D4.1).
- [x] **L11 Hard negatives** (B, due Nov 20).
  - The PolyPythias seed variants (5 sizes × 9 seeds) and same-series siblings,
    listed in `lab/models.csv` as independents with no edges.
  - Pin each one's commit.
- [ ] **L12 [HUMAN] Fingerprint lab v0** (A, due Nov 20).
  - [ ] X5 follow-up: commit/merge A's existing R2 selector repair, excluding nested adapter factors when merged primary weights exist while preserving adapter-only checkpoints; then fingerprint B's6 pending LoRAs. Reproducer/current branch check: `handoff/track-b-incremental-l12.md`.
  - B has27 unique measured real models with T1 k64/block256: original22 plus5 newly trained non-LoRA outputs (new run also repeats existing Smol base). Six additional trained LoRAs await A primary-weight selector fix.
    Incremental T2/k128 handoff and exact launch are in `handoff/track-b-incremental-l12.md`;
    this does not complete whole-lab L12.
  - First, the lab's manifests:
    - local checkpoints: agents build these in local mode, which reads only
      local headers;
    - the lab's Hub bases and the independents: agents run `manifest` for
      them (metadata only) by Nov 18.
  - T2 with free T1 blocks at 128, 256, and 512, `--with_parent`,
    `--all_experts`, and `--sketch_k 128`.
  - Before X8's `batch` command exists, agents provide a short Python loop over
    `fingerprint_model`.
- [x] **L13 Lab scorer** (B, due Dec 4). Implemented against merged schema.
  - It takes `lab/edges.csv` and predicted edges in Parquet with the zoo's
    `LINEAGE_COLUMNS`.
  - It reports per-estimand AUROC, false matches per query at TPR 0.9,
    direction accuracy, and coverage and abstention.
  - It ships with the lab as a benchmark.
- [ ] **G1 [HUMAN] Minimum lab complete** (**Dec 11**). The counts in L3–L11
  and L14 must be met. If they are not, shrink the Hub set, never the lab.

### Track B: P1 lineage engine (the plan is written first)

- [x] **E1 Write the P1 engine plan** (B, due Nov 6; a person reviews it by
  Nov 9).
  Sources: §9 ideas 7–11, §10, and §11 (P1 row). It must cover:
  - **Problem setup:** the nested estimands, and retrieval over fingerprints.
  - **Merge regression:** unconstrained on [parents, base], per tensor or layer
    group, with a layer-block bootstrap, the condition number, and collinear
    groups.
  - **Edge typing:** rank, gap, module pattern, and fraction identical,
    honoring `below_floor`.
  - **The stacked abliteration test.**
  - **Tree-likeness:** four-point violations, normalized Robinson–Foulds, and
    sibling update cosine, followed by Chu–Liu/Edmonds assembly.
  - **Calibration:**
    - an empirical null from independents and siblings;
    - a GPD tail;
    - per-query BH or e-BH;
    - cross-family calibration;
    - Mondrian or weighted conformal sets with "not in reference";
    - exact permutation tests on shortlists.
  - **Outputs:** Parquet with `LINEAGE_COLUMNS` and a `method_version`.
  - **The evaluation harness:** false parents per query against pool size, byte
    curves, and the baseline interface.
  - **Rules:** local mode only, and no access to `$ZOO_DATA/lab_blind` before
    evaluation.
- [x] **E2 Engine skeleton on synthetic fingerprints** (B, due Nov 13).
- [x] **E3 Merge regression** (B, due Nov 20). Target effective coefficients
  within ±0.05 on lab merges, and agreement with the mergekit configs.
- [x] **E4 Rank and edge typing** (B, due Nov 20).
- [x] **E5 Stacked abliteration test** (B, due Nov 27).
- [~] **E6 Tree-likeness on lab trees** (B, due **Nov 27**). A person makes
  the pivot decision on the thresholds in `GUIDANCE.md` 5.2.
- [x] **E7 Calibration** (B, due Dec 4).
  - An empirical null, a GPD tail, and per-query FDR.
  - Agree with X9 on the calibration statistic and `sketch_k = 64`.
- [x] **E8 Lineage edges into the zoo** (B, due Dec 4). Inference, atomic
  publication, CLI and upstream build integration ready. Parquet with
  `LINEAGE_COLUMNS`, loaded with `zoo build --lineage_edges`.

### Track C: study, baselines, audit (the study plan is written first)

- [x] **CS Set up track C** (C, first). Needs P0-3.
  - Create branch `track-c` and its worktree in `Zoo-Keeper` from the latest
    `main`:
    `git -C Zoo-Keeper worktree add ../worktrees/Zoo-Keeper-track-c -b track-c main`.
  - Record the branch and the worktree path in `STATUS.md`.
  - Track C's code lives in Zoo-Keeper, including the baselines. Port what
    it needs from the frozen Model-Research; never import from it.

- [x] **S1 Write the study plan: heritability plus a first exchange-rate pass**
  (C, due Nov 4; a person reviews it by Nov 6).
  - Sources: §5.1, §5.4, and §9 ideas 2–3.
  - **Data:** leaderboard v1 and v2 per-task Raw scores, Model Atlas edges, and
    the September predictions. Start from `Model-Research/data/` and
    `Model-Research/fetch_data/`, copying the small files needed into
    `Zoo-Keeper/data/`.
  - **Methods:**
    - REML mixed model with a family random effect and known binomial
      variance;
    - parent–offspring regression on observed base scores;
    - network Brownian motion for merges;
    - raw or logit accuracy;
    - scale-free comparisons.
  - **The exchange rate:** for estimation and for selection (within and across
    families), with the strict bound.
  - **Rules and output:** a frozen config, and the workshop target.
- [x] **S2 Data assembly and audit** (C, due Nov 10).
  - Count the verified revision joins, and treat zero-byte files as missing.
  - Locate the September study outputs (`research_runs/` on the cluster copy;
    they are not in the local checkout). If they cannot be found, ask a
    person.
- [~] **S3 Heritability statistics** (C, due Nov 13).
  - Exploratory observed-parent run complete:128 legacy pairs/task over49
    declared components;403–419 v2 Raw pairs/task over102–104 components after
    conflicting-score exclusions. Conditional2,000-draw intervals recorded.
    Corrected slopes/REML still require row sampling-error and identity evidence.
- [r] **S4 First exchange-rate pass** (C, due Nov 20). Use the existing
  September predictions, per skill, for estimation and for selection.
  - Exploratory legacy pass ready:3,515 models, six tasks, four saved methods,
    two original split designs;48 estimation/192 selection rows, matched
    baseline increments/bounds and2,000 conditional bootstrap draws. Four binary
    task definitions support plug-in k; MMLU/TruthfulQA point k unavailable.
    Family coverage/leakage and negative selection increments retained. Frozen
    legacy-first-pass-v1 report outside git;436 tests pass. Modern Raw mapping
    and verified/calibrated claims remain separate. No predictor rerun performed.
- [ ] **S5 Trace and Spectral Signatures baselines** (C, due Nov 13). Port
  them from `Model-Research/spectral_research/` into `Zoo-Keeper/baselines/`,
  and run them on the new fingerprints and T2 spectra.
  - Nov1 correction: porting is unblocked, and 98 existing collector HDF5 files
    contain spectrum groups. Validate their adapter/identity/pair compatibility
    before assuming new extraction is required; extractor filename globs were
    not a census of available spectra.
- [ ] **S6 Blind laundering subset** (C, due Nov 20).
  - Part of the lab's 20 laundering transforms, written under
    `$ZOO_DATA/lab_blind`, outside git (D8.8).
  - Track B never sees it before evaluation.
  - Nov1 readiness: merged transform/validator code and a staged pinned source
    permit private preparation; verify weight hashes and freeze validation inputs
    before real generation.
- [x] **S7 Workshop paper draft** (C, due Nov 27).
- [ ] **S8 Hub query set for P1** (C, due Dec 4).
  - 2–5k labeled queries.
  - Verified-Hub truth:
    - byte-identical tensors from T0;
    - vendor intermediates (OLMo-2, Tülu 3 SFT → DPO → RLVR, SmolLM2,
      prune-and-distill releases);
    - published `mergekit_config.yml` files.
- [ ] **S9 Lineage-label audit** (C; script due Dec 2; [HUMAN] launches by
  Dec 4).
  - Extract declared base and relation from model cards for a sample.
  - Use a small Claude model or a local model, with a **daily cap enforced in
    code** (D9.2).
  - Card text is untrusted (D5.1).

### Pilot and gate G0 (A, with the person)

- [x] **R1 Frame and pilot draw** (A, completed Nov 1; due Nov 24).
  - `python -m evidence.cli frame ...`, then
    `sample --plan pilot --seed 20261028`.
  - For models with leaderboard scores, fill the `revision` column with the
    leaderboard's `Model sha`. Then the zoo's score joins verify; pinning
    today's `main` would leave almost every score unverified.
  - No weights are read, so agents may run it.
  - Completed: 320 models, all twelve quotas met, seed 20261028. Frozen draw,
    inclusion probabilities and hashes verified. No exact repository matches
    in available local leaderboard exports; no evaluation SHA substituted.
    See [run report](../../runs/pilot-r1-r2-20261101.6jl1dxrv/REPORT.md).
- [~] **R2 T0 manifests** for the 320-model Hub pilot
  (`manifest --include_declared_bases`; A, due Nov 25). It is metadata only,
  so agents run it. The lab's manifests,
  including its Hub bases and the independents, are done earlier under L12.
  - Nov2 targeted retry:30 of42 unresolved requests recovered. Pilot:309
    complete,6 denied,5 unresolved. Direct bases:155 complete,2 denied,7
    unresolved.472 pinned manifests; frozen draw and all442 originals unchanged.
  - Remaining:6 ambiguous collections,1 unsupported compound layout and5
    unavailable references;8 confirmed access denials remain. HTTP307/401 and
    metadata-type failures recovered. No resampling. See the run report and
    STATUS.md; R2 remains partial.
- [ ] **R3 [HUMAN] T2 run** (due Dec 1).
  - Covers the independents, the Hub related pairs, and 10 models per stratum
    above 9B: about 4 TB.
  - Not the lab: L12 already fingerprinted it, and running it again would
    collide with L12's files.
  - Free T1 blocks at 128, 256, and 512, with `--with_parent`.
- [ ] **R4 [HUMAN] T1 run** on the rest of the 320-model Hub pilot (due Dec 1).
- [ ] **R5 G0 report and calibration file** (A, due Dec 2).
  - `write_g0_report` and `write_calibration` at `sketch_k = 64`.
  - These read existing outputs, so agents may run them.
- [ ] **R6 Zoo v0 smoke build** on the pilot outputs, with `--leaderboard_v2`
  and the calibration file (A, due Dec 3). It reads fingerprints, not
  weights, so agents run it.
  - Run each query once on a known derivative pair.
  - Record the outputs in `docs/operations/zoo.md`.
- [ ] **G0 [HUMAN] Decide gate G0** (**Dec 4**). Use the thresholds below.
  Record the decisions in the merged plan's §12 and in `STATUS.md`.
  - **Cost:**
    - T1 median bytes ≤10% of the model per stratum;
    - p90 T1 time ≤10 minutes per model ≤9B;
    - failures ≤5%, excluding no-access and packed formats;
    - no unhandled 429s;
    - a 5k-model T1 run projects to ≤4 days.
  - **T1 for lineage** (on the lab plus known-independent pretrains):
    - parent-against-independent AUROC ≥0.98, and no more than 0.01 below T2;
    - within-related Spearman between T1 and T2 cosines ≥0.9;
    - decision agreement ≥0.9;
    - false matches per query ≤0.05 at TPR 0.9 against a 5k pool;
    - parent-against-sibling and parent-against-grandparent AUROCs reported.
  - **T1 for spectra:** median |α_T1 − α_T2| ≤0.1, and per-matrix Spearman
    ≥0.8. Otherwise spectra need T2, which affects P3 only.
- [ ] **G0-a [HUMAN] Freeze the lineage method** (Dec 4).
  - Freeze its config and hash the frozen predictions.
  - Start the prospective set: every model created after 2026-12-04, taken
    from the hub-stats dump.
- [ ] **G0-b Apply any pivot** a person decided (`GUIDANCE.md` 5.2). Update the
  affected plans, and log the change in `STATUS.md`.

---

## Phase 2: G0 to ICML (Dec 7 – Jan 29)

- [ ] **H1 [HUMAN] Hub T1 candidate pool** (Dec 7–18). The target is 5–20k
  models, with a minimum of 5k. Report false matches per query at each pool
  size reached.
- [ ] **S10 Baselines: modelDNA, Witness Overlap, MoTHer** (C, Dec 7–11).
- [ ] **H2 Byte curves and the T2 subset** (A, Dec 14–18).
- [~] **E9 Full P1 evaluation** (B, Dec 14–18). Harness and first real
  held-out cohort report ready; large-pool/T2/baseline gates pending. It covers:
  - three estimands and three truth tiers;
  - false parents per query against pool size, from 10² up to the pool
    reached, extrapolated with the calibrated null;
  - accuracy against bytes, with modelDNA at matched bytes;
  - direction accuracy against Witness Overlap;
  - coverage and abstention;
  - the evidence ladder (metadata → T0 → T1 → T2 → black-box).
- [ ] **S11 Hub baselines and the black-box test on a 200-pair subset; finish
  the workshop paper** (C, Dec 14–18).
- [ ] **P-1218 [HUMAN] Pivot check** (**Dec 18**). P1 needs either ≥2× fewer
  false matches per query at TPR 0.9, or ≥10× fewer bytes at equal AUROC, than
  the best baseline. Otherwise it is reframed as a benchmark paper.
- [ ] **H3 Rerun buffer, theory remarks, related work** (all, Dec 21–Jan 1).
- [ ] **F1 Freeze results** (Jan 4–8): a frozen protocol config, input hashes,
  and the final runs.
- [ ] **F2 Figures and an anonymized code release** (Jan 11–15).
- [ ] **F3 Verify the ICML 2027 dates and policies** once the call is out. An
  agent may search the web, and a person confirms. Update §11.
- [ ] **F4 [HUMAN] Submit** the abstract (about Jan 22) and the paper (about
  Jan 28).
- [ ] **F5 [HUMAN] Submit the heritability workshop paper** (ICLR 2027
  workshop, about February; verify the deadline).

---

## Phase 3: after ICML (Feb – Sep 2027)

Each item gets its own plan, which a person reviews before work starts.

- [ ] **V1 Write the zoo v1 plan** (A, by early February). It covers:
  - `recommend` without the quick test;
  - the first predictions: the parent's score plus per-skill shifts from S3,
    plus fingerprint neighbors, with abstention;
  - pip packaging and a demo Space;
  - GGUF dequantization for lineage only: every type `gguf-py` supports,
    including the IQ types common on trending repositories, and MLX
    quantization the same way;
  - decision-model scores (System1-Arena-style: accuracy, calibration error,
    latency) through `load_scores_csv`;
  - the derivative-watch script;
  - the training callback;
  - the MCP wrapper, after v1.
- [ ] **V2 [HUMAN] Public zoo v1 release** (Feb–Mar), with a launch-week
  gallery of diffs between recent headline releases and their bases.
- [ ] **V3 Zero-byte note** (Feb–Mar).
  - The Hub's tensor-sharing graph, from T0 headers and Xet hashes.
  - Idea 5: re-measure 500–1,000 historical models with pins.
  - A person posts it.
- [ ] **V4 ICML rebuttal** (Feb–Mar). Budget the time for it.
- [ ] **V5 [HUMAN] Zoo v2 census** (Feb–Apr), sharded across hosts:
  - T0 Hub-wide;
  - T1 on 50–100k models;
  - T2 on a 2–5k probability sample;
  - phase-2 detector validation;
  - T3-lite on at most 2k models;
  - repository- and download-weighted analyses.
- [ ] **V6 [HUMAN] P3 submission** (about early May).
  - Give notice to any named party first.
  - Read the license terms for each gated family before publishing its
    sketches.
- [ ] **V7 Selection paper** (Jun–Sep; a person submits it around late
  September). It covers:
  - `recommend` with the quick test;
  - dense panels in 2–3 task families, one of them a self-hosted family with
    expensive evaluation (local judge first). A decision-task family (System
    One models against LLMs) is a candidate;
  - baselines: popularity, metadata, historical scores, BenchPress,
    ModelLens-style ranking, and bandit search without a prior;
  - the exchange rate and heritability per edge type;
  - the **G2** decision;
  - scoring the prospective set.
- [ ] **V8 [HUMAN] Decide on a hosted service or ingestion,** only after G2,
  and on evidence.

---

## Every week

- [ ] **Friday weekly result log** in `STATUS.md`: what was measured, what
  changed in the plan, and what was killed.
- [ ] **Storage check:** lab checkpoints, the parent cache, and T2 scratch are
  within bounds.
- [ ] **Interface check:** no frozen interface changed without a change-log
  entry.
- [ ] **Token hygiene:** list run-output files that contain a token-shaped string with
  `grep -rlE 'hf_[A-Za-z0-9]{30,}' <run dir>` (file names only, never the
  matches). Any hit goes to a person.
