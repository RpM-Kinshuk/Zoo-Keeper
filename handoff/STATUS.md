# Status

The live log. `CHECKLIST.md` holds the state of each item; this file holds
everything around it. Newest entries go at the top of each section. Date every
entry and name the agent or person.

## Snapshot

**2026-10-31.**
- Plans and handoff are complete. No code exists yet.
- Week 1 (Oct 29–Nov 6) is half over.
- Next per track:
  - **A:** AS, P0-4, A0, X1, and X3;
  - **B:** BS, L1, and E1;
  - **C:** CS, S1, then S2.
- Waiting on the person: P0-3.

## Tracks and sessions

Each lead fills in its row at setup (AS, BS, CS). Agents never commit; the
person reviews and commits.

| Track | Agent session (as the person's prompt named it) | Repository and branch | Worktree path |
| --- | --- | --- | --- |
| A, infrastructure | | `Model-Zoo` and `Zoo-Keeper`, `track-a` in each | |
| B, lab and methods | | `Zoo-Keeper`, `track-b` | |
| C, studies | | `Zoo-Keeper`, `track-c` | |

## Server facts

The work runs on a remote Linux server: workspace `zoo/`, data under
`$ZOO_DATA`, and the conda environment `esd_ind`. Filled by P0-4 and A0.

| Fact | Value | Checked (date, agent) |
| --- | --- | --- |
| Outbound HTTPS (login and compute nodes) (A0) | | |
| Free space under `$ZOO_DATA` (A0) | | |
| CPU `svdvals` time, 3584×18944 (A0) | | |
| Xet metadata reachable (A0) | | |
| `esd_ind` Python and key package versions (P0-4) | | |
| Baseline test run (P0-4) | | |

**Packages installed into `esd_ind` for this work:**

| Package | Version | Why | Date | Agent |
| --- | --- | --- | --- | --- |
| | | | | |

## In progress

| Item | Agent | Branch | Started | Notes |
| --- | --- | --- | --- | --- |
| | | | | |

## Blockers and questions for the person

| Date | Item | Question or blocker | Asked of | Answer |
| --- | --- | --- | --- | --- |
| | | | | |

## Runs waiting for a person

Each entry gives:
- the exact command;
- the pinned inputs;
- the expected bytes, wall-clock time, storage, and GPU-hours;
- the output path;
- how to verify success;
- how to resume.

The person marks it launched or finished.

| Date queued | Item | Command | Inputs | Expected cost | Output | Verify | Resume | State |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | |

## Decisions log

| Date | Decision | By | Recorded in |
| --- | --- | --- | --- |
| 2026-11-01 | **Two repositories.**<br>• Model-Zoo collects (collector, extractor, pilot) and builds the database (small databases may live in it).<br>• The big database lives parallel to the repositories (`$ZOO_DATA`).<br>• Zoo-Keeper builds on the collection: queries, lab, lineage engine, studies, baselines, research plans, and this handoff.<br>• Model-Research is frozen and will be retired. The repository is named Zoo-Keeper (the PyPI name `zookeeper` is taken). | team | `GUIDANCE.md` 4.1, 5.3; `README.md` |
| 2026-11-01 | **Tidy and lean.**<br>• One token (`HF_TOKEN`); `HF_GATED_TOKEN` removed.<br>• `.env` holds `HF_TOKEN`, `ZOO_DATA`, and `HF_HOME`; lab and build paths are fixed subfolders.<br>• No `/scratch` checks; P0-5 removed.<br>• Lean agent loop: test only what matters, one review for core-logic tasks, full suite once per batch, nothing extra.<br>• New portability rules: no machine-specific paths, relative artifacts, and storage and publication decided later. | team | `DIRECTIVES.md` D2.1, D7.2, D9.3, D10.2, D11; `PROMPTS.md`; `GUIDANCE.md` 4.4, 4.5 |
| 2026-11-01 | **Runs:**<br>• agents run jobs that read the weights of at most 100 models and finish within 2 hours;<br>• the person launches anything larger, or anything that spends money;<br>• GPU allocation is not a constraint, but nodes are shared. | team (the 100-model line), assistant (the 2-hour line) | `DIRECTIVES.md` D4 |
| 2026-11-01 | **Server and data:**<br>• one remote Linux server and a workspace `zoo/` holding both repositories and `.env` (tokens and paths);<br>• data under `$ZOO_DATA` (default `/scratch/zoo`);<br>• the lab folders must be durable;<br>• P0-5 becomes an agent task. | team; assistant (the `ZOO_DATA` layout) | `GUIDANCE.md` 4.5, `README.md` |
| 2026-10-31 | **Use the existing `esd_ind` environment on the work machine.** More packages may be installed into it as needed; installed versions are never changed without asking. P0-4 records the versions. | team | `DIRECTIVES.md` D7.5 |
| 2026-10-31 | **One person works with agent sessions; agents never commit.** Work stays uncommitted on each track's branch for the person to review and commit. Each track creates its own branch and worktree. Agents merge into `main` only when asked. P0-1 (tokens) and P0-2 (roles) are removed, and all `evidence/` and `zoo/` tasks move to track A. | team | `DIRECTIVES.md` D10.1, D10.3; `GUIDANCE.md` 5.3 |
| 2026-10-31 | **System One models and promising trending types are included and queryable,** after a check of 28 live repositories and 382 System One repositories. Rows were added to the extractor (heads, LoRA layouts, DeepSeek FP4/E8M0, single-file variants, ONNX), decision metadata and filters to the zoo, 6 decision-head models to the lab, and a candidate decision family to the selection paper. | team (approved), assistant (design) | merged plan §6.8 |
| 2026-10-31 | Handoff package created. A large-run threshold is defined for agents: at most 5 repositories, 10 GB, 1 GPU-hour, 1 hour, and no scheduler jobs. People may change it. | assistant (call) | `DIRECTIVES.md` D4 |
| 2026-10-31 | Lab tables contract v1: `lab/models.csv` (with `kind`), `lab/edges.csv`, a `lineage.json` per checkpoint, paths under `$LAB_ROOT`, the blind set outside git under `$LAB_BLIND_ROOT` | assistant (call) | `GUIDANCE.md` 4.4 |
| 2026-10-31 | `CHECKLIST.md` and `STATUS.md` have one copy, on `integration`. Tracks merge `integration` before any task that needs another track's work. *Superseded the same day: one copy in the main checkout, and no agent commits.* | assistant (call) | `DIRECTIVES.md` D10.1, D10.3 |
| 2026-10-31 | LLM-as-judge and human judging deferred to a later stage, with a local judge first. The coding budget is ample, not infinite. | team | merged plan §15.3 |
| 2026-10-31 | Gated-access requests filed and handled | team | merged plan §15.2 |
| 2026-10-31 | The zoo and its query framework are the core product; `diff` matters mainly between different models | team | merged plan §0.1 |
| 2026-10-29 | Review from five perspectives applied: G0 on Dec 4, the minimum lab, P6 cut, the corrected sketch and rank spec, the decode × geometry table | team | merged plan §0 |
| 2026-10-28 | Personal owner account for gated licenses. About $20/day cap on Claude evaluation calls. No standalone release of the 12.5k corpus. | team | merged plan §15 |
| 2026-10-27 | 2–3 people; framework, not a service; people launch large runs; frontier aim, not corporate tooling | team | merged plan §15.1 |

## Plan and interface change log

Every edit to a plan, or to a frozen interface, goes here: the date, the
document and section, what changed, why, and who approved it.

| Date | Document, section | Change | Why | Approved by |
| --- | --- | --- | --- | --- |
| 2026-10-31 | Extractor plan, Global Constraints | **Artifact names** use `safe_filename(f"{repo_id}@{revision}")`: `<safe>.manifest.json`, `<safe>.<tier>.fingerprint.safetensors`, and `<safe>.t2.spectra.h5` | Two revisions of one repo overwrote each other, and resume skipped the second; the manifest had no file name | assistant, pre-handoff cross-check; a person confirms at XF |
| 2026-10-31 | Extractor plan, Global Constraints | **Imports:** `evidence/__init__.py` puts `esd_experiment/src` on `sys.path`. The `PACKED_KEY` reuse claim is dropped. | The CLI and zoo could not import `measurement_config`; `esd_experiment.src` loads transformers | same |
| 2026-10-31 | Extractor plan, Global Constraints and Task 4b | **MoE subset** also covers unfused per-expert 2D tensors (`experts.<i>.`); all experts when E ≤ 8 | Mixtral, DeepSeek, and Qwen3 MoE store experts unfused, and T1 would read about 25% of bytes; `choice(E, 8)` fails for E < 8 | same |
| 2026-10-31 | Extractor plan, Task 4b | **Closed coverage vocabulary** (`measured`, `consumed`, `skipped`, `duplicate`, plus reasons). Module poisoning uses the immediate parent module, and integer buffers never poison. Added `test_unfused_expert_subset`. | The zoo's `coverage_status` depends on it; `position_ids` could poison the fixture | same |
| 2026-10-31 | Extractor plan, Task 5 | **Config subset** gains `num_hidden_layers`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, and `vocab_size`. `num_experts` is normalized, with a fallback to `text_config`. `select_file_set` considers weight files only. Fake fixtures are listed. | The zoo's config signature could never fire; VLM fields live in `text_config` | same |
| 2026-10-31 | Extractor plan, Task 6 | Fingerprints store **`shape`**; small views are zero-padded to k × k; `delta_confidence` is stored. `fingerprint_tensors` writes T1 and requires `repo_id` and `revision`. The block hash is defined. A T2 pass writes its b = 256 blocks as the standard **`.t1` file**. **The HDF5 uses the collector's `/layers` and `/eigs` conventions.** The vocabulary fixture uses 4,096-row embeddings. Added `test_revisions_do_not_collide`. | `reshaped` was undetectable; T2-only lab models would have mixed full-matrix and row-block sketches; the HDF5 claim was false; the vocabulary test could not pass | same |
| 2026-10-31 | Extractor plan, Task 7 | **Strata:** `text:<size>` for text and the modality alone otherwise. `models.csv` gets a **`revision`** column, which `manifest` honors. | `PILOT_HUB_PLAN` keys contradicted the stratum rule; scores pinned at today's `main` would almost never verify | same |
| 2026-10-31 | Extractor plan, Tasks 8–9 | **`pair_similarity`** moves to `compare.py` as the one calibrated statistic. **The calibration JSON** gains `sketch_k`, `row_block`, `tier`, and `n_pairs`; a mismatch means `uncalibrated`. Added **`check_verdict`** (consistent, inconsistent, undetermined, uncalibrated). `main(argv, reader_factory=, api=)` makes tests offline. Fixtures fixed (`unrelated_dir` declares a base; `synthetic_run`), and the syntax error in the `no_access` test fixed. | The null was being applied to a different statistic; verdict names did not match between `diff`, `check`, and the zoo; tests could not run offline | same |
| 2026-10-31 | Extractor plan, Task 10 | `day_one_checks.py` is written and run in week 1 (checklist A0) | The week-1 schedule needs it before Task 10 | same |
| 2026-10-31 | Zoo plan, Global Constraints and Task 4 | **Retrieval uses T1 only.** `select_retrieval_views` takes the views table (with `n_rows`). `nearest` breaks ties by key. | Full-matrix and row-block sketches are not comparable; the tall-view rule needed shapes | same |
| 2026-10-31 | Zoo plan, Task 5 | **`similarity_edges`** scores pairs with `pair_similarity` at the null's `sketch_k`; retrieval only proposes candidates. Its p-values are for display (floor 1/(n+1)); the P1 engine owns error control. | Calibration was being applied to the wrong statistic | same |
| 2026-10-31 | Zoo plan, Tasks 1–3 | Universe: SHA constants, `child@SHA10` is a byte copy, `org/unrelated` declares an external base, and the return type is specified. `views` gains `n_rows` and `n_cols`, and comes from T1 when present. `coverage_status` gains `not_measured` and uses the closed vocabulary. `--frame` is the sampled `models.csv`. `dup_group` means "identical on every measured block". | Fixture assertions depended on unspecified details; `inclusion_prob` lives only in `models.csv` | same |
| 2026-10-31 | Zoo plan, Task 2 | **Leaderboard:** case-insensitive repository match; an empty sha is unverified; `harness` carries the precision, so bf16 and fp16 rows stay distinct | The real data has 32 repositories evaluated at two precisions (sometimes with different values at one commit), 3 empty shas, and case mismatches | same |
| 2026-10-31 | Zoo plan, Task 7 | `family` follows directed parent and ancestor edges only. `check` adds a top-level `verdict` via `check_verdict`. `diff` raises `no_common_tier`. `info` has `predictions_note`. `find`'s skill filter averages rows per skill. | An undirected `same_root` edge could break the license chain; the verdict names did not match | same |
| 2026-10-31 | Zoo plan, Task 8 | The CLI test discards the build's stdout before parsing the JSON from `find`. `--frame` is documented as the sampled `models.csv`. | The test could not pass | same |
| 2026-10-31 | Extractor plan, Task 4b | Added an explicit **coverage precedence** list (buffers, consumed scales, poisoning, adapters, suffixes, geometry) | It was unclear which reason `position_ids` and unused scale tensors get | assistant, cold-start review; a person confirms at XF |
| 2026-10-31 | Extractor plan, Task 5 | In local mode, **`lineage.json`'s `name` becomes `repo_id`** | Lab model keys must not depend on a host's scratch path | same |
| 2026-10-31 | Extractor plan, Tasks 6–8 | `fingerprint_tensors` test passes a `revision`; the `hub_stats_sample` fixture is described; `child_dir` has the one reshaped embedding | Tests could not pass, or relied on undescribed fixtures | same |
| 2026-10-31 | Extractor plan, Task 10 | `day_one_checks.py` interface: `--scratch`, JSON keys, the Xet test, commit message; no unit test | It was underspecified for week 1 | same |
| 2026-10-31 | Zoo plan, header and Task 1 | Dependencies split (Tasks 1–4 need extractor Tasks 5–7; Tasks 5–8 also need Task 8). `edge_type` vocabulary fixed. | Z1 was scheduled before X8; `edge_type` values were unspecified | same |
| 2026-10-31 | Handoff files | Precedence includes CHECKLIST and PROMPTS. Kickoff step 0 added. `[?]` marker added. L2, L3, L4, and E1 dates moved after review. R3 drops the lab; L12 builds the lab's manifests. X4 and X7 need X1. | Contradictions and blockers found by a cold-start read | same |
| 2026-10-31 | Merged plan, header | Removed the stale "still untracked in git" note on Plan A | Plan A is tracked | assistant |
| 2026-10-31 | Extractor plan, Tasks 1, 4b, 6, and 8; Review Focus | F8_E8M0 decoding. `head` geometry. MLX LoRA and embedded PEFT LoRA (`lora_scale_unknown`). Only packed weights poison. DeepSeek-V4.x FP4-in-int8 and E8M0 FP8. `.scale` added to the suffix list. Diff summary lists `heads`. Five new tests. | System One heads were dropped, and DeepSeek-V4.x was unmeasured (merged plan §6.8) | team approved the design; a person confirms the rows at XF |
| 2026-10-31 | Extractor plan, Tasks 5 and 7 | Manifest `decision` block and `gguf` metadata. `onnx` weight format (T0-only). File reasons `precision_variant`, `bundled_adapter`, and `pickle_not_read`. Modalities `video`, `3d`, and `robotics`. Frame `system_one` and `is_decision`, and a `decision` stratum. Pilot is 320. Two new tests. | System One models must be recognizable; ComfyUI and ONNX repositories | same |
| 2026-10-31 | Zoo plan, Tasks 1, 3, 7, and 8; "After this plan" | `models` columns `architecture`, `is_decision`, `system_one`, `decision_type`, `num_labels`, and `labels`. `find` gains `decision`, `system_one`, and `architecture`; `info` gains a `decision` section. The universe gains `org/decider` and `org/parent-GGUF`, with a `test_decision_models_are_queryable` test. Zoo v1 notes cover decision scores and GGUF IQ and MLX dequantization. | Make them queryable | same |
| 2026-10-31 | Merged plan §6.6, §6.8 (new), §12, after-ICML | Section 6.8 records the check and its decisions. The lab adds at least 6 decision heads. The pilot is 320. The selection paper names a candidate decision family. | Record the scope change where research decisions live | same |
| 2026-10-31 | Handoff: DIRECTIVES D7.1, D10.1, D10.3, D10.4, stop-and-ask, reserved actions; PROMPTS loop and templates; CHECKLIST markers (`[r]`), definition of done, P0, setup items AS/BS/CS, owners (X4, X7, Z2, and R1 to A; L11 to B); GUIDANCE 4.1 and 5.3; README; STATUS tracks table | The new git workflow (no agent commits), and one person with agent sessions | team |
| 2026-10-31 | Handoff D7.5, stop-and-ask item 6, P0-4, X4b, implementer template; both plans' "Environment" constraints | `esd_ind` replaces `model_zoo`; installs allowed and logged; installed versions are not changed | team |
| 2026-11-01 | Handoff D2.1, D4.1, D4.4, D6.3; GUIDANCE 4.5; README workspace; CHECKLIST P0-5, L4, L12, L14, R2, R6; STATUS server facts; PROMPTS kickoff; zoo plan Task 8 Step 5; extractor plan "After this plan" | The new run rules and the server and data layout | team |
| 2026-11-01 | Repository split | `handoff/` and the research plans moved to Zoo-Keeper, along with the access scripts (paths fixed). Zoo plan Tasks 7–8 now target `Zoo-Keeper/zoo_keeper/`, with `zoo/testing.py` shared for fixtures; the build stays in Model-Zoo. Extractor Task 1 adds `pyproject.toml`, and its imports shim adds the repository root. Model-Research READMEs point to Zoo-Keeper. All handoff paths, tracks, and setup items are updated. | Model-Zoo collects; Zoo-Keeper queries and researches; Model-Research is frozen | team |
| 2026-11-01 | Extractor plan Global Constraints, Task 2 (`HubReader` has no `gated_token`; test renamed `test_gated_allowed_or_denied`), Tasks 8 and 10 (day-one `--data`); zoo plan Task 8 (`--zoo` default `$ZOO_DATA/builds/current`); merged plan token lines; handoff DIRECTIVES, PROMPTS (one reviewer template R), CHECKLIST (P0-5 removed, definition of done), GUIDANCE, README, STATUS | Tidy and lean, and portable | team |
| 2026-10-31 | Not changed (noted) | A per-architecture null and GPD tails are left to the P1 engine (E7). Zoo v0 uses one pooled null for display. | Keeps v0 simple; the P1 engine is where error control belongs | — |

## Weekly result log

Every Friday: what was measured, what changed in the plan, and what was
killed.

### Week of Oct 29 – Nov 6

- **Measured:**
- **Changed:**
- **Killed:**
