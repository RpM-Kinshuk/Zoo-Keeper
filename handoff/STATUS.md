# Status

The live log. `CHECKLIST.md` holds the state of each item; this file holds
everything around it. Newest entries go at the top of each section. Date every
entry and name the agent or person.

## Snapshot

**2026-11-01.**
- Track B tooling is merged locally; real lab runs and zoo integration remain pending.
- Week 1 (Oct 29–Nov 6) is half over.
- Next per track:
  - **A:** AS, P0-4, A0, X1, and X3;
  - **B:** Current tooling merged locally at `65c7399`;314 offline CPU tests PASS on main. Plans and core components committed; full E8 waits for zoo Z1/Z4/Z6, L13 for Z1 schema. Planned166-model/187-edge inventory is not completed generation; training inputs/safe Heretic and real E6/G0 outputs remain pending;
  - **C:** CS, S1, then S2.
- Waiting on the person: P0-3.

## Tracks and sessions

Each lead fills in its row at setup (AS, BS, CS). Agents never commit; the
person reviews and commits.

| Track | Agent session (as the person's prompt named it) | Repository and branch | Worktree path |
| --- | --- | --- | --- |
| A, infrastructure | | `Model-Zoo` and `Zoo-Keeper`, `track-a` in each | |
| B, lab and methods | Codex (Track B lead) | `Zoo-Keeper`, `track-b` | `worktrees/Zoo-Keeper-track-b` |
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
| immutables | 0.21 | Missing mergekit YAML-runner dependency; installed with `--no-deps`, no existing versions changed | 2026-10-31 | Codex Track B |
| mergekit | 0.1.5 | Approved L4 merge config/reference implementation; installed with `--no-deps`, no existing versions changed | 2026-10-31 | Codex Track B |
| heretic-llm | 1.4.0 | Approved L7 Heretic adapter; installed with `--no-deps`, no existing versions changed | 2026-10-31 | Codex Track B |
| | | | | |

## Ready for review

- 2026-11-01, Codex (Track B), **current batch committed and merged locally** at `65c7399` by owner request.113 files: lab generators/validators, final planned tables and pins, engine foundations and real T1 extractor adapter. Verification on main: **314 passed in57.48s**, CPU/offline, using committed Model-Zoo extractor sources. Initial merged check had one captured mergekit subprocess failure (313 passed); a direct tiny CLI reproduction passed and the instrumented full rerun passed with no code changes, so the initial cause remains unconfirmed. Approved L1/E1 plans and Track B handoff records are committed separately. BS/L1/L2/L3/L11/E1–E5/E7 marked committed; actual-generation items, E6 and blocked E8/L13 remain open. No real lab jobs, push, branch deletion, or other-track edits included.

- 2026-11-01, Codex (Track B), **E8 extractor adapter ready, zoo integration still blocked**: `lineage/upstream.py` now loads real local T1 fingerprints using committed Model-Zoo `0d24eda`, with no zoo dependency for loading or shared median scoring. Explicit index binds pinned key, authoritative caller-supplied architecture, manifest and fingerprint; validates identity/path confinement/T1/version, preserves parent/delta summaries and input hashes. Compatible-view filtering precedes the actual upstream statistic, so zero/nonfinite views cannot dilute or poison it. Six new tests failed first then passed; **full offline CPU suite314 passed in57.72s**, with committed main and collector sources on PYTHONPATH. One fresh scoped core review PASS, no findings; no extra test reruns. Changed `tests/lineage/test_upstream.py`, `docs/operations/lineage.md`, `lab/README.md`; all uncommitted in Track B. Ruling: split extractor and zoo dependency boundaries because X1–X10 are merged but Z1/Z4/Z6 are absent; do not guess architecture signatures or replicate the frozen schema. Full infer/Parquet/CLI/build still pending those APIs, L13 needs Z1 LINEAGE_COLUMNS. Suggested message: `lineage: local inference CLI and zoo-compatible edge artifacts` (adapter portion only; remaining Task7 pending).

- 2026-11-01, Codex (Track B), **L2/L11 final planned table handoff ready**: `worktrees/Zoo-Keeper-track-b/lab/{models.csv,edges.csv,pins.json,reference_pairs.json,table_manifest.json,HANDOFF.md}` contains166 models,187 construction edges,54 external pins,180 independent pairs and one published sibling pair. Frozen schema v1 unchanged. X9 can consume these files before either track merges; real G0 still requires generated checkpoints/manifests/fingerprints. **308 offline CPU tests passed in75.14s**, including frozen-revision, runtime-config and deterministic full-inventory regressions; one focused fresh core review PASS. Changes also include `lab/negatives.py`, `lab/configs/negatives.json`, `tests/lab/test_negatives.py`, and `lab/README.md`; all uncommitted. Notus/Zephyr reproduction common-SFT evidence is pinned, historical training commit explicitly unverified. Important X9 contract: `kind=independent` includes siblings under frozen L11; use `reference_pairs.json` or `pins.json`'s `reference_type=polypythias` for the independent null, excluding both sibling IDs. No Track A code imports, schema changes, blind-set reads, or real weight jobs. Suggested messages: `feat(lab): add versioned truth tables and atomic checkpoint provenance`; `feat(lab): pin independent seed and sibling references`.

- 2026-10-31, Codex (Track B), **batch handoff**: final `CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q` → **305 passed,0 warnings,57.58s**, run outside sandbox after sandbox launch stalls. Includes latest shared offline PEFT guard and fresh-process DNS/socket/state regression. Full-suite test-name collision fixed with `tests/`, `tests/lab/`, `tests/lineage/` package namespaces; observed CUDA RNG capture reproduced by a failing CPU-policy regression, then fixed with explicit empty device lists in code and reference test. No supplied scientific test weakened/skipped. Source whitespace/artifact check passed; no weights/Parquet/HDF5 in repository. All changes uncommitted in `worktrees/Zoo-Keeper-track-b`. No owned background job remains; bounded quantization download stopped, no derived outputs verified.

- 2026-10-31, Codex (Track B): final L8/L9/L14 code reviews PASS. Quantization21 tests and three-bit tiny reloads; public transforms34 tests plus tiny/resume; decision heads10 tests plus three tiny fixtures, shared causal/classification offline guard covered by final full suite. Root minimum/export helpers3+5 tests, counts reviewed with L4; runbook `lab/README.md`. Prepared inventory112 public derivatives+7bases+47refs; portable registry52 real pins, two siblings unresolved. Actual training/merge/transform counts, real calibration/pivot metrics, upstream integration/scorer and release staging remain unfulfilled. Suggested messages: `feat(lab): add bf16 quantization simulations`; `feat(lab): add audited public laundering transforms`; `feat(lab): add decoder and encoder decision-head examples`; `feat(lab): export extractor invocation inputs`; `docs(lab): record tooling runbook and handoff`.

- 2026-10-31, Codex (Track B): L4/L10 code — single combined core review PASS. Actual mergekit YAML tiny runs cover all four methods with safe reload/resume; 21 merge tests /126 lab tests PASS. Three chain/minimum tests PASS, verified completion IDs only, C contribution never inferred. TIES/DARE edge coefficients now reflect configured weights; original implementer-only blank assertion corrected to frozen column meaning. Export helper5 offline tests PASS (initial missing-module red), preserves pinned identities and confines absolute-path CSV to invocation scratch. Real generation/minimum counts remain unachieved. Suggested messages: `feat(lab): generate hashed mergekit truth recipes`; `feat(lab): verify genealogy chains and minimum counts`.

- 2026-10-31, Codex (Track B): L6 — 16 task/81 lab tests PASS plus both offline two-step CLI runs; core review PASS, FP32 master/bf16 compute-output recorded. L7 classic — 16 task tests and tiny CLI PASS, core review PASS; installed Heretic1.4.0 fails closed before unsafe loading. E5 — 17 task/72 lineage tests PASS after two failing byte-accounting regressions fixed. E6 code — 13 task/70 lineage tests PASS, core review PASS; actual lab pivot NOT_EVALUABLE. E7 code — 41 task/113 lineage tests PASS after three fresh-review provenance findings reproduced/fixed (architecture binding, full conformal content hash, frozen scorer/universe binding). No real runs or calibrated performance claimed. Suggested messages in `/tmp/track-b-*-report.md`; files limited to plan tasks.

- 2026-10-31, Codex (Track B): E4 — 13 task /42 lineage tests PASS; core review PASS. Delta features and explicit development-only rules; real lab rule fitting not performed. Files: `lineage/typing.py`, `tests/lineage/test_typing.py`. Suggested message: `lineage: evidence-aware edge typing and floor abstention`. L11 live metadata returned 45/47 pinned with two unresolved siblings; all seven bases +45 references in portable `lab/pins.json`. Actual classifier_dropout variation retained in config evidence but excluded from topology; regression failed first then15 task tests PASS. L11 package run hit concurrent L6 missing-module red, to be settled by final batch suite.

- 2026-10-31, Codex (Track B): E3 code — 16 task /29 lineage tests PASS; one review caught omitted bootstrap sampling support on gapped layer coverage, regression reproduced and fixed with explicit interval abstention. Real ±0.05 coefficient validation waits for lab merges. L5 generator — 9 task /63 lab tests PASS; review fixes bind local weights to pinned hashes, match canonical/raw config hashes, and expose device/dtype. No real 24-run batch launched; L5 remains in progress until HUMAN runs. L11 helper — 13 task tests PASS, one core review PASS; actual 45 reference pins running, two sibling slots unresolved. Suggested messages: `lineage: unconstrained merge fits and layer bootstrap`; `feat(lab): add rank-stratified LoRA generation`; `feat(lab): pin independent seed and sibling references`.

- 2026-10-31, Codex (Track B): L3 — all seven proposed bases pinned to real full SHAs; Apache-2.0 evidence hashed (standalone files or owner-approved pinned README declarations), config/tokenizer evidence under `lab/inputs/evidence/` relative to `$ZOO_DATA`, portable registry `lab/pins.json`. Metadata-only command `python -m lab.cli inventory --pin --config lab/configs/bases.json --output "$ZOO_DATA/lab/inputs/pins.json"` returned 7/7 PINNED. Pin helper 14 tests PASS; one core review found config hash/dataset-freeze/license-source gaps, reproduced and fixed; live CLI path-helper mismatch fixed with regression. No weight reads or package changes. Files: `lab/pins.py`, configs bases/data, CLI pin glue, pin tests and requirements. Suggested message: `feat(lab): pin permissive bases and training inputs`. Dataset freezes/license evidence remain pending and are not claimed complete.

- 2026-10-31, Codex (Track B): E2 — local synthetic records/retrieval, 13 tests PASS after expected missing-module failure, fresh core review PASS. Files: `lineage/{records,retrieval,upstream,testing,__init__}.py`, `tests/lineage/`. No live upstream imports until merged A code (E8 integration). Suggested message: `lineage: local synthetic records and exact retrieval`. L2 code — 27 registry / 40 lab tests PASS after reproducing and fixing local-parent identity and merge-role verification gaps; complete table snapshot still waits for L3/L11 pins. Suggested message: `feat(lab): add versioned truth tables and atomic checkpoint provenance`.

- 2026-10-31, Codex (Track B lead): L1 / E1 — drafts at `docs/plans/2026-10-31-genealogy-lab.md` (13 tasks) and `docs/plans/2026-10-31-p1-lineage-engine.md` (8 tasks), saved in main per D10.4. One contract/coverage check; all required sections/task commands present, eight Python examples parse, inventory arithmetic checked (112 public derived + 7 bases + 47 references = 166 public rows; ≥4 blind examples owned by C). No code tests, installs, or jobs: docs only. Suggested messages: `docs: plan genealogy lab and benchmark scorer`; `docs: plan P1 lineage engine and evaluation`. Implementation awaits plan review. Planning allowances exceed current reported free space; actual pinned byte inventory must fit durable storage before launches.

- 2026-10-31, Codex (Track B lead): BS — `track-b` worktree created from local `main` (`c2758c5`) at `worktrees/Zoo-Keeper-track-b`; clean after setup, `esd_ind` Python 3.10.18 activated. Changed tracking files: `handoff/CHECKLIST.md`, `handoff/STATUS.md`. Suggested message: `chore: record track B setup`.

## In progress

| Item | Agent | Branch | Started | Notes |
| --- | --- | --- | --- | --- |
| L4–L10/L14 | Codex Track B | `track-b` | 2026-10-31 | Code and tiny fixtures ready; actual generation/minimum counts remain pending frozen data and real jobs. Final L2 planned tables now populated; no completed checkpoint counts inferred from inventory or fixtures. |
| E6 | Codex Track B | `track-b` | 2026-10-31 | Tree code reviewed; actual lab pivot remains NOT_EVALUABLE until real outputs. Extractor is now merged; E8 zoo integration/L13 still need zoo APIs. |
| | | | | |

## Blockers and questions for the person

| Date | Item | Question or blocker | Asked of | Answer |
| --- | --- | --- | --- | --- |
| 2026-10-31 | L8 bounded real run | Pinned OLMo quantization attempt interrupted with SIGTERM to the exact owned PID2493943 after test/startup stalls; exit143, no derivative outputs verified. Frozen runner/config under `$ZOO_DATA/runs/track-b-olmo-quant`; owned source staging `.olmo-source-wf5inaqw` under lab inputs reached about4.641GiB. Resume remains an agent-sized option after code verification; no large run launched. | owner (record only) | stopped; partial source retained |
| 2026-10-31 | Rebuild constraints | `mergekit0.1.5` declares pydantic>=2.10.6, existing2.9.1; required backend tests pass with --no-deps install. Heretic1.4.0 additionally declares newer PEFT/psutil/rich and missing optional dependencies; genuine execution already blocked. No existing package version changed. Avoid automatic dependency resolution against the collector environment. | owner (record only; versions require explicit approval to change) | preserved current environment |
| 2026-10-31 | L7 genuine Heretic | Installed1.4.0 calls from_pretrained with trust_remote_code=None/True and lacks forced local_files_only/use_safetensors; CLI also offers uploads. Adapter rejects before invocation, classic code continues; no genuine Heretic output counted. | owner (record only; safe upstream adapter/support needed) | pending |
| 2026-10-31 | E8/L13 real schema integration | Updated2026-11-01: X1–X10 merged at Model-Zoo `0d24eda`; E8 extractor adapter complete/reviewed. Zoo package absent: L13 needs Z1's LINEAGE_COLUMNS; full E8 needs Z1/Z4 retrieval/Z6 build. | Track A zoo progression | extractor portion resolved; zoo APIs pending |
| 2026-10-31 | L2/L11 complete tables | Resolved2026-11-01: Notus and Alignment Handbook Zephyr DPO reproduction pinned with parallel-training evidence and explicit historical-revision limitation. Final planned166-row/187-edge tables and54 pins available in Track B worktree. | Codex Track B | resolved; real checkpoint generation remains separate |
| 2026-10-31 | L5/L6/L14 real training | Proposed `trl-lib/ultrafeedback_binarized` has no license metadata; `stanfordnlp/sst2` declares unknown. Frozen selected-row hashes and local token inputs also pending. No actual training batch can be queued as runnable yet; tiny synthetic code work continues. | owner (record only; will need license evidence or approved data replacement before real training) | pending |
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
| 2026-11-01 | Track B commit and local merge authorized; record current progress and remaining blockers, preserve other tracks' edits and the ongoing worktree. | owner (“record this … and merge the current changes”) | code `65c7399`; Track B plans and handoff |
| 2026-10-31 | L1 and E1 approved for implementation; proceed autonomously through unblocked Track B tasks, lean per-task loop, no commits or large launches. | owner (“go ahead”) | Track B plans; checklist L1/E1 ready for commit |
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
| 2026-10-31 | E1 Task6 new engine-only APIs | Bind null architecture; hash complete conformal artifact; require frozen scorer/universe provenance with additive optional prediction_set keywords (missing/mismatch abstains). No upstream frozen schema changes. | Fresh review reproduced false calibration guarantees from cross-architecture use, modified thresholds, or refitted scorer. | Track B routine correctness ruling under owner autonomy; existing frozen-protocol requirement |
| 2026-10-31 | L1 Global Constraints / Task 2 | Accept a matching SPDX declaration in a pinned README as base-license evidence when standalone LICENSE is absent; hash and label the source. | SmolLM2, OLMo, and ModernBERT metadata declare Apache-2.0 but lack standalone license files. Dataset ambiguities remain blocked. | owner, explicit async approval |
| 2026-10-31 | New L1 / E1 drafts in `docs/plans/` | Lab inventory/scorer and P1 engine tasks; frozen interfaces preserved. Defaults: 16 public + ≥4 C blind transforms, e-BH with explicit p-value validity, Mondrian sets, projected stacked test plus optional exact local verification. | Implement existing checklist; explicit unknown truth, sibling-v-ancestor labels, and descriptive TPR threshold avoid misleading metrics. | Proposed by Track B; awaiting plan review per D10.4 |
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
