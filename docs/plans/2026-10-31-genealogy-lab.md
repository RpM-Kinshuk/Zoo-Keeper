# Genealogy Lab Implementation Plan

> **For agentic workers:** Use the lean task loop in `handoff/PROMPTS.md`: one implementer per task, one review for core logic, and no agent commits. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** L1 approved for implementation by the owner on 2026-10-31. Suggested plan commit: `docs: plan genealogy lab and benchmark scorer`. This document authorizes no launches or publication.

**Goal:** Build the minimum genealogy lab with exact, reproducible parent edges, pinned permissive inputs, a scorer, and a benchmark release recipe.

**Architecture:** `lab/` owns the frozen CSV contract, a deterministic recipe inventory, and one Python entry point/config per edge type. Each recipe writes a complete local safetensors checkpoint plus provenance in a temporary directory and renames it into place. Scoring reads manifests and predicted lineage tables; it never needs weights.

**Tech Stack:** Python in `esd_ind`; torch, numpy, pandas, safetensors, transformers, huggingface-hub; PEFT/TRL for training, mergekit for merges, Heretic for its abliteration variant, pyarrow for predictions. Existing versions stay fixed. Add missing direct dependencies to `requirements.txt` at their installed versions only after checking that installation changes no installed version. Use JSON configs except mergekit YAML recipes.

**Spec:** `docs/research/2026-10-27-merged-research-plan.md` §§6.5, 6.8, 10, 12; `handoff/CHECKLIST.md` L2–L14; `handoff/GUIDANCE.md` §§4.2–4.5. The extractor and zoo plans in `Model-Zoo/docs/superpowers/plans/` define imported interfaces.

## Global Constraints

- `LAB_SCHEMA_VERSION = "1"`; preserve exactly `models.csv` columns `model_id,kind,path,base_family,license` and `edges.csv` columns `child,parent,edge_type,role,coefficient,params_json`. No new columns are needed.
- `kind` is `base`, `derived`, or `independent`. Lab edge types are `merge,lora,sft,dpo,abliteration,quantization_sim,laundering`; roles are `parent` or a merge's `base`. Empty coefficients for non-merges.
- Code/configs/tables live in `Zoo-Keeper/lab/`. Weights, downloads, hashes of completed runs, logs, manifests, and release staging live outside git under `$ZOO_DATA`; local model paths are relative to `$ZOO_DATA/lab`. Never store absolute paths in artifacts.
- All Hub inputs use actual 40-character commit SHAs. There are no invented pins in this plan. Local keys come from extractor manifests as `lab/<model_id>@local:<sha256>`; use its content hash rather than inventing another identity algorithm.
- Each completed local directory contains `lineage.json` with `name = "lab/<model_id>"`, `bases` naming direct parents (and merge reference), and `relation` equal to the lab edge type. Hub parents are `repo_id@SHA`; local parents are `lab/<model_id>`. References resolve through the lab registry.
- Authenticate Hub access using the environment's `HF_TOKEN`; never log it or open `.env`. Unit tests make no network calls. Only safetensors weights; `trust_remote_code=False`, no pickle loaders, no code from Hub repositories.
- Use permissive bases only, never Qwen2.5-3B. A gated base requires the person's agreement and granted access before addition. Verify actual license files at the pin, or a matching explicit SPDX declaration in the pinned README when no standalone license file exists (owner approved 2026-10-31); hash the evidence source and label its type. Registry labels below are candidates, not completed checks.
- Seed recipe generation with `20261028`; use explicit per-recipe seeds `20261028 + index`. Sort tables by model ID and edges by `(child,parent,role)`. Config hashes use canonical JSON (sorted keys, compact separators); preserve exact mergekit YAML bytes and their SHA256 too.
- Each artifact write is atomic. Completed directories are immutable; matching input/config hashes permit resume, and a different config at an existing output ID raises an error. A failed operation leaves its previous completed output intact.
- Agents may launch only jobs reading at most 100 models and expected to finish within 2 hours. All LoRA/full-tuning batches, whole-lab fingerprinting, and any larger job are `[HUMAN] launches`. Do not split a large batch to evade this rule. Use existing idle-GPU checks before GPU work.
- Track B never reads `$ZOO_DATA/lab_blind` before the evaluation phase. Allocate 16 public transforms here and at least four blind transforms to C. G1 requires the combined minimum of 20; before evaluation accept only C's aggregate completion attestation, never its files or recipe details.
- GRPO, full-parameter 7B training, packed-format dequantization, hosted services, synchronization/publishing infrastructure, and capability prediction are excluded. Quantization here means simulated quantize/dequantize to bf16.

## Review Focus

1. Missing/duplicate IDs, cycles, and path traversal must fail before any weights are opened (Task 1).
2. Unsupported architecture symmetries must fail explicitly; a matrix rotation through a nonlinear layer is not automatically function-preserving (Task 8).
3. Tied weights, GQA, and embedded adapter factors must remain loadable and retain the intended effective weights (Tasks 3, 7, 10).
4. Interrupted jobs and changed resume configs must never replace completed checkpoints (Task 1).
5. Sparse predictions, reversed edges, missing manifests, and empty positive/negative classes must produce explicit scoring denominators/statuses, not optimistic metrics (Task 12).

## Inventory and dependency choices

This is a concrete proposed matrix. License/pin checks in Task 2 happen before real execution; a failed license check blocks that input rather than silently substituting a model.

| Stable ID | Candidate repository | Family | Intended license to verify | Role |
| --- | --- | --- | --- | --- |
| `q25-05` | `Qwen/Qwen2.5-0.5B` | qwen2.5 | apache-2.0 | training/merges |
| `q25-15` | `Qwen/Qwen2.5-1.5B` | qwen2.5 | apache-2.0 | base/decision head |
| `q3-06` | `Qwen/Qwen3-0.6B` | qwen3 | apache-2.0 | training/merges |
| `smol-135` | `HuggingFaceTB/SmolLM2-135M` | smollm2 | apache-2.0 | base/tiny dry runs |
| `smol-360` | `HuggingFaceTB/SmolLM2-360M` | smollm2 | apache-2.0 | training/merges |
| `olmo-1b` | `allenai/OLMo-2-0425-1B` | olmo2 | apache-2.0 | training/negative sibling pairs |
| `modernbert` | `answerdotai/ModernBERT-base` | modernbert | apache-2.0 | two decision heads |

There are six decoder bases across four families, plus ModernBERT. `A = [q25-05,q3-06,smol-360,olmo-1b]` and `M = A[:3]` below. Extra architecture-specific support belongs to a lab transform adapter, not to the generic extractor.

| Component | Deterministic recipe expansion | Distinct outputs |
| --- | --- | --- |
| LoRA | A × two data tasks (`instruction`,`dialogue`) × ranks `[8,64,256]`; α=r, no rslora in this minimum batch | 24 |
| SFT | A × seeds `[20261028,20261029]` on instruction data | 8 |
| DPO | One continuation for each SFT output on preference data | 8 |
| Merges | M × ten recipes below, using same-base LoRA merged checkpoints | 30 |
| Abliteration | A × `[classic,heretic]` | 8 |
| Quantization simulation | A × symmetric per-row signed `[8,4,2]` bits; first three anchors use DPO seed-0 parent, OLMo uses its base | 12 |
| Laundering | A × `[mlp_permutation,rescaling,value_rotation,targeted_perturbation]` | 16 public; C contributes ≥4 blind |
| Decision heads | q25-05, q25-15, q3-06, smol-360, plus ModernBERT seeds 0 and 1 | 6 |
| Chains | For each M: base → SFT seed-0 → DPO seed-0 → 8-bit quantization simulation | 3 depth-3 paths, already counted |
| Independents | PolyPythias five published sizes × nine published seeds; add two pinned same-architecture post-training siblings with a documented common root and no ancestor relationship | 47 reference rows, no lab edges |

The public derived total is **112** (30+24+8+8+8+12+16+6), plus seven bases and 47 reference rows = **166 public table rows**. C's ≥4 blind derivatives live in its separate tables, bringing the combined derived minimum to 116. “About 120” describes generated examples, not a cap excluding hard-negative references. Chains reuse counted outputs. No blind IDs or counts are inferred.

For each merge architecture, order the six trained LoRA children by `(task,rank)` as P1…P6. Use P1…P4 as the four available parents:

- linear: P1/P2 weights `(0.25,0.75)`, `(0.5,0.5)`, `(0.75,0.25)`;
- SLERP: P1/P2 at `t=0.25`, `0.5`, and layerwise `t=0.2→0.8`;
- TIES: P1/P2 and P1…P4, reference the common base, densities `0.5`, equal configured task weights;
- DARE-TIES: P1/P2/P3 and P1…P4, reference the common base, densities `0.8`, equal task weights.

Store configured weights, interpolation schedules, and sparsification seeds in `params_json`; do not mislabel nonlinear SLERP or TIES/DARE configured weights as recovered linear coefficients. Reference-base edges have role `base`, not an extra independent parent. L4 configurations can be generated before L5, but their real execution waits for their trained parent checkpoints. A tiny synthetic merge verifies L4 tooling immediately.

## Common task execution

All commands below run from the Track B Zoo-Keeper worktree in activated `esd_ind`. Source workspace `.env` without displaying it. Each test fixture is generated under `tmp_path`. First run the named test file and observe its missing-module/function failure, then implement and rerun that file. At task end run `python -m pytest tests/lab -q` once; the lead runs `python -m pytest tests -q` once per handed-off batch. Do not run real recipes as unit tests. Suggested messages replace commit steps.

`lab/cli.py` exposes `validate`, `inventory`, `verify`, `export-models`, `score`, and `release-stage`; edge execution uses `python -m lab.scripts.<edge_type> --config PATH --lab-root "$ZOO_DATA/lab" [--dry-run] [--resume]`. `--dry-run` validates and prints the recipe plan without loading weights. Every edge script also has `--tiny`, which executes the small local synthetic fixture described in its task; it never downloads a fixture. Real SmolLM2-135M smoke jobs are optional agent-sized integration runs after Task 2, not tests.

### Task 1: Tables, inventory, validation, and atomic provenance (L2)

**Files:** Create `lab/__init__.py`, `lab/schema.py`, `lab/registry.py`, `lab/artifacts.py`, `lab/cli.py`, `lab/configs/inventory.json`, `lab/models.csv`, `lab/edges.csv`; tests `tests/lab/conftest.py`, `tests/lab/test_registry.py`. Add `tests/conftest.py` only if needed to expose the repository root.

**Interfaces:** `validate_tables(models: pd.DataFrame, edges: pd.DataFrame) -> None`; `expand_inventory(config: dict) -> dict[str,list[dict]]`; `materialize_tables(inventory: dict, pins: dict) -> tuple[pd.DataFrame,pd.DataFrame]`; `write_checkpoint(model_id: str, output_root: Path, recipe: dict, writer: Callable[[Path],None], *, resume: bool=False) -> Path`.

The inventory contains exact IDs/parents/configs for all 112 planned public children and references. Final CSV materialization needs Task 2's base pins and Task 11's independent pins. Implement/test L2 first; finish its complete, valid table snapshot as those metadata pins arrive. Do not insert fake SHAs or invalid placeholder paths to tick L2 early. Missing checkpoint directories are valid in a planned table; `lab verify` separately checks completion. Pin unresolved inventory entries explicitly and report them.

- [ ] Write `test_registry_contract` with synthetic pinned base b, derived c, and independent i. Assert exact column order and `LAB_SCHEMA_VERSION == "1"`; valid b→c passes. Parametrize duplicate IDs, absent parent, self-edge, b→c→b cycle, independent child with an incoming truth edge, unknown edge type, nonfinite merge coefficient, invalid JSON, absolute local paths, or `../escape`; each raises `ValueError` with its offending ID. Validate path syntax before resolution; physical containment is checked by `verify` and artifact access, with a separate symlink-escape test there.
- [ ] Write `test_inventory_counts`: assert counts `lora=24,sft=8,dpo=8,merge=30,abliteration=8,quantization_sim=12,laundering=16,decision=6`; all derived IDs unique; assert three specified depth-3 paths and six decoder bases/four families. A decision recipe contributes an `sft` or `lora` truth edge according to training method, never a new edge vocabulary.
- [ ] Write `test_atomic_resume`: writer creates a safetensors marker then raises; completed target does not appear. After success, identical resume is a no-op; altered config raises; previous bytes stay identical. Assert portable provenance and exact `lineage.json` name/bases/relation.
- [ ] Run `python -m pytest tests/lab/test_registry.py -q`; expected initial missing `lab.schema` failure.
- [ ] Implement validation, topological sorting, deterministic expansion, and staging/rename. Store recipe/config hashes and parent identities in `recipe.json` next to weights; hash only portable config content. Never log shell environments. CSV writes use temp files and rename; tables are produced together in a staging folder before replacing the pair through a versioned inventory directory and a single pointer file.
- [ ] Rerun the task file, then package tests once. Suggested message: `feat(lab): add versioned truth tables and atomic checkpoint provenance`.

### Task 2: Pin bases and datasets, verify licenses, freeze metadata (L3)

**Files:** Create `lab/pins.py`, `lab/configs/bases.json`, `lab/configs/data.json`; test `tests/lab/test_pins.py`. `lab/pins.json` is the small reviewed metadata registry; license copies and raw downloads stay under `$ZOO_DATA/lab/inputs`.

**Interfaces:** `resolve_pins(config: dict, *, api, download_text: Callable) -> dict`; `verify_pin(pin: dict) -> None`. Pin rows contain repo ID, 40-character commit, SPDX-like license, license-file path/hash relative to `$ZOO_DATA`, and configuration/tokenizer hashes. Dataset rows additionally contain split, selected index hash, preprocessing config, and license evidence. Read only selected metadata, never enumerate the whole Hub API.

Use candidate datasets `HuggingFaceH4/ultrachat_200k` for instruction/dialogue, `trl-lib/ultrafeedback_binarized` for DPO, and `stanfordnlp/sst2` for decision labels, conditional on license verification at actual revisions. Freeze deterministic disjoint ID slices for training, transform direction extraction, transform validation, and any scoring split; do not tune on held-out or blind data. A license ambiguity is a reported blocked input, not an assumed permissive license.

- [ ] Tests with injected fake Hub API/text fetcher assert all seven base pins are full SHAs, ambiguous/missing license fails, `Qwen/Qwen2.5-3B` is rejected before any request, branch names are resolved once, and a denied repo produces `no_access` without retry workarounds. Test metadata output excludes token marker `hf_secret_marker`.
- [ ] Run `python -m pytest tests/lab/test_pins.py -q`; expected missing `lab.pins`.
- [ ] Implement metadata-only authenticated pinning and license hashing; create reviewed pin registry without fetching weights. If a candidate does not exist or cannot be supported by installed transformers without remote code, report that concrete input issue before replacement.
- [ ] Agent command: `python -m lab.cli inventory --pin --config lab/configs/bases.json --data-config lab/configs/data.json --output "$ZOO_DATA/lab/inputs/pins.json"`. Review metadata, then copy only portable pin facts into `lab/pins.json` and materialize base table rows. No token values or raw cards enter git.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): pin permissive bases and training inputs`.

### Task 3: Reproducible LoRA recipes and complete effective checkpoints (L5)

**Files:** Create `lab/training.py`, `lab/scripts/lora.py`, `lab/configs/lora.json`; test `tests/lab/test_lora.py`.

**Interfaces:** `train_lora(recipe: dict, inputs: dict, output: Path) -> None`; `merge_adapter(base_state: dict[str,Tensor], adapter_state: dict[str,Tensor], config: dict) -> dict[str,Tensor]`. Training defaults: 128 optimizer steps, effective batch 16, sequence length 512, LR `2e-4`, AdamW, constant scheduler, zero dropout; targets `q_proj,v_proj` where supported and explicitly listed equivalents for OLMo. Resolve actual module names from pinned configs/model skeletons; unknown names fail before training.

- [ ] Create a tiny locally instantiated decoder (2 layers, hidden 32, vocab 64) and eight deterministic token sequences. A two-step rank-8 run changes only declared adapter tensors before merge. Assert effective merged matrix equals `W+(alpha/r)*B@A`, stored weights are bf16, tied embeddings reload, and `lineage.json` says `lora` with exactly its base. Tests override batch/sequence/steps, never rank-dependent scientific assertions.
- [ ] Run `python -m pytest tests/lab/test_lora.py -q`; expected missing script/module.
- [ ] Implement training with safe serialization and bounded local caches. Save adapter factors for provenance under the child directory, but make the model's primary file set a merged full checkpoint for common-shape comparisons. Mark saved secondary factors in recipe metadata so extractor file-set selection avoids double counting.
- [ ] Tiny: `python -m lab.scripts.lora --config lab/configs/lora.json --lab-root "$ZOO_DATA/lab" --tiny`. Assert two steps complete and the resulting checkpoint reloads locally.
- [ ] `[HUMAN] launches`: `python -m lab.scripts.lora --config lab/configs/lora.json --lab-root "$ZOO_DATA/lab" --resume`. Queue with actual frozen pins/config hashes and estimates below, not before inputs are frozen.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): add rank-stratified LoRA generation`.

### Task 4: Full SFT and DPO continuations (L6)

**Files:** Create `lab/scripts/sft.py`, `lab/scripts/dpo.py`, `lab/configs/sft.json`, `lab/configs/dpo.json`; modify `lab/training.py`; test `tests/lab/test_full_tuning.py`.

**Interfaces:** `train_sft(recipe: dict, inputs: dict, output: Path) -> None`; `train_dpo(recipe: dict, inputs: dict, output: Path) -> None`. SFT: full parameters, 256 steps, batch 16, length 512, LR `2e-5`. DPO: full parameters, 128 steps, beta `0.1`, batch 8, length 512, LR `5e-6`, frozen reference equal to the corresponding SFT parent. Both cap base size at 1.7B and retain deterministic sample IDs.

- [ ] Tiny two-step tests assert a selected non-adapter matrix changes for SFT and DPO, DPO leaves its reference model unchanged, parent edge is SFT→DPO, and any configured base above 1.7B is rejected before loading. Reject a training/evaluation sample-ID overlap.
- [ ] Run `python -m pytest tests/lab/test_full_tuning.py -q`; expected missing SFT/DPO modules.
- [ ] Implement safe full-checkpoint writes and explicit data/template preprocessing. A zero-step dry run is never counted as a generated lab model.
- [ ] Tiny commands: `python -m lab.scripts.sft --config lab/configs/sft.json --lab-root "$ZOO_DATA/lab" --tiny` and equivalent `lab.scripts.dpo` with `lab/configs/dpo.json`.
- [ ] `[HUMAN] launches`: the same two commands without `--tiny`, with `--resume`, in SFT then DPO order. DPO cannot start from the unmodified pretrained base by accident.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): add bounded full SFT and DPO chains`.

### Task 5: Mergekit recipes, provenance, and 30 merges (L4)

**Files:** Create `lab/merges.py`, `lab/scripts/merge.py`, `lab/configs/merge.json`, generated `lab/configs/mergekit/*.yaml`; test `tests/lab/test_merges.py`.

**Interfaces:** `mergekit_configs(inventory: dict, pins: dict) -> dict[str,str]`; `run_merge(recipe: dict, lab_root: Path, output: Path) -> None`. YAML refers to runtime-resolved local parent directories; the portable source recipe records model IDs only. Generated path-bearing invocation files are transient and never stored as artifacts.

- [ ] Tests assert exactly 30 configs, all four methods, parent counts `{2,3,4}`, at least one nonconstant layerwise `t`, and reference-base roles. Synthetic two-parent linear merge must equal the configured weighted state; SLERP must preserve the chosen endpoint at `t=0`/`1`; TIES/DARE tiny outputs must match mergekit on identical synthetic inputs and configured seed. A cross-shape merge fails before launching mergekit.
- [ ] Run `python -m pytest tests/lab/test_merges.py -q`; expected missing `lab.merges`.
- [ ] Implement mergekit calls as argument arrays (no shell interpolation), safetensors-only inputs/outputs, config hashes, and missing-parent diagnostics. Preserve all reference-base edges without claiming configured nonlinear weights are exact linear truth.
- [ ] Tiny: `python -m lab.scripts.merge --config lab/configs/merge.json --lab-root "$ZOO_DATA/lab" --tiny`; use 2-layer random local checkpoints, not downloaded models.
- [ ] Real run: same command without `--tiny` and with `--resume`. Agents may execute a genuinely bounded selected recipe through `--model-id ID` after measured tiny/Smol timing projects ≤2 hours and ≤100 distinct source models. The estimated complete 30-merge batch below is `[HUMAN] launches` if it exceeds that time; no silent fragmentation.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): generate hashed mergekit truth recipes`.

### Task 6: Classic and Heretic abliterations (L7)

**Files:** Create `lab/abliteration.py`, `lab/scripts/abliteration.py`, `lab/configs/abliteration.json`; test `tests/lab/test_abliteration.py`.

**Interfaces:** `classic_edit(state: dict[str,Tensor], directions: dict[int,Tensor], modules: dict) -> dict[str,Tensor]`; `run_abliteration(recipe: dict, inputs: dict, output: Path) -> None` dispatches `classic` or installed, pinned Heretic. Classic collects mean residual differences on disjoint direction data, normalizes r, and edits residual-writing matrices by `(I-r*r.T)W`; include relevant biases. Heretic keeps its actual per-layer/multidirection settings and optimization trace in recipe metadata; do not relabel a repeated classic edit as Heretic.

- [ ] On a synthetic residual writer, assert projection removes the r component (`norm(r.T@W_new) < 1e-6` in fp32), leaves orthogonal directions unchanged, and processes every configured writer. Use an injected fake Heretic runner to verify exact selected recipe/config and produced provenance; the adapter rejects missing safe-output support. Add local installed-package integration with a tiny model when implementing the actual adapter; if upstream cannot process it, document the measured restriction rather than weakening the core projection test.
- [ ] Run `python -m pytest tests/lab/test_abliteration.py -q`; expected missing `lab.abliteration`.
- [ ] Implement classic and a thin installed-Heretic adapter. Check installed version's supported CLI/API and pin its actual dependency before writing the adapter; invoke argument arrays, use local inputs, disable remote code and unpickling, and restrict optimization to the frozen transform-training slice. The validation slice only reports behavioral drift; no test/blind optimization. Pin its algorithm/config identity in outputs.
- [ ] Tiny: `python -m lab.scripts.abliteration --config lab/configs/abliteration.json --lab-root "$ZOO_DATA/lab" --tiny`; test both dispatches and all artifact checks. Run a real Smol smoke for each algorithm before queuing the four-base expansion. A failure on pretrained bases is reported; no refusal-quality claim follows from a rank test.
- [ ] Full eight-output batch is `[HUMAN] launches` under the provisional estimate; queue exact frozen config and parent pins.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): add classic and Heretic abliteration recipes`.

### Task 7: Quantize/dequantize simulations (L8)

**Files:** Create `lab/quantization.py`, `lab/scripts/quantization_sim.py`, `lab/configs/quantization_sim.json`; test `tests/lab/test_quantization.py`.

**Interfaces:** `simulate_quantization(weight: Tensor, bits: int) -> Tensor`; `run_quantization(recipe: dict, inputs: dict, output: Path) -> None`. For a matrix row, `qmax=2**(bits-1)-1`, scale=max(abs(row))/qmax, ties-to-even rounding/clamping to `[-qmax,qmax]`, then dequantize; zero rows remain zero. Quantize 2D floating matrices, preserve other tensors and tied aliases; save all floating checkpoint weights as bf16. This intentionally simple signed simulation is not GPTQ/AWQ equivalence.

- [ ] Tests assert hand-computed values for a small 4-bit row, zero-row behavior, rejection of unsupported bits, bf16 output, unchanged nonmatrix tensors, safe reload and tied embeddings, and `quantization_sim` lineage with correct DPO/base parent.
- [ ] Run `python -m pytest tests/lab/test_quantization.py -q`; expected missing module.
- [ ] Implement tensor streaming without a packed library. Record quantizer, bits, scaling axis, input hashes, and relative weight change; output format remains plain safetensors bf16.
- [ ] Tiny: `python -m lab.scripts.quantization_sim --config lab/configs/quantization_sim.json --lab-root "$ZOO_DATA/lab" --tiny`. Full batch command drops `--tiny` and adds `--resume`; agents may launch only after wall-time projection satisfies D4, otherwise queue `[HUMAN] launches`.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): add bf16 quantization simulations`.

### Task 8: Public laundering transforms (L9)

**Files:** Create `lab/transforms.py`, `lab/scripts/laundering.py`, `lab/configs/laundering.json`; test `tests/lab/test_transforms.py`.

**Interfaces:** `transform_state(state: dict[str,Tensor], model_config: dict, recipe: dict) -> dict[str,Tensor]`; `check_equivalence(original, transformed, token_batches: list[Tensor]) -> dict`. Explicit family adapters for these lab bases only; unsupported tensor/layout combinations raise `NotImplementedError` naming the module.

Permute SwiGLU MLP intermediate units consistently across gate/up rows and down columns. Rescale MLP up rows by powers of two with inverse down columns, leaving nonlinear gate rows alone. Rotate value-head channels using seeded orthogonal matrices and inverse output-projection blocks, consistently across GQA groups. Do not apply arbitrary residual-space rotation across RMSNorm, RoPE, or nonlinearities. Targeted perturbation adds a seeded rank-1 update with configured relative Frobenius norm `1e-3` to four selected projection matrices; it is a deliberately bounded perturbation, not function-preserving or an assumed successful evasion.

- [ ] Synthetic fp32 decoder tests assert MLP-permutation/rescaling/value-rotation logit agreement at `atol=2e-5, rtol=2e-5`, including a GQA case. Record bf16 agreement separately with `atol=2e-2,rtol=2e-2` on the same tiny fixture; these tolerances are test requirements, not a scientific equivalence guarantee on all bases. Verify targeted relative norm `1e-3` within `1e-6` before bf16 casting, and report post-cast norm rather than asserting the request survived quantization. Reject unsupported rotations before writing.
- [ ] Run `python -m pytest tests/lab/test_transforms.py -q`; expected missing module.
- [ ] Implement transforms, seed preservation, and recorded fp32/bf16 drift on frozen validation tokens. Store transform/algebra details and measured deviations in `recipe.json`; mark only mathematical symmetries as function-preserving in exact arithmetic. Failed real-model equivalence checks are failures, never silently accepted as new successful examples.
- [ ] Tiny: `python -m lab.scripts.laundering --config lab/configs/laundering.json --lab-root "$ZOO_DATA/lab" --tiny`. Full 16-transform public batch is provisionally `[HUMAN] launches` because of aggregate runtime; bounded individual integration recipes may be agent-sized.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): add audited public laundering transforms`.

### Task 9: Counted multi-step chains (L10)

**Files:** Create `lab/chains.py`, `lab/configs/chains.json`; test `tests/lab/test_chains.py`.

**Interfaces:** `chain_paths(models: pd.DataFrame, edges: pd.DataFrame) -> list[list[str]]`; `verify_minimum(models: pd.DataFrame, edges: pd.DataFrame, completed_ids: set[str]) -> dict`. Return counts, missing IDs, depth paths, and PASS/FAIL for each checklist minimum. Report the public laundering count against 16, and the combined G1 target of 20 as pending C attestation; a public-only report never declares that combined target passed. Reference-base merge edges are excluded when counting transformation depth; DAG ancestry still includes them for scorer truth.

- [ ] Tests assert the three base→SFT→DPO→quantized paths have depth 3, absent intermediate completion fails, no standalone duplicate “chain output” is counted, and a merge's base edge does not create fictitious extra depth.
- [ ] Run `python -m pytest tests/lab/test_chains.py -q`; expected missing module.
- [ ] Implement verification using existing outputs only. `python -m lab.cli verify --models lab/models.csv --edges lab/edges.csv --lab-root "$ZOO_DATA/lab" --minimum` prints all thresholds including failures and does not launch missing jobs.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): verify genealogy chains and minimum counts`.

### Task 10: Six decision heads and embedded adapter factors (L14)

**Files:** Create `lab/scripts/decision.py`, `lab/configs/decision.json`; modify `lab/training.py`; test `tests/lab/test_decision.py`.

**Interfaces:** `train_decision(recipe: dict, inputs: dict, output: Path) -> None`. Full sequence-classification SFT on five recipes, LoRA rank 8 on q3-06 for the embedded-factor recipe. Two labels, 32 steps, batch 16, max length 128, LR `2e-5` full/`2e-4` LoRA. Each records label names and config's classification architecture; four decoder outputs and two ModernBERT outputs.

- [ ] A tiny encoder and tiny decoder produce `[batch,2]` logits and a `[2,hidden]` classifier. Assert ordinary recipes have `sft` truth and embedded recipe `lora` truth. Embedded full checkpoint retains frozen base plus PEFT `lora_A/lora_B` factors, head, and enough adapter scaling metadata for the installed model loader to reconstruct the effective model safely; effective logits match its live PEFT model. Extractor must expose head and lora-product views when integration is available.
- [ ] Run `python -m pytest tests/lab/test_decision.py -q`; expected missing script.
- [ ] Implement safe classifier loading, explicit pad/pooling configuration, and serialization. Embedded factors are in the full checkpoint's selected file set, not a separate adapter-only repo; `recipe.json` states α/r and target modules. Preserve known scale metadata instead of creating an intentional unknown-scale fixture.
- [ ] Tiny: `python -m lab.scripts.decision --config lab/configs/decision.json --lab-root "$ZOO_DATA/lab" --tiny`. Agents may run the full six-job command with `--resume` after a smoke projects <2 hours; expected minutes per job. Queue it if measurements exceed that bound.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): add decoder and encoder decision-head examples`.

### Task 11: Hard-negative registry (L11)

**Files:** Create `lab/negatives.py`, `lab/configs/negatives.json`; test `tests/lab/test_negatives.py`.

**Interfaces:** `resolve_independents(config: dict, *, api, download_text: Callable) -> list[dict]`. Resolve the actual published PolyPythias five-size/nine-seed roster from its authoritative metadata, pin each selected repository, and record independence evidence in the portable pins registry. Do not fabricate repository names from guessed naming patterns. Same-series siblings are two distinct published same-architecture post-training branches with a documented common root and no ancestor relationship; select them from authoritative lineage metadata and mark that evidence separately in pin metadata. Consecutive training-step checkpoints are ancestors, not sibling negatives.

- [ ] Fake authoritative roster tests assert 45 unique `(size,seed)` references, full pins/license evidence, independent rows have no exact-truth edges, duplicate/missing seeds fail, and siblings are excluded as negatives for `same_root` while usable for immediate-parent discrimination when nonancestry is known. Different architectures are never weight-comparison negatives.
- [ ] Run `python -m pytest tests/lab/test_negatives.py -q`; expected missing module.
- [ ] Implement metadata-only pinning and final L2 table materialization. If published same-series references have a known ancestor relation, record that truth in a separate evaluation-label sidecar, not false edges in the lab construction graph. Metadata status and reasons remain explicit.
- [ ] `python -m lab.cli inventory --pin-negatives --config lab/configs/negatives.json --output "$ZOO_DATA/lab/inputs/negative-pins.json"`; review portable metadata then update `lab/pins.json` and tables. No weights needed; A fingerprints references under L12.
- [ ] Rerun task file/package tests. Suggested message: `feat(lab): pin independent seed and sibling references`.

### Task 12: Extractor handoff and benchmark scorer (L12 helper, L13)

**Files:** Create `lab/export.py`, `lab/score.py`, `lab/configs/scoring.json`; tests `tests/lab/test_export.py`, `tests/lab/test_score.py`.

**Interfaces:** `export_models(models: pd.DataFrame, lab_root: Path, destination: Path) -> None`; `score_lab(models: pd.DataFrame, edges: pd.DataFrame, predictions: pd.DataFrame, manifests: list[dict], universe: pd.DataFrame, *, threshold: float | None=None) -> dict`. `universe` requires `query_key,candidate_key,eligible` and explicitly defines all evaluated query/candidate pairs. Optional evaluation-only columns `truth_parent,truth_ancestor,truth_same_root` are nullable booleans, with `truth_tier` equal to `lab_exact`, `verified_hub`, or `declared`. Task 11's sibling sidecar is joined into these optional labels before scoring; conflicting explicit labels and known-complete construction truth raise. Isolated external reference rows are not a complete construction DAG: their root/ancestor labels require certified evidence, so an absent edge does not assert independence. Unknown external labels remain unscored for that estimand, with counts reported; rows with `eligible=False` remain in coverage denominators but cannot be given positive scores. Duplicate universe pairs fail. Predictions use the zoo's exact `LINEAGE_COLUMNS`, imported when committed Model-Zoo is installed. Higher score always means stronger relation.

Exporter columns are `repo_id,revision`: runtime absolute path + empty revision for local checkpoints, Hub ID + its pinned SHA otherwise. This is an **invocation-only CSV**, placed under caller-provided temporary scratch and removed after manifest extraction; it is not a portable stored artifact. Manifests resolve local model IDs by their `repo_id = lab/<model_id>` and pin keys. Missing/duplicate manifest mappings raise, and no path-based guesses are allowed.

For the complete generated public DAG, truth labels use direct edges for `parent`, strict transitive closure for `ancestor`, and intersecting construction roots for `same_root`; self-pairs are excluded. Merge reference-only `base` edges contribute ancestry/root context but are excluded from immediate-parent positives. In known-independent references, roots are independent; sibling root/ancestor labels require the explicit sidecar in Task 11. Never turn merely absent Hub declarations into negative truth. Score lab-exact, verified-Hub, and declared tiers separately; the latter two require an explicitly labeled external universe and never use this complete-lab closure by default.

For each estimand, missing predictions in the eligible declared universe get score `-inf` (abstention), not dropped rows. Collapse duplicate claims only when all values agree; conflicting duplicates fail. AUROC uses midranks for ties; compute the highest score threshold reaching recall ≥0.9 across positives and count all false matches at that threshold, including ties. Divide by **all query IDs** in that estimand's universe. Also report the attained TPR, threshold, positive/negative/query counts, eligible fraction, scored fraction (coverage), and abstention fraction. `threshold` additionally gives metrics at a frozen development threshold; do not choose a test threshold for operational claims. An undefined AUROC/TPR is JSON null plus `undefined_no_positives`/`undefined_no_negatives`, never NaN or a perfect score.

Direction accuracy is correct higher-scored direction / paired directed ancestor-or-parent examples with both directions scored; ties are wrong. Report direction availability separately; same_root has no direction metric. Lab→zoo edge types: lora→adapter, sft/dpo→finetune, quantization_sim→quantized, others unchanged; laundering is evaluated for relation recovery, not type correctness.

- [ ] Export test maps a Hub base and local child to the exact CLI rows; confirm persisted manifests keep portable IDs, not absolute checkpoint paths. Reject path escapes and missing pins.
- [ ] Scorer fixture: b→c→d and independent i, with explicit universe for d including b,c,i and reverse pairs. Perfect complete scores give AUROC 1 for all defined estimands and false matches/query 0; d→b is ancestor-positive and immediate-parent-negative. Reverse the d/c scores and assert direction accuracy falls. Omit d→c and assert reduced coverage plus positive abstention is retained; make all positive scores tied at `-inf` and assert false matches are not hidden. Empty-class tests return null/status and stable denominators. Check merge-reference exclusion and same-root sibling inclusion.
- [ ] Run `python -m pytest tests/lab/test_export.py tests/lab/test_score.py -q`; expected missing `lab.export`/`lab.score`. During fixes run only the failing file.
- [ ] Implement deterministic scorer/report JSON with atomic writes. No new model inference or calibration fitting in the scorer. Import `LINEAGE_COLUMNS` from `zoo.schema`; do not add fields to that contract. Per-example eligibility and truth metadata live in separate reports/configs.
- [ ] Agent command: `python -m lab.cli score --models lab/models.csv --edges lab/edges.csv --predictions "$ZOO_DATA/runs/p1/predictions.parquet" --manifests "$ZOO_DATA/runs/lab/manifests" --universe "$ZOO_DATA/runs/p1/universe.parquet" --config lab/configs/scoring.json --output "$ZOO_DATA/runs/p1/lab-scores.json"`. Inputs must already exist; this does not create fake evidence.
- [ ] A owns `[HUMAN]` L12: T2, row blocks 128/256/512, k=128, parents, all experts. Hand it the invocation CSV and portable truth tables. Never start whole-lab fingerprinting from this task.
- [ ] Rerun both task files, then package tests once. Suggested message: `feat(lab): export extractor inputs and score nested lineage estimands`.

### Task 13: Runbook and local benchmark release staging

**Files:** Create `lab/README.md`, `lab/release.py`, `lab/configs/release.json`; test `tests/lab/test_release.py`.

**Interfaces:** `stage_release(models: pd.DataFrame, edges: pd.DataFrame, pins: dict, source_root: Path, output: Path, *, include_weights: bool=False) -> dict`. Stage under `$ZOO_DATA/releases/lab-v1`, never publish. Include schema/version, scorer, source/config hashes, environment pins, provenance, known limitations, counts, license attribution, fixed query universe/split protocol, and regeneration commands. Model data remains opt-in and license-reviewed; no parent raw blocks/caches are included.

- [ ] Synthetic staging test asserts stable file hashes across repeated builds, no absolute paths, no cache/raw-row-block files, no blind data discovery, and explicit refusal if an output directory would overwrite unrelated data. Verify a missing checkpoint remains a visible incomplete row in the count report, not silently omitted.
- [ ] Run `python -m pytest tests/lab/test_release.py -q`; expected missing `lab.release`.
- [ ] Implement staging, a dataset-card draft, and the runbook below. Document approximate preservation, simulated quantization scope, incomplete quality guarantees, parent/reference semantics, and the separate blind evaluation managed by C. Freeze near-duplicate groups before root-held-out splits; a connected merge spanning roots stays wholly in one partition or is marked excluded, never split across train/test.
- [ ] `python -m lab.cli release-stage --models lab/models.csv --edges lab/edges.csv --pins lab/pins.json --lab-root "$ZOO_DATA/lab" --output "$ZOO_DATA/releases/lab-v1" --config lab/configs/release.json`. Default excludes weights and hidden examples. A person decides final inclusion and all uploads/publication at the May benchmark release.
- [ ] Rerun task file/package tests. At the milestone run the full Zoo-Keeper suite once. Suggested message: `docs(lab): add reproducible runbook and benchmark staging`.

## Run estimates and launch queue recipe

These are planning allowances, not measurements. Assume bf16 weights, one accelerator, cached inputs, typical 0.36–1B training anchors, and effective short sequence lengths above. Total trainable-anchor parameters are about 2.46B (0.5+0.6+0.36+1), so a full copy of all four is roughly 4.9 GB. Six LoRA-derived merged copies per anchor are about 30 GB; other descendants add copies. Reserve **1–2 TB durable lab space** for checkpoints, all provenance, independent references, staging, and headroom, plus **100 GB bounded disposable cache**; actual CSV inventory byte totals replace this allowance after pins and headers. No 7B full training is included. A0 currently reports about 507 GB free (see `handoff/STATUS.md`); the 1–2 TB allowance is not verified capacity. Before launches, reconcile the pinned inventory byte estimate and retention plan with durable free space; do not launch a batch whose outputs cannot fit.

| Batch | Weight bytes read / new storage, provisional | Wall time, one worker | GPU-hours, provisional | Launcher |
| --- | --- | --- | --- | --- |
| Base/reference fetches | 30–150 GB transfer / same cached; exact PolyPythias roster determines bound | 0.5–4 h | 0 | person if >2 h; metadata pinning agents |
| 24 LoRA + full merges | ~30 GB local reads / ~30–40 GB outputs incl adapters | 40–80 h | 40–80 | `[HUMAN]` |
| 8 SFT + 8 DPO | ~40–80 GB local reads / ~20–30 GB checkpoint outputs | 70–130 h | 70–130 | `[HUMAN]` |
| 30 mergekit merges | ~80–160 GB local reads / ~30 GB | 2–6 h CPU | 0–2 | `[HUMAN]` full batch if >2 h |
| 8 abliterations | ~10–20 GB reads / ~10 GB outputs | 16–40 h | 16–40 | `[HUMAN]` full batch |
| 12 quantization simulations | ~15 GB reads / ~15 GB outputs | 0.5–2 h CPU | 0 | agents if measured projection ≤2 h |
| 16 public transforms + validation | ~20 GB reads / ~20 GB outputs | 3–12 h | 3–12 | `[HUMAN]` full batch |
| 6 decision heads | ~7 GB reads / ~7 GB outputs | 0.3–1.5 h | 0.3–1.5 | agents if measured projection ≤2 h |
| Scoring/release metadata | existing fingerprints/tables only / <1 GB initially | <1 h | 0 | agents |
| L12 lab/reference T2 | full pinned weights once; all ~166 public entries plus C's separately managed blind entries, exact bytes after headers | A's measured SVD projection | CPU first | `[HUMAN]`, Track A |

Training/transform estimates total roughly **130–266 GPU-hours**; keep the research allowance of 150–300 GPU-hours including retries. Do not force measured numbers into that range. If projected runtime/storage exceed allowances, record actual evidence and queue the run; shrinking the lab below G1 minimums needs a person.

After implementation and smoke timing, each queued batch entry in main `handoff/STATUS.md` must contain the exact `python -m lab.scripts.<type>` command from its task with `--resume`, the config SHA256, parent pins/local manifest keys, data pins/sample hash, expected bytes/time/GPU-hours/storage, output `$ZOO_DATA/lab/<model_id>`, verification `python -m lab.cli verify ... --minimum` (or selected model checks before all batches finish), and the same resume command. Until actual pins/configs exist, these are prepared runbook templates, not runnable frozen jobs.

## After this plan

The person reviews L1; implementation then proceeds on `track-b`, uncommitted. L2 can begin with synthetic fixtures independently of extractor code; finish complete table materialization after L3/L11 pins. Generate L4 configs early and queue L5/L6 as soon as their safe tiny runs and frozen inputs are ready. Real merges wait for trained same-base parents. Finish public transforms and chains, then coordinate L12 with A without reading C's hidden set. The scorer/import integration waits until required `evidence`/`zoo` code is committed and merged into Model-Zoo main by the person; use the frozen interfaces to plan meanwhile.

No new research decision is required before implementation. Concrete input failures (license ambiguity, missing published pin, package incompatibility, unsupported safe Heretic outputs) are surfaced with evidence at their owning task. G1, pivots, publication, and large launches remain the person's actions; this plan does not waive them.
