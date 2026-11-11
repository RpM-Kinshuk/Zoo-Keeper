# Technical contracts and measurement reference

Current operational commands live in this repository's README and docs/operations, and in Model-Zoo/docs/operations. This reference retains useful technical content from the retired handoff; CHECKLIST.md sets priorities and DIRECTIVES.md sets scientific/safety rules. Historical T0/T1/T2 labels in saved runs do not authorize replay against unpinned current source.

## Current names and compatibility

- Modes: `metadata_only`, `row_blocks`, `full_weights`. Use `--measurement_mode row_blocks|full_weights` and `--also_row_block_sizes` at extraction boundaries.
- Fingerprints: `.row_blocks.fingerprint.safetensors`, optional `_128`/`_512` row-block variants, `.full_weights.fingerprint.safetensors`; full spectra: `.full_weights.spectra.h5`.
- Model columns: `row_block_fingerprint`, `full_weight_fingerprint`, `full_weight_spectra`; the store also has `layers`.
- Evidence writer version3 reads1/2/3; zoo writer version2 reads0/1/2. Serialization version alone does not establish scientific compatibility. Preserve input identities, exact source revision/local hashes, row geometry, sketch seed/size and artifact hashes.
- Reuse `evidence.compatibility` (`canonical_mode`, `companion_spectra_path`, table normalization). Never rewrite an old source/checksum as current merely to pass validation.
- Build from explicit artifacts or selected package conventions. A package-default scan does not include arbitrary historical run trees. Keep stored-view spectra separate from package `esd/metrics.h5` and its layer indices; preserve spectrum-source and missing-parent/delta status.

## Stable interfaces

Changes require owner approval, a schema version bump and a short migration record with the result:

| Interface | Source of truth |
| --- | --- |
| Model key `repo_id@revision`, full40-character commit or `local:<sha256>` | Model-Zoo evidence identities and zoo schema |
| Artifact names from `safe_filename` and manifest/fingerprint layout | Model-Zoo/evidence serialization and compatibility helpers |
| Coverage statuses/reasons and decode×geometry rules | Model-Zoo/evidence code/tests |
| Calibration (`statistic`, `sketch_k`, `row_block`, `tier`, `n_pairs`, `null_quantiles`, `related_threshold`) | Existing calibration reader/writer; only the exact statistic/config it describes |
| `diff/check` JSON and verdict vocabulary | Existing API/code/tests |
| Lab models/edges and `LAB_SCHEMA_VERSION` | Zoo-Keeper/lab/schema.py and tracked table manifests |
| LINEAGE_COLUMNS and table versions | Model-Zoo/zoo schema |

## Measurement conventions

### Measurement modes (historical T0–T3 labels)

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

### 3.4 Block delta and the noise floor

The delta between two bf16 checkpoints is the update quantized to the bf16
grid. Below a relative update ρ ≈ 3e-3 it is mostly rounding, and a raw rank
estimate is wrong. The procedure:

1. All elements bit-identical: status `identical`.
2. More than half bit-identical: status `below_floor`, and no rank is reported.
3. Otherwise: biwhiten the delta with the parent's ulp²/12 profile (Sinkhorn),
   threshold at `1.1·(√rows + √cols)`, and report the rank with a gap-based
   confidence.

A rank-1 difference in one tensor is not a general edit detector. Classic shared-direction projections and recipe-specific Heretic changes need separate models/tests (CHECKLIST.md item1). Do not assume every directional edit has one shared direction.

### 3.5 The zoo store and queries

- Parquet tables (`models`, `views`, `coverage`, `lineage`, `scores`, `predictions`, plus current `layers`) and one retrieval matrix per architecture signature, in one directory with no database server.
- Rebuilt atomically and deterministically from the extractor's outputs.
- Queried through `Zoo(path)` over in-memory DuckDB.
- Weight similarity compares only models with the same architecture signature,
  which ignores vocabulary size.
- v0 lineage edges:
  - declared (from metadata);
  - verified (shared identical tensors);
  - detected (sketch similarity), labeled "uncalibrated" until the G0
    calibration file exists.


## Lab and data layout

Local lab paths live under `$ZOO_DATA/lab`. `models.csv` contains stable model_id, kind(base/derived/independent), path, base_family and license. `edges.csv` has child/parent, edge_type, role, coefficient and params_json; actual schema/validator and manifests are authoritative. Each constructed checkpoint's `lineage.json` binds its name, bases and relation; local keys are content-pinned. Lab lora maps to zoo adapter, SFT/DPO to finetune, quantization_sim to quantized. Laundering/disguise is scored as a limitation, not a robustness claim. Schema changes bump LAB_SCHEMA_VERSION.

`$ZOO_DATA/runs` preserves inputs/reports/protocols/evidence; builds are under `builds`; checkpoints under `lab`; any reserved final set stays separate with exposure tracked; cache contains scratch and internal parent blocks. Never discard scientific artifacts merely because an old task ended. DATA.md holds archive/restore details. Model-Research remains needed for existing Trace/Spectral imports; its old corpus is exploratory.

## Known traps

- BF16 updates below roughly3e-3 relative size are dominated by rounding; rank alone does not type an edit. Sketch rank is readable only to aboutk/4; use measured block deltas when justified.
- Quantization scales/indices are not matrices. Not every3D tensor is an expert stack; orient experts using hidden_size and honor fused/interleaved layouts. Do not drop tiny decision heads or poison modules because an integer buffer exists.
- FP8 block size comes from config; file sets may store duplicate weights. GGUF llama-family Q/K rows may be reordered and need alignment before cross-format comparison. Unsupported formats are explicit coverage gaps.
- Shared sketch operators need an empirical null, not a zero-centered theoretical assumption. Tensors and reversed model pairs are dependent. A calibration does not transfer automatically across statistic/sketch/family/config.
- Merges need deltas and sometimes a base term; they need not sum to one. Collinear sources are unidentifiable, so report groups/conditioning. TIES/DARE unchanged-share overlaps normal controls in existing data; do not tune that shortcut into a pass.
- Declared bases are noisy; distinct labels do not prove independence. Matching weight hashes does not prove equivalent benchmark settings. Score normalized columns may be clipped; use Raw values.
- Sampling-based prevalence requires inclusion probabilities and detector-error correction; targeted scaling pools do not estimate Hub prevalence. Report repository-weighted and download-weighted quantities separately where justified.
- Cross-architecture distillation is outside weight-coordinate lineage. Public-seed coordinate sketches can be evaded by permutation/targeted perturbations; no adversarial robustness claim.

Reproduction of old runs uses their pinned source/config/hash records, not an obsolete implementation plan or handoff queue. Preserve original negative/failed results; a new policy/test uses a new output directory.
