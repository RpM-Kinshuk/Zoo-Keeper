# Incremental L12: small real cohort

Owner priority, 2026-11-01: verified real sources, ready L8/L9 derivatives,
evaluation harness and first scoped inference/scoring report. Release staging
is deferred. A retains ownership of whole-lab L12; this bounded incremental
pass does not claim L12 or G0 completion.

B owns generation and the first local T1 pass under
`$ZOO_DATA/runs/track-b-real-cohort-v1/`, avoiding A's existing smoke and future
whole-lab output directories. Do not duplicate that T1 work. The atomic
`ready.json` will list completed source/output rows with their verified pins,
paths, parents and hashes; only those rows enter scoped tables.

The targeted control family is Pythia-14M seeds1–4 with q8/q4 derivatives:
seed1/2 are development roots; seed3/4 are held-out test roots. Roots and all
derivatives stay in one partition. Safetensors availability is a prerequisite;
pickle loading is forbidden. Smol/OLMo transforms are separate descriptive
strata until comparable certified controls exist. The pre-score protocol is
`predeclared-protocol.json` in that run directory; it freezes the illustrative
0.75 development threshold before any cohort scores are observed.

Incremental evidence is written to `evidence/manifests/` and `evidence/t1/`
within this run. B uses the committed Model-Zoo extractor at cda3705, local
verified sources only, k64 and T1 block256, with an explicit parent manifest
for each derivative. Each completion records model identity, input/output
hashes, selected bytes and elapsed time in `evidence/progress.json`.

A can consume the ready scoped `models.csv`, `edges.csv`, manifest directory
and cost/progress records directly through the frozen file contract. A's next
increment should add T2 and the prescribed free T1 blocks128/256/512 at k128
in a separate output directory after a measured runtime projection; first
measure one small source and one derivative. Preserve the pinned input list
and provenance, and coordinate writer ownership here before launching. B will
not label missing T2 or5k-pool measurements as successful gates.

Exact ready counts, invocations and final report paths are appended as the
cohort completes. No blind paths are required or inspected.

## Completed and ready for A

B completed T1 k64/block256 on **22 real models** across two scoped cohorts:

- `runs/track-b-real-cohort-v1`: Smol135/OLMo sources and eight derivatives;
  current report `REPORT-v2.md` (earlier report retained).
- `runs/track-b-safe-control-cohort-v1`: four independent Pythia14M seeds,
  each with q8/q4 derivatives. Existing official safetensors conversion PR1
  commits were pinned and verified after main lacked safe files. Seeds1/2 are
  development; seeds3/4 are held-out roots. Current report `REPORT.md`.

Each has `scoped/models.csv`, `scoped/edges.csv`, `ready.json`,
`evidence/manifests/`, `evidence/t1/`, `evidence/progress.json` and frozen
evaluation outputs. The control report's two held-out roots yield12 positive
and18 independent-negative same-root pairs; AUROC1.00, TPR1.00 and0 false
matches at the unchanged0.75 threshold. Parent/ancestor direction abstains;
neither error control nor G0/P1 completion is claimed.

A's exact two-model launch is
`$ZOO_DATA/runs/track-b-safe-control-cohort-v1/incremental-A-l12-launch.json`.
It uses **only committed Model-Zoo code** and does not depend on B's helpers. Its input CSV is invocation-only at
`$ZOO_DATA/cache/track-b-incremental-l12/models.csv`, with a public pinned
seed1 source and the actual local q4 derivative. Source/CSV hashes and exact
argument arrays are recorded in the launch JSON. From the Model-Zoo checkout
with its collector sources importable and workspace environment loaded:

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
python -m evidence.cli manifest \
  --models "$ZOO_DATA/cache/track-b-incremental-l12/models.csv" \
  --out "$ZOO_DATA/runs/track-b-safe-control-cohort-v1/incremental-A-t2/manifests" \
  --include_declared_bases --workers 1
python -m evidence.cli batch \
  --models "$ZOO_DATA/cache/track-b-incremental-l12/models.csv" \
  --manifests "$ZOO_DATA/runs/track-b-safe-control-cohort-v1/incremental-A-t2/manifests" \
  --tier t2 \
  --out "$ZOO_DATA/runs/track-b-safe-control-cohort-v1/incremental-A-t2/fingerprints" \
  --workers 1 --sketch_k 128 --all_experts \
  --also_t1_blocks 128,256,512 --with_parent
```

This launch is **prepared, not started by B**. A should claim writer ownership
here before executing, measure the source first and keep the projected job
below two hours. It reads two distinct weight models (about106.6MB stored
source files); conservative provisional read allowance2GB, planning time1–30
minutes CPU,0 GPU hours. Full T2 timing is not yet measured. Rerun the exact
commands to resume matching upstream artifacts; verify both complete summary
rows, T2/H5 and all three T1 block outputs before extending the batch. Keep
this k128 output separate from B's frozen k64 evidence. Network may be needed
for the public pinned parent; no pickle file is selected or loaded.
Compare completed manifest/fingerprint identities with `expected_model_keys`
in the launch JSON, not inferred filesystem names. Use the existing workspace
token environment without displaying it. Delete the invocation-only cache CSV
after its manifest handoff has completed.

## GPU training follow-up: primary checkpoint selection (2026-11-02)

**Resolved and merged,2026-11-02 (Codex A):** Model-Zoo main `00ad4af`
contains the reviewed R2/X5 repair. The exact reproducer below now selects only
`model.safetensors`, records saved factors as `bundled_adapter`, and preserves
a standalone nested adapter-only checkpoint. No duplicate fix is needed; B can
consume committed upstream code. The original six-model evidence/scoring freeze
and all trained files remain unchanged.

**Next writer/handoff:** A fingerprints the six pending LoRAs at existing
T1 k64/block256 settings, without repeating the six completed non-LoRA/source
outputs. A then measures the prepared two-model T2/k128 pilot in its separate
output directory above. Claim output ownership and freeze exact inputs before
launch; no such run was launched during integration. B consumes the six added
fingerprints for a new scoped evaluation; C uses identity-matched T2 spectra for
S5. Keep the original held-out controls and reports immutable. Whole-lab L12,
E6/E9, G0 and P1 remain incomplete.

The following describes the pre-integration reproducer and diagnosis.

Before this integration, the extractor selected both the merged checkpoint and the
provenance adapter factors written by B's LoRA trainer. The following pure
selector reproducer uses the installed committed Model-Zoo sources:

```python
from evidence.manifest import select_file_set
select_file_set([
    'model.safetensors', 'config.json', 'training.json',
    'adapter/adapter_model.safetensors', 'adapter/adapter_config.json',
])
# Before 00ad4af: both files selected.
# Now: (['model.safetensors'],
#       [{'filename': 'adapter/adapter_model.safetensors', 'reason': 'bundled_adapter'}])
```

`training.json` explicitly declares `primary_weights: ['model.safetensors']`
and `secondary_weights: ['adapter/adapter_model.safetensors']`, with the
secondary reason identifying provenance factors excluded from the effective
checkpoint tensor set. The selector does not currently consume these fields;
its filename heuristic also misses this nested adapter layout. Including the
factors contaminates parameter counts, tensor views/fingerprints, and
`is_adapter` classification for the merged full checkpoint.

A owns this **X5 file-selection follow-up needed by L12**. On2026-11-02,
B inspected A's existing uncommitted R2 repair in `worktrees/Model-Zoo-track-a`:
it already returns only `model.safetensors`, records nested factors as
`bundled_adapter`, and preserves a standalone nested adapter-only checkpoint.
The same two-case pure-function check passed. **No duplicate implementation is
needed:** A should finish its existing repair review/tests and commit/merge it
into Model-Zoo main, then resume the six LoRA extractions and queued incremental
T2/k128 pilot. B continues to use committed upstream code only.
The desired effective selection is only `model.safetensors`, with the nested
adapter weights recorded as excluded provenance. Preserve the immutable
trained output directories and their files. B will extract the base and
completed SFT, DPO, and decision checkpoints meanwhile; LoRA fingerprints wait
for A's committed selector correction.

There are also two different local content identities: the DPO launcher's
`local_parent.model_key` hashes its verified SFT provenance pin, while the
extractor's local revision hashes all files in the checkpoint directory.
They need not match. This does not block training: the trainer binds the
declared parent to the completed SFT's actual weight, config, recipe, lineage,
and training hashes. Evidence and scoped evaluation must use the actual
extractor identity, retaining the launch provenance identity separately;
neither should be substituted for the other. Local `lineage.json` bases are
portable `lab/<parent-id>` paths and require resolution to that verified parent
when constructing evidence tables.


## Additional completed training evidence,2026-11-02

`runs/track-b-training-gpu-v1/ready.json` records11 completed GPU-trained
outputs (2SFT,6LoRA,2DPO,1decision) plus the existing Smol135 source; scoped
12-model/11-edge verification PASS. B has completed T1 k64/block256 on
source+2SFT+2DPO+decision, with actual portable extractor keys, manifests,
matched-parent fingerprints and870,584,832 selected tensor bytes. Its exact
six-model tables, universe and report are under `evaluation/`; full training
tables are under `scoped/`. B writer is finished; no T2/GPU workers remain.
Do not duplicate these six T1 outputs. Six LoRA outputs remain untouched and
unfingerprinted pending the primary-only selector fix above. Union across all
three B runtime cohorts:33 unique real models,27 with T1. Original Pythia
held-out controls and A's separately queued two-model T2 command are unchanged.


## Next Track B batch after integration

First consume A's committed selector repair to fingerprint the six completed
LoRAs and refresh the scoped evaluation. In parallel, use existing SFT→DPO
outputs to prepare a scoped depth-three-chain experiment (L10), then measure
E6 tree/direction behavior and extend E9 with frozen development/test controls.
These extra Smol paths remain distinct from the formal lab minimums. Preparing
verified official-anchor weights and tokenizer-specific launch inputs is the
parallel route to complete the original L5/L6/L14 quotas. Full E9 still needs
A's T2 measurements, the larger reference pool and C's matched baselines.
Release staging remains deferred.


Integration checkpoint2026-11-02: B source merged locally as `f6f3e08`; full
CPU/offline suite494 PASS. L13/E8 complete; L12 remains A-owned and incomplete.
The GPU completion pointer is `lab/configs/launches/track-b-gpu-v1.json`.

## Active A execution,2026-11-02

Owner authorized six LoRA T1, two-model T2 and L12. A claims six missing T1
outputs and progress rows in the training run; the six prior rows and frozen
evaluation stay unchanged. T2 uses separate outputs; exact freeze/logs at
`runs/track-a-l12-20261102.mnbqfy_y`. GPU launch rule: utilization<80% and
occupied-memory fraction<80%, checked twice five seconds apart. Prepared
pilot source/derivative identities are retained. Full inventory coverage and
missing inputs will be reported separately; no duplicate B writer should run.

## B can consume six LoRA T1 outputs now

A completed all6 pending LoRAs (k64/block256, matched parent), preserving the
original6 evidence files. All12 source/training models now have evidence.
Actual keys/hashes plus12-model/11-edge tables: `runs/track-a-l12-20261102.mnbqfy_y/t1-training-12`.
B should freeze a new scoped evaluation output before refreshing scores; the
original six-model evaluation remains unchanged. A's two-model T2/k128 pilot
also passed;33-model full-spectrum extension is running separately. All54
formal external T0 manifests completed at original pins,42 T0-only.

## Completed A handoff,2026-11-02

All43 readable checkpoints are published at T2/k128, with full HDF5 spectra
and free T1 blocks128/256/512. This covers the original33-model union and10
additional exact-pin formal sources. All6 pending LoRAs also have T1 k64/b256;
the earlier6 artifacts and frozen evaluation remain unchanged.

- Primary report: `runs/track-a-l12-20261102.mnbqfy_y/REPORT.md`.
- Exact keys/manifest/T2/HDF5/freeT1 paths: `artifact-index43.csv` in that run.
- Observed construction truth: `construction-edges27.csv`; reference/sibling
  metadata does not invent immediate parents or certify independent roots.
- B's12-model training T1 inputs: `t1-training-12/`. Freeze a new scoped
  evaluation rather than overwriting the original six-model report.
- C can use matched spectra for S5. Published artifacts were SHA-verified.
- Formal166 coverage:15 measured,109 checkpoints not generated,42 original
  pins T0-only. Exact gaps: `formal-coverage166.csv`. B owns generation/source
  readiness; A retains incremental fingerprinting as new ready inputs arrive.

Optional source-weight cache copying was stopped to avoid delaying delivery;
partial caches are outside this completion contract. GPU option source is
reviewed/uncommitted on Model-Zoo track-a. All execution workers have finished.

Integration checkpoint,2026-11-02: optional CUDA spectrum support is now committed/merged in Model-Zoo main as `48a70fc`. All43 published artifacts and original evaluation freezes are unchanged. B/C may consume the published paths/keys; A continues with Z7 queries while L12 awaits the explicit remaining inputs.
