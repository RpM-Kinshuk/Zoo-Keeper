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


## B consumed delivery and extended the measured cohort,2026-11-02

B consumed A's six LoRA T1 outputs and all applicable ready-model T2/freeT1
artifacts through a new immutable evaluation at
`runs/track-b-l12-evaluation-v2`. The original six-model training evaluation,
original Pythia held-out report and all33 ready checkpoints remain untouched.
Training now scores12 models/11 edges; the overall evaluated union is36 models
and30 edges after three new scoped Smol135 DPO quantization endpoints. All36
ready models have T1/T2; global T2 includes A's extra10 formal sources:46 total.
The new endpoints and exact T2/HDF5/freeT1 paths are in
`runs/track-b-chain-cohort-v1/artifact-index3.csv`; only3 new models were
extracted, with inherited evidence reused. No A rerun is needed for them.

Held-out Pythia same-root AUROC is1.00 for T1 and T2; related-pair cosine
Spearman0.937063 over24 ordered related pairs. Threshold0.75 and root splits
are unchanged. The expanded20-node Smol tree diagnostic flags4193/4845(T1)
and4201/4845(T2) quartets at1e-6. Tolerance sensitivity is separately reported;
no thresholds were relaxed. Direction still abstains; nRFnull. These scoped
results do not decide the formal E6 pivot or pass G0/P1/full E9 gates.

B's new code passed one fresh core review/fix pass and full suite508 PASS,
0warnings,91.78s CPU/offline. Source changes remain uncommitted. Scientific
source readiness is now B-owned: exact official Smol360 source is promoted
and11 official-ID training configs prepared,9 ready/2 awaiting actual SFT.
The first formal SFTs0→DPOs0→q8 path is being executed in a separate run,
`runs/track-b-formal-smol360-launches-v1`; new formal outputs will be handed
off separately and do not change this36-model evaluation freeze. GPU admission
requiresboth<80% checks and48GiB free for that360M pilot. No additional A
code dependency is blocking B. Release staging remains deferred.


## New formal Smol360 chain ready for incremental A T2,2026-11-02

B completed the exact official chain
`smol-360 → smol-360-sft-s0 → smol-360-dpo-s0 → smol-360-quantization-s8`.
All3 new checkpoints are verified against the original roster, safely reloaded,
and T1/k64 measured. A's base Smol360 T1/k128 was reused with its k64 prefix.
Exact new keys, selected file hashes, paths and native settings:
`runs/track-b-formal-smol360-launches-v1/execution/model-id-map.json`.
No generation/training/extraction worker remains. Do not repeat these T1 jobs.

The new3 T2/k128/full-spectrum/free128/256/512 launch is PREPARED, NOT STARTED:
`runs/track-b-formal-smol360-launches-v1/incremental-A-l12-launch.json`.
It binds invocation and single-child pilot CSV hashes, expected actual keys,
exact manifest/batch argv, output directory, verified local source mirror,
GPU checks, verification and resume policy. Source mirror:
`lab/inputs/cohort-v1-smol-360-f8027fd0eaee` at the original public pin.
A should claim the new output writer, run the single-child pilot first, then
extend only if its full3-model projection stays below2h. Use paired≥5s
utilization<80%/memory<80% checks and≥24GiB free before selecting the GPU;
CUDA_VISIBLE_DEVICES supplies the physical UUID, compute_device uses cuda:0.
Only3 new derivative models enter the CSV; base metadata/parent reads are
permitted, but no base fingerprint should be recomputed. Invocation-only CSVs
are under `cache/track-b-formal-chain-l12`, not release tables.

This adds formalT1 coverage to18/166 while fullT2 remains15/166. Formal missing
generation is106 checkpoints;42 external pins remainT0-only. WholeL12 stays
in progress. No further upstream code change is blocking B. These new formal
outputs are separate from B's frozen36-model evaluation and its three extra
Smol135 chain endpoints, which already haveT1/T2 and need no A rerun.


## A claims the three formal Smol360 T2 outputs,2026-11-02

A is starting the prepared single-child pilot, then the remaining two only
if the full3-model projection stays below2h. Existing B T1 and the Smol360
base fingerprint are reused. GPU2 is released by B and excluded from A's
separate R3 farm; paired<80%/24GiB checks still apply. Verified source copies
and compute scratch stay under a private /tmp directory and are deleted
after publication. Progress/report/index at the prepared incremental-A-t2
path; no duplicate writer should launch.

## Three formal Smol360 T2 outputs delivered,2026-11-02

Completed in507.56s; all3 exact keys/parent pins verified. Each has225 views,
175,040 full eigenvalues, T2/k128 and free T1 blocks128/256/512. Full HDF5
spectra are retained. Selected tensor reads including parents total5,706,792,960
bytes; peak GPU allocation1,005,589,504bytes. Both GPU checks passed before
admission. The single-child pilot passed before extending; all workers exited
and private source/compute scratch was removed. No network reads, base/T1
rerun, source changes or evaluation overwrite.

At `runs/track-b-formal-smol360-launches-v1/incremental-A-t2`:

- `REPORT.md` / `A-report.json`: verification, timings, selected bytes and hashes.
- `artifact-index3.csv`: exact new keys and T2/HDF5/freeT1 paths.
- `formal-coverage166.csv`:18 fullT2,106 ungenerated,42 original-pin T0-only.

B can consume these3 in a new evaluation freeze. Original43-model A delivery,
B's prior scoped36-model evaluation and all standalone T1 outputs remain
unchanged. Full calibration still needs the remaining formal lab/reference
inputs.

## Owner paused the Hub pilot,2026-11-02

The separate R3/R4 pilot is halted; resume requires a new owner instruction.
Saved83 new T1 and85 new T2 completions, retaining85 full-spectrum HDF5 files.
Four T1 and4 T2 jobs were interrupted; one other T1 job had a short read.
Controllers, workers and report/build follow-up are stopped. Frozen inputs,
original results and completed artifacts are preserved. Exact identities and
508 artifact paths: `runs/track-a-r3-r6-20261102.rcsy95ap/PAUSE_REPORT.md`.
The three formal Smol360 outputs above remain complete and ready to consume;
formal fullT2 coverage remains18/166. No report/calibration/fullR6 gate claim.


## Nine more formal Smol360 outputs ready,2026-11-03

All remaining11-job-bundle training is now complete (nine new plus two prior).
The nine new checkpoints and T1/k64 fingerprints passed final audit; all61
prior frozen artifacts are unchanged. Scoped registry has13 models/12 edges.
New T1 selected2,599,329,280 bytes in102.34s; training wall788.19s, active661.07s,
maximum allocated23.08GiB. No new T2 was launched and no Smol worker remains.

A can claim the new9-child writer using
`runs/track-b-formal-smol360-launches-v1/resume-20261103/incremental-A-l12-launch.json`.
It binds exact child keys, existing manifest/index hashes, one-child SFTs1 pilot
and full9 commands, full HDF5/freeT1 blocks, original pinned source mirror,
paired GPU checks and <2h projection guard. Reuse all prior base/child T1/T2.
Formal coverage is now27/166 at T1-or-T2,18/166 at fullT2,97 ungenerated and42
external T0-only. Whole L12 remains in progress. Qwen two-GPU chains are in a
separate active run and will be handed off only after actual outputs verify.

## Qwen2.5 chain ready for incremental T2,2026-11-03

The exact q25-05→SFTs0→DPOs0→quantization-s8 chain is complete. Both training
jobs ran on two L40 GPUs through NCCL, unchanged global batches16/8 and full
256/128 steps; frozen DPO reference unchanged. Verified4-model/3-edge tables,
safe BF16 q8 reload and3 new T1 fingerprints are at
`runs/track-b-formal-q2505-launches-v1/execution/`. New T1 selected691,666,944
bytes in72.51s; A existing base/k128 is reused. Three-child T2 launch:
`runs/track-b-formal-q2505-launches-v1/incremental-A-l12-launch.json`,
PREPARED_NOT_LAUNCHED. It binds exact keys, manifests and CSV hashes, firstSFT
pilot and full3 batch; standard k128/full HDF5/freeT1 blocks and admission
policy apply. No base/T1 rerun. Together with Smol360 new9,12 children await A
writer claim; Qwen3 is still training and is not yet in this queue.

## All three prescribed chains complete;15 new children queued,2026-11-03

Qwen3 is now also complete:4-model/3-edge registry,256-step SFT,128-step DPO
with unchanged reference, official q8 safe BF16 reload and3 new T1 fingerprints.
New T1 selected883,949,568 bytes in65.04s; existing A base evidence reused.
Its exact three-child T2 launch is
`runs/track-b-formal-q306-launches-v1/incremental-A-l12-launch.json`,
PREPARED_NOT_LAUNCHED. No B workers remain; driver/launcher/rank PIDs absent
and both GPU locks released. CPU and training sessions exited zero.

Together the three incremental queues cover15 new formal children (9+3+3).
No base or completed B T1/T2 should be repeated. New report, exact queue paths,
checksums, minimum verdict and166-row coverage:
`runs/track-b-resume-20261103/{REPORT.md,summary.json,minimum.json,formal-coverage166.csv}`.
All3 prescribed depth-three paths are verified; whole public minimum remains
incomplete. Formal T1-or-T2 is33/166; fullT2 remains18/166,91 derivatives are
ungenerated and42 external inputs are T0-only. A needs these incremental
measurements; no A source/interface change is blocking B. Re-evaluate new
formal chains in a new freeze after T2; retain original36-model evaluation
and independent-root held-out control splits/thresholds unchanged.


## Ten official Smol360 merge outputs ready for incremental T2,2026-11-03

B completed10 official L4 recipes (8 unique primary-weight sets), safely
reloaded the models and available native tokenizers, verified a17-model/41-edge
scoped DAG, and measured10 raw T1 fingerprints in84.05s (2,736,128,000bytes).
Seven existing base/LoRA fingerprints and all construction truth are reused.
Exact duplicate groups are recorded; these are correlated recipe cases.

A can claim `runs/track-b-formal-smol360-merges-v1/incremental-A-l12-launch.json`
(PREPARED_NOT_LAUNCHED). It binds the exact10 keys, child/prior manifest metadata,
CSV/index hashes, linear05 pilot and full10 commands for k128/full HDF5/freeT1
blocks128/256/512. No base/LoRA/T1 rerun. The existing --with_parent CLI selects
the first declared parent; label that delta coverage explicitly, retain the
full merge DAG/reference, and never describe it as a unique-parent inference.

Together with the prior three queues,25 new children await A. Global formal
coverage is43/166 measured model IDs (18 fullT2 plus25 T1-only),81 ungenerated
and42 external T0-only. Updated proof/minimum/coverage and detailed E3 report
are under the new run directory. All10 individual E3 fits abstain under the
existing correlation guard despite full numerical rank; combined layer CIs
are available. No new calibration/direction/coefficient-recovery claim.
No B worker, GPU allocation or T2 run remains.
