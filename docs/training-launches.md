# Frozen training launch bundles

The extra GPU cohort completed on2026-11-02:11 real checkpoints,3,379,938,264 output-weight bytes,0.743 aggregate worker wall-hours and scoped verification PASS. The held-out classifier scored71.48% on256 excluded examples (majority baseline53.52%). See `$ZOO_DATA/runs/track-b-training-gpu-v1/TRAINING-REPORT.md` and `decision-heldout-score.json`. A's merged selector repair and completed six LoRA fingerprints are consumed in the new36-model analysis; see `docs/l12-evaluation.md`. Trained checkpoints remain intact.

The tracked completion pointer is `lab/configs/launches/track-b-gpu-v1.json`; it binds thirteen run artifacts by SHA256, the completed counts and remaining A dependency.

The original CPU bundle, prepared2026-11-01, remains unchanged:
`$ZOO_DATA/runs/track-b-training-launches-v1` contains actual licensed data,
frozen safe token files, 11 selected SmolLM2-135M configurations, their hashes,
exact command arrays, preflight results, and the 46 official inventory recipes.
No training was executed. Nine extra small-base jobs pass preflight; two DPO
jobs require their actual SFT output first. These extra Smol135 IDs provide
**no G1 quota credit**. All official 24 LoRA, 8 SFT, 8 DPO, and 6 decision jobs
remain blocked on their anchor-specific weights and token artifacts.

The tracked pointer is `lab/configs/launches/track-b-v1.json`. The bundle's
`manifest.json` and `manifest.sha256` bind dataset/preprocessing facts and config
hashes; `bundle-index.json` binds all final artifacts. `commands.json` contains
argument arrays; `commands.txt` renders exact commands for a person. Treat the
latter as a queue: it includes deliberately blocked official commands. It is
not a script to execute blindly. All selected output paths are
`$ZOO_DATA/lab/smol-135-launch-*`.

## Authorized GPU execution, 2026-11-02

The owner subsequently authorized executing this small cohort on cluster GPUs
with **both utilization below80% and occupied memory below80%**. The original
CPU bundle remains immutable. `lab.scripts.gpu_training prepare` creates the
separate `$ZOO_DATA/runs/track-b-training-gpu-v1` bundle with new
`smol-135-gpu-v1-*` IDs and logical `cuda:0` recipes. Its manifest binds original
and GPU config hashes; data selections, all step counts, batches and seeds
remain fixed. Each job receives one physical GPU UUID through
`CUDA_VISIBLE_DEVICES`, passes two load checks at least five seconds apart,
and requires24GiB free headroom. GPU locks prevent overlapping owned jobs.

The full256-step SFT pilot completed in120.553s wall, with0.08977s per step
between synchronized completions and13.36GiB peak CUDA allocated memory
(14.72GiB peak reserved). Startup/loading/export are included only in wall time.
This is a small Smol135 measurement, not a prediction for larger anchors.
All11 jobs completed with zero failures. The remaining ten jobs used at most two GPUs concurrently; DPO binds actual
completed SFT checkpoint hashes. The runtime `launch_batch.py`,
`batch-progress.json`, per-attempt profiles and logs record commands, PIDs,
GPU UUIDs, checks, timings and completion/failure states. Each worker has a
7200s external timeout and rejects a measured training projection above2h.

Run one prepared GPU job with the actual manifest hash, for example:

```bash
python -u -m lab.scripts.gpu_training run \
  --config "$ZOO_DATA/runs/track-b-training-gpu-v1/configs/smol-135-gpu-v1-sft-s0.json" \
  --data-root "$ZOO_DATA" --gpu-uuid GPU-a9fec090-9da5-7a9d-bb61-f1428678a54b \
  --profile "$ZOO_DATA/runs/track-b-training-gpu-v1/profiles/new-attempt.json" \
  --expected-config-sha256 67253ec8b45c6e51dbd0713761c72163d8861db8d6b2aa5173fc347c04792af4
```

Use a new profile path for each attempt and the existing environment/offline
settings below; wrap execution in `timeout --kill-after=30s 7200s`. Add
`--resolve-parent` for DPO. Completed-checkpoint resume checks identical
provenance; partially optimized jobs restart their full recipe. No additional
training quota is earned by these extra Smol135 cases. The sections below
describe the original CPU preparation checkpoint.

## Pinned datasets and replacements

| Task | Publisher and exact revision | License | Training/direction/validation/scoring accepted rows |
| --- | --- | --- | --- |
| Instruction | `HuggingFaceH4/ultrachat_200k@8049631c405ae6576f93f445c6b8166f76f5505a` | MIT | 1880 / 110 / 120 / 106 |
| Dialogue | same UltraChat pin | MIT | 3746 / 237 / 234 / 227 |
| Preference | `HuggingFaceH4/ultrafeedback_binarized@3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` | MIT | 665 / 39 / 49 / 35 |
| Decision | `fancyzhx/amazon_polarity@9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289` | Apache-2.0 | 4096 / 256 / 256 / 256 |

License evidence is the actual publisher's pinned README declaration, with
publisher API metadata saved alongside it. Sources: [UltraChat publisher](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/blob/8049631c405ae6576f93f445c6b8166f76f5505a/README.md),
[UltraFeedback publisher](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/blob/3949bf5f8c17c394422ccfab0c31ea9c20bdeb85/README.md),
[Amazon Polarity publisher](https://huggingface.co/datasets/fancyzhx/amazon_polarity/blob/9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289/README.md).
Amazon's original publisher card identifies this as original data, declares
Apache-2.0, and explicitly maps 0 to negative and 1 to positive.

The proposed `trl-lib/ultrafeedback_binarized` revision
`47124cb5778f5d50de1c7676a412828f3ea7c555` had no publisher license metadata.
It was replaced with the licensed H4 preference dataset, not relabeled.
`stanfordnlp/sst2@8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb` declared
`unknown`; its rows were not used. Amazon Polarity is an explicit binary
sentiment task replacement, not an SST2 experiment. No upstream license was
assumed to cover an unlicensed derivative.

Each source contributes exactly 4864 raw rows from the first pinned Parquet
shard: raw positions `[0,4096)`, `[4096,4352)`, `[4352,4608)`, and `[4608,4864)`
form disjoint partitions before filtering. IDs include repository, commit,
source filename and row index; actual selected-ID hashes bind accepted rows.
Parquet LFS source hashes, exact source URLs, original publisher metadata,
selected JSONL hashes, accepted/rejected counts, and every token-file hash
are in the bundle. HTTP byte-range reads transferred 17,872,604 bytes for
UltraChat, 18,955,846 for UltraFeedback, and 2,563,335 for Amazon Polarity;
the full source shards were not downloaded. Acquisition timings were
10.86, 14.96, and 5.02 seconds, respectively.

Instruction renders the first user/assistant pair as `role: content\n`;
dialogue renders all turns. Both take the first 512 tokens, exclude shorter
rows, and train all tokens without padding. SFT uses instruction rows. DPO
renders the shared prompt plus `assistant: `, keeps its first 256 tokens,
and truncates each completion to fill 512 tokens; rows with insufficient
completion tokens are excluded. Its masks are contiguous completion-only
suffixes with an unmasked identical nonempty prompt. Classification renders
`title\ncontent`, truncates to 128, and right-pads with token 0 with an
explicit attention mask. Text rows containing the padding token are excluded.
All four partitions have actual separate safetensors files; held-out tokens
are never passed to a trainer.

The tokenizer is the local pinned Smol135 tokenizer, identity
`c7c6d2ad363def654e7c6b8463d5bdeafc01732c8da2fabe012ff76efa87e88d`.
Its individual metadata hashes are recorded. No chat template is silently
inherited. CPU preprocessing took 2.92s instruction, 6.47s dialogue, 3.93s
preference, and 0.64s decision; token artifacts total 50,193,328 bytes.
These are measured preprocessing times, not training benchmarks.

## Preflight and human launch

Use the existing Python environment, without upgrades:

```bash
cd "$ZOO_DATA/worktrees/Zoo-Keeper-track-b"
PY=/scratch/kinshuk/anaconda3/envs/esd_ind/bin/python
BUNDLE="$ZOO_DATA/runs/track-b-training-launches-v1"
"$PY" -m lab.launches --config "$BUNDLE/configs/smol-135-launch-sft-s0.json" --data-root "$ZOO_DATA" --load-model
```

This checks evidence and license declarations, source/model/token hashes,
all four disjoint selection lists, exact token shape/vocabulary, preprocessing,
and task-specific masks. `--load-model` additionally loads only local safe
weights and checks actual installed architecture/input support. CPU loading
passed for real Smol135 instruction and dialogue LoRA, SFT and decision
classification, without a forward/backward optimization step. Classification
correctly initializes its new two-label head. The source is
`lab/inputs/cohort-v1-smol-135-93efa2f097d5`, with actual SHA-bound weights.

Only a person's explicit launch executes training:

```bash
"$PY" -m lab.launches --config "$BUNDLE/configs/smol-135-launch-sft-s0.json" --data-root "$ZOO_DATA" --execute --resume
```

Every selected command and config hash is in `commands.json`. Settings retain
128 LoRA steps/ranks 8,64,256, 256 SFT steps, 128 DPO steps/beta0.1, and 32 decision
steps; CPU/bfloat16 is explicit. The helper uses existing `write_checkpoint`
and trainers; outputs receive the same complete safe checkpoint and provenance
contract as existing batch CLIs. `--resume` requires identical recipe/input
provenance. No step count was reduced for these prepared jobs.

After SFT actually completes, bind DPO to the actual matching parent:

```bash
"$PY" -m lab.launches --config "$BUNDLE/configs/smol-135-launch-dpo-s0.json" --data-root "$ZOO_DATA" --resolve-parent-output "$BUNDLE/resolved-configs/smol-135-launch-dpo-s0.json"
"$PY" -m lab.launches --config "$BUNDLE/resolved-configs/smol-135-launch-dpo-s0.json" --data-root "$ZOO_DATA" --execute --resume
```

Resolution requires completed full-parameter SFT provenance, matching original
base pin, and actual local safetensors/config bytes. It creates a new immutable
resolved config with a content-derived parent key and metadata hashes; it
never edits the original config or substitutes the pretrained base. It fails
now because no SFT training has been run. Repeat the same two-stage sequence
for `s1` after its matching SFT. The DPO reference starts exactly at that SFT
checkpoint under the existing trainer.

## Resources and remaining prerequisites

Each extra job reads the measured 269,060,552-byte source weight file plus its
small token input (DPO instead reads its eventual SFT file). Expected primary
output weights are approximately 269 MB each; all 11 extra outputs require about
3GB plus adapters, atomic staging, and operating headroom. Weights are written
only by human training. Source/header and bundle byte facts are recorded;
actual output sizes cannot be measured before training.

Training time and peak memory have **not been benchmarked**. Rough planning
allowances on the current CPU are LoRA 0.25–4 h/job, SFT 0.5–8 h/job,
DPO 0.5–8 h/job, and decision 0.05–2 h/job, approximately 3.55–58 CPU-hours for
all 11. These broad unmeasured estimates do not establish a two-hour bound.
CPU commands request no GPU. Full tuning's fp32 parameters/gradients/Adam
moments alone need approximately 16 bytes/parameter (2.15 GB at 135M); activations,
logits and DPO's extra frozen reference add substantial unmeasured memory.
The scientific plan's official full-batch GPU estimates remain provisional:
LoRA 40–80 GPUh, SFT+DPO 70–130 GPUh, decision 0.3–1.5 GPUh.

The official batch configs and exact existing `lab.scripts.{lora,sft,dpo,decision}`
commands are frozen under `official-configs` and `commands.json`; they retain
explicit unresolved input states. Preparation does not make those 46 jobs
runnable. Acquire their pinned anchor safetensors, materialize equivalent
anchor-tokenizer-specific rows/hashes, and bind SFT parents before replacing
those blocked configurations in a new bundle version. Do not reuse Smol135
token IDs with another tokenizer. Decisions also need each actual pad ID and
encoder special-token handling, with new preprocessing pins.

Every selected job has its own frozen `verification/<model_id>-models.csv` and
`-edges.csv`; `commands.json` records the exact `lab.cli verify` invocation.
DPO verification includes its completed SFT parent and the original base.
These small tables are separate from official CSVs and preserve the existing
schema. Run the recorded verification command after training; absent outputs
must fail now. Full `--minimum` cannot pass from extra Smol135 jobs. Actual
checkpoint counts and the complete official inventory remain separate.
