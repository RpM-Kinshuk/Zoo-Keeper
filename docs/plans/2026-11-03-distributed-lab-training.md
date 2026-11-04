# Two-GPU lab training extension,2026-11-03

User explicitly requested model training across multiple GPUs and delegated
implementation choices. Use PyTorch DDP for the current Qwen0.5/0.6B formal
SFT/DPO chains, with one process per GPU and explicit contiguous partition of
the existing global batch. Global batch16/8, steps256/128,512-token inputs,
learning rates, seeds, data IDs and held-out exclusions remain fixed. DDP
averages gradients; local batches8/4 retain the global objective. This does
not shard parameter/optimizer replicas; models exceeding that footprint need
a separate FSDP extension. Missing approved pinned dependencies may be restored with --no-deps under
DIRECTIVES D7.5; installed versions must not be upgraded, downgraded or removed.

Add new distributed trainer/worker modules only. Keep the live Smol360
single-GPU trainer and frozen inputs/outputs unchanged. Publish a versioned
DDP config with explicit world/backend/local/global batch/partition metadata;
floating-point reduction order can differ from single-GPU runs. Rank0 alone
publishes safe BF16 weights and standard recipe/lineage/training provenance.
DPO binds its actual completed SFT parent and freezes the reference on every
rank. All ranks must agree on resume success/failure. CPU export releases GPU
optimizer/reference/model allocations before persistent filesystem writes.

Track admission/loading/training/CPU-export phases, last-step progress,
synchronized throughput, global losses and per-rank peak VRAM/UUID/name.
GPU admission checks BOTH utilization<80% and occupied memory<80% twice at
least5s apart; reserve two distinct UUID locks together. Initial L40 Qwen
pilot initially required40GiB free perGPU and profiles full frozen steps with7200s
external bound and measured<2h projection. Smol360 uses its measured23GiB
peak plus32GiB free headroom. Do not change batch/sequence settings afterOOM.

Verify with real two-process CPU/Gloo SFT+DPO comparisons against a one-process
global-batch baseline, frozen-reference/strict-input/rank0 publication/resume
regressions; one fresh core review and one fix wave. Then execute actual
Qwen2.5 SFT→DPO→q8 and Qwen3 SFT→DPO→q8 as separate formal chains. Native
Qwen sources/IDs are newly verified at original pins; CPU templates and old
reports remain immutable. First real two-GPU run verifies NCCL/hardware path.

Current Eniac preexisting runtime is torch2.11.0+cu128, whereas the prior
Voyager run used the repository pin2.10.0+cu130. Record actual runtime in new
execution artifacts and retain the old requirements/frozen run provenance.
No installed package was changed. Pure system pytest is appended only to
test-process import paths; numerical libraries remain the target environment.

Execution amendment: initial40GiB admission stopped before optimizer startup.
Attempt2 uses38GiB free per card, with unchanged paired<80% criteria and
unchanged recipes. Local batches are halved relative to the measured23.08GiB
Smol360 single-GPU DPO; larger Qwen parameter/logit state retains a substantial
initial buffer. First real Qwen2.5 SFT steps reserve24.25GiB per rank. Record
full-run peaks before deriving future admission limits; OOM stops the pilot.

Scope review,2026-11-03: DDP is an execution aid for the existing lab/P1 plan.
Do not add FSDP infrastructure before a required recipe demonstrably exceeds
replica memory. The next experiment is the existing L4 Smol360 merge batch,
using completed formal LoRA parents; Qwen LoRA follows to unlock other merges.
The fourth training/LoRA anchor is OLMo-1B; Qwen2.5-1.5B is base/decision-only.
The four completed DDP jobs made no package changes. Subsequent merge
verification restores absent approved packages under D7.5 and records installs.


Measured execution extension,2026-11-04: native OLMo1.485B SFT fits two L40s
(39.515GiB allocated/41.912GiB reserved), but DPOs0 fails at its second
forward while scoring the frozen reference, requesting1.53GiB with only
549.56MiB free. Both failed-rank traces and exact peak counters are retained
under `runs/track-b-l6-remaining-20261104/`. No checkpoint was published.

The owner's existing request to parallelize larger models across GPUs covers
increasing the execution rank count to four. Extend only the launcher to
accept two or four single-node ranks and protect every rank's profile path.
The trainer's existing contiguous global-batch partition and averaged gradient
objective stay fixed: globalDPO batch8 becomes local2 at world4. Parameter and
optimizer replicas remain unsharded; smaller local activations/logits address
the measured forward-memory failure. Preserve dtype, tokens, loss, seeds,
steps, LR and actual frozen SFT references. Use distinct world4 frozen configs
and attempt profiles; retain the original two-rank OOM and successful outputs.

Verify four-process CPU/Gloo SFT and DPO against the same independent global
baseline, plus supported topology and fourth-rank profile collision checks.
One fresh core review and one full suite cover this small extension. Retry
only the two missing OLMo DPOs after all four eligible physical UUID locks
are jointly acquired, fresh paired usage/memory checks pass and the measured
whole-action/storage bounds still fit. Record four-rank hardware/reduction
order provenance. Do not add an FSDP subsystem or alter fixed science.


Hardware follow-through,2026-11-04: capacity returned and both native OLMo
DPO seeds completed128 steps on4L40s, global8/local2. Training750.251s and
748.043s; peak38.80GiB allocated/43.385GiB reserved per rank. Saved checkpoint,
frozen reference, full-update, held-out/provenance checks and new2T1 PASS.
The original two-rank failure remains preserved. All16 SFT/DPO identities
now verify (8+8), with source599tests/fresh review unchanged and all workers
exited. Proofs: `runs/track-b-l6-remaining-20261104/world4/future-capacity/execution2/`.
