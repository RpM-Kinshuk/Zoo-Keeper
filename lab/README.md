# Genealogy lab tooling

The lab records exact construction truth for model-lineage experiments. Code,
recipes, and the small pin registry live here; checkpoints and evidence live
under `$ZOO_DATA/lab`. The approved L1 plan is in the main checkout's
`docs/plans/2026-10-31-genealogy-lab.md`.

The final planned inventory contains 166 models and 187 construction edges:
112 public derivatives, seven bases, and 47 references. All 54 external models
are frozen in `pins.json`. The tables describe planned construction; they do
not establish checkpoint or measurement completion. See [HANDOFF.md](HANDOFF.md)
for Track A's file contract and the reference labels used for calibration;
`table_manifest.json` records counts and content hashes.

Run commands from this repository in `esd_ind`, with the workspace environment
loaded by sourcing `.env` without displaying it. The package requirements pin
the installed versions. For the full suite, also expose the committed extractor
as described in [the lineage runbook](../docs/operations/lineage.md).
No existing package version was changed for this work. Additional packages
were installed with `--no-deps`; mergekit declares a newer pydantic minimum
than the existing environment, although its exercised backend tests pass.
Heretic is not runnable under the required safety policy. Do not let a default
dependency resolution upgrade the collector environment; clean-environment
compatibility remains a recorded handoff item.

```bash
python -m lab.cli validate
python -m lab.cli inventory --config lab/configs/inventory.json
python -m pytest tests -q
```

Each generator accepts `--config`, `--lab-root`, `--dry-run`, `--tiny`, and
`--resume`. Dry runs validate and describe recipes without loading weights.
Tiny runs use synthetic local fixtures; they do not count toward the real lab.

| Module under `lab.scripts` | Configuration | Planned real outputs |
| --- | --- | --- |
| `lora` | `lab/configs/lora.json` | 24 rank-stratified adapters and effective checkpoints |
| `sft`, `dpo` | `lab/configs/sft.json`, `lab/configs/dpo.json` | Eight full SFT and eight DPO continuations |
| `merge` | `lab/configs/merge.json` | 30 linear, SLERP, TIES, and DARE-TIES merges |
| `abliteration` | `lab/configs/abliteration.json` | Classic and Heretic variants |
| `quantization_sim` | `lab/configs/quantization_sim.json` | 12 quantize/dequantize simulations, stored bf16 |
| `laundering` | `lab/configs/laundering.json` | 16 public transforms; C manages the blind contribution |
| `decision` | `lab/configs/decision.json` | Four decoder and two encoder decision heads |

For example, an offline tiny LoRA run writes only to a chosen scratch lab:

```bash
python -m lab.scripts.lora --config lab/configs/lora.json \
  --lab-root "$ZOO_DATA/cache/track-b-smoke" --tiny
```

Real recipes require prepared local inputs, immutable parent identities,
expected configuration and weight hashes, and frozen data selections. Loaders
reject missing or mismatched provenance before training. No script implicitly
downloads a model or executes checkpoint code. Approved launch bundles freeze
licensed data, tokenizer-native selections, held-out splits and preprocessing
separately for each source. An inventory entry alone does not make a run ready.

Real LoRA/SFT/DPO batches require the person's authorization. Runs require a
measured projection within both limits: at most100 weight models and two hours,
within the authorized scope. Select an allocated
device explicitly and obey the shared-GPU checks. Completed runs, exact pins,
counts and artifact locations are recorded in the main handoff status.

Full SFT/DPO can use two or four GPUs through `lab.scripts.distributed_training`.
Its caller must reserve all physical UUID locks and check each GPU's
utilization and occupied-memory fraction below80% twice, at least5s apart,
with enough free memory for the measured recipe. Freeze a new config with
`recipe.distributed` containing `world_size=2` or `4`, `backend="nccl"`, the original
`global_batch`, `local_batch=global_batch/world_size`, and
`partition_policy="contiguous_global_step_batch_v1"`. Keep global batch,
steps, learning rate, data and seeds unchanged. DDP replicates model/optimizer
state; models exceeding one replica's memory need a separate sharding path.

After CPU preflight and GPU admission, the reserved caller sets
`CUDA_VISIBLE_DEVICES` to the reserved UUIDs and starts the worker:

```bash
timeout --signal=TERM --kill-after=30s 7200s \
  python -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node=2 --max_restarts=0 -m lab.scripts.distributed_training \
  --config "$CONFIG" --expected-config-sha256 "$CONFIG_SHA256" \
  --data-root "$ZOO_DATA" --profile "$NEW_PROFILE" --execute --resume
```

For four ranks, use `--nproc_per_node=4` with matching frozen world4 metadata.
Smaller local activations and logits can fit a recipe that exceeds two GPUs;
parameter and optimizer replicas remain unsharded.

Resolve the actual completed SFT parent before admitting DPO. Each launch
attempt needs a new profile path. Rank0 alone publishes the atomic checkpoint;
each nonzero rank writes a separate profile. Profiles report phases, step progress,
synchronized timing, actual runtime, current GPU allocation/reservation and
per-rank peaks. GPU tensors are released before CPU checkpoint serialization.

Real merge recipes may set `compute_device` to `cuda:0` to use mergekit's
existing GPU arithmetic. The caller admits and locks one physical GPU, sets
`CUDA_VISIBLE_DEVICES` before launch, and freezes that execution choice with
the recipe. Omitting the field preserves the existing CPU invocation. Record
device/runtime and measured pilot costs; floating-point results and seeded
DARE masks can differ between CPU and GPU execution.

The installed Heretic 1.4.0 loader lacks the required safe local-loading
controls. Its adapter refuses execution; synthetic dispatch tests are not
genuine Heretic results. Classic ablation remains available. Permutation,
rescaling, and value rotation have architecture-specific equivalence checks;
targeted perturbation is not labeled function-preserving. Simulated
quantization is not a claim of equivalence to a packed quantization format.

Checkpoints are published atomically with `lineage.json` and `recipe.json`.
Matching resume is a no-op; changed inputs or recipes do not replace completed
outputs. Versioned table pairs use a single `current.json` pointer:

```bash
python -m lab.cli inventory --config lab/configs/inventory.json \
  --pins lab/pins.json --output "$ZOO_DATA/lab/tables"
python -m lab.cli verify --tables-root "$ZOO_DATA/lab/tables" \
  --lab-root "$ZOO_DATA/lab" --minimum
```

The materialization command reproduces the checked-in planned inventory.
The minimum checker reports missing outputs rather than launching jobs. The
combined laundering target stays pending C's aggregate attestation; it never
opens the blind directory or declares G1 itself.

`lab.cli export-models` writes an invocation-only CSV for the extractor. Local
paths are absolute only in that temporary CSV; durable manifests use portable
`lab/<model_id>@local:<sha256>` identities. Put the CSV in temporary scratch or
`$ZOO_DATA/cache`, and remove it after manifest extraction. The scorer and
lineage runner now import the committed Model-Zoo schema/retrieval/build APIs.
See [the lineage runbook](../docs/operations/lineage.md) for inference and edge
publication. Score existing predictions with:

```bash
python -m lab.cli score --models lab/models.csv --edges lab/edges.csv \
  --predictions "$ZOO_DATA/runs/p1/edges.parquet" \
  --manifests "$ZOO_DATA/runs/lab/manifests" \
  --universe "$ZOO_DATA/runs/p1/universe.parquet" \
  --config lab/configs/scoring.json --output "$ZOO_DATA/runs/p1/lab-scores.json"
```

The universe must explicitly list query/candidate keys and boolean eligibility.
Join external reference labels from `reference_pairs.json` into its nullable
`truth_parent`, `truth_ancestor`, `truth_same_root`, and `truth_tier` columns
before scoring. Missing external labels stay unknown; an absent edge is not a
negative. Construction truth, verified Hub evidence and declarations are
reported separately. Missing predictions remain abstentions in coverage and
ranking. TPR90 thresholds are descriptive, including ties; an optional supplied
development threshold gives frozen-threshold metrics. An all-abstention cutoff
is represented as the JSON string `"-inf"`, and undefined metrics are null.

The planned inventory needs matching manifests for every row before real lab
scoring. No metrics are claimed from the checked-in inventory alone. Local
benchmark release staging remains the next lab tooling task; all publication
remains a person's action.


The completed L12 delivery and real depth-three-chain evaluation are described
in `docs/l12-evaluation.md`. `lab.scripts.chain_cohort` creates the three
scoped extra endpoints; `lab.scripts.evaluate_delivery` consumes A artifacts
and publishes a new immutable report without changing old held-out protocols.

`lab.heretic_safe` and `lab.heretic_objective` provide the reviewed binding for
installed `heretic-llm==1.4.0`: preload a builtin safetensors model with
`trust_remote_code=False`, call `prepare_stock_run` with separately frozen,
licensed harmful/harmless direction and objective records, optimize the stock
refusal/KL objective, and merge the selected in-memory PEFT edit safely. Freeze
runtime controls and source hashes before optimization. The original CLI's
unprepared Heretic path remains fail-closed; sentiment token inputs do not bind
the stock refusal objective. Record the actual sampler seed from the nested
runtime config alongside any inventory recipe seed. A zero-refusal pretrained
baseline uses the upstream absolute-count rule and supports no quality claim.

`lab.transform_drift` implements the separately approved generation-only
`public-transform-drift-v2` cohort. It reuses the frozen permutation/rotation
recipes and unchanged numerical checks, then requires exact finite BF16 state,
safe reload, complete matching token audits, and immutable checkpoint files.
Its version-2 `cohort.json` separates verified construction from functional
failure. New IDs end in `-drift-v2`; original formal slots and G1 credit are
unchanged. See the public-transform-equivalence-v2 protocol in the main
checkout's `docs/plans/` for the owner's approval boundary.
