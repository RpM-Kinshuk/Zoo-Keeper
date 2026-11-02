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
downloads a model or executes checkpoint code. The training configurations
remain explicitly unfrozen: the proposed DPO and decision datasets lack clear
license evidence, and token selections/preprocessing still need freezing.

Real LoRA/SFT/DPO batches are launched by a person. Other runs require a measured
projection within both limits: at most 100 weight models and two hours. Select
an already allocated device explicitly and obey the shared-GPU checks. No large training or full generation batch has been launched. A bounded OLMo
quantization attempt was interrupted before derivative verification; its frozen
configuration and partial source are recorded in the main handoff status.

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
