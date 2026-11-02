# Local lineage engine

The runner reads existing T1 fingerprints through the committed Model-Zoo
extractor and zoo APIs. It retrieves candidates, scores aligned views, records
typing/stacked diagnostics, and publishes detected edges in the frozen zoo
schema. It never loads checkpoints or reads lab truth to infer edges.

The supplied `configs/lineage/method.json` is an unfitted development method.
Its zero reporting threshold is a software default, not a calibrated operating
point. Real lab calibration and empirical performance remain pending.
Changing the inference method requires a method-version bump; configuration
changes are additionally captured by the complete config hash.

## Inference and publication

With the committed Model-Zoo package importable, run:

```bash
python -m lineage.cli dry-run --out "$ZOO_DATA/runs/lineage-smoke"
python -m lineage.cli infer \
  --manifests "$ZOO_DATA/runs/lab/manifests" \
  --fingerprints "$ZOO_DATA/runs/lab/fingerprints" \
  --data-root "$ZOO_DATA" --config configs/lineage/method.json \
  --out "$ZOO_DATA/runs/p1/edges.parquet"
```

`dry-run` generates synthetic records and one explicitly synthetic asymmetric
direction example; it is not a real lab result. `infer` discovers real T1
fingerprints using the upstream model tables and architecture signatures,
builds a temporary retrieval index, then deletes that index when finished.
Missing measurements, duplicate exclusions and unresolved directions appear
in diagnostics rather than becoming fabricated edges.

The requested edge path points to an immutable bundle containing
`edges.parquet` and `provenance.json`. Resolve the edge path and read the adjacent
provenance file for the matching config, method/input/calibration hashes and
diagnostics. The pointer switches only after the complete bundle exists;
interruption preserves the prior output. Old complete bundles are retained for
readers. `--resume` requires identical hashes and keeps the existing bundle;
changed inputs or settings fail. Use a new output path for a changed method.
Pass the edge path as `lineage_edges=` to `zoo.build.build_zoo`.

Symmetric cosine supports same-root scores. Parent claims additionally require
an explicit development-frozen asymmetric score in the record's
`direction_evidence`, whose `method_hash` matches the config's
`direction_method_hash`. The default has no fitted direction method. Parent
metadata, timestamps and symmetric similarity alone never resolve direction.
Ancestor scores propagate supported parent paths with the configured decay;
they retain path evidence and are uncalibrated. Contradictory directional
cycles are removed and reported. Optional `typing_rules`, `residual_views`,
`merge_proposals` and `assemble_tree` settings expose the existing diagnostic
components; regression fit alone never supplies direction.

`--calibration FILE` accepts artifacts keyed by estimand. A calibrated artifact
must match the method hash, candidate-universe hash, architecture and statistic
settings, and preserve validated exchangeability. Per-query e-BH includes all
same-architecture T1 candidates in its denominator, including candidates that
could not be scored or retrieved. Calibrated rows are exported only when the
query control rejects; unsupported or absent calibration leaves development
scores explicitly uncalibrated, with null p-values. No control guarantee is
borrowed by ancestor closure.

## Extractor input

Call `lineage.upstream.load_records(index, fingerprints, data_root=...)` with a
JSON index. Each real row explicitly binds these four fields:

```json
[
  {
    "key": "publisher/model@0123456789abcdef0123456789abcdef01234567",
    "arch": "cfg:<authoritative architecture signature>",
    "manifest": "manifests/model.manifest.json",
    "fingerprint": "fingerprints/model.t1.fingerprint.safetensors"
  }
]
```

The example identity is illustrative. Use actual pinned identities; local
models use `lab/<model_id>@local:<sha256>`. Manifest and fingerprint identities
must both match the index. The index, manifest, and fingerprint paths stay
within `data_root`, and each fingerprint stays within the supplied fingerprint
directory. Absolute paths and escaping symlinks in index entries are rejected.
T2 is unsupported by this adapter. JSON fingerprint payloads are reserved for
synthetic fixtures.

Supply the architecture signature from the authoritative model index. The
adapter does not infer architecture from a partial fingerprint. Alternatively,
pass a manifest directory; its identities and architecture signatures come
from the merged zoo model tables. The CLI uses this directory form.

The adapter preserves extractor delta summaries and the pinned parent identity.
It records SHA256 hashes of the input manifest and fingerprint in each record's
metadata. Pair scoring uses the shared extractor median cosine at `k=64` after
filtering aligned views for engine eligibility. Zero/nonfinite views and
incompatible shapes/settings are excluded with the existing view diagnostics.
Scores do not certify direction or calibration. Retrieval uses the merged zoo
index API; loading an explicit index and pair scoring need only the extractor.

## Local validation

Make the committed Model-Zoo main checkout and collector sources importable;
do not use another track's uncommitted checkout. From this Track B worktree in
the prepared `esd_ind` environment:

```bash
MODEL_ZOO_ROOT="$(cd ../../Model-Zoo && pwd)"
export PYTHONPATH="$MODEL_ZOO_ROOT:$MODEL_ZOO_ROOT/esd_experiment/src${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests -q
```

The extractor integration tests create tiny CPU tensors and fingerprints in
temporary directories, with socket connections forbidden. They perform no Hub
downloads or real lab generation. The integration check also imports generated
edges into a tiny upstream zoo build and verifies their schema and values.

## Real cohorts and frozen evaluation

The real path is separate from release staging:

- `$ZOO_DATA/runs/track-b-real-cohort-v1/REPORT-v2.md`: ten real models,
  including Smol rescaling/perturbation and three planned OLMo quantizations.
  The rejected MLP permutation is not counted.
- `$ZOO_DATA/runs/track-b-safe-control-cohort-v1/REPORT.md`: twelve real
  models from four independent Pythia seeds, each with q8/q4 descendants.
  Existing official safe-conversion commits are pinned separately from the
  original inventory; no pickle or new conversion was used.

Seeds1/2 are development roots and seeds3/4 are held-out roots, frozen before
generation/scoring. The unchanged0.75 threshold gives same-root AUROC1.00,
recall1.00 and0 false matches on twelve positive and eighteen independent-negative
held-out pairs. This is only two held-out roots and six model queries.
Parent/ancestor direction still has zero coverage/TPR; no lineage gate or error
control is established. Independent-seed comparisons conservatively use the
verified-Hub truth tier. Construction edges have actual checkpoint provenance;
cross-architecture negatives are not fabricated.

Each run contains scoped tables, pins/weight hashes, method/input hashes,
predictions, per-model extraction costs and partitioned evaluation outputs.
`lineage.cli freeze` creates an immutable directory before metrics; `evaluate`
checks its inputs, executed scorer components and numerical runtime. For a
new evaluation of the existing test artifacts:

```bash
COHORT="$ZOO_DATA/runs/track-b-safe-control-cohort-v1"
python -m lineage.cli freeze \
  --config "$COHORT/scoped/test/evaluation.json" \
  --inputs "$COHORT/scoped/test/inputs.json" --out "$COHORT/evaluation-new/test"
python -m lineage.cli evaluate \
  --predictions "$COHORT/scoped/test/predictions.parquet" \
  --truth "$COHORT/scoped/test/truth.parquet" \
  --universe "$COHORT/scoped/test/universe.parquet" \
  --costs "$COHORT/scoped/test/costs.parquet" \
  --models "$COHORT/scoped/models.csv" --edges "$COHORT/scoped/edges.csv" \
  --manifests "$COHORT/evidence/manifests" \
  --config "$COHORT/scoped/test/evaluation.json" --out "$COHORT/evaluation-new/test"
```

Outputs are `metrics.json`, `curves.parquet`, `coverage.parquet`, `protocol.json`
and `input_hashes.json`. Missing predictions remain abstentions; unknown truth
stays unknown. Root/merge components and duplicate groups cannot cross splits.
Unattainable finite-score TPR90 has null operational metrics; separate
`ranking_extension` retains negative-infinity tie accounting. Frozen deployment
metrics stay separate. Input paths are relative to the freeze directory; old
incomplete freezes are rejected rather than silently upgraded.

All22 models have measured T1 tensor reads and parent-read breakdowns. Full
preparation/cold-reference/cached-inference IO remains incomplete, so complete
bytes-per-answer is unavailable. Missing baselines are `not_run`; absent
T2/5k-pool/baseline results leave gates `NOT_EVALUABLE`. A's queued incremental
k128/T2 launch uses committed extractor code and separate output; see the main
checkout's `handoff/track-b-incremental-l12.md`. Whole-lab L12 remains pending.

Training launches are in `$ZOO_DATA/runs/track-b-training-launches-v1`, documented
in `docs/training-launches.md`: nine scoped Smol jobs are human-ready, two DPO
jobs need actual SFT parents, and official batches need anchor-specific inputs.
No optimizer step was run. Use each run's frozen runner/config hashes; changed
code/configuration requires a new version. Release staging remains deferred.
