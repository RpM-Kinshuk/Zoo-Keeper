# Local lineage engine

The inference components work on synthetic records. The extractor adapter also
reads real T1 fingerprints through the committed Model-Zoo X6/X8 APIs. Full
edge inference, Parquet publication, CLI orchestration, and zoo build integration
still await the zoo schema, retrieval, and build modules (Z1/Z4/Z6).

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
adapter does not infer architecture from a partial fingerprint. Automatic
manifest discovery and architecture computation will use Z1 once merged.

The adapter preserves extractor delta summaries and the pinned parent identity.
It records SHA256 hashes of the input manifest and fingerprint in each record's
metadata. Pair scoring uses the shared extractor median cosine at `k=64` after
filtering aligned views for engine eligibility. Zero/nonfinite views and
incompatible shapes/settings are excluded with the existing view diagnostics.
Scores do not certify direction or calibration. Retrieval through the real zoo
still requires Z4; loading and pair scoring require only the extractor.

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
downloads or real lab generation. The full inference CLI and resumable output
bundle are pending the remaining E8 integration work; they are not available
commands yet.
