# Frozen heritability and exchange-rate study

This study is parked, not a current task. Run reproduction from the main Zoo-Keeper checkout in the existing `esd_ind` environment. Source the workspace `.env` without opening or printing it when data paths are needed. The commands read selected CSV/JSON tables only: no weights, model evaluation, downloads, paid APIs or predictor refits.

## First frozen audit (2026-10-31)

The completed first pass is `$ZOO_DATA/studies/heritability/track-c-first-pass-v1/final/report.md`, status `PARTIAL`. Its protocol SHA256 is `4f3b6d804608d77e0ed96efa661219f8a70e04782cab7d1360f3695baed7146a`. It retains 3,288 score observations, excludes 2,316 conflicting identities, and explicitly quarantines six metadata self-parent declarations. Supplied inputs support no independently verified deployment joins or known sampling variances; historical prediction targets also lack eligible Raw semantics/provenance. The resulting unavailable estimates are deliberate coverage results. See the [parked methods/audit](../research/heritability-audit.md) for the audit table and source-field references.

The run includes its portable `config.json`, assembly/audit snapshots and frozen inputs. Resume its verified analysis from this worktree:

```bash
HERITABILITY_AUDIT_ROOT="$ZOO_DATA/studies/heritability/track-c-first-pass-v1"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.heritability.cli analyze --run "$HERITABILITY_AUDIT_ROOT/final" --resume
```

The original modern/legacy historical targets remain unvalidated as Raw in this audit. No inverse clipping or implicit percentage conversion occurs. Original historical source documents are retained byte-for-byte as evidence; old path strings inside those documents are not operational references in the new portable inventory. Use a new run version and explicit source evidence when adding validated v1 inputs, denominator/variance metadata or independent pinned deployment records.

## Tiny offline check

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.heritability.cli dry-run --out /tmp/heritability-tiny
python -m studies.heritability.cli analyze --run /tmp/heritability-tiny/final --resume
```

The dry run generates eight synthetic families with three observed models each, independent pinned deployment records, documented binary counts and a composite with unavailable sampling variance. It assembles scores and ancestry, audits two paired Raw prediction methods with explicit held-out split membership, freezes all selected source/provenance snapshots, and invokes the ordinary analysis. The config records `dry_run.synthetic=true` and 20 bootstrap replicates. Family and network REML, observed-parent regression and exchange rates use the same functions as a regular run. Unavailable composite results remain visible. Add `--without-historical` to exercise an honest partial report whose exchange rows say `historical_predictions_missing`.

The output contains `config.json`, `sources/`, `assembly/`, optional `synthetic_history/` and `historical/`, and `final/`. Reusing a directory with differing immutable snapshots is rejected; choose a fresh output directory for changed inputs.

## Source assembly

```bash
python -m studies.heritability.cli assemble   --config configs/heritability_v1.json   --source-root ../Model-Research   --out "$ZOO_DATA/studies/heritability/assembly-v1"
```

Source names in the config must be relative, remain inside their source root, and avoid symlinks and environment files. The root is the config's directory (its parent repository for `configs/`); `--source-root` is an explicit fallback for the same relative names, not a recursive search. The provided config points to approved repository snapshots and their source manifest. Every selected source, registry, evidence document, pinned manifest, duplicate table and historical provenance config is hashed and copied under `source_inputs/` before use; any supplied source-manifest digest must match. Missing sources are retained in the inventory and exclusions.

`task_registry` refers to the explicit schema/Raw task mapping; denominator and variance proofs remain relative to the documented source root. `pinned_models` is a relative CSV with independent checkpoint/deployment evidence. No source's own SHA becomes an independently verified score join. Optional documented parameter counts require both `parameter_count` and `parameter_count_source` in independent pinned records. The primary deployment selector retains its alternatives as sensitivity/descriptive rows and refuses missing or ambiguous harness matches.

Assembly writes scores, primary/sensitivity scores, declared edges, conservative groups, exclusions, coverage, portable config, source inventory and an assembly hash manifest. Raw metadata and `raw_declared_edges.csv` retain self-parent declarations; each such metadata row is explicitly quarantined as `self_parent_declaration` before graph validation. Any remaining graph cycle is `ancestry_cycle`: score assembly survives, all ancestry groups share an unresolved split group, root groups remain empty, and lineage-dependent fits abstain. No cyclic graph is simplified into fabricated independent roots.

## Audit existing historical outputs

```bash
python -m studies.heritability.cli historical   --run-root "$SEPTEMBER_RUNS"   --assembled "$ZOO_DATA/studies/heritability/assembly-v1"   --out "$ZOO_DATA/studies/heritability/historical-v1"
```

`SEPTEMBER_RUNS` is an explicit argument supplied by the person; it is not a mandatory environment key and is never stored as an absolute path. Discovery is bounded and metadata-only. The config's optional `historical_mapping` supplies the Task 4 audit mapping: `prediction_file`, `protocol_file`, `manifest_file`, `split_file`, `baseline_method`, explicit `prediction_scale`/`prediction_semantics`/`setting`, and canonical-to-source `columns`. These file names must be relative to `--run-root`. No unrecorded scale, harness, fold semantics or held-out status is guessed. A normalized performance artifact can be mapped explicitly and preserved in `audit.json`'s descriptive rows; it never becomes Raw by rescaling.

The command writes `predictions.csv`, exclusions, discovery, audit, mapping, copied selected source artifacts and `audit_manifest.json`. Only provenance-eligible Raw paired rows support primary exchange rates; incomplete provenance remains descriptive. A missing historical directory produces an unavailable audit rather than blocking score analysis.

To include this audit in a final run, set the config's `historical_audit` to its relative directory under the documented source root. Reassemble using that same config before freezing: assembly and freeze configs must match exactly. A convenient portable layout puts the config, `historical-v1/`, and `assembly-v1/` under a shared study root, with other source paths relative to that root. No absolute historical path belongs in the config. Omitting `historical_audit` yields explicit missing-historical exchange rows.

## Freeze and analyze

```bash
python -m studies.heritability.cli freeze   --config configs/heritability_v1.json   --assembled "$ZOO_DATA/studies/heritability/assembly-v1"   --out "$ZOO_DATA/studies/heritability/workshop-v1"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m studies.heritability.cli analyze   --run "$ZOO_DATA/studies/heritability/workshop-v1" --resume
```

Freeze first verifies assembly and optional historical audit inventories. It copies selected inputs beneath `inputs/`, including `inputs/scores.csv`, original source snapshots and historical provenance artifacts. `protocol.json` uses the canonical `{config, inputs}` wrapper, with actual optimizer controls recorded in `effective_optimizer`; `inputs.json` and `protocol.sha256` provide the portable verification records. Equal freezes are accepted. Changed config, source/input hashes or existing immutable snapshots are rejected with a nonzero exit status. Missing evidence is recorded in a frozen partial scope rather than invented to satisfy a completeness checklist.

Every analyze/resume validates the protocol digest, inventory equality, frozen config and every available input hash before fitting. It recomputes deterministic small stages; `--resume` does not bypass verification. A corrupted freeze returns nonzero without replacing an earlier report. Every artifact write uses a temporary sibling then rename; report publication occurs last, so a report-stage exception preserves the previous completed report.

Analysis outputs:

- `heritability.csv`: family/network rows for every registry task and raw/logit scale, fixed-effect design label, support/exclusions, convergence/boundary flags, components, kernel normalization, latent familial fraction, measurement noise fraction, reliability ceiling and bootstrap intervals where supported.
- `parent_offspring.csv`: verified compatible single-parent observed-score pairs; naive/corrected population-moment slopes and intercepts, measurement denominator, counts, paired-error assumption and bootstrap diagnostics. Multi-parent children do not become single-parent pairs.
- `exchange.csv`: audited prediction-based estimation/selection exchange rates, comparison with the paired `no_weight` baseline, evidence tier, bounds, support and unavailable reasons. Every registry skill remains visible if predictions omit it.
- `skill_errors.csv`: per-task MAE, RMSE, MAE/population-SD and untruncated R² from the existing statistical primitive. Compare the common audited model-key intersection across supported tasks and methods within each benchmark/harness/precision/quantization/setting/fold stratum; `common_tasks` and `common_methods` identify that cohort. Preserve full descriptive errors, `n_observed`, `n_audited`, `n_common` and `excluded_from_common_support`. Unsupported requested tasks have explicit unavailable rows. Constant targets retain observed errors while leaving scale-free metrics unavailable. Normalized or unverified historical rows never enter the primary common support.
- `exclusions.csv`, `coverage.json`, `fit_diagnostics.json`, `report.md`: explicit exclusions, observed sample counts, score/prediction verified join fractions, convergence and failed-bootstrap counts. Honest partial results return zero and are labeled `PARTIAL`; invalid/corrupted inputs return nonzero.

Raw accuracy is primary. Logit requires documented ordinary binary counts and uses `(p*n+.5)/(n+1)` with recomputed binomial and delta-method variances. Composite variances never authorize fabricated binomial counts. Family and network models use nonnegative components with fixed known D and explicit boundary solutions; kernel normalization is mean diagonal. Parameter counts enter as centered log counts only when documented for all retained rows; otherwise the compared fits are labeled intercept-only. Familial fractions are covariance descriptions, not causal heritability claims.

Bootstrap units are entire split groups. Each repeated draw is an independent latent family/network covariance block, retaining its within-draw covariance and zero covariance to other draws; resampled family counts count draws. Private draw IDs do not change original source-group labels. Final default is 2,000 replicates, seed 20261028; the 20-replicate override is synthetic-only and recorded. Percentile 95% intervals require at least 90% successful replicates and at least two independent clusters. Small/unsupported designs remain unavailable, with failed replicate counts preserved.

## Compute and launch boundaries

Inputs are the approved small table/graph snapshots plus audited historical CSV/JSON artifacts; no weights or GPU-hours are needed. Before a final 2,000-replicate run, time 20 replicates on representative eligible support in a separately labeled timing protocol and extrapolate without changing the paper freeze. Expected initial assembly reads about 12 MB; measure actual inventory bytes and output storage. If the projected final run exceeds two hours, the person launches it. Queue the exact command, pinned hashes, measured bytes/time estimate, storage, 0 GPU-hours, verification and the same `--resume` command in a concrete launch card only if this parked work is authorized; do not silently lower the final bootstrap count.

The lead runs task/package/batch checks in the sequence required by D7. Task-level validation is `python -m pytest tests/heritability/test_cli.py -q`; the coordinated package check is `python -m pytest tests/heritability -q`, and the batch check is `python -m pytest tests -q`.
