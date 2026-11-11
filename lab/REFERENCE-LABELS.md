# Lab table contract and reference labels

Original inventory snapshot, 2026-11-01. Current priorities and completed coverage are in ../steering/CHECKLIST.md and RESULTS.md; the label cautions below still apply.

- `lab/models.csv`: 166 models (seven bases, 112 planned local derivatives,
  45 independent pretrains, and two published sibling references).
- `lab/edges.csv`: 187 planned construction edges.
- `lab/pins.json`: all 54 external models at full commit SHAs, with license
  and configuration evidence hashes.
- `lab/reference_pairs.json`: 180 canonical unordered independent pairs and
  one published sibling pair, with pinned model keys.
- `lab/table_manifest.json`: schema version, counts, content hashes, and the
  immutable table snapshot path relative to `$ZOO_DATA`.

The CSV schema remains `LAB_SCHEMA_VERSION = "1"`. Pass the main checkout's `lab/edges.csv` to `lineage_pairs`; its adjacent `models.csv` supplies the ID mapping. No separate track checkout or merge is needed.

These are the final **planned L2 inventory tables**, not an assertion that the
local checkpoints, manifests, or fingerprints have been generated. Real calibration/performance metrics need the corresponding measured outputs. Do not treat an inventory row as completed
measurement coverage.

## Reference labels and the independent null

The frozen CSV `kind=independent` category contains the external hard-negative
references, including the two siblings, as required by L11. It alone is not a
certificate of independent roots. For the independent null use the 45
PolyPythias references, identified by `reference_type=polypythias` in `pins.json`
and by the independent model IDs/pairs in `reference_pairs.json`.

Exclude both `posttraining-sibling-slot-*` rows from that null. They are separate
DPO continuations of the published Zephyr SFT model: Notus and the Alignment
Handbook Zephyr DPO reproduction. Their same-root label is positive; immediate
parent and ancestor labels between them are negative. Published training records
support that relationship; no byte-level historical training-revision claim is
made. Their pinned public SFT reference and the historical revision limitation
are recorded in `lineage_evidence`.

Sibling/grandparent pairs within the constructed lab still come from the exact
construction edge graph. Published-reference evidence stays separate from lab
construction truth. Any reserved final subset is absent from these tables; preserve its exposure boundary and use independent scoring under WORKFLOW.md.
