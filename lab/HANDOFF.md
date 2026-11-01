# L2 table handoff for Track A

Prepared on 2026-11-01 in Track B:

- `lab/models.csv`: 166 models (seven bases, 112 planned local derivatives,
  45 independent pretrains, and two published sibling references).
- `lab/edges.csv`: 187 planned construction edges.
- `lab/pins.json`: all 54 external models at full commit SHAs, with license
  and configuration evidence hashes.
- `lab/reference_pairs.json`: 180 canonical unordered independent pairs and
  one published sibling pair, with pinned model keys.
- `lab/table_manifest.json`: schema version, counts, content hashes, and the
  immutable table snapshot path relative to `$ZOO_DATA`.

The CSV schema remains `LAB_SCHEMA_VERSION = "1"`. Track A can pass
`worktrees/Zoo-Keeper-track-b/lab/edges.csv` to `lineage_pairs`; its adjacent
`models.csv` supplies the ID mapping. Neither Track A nor Track B needs to be
merged for this file handoff. No Track A Python code was imported to produce it.

These are the final **planned L2 inventory tables**, not an assertion that the
local checkpoints, manifests, or fingerprints have been generated. X9
implementation can use this contract now; real calibration and G0 metrics need
the corresponding measured outputs. Do not treat an inventory row as completed
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
construction truth. The blind subset remains C's responsibility and is absent
from these tables.
