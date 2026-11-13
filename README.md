# Zoo-Keeper

Queries and research built on the Model Zoo: a measured, queryable database of
open models. [Model-Zoo](https://github.com/RpM-Kinshuk/Model-Zoo) collects the
measurements and builds the database; Zoo-Keeper queries it and holds the
research that builds on it.

The end product is the database **and its query framework**, for finding, comparing, understanding and choosing models. The long-term ambition is50k–100k+ models; expansion follows useful query results and measured costs.

- Start with [steering/README.md](steering/README.md): one doer, independent reviewer, one checklist.
- [steering/DIRECTION.md](steering/DIRECTION.md): product/research goals and paper readiness.
- [steering/BOARD.md](steering/BOARD.md): concise current outlook.
- [docs/README.md](docs/README.md): technical contracts and preserved scientific references.
- `scripts/`: scientific reproduction drivers and current access checks, not a second queue.

**Setup on a server:**
1. Clone this repository next to `Model-Zoo/` in one workspace.
2. Add a `.env` there with `HF_TOKEN`, `ZOO_DATA`, and `HF_HOME`.
3. In the `esd_ind` environment, run `pip install -e ../Model-Zoo`.

The data lives under `$ZOO_DATA`, outside the repositories.

## Query the zoo

Queries require DuckDB (pinned in `requirements.txt`) and the editable
Model-Zoo dependency from setup above. They read tables and fingerprints
offline; no model weights or GPU are needed. Build instructions and table
descriptions are in Model-Zoo's `docs/operations/zoo.md`.

Current snapshots expose `row_block_fingerprint`, `full_weight_fingerprint`,
and `full_weight_spectra`. Readers also support schema 0/1 snapshots, normalizing
legacy columns and measurement labels in memory without rewriting published files.
Historical reports use T0 for metadata-only evidence, T1 for row-block evidence,
and T2 for full-weight evidence; their recorded artifact paths stay unchanged.

Set `ZOO_DATA` to the data root holding the original evidence. Every command
defaults to `$ZOO_DATA/builds/current`; pass `--zoo "$ZOO"` for another build.
`--data_root DIR` overrides the evidence root. For example:

```bash
ZOO="$ZOO_DATA/builds/current"
MODEL=lab/smol-135-gpu-v1-lora-dialogue-r8
BASE=HuggingFaceTB/SmolLM2-135M

python -m zoo_keeper.cli find --zoo "$ZOO" --text smol --limit 10
python -m zoo_keeper.cli info --zoo "$ZOO" "$MODEL" --json
python -m zoo_keeper.cli similar --zoo "$ZOO" "$MODEL" -k 5
python -m zoo_keeper.cli family --zoo "$ZOO" "$MODEL" --depth 2
python -m zoo_keeper.cli diff --zoo "$ZOO" "$MODEL" "$BASE" --json
python -m zoo_keeper.cli check --zoo "$ZOO" "$MODEL" --json
```

`MODEL` and `BASE` above have one revision in the current snapshot. A bare
repository name matching several revisions is an error listing the choices;
use its exact `repo_id@revision` key. Add `--json` to any command for one JSON
document; otherwise tables or indented cards are printed. Query/input errors
return 2 on stderr; an `inconsistent` check returns 1, and success returns 0.

Weight neighbors use compatible row-block views within the same architecture
signature. `info` separates measured scores (with revision verification)
from predictions, which are empty in v0. `diff` reports measured changes,
without predicting skill changes. Lineage records carry `verified`,
`detected`, or `declared` evidence tiers. `diff` and `check` remain
`uncalibrated` without matching empirical calibration; supply `--null FILE`
to apply one. Current result scope and remaining calibration work are recorded in steering/RESULTS.md; a smoke workflow does not establish broad reliability.

### Find candidates, then inspect evidence

For example, shortlist accessible Apache-2.0-labelled Qwen2 text models below
one billion parameters:

```bash
python -m zoo_keeper.cli find --zoo "$ZOO" --modality text \
  --license apache-2.0 --params_max 1000000000 --format safetensors \
  --access granted --architecture qwen2 --json
```

These filters use recorded metadata. Missing size or modality stays unknown and
is excluded by that filter; an empty result does not prove that suitable models
do not exist. Supply the builder's `--frame` input for those metadata fields.
License strings describe the catalog metadata, not a legal conclusion.

`find --min_score` filters the mean of matching score rows (`--skill` narrows
those rows); results are ordered by downloads, not score. Default score filters
use revision-verified rows; `--include_unverified` opts into other reported rows.
An average can mix
benchmarks or evaluation settings. Use `info MODEL --json` to inspect each
benchmark, metric, source revision and harness before comparing performance.
Revision verification links a score to a pinned model revision; it does not establish
matching prompts, task versions, shots or other evaluation settings. Unknown
settings and missing scores remain unknown, and predictions are separate.

Use `diff` for measured weight changes and `similar` for compatible neighbors.
A close neighbor or mixture fit does not establish the actual construction
source or a skill gain. `check` compares the first declared base it can resolve;
it does not validate every merge ingredient or the full model history.

### Find by recorded weight-file storage

From this checkout, shortlist recorded nongated Apache-2.0/MIT safetensors
entries whose selected weight files total at most2GB:

```bash
python -m scripts.find_weight_budget \
  --zoo "$ZOO_DATA/runs/track-a-expanded-r6-20261105/zoo" \
  --data-root "$ZOO_DATA" --max-weight-bytes 2000000000 --limit 10
```

This offline adapter checks each manifest against the snapshot's hash and exact
model identity, then applies the budget before taking the first results in
recorded download-count order (unknowns last). Local pins and recorded adapters are excluded; unknown
file sizes stay unknown. The budget covers selected files, not a complete model
package, RAM/VRAM or proven loadability. Access/license/adapter flags are recorded
metadata. Missing parameter counts do not become guessed sizes or skill scores.
The703-entry check found95eligible entries; all ten returned entries lack stored
benchmark scores. Evidence and fixed info/compare outputs:
`$ZOO_DATA/runs/catalog-budget-20261112/REPORT.md`.

### Fresh query example (2026-11-12)

A measured local example is available at
`$ZOO_DATA/runs/query-journey-20261112/catalog`. It contains three pinned
Qwen0.5B models, each with explicit size/modality metadata. The new model was
measured during this run; the two reference fingerprints and validated headers
were reused. One cold driver, including fresh metadata, weight reads, catalog
building and seven in-process CLI calls, took62.13s and requested115,277,824
new tensor bytes. This is a small reference catalog, not703-way retrieval.

```bash
ZOO="$ZOO_DATA/runs/query-journey-20261112/catalog"
MODEL=adityabanerjee13/qwen2.5-0.5b-cpt-mix-1to1@56d3eaf20ab495fa26e63e82412268b6de537d48
python -m zoo_keeper.cli info --zoo "$ZOO" "$MODEL" --json
python -m zoo_keeper.cli similar --zoo "$ZOO" "$MODEL" -k 2 --json
python -m zoo_keeper.cli diff --zoo "$ZOO" "$MODEL" Qwen/Qwen2.5-0.5B --json
```

The constrained `find` example above returns all three models. The fresh model
has no benchmark scores or predictions; `diff` reports169 changed measured
views and uncalibrated relatedness. The default score filter returns no models:
the six preserved base scores have unverified evaluation revisions. Full
outputs, provenance, timing and limitations are in the adjacent run report.
This validates one query path, not a performance recommendation or broad source
accuracy.

### Validated example (2026-11-02)

The original Smol command sequence in “Query the zoo” passed with
`ZOO="$ZOO_DATA/runs/track-a-z8-cli-20261102.c45u9pog/zoo"`.
This snapshot has 85 models: 43 measured and 42 metadata-only. It retains all
43 full-spectrum links and imports 18 leaderboard score rows, six with matching
evaluation revisions. Observed results for the example model:

| Command | Result |
| --- | --- |
| `find --text smol --limit 3` | 3 models |
| `info` | Measured coverage; no predictions |
| `similar -k 3` | 3 neighbors; closest is the instruction LoRA r8, cosine 0.999992 |
| `family --depth 2` | 1 ancestor, 0 descendants, 10 same-root models |
| `diff` against the base | 151 identical views, 60 changed; uncalibrated |
| `check` | Uncalibrated |

Each command returned 0 and emitted one valid JSON document with `--json`.
Identical views match on their measured blocks.
Unknown and ambiguous model names returned 2 with stderr only. Exact commands,
JSON outputs and timings are saved beside the snapshot in `commands.json`,
`*.stdout` and `cli-smoke-report.json`. This is an artifact/CLI check; R6 remains
open until the pilot outputs and calibration are available.
