# Zoo-Keeper

Queries and research built on the Model Zoo: a measured, queryable database of
open models. [Model-Zoo](https://github.com/RpM-Kinshuk/Model-Zoo) collects the
measurements and builds the database; Zoo-Keeper queries it and holds the
research that builds on it.

- `handoff/`: start here. Guidance, directives, the checklist, agent prompts,
  and the live status.
- `docs/research/`: the merged research plan and its two source plans.
- `docs/plans/`: implementation plans, added as the tracks write them.
- `scripts/`: small utilities (Hugging Face access checks).

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

Weight neighbors use compatible T1 views within the same architecture
signature. `info` separates measured scores (with revision verification)
from predictions, which are empty in v0. `diff` reports measured changes,
without predicting skill changes. Lineage records carry `verified`,
`detected`, or `declared` evidence tiers. `diff` and `check` remain
`uncalibrated` without matching empirical calibration; supply `--null FILE`
to apply one. This smoke workflow does not complete R6: the R3/R4 pilot
outputs and R5 G0 calibration are still pending.

### Validated example (2026-11-02)

The commands above passed with
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
