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
