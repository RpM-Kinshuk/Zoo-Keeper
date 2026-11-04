# Handoff: building the Model Zoo (start here)

**Continuing Track A:** start with [TRACK-A.md](TRACK-A.md). The dated orientation below is historical.

This folder holds what a new group of agents needs to continue the Model Zoo
work: what we are building and why, the rules, and the checklist of all the
work. It was written on 2026-10-31, when the plans were complete and no code
existed yet.

## What we are building

The **Model Zoo** is a measured, queryable database of open models, plus a
library and CLI that answer practical questions about them:

- which models fit my constraints;
- what this model is like;
- where it came from;
- what changed between two models;
- whether its declared base is right.

Each model gets a small **fingerprint** read from a few percent of its weight
files over HTTP range requests, so the zoo can grow to tens of thousands of
models of every type. The research papers (lineage first, then the zoo
release, then model selection) are built on the zoo and organize it.

## Read in this order

| # | File | What it gives you | Time |
| --- | --- | --- | --- |
| 1 | `README.md` | Orientation, current state, key dates | 5 min |
| 2 | `DIRECTIVES.md` | The rules: what you must and must never do, how to stay lean and portable, and when to stop and ask | 10 min |
| 3 | `GUIDANCE.md` | The product, the key ideas, interfaces, settled decisions, known traps | 20 min |
| 4 | `CHECKLIST.md` | All the work, with owners, dependencies, due dates, and "done when" | 15 min |
| 5 | `PROMPTS.md` | Agent roles, the per-task loop, and ready-to-paste kickoff prompts | 10 min |
| 6 | `STATUS.md` | The live log: in progress, blockers, decisions, changes, runs waiting for a person | 2 min |

Then read the source document for the task you pick (table below). Read only
the sections you need. The plans are long, but each task in them is
self-contained.

## Source documents

All paths are relative to the workspace root, `zoo/`, which holds the
repositories side by side (`Model-Zoo/`, `Zoo-Keeper/`, and the frozen
`Model-Research/`).

| Document | Path | Use it for |
| --- | --- | --- |
| Merged research plan (v3) | `Zoo-Keeper/docs/research/2026-10-27-merged-research-plan.md` | Research decisions, the rationale behind them, gates, and thresholds. Section 1.1 is the committed plan; section 12 is the schedule. |
| Extractor implementation plan (v2) | `Model-Zoo/docs/superpowers/plans/2026-10-28-weight-evidence-extractor.md` | Code-level spec for `Model-Zoo/evidence/`, Tasks 1–10 |
| Zoo store and v0 queries plan | `Model-Zoo/docs/superpowers/plans/2026-10-31-zoo-store-and-queries.md` | Code-level spec for the database build (`Model-Zoo/zoo/`, Tasks 1–6) and the queries (`Zoo-Keeper/zoo_keeper/`, Tasks 7–8) |
| Source Plans A and B | `Zoo-Keeper/docs/research/2026-10-23-census-and-query-plan.md`, `.../model_zoo_frontier_research_plan_2026-10-23.md` | Background only. Superseded by the merged plan; treat their claims as hypotheses. |
| Existing collector | `Model-Zoo/README.md`, `Model-Zoo/docs/operations/analysis.md`, `Model-Zoo/esd_experiment/`, `Model-Zoo/net_esd/` | Code the extractor reuses (T2 spectra, pins, file names) |
| Frozen study code and data | `Model-Research/README.md`, `Model-Research/spectral_research/`, `Model-Research/data/` | Source to port from: the Trace and Spectral Signatures baselines, and the leaderboard data. Nothing new goes into Model-Research. |

## When documents disagree

1. A human's direct instruction wins.
2. Then `DIRECTIVES.md`.
3. Then the implementation plan for the task. It is authoritative for names,
   signatures, file formats, and tests.
4. Then the merged research plan. It is authoritative for research decisions
   and thresholds.
5. Then `CHECKLIST.md`, `PROMPTS.md`, and `GUIDANCE.md`. They organize and
   summarize; if one disagrees with 2–4, it has a bug. Log it in `STATUS.md`.

If two sources at levels 3–4 disagree in a way that changes behavior, stop and
ask (`DIRECTIVES.md`, "Stop and ask a person when").

## State on 2026-10-31

**Done:**
- the research plan (v3, re-centered on the zoo);
- the extractor plan (v2) and the zoo store plan;
- the gated-access requests for the owner's account (filed and handled by the
  team);
- this handoff, including interface fixes to both plans (logged in `STATUS.md`);
- support for System One decision models and other trending model types,
  checked on 28 live repositories (merged plan §6.8).

**Not started:**
- all code (`evidence/`, `zoo/`);
- the genealogy lab;
- the P1 lineage engine;
- the heritability and exchange-rate study.

The lab, engine, and study still need implementation plans, and writing those
plans is the first item in each of their checklists.

**Behind schedule:** week 1 (Oct 29–Nov 6) is half over. The fingerprint
format still freezes on Nov 6 so that track B can build on it, which makes
extractor Tasks 1–3 and the day-one checks urgent.

**Uncommitted:** these files are not in git yet:
- in `Model-Zoo`: the two plans (untracked);
- in `Zoo-Keeper`: this folder, the research plans under `docs/research/`, and
  the access scripts under `scripts/`;
- in `Model-Research`: the moved research plans (shown as deleted), plus
  pointer edits to its READMEs.

A person commits them before agents start (`CHECKLIST.md`, P0-3).

## Workspace

- `Model-Zoo/` collects: the existing collector, the extractor
  (`evidence/`), and the database build (`zoo/`). Small databases may live in
  it, like today's curated CSVs.
- `Zoo-Keeper/` builds on the collection: the queries (`zoo_keeper/`), the
  lab, the lineage engine, studies, baselines, the research plans
  (`docs/research/`), new plans (`docs/plans/`), and this handoff.
- `Model-Research/` is frozen and will be retired. It is a source to port
  from.
- **The work runs on a remote Linux server,** in a workspace directory `zoo/`
  that holds the three repositories, `worktrees/` (made by the tracks), and
  `.env`. The big database lives parallel to the repositories, under
  `$ZOO_DATA`.
  - `.env` holds three keys: `HF_TOKEN`, `ZOO_DATA` (the data root), and
    `HF_HOME` (the Hugging Face cache).
  - Agents start in `zoo/` and load it with `set -a; . ./.env; set +a`. They
    never open or print it (`DIRECTIVES.md` D2).
- Where data goes is in `GUIDANCE.md` 4.5.
- A cluster copy at `/jumbo/yaoqingyang/kinshuk/` is the source of truth for
  historical collector outputs. The local `svd_results` copy is incomplete:
  1,606 of 3,817 files are zero bytes.

## How to work

1. Pick the next unblocked checklist item for your track.
2. Follow the lean per-task loop in `PROMPTS.md`: test first, implement, and
   one review for core-logic tasks only. Leave the changes **uncommitted** on
   your track's branch; the person reviews and commits (`DIRECTIVES.md`
   D10.1).
3. Mark the item `[r]` in `CHECKLIST.md` and add a line to `STATUS.md`. Both
   are edited only in the main `Zoo-Keeper` checkout (`DIRECTIVES.md` D10.3).
4. Anything marked `[HUMAN]`, anything above the large-run threshold, and
   anything outward-facing goes to a person. Write the exact command and its
   expected cost under "Runs waiting for a person" in `STATUS.md`.

## Key dates

| Date | What |
| --- | --- |
| **Nov 6** | Fingerprint format frozen (extractor Task 6 layout). Lab, engine, and study plans reviewed by a person. |
| **Nov 27** | Pivot check: is model evolution tree-like on the lab? |
| **Dec 4** | **Gate G0.** Zoo v0 is usable internally. The lineage method is frozen and the prospective set starts. |
| **Dec 11** | **Gate G1.** The minimum lab (about 120 models) is complete. |
| **Dec 18** | Pivot check: does P1 beat the baselines by the stated margin? |
| **Jan 4–8** | Results freeze |
| **~Jan 22 / ~Jan 28** | ICML 2027 abstract and paper deadlines (estimated; verify when the call is out) |
| **Feb–Mar** | Zoo v1 goes public, plus the zero-byte note. ICLR 2027 workshop paper (heritability). |
| **~early May** | P3, the zoo release, at NeurIPS 2027 Evaluations & Datasets (estimated) |
| **~late Sep** | Selection paper at ICLR 2028 (estimated) |
