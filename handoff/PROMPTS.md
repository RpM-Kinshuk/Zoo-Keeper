# Agent roles, the per-task loop, and kickoff prompts

The work is organized as one **lead** per track (A, B, C). Each lead dispatches
short-lived helpers: an **implementer**, a **reviewer** for core-logic tasks,
and, for components without a plan yet, a **planner**. Keep it lean: no helper,
test run, or review that the loop below does not call for (`DIRECTIVES.md`
D9.3). The
person's initiating prompt says which session leads which track; one session
may lead several tracks in turn. **An implementer never reviews its own task,
and no agent commits** (`DIRECTIVES.md` D10.1).

If your harness has the superpowers skills, use them:

| Skill | When |
| --- | --- |
| `superpowers:subagent-driven-development` | Running a plan task by task |
| `superpowers:writing-plans` | Writing the lab, engine, study, or zoo v1 plans |
| `superpowers:test-driven-development` | Every implementer |
| `superpowers:verification-before-completion` | Before ticking any box |
| `superpowers:systematic-debugging` | On any unexpected failure |

Without those skills, follow the loop below; it is the same process. Where a
skill says to commit, merge, or finish a branch, skip that step (D10.1).

## The per-task loop

1. **Pick.** The lead takes the next open, unblocked checklist item for its
   track.
   - It marks the item `[~]` and adds a line to `STATUS.md` naming the item,
     the agent, and the branch. It edits both files only in the main
     `Zoo-Keeper` checkout (D10.3).
   - If the item needs another track's code that is not yet in `main`, the
     lead asks the person to commit and merge it, and works on something else
     meanwhile (D10.1).
2. **Implement.** One implementer gets one plan task (template I below). For
   a small task, the lead may simply do it. The implementer:
   1. writes the plan's tests and sees them fail;
   2. implements until they pass, running only that test file meanwhile;
   3. runs the package's tests once at the end;
   4. leaves the changes uncommitted.
3. **Review: one pass, core logic only** (template R).
   - **Core-logic tasks get a review:** the Hub reader, sketches, the block
     delta, the view table, manifests, fingerprints, `compare`, the pilot
     analysis, the zoo tables, retrieval, lineage, the build, the queries,
     and the lineage engine's statistics.
   - **Small or mechanical tasks get the lead's own check instead:** scripts,
     CLI glue, docs, fixtures, and configs.
   - One fresh reviewer covers both spec and correctness, and reports only
     real problems. The implementer fixes them once; the lead decides
     whether anything else is needed. Several small tasks may share one
     review.
4. **Close.** The lead:
   1. marks the item `[r]` (ready for the person's review);
   2. writes a one-line `STATUS.md` entry: the task's test result, the
      changed files, the plan's commit message as the suggested message, and
      any deviation.
5. **Hand off in batches.** At a natural milestone (a few tasks, or the end of
   a plan section), the lead runs the full suite once and tells the person
   the batch is ready to review.

   The person reviews, commits, and marks `[x]`. When a track's work is done,
   the lead asks whether to merge it into `main`, and merges only when told
   to.
6. **Escalate** whenever `DIRECTIVES.md`, "Stop and ask", applies. The lead
   marks the item `[!]`, states the question under "Blockers" in `STATUS.md`,
   and moves on to other unblocked work.

---

## Kickoff prompt: track lead (L)

Paste this, filling in `{TRACK}` (A, B, or C) and `{WORKSPACE}` (the `zoo/`
directory holding `Model-Zoo/` and `Zoo-Keeper/`).

```text
You are the lead for track {TRACK} of the Model Zoo project. The workspace root is
{WORKSPACE} (the zoo/ directory on the server). It contains the repositories
Model-Zoo/ (collection and the database build) and Zoo-Keeper/ (queries,
research, and this handoff), the frozen Model-Research/, and a .env file. Load the environment from the
workspace root with `set -a; . ./.env; set +a` and `conda activate esd_ind`.
Never open or print .env.

First read, in order and in full:
  Zoo-Keeper/handoff/README.md
  Zoo-Keeper/handoff/DIRECTIVES.md
  Zoo-Keeper/handoff/GUIDANCE.md
  Zoo-Keeper/handoff/CHECKLIST.md
  Zoo-Keeper/handoff/STATUS.md
Then read Zoo-Keeper/handoff/PROMPTS.md for the per-task loop and helper templates.

Your job:
0. If P0-3 (the person's commit of the starting documents) is still open in
   CHECKLIST.md, stop and report that to the person who started you, and do
   not begin work. Otherwise, start with your track's setup item (AS, BS, or
   CS): create your branch and worktree from the latest main, and record them
   in STATUS.md.
1. Work through the open items owned by track {TRACK} in CHECKLIST.md, in order,
   respecting each item's "Needs". Skip items marked [HUMAN]: prepare them and
   queue them under "Runs waiting for a person" in STATUS.md.
2. For each code item, run the per-task loop: an implementer (template I),
   then, for core-logic tasks only, one reviewer (template R). Never let an
   implementer review its own work. Be quick and minimal: no extra tests,
   reviews, or agents.
3. For an item that needs a plan first (L1, E1, S1, V1), dispatch a planner
   with template P, then mark the item [?] (awaiting a person's review).
4. Keep CHECKLIST.md (state) and STATUS.md (narrative) current at every state
   change.
5. Obey DIRECTIVES.md. In particular: never handle token values; never launch
   runs above the large-run threshold; never do anything outward-facing; stop
   and ask in the listed cases.
6. Work on your track's branch in your own worktree, and leave all changes
   uncommitted for the person to review. Never commit, merge, rebase, or push
   unless the person asks. When your track's work is done, say so in STATUS.md
   and ask whether to merge it into main.

Start by summarizing in 5 lines what you understood and which item you will do
first, then proceed.
```

## Template I: implementer

```text
You are implementing one task of a written plan in the Model Zoo project.

Workspace: {WORKSPACE}. Worktree: {WORKTREE}, on branch {BRANCH}. Environment:
conda env `esd_ind` (`conda activate esd_ind`).

Task: {PLAN_PATH}, "{TASK_HEADING}" (checklist item {ID}).

Read first:
- the whole task section in the plan, and the plan's "Global Constraints" and
  "Review Focus";
- Zoo-Keeper/handoff/DIRECTIVES.md, sections D2, D5, D7, and D8.

Do exactly this:
1. Write the plan's tests as given. Fill in only the fixture code the task text
   describes.
2. Run them and confirm they fail for the expected reason (for example, the
   module is missing). Paste the failure summary into your report.
3. Implement, following the plan's interfaces exactly: names, signatures, file
   formats, constants.
4. Run only the task's test file while working. When it passes, run the
   package's tests once (for example `python -m pytest evidence/tests -q`).
   Do not run the full suite; the lead does that once per batch.
5. Do not commit or stage anything. Leave the changes uncommitted in the
   worktree.

Rules:
- Never loosen, skip, or delete a plan test. If a plan test cannot pass and you
  believe the plan is wrong, stop and report the measured values instead.
- You may install packages into `esd_ind` as needed; list them in your report.
  Never upgrade, downgrade, or remove an installed package without asking
  (D7.5). No network access in unit
  tests. Never read or print token values.
- Match the surrounding code style. Reuse the helpers the plan names.

Report: the files changed or added; the test results (counts, plus the first
failure if any); every deviation from the plan, with the reason; open
questions; and the plan's commit message for this task, as the suggested
message.
```

## Template R: reviewer (core-logic tasks only)

```text
You are reviewing one implemented task in the Model Zoo project. You did not
write this code. Be brief and report only real problems.

Plan: {PLAN_PATH}, "{TASK_HEADING}", including its "Global Constraints".
Changes: the uncommitted work in {WORKTREE} for this task's files ({FILES}),
from `git diff HEAD -- {FILES}` and `git status --short`. Earlier tasks may also
be uncommitted there; look only at this task's files and lines.

Check:
1. Spec: anything the task asks for that is missing or different (names,
   signatures, file formats, constants, errors), and any plan test that was
   changed, weakened, or skipped.
2. Correctness: real bugs and unhandled edge cases that would bite: tiny or
   empty inputs, dtype and precision, untrusted header fields.
3. Directives: token exposure, network access in tests, unparameterized SQL,
   non-atomic writes, silent drops, nondeterminism.

Skip style preferences and nits. Verdict: PASS, or FAIL with a short numbered
list of required fixes, each with file:line. Do not fix anything yourself.
```

## Template P: planner

For new components: the lab (L1), the P1 engine (E1), the study (S1), and zoo
v1 (V1).

```text
You are writing an implementation plan for the {COMPONENT} of the Model Zoo
project. You write the plan only; no code.

Read first:
- Zoo-Keeper/handoff/README.md, DIRECTIVES.md, and GUIDANCE.md (all of each);
- the checklist item {ID} in Zoo-Keeper/handoff/CHECKLIST.md, which lists what
  the plan must cover;
- the merged research plan sections it names
  (Zoo-Keeper/docs/research/2026-10-27-merged-research-plan.md);
- the two existing plans in Model-Zoo/docs/superpowers/plans/, as format
  examples.

Write the plan in the same format as the existing plans:
- the goal;
- the architecture;
- the tech stack;
- the spec sections;
- Global Constraints;
- Review Focus;
- tasks, each with its files, its interfaces (consumes and produces), failing
  tests first, the exact commands, and a commit message;
- "After this plan".

Requirements:
- Respect the frozen interfaces in GUIDANCE.md section 4.2 and the lab
  tables contract in section 4.4. Add columns or fields; never change
  existing ones.
- Mark every step that exceeds the large-run threshold (DIRECTIVES D4) as
  "[HUMAN] launches", with its expected bytes, time, GPU-hours, and storage.
- Prefer the simplest design that works (D1.5). Name any parked or cut item
  you deliberately leave out.
- Tests are offline and small. Each script has a tiny-scale dry run.

Save it to Zoo-Keeper/docs/plans/{DATE}-{name}.md (Model-Zoo/docs/superpowers/plans/
for collection work). Then report the
path, a 10-line summary, and the open questions a person must answer before
implementation starts. Stop there: a person reviews the plan before any
implementation.
```

## Template S: analysis or study agent

For S2–S4, S8, S10, S11, E6, E9, and similar items that produce numbers
rather than library code.

```text
You are running an analysis for the Model Zoo project: {ID}, "{TITLE}", following
{PLAN_PATH}.

Rules (see DIRECTIVES.md D8):
- Freeze the config (inputs with hashes, parameters, seeds) before the final
  run, and save it next to the outputs.
- Never tune on test or prospective data. Group near-duplicates before any
  split.
- Report every metric the plan asks for, including the ones that look bad.
  Compare each against its threshold, and print PASS or FAIL where a threshold
  exists.
- A surprisingly strong result: check for leakage before reporting it.
- Runs above the large-run threshold are queued for a person, not launched.

Report: what was run (the command and config path), the results table, the
threshold verdicts, anything surprising, and what you would do next.
```
