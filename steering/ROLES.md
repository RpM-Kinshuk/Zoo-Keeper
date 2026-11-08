# Roles

Four roles. One person, the overseer, runs the agent sessions.

## Overseer (the owner)

- Reads `BOARD.md`, and opens a card only to decide it.
- Answers cards: "approve", "pick B", "you decide" (where allowed), or a
  question back to the steward. The overseer can also ask the steward
  anything in plain words, or paste an agent's question for it to handle.
- Runs launch cards: pastes the command, then tells the steward it finished.
- To guide a track, tells it "read your brief". The steward writes the brief.
- Reviews and commits the work. Agents never commit.
- Owns the direction. Claims, paper targets, dates, money and publishing are
  always the overseer's call (class 3).
- Checks in with the assistant once in a long while, using the board's
  "Since the last check-in" section.

## Steward

The steward coordinates the tracks day to day, guides them through briefs,
handles questions the card way, keeps the research verified, and protects the
overseer's attention. **Its full manual, context and glossary are in
`STEWARD.md`.** The overseer starts it on Mondays and Thursdays, and whenever
a track reports a claim-changing result.

## Track leads (A, B, C)

Scopes through Dec 18 (`PLAN.md` holds the items):

| Track                                    | Scope                                                                                                                                                                                                                                                                                                                                                                                    | Repository                             |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **A**: tool and data                     | Lab data to Hugging Face (K0); real look-alike models (K6); score verification by weight hash (K10); cost gate and the 5k candidate pool (K8: launch card, then analysis); zoo builds. From mid-December: the zoo v1 plan (K15: `check` command, GGUF, duplicates report).                                                                                                               | Model-Zoo, plus Zoo-Keeper for queries |
| **B**: lab methods and the P1 evaluation | In order: the quick re-tests on existing data (K3 merges on deltas, K4 tree test, K14 disguise check); then edge typing, the exact-mark tests and leave-the-parent-out (K5, the main job); the P1 pre-registration (K12); order signals for ordinary fine-tunes (K9). If B falls behind, cut K9 to "order by Hub upload time only", and leave the Witness Overlap comparison to track C. | Zoo-Keeper                             |
| **C**: baselines and truth sets          | In order: finish the literature check (`CLAIMS.md`); Trace and Spectral Signatures (ported from Model-Research); a first verified Hub truth set at 7B and above, including refusal-removed and quantized Hub models with known parents (K7); modelDNA and "Training Leaves Traces"; Witness Overlap (K2). Cut list in card D3. Heritability is parked.                                   | Zoo-Keeper                             |

**Each lead:**

- **At the start of every session,** reads `briefs/<track>.md`, the steward's
  current directions, and confirms with one line in its track note
  ("Brief of <date> read").
- Works `PLAN.md` items in order, using the lean loop in `handoff/PROMPTS.md`.
- Writes every test's pass/fail rule **before** running it, as `CLAIMS.md`
  requires, together with the noise level each threshold is set against.
- **Keeps `tracks/<track>.md` to 30 lines or fewer,** rewritten (not
  appended), with four parts:
  - **Now.**
  - **Results since the last steward run:** one line each, in the form "what
    was measured: number vs target, what it means", with a link.
  - **Next.**
  - **Blocked or needs.**
- Adds one row per reportable number to `RESULTS.md`.
- **Decides class 1 alone.** Notes class 2 in its track file. For class 3,
  drafts a card in `cards/` marked `draft`, then moves to the next item.
- Never writes to `handoff/STATUS.md` or `handoff/CHECKLIST.md` (frozen).
  Never changes truth tables, held-out splits, or evaluation thresholds; those are class 3.

## Skeptic (on demand)

A fresh agent session, or a subagent the steward starts, that did not do the
work. It gives the second opinion on every class-3 card and on any result that would change a claim, in two lines.

---

## Kickoff prompts

Replace `{WORKSPACE}` with the `zoo/` directory.

### Steward

```text
You are the steward of the Model Zoo project. Workspace: {WORKSPACE}.
Read Zoo-Keeper/steering/STEWARD.md in full first: it is your manual,
context and glossary. Then read README.md, DIRECTION.md, CLAIMS.md, PLAN.md,
DECISIONS.md, BOARD.md, RESULTS.md, tracks/*.md, briefs/*.md and the open or
draft cards in cards/, all in Zoo-Keeper/steering/.

Do one run as described in STEWARD.md section 4. Write only inside
Zoo-Keeper/steering/. Never open or print .env, and never implement, launch,
commit or edit code. End with a summary of at most five lines for the
overseer, in plain words. If the overseer asks you anything, answer the card
way (STEWARD.md section 3).
```

### Track lead

```text
You lead track {TRACK} of the Model Zoo project. Workspace: {WORKSPACE}.
Load the environment with `set -a; . ./.env; set +a` and `conda activate esd_ind`; never open or print .env.

Read, in order: Zoo-Keeper/steering/briefs/{TRACK}.md (your current
directions), steering/README.md, DIRECTION.md section 1, CLAIMS.md (the
claims your items test), PLAN.md (your items), DECISIONS.md, ROLES.md (Track
leads), DATA.md if you store data, and tracks/{TRACK}.md. For technical
rules, read Zoo-Keeper/handoff/DIRECTIVES.md; it still applies except where
steering/README.md says otherwise.

Work your items in order on your track branch and worktree. Leave everything
uncommitted for the overseer. Write each test's pass/fail rule before running
it. Keep tracks/{TRACK}.md current (30 lines or fewer, rewritten), and add
one row to RESULTS.md per reportable number. Decide class-1 choices yourself;
for class 3, draft a card in cards/ and move on. Be quick and minimal. Start
by confirming the brief and stating in three lines what you will do first.
```

### Skeptic

```text
You are a skeptic giving a second opinion. You did not do this work.
Read the card or result at {TARGET} and the evidence it links. In at most two
lines: agree or disagree, and name the single most important thing it
misses, overclaims, or underestimates for the project's big picture (the
compass in Zoo-Keeper/steering/README.md). Edit only the "Second opinion"
line, if it is a card.
```
