# Decisions

This protocol fixes two failure modes:

- the overseer receives choices with too little or too much detail, and has
  to hand them back;
- agents decide alone and miss the big picture.

The split: **agents supply the technical knowledge and a recommendation; the
overseer decides on the big picture** (the compass in `README.md`). The steward sits between them and makes every card readable and honest.

## 1. Who decides what

| Class | Who decides                                                     | When it applies                                                                                                                                                                                            | How it is recorded                                               |
| ----- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **1** | The track agent alone                                           | Reversible within a day; inside its own track; no change to claims, papers, dates, money, shared formats, or another track                                                                                 | One line in the track's note                                     |
| **2** | The track agent, then the steward checks it against the compass | Reversible, but touches another track, the order of `PLAN.md`, a shared file format, or more than 20 GPU-hours                                                                                             | A one-line "for your information" entry on `BOARD.md`            |
| **3** | The overseer                                                    | Changes a claim, paper, venue, or date; spends money; cannot be undone or is outward-facing (making anything public, contacting anyone); decides a gate or a pivot; deletes data that may be the only copy | A card in `cards/`, listed under "Waiting for you" on `BOARD.md` |

**When unsure, pick the higher class.** An agent that is unsure which class
applies asks the steward, not the overseer.

**Handing a decision back.** The overseer may answer a class-3 card with "you
decide". The steward then decides by the compass, after a skeptic's check, and writes the reasoning in the log.

- This is allowed only when the choice is reversible and not outward-facing.
- Money, publishing, deleting a sole copy, and the P1 venue call are never
  handed back.

**Batching.** Cards reach the overseer in batches, at most twice a week
(Monday and Thursday). The exceptions: work blocked for more than a day, a
safety or data-loss risk, or a deadline within 48 hours.

## 2. Rules for anything sent to the overseer

- Lead with the question and the recommendation. Put reasons after.
- Use plain words. Explain any technical term in a few words in parentheses,
  or leave it out.
- Put every number next to its target, then say in one line what it means
  for the claims or papers.
- Offer at most three options. If the agent cannot recommend one, it says
  what information would let it.
- Fit everything on one screen, with technical detail behind a link
  ("Details for agents").

Overseer preference (Nov 8): keep language plain and detail low, but include
one short sentence saying what the work is for and why the update matters.
Do not assume the overseer remembers task IDs, method names or research terms.
Use a plain description first; explain any necessary unfamiliar term where it
appears. Put technical IDs in parentheses or links, not in place of context.

- Always name a default for when no answer arrives. A choice that cannot be
  undone has no default: the card says what it blocks.

## 3. Card templates

Cards live in `cards/` as `D<n>-<short-name>.md` (decision),
`G<n>-<short-name>.md` (gate), or `L<n>-<short-name>.md` (launch). The
steward numbers them and marks them `open` or `decided` at the top.

### Decision card (class 3)

```
# D<n>: <the question in plain words, 15 words or fewer>
Status: open | Ask: <the exact answer needed, e.g. "approve" or "pick A or B">
Due: <date> | If no answer: <default action, or "none: blocks X">

Recommendation: <option> (confidence: high / medium / low)
Why: <three lines at most, plain words>

Big-picture effect (write "no change" where none):
- Claims: …
- Papers and dates: …
- Real-world use: …
- Cost (compute, money, your time): …
- Risk: …

Options:
- A: <one line>. Gain: … Lose: …
- B: <one line>. Gain: … Lose: …

Can it be undone? <yes, and the cost of undoing / no>
What we are unsure about: <one line>
Second opinion (skeptic): <agree or disagree, plus one line>
Details for agents: <link to the run report or plan section>
```

### Gate card (a planned checkpoint, or a pivot trigger)

```
# G<n>: <gate name>, <date>
Status: open | Ask: "pass", "fail", or "pass with conditions"

| Check (plain words) | Measured | Target | Result | What it means |
| --- | --- | --- | --- | --- |
| …                   | …        | …      | PASS / FAIL / NOT MEASURED | one line |

Recommendation: <pass / fail / conditions>, and the consequence of each answer.
Is any check unfair or broken? <yes or no; if yes, say why in one line>
Second opinion (skeptic): <one line>
Details for agents: <link>
```

"NOT MEASURED" is never counted as a pass.

### Launch card (a large run the overseer starts)

```
# L<n>: <what the run does, in plain words>
Status: ready | Why now: <one line, linked to a PLAN item>

Command (paste as is, from the workspace root):
    <one block>

Expected cost: <GPU-hours> · <wall-clock hours> · <data read> · <storage written>
You will know it worked when: <one line, e.g. "REPORT.md says 5,000/5,000 done">
Stop it if: <one line>   Safe to stop and resume: <yes / no>
After it finishes: tell the steward. No other action is needed from you.
```

The overseer needs no technical judgment to run a launch card. If running it
safely needs judgment, it is a decision card first.

## 4. The second opinion (the skeptic)

Every class-3 card, and every result that would change a claim, gets a
skeptic's opinion before it reaches the overseer.

- The skeptic is a fresh agent session that did not do the work (`ROLES.md`).
- It reads the card and the linked evidence, and writes one or two lines:
  agree or disagree, and the single most important thing the card misses.
- Disagreement stays on the card and is never smoothed over.

For science questions, the overseer is not expected to judge the math. They
choose on the compass, with both opinions in view.

## 5. The old `[HUMAN]` items

Each `[HUMAN]` item in `handoff/CHECKLIST.md` becomes one of these:

- **Launches** (large runs): a launch card.
- **Gate and pivot calls:** a gate card, with the steward's recommendation and the skeptic's opinion.
- **Money, publishing, and outside contact:** a decision card.

"Track agent decides and asks for approval later" is not one of the options.

## 6. Decision log

Newest first. One line per decision: date, ID, decision, who decided.

| Date       | ID  | Decision                                                                                                                                                                                                        | By                                           |
| ---------- | --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 2026-11-08 | class 2 | Holistic review aligns tracks on Nov 14 shared evaluation summary; C interleaves K2/K7, B owns detector/noise rules and reviews A's protocol, completed work reopens only for a decision-changing gap. Existing targets/dates unchanged; fresh skeptic agrees. | steward, under overseer request |
| 2026-11-08 | D6 | Test-plan/claim timing conflict screened; recommend separating open questions from approved conclusions, pending overseer decision. Current K12 restriction remains. | steward and fresh skeptic |
| 2026-11-08 | D5 | Approved factual corrections to research comparison; no generic largest-collection or first-pool-curve claim. Exact construction-record advantage remains to be checked; claim status and dates unchanged. | overseer |
| 2026-11-08 | communication | Brief updates must give enough surrounding context, use plain language and explain necessary unfamiliar terms while keeping detail low. | overseer |
| 2026-11-08 | class 2 | A/B prepare one proposed K8 evaluation freeze; B prioritizes K5 while closing bounded K4/K14 gaps. Existing class-3 truth/split/threshold rules stay in force. | steward |
| 2026-11-08 | D5 | Literature correction screened against primary source and fresh skeptic; open for overseer, unsupported novelty wording suspended meanwhile. No claim ladder change. | steward (screening only) |
| 2026-11-07 | D1 continuation | Owner approved the prepared remaining K0 payload; Track A records completed remote verification. Settled continuation draft consolidated into D1 after the holistic review. | overseer (approval recorded by track A) |
| 2026-11-07 | class 2 | A may do K6 then K10 while external K0 setup is pending; K0 resumes first when setup is ready. No claim, date or scale change. | steward |
| 2026-11-07 | —   | Verify claims before solidifying them (`CLAIMS.md`). The steward's way of handling questions is codified in `STEWARD.md`. The overseer checks in with the assistant rarely; the steward coordinates day to day. | overseer                                     |
| 2026-11-07 | D4  | Disguise claim dropped; the disguise check runs (K14)                                                                                                                                                           | overseer                                     |
| 2026-11-07 | D3  | Track C moves to baselines and truth sets. Heritability parked. No workshop papers.                                                                                                                             | overseer                                     |
| 2026-11-07 | D2  | Corrected direction adopted as the working direction                                                                                                                                                            | overseer                                     |
| 2026-11-07 | D1  | Public Hugging Face storage, gated until the P1 submission (the gating is the steward's default; the overseer may say "fully open")                                                                             | overseer (gating: assistant's default)       |
| 2026-11-07 | —   | Cards D1–D4 written and screened. The skeptic agreed with a caveat on each, and the caveats were applied.                                                                                                       | assistant and skeptic; awaiting the overseer |
| 2026-11-07 | —   | Steering folder created at the overseer's request                                                                                                                                                               | overseer                                     |
