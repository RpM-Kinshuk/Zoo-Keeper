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

## 6. Settled decisions

| ID | Approved | Effective decision | By |
| --- | --- | --- | --- |
| D1 | Nov 7 | Public, manually gated storage for the approved lab/evidence/database scope; preserve restricted-data exclusions. Archive verified; no storage setup remains. | overseer |
| D2 | Nov 7 | Corrected research direction; test what weights reveal, with real-model evidence before practical claims. | overseer |
| D3 | Nov 7 | C owns comparisons/real-model truth; heritability parked; no workshop papers. | overseer |
| D4 | Nov 7 | Drop disguise robustness claim; retain the bounded limitation check. | overseer |
| D5 | Nov 8 | Correct prior-work size/search comparisons; exact construction-record advantage remains to be verified. | overseer |
| D6 | Nov 8 | Separate approved conclusions from prospective open tests. Later conclusions need verification and explicit owner approval; dates unchanged. | overseer |
| D7 | Nov 9 | Informative evidence may support compatibility; a unique immediate source requires distinguishing relatives within the stated pool. Inconclusive stays unresolved. | overseer |

Closed card options/reviews and the full earlier log are preserved at Git
`fbf40e0a5097bf243f1fcb162c0185c1e171e0f4` under `steering/cards/` and
`steering/DECISIONS.md`. Cards now hold only actual open decisions or ready launches.

## Current coordination and preferences

- Nov 9, steward/fresh skeptic: close B/C applicability audits; treat the pilot as mathematical projection/source consistency without refusal-behavior validation. Separate within-family transfer from ancestry error-test prerequisites; no current rule/target/split changed. No indefinite producer-record hunt or unapproved contact.

- Nov 9, steward: completed A/B/C results carried into one curated ledger and shared dependency note. Stop passed pilots; assess applicability/independent validation before expanding.
- Nov 9, overseer cleanup request: remove superseded handoffs/cards and reduce old logs. Preserve raw evidence, protocols and unique data. Old documentation remains recoverable at the exact revision above.
- Scientific necessity: every task/statement needs a necessary concrete outcome; reuse evidence, consider different framings and state the stopping point. No approval-to-prepare bureaucracy (overseer, Nov 8).
- Four idle GPUs project-wide, initial A0/B2/C2 with explicit handoffs. Use only when faster/useful; owner launch rules still apply (overseer, Nov 8).
- Broad collection and capacity are parked, not pending purchases. Only a concrete necessary information gap can revive them; no source/split/threshold changes implied (overseer direction, steward/A/B implementation).
- Plain contextual updates, verified claims before solidification, short briefs and no repeated completed work remain in force. Historical class-2 sequencing choices are preserved in the cited revision.
