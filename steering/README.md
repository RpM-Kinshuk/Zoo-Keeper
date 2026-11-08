# Steering

This folder is the control layer for the Model Zoo work, started 2026-11-07.
It sets the corrected research direction, the forward plan, how decisions
reach the overseer, and how the three tracks stay reconciled.

**Where this folder and `handoff/` disagree, this folder wins.** The
`handoff/` folder stays the reference for technical rules: tokens, untrusted
inputs, engineering standards, and the frozen file formats.

## Who reads what

| Reader | Reads | Time |
| --- | --- | --- |
| **Overseer** (the owner) | `BOARD.md` only: one page covering what happened, what is happening, what comes next, and what waits for you. Open a card in `cards/` only when you decide it. | 3 min |
| **Steward** (new agent role) | `STEWARD.md` first (its manual, context and glossary), then everything here | as needed |
| **Track leads** (A, B, C) | Their brief in `briefs/` at the start of every session; this README; `DIRECTION.md` section 1; `CLAIMS.md`; their items in `PLAN.md`; `DECISIONS.md`; their section of `ROLES.md`; and `DATA.md` when they store data. For technical rules, `handoff/DIRECTIVES.md`. | 20 min |

## The compass

The overseer holds the big picture. Agents hold the technical knowledge.
Every decision and recommendation is weighed against these six points, in
this order, and every card says how it affects each one.

1. **Real-world use.** Would a practitioner actually use the output: for a
   license or compliance check, a safety check, de-duplication, "what did
   this fine-tune change?", or choosing a model?
2. **Honest claims.** Claim only what the evidence shows. Report negative
   results as plainly as positive ones. A result that fails its target is
   reported as failing.
3. **Publishable results on the dates.** P1 is decided on Dec 18: ICML around
   Jan 22–28, or COLM around late March. The zoo paper is due around early May.
   The checkpoints and dates are in `PLAN.md`.
4. **Simple and fast.** Use the simplest thing that answers the question. No
   infrastructure, bookkeeping, or completeness work beyond what an answer
   needs.
5. **Cost.** Compute, storage, money, and above all the overseer's attention.
6. **Safety and legal.** Tokens, licenses, untrusted code, and data that must
   never be published.

**Warning signs** that a decision has lost the big picture:
- making a test pass by redefining it;
- counting "credit" instead of answering a question;
- adding machinery before a result needs it;
- deciding something irreversible because nobody objected;
- treating "the code exists" as "the result is known".

## What changes from `handoff/` (2026-11-07)

1. **Priorities** now come from `PLAN.md`. The open items in
   `handoff/CHECKLIST.md` are mapped there: continued, changed, or dropped.
2. **`handoff/STATUS.md` and `handoff/CHECKLIST.md` are frozen as history.**
   Tracks no longer write to them. Each track keeps one short note in
   `tracks/`, which it rewrites rather than appends to. Numbers go to
   `RESULTS.md`. The steward keeps `BOARD.md`. Run details stay in the run's
   own folder.
   - Drop roster "credit" accounting, approval records for reversible
     choices, and per-step hash proofs. Keep one input-hash record per
     reported result, and one manifest per release.
3. **Decisions follow `DECISIONS.md`.** Each decision has a class (who
   decides) and a one-screen card. The old `[HUMAN]` items become launch
   cards, gate cards, or decision cards.
4. **Status words mean exactly one thing:**
   - **built:** the code exists and its tests pass;
   - **tested:** it ran on real data and the numbers are recorded;
   - **decided:** a claim's verdict is settled.

   The old "done" mixed these up.
5. **Data goes to public Hugging Face repositories, gated until the P1
   submission** (card D1, `DATA.md`). Agents may upload to the three
   designated repositories. Removing the gate, or publishing anything else, is
   the overseer's call.
6. **The verified-join rule** (`handoff/DIRECTIVES.md` D8.4) applies only when
   scores are joined to our weight measurements. A score-only study may use
   the leaderboard's own commit as the model's identity, and the benchmark's
   fixed question count as its sampling size (`PLAN.md`, K11; heritability itself is parked).
7. **Agents make reversible, in-track choices without asking** (`DECISIONS.md`,
   class 1). Approvals and roster amendments are no longer needed for choices
   that can be undone.

8. **Verify before solidifying.** Every claim climbs the ladder in
   `CLAIMS.md` (hypothesis, checked, piloted, verified, solid), with pass/fail
   rules written before each test. Only the overseer makes a claim solid, at
   the gate on Nov 25.
9. **The steward guides the tracks** through `briefs/`, and handles all
   questions the card way (`STEWARD.md`).

Everything else in `handoff/` still applies: no agent commits, no `.env`
reads, the person launches large runs, and untrusted inputs stay data.

## Files

| File | What it is |
| --- | --- |
| `BOARD.md` | The overseer's one page, kept by the steward |
| `DIRECTION.md` | The research reassessment and the corrected direction |
| `PLAN.md` | Corrective actions, milestones, the plan index, and the mapping of old checklist items |
| `RESULTS.md` | The results ledger: every reportable number, its target, and its source |
| `CLAIMS.md` | The claims and hypotheses, the literature check, theory notes, pass/fail tests, and each claim's status |
| `STEWARD.md` | The steward's manual: its job, how it handles questions, its runs, the context, and a glossary |
| `DECISIONS.md` | Who decides what, the card templates, and the decision log |
| `ROLES.md` | The roles of the overseer, the steward, the track leads, and the skeptic, with their kickoff prompts |
| `DATA.md` | Storage and publication on Hugging Face |
| `cards/` | One file per open decision, gate, or launch |
| `tracks/` | One short note per track, rewritten by its lead |
| `briefs/` | The steward's current directions for each track |
