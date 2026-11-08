# D2: Adopt the corrected research direction and paper plan?

Status: decided | Ask: "approve", or name the part you disagree with
Due: Nov 10 | If no answer: tracks work the PLAN items. They are useful under
either direction. No claim or paper changes until you answer.

Recommendation: approve (confidence: medium-high)

Why:
- The original P1 headline ("typed, calibrated lineage scales") is not
  supported. Naming the parent fails, direction was never tried, and no
  baselines exist.
- What weights reliably show is the family. Some edits also leave exact marks
  that make the parent, and sometimes the order, provable.
- The new direction claims those, tests them properly, and stops the claims
  that the evidence or the real world does not support.

Big-picture effect:
- Claims:
  - New: family at scale (C1); exact-mark edits (C2); calibration against
    relatives (C3); the lab as a benchmark (C4); the measured Hub (C5).
  - Stopped: tree recovery; parent and order for ordinary fine-tunes;
    10^4-scale guarantees; byte efficiency; disguise robustness.
- Papers and dates: P1 is decided Dec 18, ICML (only if the edge tests and at
  least 2 baselines are in) or COLM in late March. The zoo paper (May) is
  sized to 10–20k models instead of 50–100k.
- Real-world use: the main product becomes a `check` command (family, license
  chain, declared-base verdict, near-duplicates, "was refusal removed?"), with
  GGUF support in zoo v1.
- Cost: no new compute beyond what was already planned (the 5k pool and the
  7B-plus truth set were in the old plan). It saves time on work that would
  not have paid off. Track B's list is long, so its order and cut rule are in
  `ROLES.md`.
- Risk: ICML may slip to COLM. That is honest, and COLM is a good venue for
  this work.

Options:
- A (recommended): adopt `steering/DIRECTION.md` as written.
- B: keep the original claims.
  - Lose: high risk of an unsupported paper, and wasted weeks.
- C: target COLM now and skip the ICML attempt.
  - Gain: calmer schedule.
  - Lose: the chance at ICML if the edge tests work early.

Can it be undone? Yes, until Dec 18. The paper framing is decided then.
What we are unsure about: whether the exact-mark tests work on our lab. That
is known by Nov 21 (first results) and Dec 12 (the method check).
Second opinion (skeptic): Agree with a caveat — the new headline rests on "exact marks" that will likely show up cleanly on lab models we built with our own scripts, but may be blurred on real Hub models (for example, refusal-removed models that were fine-tuned afterwards, or compressed files that a paper we cite says cannot be rebuilt exactly). Require these tests to pass on real Hub examples, not only the lab, before they become the paper's main claim.
Details for agents: `steering/DIRECTION.md`; `steering/PLAN.md`

**Decision:** Decided 2026-11-07 by the overseer: **approved** as the working direction. The overseer added: verify the claims before solidifying them (`CLAIMS.md`; solidify gate Nov 25).
