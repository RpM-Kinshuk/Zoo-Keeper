# D4: Drop "robust to disguised models" from P1, and count the lab as complete?

Status: decided | Ask: "approve", or "keep building the disguise set"
Due: Nov 14 | If no answer: track B does only the check below, which takes
hours, and nothing more on disguises.

Recommendation: approve, plus a check that takes hours (PLAN K14): run the
fingerprint on the 8 existing disguises and on one exact neuron reordering.
If the fingerprint survives, we gain a robustness result for free. If not, we
report the limit plainly. Confidence: medium-high.

Why:
- All 8 attempts to disguise a model while keeping its behavior failed the
  behavior check. The planned blind disguise set depends on them.
- Two other tests turned out to be broken by over-strict tolerances, so first
  check whether these 8 failed only on tolerances too strict for normal
  floating-point rounding.
- The 8 disguised models already exist, and differ from their originals
  only by rounding-level amounts. Testing whether our fingerprint still
  matches them takes hours.
- Also run one exact disguise: reorder the neurons inside one layer, which
  keeps behavior exactly.
- This matters most for compliance, our top use, because that is where
  someone has a reason to hide a model's origin.
- The rest of the lab (166 models) is complete.

Big-picture effect:
- Claims: P1 says it assumes nobody is deliberately hiding a model's origin,
  as most published methods do, and reports the one disguise test as a known
  limit.
- Papers and dates: removes a time sink before Dec 18. Gate G1 ("minimum lab
  complete") passes without the disguise quota.
- Real-world use: the tool states that it does not defend against
  adversaries. That is honest and standard.
- Cost: saves an estimated 1–2 weeks across tracks B and C. The check takes hours.
- Risk: reviewers may raise robustness. We cite the activation-based methods
  that target it (for example TrFP) and list it as future work.

Options:
- A (recommended): drop the disguise claim and the blind set; run the check
  that takes hours; keep the 8 heavy edits as "heavily changed" examples.
- B: keep building behavior-preserving disguises.
  - Gain: a robustness section.
  - Lose: likely 1–2 weeks, with an uncertain outcome.

Can it be undone? Yes. Later work can add it back.
What we are unsure about: whether the fingerprint survives neuron
reordering. Row blocks would shift, so probably not.
Second opinion (skeptic): Agree with a caveat — the eight "failed" disguises already exist and, at full precision, differ from the originals only by rounding-level amounts, so testing the fingerprint on all eight takes hours, not weeks, and gives a real answer now. That answer matters most for compliance checks, our top use, because that is exactly where someone has a reason to hide where a model came from.
Details for agents: `Zoo-Keeper/docs/plans/2026-11-05-public-transform-equivalence-v2.md`; `steering/PLAN.md` K14

**Decision:** Decided 2026-11-07 by the overseer: **approved**, including the disguise check that takes hours (PLAN K14).
