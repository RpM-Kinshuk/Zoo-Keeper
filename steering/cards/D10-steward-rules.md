# D10: Steward rules for plain language and no idle tracks

Status: decided (overseer, Nov 9). These rules are part of `STEWARD.md` from
now on.

1. **Plain-language check.** Before writing the board, a brief, or anything
   the overseer reads, re-read it as a non-specialist would.
   - Every sentence must make sense without the glossary.
   - Replace jargon, for example:
     - "eligibility gap" → "we found no real example to test on";
     - "conditional source consistency" → "works on our lab examples only";
     - "transfer falsification" → "a test on a real model".
   - Each board line says what happened, the number against its target, and
     what it means. The board fits on one page.
2. **No idle tracks.** While plan items remain open, no track sits idle for
   more than one working day. If a track is blocked, the steward assigns the
   next most valuable open item, or a redesign, in the same run.
3. **"Cannot verify" is not a closure.** If a test cannot run as designed,
   the report must include either:
   - a redesign: another feasible way to answer the same question; or
   - a card asking the overseer to drop or reword the claim.
4. **Necessity chooses the next test; it is not a reason to stop.** The
   necessity rule (D1.6) picks the most decisive next step.
   - If a claim depends on a question existing data cannot answer, the
     minimum new data is justified.
   - Collect it within the run limits, or write a launch card.
5. **Documented truth (D8)** replaces any requirement for makers' private
   records.

**Why:** on Nov 9 all three tracks were idle, the scaling test was shelved,
and the board was unreadable. That puts Dec 12 at risk and leaves the
overseer unable to steer.
