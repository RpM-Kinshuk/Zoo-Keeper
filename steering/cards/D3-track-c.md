# D3: Move track C from heritability to baselines and truth sets?

Status: decided | Ask: "approve", or "keep heritability"
Due: Nov 10 | If no answer: track C starts baselines (this is reversible).

Recommendation: approve. Revisit the heritability workshop paper in January
(default there: skip it). Confidence: high.

Why:
- No published baseline has been run, and without them no version of P1 can
  be reviewed.
- Heritability produced no numbers. Its rules were stricter than the question
  needs, and fixing that (PLAN K11) can wait until January.

Big-picture effect:
- Claims: makes it possible to test every lineage claim against published
  methods.
- Papers and dates: protects the Dec 18 P1 decision. The ICLR 2027 workshop
  paper (around Feb) probably slips or is skipped.
- Real-world use: baselines show users whether our tool beats what already
  exists.
- Cost: no new spending. It is the same track, but its load is heavy, so
  the order and the cut list are set below.
- Risk: we lose the workshop slot. The useful finding (weights are worth only
  about 3–39 benchmark questions) is kept for the selection paper or a short
  note.

Options:
- A (recommended): C does, in this order:
  1. Trace and Spectral Signatures on the lab (Nov 21);
  2. a first verified Hub truth set: the vendor pipelines OLMo-2 and Tülu 3,
     plus Hub models with known parents that were refusal-removed or
     quantized (Dec 1);
  3. modelDNA (Dec 5);
  4. Witness Overlap (Dec 5).

  **If C falls behind, cut in this order:** MoTHer, then merges beyond 20,
  then Witness Overlap on the Hub tier (keep it on the lab). The real
  look-alike models (K6) move to track A. Heritability resumes in January
  only if this is decided again.
- B: C keeps heritability, and B runs baselines on the side.
  - Lose: B is already the critical path for K3–K5 and K9.

Can it be undone? Yes.
What we are unsure about: whether the workshop deadline is in early or late
February.
Second opinion (skeptic): Agree with a caveat — track C is handed three large jobs due within four weeks (published baselines, 30 look-alike models, and a 7B-plus truth set), and the Dec 18 decision needs the baselines run on that truth set too, so all three are urgent at once. The card should say what C does first and what gets cut if it falls behind, instead of saying the move costs nothing extra.
Details for agents: `steering/PLAN.md` (K2, K6, K7, K11)

**Decision:** Decided 2026-11-07 by the overseer: **approved**. Skip the workshop paper; workshop papers are not a target for this project. Heritability is parked with no paper target.
