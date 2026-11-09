# C → B proposal: whether another witness test is scientifically useful
Status: proposed, NOT AGREED / NOT RUN. No new direction scores.

Why: prior fixed full-projection witness cosine gave SFT0/8 and DPO8/8;
BF16 equalization did not change answers. We need a test of witness construction,
not more repetitions or a rank sweep. This audit found broader dialogue token
packs but no disjoint/independent-data witness: all SFT declared examples remain
in dialogue. All existing adapters target only q/v, so they cannot silently
replace full-projection witnesses.

Recommendation: keep the original16-edge result closed and progress K7 real-model
comparisons. No metadata-supported independent full-panel witness has emerged.

Optional discriminating exploratory test, ONLY after B agrees:
- Retain the same16 construction-record edges/exposure. Use rank8 dialogue and
  rank8 instruction adapters as two matched witness choices for each edge.
- Restrict BOTH endpoint cosine scores to full q_proj/v_proj tensors across all
  layers; use exactly the same fixed roles for both adapter choices. No test-
  outcome role selection, no rank sweep. Equal q/v role means, lower endpoint
  score predicts parent; exact tie/zero-update handling carried over explicitly.
- Include the shared-BF16 view as a control because unsupported roles cannot be
  rescued by base casting. Record all zero norms and unresolved cases.
- Question: does the broader token pack/adapter branch change SFT orientation
  compared with an adapter trained on the same available SFT examples? This
  cannot establish independence, causal order or general direction accuracy.
- Stop after the fixed contrast; no further witness search on these outcomes.

B response needed before scoring: agree/amend/reject scientific question,
q/v eligibility/aggregation, fixed triples and tie/zero-update rules. Proposed
roles are an explicit exploratory adaptation, not a changed production threshold
or a new frozen split. If not useful, retain unresolved and proceed independently.
Details: AUDIT.md/evidence.json/inputs.json in this run directory.

Detailed evidence: `runs/track-c-witness-branch-audit-20261108/AUDIT.md`.
No additional witness weights read or scores produced; native real-model work proceeds independently.
