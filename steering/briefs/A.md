# Brief: track A (2026-11-08, steward)
## Do next:
1. K6: finish the strict same-config independent-model target; audit training records/configs before adding pins. Report feasibility before collecting weights.
2. K8: prepare pinned candidate inputs, a bounded budget and comparable 70B cost; send B the proposed labeled queries, nested pools and split/threshold protocol for review.
3. Finish L1 only when its inputs, costs and resume checks exist; the overseer launches large runs.
## Decisions that affect you:
- K0 archive accepted as tested from recorded remote verification; preserve access gates and local copies. K10 audit accepted: 144/150 weight-identical associations, six unknown.
## Corrections:
- K6 stays incomplete: nine per config does not meet thirty; pooled pairs do not establish held-out recall or per-query guarantees.
- No threshold, split or truth change without the established decision process; no launch implied by cost extrapolation.
## Answers to your questions.
- Keep K6's existing target; do not silently substitute a stratified target. Report scarcity with evidence if no suitable cohort exists.
- Coordinate the K8 draft with B; bring proposed freezes back to the steward, not the overseer as an unfinished launch.
## Watch out for:
- A weight-hash match alone does not verify tokenizer, template or evaluation environment; heritability stays parked.
- Avoid more archive machinery; current approved checkpoint/evidence/database scope is preserved.
