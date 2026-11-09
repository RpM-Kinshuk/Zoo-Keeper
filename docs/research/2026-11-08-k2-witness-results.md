# Witness Overlap on constructed ordinary-fine-tune edges — 2026-11-08

Scientific question: does a third same-family checkpoint usefully orient known
ordinary-fine-tune edges where pairwise relatedness cannot? The fixed panel
answers this for opposite-seed SFT witnesses, not for arbitrary witnesses.

All16 frozen construction-record edges measured,20 distinct models,4 anchor
families; cosine adaptation of arxiv2609.31784v2 equations1/5 and Algorithm1.
Operational completeness PASS. Scientific direction remains exploratory: no
accuracy gate or detector threshold was introduced or changed.

| Measurement | Observed | Interpretation |
| --- | --- | --- |
| Correct/answered/eligible |8/16/16|No general directional success on this panel|
| Wrong/ties/abstain/not_run |8/0/0/0|Every SFT edge reversed; no silent exclusions|
| SFT correct |0/8|Sibling-seed witness does not orient these SFT edges|
| DPO correct |8/8|Conditional success with this specified witness construction|
| Each anchor family |2/4 correct|Pattern holds in all four small lab groups|
| Shared-BF16 control |8/16 correct;0/16 answers changed|Unequal storage precision does not explain these reversals by itself|
| Nearest-witness distance control |8/16 correct;16/16 agrees|Same decisions without implying a causal order test|
| Primary full projection coverage |2800/2800 tensors|Every required projection has defined anchor scores|

The primary parent-anchor cosine is high for the SFT sibling witnesses, whereas
the child-anchor score is lower. Their updates are co-aligned in the measured
coordinates. This is consistent with the paper's stated high-alignment failure
condition (section5.4), not a refutation of its differently sampled benchmark.
Witnesses share data/training choices and are not independent roots; these are
previously exposed lab cases, not an unseen test set. No direction precision,
false-parent guarantee, pair-type classifier or parent-retrieval claim follows.

Why the distance check was necessary: for one tensor the lower endpoint cosine
is exactly the endpoint closer to W by triangle geometry. Block aggregation can
differ, but did not change decisions here. Thus a same-family witness alone is
not evidence of a reliable chronology; which training branch it samples matters.
This is an interpretation/control, not a new method or novelty claim.

Primary scores use stored values decoded to float64. All seven q/k/v/o/up/down/
gate roles at every layer are averaged within role, then equally across roles.
Embeddings/head/norms excluded with names/reasons in results.json. Scalar
per-tensor scores/norms, all margins and the BF16 control are retained; no raw
blocks or deltas retained. Zero/tie rules were written before scoring; none
occurred in this panel. No fine-tuning, refitted cutoff or model downloads.

## Execution and integrity
Protocol: worktree docs/research/2026-11-08-k2-witness-protocol.md, saved verbatim
before scoring in protocol.json. Recipes and witness choices match the existing
construction metadata; all weight hashes match B's prior input record. One input
record: inputs.json. Equation-level adaptation: official code link unavailable.
Core review PASS; six algebra tests RED missingmodule then GREEN. Baseline
package23passed. The streamed implementation uses128 rows with float64 sums;
full tensors, not partial row-block geometry, define the scores.

Earlier preparations stopped before scoring: derived-quantizer recipe-schema
assumption corrected to training inputs.local_parent/model_pin; equal-dtype
restriction corrected because it is absent from the published method. OLMo's
float32 base/BF16 descendants prompted the documented shared-BF16 control BEFORE
first scores. Original protocols/logs retained; no failed attempt is evidence.
Successful v3 cost excludes their extra metadata/hash-validation IO.

CPU8threads, eniac.cs.dartmouth.edu; GPU driver unavailable, no reservations.
Successful run elapsed 146.637193s; logical full-weight payload
24,012,914,688bytes; separate verification/input scan
32,653,895,960bytes. Shared-BF16 control requires no additional
weight reads. No claim about network search cost or isolated GPU performance.

## Decision and next dependency
Stop tests on these unchanged triples: they resolved the chosen uncertainty.
Do not add SVD just to repeat a known high-alignment witness failure; revisit it
only for a failure it can discriminate or required final variant coverage.
Next useful input is a genuinely different training branch as witness (C source
selection, B shared evaluation), or explicit abstention when none is supported.
Do not retrain or silently select witnesses on these observed test outcomes.
Real-model multi-family comparisons and remaining edit-method coverage stay open.
No claim, truth, split or scientific threshold changed; all work uncommitted.
