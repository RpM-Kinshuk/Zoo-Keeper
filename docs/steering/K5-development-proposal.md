# K5 development proposal: informative edit/source tests
Internal draft, 2026-11-08. Preparation authorized; no final rule adopted.
Question: can known edit structure distinguish a source from its relatives,
rather than merely find family support? Nov14/Nov21/later dates unchanged.

## Existing evidence and decisions to stop

- Untouched-tensor support survives parent removal in all24 adapter cases.
  Keep it as a cheap support filter, never a unique-source rule.
- A high unchanged-entry-share cutoff cannot flag all12 TIES/DARE cases with
  zero control flags, even separately within each of the three measured anchors.
  Stop calibrating that scalar rule. No cutoff search or richer-pattern project.
- Eleven of12 earlier Heretic writer samples were inactive by construction.
  The bounded active-layer correction uses fixed nonzero construction layers
  and unchanged gates. It repairs coverage, not scientific decision thresholds.
- Actual classic outputs record one shared direction. All four Heretic recipes
  instead specify FULL row normalization and configured LoRA rank3. The installed
  pinned-version implementation uses a rank3 approximation in this branch.
  A universal per-matrix rank1 expectation is not justified for those recipes.
  See the separate scientific-expectation draft; stored rank is not silently3.

## Fixed development cohort, controls and exposure

Use the existing four anchors, eight projection constructions and40 same-anchor
SFT/DPO/LoRA controls. All are exposed development data; preserve existing roots,
duplicates and inherited splits. Do not call a new resampling split unseen.
No fresh holdout is necessary for descriptive work; confirmation is a later issue.
Positive sampling for the immediate gate audit is explicitly construction-aware:
classic retains its existing three writers; Heretic takes the first three numeric
active down_proj layers. Controls use the identical three indices within anchor.
This selection cannot be used as a blinded detector's inference shortcut.

For a later source-discrimination prototype, FIX candidate pools before scores:
same-anchor identities in the existing64-model joined cohort, remove query and
observed duplicate group, then remove ALL true-source groups in the paired branch.
Keep siblings/grandparents/other edits. Use the existing216-key group map, with
exact pinned identities and no metadata substitution. Report the finite candidate
universe and all unknowns. No extra collection, training or weight fingerprints.

Negatives must include nonremoved control-vs-anchor pairs AND nonparent relatives
for each positive query. Positives alone cannot calibrate source specificity.
Stop at these fixed pools; do not add negatives until a threshold looks good.

## Proposed statistic and noise allowance — NOT an adopted rule

Let D be a matched observed writer delta in its declared native/storage view.
Measure relative update and the best rank-r Frobenius residual R_r, using the
existing SVD path, never spectra of separate models as delta singular values.
Hypotheses: classic r=1 with a common left span across at least three layers;
FULL-normalized Heretic r<=configured rank3, per writer. This proposed expectation
needs scientific review; it is not a change to the current rank/confidence gate.
An adapter's configured rank is development truth, not supplied to an unknown
query as privileged inference evidence. Module patterns remain part of the test.

Candidate storage allowance: for each stored value z, form its exact round-to-
nearest preimage interval in the declared format. Let b(z) be its maximum distance
from the interval endpoints (half the larger adjacent spacing, with subnormals,
zero and bin boundaries handled). The triangle inequality gives
    B_storage = ||b(child)||_F + ||b(parent)||_F
ONLY for the stated one-export-at-each-endpoint latent-value model. If parent
conversion was used, model that explicit conversion; do not infer it from a name.
An operator arithmetic term A must separately bound the actual FP32 projection /
low-rank merge arithmetic. The total candidate allowance is B=B_storage+A.
Do not silently assume A=0, derive it from the observed residual, or use a loose
epsilon multiple that makes every delta acceptable. Unknown export/operator
history means this bound is unsupported and the test remains unresolved.

Before adopting even a development decision, validate the allowance against
known operator controls/oracle FP32-to-storage round trips and CPU numerical
repeat/reconstruction error. Existing recorded constructor residuals are evidence
to validate the model, not a per-query acceptance tolerance. Candidate interval
math/FP32 arithmetic justification must pass an independent theory/code check.
No historical source-hash proof is inferred from the currently installed package.

Nonvacuity and interpretation: retain the3e-3 update gate; require a validated
finite B < ||D||_F and resolved numerical boundary before considering R_r<=B.
If B>=||D||, zero rank would satisfy it: report UNINFORMATIVE, not compatible.
A passed inequality is only low-rank consistency under stated assumptions; it
does not itself identify an edit, direction, immediate source or provenance.
Use the same noise model/selection for positive, nonremoved and relative controls.

## Nov9 conditional plain-projection validation

The separate plain FP32 rank-one outer/subtraction derivation now supplies a
conditional arithmetic term A, with BF16 rounding cells and explicit input
conversion H: see K5-classic-noise-validation.md. Independent proof/numerical
validation and the fixed sampled classic/control panel completed in
`runs/track-b-classic-noise-20261109-v4/REPORT.md`.
Per predeclared input model,12/12 sampled classic blocks have informative
rank-one witnesses within allowance, versus0/18 eligible control blocks;
18/36 controls remain unresolved. These are exposed, dependent block results,
not source decisions, final rule adoption or a verified historical replay.
The earlier rank/confidence gate was not changed. Heretic FULL/rank3 arithmetic
remains outside the derived model and needs its own justification.

Writer-only source specificity is insufficient: saved native full-writer hashes
give six nonparent equivalents in each of three classic families. Under the
separately declared BF16 input model, all four have equivalents. Do not run a
writer-only source-threshold sweep: identical inputs cannot be separated.
A next source test must predeclare joint unchanged-nonwriter and informative
writer evidence against these actual relatives. No full-layer expansion or
additional collection is justified by this conditional pilot alone.

## Evaluation and stopping criteria

1. First finish the already-declared active-layer audit with the ORIGINAL gate.
   Report resolved/eligible, rank-one/control flags and unavailable blocks.
2. Validate the mathematical/implementation noise model before a new decision
   rule. If operator arithmetic cannot be bounded informatively under the actual
   construction/export, stop as unresolved; no more/full reads to hide that gap.
3. Only after review, a development prototype may report low-rank-consistency
   sets, false-source answers, useful answers and abstentions in the FIXED pools.
   Require unique informative support to name a source; otherwise retain the set
   or abstain. Apply the existing C3 empirical target (<=.05 false per answer,
   >=50% source-present answer coverage) without calling this fresh validation.
   If the target fails, report FAIL and stop: no alternate cutoffs/rank sweeps.
4. Fuller matrices/layers are justified only if the validated rule is informative
   but sampled coverage blocks a specified result. State expected distinguishing
   outcome, fixed eligible tensors and bytes/time first. Owner launches anything
   >100 weight identities or >2h. No such launch is requested now.

Before FINAL use, route actual changes to noise/decision rules through scientific
review, preserve exposure and separately establish confirmation feasibility.
C's declared Hub cases use the SAME validated rules; unknown construction inputs,
insufficient writers or uninformative bounds stay unresolved. No Hub truth upgrade.
Quantization/replay and known-source merge-coefficient successes remain closed.
