# Which skills move after post-training? A measurement-aware study protocol

Status: internal methods and data-audit draft, 2026-10-31. Empirical estimates
remain unavailable. The analysis distinguishes supported results
from missing evidence; synthetic validation is not an empirical finding.
Intended target: an ICLR 2027 workshop. The person verifies the venue, deadline,
and submission requirements under checklist F5 before submitting.

## Abstract

We study association between model ancestry and per-task benchmark accuracy,
accounting for score sampling noise and ambiguous checkpoint identity. The
protocol combines a family random-effect model, observed parent–offspring
regression, and a network Brownian covariance for documented merges. A separate
audit assesses whether historical prediction errors can be expressed in
equivalent evaluation items. Results are conditional on evidence for benchmark
semantics, checkpoint and deployment identity, sampling variance, and held-out
splits. An unavailable estimate remains a reported coverage gap. The current
release is an auditable analysis pipeline and data audit; it does not establish
causal inheritance or a ranking of skill heritabilities.

## Questions and contribution

The hypothesis is that knowledge and reasoning scores move less after
post-training than instruction-following scores. Different score ranges and
floor-clipped normalizations can manufacture this pattern. We therefore compare
Raw per-task accuracy and scale-free prediction errors on shared support.

Prior work includes PhyloLM, Tamura et al., and Laufer et al.; our source ledger
and intended comparisons are recorded in the [merged plan](../research/2026-10-27-merged-research-plan.md),
sections 3.3, 5.4 and 9. We do not claim novelty for familial similarity, network
Brownian models, or the prior effective sample-size formula. The intended
contribution is a measurement-aware comparison per skill and supported edge
type, with identity, variance and coverage audits alongside every estimate.
The final bibliography must resolve the references in that ledger before
submission; this draft makes no new literature priority claims.

## Data and provenance

The starting inputs are the selected leaderboard snapshots and Model Atlas
subgraph, copied from frozen Model-Research with hashes recorded before copying.
The runner retains source provenance, contradictory records, absent denominators,
zero-byte files and missing revision evidence in explicit audit outputs. It does
not import or execute the historical study package.

A benchmark score's own SHA is recorded identity, not independent verification.
A verified join additionally requires a matching independent pinned model record
and compatible harness, precision and quantization. Repository matching ignores
case; exact checkpoint and deployment distinctions remain. Different precision
rows do not count as independent model replicates. The prespecified primary
deployment is selected explicitly; alternatives remain in sensitivity outputs.

Atlas ancestry is declared evidence. Pinning its endpoints does not verify the
relationship. Self-parent metadata declarations are invalid and quarantined with
record-level reasons. Other cycles reject the network analysis rather than being
silently converted into a tree. Duplicate evidence and all roots of a merge are
joined before splitting. Lack of duplicate evidence is disclosed; distinct
repository IDs do not establish independent weights.

Historical prediction artifacts retain their own protocol and hashes. Modern
normalized predictions cannot be converted into Raw predictions by dividing by
a percentage scale or reversing clipping. Incomplete historical provenance is
reported as exploratory, never as a verified held-out result. No predictor is
refit and no test or prospective outcomes select a new model here.

## Methods

### Familial variance and observed parents

For one task, benchmark, harness, lineage evidence tier and scale, let y be the
score vector and D its documented sampling covariance. The primary model fits

`V = tau2 * (Z @ Z.T) + sigma2 * I + D`

by restricted maximum likelihood. A binomial task with documented item count n
uses `p*(1-p)/n`. Composite, macro-averaged or correlated metrics require sampling
variance evidence; no effective item count is invented. Raw accuracy is primary.
A prespecified logit sensitivity uses the documented half-count correction and
recomputes its sampling variance, including at accuracy endpoints.

Kernels are normalized by mean diagonal before fitting. The reported familial
fraction is `tau2/(tau2+sigma2)` conditional on that covariance model, not a
causal genetic heritability. Component boundaries, convergence failures and
nonidentifiable designs are reported explicitly. Documented parameter counts
permit an intercept and centered log-size covariate; otherwise the matched
comparison uses a labeled intercept-only design.

Observed parent–offspring regression uses compatible measured parent scores,
not imputed family averages. It reports both naive and measurement-corrected
slopes. The corrected population-moment denominator subtracts mean parent score
variance. Nonpositive denominators are unavailable. Paired error covariance is
used only when documented; otherwise the independence assumption is stated.
Merges are not treated as single-parent observations.

### Merge covariance and uncertainty

Network Brownian inheritance propagates each parent's innovation vector with
documented convex coefficients and adds the child's branch innovation. Its
covariance is `A @ A.T`; observed ancestors remain in the matrix. Unit branch
increments are a prespecified sensitivity model, not measured evolutionary
time. Unknown or unsupported coefficients exclude a merge and its affected
descendants from the primary network fit. This does not constrain the P1
engine's separate unconstrained merge regression.

Uncertainty resamples whole source split groups. Repeated parents and related
pairwise differences are not independent replicates. Failed bootstrap fits and
interval coverage are retained. Estimates from insufficient support remain
unavailable. Per-skill error summaries include MAE, RMSE, MAE divided by target
standard deviation, and R-squared; constant targets have explicit undefined
scale-free statistics, and negative R-squared values remain visible.

### Evaluation-item equivalence

For absolute prediction mean-square error s-squared and binary-item variance q,
the estimation equivalence is `k = q/s_squared`. The strict MAE upper bound is
`q/MAE_squared`, since RMSE is at least MAE. The historical assumption
`RMSE = 1.25*MAE` is labeled an approximation and never replaces measured RMSE.

Selection uses errors in pairwise score differences. Under independent item
evaluations the numerator is `p_i*(1-p_i) + p_j*(1-p_j)`; k is items per candidate,
and total items are twice k. Paired-item covariance requires item-response
evidence. Within-family and across-family pairs are reported separately from
the training/holdout setting. Duplicate pairs are removed. Weight and no-weight
comparisons share audited held-out support, including folds and deployment
identity; negative incremental value is retained. Perfect prediction on a finite
sample does not establish infinite certainty.

## Results and coverage

The first frozen audit is `studies/heritability/track-c-first-pass-v1/final/`,
relative to `$ZOO_DATA`. Its protocol SHA256 is
`4f3b6d804608d77e0ed96efa661219f8a70e04782cab7d1360f3695baed7146a`.
Every count below traces to that run's `coverage.json`, `exclusions.csv` and
hashed input snapshots. The run completed with status `PARTIAL`.

The two source snapshots supplied 11,232 task observations before duplicate and
conflict handling. Assembly retained 3,288 unique deployment/task observations
and excluded 2,316 conflicting identities. These counts use different units:
the input count includes overlapping copies; each conflicting identity can
represent multiple source rows. Verification against an independent deployment
record succeeded for 0/3,288 retained observations. Documented sampling variance
was available for none of the retained observations. All original source rows
remain in hashed snapshots.

| v2 task | Retained score observations | Family/network, raw/logit | Observed-parent slopes | Raw prediction errors and item equivalence |
| --- | ---: | --- | --- | --- |
| BBH | 757 | unavailable | unavailable | unavailable |
| GPQA | 504 | unavailable | unavailable | unavailable |
| IFEval | 692 | unavailable | unavailable | unavailable |
| MATH level 5 | 265 | unavailable | unavailable | unavailable |
| MMLU-PRO | 559 | unavailable | unavailable | unavailable |
| MuSR | 511 | unavailable | unavailable | unavailable |

Task counts come from `coverage.json:retained_by_task` and `inputs/scores.csv`.
The 24 covariance/scale rows in `heritability.csv` have no eligible observations;
the 12 rows in `parent_offspring.csv` have insufficient compatible pairs.
`exchange.csv` records missing eligible historical tasks. `skill_errors.csv`
retains explicit unavailable Raw/common-support summaries rather than substituting
normalized errors. Sampling-noise fractions, uncertainty intervals, per-edge-type
effects and within/across-family selection equivalences are consequently
unavailable. These are coverage findings, not estimates of zero effect.

Declared ancestry becomes acyclic after quarantining six metadata self-parent
declarations; the raw declarations and exclusion reasons are retained.
`coverage.json:ancestry_status` is `available`, while documented merge coefficients
remain incomplete. A valid declared graph alone does not verify score identity
or supply measurement variance.

The located September artifact contains 290,496 per-task prediction rows,
including modern and legacy targets. Its original percentage-point targets and
incomplete deployment/split evidence are retained as historical descriptive
material; the audit does not assert that every legacy target is normalized.
Modern target construction uses floor-clipped normalized scores, which cannot
be relabeled Raw. Verified prediction-to-score joins are 0/290,496; the denominator
includes every historical row, not just eligible Raw rows. See
`inputs/historical/audit.json`, its copied originals and the corresponding
exclusions for semantics and provenance details. The selected Raw assembly has
no explicit v1 source, recorded as `v1_status=unavailable_no_explicit_source`;
legacy historical tables remain distinct from a validated v1 input.

The software's synthetic validation exercises actual family/network fits,
observed parents, common-support errors and item equivalence. It establishes
implementation behavior only and contributes no empirical claim to this table.

## Limitations and next evidence

Declared Atlas links, unit branch lengths, benchmark sampling assumptions and
incomplete deployment identity limit interpretation. Matching benchmark commits
does not verify ancestry. Historical unpinned predictions remain exploratory;
normalization gaps cannot be repaired by relabeling targets. Tasks without
documented sampling variance support descriptive summaries only.

Per-edge-type estimates wait for defensible edge labels. T1 distance comparisons,
new predictor fitting, matched-cost item-baseline evaluations and end-to-end
model selection belong to later reviewed work. This first pass does not run new
model evaluations, use paid judges, or inspect the blind laundering subset.

## Reproduction and baseline handoff

Use the [operations guide](../operations/heritability.md) for the frozen runner,
hash verification, resume behavior and tiny dry run. Final tables are generated
under `$ZOO_DATA/studies/heritability`; run outputs and historical predictions
are not committed.

For S5, align the eventual adapters to the reviewed P1 engine's evaluation
protocol: `Baseline.run(query_keys, candidate_keys, *, budget_bytes, seed)` returns
prediction rows with the frozen `LINEAGE_COLUMNS` and a query-cost table
(`query_key, bytes_read, seconds, status, reason`). Candidate ranking may be an internal
adapter detail; it does not replace the engine's scoring contract. Missing
baseline artifacts are `not_run` and coverage gaps, not zero error. Port the
actual Trace/Spectral Signatures math from Model-Research with provenance;
do not import its frozen runner. Charge reference construction and additional
verification reads as well as query bytes. No empirical baseline comparison is
claimed until the upstream artifacts and complete pool are available.
