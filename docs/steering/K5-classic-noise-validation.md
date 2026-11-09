# Plain rank-one noise model: development validation, Nov9
Status: proposed conditional model; not a final detector/noise-rule change.

Necessary question: can the plain FP32 projection have a mathematically justified,
informative storage/arithmetic allowance, rather than the loose bound that accepts
an entire delta? Validate this before additional detector or full-layer work.
Heretic FULL/rank3 arithmetic is outside this model; do not silently apply it.

## Conditional arithmetic derivation

Let P and r be finite FP32 operator inputs. The computed dot product v=fl(r^T P)
is just a finite FP32 vector: its error changes v but does not change the fact
that the exact outer product M=r v^T has rank<=1. No dot-product error allowance
is needed for this rank-only hypothesis. This does NOT validate historical v.

For the unfused operations T=fl32(M), X=fl32(P-T), round-to-nearest with gradual
underflow gives |fl(z)-z|<=u|z|+eta, u=2^-24, eta=2^-150. No overflow allowed.
For N entries define alpha=u/(1-u). Triangle inequalities and
||P-T|| <= (||X||+eta*sqrt(N))/(1-u) give
    ||X-(P-M)||_F <= A
    A = alpha*||P||_F + alpha*(2+alpha)*||X||_F
        + (2+3*alpha+alpha^2)*eta*sqrt(N).
The same formula applies to a row slice with that slice's P, M, X and N.
No empirical residual is used to choose A. Unsupported arithmetic/overflow/FTZ
history means unsupported, not an enlarged tolerance or affirmative decision.

If C is round-to-nearest BF16 export of FP32 X, let b(C) be half the larger
finite adjacent BF16 spacing elementwise. Then Q=||b(C)||_F bounds ||C-X||_F.
Use ||X||<=||C||+Q in A. Max-finite endpoints adjacent to infinity are unsupported.
Zero, subnormal values and exponent boundaries must be explicitly validated.

For observed source W, separately declare operator input P=W.float() or
P=W.to(BF16).float(). The explicit input conversion contributes H=||P-W||_F.
Thus B=H+Q+A bounds the observed delta's distance from the rank-one oracle -M.
This is conditional on the declared input/export/operator model. Evaluate BOTH
predeclared input conversions, without selecting one to claim historical fidelity.
A stored child's passing residual is only consistency under that model, not
proof of its construction, edit, source or direction.

## Stage1: numerical validation before checkpoint matrices

Use CPU, one numerical thread, esd_ind. Validate all finite BF16 rounding cells
with finite neighbors using exact BF16 values plus FP32 midpoint/adjacent probes.
PASS iff every finite probe's error is within Q's elementwise bound; overflow
endpoints are explicitly unsupported. Test FP32 rank-one product/subtraction on
fixed-seed normal, cancellation, subnormal and exponent-boundary arrays. Compare
actual error to the derived A with explicit floating-evaluation guard; no fitted
coefficient, cutoff or seed search. Any supported counterexample FAILS the model
and STOPS model-weight work. The proof also gets one independent fresh review.

## Stage2: only after Stage1/proof pass

Use existing four classic constructions and their exact native anchors. Keep the
old sampled writer names/rows (first three lexicographic down_proj, layers0/1/10),
which the previous metadata audit showed are active. Select same-anchor SFT-s0,
DPO-s0 and dialogue-LoRA-r8 as fixed controls; no outcome-based replacement.
At most20 existing weight identities, expected<15 minutes, hard stop6600s.
No network/GPU/new fingerprints. Original file hashes and direction sidecars
must match the archived pins. Persist only scalars, never directions/weight blocks.

Full source writer matrices are needed ONLY to compute the known projection's
v=r^T P for an oracle round trip; loading more child layers is unnecessary.
Retain row256 observed child/control measurements. Reuse the existing classic_edit
operator and standard FP64 SVD, not a new model/extractor. Check numerical SVD/
reconstruction error separately. All four roots/cases remain exposed development.

For each fixed matrix/input-conversion model, record the simulated writer's
actual oracle error versus B, B/||delta||, relative update, best-rank1 residual,
and whether allowance is informative (B<||delta||). For stored child/control
rows report the same scalar diagnostics as conditional measurements only.
Below3e-3 update, unsupported bound or numerical boundary uncertainty is unresolved.
Do not count an uninformative inequality as compatibility. No final acceptance
rule, confidence guarantee or C3 parent decision is adopted in this validation.

Stopping: if a supported oracle violates the bound, FAIL and stop. If the bound
is valid but uninformative, report that and stop. If informative, bring the actual
conditional statistic/noise proposal and matched-control outcomes for scientific
review before a final rule. Do not expand to all tensors, Heretic or real Hub
cases without a separate justified operator/coverage need. Existing successes,
truth, splits, final thresholds and dates remain unchanged.

Execution correction after an incomplete v2 attempt: numerical-kernel checks
showed full SVD0.17s at the largest selected block shape, while a64MiB sequential
source-read probe completed0.29s. Neither diagnoses the old stall conclusively.
Use existing guarded safetensors_header parsing/decode helpers for explicit
buffered tensor ranges, with fresh whole-file pins, finite file bounds and
coordinate/value equivalence checks against safe_open. Avoid relying on cold
memory-mapped pages; emit file/writer progress. This changes input materialization
only, not values, matrices or statistics. Freeze a new run; preserve v2 as
incomplete. An actual900s external deadline caps this single instrumented retry.
If it still cannot complete, retain partial outputs and stop; no broader retry.
The final reader review required exact full child/parent shape equality before
decoding a row block, preventing broadcasting. v3 numerical checks preceded
that guard and no matrix stage ran; v4 freezes the corrected complete adapter.
