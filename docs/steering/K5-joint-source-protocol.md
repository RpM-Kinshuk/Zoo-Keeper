# Joint classic-source test — DEVELOPMENT ONLY, predeclared Nov9

Question: can unchanged nonwriters plus informative shared-writer evidence
distinguish the true source from the specific relatives that defeated either
mark alone? No new collection, final rule, truth or split is authorized here.

Four classic queries; candidate universe is the existing64 joined identities
plus12 closed quantization identities. Use only same-anchor candidates. Remove
query/observed-duplicate groups; in source-out remove the exact true-source
group too, retain all other relatives. Keep all remaining identities and report
group counts; observed-block duplicates are not full-weight identity. Do not
rerun passed quantizer/replay or coefficient tests. Quantized copies are merely
source candidates here. No model chosen by this test's outcomes.

## Fixed conditional rule

Each input hypothesis remains separate: native stored source converted to FP32,
or converted to BF16 then FP32. The declared classic operator changes only its
recorded writer weights and finally exports floating tensors to BF16. Candidate
nonwriter floats must therefore match the query exactly AFTER that predeclared
BF16 export, not necessarily in their original dtype. Never call cast equality
native byte identity. Reject on one proven nonwriter counterexample.

Use cached full-tensor BF16 hashes wherever available, excluding writer matrices
and q/v weights absent from that cache. Use cached row-block counterexamples
only when query/candidate manifest dtypes both BF16, geometry/shape/coordinates
align and the block hash convention matches. A mismatch suffices to reject;
a matching row is NOT full-tensor equality. Stream remaining nonwriter tensors
with existing guarded decoder; stop at first mismatch. Ignore nonfloating
buffers explicitly: this test covers floating weights, not a full-file replay.
Supported source float formats are BF16/F32 (F16 also expands exactly through
FP32); other formats are unresolved BEFORE caches are used. Streamed comparison
always follows FP32 then BF16. No F64 source exists in the fixed cohort; its
direct-to-BF16 rounding is not silently substituted for the declared path.

Tied-head handling is explicit and conditional: require config tie_word_embeddings
true, an embedding tensor, and equal shape/BF16 bytes when both embedding/head
are stored. Then omit the redundant lm_head alias from the comparison. Conflicts
or unsupported alias layouts remain unresolved, not silently repaired. Configs,
cache artifacts and freshly read weight files are hash-bound. Cache-only
exclusions inherit their original verified weight pins; do not claim rehashing.

Writer panel: union of old numeric layers0/1/10 and the first three active Heretic
down_proj layers from the saved construction audit, fixed per anchor before this
test. This targets the known hard relative instead of inspecting SVD outcomes.
Each has the same row256 bounds. This is exposed construction-aware development,
not a blind inference panel or a final full-layer detector.

Reuse the reviewed classic allowance B_i=H_i+Q_i+A_i. Stack row-aligned observed
writer deltas horizontally. Rank1 of the stack tests a shared left span; the
aggregate error allowance is sqrt(sum(B_i^2)), without independence assumptions.
For support, one computed rank1 witness must fit EACH block's allowance, every
block must be informative and above the existing3e-3 update floor. Apply numeric
guards. A failure to find such a witness is UNRESOLVED, not exclusion.

For exclusion, use a separately validated lower certificate: for arbitrary
two-column test matrices U,V, J=U^T D V. Rank1 stays rank1 under this map, hence
distance_to_rank1(D) >= sigma2(J)/(||U||2 ||V||2).
sigma2(J)>=abs(det J)/||J||F. Guard determinant/projection errors and upper-bound
operator norms using guarded Gram row sums. Test vectors may come from SVD;
no accurate/optimal SVD assumption is required. A guarded lower bound exceeding
the aggregate B proves inconsistency only under the declared operator model.
Validate this certificate on fixed synthetic controls and obtain independent
proof review BEFORE checkpoint reads. Any counterexample stops the test.

Candidate states: EXCLUDED (nonwriter counterexample or certified writer lower
bound), SUPPORTED (complete nonwriters and informative shared witness), otherwise
UNRESOLVED. Name a source only with exactly one supported identity AND zero
unresolved candidates. Multiple supports, missing data or unresolved competitors
abstain. Source-out reuses the same comparisons, not new fitting or reads.

## Outcomes and stopping

Report eligible/tested query and candidate counts, unique supported sources,
wrong answers, abstentions, unresolved cases, rejection reasons, bytes/time and
per-input-model results. Existing C3 target: <=.05 false per answered query and
>=50% present-source coverage. Empty answered denominator is undefined, not a
pass. Only four dependent exposed roots; no statistical guarantee or claim lift.
This is a conditional development test, not approved final source decisions.

Use CPU because no admitted GPU is available. Cache filters first; at most76
existing weight identities, expected<15 minutes, external900s deadline. Record
actual identities read; no training/new fingerprints/raw-block persistence.
No outcome-driven change to panel, aliases, noise, cutoffs or pool. If target
fails or execution is incomplete, report it and stop; no threshold sweep,
extra layers or hidden candidate removal. Further rule changes need review.
