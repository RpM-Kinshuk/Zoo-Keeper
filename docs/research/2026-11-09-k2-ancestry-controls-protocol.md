# Do unresolved-source methods still discriminate shared ancestry?
Predeclared before new weight reads/extraction/scores,2026-11-09.

Scientific question: the recent source-neighborhood tests contained only
relatives. They cannot show whether modelDNA or centered residual signatures
separate weight ancestry from same-design independent-training controls. Use
existing fixed control cases to distinguish useful relatedness evidence from
merely uninformative scores. No additional Hub collection or training is needed.

Reuse EXACT development and inherited-test universe.parquet in
runs/track-b-l12-evaluation-v2/controls/t2-k64/{development,test}/. Each contains
30 ordered pairs,12 same-root and18 different-seed-root labels. Twelve distinct
local Pythia14m checkpoints across both splits: four root snapshots, their q4/q8
constructed descendants. Preserve every row/order/eligibility/truth/tier and
existing partition. Root independence is INHERITED from the frozen labels and
publisher seed provenance, not newly established construction truth. Only two
independent-root labels per partition; dependent/reversed pairs are not trials.
Existing test was exposed by earlier work: no new unseen/prospective guarantee.

Run ONLY the two methods missing from this fixed-panel comparison: published
modeldna0.1.0 native fast/default seed/default synthetic-bootstrap calibrator,
and existing Training Leaves Traces canonical-MLP signature/gated assignment
adapter. No method refitting or new thresholds. Trace/Spectral prior runs stay
closed; their archived results can be compared on the same frozen cases, not
rerun. Training132-pair single-class panel omitted: it cannot answer this question.

GPTNeoX factors: gpt_neox.layers.L.mlp.dense_4h_to_h.weight times
...dense_h_to_4h.weight, ALL blocks/full factors, exclude biases/attention/head/
embedding explicitly. Same residual equations3–6/B.1/reference gate as the
previous adapter. Zero/nonfinite/incompatible signatures are not_run, never0.
Native modelDNA reads local safetensors only; no unpickling or model code.
Verify each entire local safetensors SHA256 against the exact-key catalog
manifest and base parent pins, bind config/recipe/lineage metadata. No name join.

Operational PASS:12 identities verified; all60 original rows/method accounted
for with finite scores or explicit not_run; no pair/partition/label changed.
Complete scored coverage requires60/60 per method. Scientific metrics are
DESCRIPTIVE: AUROC and positive/negative score ranges per partition. No new C1
pass gate, per-query guarantee or parent/type/direction result. If only one class
or incomplete scores, AUROC is N/A or explicitly incomplete, never silently
filtered to a passing subset. No calibration of residual scores on test nulls.

For modelDNA also retain each native pair verdict and its default probability.
Report native-default positive/abstention-band/no-match counts by frozen truth,
using unchanged Thresholds values; do not equate every SAME_FAMILY_UNRESOLVED
class with an affirmative ancestry decision. No empirical threshold is chosen
from these scores. No confusion of immediate-parent nonanswers with ancestry
misses. Preserve methods' applicability and native feature gaps explicitly.

CPU8threads,12small local models, expected<5minutes, stop110minutes. Logical
weight bytes (full factors/native reader), separate identity verification bytes,
wall time and reused old spectral cost limitations recorded. No GPU reservation,
network model reads, raw blocks/deltas retained or publication. Native local
sample fingerprints remain private run artifacts. One input hash record/result.
modeldna0.1.0 was absent in current esd_ind and restored via published wheel
--no-deps; no existing package/version changed. Stop after the fixed panel,
report whether ancestry and immediate-source questions differ; no next seed,
new training, rank sweep, or test-set tuning to rescue an outcome.
