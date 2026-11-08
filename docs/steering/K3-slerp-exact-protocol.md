# K3 SLERP endpoint audit — rules before execution

Read only the nine existing SLERP children and their six unique LoRA parents
(15 models, CPU, expected under ten minutes). Compare each stored child tensor
with its published configured linear interpolation, using the published per-layer
t schedule for layerwise recipes. This tests the actual stored linear-fallback
relation, without assuming that the currently installed mergekit source is the
historical implementation. Hash inputs; preserve originals.

Per tensor PASS: max absolute discrepancy <= 8 float32 epsilon times
max(1, max absolute expected value). This is a conservative float32 arithmetic
roundoff allowance; no bf16 equivalence claim. Per case PASS iff every tensor
passes. Also measure parent tensor cosine, excluding zero norms, and report how
many fall at or below mergekit's current default fallback boundary 0.9995.
Those angles are descriptive; historical implementation provenance is separate.
If all child tensors pass, the prior delta-fit targets [1-t,t] are empirically
validated to float32 arithmetic noise. Otherwise retain SLERP as unresolved;
do not redefine targets, thresholds, or truth. Coefficient error stays <=0.05
with no more than four linear/SLERP misses, as CLAIMS.md specifies.
For per-layer schedule, inspect the existing recipe execution helper before
running; never infer the schedule from child values.
