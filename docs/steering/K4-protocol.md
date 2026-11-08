# K4 numerical repeat diagnostic, 2026-11-07

Written before execution. Scope: eight existing E6 distance matrices; no new
weight acquisition. Repeat the shared-coordinate distance calculation from
fresh fingerprint loads and compare with saved matrices. This measures numerical
repeatability only: it does not estimate extraction, storage, or sketch sampling
noise. Those components remain unmeasured and K4 cannot be closed by this run.

For each family/mode, absolute distance noise e is the maximum absolute repeat
minus original distance. Quartet tolerance is 4e plus 64 float64 eps times
max(1, maximum distance). Report absolute gaps between the largest two four-point
sums, their median/p95/max, and violation fraction above this numerical floor.
No frozen production tolerance or pivot threshold changes.

Positive control: construct the exact additive tree from the same truth edges,
using original parent-child distances as edge lengths. Add symmetric independent
uniform noise in [-e,e], clip nonnegative, zero diagonal, seed 20261028.
PASS control iff no quartet gap exceeds 4e plus the same roundoff allowance;
otherwise FAIL diagnostic validity. Real data cross the inherited diagnostic
flag iff >20% quartets exceed the floor, but this is not an ancestry conclusion.
Extraction-noise comparison and any resulting claim decision remain pending.
