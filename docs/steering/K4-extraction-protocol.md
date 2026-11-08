# K4 Smol360 row-block extraction repeat — before execution

Scope: the existing fourteen-node Smol360 E6 tree, T1 k64 only. Read selected
stored weight row blocks anew twice on CPU using the existing two_sided_sketch
helper and the original per-view rows; no spectra or rank inference. Require
plain matrix geometry and exact tensor shape/name agreement; no view imputation.
Use the original shared-view distance helper without relaxing version guards.

Measure distance discrepancies: CPU repeat 1 versus repeat 2 (same-backend
repeat noise), and each versus original extraction (backend/extraction
sensitivity, not stochastic sampling uncertainty). The conservative distance
noise e is the maximum absolute discrepancy across these comparisons.
Absolute quartet tolerance = 4e + 64 float64 eps * max(1,max original distance).
PASS simulated exact tree iff every quartet gap is within this allowance after
symmetric uniform distance noise [-e,e], seed 20261028, zero diagonal and clip
nonnegative. Real diagnostic flag: >20% quartet violations, unchanged historical
threshold. Report gap median/p95/max and violation sizes, not ancestry claims.

Fourteen models, two fresh selected-block reads each; CPU only, expected <10 min,
stop by 110 minutes. Original inputs and every exclusion remain recorded.
This limited one-family, partial-read test does not close whole-lab K4 or measure
finite-sketch uncertainty. No threshold, split, truth or production change.
