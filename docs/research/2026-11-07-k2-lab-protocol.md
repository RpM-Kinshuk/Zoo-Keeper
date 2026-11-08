# K2 lab spectral comparison: protocol and implementation plan

Spec: steering brief C (2026-11-07), K2, and published equations at
https://arxiv.org/html/2608.07786v1#S4 and https://arxiv.org/html/2607.03377v1#S3.
Class 1: bounded adapter; native-module adaptation explicitly labeled.

Design: reuse Model-Research `_unit_profiles` for Trace interpolation/normalization;
use raw saved full eigenvalues for exact Frobenius energy and fixed-half Hill.
Group by native module suffix after contiguous transformer layer index.
Do not split fused QKV spectra or invent a missing gate projection.
Alternative (seven canonical roles only) excludes the frozen Pythia controls;
new weight reads/SVDs are unnecessary for this native-module comparison.
These adaptations are not a full reproduction of either paper's evaluation.

Pass/fail written before execution:
- Unit formula checks pass within 1e-12 (float64 synthetic arithmetic noise).
- Hill uses k=floor(n/2), threshold lambda[n-k-1] in zero-based indexing;
  nonpositive, nonfinite or flat tails abstain, never become zero scores.
- Every original frozen pair remains in output in its original partition;
  no split, truth or deployed threshold changes. Missing profiles yield not_run.
- Complete contiguous layers and identical native module sets required per pair;
  Spectral Signatures requires equal depths (no unapproved layer matcher).
- AUROC is descriptive, not a new gate; single-class partitions yield N/A.
  Parent/direction/edge decisions N/A: these similarity baselines do not infer them.
- Byte counts: full companion-file scan bytes (includes hashing) and logical
  eig-array payload separate; no inference of historical weight-read cost.
  New weight bytes=0; original extraction/verification costs unknown, not zero.
- One hash manifest binds universes, model index, used spectra and adapter/helper.

Implementation plan (execute inline, no commits):
1. Add tests for Hill formula/invariance/invalids, native-role alignment,
   identity/reversal scores, constant-profile abstention and frozen pair retention.
2. Observe expected missing-module failure; implement `studies/baselines/lab_spectral.py`.
3. Run task tests, package tests once, fresh core review; fix actual defects.
4. Run both methods on L12 t2-k64 development/test frozen universes and training
   if present. Hash inputs before scoring. Report every number in RESULTS.
5. Keep tracks/C.md below 30 lines and leave all work uncommitted.

Execution: all implementation steps completed; full suite: 444 passed, 2 inherited upstream-loader failures (see result report).
Core review fixes: isolate invalid Hill profiles from Trace; include training-12
when available; retain consumed payload counters on rejected inputs.
Expected-red regressions and ten green task tests observed under esd_ind.
Ruling: streaming hashlib digest used for Python 3.10 compatibility; no install.
