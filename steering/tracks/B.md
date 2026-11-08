# Track B: lab methods and P1 evaluation
Brief of 2026-11-08 read; K5 now first. No claim advanced.

**Now:** K5 quantization test recorded; nothing running.
Class 1: reused quantizer with exact stored-value checks; rules fixed before run.

**Results since the last steward run:**
- K5 quantization: 12/12 grid-compatible; false flags 0/30 controls.
- Known parent replay bit-exact 12/12; complete tensor coverage and pins verified.
  [Report](../../../runs/track-b-k5-quantization-20261108/REPORT.md).
  Necessary mark and known-source replay only; retrieval/order not established.
  Fresh review checked arithmetic/coverage; Hub transfer still needs K7.
- K3 linear/SLERP coefficient tests pass 18 cases; no broader parent claim.
  [Report](../../../runs/track-b-k3-slerp-trig-20261107-v2/REPORT.md).
- K4 Smol360 repeats exact; 1001/1001 real vs 0/1001 exact-tree violations.
  Repeatability is not finite-sketch uncertainty or ancestry evidence.
- K14 behavior failures unchanged; only compatible-source comparisons and
  tensor sketch drift measured, not full-network robustness.

**Next:** K5 parent-out replay and remaining exact marks/edge typing.
Parent-out proposal in track-b docs/steering/K5-parent-out-proposal.md; not run.
Bounded K4/K14 gaps follow without displacing K5; K8 freeze review with A.
K12 final freeze awaits Nov 25 decisions; K9 stays exploratory.

**Blocked or needs:** no external blocker. General C3 calibration remains open.
All source edits uncommitted in track-b worktree; run artifacts external.
