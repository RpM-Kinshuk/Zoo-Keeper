# Track A roadmap

Updated 2026-11-04. Current reports: `runs/track-a-r5-readiness-20261104T092928Z/REPORT.md` and the recovery report below.

| Work | State | Next action |
| --- | --- | --- |
| Extractor X1–X10; zoo Z1–Z8 source | Complete and merged | Retain full spectra in artifact storage and database links. |
| R1 sample; R3/R4 execution | Closed with recorded gaps | Keep missing observations and frozen inclusion weights; optional retries stay stopped. |
| L12 formal fingerprints | 87/166 full T2 | Measure ready new inputs; reuse completed artifacts. |
| Required independent references | 42 original pins are pickle-only | Safe conversion candidates audited; bounded GPU pilot/batch underway at actual new pins. Four existing scoped aliases are reusable. |
| B's remaining formal generation | 37 checkpoints ungenerated | B prepares immutable outputs; A consumes new queues. |
| R5 formal calibration | Blocked by cohort/source readiness | Record explicit source amendment, preserve held-outs, finish required mapping, then calibrate. |
| R6 / G0 | Artifact smoke passed; calibrated build pending | Apply valid calibration, rerun six queries, prepare owner's G0 decision. |
| A0 / residual R2 | CPU-node route / source-layout gaps pending | Lower priority; no optional Hub tail retries. |

## Current checkpoint

Model-Zoo source `5262c90`: exact CUDA QR/SVD recovery, packed JSON shape normalization and bounded operator caches. Full suite **1,280 passed, 1 skipped**; fresh core review passed. Source is merged locally; nothing pushed.

Hub: **284/287 new T2 successes**, **6 prior bundles reused**; one NaN source and two timeout omissions. Formal handoff: **44/44 delivered**, all **87** formal T2 identities mapped. Artifact zoo: **628 models, 129,623 views, 399 full-spectrum links**, all six queries passed. Full calibration/G0 remains incomplete.

## Next handoff

A is measuring the 38 newly readable safe-reference candidates separately, with full spectra, exact source hashes, at most four eligible GPUs and a two-hour whole-action bound. The largest-model pilot must justify expansion. Current source proposal and runtime records are under `runs/track-a-r5-readiness-20261104T092928Z/`.

Formal source changes need an explicit recorded amendment before credit: original and actual keys stay distinct, original registry/provenance is preserved, and prior 14M seed 3/4 held-outs remain held-out. B owns the 37 ungenerated checkpoints. No partial null or gate claim is produced.

Recovery evidence: `runs/track-a-r3-r6-20261102.rcsy95ap/recovery-20261104T042716Z/`.
