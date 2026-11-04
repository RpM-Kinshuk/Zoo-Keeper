# Track A roadmap

Updated 2026-11-04. Current reports: `runs/track-a-r5-readiness-20261104T092928Z/REPORT.md` and the recovery report below.

| Work | State | Next action |
| --- | --- | --- |
| Extractor X1–X10; zoo Z1–Z8 source | Complete and merged | Retain full spectra in artifact storage and database links. |
| R1 sample; R3/R4 execution | Closed with recorded gaps | Keep missing observations and frozen inclusion weights; optional retries stay stopped. |
| L12 formal fingerprints | 129/166 full T2 | Measure ready new inputs; reuse completed artifacts. |
| Required references | Complete: 47/47 | Approved safe pins applied; all 180 independent pairs mapped. |
| B's remaining formal generation | 37 checkpoints ungenerated | B prepares immutable outputs; A consumes new queues. |
| R5 formal calibration | Blocked by 37 ungenerated checkpoints | B supplies outputs; A measures them, preserves held-outs, then calibrates. |
| R6 / G0 | Artifact smoke passed; calibrated build pending | Apply valid calibration, rerun six queries, prepare owner's G0 decision. |
| A0 / residual R2 | CPU-node route / source-layout gaps pending | Lower priority; no optional Hub tail retries. |

## Current checkpoint

Model-Zoo source `5262c90`: exact CUDA QR/SVD recovery, packed JSON shape normalization and bounded operator caches. Full suite **1,280 passed, 1 skipped**; fresh core review passed. Source is merged locally; nothing pushed.

Hub: **284/287 new T2 successes**, **6 prior bundles reused**; one NaN source and two timeout omissions. Formal handoff: **44/44 delivered**, all **87** formal T2 identities mapped. Artifact zoo: **628 models, 129,623 views, 399 full-spectrum links**, all six queries passed. Full calibration/G0 remains incomplete.

## Next handoff

A completed all 38 newly readable safe references in a separate bounded GPU batch, plus 4 reused aliases. Full spectra, exact hashes and all three T1 companions are retained; no workers remain. The largest-model pilot took 25.8s; the whole batch took 350.2s. Exact 42-row index, source-amendment preview and proof: `runs/track-a-r5-readiness-20261104T092928Z/`.

Approved source amendment merged at `7a0d786`: **129/166 formal T2**, **47/47 references**, **180 independent pairs**, **0 formal T0-only**. Original pin provenance and prior held-outs are retained. No calibration fitted. The 37 remaining B checkpoints are 16 laundering, 8 abliteration, 5 decision, 4 SFT and 4 DPO.

Current exact map: `runs/track-a-r5-readiness-20261104T092928Z/formal129-artifact-map.json`. Current preflight: `calibration-status-current.json` beside it. Authoritative additive coverage refresh: `refresh-reference-coverage.py` in that directory; older refreshers do not include the approved references.

Next: consume B's new immutable queues without repeating completed bases, complete R5 calibration with the frozen allocation, then run the calibrated R6 build and prepare G0 evidence. No optional Hub retry is queued.

Recovery evidence: `runs/track-a-r3-r6-20261102.rcsy95ap/recovery-20261104T042716Z/`.
