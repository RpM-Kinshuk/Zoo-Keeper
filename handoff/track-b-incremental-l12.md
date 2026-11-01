# Incremental L12: small real cohort

Owner priority, 2026-11-01: verified real sources, ready L8/L9 derivatives,
evaluation harness and first scoped inference/scoring report. Release staging
is deferred. A retains ownership of whole-lab L12; this bounded incremental
pass does not claim L12 or G0 completion.

B owns generation and the first local T1 pass under
`$ZOO_DATA/runs/track-b-real-cohort-v1/`, avoiding A's existing smoke and future
whole-lab output directories. Do not duplicate that T1 work. The atomic
`ready.json` will list completed source/output rows with their verified pins,
paths, parents and hashes; only those rows enter scoped tables.

The targeted control family is Pythia-14M seeds1–4 with q8/q4 derivatives:
seed1/2 are development roots; seed3/4 are held-out test roots. Roots and all
derivatives stay in one partition. Safetensors availability is a prerequisite;
pickle loading is forbidden. Smol/OLMo transforms are separate descriptive
strata until comparable certified controls exist. The pre-score protocol is
`predeclared-protocol.json` in that run directory; it freezes the illustrative
0.75 development threshold before any cohort scores are observed.

Incremental evidence is written to `evidence/manifests/` and `evidence/t1/`
within this run. B uses the committed Model-Zoo extractor at cda3705, local
verified sources only, k64 and T1 block256, with an explicit parent manifest
for each derivative. Each completion records model identity, input/output
hashes, selected bytes and elapsed time in `evidence/progress.json`.

A can consume the ready scoped `models.csv`, `edges.csv`, manifest directory
and cost/progress records directly through the frozen file contract. A's next
increment should add T2 and the prescribed free T1 blocks128/256/512 at k128
in a separate output directory after a measured runtime projection; first
measure one small source and one derivative. Preserve the pinned input list
and provenance, and coordinate writer ownership here before launching. B will
not label missing T2 or5k-pool measurements as successful gates.

Exact ready counts, invocations and final report paths are appended as the
cohort completes. No blind paths are required or inspected.
