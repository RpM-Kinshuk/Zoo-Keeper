# Track C local integration — 2026-11-09

User authorized local integration of the completed Track C batch. Feature commit
319ec84 was merged into main as 2df386e; nothing pushed. Research data,
fingerprints, manifests and raw samples remain outside git. Track branch/worktree
are retained for continuing C work; unrelated B dirty files are preserved.

Pre-merge core review found one integration defect: seven one-off runners used
a fixed worktree depth to find the workspace, which would fail from main. Added
studies.baselines.paths.workspace_root and switched those runners. Three regression
cases cover main layout, track layout and invalid standalone layout: missing-module
RED then GREEN; entire baseline package34passed. The scientific calculations,
inputs and recorded results were not rerun or changed. Original runner bytes and
SHA256s are archived in runs/track-c-integration-20261109/original-runners/ so old
input records remain auditable after this reproduction-only path correction.
Current shared owner directives were copied before staging, preserving scientific
necessity and newer main wording without duplicate policy changes.

## Verification and known baseline limitation
Full merged suite: **690 passed,2 failed**,3warnings,84.75seconds. It is NOT an
all-green suite. Both failures are pre-existing CPU/Gloo SFT equivalence cases:
- test_global_loss_update_frozen_reference_and_collective_resume[sft-2]
- test_global_loss_update_frozen_reference_and_collective_resume[sft-4]

The same cases failed on unchanged main before this merge (2failed,17passed in
the distributed package). FP32 parameter-equivalence assertions differ slightly;
other ranks can report the resulting closed-peer collective. Single-thread BLAS
and a reference-attention diagnostic did not remove all failures. No assertion,
tolerance, training algorithm, label, split or scientific threshold was changed.
These baseline failures remain open and are not attributed to the C comparison
code. Local-loopback Gloo tests ran outside the socket sandbox. The full merged
run kept optional Hub-kernel discovery offline during installed Heretic imports;
no remote model/kernel downloads were needed. All tests were collected/run.

The environment lacked twelve dependencies already pinned in requirements.txt.
Installed missing versions only with --no-deps: pydantic-settings2.13.0,
python-dotenv1.2.1,typing-inspection0.4.2,py-cpuinfo9.0.0,questionary2.1.1,
tomli-w1.2.0,optuna4.7.0,sqlalchemy2.0.46,alembic1.18.1,colorlog6.10.1,
Mako1.3.10,greenlet3.3.1. No existing package version changed. Heretic package
checks then15passed. Environment restoration is not a model/detector change.

Logs and preservation checks: runs/track-c-integration-20261109/. Before/after
hashes cover13 unrelated dirty B files. Scientific numbers remain in RESULTS
and existing C reports; routine test counts are recorded here rather than
presented as scientific outcome improvements. No remote push or branch deletion.
