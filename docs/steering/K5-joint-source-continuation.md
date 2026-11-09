# Joint source continuation — predeclared Nov9, development only

Goal: finish the fixed four-query joint source test after exact input recovery.
The owner explicitly authorized continuation and GPU use on Nov9. This replaces
only the earlier execution stop/budget for this new run; original v2 stays frozen.
Use the same method,76 identities,72 candidate pairs, source groups, panel,
conditional input models, noise/cutoffs and answer rule. No claim/split changes.

Architecture: retain saved comparisons with complete evidence, re-evaluate
only the three saved `no weights` pairs and the36 unsaved pairs. Thus33 pairs
are inherited and39 newly evaluated. Never retry statistical unresolved states.
Bind the old progress, frozen driver and protocol, and check all overlapping
fresh evidence hashes against the old pins. Reused weight measurements inherit
their original pins; any freshly read checkpoint is rehashed. Do not claim
freshly re-reading inherited measurements. No candidate is removed.

Execution: use conda esd_ind. Check CUDA and reservations; absent admission use
CPU1thread. Reuse file handles for repeated ranges, preserving exact decoder,
coordinates and numeric calculations. Record attempted and successful identities
separately, hash/stream times, payload bytes and progress after EVERY pair.
At most76 identities across original+continuation; expected<40 minutes; new
external2400s limit (internal2350s), total original+new bound3300s, under2h.
If incomplete or a pin/certificate fails, record and stop; no automatic retry.
No collection, new fingerprints, training, raw-block persistence or packages.

Pass/fail BEFORE running: original criterion remains >=50% source-present
coverage and <=.05 false per answered query. Report all4 cases under EACH
hypothesis; source-out wrong/useful/abstain separately; zero answered denominator
is undefined. Any missing input stays unresolved. Small exposed dependent roots
cannot establish a general error guarantee or approve a final detector.

Implementation plan (inline; no commits):
- Test resume on complete paired evidence, missing inputs, invalid/duplicate
  hypothesis rows; test nonsequential buffered ranges and truncated input.
- Add minimal resume/output/budget plumbing; keep all scientific formulas fixed.
- Review continuation invariants once before launch; validate certificate.
- Run once within bound, verify72 unique pairs/144 hypothesis rows/16 decisions;
  report outcomes, provenance, costs and limits in RESULTS and shared table.

GPU execution clarification before launch: sandbox hides CUDA; activated esd_ind
outside sandbox exposes8 idle GPUs. Reserve ONE within B's2-slot budget using
existing training-gpu-v1 physical UUID flock and gpu_training admission helpers,
plus no-compute-process check twice >=5s apart. A and C report no reservations.
GPU FP64 torch.linalg.svd(gesvd) only supplies arbitrary finite test vectors;
all noise/norm/projection/witness certificates remain CPU FP64, preserving their
reviewed validity independently of GPU SVD accuracy. Validate25 synthetic cases
with GPU vectors. Compare one fixed synthetic256x12288 CPU/GPU SVD including
transfers; adopt GPU only if faster, otherwise release it and retain CPU.
Record timings and backend; no device use merely to occupy an idle slot.
