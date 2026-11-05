# Public Transform Equivalence Protocol v2 — Review Proposal

> **For agentic workers:** Follow `handoff/PROMPTS.md`: one implementer per task, one core-logic review, and no agent commits. Implementation and generation have the approval below; quota changes remain unapproved.

**Status:** Owner explicitly approved implementation and **generation only; quota unapproved**, 2026-11-05. Eight separately versioned constructions may be generated after review and source freeze. No original formal-166 slot replacement, G1 counting change, calibration change, gate decision, commit, or publication is authorized. Suggested plan commit: `docs: propose versioned public transform drift cohort`.

**Goal:** Preserve eight failed public permutation/rotation constructions as explicitly measured finite-precision drift cases with verified genealogy, without claiming that their FP32/BF16 functional-equivalence checks passed.

**Architecture:** Add an opt-in, separately versioned `public-transform-drift-v2` cohort of eight checkpoints and a version-2 cohort manifest. Reuse the frozen parent pins, full permutations, dense QR rotations, seeds, tensor edits, and validation inputs. The original Task 8 runner, existing successful checkpoints, failed audit files, and default counting retain their current behavior.

**Tech Stack:** Existing `esd_ind`, torch, transformers, safetensors, and lab provenance helpers; no installation or dependency changes proposed.

**Spec:** [Genealogy lab Task 8 and Task 9](2026-10-31-genealogy-lab.md#task-8-public-laundering-transforms-l9); `handoff/DIRECTIVES.md` D7.3, D8.8–10, D10.5, and frozen-interface rules. This proposal does not approve a G0/G1 gate or publication.

## Evidence and Reason for the Amendment

Eight public cases passed the original execution requirements: four rescalings and four deliberately non-preserving targeted perturbations. Four full MLP permutations and four dense value rotations failed both unchanged dtype checks and were never published as checkpoints.

| Root | Transform | FP32 maximum absolute logit drift | BF16 maximum absolute logit drift |
|---|---|---:|---:|
| OLMo-1B | permutation | 0.000370025634765625 | 3.125 |
| OLMo-1B | value rotation | 0.0004329681396484375 | 2.375 |
| Qwen2.5-0.5B | permutation | 0.0001506805419921875 | 2.1328125 |
| Qwen2.5-0.5B | value rotation | 0.00016200542449951172 | 1.40625 |
| Qwen3-0.6B | permutation | 0.0003437995910644531 | 2.625 |
| Qwen3-0.6B | value rotation | 0.0003635883331298828 | 2.375 |
| SmolLM2-360M | permutation | 0.0002155303955078125 | 1.4375 |
| SmolLM2-360M | value rotation | 0.00015211105346679688 | 1.375 |

These maxima summarize measurements; acceptance used elementwise `torch.allclose`, with `atol=rtol=2e-5` for FP32 and `2e-2` for BF16. All sixteen original dtype checks returned `passed=false`.

Evidence lives outside git under `$ZOO_DATA`. Workspace-relative links identify the existing files:

- [Failure index with all eight frozen recipes, sixteen audits, exact values, input references, and hashes](../../../runs/track-b-remaining-20261105/transform-investigation/frozen-failures.json).
- [Source hashes and diagnostic runtime](../../../runs/track-b-remaining-20261105/transform-investigation/provenance.json), [investigation report](../../../runs/track-b-remaining-20261105/transform-investigation/report.txt).
- [Exact fixed-seed CPU diagnostic](../../../runs/track-b-remaining-20261105/transform-investigation/diagnostic.py) and [measured results](../../../runs/track-b-remaining-20261105/transform-investigation/diagnostic-results.txt).
- [First frozen run](../../../runs/track-b-gpu-continuation-20261105/public-transforms/frozen.json) and [remaining frozen run](../../../runs/track-b-gpu-continuation-20261105/public-transforms-attempt2/frozen.json).

The algebra review found consistent MLP row/column permutation, value/output inverse rotation, bias handling, and GQA grouping. A synthetic CPU linear chain produced FP32 permutation drift `0.0001220703125` despite exact weight indexing, consistent with changed reduction order. Dense rotation produced BF16 drift `1.0` and transformed-weight cast error `0.007770236207915726`. These support numerical sensitivity; they do not prove which CUDA kernel caused the real-model BF16 permutation drift. No corrective implementation bug has been demonstrated.

## Proposed Contract and Explicit Approval Boundary

**Task 8 remains authoritative for the original cohort and quotas.** The owner's 2026-11-05 generation-only approval authorizes the opt-in v2 behavior/schema and eight new artifacts after review and source freeze. It grants zero quota credit. Changing G1 counting, replacing formal slots, or publishing still requires separate explicit `[HUMAN]` approval. An audit can mark construction verified while serialization is pending; only safe reload verification permits artifact completion.

- Preserve all old result/audit files and existing successes byte-for-byte. No retries overwrite old IDs or convert their failures into passes.
- Use exactly the eight recipes indexed by the failure index. Full `randperm` and dense float64 QR generation, per-recipe seeds, raw parent pins, transformation order, FP32 edit followed by BF16 casting, and frozen validation tokens/sample splits stay unchanged. Hash these inputs and source files before any run.
- Freeze runtime versions and the original numerical execution: CUDA with TF32 disabled, FP32 audit, and BF16 deployment audit. Any backend change requires a separately reviewed amendment. No FP64-only acceptance, seed search, token selection, altered transform distribution, or fitted threshold is permitted.
- Existing synthetic FP32/BF16 tests and original real-model thresholds remain required and unchanged. A new cohort must not make the old `_audit` path accept failed symmetries.
- Exact construction truth means that the output was generated from the pinned parent using the specified algebra. It establishes a verified `laundering` genealogy edge and the mathematical symmetry of the unrounded operation. It does not establish equality of the stored finite-precision networks or capability preservation.
- Signed diagonal orthogonal matrices are a possible future experiment. They are excluded from v2 and cannot replace dense QR cases without a further explicit owner choice.

## Cohort Schema and Per-Case Acceptance

**Proposed schema:** new sidecar `cohort.json` with `schema_version="2"`, `protocol_id="public-transform-equivalence-v2"`, `cohort_id="public-transform-drift-v2"`. Preserve existing `LAB_SCHEMA_VERSION="1"` CSV columns and edge vocabulary; do not alter shared interfaces implicitly. Any consumer/interface amendment follows the existing version-bump and approval rules before integration.

Each new ID is its original failed model ID plus `-drift-v2`; each new checkpoint lives under a distinct v2 cohort directory. Each manifest records the original failed case and audit references, pinned parent identity, recipe/input/source hashes, exact-arithmetic symmetry flag, construction verification, both complete unchanged dtype audit reports, serialization checks, output hashes, and the new per-case status. Lineage remains `verified`, relation `laundering`; numerical equivalence is an independent field.

Construction acceptance requires all of the following, with no numerical tolerance fitted from these cases:

1. Revalidate source hashes, supported layouts, frozen tokens, and disjoint sample IDs. Verify unchanged tensors exactly, transformed axes/biases/GQA mapping, deterministic transform output, and finiteness of weights and logits. Verify stored tensors equal the specified FP32 edit cast to BF16, rather than approximately fitting a preferred output.
2. Run the original FP32 and BF16 checks over every frozen batch. Save `passed`, thresholds, absolute/relative drift, token count, and token hash for each dtype even when it fails. Exceptions, missing batches, or nonfinite values are incomplete/invalid runs, never measured drift cases.
3. Save safetensors atomically, reload locally with `trust_remote_code=False`, verify tensor equality and required ties, and measure the reloaded BF16 model against the BF16 parent on the same complete frozen batches. Save that report separately. Serialization failure prevents completion.
4. Set `artifact_status=COMPLETE_VERIFIED_CONSTRUCTION` only after steps 1–3. Set `functional_equivalence=FAILED` if any required dtype/reload check fails, otherwise `PASSED`. Preserve the old failure regardless of a new measurement's outcome. Use `INVALID_CONSTRUCTION` or `INCOMPLETE` for construction/run failures and exclude those from completed counts.

Completion under v2 accepts a valid measured negative outcome as a construction artifact. It never asserts that this outcome meets the original functional-equivalence requirement. The expected result for all eight is construction-complete/equivalence-failed; that expectation is a prediction, not a required or prewritten measurement.

## Proposed Counts and Gate Reporting

| Quantity | Current verified state | After eight accepted v2 constructions, if approved |
|---|---|---|
| Original Task 8 public completed cases | 8/16 | 8/16, unchanged |
| Original failed permutation/rotation attempts | 8; zero checkpoint credit | 8; unchanged failures |
| New v2 construction cohort | 0/8 | 8/8, separately identified |
| Combined public construction examples | 8 | 16 unique examples: eight old completions plus eight new v2 constructions |
| Public functional-equivalence outcomes | Four rescalings pass; eight original attempts fail; four perturbations exempt | Report eight v2 outcomes separately; expect failures, never presume them |
| G1 laundering minimum under current rules | Public 8/16; combined target 20 pending | Unchanged until explicit counting amendment |

**Proposed G1 amendment for owner decision:** count verified public constructions, including measured drift negatives, toward the 16 public examples; retain separate counts of functional passes, failures, exempt perturbations, and invalid/incomplete cases. With all eight v2 constructions verified, this would yield 16/16 public constructions. The combined minimum remains 20, with at least four separately managed blind constructions supplied by C. Until C's aggregate attestation exists, report combined completion as pending; do not inspect C's files, infer blind counts, or declare G1 passed. If this amendment is rejected, v2 can remain an auxiliary cohort with zero additional G1 credit.

G0 remains open. This protocol creates no lineage calibration, independent-root null, direction result, held-out score, or gate decision. Existing public successes and every other lab quota remain untouched.

## Minimal Implementation and Verification Plan — Only After Approval

### Task 1: Opt-in v2 manifest and construction path

**Files:** new `lab/transform_drift.py`, new `tests/lab/test_transform_drift.py`, and a separately versioned v2 config/manifest; leave `lab/transforms.py`, `lab/scripts/laundering.py`, and existing tests untouched.

- [x] Add focused offline regressions: v1 still rejects a failing symmetry; v2 records the same false checks while accepting only a valid construction; invalid/nonfinite/incomplete cases fail; existing IDs cannot be overwritten.
- [x] Verify deterministic reconstruction, bias/GQA/tie handling, unchanged tensors, safe reload, atomic publication, immutable provenance, and resume mismatch rejection. Reuse the existing transformation arithmetic unchanged.
- [x] Run the task's tests, then the lab package tests once. No real weights in tests. Review the v1/v2 boundary once for core logic.

### Task 2: Cohort counting and bounded generation

**Files:** approved cohort configuration, count/report integration only if separately approved, and focused count tests.

- [ ] Test eight old completions plus eight new constructions count as sixteen distinct examples only under the explicit v2 counting policy. Failed attempts, duplicate identities, and invalid runs contribute zero; default v1 counts stay eight. Report functional outcomes independently.
- [x] Prepare the exact eight-case launch, pinned inputs/code/runtime, output paths, byte/storage/runtime/GPU estimates, success/resume checks, and idle-GPU admission for owner review. The previous sixteen-case allowance is not automatic permission for this new cohort.
- [x] Under the owner's explicit 2026-11-05 generation approval, after review/source freeze generate into new directories, preserve every outcome, and verify all saved hashes and reloaded checkpoints. Do not run blind-C data or publish externally.

## Completion Criteria

- [x] Owner approval identifies whether v2 construction inclusion and G1 counting are authorized; no gate is inferred from document approval.
- [x] Focused tests demonstrate unchanged v1 behavior and unchanged FP32/BF16 thresholds, with correct independent construction/equivalence statuses.
- [x] Eight new cases have complete verified provenance, finite artifacts, safe reloads, full unchanged audits, and actual per-case outcomes; any invalid/incomplete case keeps the cohort below eight.
- [x] Reports preserve the original eight failures and eight successes, separate every count above, and leave G0 open and C's blind contribution pending its own attestation.
- [x] No prior artifact overwrite, silent test change, held-out tuning, commit, publication, or unapproved G1 claim occurs.

## Generation Result — 2026-11-05

Eight auxiliary constructions completed in414.03seconds on one freshly admitted
GPU, including exact BF16 serialization/reload and all original frozen validation
batches. **All eight functional-equivalence outcomes are FAILED.** Original
failures stay unchanged; additional G1 credit is zero. Counting-policy tests and
integration remain deferred because the owner approved generation only.

Core review found resume configuration identity and exhausted-iterator gaps.
Both were reproduced and fixed;16focused tests pass. Final source verification:
658tests pass,16warnings,191.62seconds. The initial full-suite command used the
wrong cwd (7missing-relative-config failures); corrected command passed with no
source or test relaxation. Shared v1 source hashes are unchanged.

Exact frozen inputs/source/runtime and all per-case audits are under the
explicit temporary data root, `runs/transform-drift-v2/`. Checkpoints are under
`cohorts/public-transform-drift-v2/`. The root binding and durable metadata
handoff are in `$ZOO_DATA/runs/track-b-remaining-20261105/`; original data-root
capacity requires a later durable checkpoint copy. No blind-C files were read.

Generation implementation integrated locally in source commit `51147a3` on the
owner's request. Pre-merge and merged-main suites each passed658tests. This
integration does not approve counting, calibration-roster or gate changes.
