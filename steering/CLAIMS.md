# Claims and hypotheses: verify before we solidify

No claim in `DIRECTION.md` is frozen until it passes four checks:

1. **Literature:** is it new, and what exists already?
2. **Theory:** what does it assume, and where should it break?
3. **A cheap test on data we already have,** with pass and fail written down
   before running.
4. **A skeptic's review.**

Claims about real-world use must also hold on real Hub models. The doer updates this file after independent review. Only the owner makes a verified claim **solid** for a paper. The final evaluation plan lists approved conclusions separately from open questions; a scheduled date or completed checklist never verifies a claim. The end product is the database plus query framework, not a lineage result alone. Model-choice/clustering/prediction ideas need practical evaluation before claims.

**Status ladder:**

- **hypothesis:** stated, not checked;
- **checked:** literature and theory done;
- **piloted:** the cheap test passed;
- **verified:** the full test passed and the skeptic agreed;
- **solid:** verified and explicitly approved for the paper claim list;
- **dropped:** with the reason.

## Current scientific position

D8 admits published construction as documented real truth; private execution details are caveats. New evaluations keep pinned recipes, prospective rules and unknowns explicit. D9 uses two same-architecture pools, not the mixed5k roster. CHECKLIST.md orders work; D11 retires old track/date machinery without loosening evidence standards.

The four Heretic edits need recipe-rank/control scoring; merge-source search and documented real-mark tests remain open. The four sentiment projections do not establish refusal behavior. Saved SFT witness rows already use siblings; the proposed descendant-witness explanation for0/8 is unconfirmed. A changed test needs a scientific reason.

Prior12/12 exact-quantizer source results and18/18 known-source merge fits retain their narrow scopes. The best comparator source ranking7/12 versus our12/12 exact-replay answers is a lab task result, not general precision or every method's native claim. C1–C5 remain checked; C6 is a later hypothesis. No broad claim is promoted by this workflow change.

## Literature check (Nov 7, documentary follow-up Nov 8)

| Work                                                                                                                                         | What it does                                                                                                                                                                                    | Effect on us                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| modelDNA (2607.10617), LineageBench, modelDNA Atlas                                                                                          | Weight-based lineage verification and merge decomposition. LineageBench has 15 suspect models and 8 candidate parents (AUROC 1.0, 13/13 correct). The Atlas covers 68 Hub models and 123 edges. | Our controlled cohort includes 112 constructed children among 166 entries; the measured snapshot has 703 models. Exact construction records, not generic size, are the distinction to check (D5). Same-family detection alone is not new.    |
| "Training Leaves Traces" (2608.14929)                                                                                                        | Weight-only relatedness score. It separates fine-tuned, LoRA-merged, pruned and quantized descendants from independent and distilled models. The score is symmetric: no direction or edit type. | **New to our plan; must be a baseline for claim C1.** Typing and direction remain open.                                                                     |
| Witness Overlap (2609.31784, arXiv 2026; NeurIPS acceptance not confirmed)                                                                   | Orients parent→child among fine-tunes using a third family member (a "witness"): 95.3% on 176 checkpoints in 16 families                                                                        | **Corrects us:** order for ordinary fine-tunes is not hopeless from weights when a witness exists. It is a baseline and a possible component for C2 and K9. |
| TrFP (2610.34675)                                                                                                                            | Fingerprints from activation trajectories. It needs forward passes. Claimed robust to fine-tuning, pruning, quantization, merging, permutation and adaptive attacks.                            | The robustness comparison for disguise. Different cost: it runs the model; we read weights only.                                                            |
| [Provenance sets (MPS, 2602.00772)](https://arxiv.org/html/2602.00772v1) | Returns possible source/ancestor sets; tests coverage and set size over candidate pools 5, 10, 20, 50, 80, 100. Collection: 455 models labeled from declared bases. | Pool-size curves and calibrated source sets already exist. Their ancestor coverage differs from our immediate-parent error test; exact construction records remain a possible distinction (D5 approved). |
| Cisco `model-provenance-kit` (GitHub)                                                                                                        | A practical provenance tool with an edit-type taxonomy (fine-tune, adapter, quantization, merge, and more)                                                                                      | **New to our plan:** a real-world competitor for the `check` command. Learn what it measures before we position ours.                                       |
| Refusal-removal ("abliteration") detection                                                                                                   | Found only activation-based or behavior-based detection                                                                                                                                         | A weight-only detector looks open.                                                                                                                          |
| AttnPrint/DistillTrace, SCOUT, BULLPEN and GGUF reproducibility | C completed the primary-source access/modality checks; detailed findings in its Nov 7 K2 primary-check note | Do not repeat that completed documentary check. Exact reconstruction needs build inputs; output-based approaches do not establish weight-only edit detection. |

## Claims

### C1: family at scale

**Hypothesis.** Fine-tunes stay very close to their base: updates are small
next to the weights. Independently trained models of the same architecture
are nearly unrelated. So partial-read sketches separate the two at any pool
size, with measured false positives.

**Where theory says it breaks:**

- long continued pretraining or heavy re-training, where updates stop being
  small;
- up-scaled or pruned models, where layer names no longer line up;
- models that copied only the embedding table from another model (partial
  sharing);
- naming differences across formats;
- **GGUF files from llama.cpp reorder the rows of the query and key
  matrices** for llama-family models. Row blocks must be un-reordered before
  comparing; this matters for zoo v1, not for P1;
- fused versus split attention matrices.

**Tests:**

- **K6:** at least 30 real independently trained models with the same config.
  - Pass: no false "same family" at the chosen threshold.
  - Fail: any false positive the threshold cannot remove without losing more
    than 5% of true pairs.
- **K8/D9:** false family/source answers per query at 10,100,1,000/all in two same-architecture pools, capped near 2,000 per base; fixed subsamples and actual reached sizes reported.
- **Baselines (K2):** Trace, Spectral Signatures, modelDNA, and "Training
  Leaves Traces", on the same pairs.

**Status:** checked. Current family coverage and exposure limits are in RESULTS.md; broad scaling remains unverified.

### C2: edits that leave an exact mark

Each edit type is its own hypothesis. Lab results count only together with
real Hub examples (K7).

**D7 approved (Nov 9):** an edit mark supports source compatibility only when
its evidence is informative. A unique immediate source requires tests that
distinguish relatives within the stated candidate pool; otherwise the source
is unresolved. Neither a shared mark nor an inconclusive test proves a source.
Successful lab quantization/replay and merge-coefficient results retain their
original scope. Existing pass/fail rules below are unchanged; no claim is promoted.

| Edit                                       | Exact mark (theory)                                                                                                                                                                                                                                                                         | Parent?                            | Order?                                                                                 | Where it breaks                                                                                                                                                                                                                        | Cheap test (pass rule written first)                                                                                                                                     |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Quantization** (quantize, then store)    | Within each quantization block, the child has at most 2^bits distinct values (16 for 4-bit). This survives bf16 storage, because each grid level maps to one stored value. Re-quantizing the true parent reproduces the child.                                                              | informative replay supports a candidate; unique only when relatives are distinguished in the stated pool | yes: the child is the one on the grid                                                  | **Native low-precision releases:** if the original was released in FP8, its bf16 copies are the children and also sit on the grid, so the order needs metadata. Quantize-then-fine-tune (QLoRA merged) erases the grid.                | 12 lab quantizations against 30 unquantized models. Pass: 12/12 detected, 0 false; re-quantizing the parent matches.                                                     |
| **Directional ablation** (weight-edit operator) | Classic removes a shared direction; Heretic can apply recipe-specific low-rank changes separately per edited matrix. A weight signature does not establish refusal behavior. | informative edit evidence may support a source; unique source requires distinguishing relatives (D7) | test where operator evidence supports order; do not infer it from low rank alone | Noise, export and later edits can blur marks. Use the actual recorded recipe rank; do not force FULL/rank3 Heretic into rank1. | Score all 4 Heretic models under a prewritten recipe-rank/noise rule; SFT/DPO/adapter/unedited controls must not be flagged. Report four classic sentiment projections separately; then apply tests to D8 cases. |
| **Merge**, linear or SLERP                 | Child minus base is a mix of each parent minus base (SLERP adds a small base term)                                                                                                                                                                                                          | informative coefficient fit supports candidate sources; unique sources require relative discrimination | yes, **only if** merge weights are non-negative: an affine relation alone is symmetric | Parents that are near-duplicates are not identifiable; report them as a group.                                                                                                                                                         | K3 refit on deltas. Pass: coefficient error ≤0.05 on most linear and SLERP merges (no more than 4 misses).                                                               |
| **Merge**, TIES or DARE                    | Trimming or dropping leaves many entries **bit-identical to the base** in a random-looking pattern, an extra exact mark                                                                                                                                                                     | informative edit evidence may support candidate sources; unique sources require relative discrimination | yes                                                                                    | Small merges may have few trimmed entries                                                                                                                                                                                              | Pass: all TIES and DARE merges flagged by the identical-to-base share, with 0 false flags among the SFT and LoRA models                                                  |
| **Adapter** (LoRA merged into the weights) | Tensors the adapter did not touch are **bit-identical to the parent**. Touched tensors change at low rank, but only above the bf16 noise floor.                                                                                                                                             | family/base support; not immediate source among sibling adapters unless readable rank and other evidence distinguish candidates | **no:** a low-rank difference looks the same in both directions                        | Small LoRA updates fall below the floor and the rank test abstains. Report each LoRA's update size first.                                                                                                                              | Report family/base support, sibling ambiguity and readable-rank coverage for 24 lab adapters. Where above rounding noise, retain configured-rank accuracy target ±25%; unchanged tensors alone are not a24/24 unique-source pass rule (approved check-in). |
| **Ordinary fine-tune** (SFT, DPO)          | No exact mark. Two candidate signals for order: (a) the witness geometry (Witness Overlap); (b) a hypothesis from our own spectral work, that heavy-tail statistics shift in one direction with training. Spectral numbers need full reads (lab has them).                                  | family and closest candidates only | possibly, with a witness or spectral signal; otherwise from Hub upload time            | No witness in the pool; tiny updates (DPO); spectral drift may be too weak                                                                                                                                                             | 16 lab SFT/DPO edges. Report witness-method and spectral-cue order accuracy. No pass rule yet: this is exploratory until a prospective final protocol is fixed.                                          |

**Status:** checked overall. Quantization and known-source linear/SLERP tests have passed in the lab; the remaining edit tests and real-model checks are open. Each hypothesis moves separately; no parent/order guarantee follows from these subtests.


### C3: parent decisions calibrated against relatives

**Hypothesis.** If updates are roughly independent, a child is closer to its
parent than to its siblings or grandparent. So "nearest relative = parent"
holds **when the parent is in the pool.**

**Where it breaks:**

- **The parent is missing from the pool.** Then the nearest relative is a
  sibling or grandparent, and naming it is a false parent. This likely
  explains much of the 8.28 false parents per query on the pilot, where many
  true parents are absent.
- Merges, where several parents are close.

**Tests:**

- **Leave-the-parent-out on the lab** (new; part of K5). Remove each true
  parent from the pool, then measure how often a sibling or grandparent is
  wrongly named, and whether a "parent not in pool" rule catches it.
  - Pass: a rule exists that keeps false parents ≤0.05 per answered query
    while answering at least 50% of queries where the parent is present.
- **The comparison set** is close relatives (siblings, grandparents,
  quantized copies), not unrelated models.
  - Note: 15 comparison pairs with 0 false alarms still allows a false-alarm
    rate up to 20%. About 300 relative pairs are needed to show a rate below
    1%.

**Status:** checked; handling a missing source is important, but not itself new. Our close-relative and immediate-parent tests must be compared with prior source-set methods (D5).

### C4: the lab as a benchmark

**Hypothesis.** A useful lineage benchmark with exact construction records
across many edit types. Any claim that it is the largest of its kind requires
a separate comparison; size alone is not a novelty claim (D5).

**Literature:** LineageBench has 15 suspect models; Witness Overlap evaluates
176 checkpoints; MPS collects 455 models labeled from declared bases. Ours has
a 166-entry cohort including112 formal constructed children and bases/reference
controls; the documented real-model tier is being built under D8 (K7).
Exact construction records and coverage of edit types need direct comparison.

**Test:** at least 4 published methods run on it (K2), with the scorer
released.

**Status:** checked.

### C5: the Hub measured against what it declares

**Hypothesis.** Exact file and block hashes reveal duplicates, re-uploads,
quantized copies and tensor sharing at almost no read cost. The random sample
gives honest prevalence estimates.

**Literature:** the modelDNA Atlas covers 68 models; nothing at our planned
scale was found.

**Test:** run on the 320-model sample first, then scale.

**Status:** checked; mostly engineering risk.

### C6 (later): weights are worth few benchmark questions

**Hypothesis.** As a score predictor, weights are worth about 3–39 questions,
and nothing within a family.

**Where it is weak:**

- the comparison baseline was a plain training median;
- the data is old and unpinned.

**Test before claiming:** pinned scores (K10), and strong no-weights baselines
(family averages, and question-profiling services such as BULLPEN once it is
verified).

**Status:** hypothesis.

## Theory notes for agents

- **Never use a fixed tolerance without a noise model.** The tree test and the
  disguise checks both failed on tolerances below normal bf16 rounding. Every
  pass/fail threshold states the noise it is set against.
- **"The code exists" is not "the result is known".** Record runs in
  `RESULTS.md` with their status.
- **Check the update size before any rank-based conclusion.** Below a relative
  update of about 3×10⁻³ in bf16, rank estimates are rounding noise; the
  existing abstention rule handles this.
- **Identical tensors are the strongest evidence we have** (adapters,
  partial fine-tunes, TIES and DARE patterns). Use them before statistical
  similarity.
