# Plan (from 2026-11-07)

This file holds the corrective actions, the milestones, and the reconciliation
of every earlier plan and checklist item. Status words: **open**, **built**
(code only), **tested** (ran on real data, numbers in `RESULTS.md`),
**decided** (claim settled), **dropped**. The steward keeps the status column
current.

## Current execution focus (Nov 8 holistic review)

The collection/database and queries are an internal prototype; the controlled
lab and its backup are valuable assets. Reliable real-world verdicts and the
main paper still need broader evidence. Model-Zoo and Zoo-Keeper serve that
same outcome; Model-Research is frozen, and selection and heritability work stay parked.

**Next shared output, at the existing Nov 14 review:** one evaluation summary
in RESULTS.md, linked to existing run reports: coverage of every planned lab
edit group, the usable real-model cases and their evidence tiers, matched
published-method progress, and A/B's proposed query protocol with actual costs
and resource gaps. Include a trace through the existing query/scorer path to
its evidence and uncertainty. This is a progress review, not a new pass gate,
sample quota, interface feature, or earlier deadline for all K5/K7/K2 work.

- A finishes inputs/resources and brings one concrete protocol to B; B returns
  a prompt scientific review. L1 remains blocked until the review and storage
  feasibility are resolved. A mixed 5k roster is not 5k comparable candidates.
- B covers the existing lab edit groups and missing-parent cases. C supplies
  complete source-tiered real-model cases and runs comparisons; detector/noise
  decisions belong to B. Do not multiply probes on a single model.
- C interleaves K2 with K7 whenever historical provenance is blocked. Missing
  construction records restrict conclusions, not all descriptive evaluation.
- Keep K4/K14 to the already-required bounded checks; record an inconclusive
  outcome when appropriate. No new tree/disguise programme. Reopen completed
  work only for contrary evidence or a gap that changes the next verdict.
- Judge progress by coverage, wrong answers, useful answers/abstentions, real
  model transfer, comparisons and full workflow cost. Routine unit-test counts,
  per-file inventories and per-block diagnostics stay in linked run reports.

**Freeze conflict:** Nov 25 can solidify only claims already verified. K7 is
due Dec 1–5 and K8 Dec 10, so those later results cannot be assumed at Nov 25
or silently admitted to the Dec 4 claim list. [D6](cards/D6-test-plan-and-claims.md)
asks to separate approved claims from unanswered questions in the evaluation
plan. Until decided, current K12 rules and separate prospective run protocols
apply. Dates, thresholds, truth tiers and held-out partitions are unchanged.

## 1. Corrective actions

Each action names its owner track, what "done" means, and a date. They all
serve the corrected claims in `DIRECTION.md` section 6.

| ID | Action (and why, in one line) | Owner | Done when | Due | Status |
| --- | --- | --- | --- | --- | --- |
| **K0** | **Save the lab.** Approved archive now verified in public, manually gated Hugging Face repositories; measured inventories replace the early estimates (`DATA.md`). Preserve local copies and restricted-data exclusions. Never re-upload third-party models. | A (after the overseer creates the repositories and the token) | Remote copies verified by file list, sizes and safetensors sha256; revisions recorded in `RESULTS.md`; the real storage figures replace the estimates in `DATA.md` | /tmp copies within a day of the token; irreplaceable Nov 14; all Nov 21 | tested: approved archive verified, 139 constructed checkpoints; local copies retained (Nov 7) |
| **K1** | **Start steering.** First steward run: fill `BOARD.md`, screen cards D1–D4, and confirm the track notes. | Steward | `BOARD.md` filled; cards screened, with skeptic lines | Nov 10 | tested: first run complete; all three leads acknowledged Nov 7 brief |
| **K2** | **Run the published baselines on the same questions.** Without baselines, no P1 framing can be reviewed. Order: Trace and Spectral Signatures (ported from Model-Research), then modelDNA and "Training Leaves Traces" (2608.14929), then Witness Overlap (direction). MoTHer only if cheap. No black-box test for January. The initial documentary check is complete (D5); use its findings in `CLAIMS.md`, rather than repeating it. Continue targeted reading only for unresolved comparison questions. | C | Each baseline has same-family, parent and edge-type numbers (where it supports them) on the lab, plus bytes read per answer, in `RESULTS.md`. A one-page note on what the new papers already cover. | first two Nov 21; all four Dec 5 | tested (partial): documentary check and two native adaptations on small L12 cohort; final lab/Hub and remaining methods open |
| **K3** | **Redo the merge fits on deltas.** Fit "child minus base" against "each parent minus base" (base term kept for SLERP). The old raw-weight design made the columns look identical. Compare with the published mergekit configs. | B | Coefficient error per merge type against each config; abstain only for exact-duplicate parents | Nov 14 | tested: 9 linear + 9 SLERP meet coefficient target; narrow lab scope only |
| **K4** | **Redo the tree test with a tolerance set by measurement noise.** Run a simulated exact tree with the same noise as a check that the test can pass at all. Report how large the violations are, not only how many. | B | Violation sizes for real lab trees vs the simulated tree; a gate card if the pivot conclusion changes | Nov 14 | tested (partial): Smol360 partial-read repeat/control; other families and full reads open |
| **K5** | **Edge typing and the exact edge tests on the lab** (claim C2). "What was done to this model?" is a distinctive, practical claim, and the code already exists. Add four exact tests: <br>• quantized child on a value grid (parent and order); <br>• refusal removal leaving one removed direction shared across layers (parent and order); <br>• a merge as a mix of parents and base (parents and order); <br>• an adapter as a low-rank difference (parent only). <br>Use the exact pass rules in `CLAIMS.md` (C2 table), including the identical-to-base pattern for TIES and DARE merges and identical untouched tensors for adapters. Calibrate parent decisions against close relatives, not unrelated models. Run **leave-the-parent-out**: remove each true parent from the pool to test the "parent not in pool" rule (C3). | B | A table of predicted vs true type over all edge types (LoRA, full SFT/DPO, merge, quantization, ablation, decision head). LoRA rank accuracy at 8/64/256. Per edge type: parent and direction coverage and precision. Ablation-detector precision and recall on the lab and on Hub models named "abliterated" or "uncensored" (name matching is the baseline to beat). | Nov 21; method check Dec 12 | tested (partial): lab quantization 12/12 plus 0/30 controls and 12/12 exact replays; parent-out subset passes; other edit groups and real-model validation open |
| **K6** | **Find real same-architecture independent models.** These are the hard negatives a compliance check meets in practice: same config, trained from scratch by someone else. | A (moved from C by card D3) | At least 30 pinned models with partial-read fingerprints; same-family false-positive rate against them | Nov 21 | tested: 30 publisher-supported scratch-trained BERT controls, 0/435 false matches; narrow diagnostic only, seed independence/recall/decoder transfer unresolved |
| **K7** | **Build verified Hub truth at 7B and above.** The lab is ≤1.7B and reviewers will ask whether results transfer. Use vendor pipelines that publish their intermediates (OLMo-2 7B/13B SFT→DPO→RLVR, Tülu 3 8B, SmolLM2-1.7B) and at least 20 (stretch 50) merges that ship `mergekit_config.yml`. Include Hub models with known parents that were refusal-removed or quantized: the exact-mark tests (K5) must hold on real Hub models, not only on our lab. | C (launch card if over 100 models) | Truth table plus fingerprints; same-family, edge-type, exact-mark and merge results on it | first set Dec 1; complete Dec 5 | tested (partial): source-tiered Hub candidates; 0 verified construction edges; one selected-block diagnostic uninformative; usable full comparisons open |
| **K8** | **The 5k candidate pool and the cost gate.** This answers "does same-family detection hold as the pool grows?", which is the scaling plot. Also report bytes and seconds per model by size stratum, including the cost of checking a new 8B and a new 70B model. | A prepares a launch card; the overseer launches | Launch card by Nov 21; run finished and analyzed by Dec 10; false matches per query at each pool size; cost table | Dec 10 | open: 8B/70B cost and completed-artifact resume tested; metadata/header preparation under way; query review, final eligible pool and storage/budget unresolved |
| **K9** | **Parent ordering, done realistically.** Weights narrow the family and the closest candidates. Hub metadata (upload time, declared base, author, tokenizer and chat template) orders them. Test two order signals for ordinary fine-tunes: Witness Overlap's witness method, and the spectral-drift cue on full-read spectra (`CLAIMS.md`, C2 last row). Exploratory until Dec 4. Claim only what this shows. | B | Parent-ranking accuracy on the lab and on verified Hub truth, with and without metadata | Dec 5 | open |
| **K10** | **Verify scores by weight hash, not only by commit.** A leaderboard score applies to our pinned revision when the weight files' hashes match at both commits. New commits rarely change weights. | A | Verified score joins re-counted (was 24/150) | Nov 21 | tested: 144/150 weight-hash joins verified; six lack evaluation revision |
| **K11** | **Heritability: parked.** Card D3: no paper target, and no workshop papers. If it is ever resumed, the rule fix stands: the benchmark's fixed question count is the sampling size, and the leaderboard commit is the identity for score-only analysis. | — | — | — | dropped (parked) |
| **K12** | **P1 pre-registration.** One file lists the headline evaluations, metrics, splits, thresholds and baselines, frozen before the final runs. **It may contain only claims the overseer moved to solid at the Nov 25 gate.** It replaces the many small freezes. | B, with the steward | File frozen and its hash recorded in `RESULTS.md` | Dec 4 | open: Nov 25 verified-only claim restriction preserved; D6 proposes separate open questions in the Dec 4 document |
| **K14** | **Disguise check, in hours** (card D4). First check whether the 8 existing disguised models failed their behavior check only on tolerances too strict for floating-point rounding. Then test whether our fingerprint still matches all 8, and one exact neuron reordering inside one MLP layer. | B | Rows in `RESULTS.md`: the tolerance finding, plus survives or breaks for each case | Nov 21 | tested (partial): stored evidence and tensor permutation; behavior-noise and network comparison open |
| **K15** | **Plan zoo v1:** a `check` command, GGUF support (dequantized for lineage), a duplicates report, a diff gallery, pip packaging. It replaces the old V1 item. | A | Plan written and reviewed by the steward, with a decision card for the release | plan Jan 15; release Feb–Mar | open |
| **K13** | **Clean up the process** (`README.md`, "What changes"). Freeze the old STATUS and CHECKLIST. Keep track notes at 30 lines or fewer. Keep the results ledger. Stop credit and roster accounting. | All | First steward run finds no new writes to the frozen files | Nov 10 | tested: lead notes within cap; frozen handoff files match tracked versions Nov 8; ongoing rule |

## 2. Milestones and checkpoints

| Date | Checkpoint | Card |
| --- | --- | --- |
| **Nov 14** | K3 and K4 results. Merge and tree claims are decided by Dec 5 (`DIRECTION.md` section 7). | gate card if any claim changes |
| **Nov 21** | First baselines (K2); edge typing (K5); real independents (K6); score joins (K10); 5k launch card ready (K8) | L1 (5k pool launch) |
| **Nov 25** | **Solidify gate:** each claim in `CLAIMS.md` is moved to solid, reworded or dropped, using the cheap tests (K3–K6, K14), the literature check and the skeptic. Until then `DIRECTION.md` is the working direction, not a commitment. | G-card (solidify) |
| **Dec 4** | **Revised G0.** Cost gate on the pilot; same-family false positives against real independents; edge typing; merges; first baselines; pre-registration frozen | G1 (revised G0) |
| **Dec 10** | 5k pool analyzed (K8); verified Hub truth (K7, Dec 5); all four baselines (Dec 5) | — |
| **Dec 12** | **Method check (claim C2):** typed-edge direction coverage ≥50% at ≥95% precision, and parent false matches ≤0.05 per query on the edits where the method answers (abstentions reported), on the lab; the exact-mark tests also hold on the real Hub examples from K7 | G-card |
| **Dec 18** | **P1 go/no-go:** ICML (around Jan 22 abstract, Jan 28 paper) or COLM (around late March). Criteria in `DIRECTION.md` section 7, including the scale check (pool ≥5,000 and ≥200 labeled queries) before any freeze. | G2 (P1 venue) |
| Jan 4–8 | Results frozen (if ICML) | — |
| Jan 15 | Final ICML abstract go/no-go | G-card |
| Feb–Mar | **Zoo v1 public (K15):** pip install and a `check` command (family, license chain, declared-base verdict, near-duplicates, "was refusal removed?") for safetensors and GGUF; a Hub duplicates report; a `diff` gallery. The lab benchmark goes public only after the P1 submission (January if ICML, late March if COLM). Each release is a decision card. | decision cards |
| ~early May | **P3, the zoo paper:** the lab plus 10–20k measured models (not 50–100k). Scale is confirmed by K8 throughput. | G-card in March |
| Later | **Selection paper.** `recommend` uses measured scores plus a quick test on the user's examples. Weight-based score predictions are included only if they beat no-weights baselines. | — |

## 3. Plan index (existing plan documents)

| Plan | State | What happens now |
| --- | --- | --- |
| `Model-Zoo/docs/superpowers/plans/2026-10-28-weight-evidence-extractor.md` | built (X1–X10) | Reference only |
| `Model-Zoo/docs/superpowers/plans/2026-10-31-zoo-store-and-queries.md` | built (Z1–Z8) | Reference only |
| `Model-Zoo/docs/superpowers/plans/2026-11-03-gpu-fingerprint-pipeline.md` | built | Reference only |
| `Model-Zoo/docs/superpowers/plans/2026-11-04-full-mode-evidence-packages.md` | Tasks 1–6 built; 7–9 open | Task 7 (deltas for similarity-found parents) only if K9 needs it. Tasks 8–9 (full-scale sizing) only when K8 needs them. |
| `Model-Zoo/docs/superpowers/plans/2026-11-04-t2-recovery.md`, `2026-11-05-descriptive-evidence-names.md` | built | Reference only |
| `Zoo-Keeper/docs/plans/2026-10-31-genealogy-lab.md` | tested (166 models), except disguise transforms | Disguise transforms: card D4 |
| `Zoo-Keeper/docs/plans/2026-10-31-p1-lineage-engine.md` | built; E6 and E9 open | Continues as K3, K4, K5, K9 and K12. Tree recovery dropped. |
| `Zoo-Keeper/docs/plans/2026-10-31-heritability-and-exchange-rate.md` | built; results blocked | Paused (card D3); K11 if resumed |
| `Zoo-Keeper/docs/plans/2026-11-03-distributed-lab-training.md`, `2026-11-05-public-construction-roster-v2.md`, `2026-11-05-public-transform-equivalence-v2.md` | done | Reference only |

New plans for K-items are written only when the action needs more than a page.
The steward adds them to this index.

## 4. Open items in `handoff/CHECKLIST.md`, mapped

| Old item | Now |
| --- | --- |
| A0 day-one checks | dropped (server facts recorded) |
| L9 disguise transforms, S6 blind set | card D4 |
| L12 lab fingerprints | done (166/166 measured) |
| G1 minimum lab | part of card D4, then the revised G0 (Dec 4) |
| E6 tree-likeness | K4 |
| E9 full P1 evaluation | K12, then the Dec 18 gate |
| S3 heritability | parked (card D3) |
| S5, S10 baselines; S11 Hub baselines | K2 and K7. The black-box test is dropped for January. |
| S8 Hub query set | K7 |
| S9 label audit | later, after the P1 decision |
| R2, R5, R6 pilot manifests, calibration, smoke build | R2 and R6 done; calibration folds into the revised G0 |
| G0, G0-a, G0-b | revised G0 on Dec 4 (G1 card); the method freeze becomes K12 |
| H1 5k pool | K8 |
| H2 byte curves | part of K8 |
| P-1218 pivot check | Dec 18 gate (G2 card) |
| F1–F4 freeze and submit | unchanged, if Dec 18 says ICML |
| F5 heritability workshop | dropped (card D3: no workshop papers) |
| V1–V8 after ICML | Feb–Mar rows above; each release is a decision card |
| Weekly items | the steward's runs |
