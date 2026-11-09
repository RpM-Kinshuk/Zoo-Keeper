# Plan (from 2026-11-07)

This file holds the corrective actions, the milestones, and the reconciliation
of every earlier plan and checklist item. Status words: **open**, **built**
(code only), **tested** (ran on real data, numbers in `RESULTS.md`),
**decided** (claim settled), **dropped**. The steward keeps the status column
current.

## Current execution focus (approved Nov 9 check-in)

`CHECKIN-2026-11-09.md` and decisions D8–D10 supersede the earlier stop/park
instructions. They authorize the next tests and same-architecture scaling design;
they do not report new measured success. No track remains idle while useful plan
items are open. A blocked design needs another feasible test or a claim decision.

- **A, Nov 12:** metadata counts for four lab architectures; choose the two
  largest readable pools near 2,000 candidates each. Prepare owner launch L1
  with pinned inputs, comparable pool sizes, cost/space, stop and resume rules.
  Measure C's incoming D8 cases within agent-run limits; owner starts larger runs.
- **B, Nov 12:** all 4 Heretic recipe-rank tests against SFT/DPO/adapter/unedited
  controls; prewrite rules and rounding limits. **Nov 21:** merge-source search
  among candidates, not only fits with sources supplied. Test D8 cases as delivered.
- **C, Nov 11:** check the SFT order setup and a justified sibling-witness test.
  Saved Nov 8 SFT rows already name sibling SFTs, not DPO descendants; the
  suggested explanation is a hypothesis, not a correction established by records.
  **Nov 14:** first D8 batch, delivered incrementally to A/B, then matched methods.
- **D8 real cases:** published vendor compression methods, merge recipes,
  official training stages and public-tool ablations with recorded settings
  count as documented truth. Missing private execution details are caveats.
  Declared-only cases stay descriptive; they do not hold up other work.
- **D9 scaling:** 10/100/1,000/all same-architecture rivals with repeated fixed
  subsamples; report the largest pool reached and false-family/source answers.
  Unknown matches remain unknown; metadata base declarations do not prove independence.
- **D10:** plain language, a next useful task within one working day, and no
  closure by “cannot verify.” Necessity chooses the decisive next test; it does
  not shelve a claim whose required data can be collected or its test redesigned.

Keep completed results and their caveats. Do not overwrite raw protocols, split
assignments or historical labels; new D8/D9 evaluations record their own pinned
inputs and prospective rules. D7 still requires distinguishing relatives before
naming a unique source. The failed high-unchanged-share TIES/DARE shortcut is
not revived by the new merge search.

**D6:** Dec 4 plan separates approved verified conclusions from open tests;
later conclusions need verification, skeptic review and explicit overseer approval.
Nov 25 can keep claims open when evidence is still arriving; no claim is solid yet.
**Compute:** up to four idle GPUs project-wide, initial A0/B2/C2 with handoffs.
Use existing accelerated paths when useful; metadata may stay CPU. Agent jobs
must meet BOTH ≤100 weight models and ≤2 hours; larger work requires owner launch.
Resource estimates must fit the new selection, not the retired mixed-pool budget.

## 1. Corrective actions

Each action names its owner track, what "done" means, and a date. They all
serve the corrected claims in `DIRECTION.md` section 6.

| ID | Action (and why, in one line) | Owner | Done when | Due | Status |
| --- | --- | --- | --- | --- | --- |
| **K0** | **Save the lab.** Approved archive now verified in public, manually gated Hugging Face repositories; measured inventories replace the early estimates (`DATA.md`). Preserve local copies and restricted-data exclusions. Never re-upload third-party models. | A (after the overseer creates the repositories and the token) | Remote copies verified by file list, sizes and safetensors sha256; revisions recorded in `RESULTS.md`; the real storage figures replace the estimates in `DATA.md` | /tmp copies within a day of the token; irreplaceable Nov 14; all Nov 21 | tested: approved archive verified, 139 constructed checkpoints; local copies retained (Nov 7) |
| **K1** | **Start steering.** First steward run: fill `BOARD.md`, screen cards D1–D4, and confirm the track notes. | Steward | `BOARD.md` filled; cards screened, with skeptic lines | Nov 10 | tested: first run complete; all three leads acknowledged Nov 7 brief |
| **K2** | **Run the published baselines on the same questions.** Without baselines, no P1 framing can be reviewed. Order: Trace and Spectral Signatures (ported from Model-Research), then modelDNA and "Training Leaves Traces" (2608.14929), then Witness Overlap (direction). MoTHer only if cheap. No black-box test for January. The initial documentary check is complete (D5); use its findings in `CLAIMS.md`, rather than repeating it. Continue targeted reading only for unresolved comparison questions. | C | Each baseline has same-family, parent and edge-type numbers (where it supports them) on the lab, plus bytes read per answer, in `RESULTS.md`. A one-page note on what the new papers already cover. | first two Nov 21; all four Dec 5 | tested (partial): four methods matched on quantization and small ancestry controls; fixed witness panel complete, correlated contrast deferred; full independent real-model comparison open |
| **K3** | **Redo the merge fits on deltas.** Fit "child minus base" against "each parent minus base" (base term kept for SLERP). The old raw-weight design made the columns look identical. Compare with the published mergekit configs. | B | Coefficient error per merge type against each config; abstain only for exact-duplicate parents | Nov 14 | tested: 9 linear + 9 SLERP meet coefficient target; narrow lab scope only |
| **K4** | **Tree-distance retest.** Retain the measured noise/control comparison and scope limits; approved research direction does not resume tree recovery. Further checks need a concrete unresolved question, not repeated tests by habit. | B | Noise/control findings reported with violations; no unsupported claim that distance violations disprove ancestry trees. | review Nov 14 | tested in reported scope; check-in retains the pivot; not the priority over Heretic/merge/real cases |
| **K5** | **Test edit marks and find sources.** By Nov 12 score the4 existing Heretic models against matched SFT/DPO/adapter/unedited controls, testing changes within each recipe's rank in active edited weights. Predeclare noise/pass rules. Add merge-source search among candidates by Nov 21, including absent-source trials and hard relatives. Test marks on D8 real cases; do not substitute known-source fits for source search. | B | Heretic pass/fail and control flags; lab edit/source table; merge-search errors, useful answers and abstentions; same tests on documented real cases. Adapters support family/base, not automatic sibling-source identity. | Heretic Nov 12; merge search Nov 21; method check Dec 12 | open next tests: Heretic and merge search authorized by check-in; previous4 projection cases and12 quantizer cases remain limited lab evidence |
| **K6** | **Find real same-architecture independent models.** These are the hard negatives a compliance check meets in practice: same config, trained from scratch by someone else. | A (moved from C by card D3) | At least 30 pinned models with partial-read fingerprints; same-family false-positive rate against them | Nov 21 | tested: 30 publisher-supported scratch-trained BERT controls, 0/435 false matches; narrow diagnostic only, seed independence/recall/decoder transfer unresolved |
| **K7** | **Documented real-model set (D8).** First batch: ≥5 vendor quantized/FP8 releases with originals in readable formats,10–20 small-base merges with recipe files, official OLMo/Tulu stages and2–4 directional ablations with public tool/base/settings. Published construction counts; missing private records do not block. Declared-only cases are separate. Deliver cases to A/B as available. | C selects/records; A measures; B tests; C compares | Pinned model/source records, documented-versus-declared classification, readable-format coverage; real edit/source and applicable-method results with caveats. | first batch Nov 14; complete Dec 5 | open under D8: reclassify/admit from published construction; previous private-record rejection is superseded, old measurements preserved |
| **K8** | **Same-architecture scaling (D9).** Count matches for Qwen2.5-0.5B, Qwen3-0.6B, SmolLM2-360M and OLMo-2-1B; select the two largest readable pools, capped near 2,000 each. Query lab plus D8 cases at 10/100/1,000/all with fixed subsamples. Record false-family/source answers, actual comparability, exclusions and cost; measured-versus-declared counts remain descriptive. | A prepares L1; overseer launches; B reviews analysis | Executable pinned owner command, new scope cost/space and stop/resume checks; post-run scaling curves at largest reached pools with label/unknown coverage. | L1 Nov 12; analysis Dec 10 | open redesign authorized; no new pool collected yet. Old mixed5k roster/space gate are superseded, not prerequisites |
| **K9** | **Model order and witness setup.** C audits the8 SFT failures and exact witness ancestry/scoring by Nov 11; use a valid separate sibling branch and predeclare changed tests. Stored old triples already use sibling SFTs, so do not assume a DPO-descendant error. B keeps metadata/spectral work and the broader order evaluation. | C witness audit/retest; B broader order | Verified setup, predeclared pass rule and corrected result or documented limitation/redesign; order results remain separate from source retrieval. | witness check Nov 11; broader results Dec 5 | open approved check; old SFT0/8 and DPO8/8 remain historical until new evidence |
| **K10** | **Verify scores by weight hash, not only by commit.** A leaderboard score applies to our pinned revision when the weight files' hashes match at both commits. New commits rarely change weights. | A | Verified score joins re-counted (was 24/150) | Nov 21 | tested: 144/150 weight-hash joins verified; six lack evaluation revision |
| **K11** | **Heritability: parked.** Card D3: no paper target, and no workshop papers. If it is ever resumed, the rule fix stands: the benchmark's fixed question count is the sampling size, and the leaderboard commit is the identity for score-only analysis. | — | — | — | dropped (parked) |
| **K12** | **P1 pre-registration.** One file lists the headline evaluations, metrics, splits, thresholds and baselines, frozen before the final runs. **Two separate sections: approved verified conclusions, and open questions with prospective test rules (D6).** Open questions are not established claims; later conclusions require explicit overseer approval after verification. It replaces the many small freezes. | B, with the steward | File frozen and its hash recorded in `RESULTS.md` | Dec 4 | open: D6 approved; prepare separate approved-conclusion and open-question sections, with tests fixed before final data inspection |
| **K14** | **Disguise check, in hours** (card D4). First check whether the 8 existing disguised models failed their behavior check only on tolerances too strict for floating-point rounding. Then test whether our fingerprint still matches all 8, and one exact neuron reordering inside one MLP layer. | B | Rows in `RESULTS.md`: the tolerance finding, plus survives or breaks for each case | Nov 21 | tested (partial): stored evidence and tensor permutation; behavior-noise and network comparison open |
| **K15** | **Plan zoo v1:** a `check` command, GGUF support (dequantized for lineage), a duplicates report, a diff gallery, pip packaging. It replaces the old V1 item. | A | Plan written and reviewed by the steward, with a decision card for the release | plan Jan 15; release Feb–Mar | open |
| **K13** | **Clean up the process** (`README.md`, "What changes"). Freeze the old STATUS and CHECKLIST. Keep track notes at 30 lines or fewer. Keep the results ledger. Stop credit and roster accounting. | All | First steward run finds no new writes to the frozen files | Nov 10 | tested: lead notes within cap; frozen handoff files match tracked versions Nov 8; ongoing rule |

## 2. Milestones and checkpoints

| Date | Checkpoint | Card |
| --- | --- | --- |
| **Nov 11–14** | Nov 11 witness setup check; Nov 12 Heretic results and same-architecture L1; Nov 14 first documented real cases and shared evidence review. | L1; scientific cards only for actual unresolved decisions |
| **Nov 21** | First baselines (K2); edge typing (K5); real independents (K6); score joins (K10); same-architecture launch preparation checked (K8, due Nov 12) | L1 (same-architecture pools) |
| **Nov 25** | **Solidify gate:** each claim in `CLAIMS.md` is solidified only if verified, reworded, dropped or kept open for missing evidence, using the cheap tests (K3–K6, K14), the literature check and the skeptic (D6). Until then `DIRECTION.md` is the working direction, not a commitment. | G-card (solidify) |
| **Dec 4** | **Revised G0.** Cost gate on the pilot; same-family false positives against real independents; edge typing; merges; first baselines; pre-registration frozen | G1 (revised G0) |
| **Dec 10** | Two same-architecture pools analyzed at largest sizes reached (K8); documented real cases (K7, Dec 5); all four baselines (Dec 5) | — |
| **Dec 12** | **Method check (claim C2):** typed-edge direction coverage ≥50% at ≥95% precision, and parent false matches ≤0.05 per query on the edits where the method answers (abstentions reported), on the lab; the exact-mark tests also hold on the real Hub examples from K7 | G-card |
| **Dec 18** | **P1 go/no-go:** ICML (around Jan 22 abstract, Jan 28 paper) or COLM (around late March). Criteria in `DIRECTION.md` section 7, with the D9 scaling curves at largest reached same-architecture pools and documented query coverage. The old mixed5k-count condition is superseded; report the existing ≥200 labeled-query target explicitly and flag any shortfall for the gate decision. | G2 (P1 venue) |
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
| H1 old5k pool | K8, redesigned as D9 same-architecture pools |
| H2 byte curves | part of K8 |
| P-1218 pivot check | Dec 18 gate (G2 card) |
| F1–F4 freeze and submit | unchanged, if Dec 18 says ICML |
| F5 heritability workshop | dropped (card D3: no workshop papers) |
| V1–V8 after ICML | Feb–Mar rows above; each release is a decision card |
| Weekly items | the steward's runs |
