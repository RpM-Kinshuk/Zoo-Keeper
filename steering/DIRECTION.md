# Direction: reassessment and correction (2026-11-07)

**Status:** the overseer approved this as the working direction on 2026-11-07
(card D2). Claims remain hypotheses until verified and explicitly approved
by the overseer in `CLAIMS.md`, at Nov 25 or a later review. Under D6 (approved
Nov 8), the Dec 4 plan separates approved conclusions from unanswered questions
and the tests that will settle them; dates and evidence standards are unchanged. A literature check on 2026-11-07 already
corrected two points; they are marked "(corrected)".

This document re-checks every claim, promise and plan against the real world,
the research frontier, and the evidence so far. It then sets the corrected
course. It supersedes the claims and paper plan in
`docs/research/2026-10-27-merged-research-plan.md` (sections 1.1, 3.3 and 11)
where they conflict.

Inputs:
- the agents' recorded results (`RESULTS.md`);
- two independent reviews written for this reassessment: a skeptical
  area-chair view, and a real-world practitioner view;
- a web search for new work.

The merged plan remains the reference for methods detail.

## 1. In one minute

**What we have.** The infrastructure is strong:
- a fingerprint extractor for every model type;
- a queryable zoo of 703 models;
- a 320-model random sample of the Hub;
- a 166-model lab cohort: 112 formal constructed children plus bases and reference controls. Construction records apply to the built children; control provenance has separate limits.

**Current evidence, updated Nov 8:**
- Tools and the protected lab are a working internal foundation; broad
  real-world reliability and the paper's strongest claims remain unverified.
- Quantization replay/source-absence tests and 18 known-source merge fits pass
  their lab targets. They do not establish all-edit source or order recovery.
- The joined table exposes broader limits: adapter clues also match relatives,
  sampled refusal tests cannot resolve rank, and merge-entry shares overlap
  controls. Most immediate-source decisions remain untested. D7 approved
  separating informative edit evidence from unique-source proof (Nov9).
- Four priority comparison methods now have scoped results, with two recently
  matched on the quantization cases. Whole-benchmark and real-model coverage
  remain unfinished; lack of independent negatives restricts error claims.
- Additional collection of the prepared 5k is scientifically deferred; storage
  is parked. Evaluate saved source/relative/control artifacts before defining
  any minimum decision-changing data addition. Preparation is not a reason
  to collect, and missing documented exact matches do not imply absent family.
- Family detection/control results remain narrow; score-link improvements do
  not revive prediction or heritability work.

**Execution emphasis:** every statement/task needs a strong scientific reason
and a necessary outcome (DIRECTIVES.md D1.6). Challenge the framing, reuse
existing evidence, consider simpler or different approaches, and stop when a
useful uncertainty is resolved. Do not turn inconclusive tests into passes or
launch another collection to fill a round count. Existing dates and scientific
rules remain; PLAN.md and briefs hold the necessary next work.

**The correction.** Claim what weights can actually show, and test it
properly:
1. **Family, at scale.** Which base family a model descends from, with error
   rates measured against the hardest real look-alikes and against every
   published method. This is the question compliance and license checks ask.
2. **Edits that leave a mark.** Quantization, refusal removal, merges and
   low-rank adapters may leave informative edit evidence. Under approved D7,
   this supports source compatibility only when informative; a unique immediate
   source requires tests distinguishing relatives within the stated candidate
   pool, otherwise unresolved. Order is a separate hypothesis. Ordinary
   fine-tunes leave no exact mark. Their order may still come from a third
   family member (the "witness" method published as Witness Overlap) or from
   a spectral cue. We test both, and fall back to the Hub's upload times
   (corrected). This is new science and a practical "what was done to this
   model?" tool.
3. **The lab as a benchmark.** A 166-model cohort with 112 constructed
   children across seven edit groups, plus bases/reference controls. Earlier
   work includes a 455-model collection labeled from declared
   bases. We do not claim the largest collection; the value of our exact
   construction records across edit types still needs direct comparison (D5).
4. **The ecosystem measured.** Duplicates, quantized copies, tensor sharing,
   and whether declared bases are right, on 10–20k models (down from 50–100k).

**What we stop:**
- tree recovery;
- exact-parent and order claims for ordinary fine-tunes, unless the witness
  or spectral tests support them;
- per-query guarantees at 10,000 candidates;
- "reads fewer bytes" as a selling point;
- robustness to deliberate disguise;
- spectral science from partial reads;
- the 50–100k census by May;
- weight-based score predictions in `recommend`.

**Papers:**
- **P1** is decided on Dec 18: ICML, only if the edge tests and baselines are
  in, otherwise COLM in late March.
- **Zoo v1** goes public Feb–Mar, with a `check` command and GGUF support.
- **The zoo paper** is due in May, at 10–20k models.

## 2. Original reassessment: promises versus evidence (Nov 7)

This table records the original diagnosis; use section 1 and RESULTS.md for
current evidence. Do not re-run a resolved repair because its old failure is
still recorded here.

| What the plan promised (P1 and around it) | Evidence at reassessment | Verdict |
| --- | --- | --- |
| Published methods give many false parents as the pool grows | No baselines run; pool about 700 | **untested.** It is also close to a statistical truism, so on its own it is weak. |
| Our method keeps false parents at or below 0.05 per query at 10^4 | 8.28 per query on a small pool | **undermined** |
| Same family is detected | AUROC 1.00, but 2 held-out families | **supported, thin, not new** |
| Immediate parent; ancestor | Parent fails; ancestor not tried | **undermined; untested** |
| Direction (which came first) | 0% coverage | **untested** |
| Edge typing (LoRA, full fine-tune, quantization, merge, ablation) | Code exists, never run on the lab | **untested** |
| Merge recipe recovery | 30/30 refused because of how the fit was set up | **untested (broken test)** |
| Per-query error control | 15 null pairs with 0 false alarms; that still allows a false-alarm rate up to about 20%. And it is the wrong comparison: a parent is confused with its relatives, not with unrelated models. | **unsupported** |
| Reads a few percent of the bytes | 27% on a tiny model; modelDNA reads about 1–2% | **undermined, and not our edge** |
| Robust to disguise | 0/8 disguises built; a row-block sketch is not robust to neuron reordering | **untested; a real weak spot** |
| Model evolution is "not tree-like" | 82–96% of quartets fail, at a tolerance with no noise model | **undermined as measured.** The pivot rests on uninformative evidence. |
| One pipeline for every model type | Extraction supports it; science tested on small LLMs only | **untested as a claim** |
| Heritability per skill and edge type | 0 estimates under the strict rules | **blocked** |
| Weights as a priced prior for scores | About 3–39 benchmark questions' worth; within a family, worse than no weights | **sobering and useful** |

## 3. The real-world check

Who would use this, ranked by value:
1. **Compliance and legal teams** adopting open models. "Is this renamed
   fine-tune or GGUF derived from Llama, Gemma, Qwen or DeepSeek, and which
   license applies?" High stakes, asked monthly. *Partly supported:* family
   detection works; it is untested on real look-alikes and on GGUF.
2. **Hubs and model publishers.** "Is this upload a re-upload, re-shard or
   quantized copy? Is its declared base right? What derives from our model?"
   Asked continuously. *Mostly supported* by file and block hashes, which are
   cheap.
3. **Safety teams.** "Was refusal removed?" *Untested,* but probably the
   easiest win. Refusal removal leaves an exact mark, and the lab has eight
   examples. The baseline to beat is just "the name contains 'abliterated'".
4. **Researchers and fine-tuners.** "What did this fine-tune change?" The most
   shareable output (a diff gallery). *Works, but uncalibrated.*
5. **Model choice.** The largest audience, and the weakest fit for weights.
   Leaderboards and model-profiling services already answer it, and our own
   data says weights add little. Measure, do not predict.

**Almost nobody needs the exact parent from weights alone.** In practice the
order comes from upload times and declared bases. Users need "family X, these
closest candidates, declared base consistent or not".

**Missing for real use:**
- GGUF (16% of top repositories, and half of the top decision models);
- Hub metadata as evidence;
- a published cost per query for a model not yet in the zoo;
- standard integration points (a pip command, a Space, a GitHub Action).

## 4. The frontier check

**Related work, checked on 2026-11-07.** The full table and its effect on
each claim are in `CLAIMS.md`. In short:
- **Verified:**
  - **modelDNA** with **LineageBench** (15 suspect models) and the **modelDNA
    Atlas** (68 Hub models);
  - **TrFP** (activation fingerprints; runs the model; robust to disguise);
  - **provenance sets (MPS)**: source-set coverage and set-size curves for
    5–100 candidates; 455 models labeled from declared bases (D5).
- **Corrections:**
  - **Witness Overlap** is arXiv only (NeurIPS acceptance not confirmed). It
    orients ordinary fine-tunes using a witness model, so order is not
    hopeless from weights (corrected).
  - Two works were missing from our plan:
    - **"Training Leaves Traces"** (2608.14929): weight-only relatedness, but
      symmetric, with no type or direction. It is now a C1 baseline.
    - **Cisco's `model-provenance-kit`:** a practical provenance tool with
      edit types, and a real-world competitor for `check`.
- **Not found:** any weight-only detector for refusal removal.
- **Documentary follow-up complete (Nov 8):** C checked AttnPrint/DistillTrace,
  SCOUT, BULLPEN and the GGUF reproducibility paper; see its K2 primary-source
  note and CLAIMS.md. This is reading, not new method-performance evidence.

**Where we can lead:**
- **Construction records across many kinds of edit:** 112 formal constructed
  children within the 166-model cohort. Earlier work includes 455 models with declared bases; our possible
  distinction is exact construction records across edit types, not size (D5).
- **Edge-type identifiability:** which edits make the parent and order
  provable from weights, and which provably do not.
- **Calibration against relatives.**
- **The measured Hub, with a random sample** for honest prevalence estimates.

Whether exact construction records across edit types distinguish this lab
from earlier benchmarks remains unverified. Track C must compare them before
we claim novelty (D5, approved Nov 8).

**Where we do not lead:** reading fewer bytes, same-family detection on its
own, and robustness to disguise.

## 5. Over- and under-estimates

**Oversold:**
- "Typed, calibrated lineage scales";
- per-query error control;
- 10^4+ pools by January;
- direction from weights for every kind of edit (we test it only for edits
  that leave an exact mark);
- byte efficiency;
- one validated pipeline across modalities;
- tree-likeness as a finding;
- the 50–100k census by May;
- `recommend` with weight-based predictions.

**Undersold:**
- **The lab** (the strongest asset).
- **The honest split** between "family is easy, parent is hard", which
  reframes the field.
- **Within a family, weights do not tell which sibling is better.** A crisp,
  sobering result.
- **The random Hub sample,** which makes honest prevalence estimates possible.
- **Refusal-removal detection,** a practical win.
- **Score verification:** weight identity now supports 144/150 score links,
  compared with 24/150 matching revision names alone; six remain unknown.
  This is not a verification of every evaluation setting.
- **The tool itself:** an interpretable model-origin check using the existing
  queries. Whole-workflow latency and real-world accuracy still need measurement.

## 6. The corrected course

### Claims we will test, and make only if they hold

| ID | Claim | Real-world use | Tested by |
| --- | --- | --- | --- |
| **C1** | From a small part of the weights, a model's base family is identified reliably. False-positive rates are measured against independently trained models of the same architecture (the hardest real look-alikes), at the largest pool reached (target 5,000), alongside every priority published method. | Compliance, licensing, de-duplication | K2, K6, K8 |
| **C2** | Some edits make the parent, and sometimes the order, provable from weights. Each has its own test: <br>• quantized children sit on a value grid (parent and order); <br>• refusal removal leaves one removed direction shared across layers (parent and order); <br>• a merge is a mix of its parents and base (parents and order); <br>• adapters leave a low-rank difference (parent, but not order). <br>For ordinary fine-tunes, weights give the family and the closest candidates. Order is tested with a witness model (Witness Overlap) and a spectral cue; otherwise the method abstains and Hub metadata supplies the order (corrected). | "What was done to this model?"; refusal-removal checks; quantized-copy detection; merge-config verification | K3, K5, K9 (pass rules in `CLAIMS.md`) |
| **C3** | Parent decisions are calibrated against close relatives (siblings, grandparents, quantized variants), with coverage reported at the pool size reached. There are no guarantees beyond what the calibration set supports. | Trustworthy "consistent or not" verdicts | K5, K12 |
| **C4** | The lab, plus a verified Hub tier at 7B and above, is released as a lineage benchmark with a scorer, with all priority methods measured on it. | Shared yardstick for the field | K2, K7 |
| **C5** | The Hub measured against what it declares: duplicates, quantized copies, tensor sharing, declared-base consistency and revision drift, with prevalence estimated from the random sample. | Hubs, publishers, researchers | zoo v1 and P3 |

C2 becomes P1's headline only if it holds on real Hub examples as well as on
the lab (section 7).
| **C6** (later) | Weights are worth only a few to a few dozen benchmark questions as a score predictor, and do not separate siblings. So measure, do not predict. | Model choice | selection paper |

### Stop

- Tree recovery. "How tree-like" is reported only if the K4 retest makes it
  meaningful.
- Parent and order claims for ordinary fine-tunes from weights alone, unless
  the K9 witness or spectral tests support them.
- Per-query guarantees at 10^4 candidates.
- Byte-efficiency claims.
- Robustness to disguise. We state that we assume no adversary, and run one
  exact neuron-reordering test to show the limit honestly.
- Spectral numbers from partial reads.
- Validating every modality before LLM lineage is solid. Extraction support
  stays, but makes no claim.
- The 50–100k census by May. Instead: 10–20k plus checks on request.
- The black-box test for January.
- Weight-based score prediction in `recommend`, unless it beats a no-weights
  baseline, including question-profiling services.

### Start

The K-actions in `PLAN.md`, plus:
- the exact edge tests (grid, removed direction, merge mix, rank) inside K5;
- the relatives comparison set inside K5 and K12;
- one exact neuron-reordering test (part of card D4);
- reading and verifying the new papers inside K2.

### Papers and releases

| What | Target | Odds as judged now |
| --- | --- | --- |
| **P1:** "What model weights reveal about lineage" (working title). Family at scale, exact-mark edits, honest limits, and the benchmark with baselines. | ICML (Jan 22/28) if Dec 18 says go; otherwise COLM (late March) | ICML plausible if the edge tests and baselines land by about Dec 12; COLM likely |
| **Zoo v1:** pip install, a `check` command, GGUF, a duplicates report, a diff gallery (planned in K15) | Public Feb–Mar. The tool can go public before P1. The lab data goes public only after the P1 submission: January if ICML, late March if COLM. | likely |
| **Measured-vs-declared note** (tensor sharing, re-uploads, revision drift) | arXiv, Feb–Mar | likely |
| **P3:** the zoo paper, the lab plus 10–20k measured models | NeurIPS E&D, around May | plausible |
| ~~Heritability workshop paper~~ | Dropped (card D3: no workshop papers) | — |
| **Selection paper** | Decided after zoo v1 | too early to judge |

## 7. Kill criteria and checkpoints

Each line is decided on its date with a gate card. Results that miss a target
are reported as misses.

| Date | Check | If it fails |
| --- | --- | --- |
| Nov 14 → Dec 5 | Merge fits on deltas: coefficient error ≤0.05 on most linear and SLERP merges (no more than 4 misses) | Drop recipe recovery. Keep only "the published config is consistent with the weights". |
| Nov 14 → Dec 5 | Tree test with a tolerance set by repeat-measurement noise, on subtrees without merges | Keep the pivot if violations stay large against the simulated exact tree; reopen it if not |
| Dec 12 | Method: typed-edge direction coverage ≥50% at ≥95% precision, and immediate-parent false matches ≤0.05 per query on the edits where the method answers (the original target, with abstentions reported), on the lab; the exact-mark tests also hold on real Hub examples (refusal-removed and quantized models with known parents, from K7). Lab-only success does not make the headline: marks may blur on real models, for example refusal removal followed by more fine-tuning. | Drop the method framing for ICML. P1 becomes an empirical study plus benchmark (ICML), or moves to COLM. |
| Nov 21 | Disguise check (K14): does the fingerprint survive the 8 existing disguises and one exact neuron reordering? | It is a limitation either way. Report it. Compliance is where disguise matters most. |
| Dec 18 | At least 2 priority baselines run on the lab and the Hub tier | No ICML; target COLM |
| Dec 18 | Pool of at least 5,000 and at least 200 labeled queries (checked before the January freeze) | Remove "scale" from the title and claims |
| Jan 15 | Final go or no-go on the ICML abstract | COLM |

## 8. Risks

1. **Losing the lab data.** It is in `/tmp` today. Card D1.
2. **No baselines means no paper.** Card D3 and K2.
3. **Overlap with new work:** provenance sets on pool size; the modelDNA
   Atlas; "Training Leaves Traces"; Cisco's provenance kit for the `check`
   product. Track C verifies before we claim.
4. **Results support only the empirical framing by Dec 12.** That is still
   publishable, at COLM or as a benchmark.
5. **Releasing data** under gated licenses, ablated models, naming model
   authors, and the institution's policy: each release is a decision card with
   a license check and prior notice to named parties.
6. **One person cannot keep a 100k census fresh.** The product is an index of
   important bases, plus a `check` on request.
