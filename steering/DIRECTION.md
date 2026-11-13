# Direction: a measured model database and query framework

## End goal

Build a framework people can use to query a large, measured database of open models. Someone should be able to find candidates, compare them, understand their origins and changes, and make a better model choice. The database and the query framework are joint deliverables. Collecting models or writing papers alone is not enough.

The owner's original ambition is **50k–100k models or more**, with practically useful research that attracts attention at top venues and helps frontier companies and other users. Keep that ambition. Expand when query value, coverage, measurement quality and maintainable costs justify it; do not confuse model count with progress. The earlier 10–20k zoo-paper scope is a near-term planning envelope, not a permanent ceiling. No 100k collection run is authorized merely by restating the ambition.

## What research contributes

| User question | Research contribution to test | Present position |
| --- | --- | --- |
| “Find models that meet my needs.” | Filters, measured scores and useful retrieval; later validate clustering or task-based recommendations against simple alternatives. | Search and information commands exist. End-to-end model-choice usefulness is still unproven. |
| “What is this model based on?” | Family identification with explicit uncertainty, difficult same-shape controls and realistic candidate pools (C1/C3). | Useful small-catalog results; broad reliability and scaling remain open. |
| “What changed, and is the declared origin consistent?” | Measured differences, duplicate detection and informative edit/source evidence (C2/C5). | Strong narrow quantizer and known-source merge results; real-model source discrimination remains open. |
| “Which model should I choose?” | Comparable measured evaluations and, later, a quick test on the user's examples. Predict scores from weights only if this beats strong no-weight alternatives. | Recommendation is a product goal, not an established weight-prediction result. Selection research follows evidence readiness. |
| “Can others test these answers?” | A reproducible lab benchmark, documented real-model cases and applicable published comparisons (C4). | 166 cohort entries, including 112 formal constructed children; evidence protected, comparison coverage incomplete. |

Prefer research that improves an actual query answer or establishes an important limit. Simple, mathematical, statistical or learned approaches are all allowed when the useful question warrants them. Do not add a technique just because it is sophisticated. Scientific necessity applies to everything said and done.

## Current research strategy

1. Finish decisive tests using existing lab assets: the four realistic directional-ablation edits (Heretic) with normal-training controls, and merge-source search among candidates. Known-source coefficient fits are a separate successful result.
2. Establish transfer on documented real models and compare applicable methods on the same cases. A public vendor method, merge recipe, official training stage or public tool/base/settings is documented truth. Missing private execution details are caveats; a missing expected mark is a reported result. Declared-only relations remain descriptive.
3. Test search at scale with two same-architecture candidate pools, selected from metadata around the actual pinned lab bases, capped near 2,000 each for this experiment. Measure 10/100/1,000/all with fixed subsamples, actual sizes, unknowns and costs. This replaces the uninformative mixed 5k roster, not the long-term database goal.
4. Demonstrate useful query journeys, including a fresh model, honest “not sure” answers, measured score filters, comparisons and total cost. Evaluate model-choice usefulness before adding prediction or a larger recommendation system.
5. Grow the database and prepare papers/releases when these results justify them. Revisit breadth, methods, query usefulness and cost together, rather than optimizing one small exposed panel repeatedly.

## Next milestones toward the larger goal

The three-model/two-task probe closed as an engineering failure; the shared coverage map and matched real-model comparison batch are now complete for their fixed subsets. Move to useful measured query journeys and remaining claim/readiness gaps rather than another stream of isolated comparisons. Use the existing checklist and reusable code; keep one research record per question.

| Milestone | Concrete deliverable and why it matters | Stop or decision |
| --- | --- | --- |
| One common benchmark and claim map | Reconcile existing lab/real cases into one table keyed by weight identity, root, operation, exposure, truth strength, available information, candidate/removal controls and applicable method. Reuse protected inputs and four-method comparisons; include failures, unsupported cases and abstentions. This reveals which missing experiment can actually change a claim. | Resolve identity/contract mismatches before more collection. Do not count repeated pools or aliases as new independent queries. |
| One decisive method campaign | Test only the remaining informative questions together: relatives versus source, source absence, independent same-design controls and a query-applicable directional-edit/noise rule if one can be justified. Keep released-factor checks separate from weight-only methods. Preserve existing precision/coverage targets and root/exposure boundaries. | Stop uninformative rules without tolerance/rank/layer sweeps. If the method headline remains unsupported, prepare an evidence-backed method-versus-benchmark framing recommendation for the owner; do not silently drop claims. |
| Useful framework over the measured catalog | Connect pinned measurements, exact evaluation contexts and explicit reported/verified/unknown status to find→compare→inspect→choose. Validate a small fixed set of real user questions across the current catalog; distinguish storage from deployment fit. Use the probe result only within its actual task/method scope. | No new predictor, hosted service or schema expansion without an observed need. A frozen-interface change needs a concrete versioned proposal and owner approval. |
| Justified search at scale | Complete scientific labels/scoring/controls before the owner launches the prepared two same-architecture pools. Retain the at-most60 versus≥200 query gap and independent-control gap. Refresh actual disk/throughput/budget estimates; the old771.47GB/32–45h estimate is not a ready launch. | Large/paid runs remain owner-launched; no splitting to evade≤100weight-identities/≤2h agent limits. Use results to support or delimit scaling claims, not a round model count. |
| Staged database growth and research delivery | Grow pinned metadata/inventories from703 toward10–20k and50–100k+, with duplicate links, explicit unknowns and probability-sampled quality/coverage checks. Select expensive weight measurements by query value. Prepare the paper/usable release from the common evidence table without waiting for100k. | Owner approves collection/release/framing decisions when concrete commands, measured cost and reviewable outputs are ready. Preserve the larger ambition; population claims need an actual sampling design. |

Speed comes from batchable questions, one evaluation contract, reuse and clear stopping rules. Metadata growth can proceed alongside scientific work when it answers a stated coverage question; it does not substitute for trustworthy query answers.

## Claims and limits

C1: family at scale; C2: informative edit evidence and source discrimination; C3: decisions tested against close relatives and missing sources; C4: benchmark; C5: measured-versus-declared ecosystem; C6: later score/selection hypotheses. Detailed tests and status are in CLAIMS.md. C1–C5 remain checked, not broadly verified or approved as solid conclusions.

An edit mark may establish compatibility with an operation under stated assumptions. A unique immediate source requires distinguishing relatives in the stated pool; inconclusive evidence stays unresolved (D7). A directional weight edit does not establish a behavioral loss of refusal. Ordinary fine-tunes do not have a general exact-parent/order guarantee. Published method rankings and calibrated parent answers are not interchangeable.

Retain the lab successes, failures and limits in RESULTS.md. Do not require a single method to answer every query. Combine signals only under a justified, prospectively tested rule; do not claim recovered accuracy from a metadata join.

## Papers, releases and readiness

- **P1, “What model weights reveal about lineage” (working title):** useful source/family/edit answers, real-model transfer, fair comparisons and honest limits. A method framing requires the existing target: typed-edit direction coverage ≥50% at ≥95% precision, immediate-parent false matches ≤0.05 per answered query on the relevant edits, abstentions reported, plus documented real-model tests. These thresholds do not redefine each other.
- **P1 venue:** ICML 2027 if evidence and comparisons are ready; otherwise COLM. Prior schedule estimates were January and late March; verify official deadlines before any submission decision. No arbitrary internal dated gate is required. An empirical-study/benchmark framing is legitimate if the method headline fails. The owner decides framing and venue.
- **Scaling claim:** report largest reached same-architecture pools and explicitly assess the existing ≥200 labeled-query target. A coverage shortfall is a shortfall, not a reason to relabel unknowns. Remove a scale claim only by an owner decision.
- **Zoo v1:** installable library/CLI with useful `check`, GGUF lineage, duplicates and comparison examples; validate fresh-query cost, supported scope and user usefulness. Earlier public target February–March remains an aim, conditional on readiness and owner release approval.
- **Zoo/ecosystem paper:** database plus query framework, useful measurements and measured-versus-declared findings; earlier target around May at 10–20k models, with expansion supported by throughput and quality. A sound sampling design is required for prevalence claims; selected same-shape scaling pools are not a population census.
- **Selection paper:** later, once end-to-end query/model-choice evaluation supports a distinctive contribution. Clustering and performance prediction are possible tools to test, not assumed solutions. Heritability remains parked. No workshop papers.

Approved conclusions and open questions stay separate (D6). Every final evaluation has prospective rules, held-out/root/duplicate separation and an independent review. Only the owner promotes a verified conclusion into the paper's approved claim list.

## Do not revive by habit

Tree recovery; exact parent/order guarantees for ordinary fine-tunes; 10k-candidate guarantees; byte-efficiency novelty; robustness to adversarial disguise; spectra inferred from partial reads; a 50–100k census promised by May; paid/black-box evaluation for the current P1; unvalidated weight-based score prediction; workshop or heritability papers. A broader database and query framework remain the goal; these cuts constrain unsupported claims and premature work, not ambition.

## Assets and main risks

Model-Zoo holds collection/evidence/database building; Zoo-Keeper holds queries, the lab, methods and studies. Model-Research remains a historical source and comparison dependency. Reuse these assets before new code. The lab archive is already verified, not awaiting storage setup.

The main risks are real-model marks failing, source ambiguity among relatives, limited independently trained controls, inflated counts from dependent cases, narrow format coverage, unfair comparisons and database growth outrunning useful queries. Report these plainly and choose a decisive test or a useful redesign.
