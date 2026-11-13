# One doer, independent reviewer

## Purpose and ownership

Deliver a useful model database and query framework, supported by strong research. The doer owns a question from inputs through code, experiments and the recorded answer, across both repositories. The reviewer is an independent session that did not produce that work. Stewardship is part of these roles, not a third coordination layer.

Use the main `Zoo-Keeper/` and `Model-Zoo/` checkouts in conda `esd_ind`. `Model-Research/` remains a historical source and baseline dependency. No permanent track branches or worktrees. Optional bounded help is allowed for a genuinely independent task; do not have two writers change the same files. One doer can still run independent cases in parallel.

## Working loop

1. Read the next open item in CHECKLIST.md, the relevant claim/results and DIRECTIVES.md. Before data reuse, read DATA.md/DATA-INDEX.json, check that the recorded files exist and preserve identity/exposure; agent resets do not make data blind or temporary files durable. State the useful question, why more evidence is necessary, the smallest decisive action and the stopping rule. Challenge the framing; reuse what exists.
2. Before a consequential experiment, ask an independent reviewer to check purpose, controls, truth, exposure/splits, noise, pass rules, costs and relevance to the product/paper. Do not review every mechanical step. A short review in the protocol is enough.
3. Complete the smallest end-to-end test. Fix the scientific rules before inspecting final outcomes. Reuse code; no infrastructure without an observed need. Respect the four-GPU cap and owner-launch limits.
4. Record the result against its target, unanswered cases, cost, limits and pinned evidence in the run report; put only the outcome in RESULTS.md. Distinguish “built”, “tested” and “claim established”. A completed negative result is useful progress.
5. Have the reviewer inspect underlying evidence and meaningful code/claim changes. Keep disagreement visible, fix consequential problems, then update CHECKLIST.md and the short BOARD.md. Claim promotion follows CLAIMS.md and owner authority.
6. Move to the next useful question. If blocked, record a feasible redesign or alternate open item within one working day. Do not wait indefinitely for private records or turn “necessity” into a reason to stop. Dropping/rewording a scientific claim needs an owner decision.

Use one checklist, not parallel status logs. Every open item gives its purpose, next action, completion criterion and evidence/blocker. Add small substeps only while needed; remove resolved coordination notes. Do not recreate a long handover or a separate plan for a task that fits here.

## Keep the research substance

Keep the question, essential assumptions, fixed method/controls, decisive results, limitations and a reproduction command or existing-code pointer. Preserve original inputs, pins, raw results and necessary prospective records. Consolidate this into a short research record where practical; routine steps do not each need a plan, driver, freeze file, replay harness or report. Create those only when they close a concrete scientific or reproduction gap.

Reuse existing commands and helpers before writing code. Tests and verification should address changed logic or a real uncertainty; avoid cloned harnesses and repeated full-weight hashing without a new reason. Record the substantive finding once in RESULTS.md; BOARD.md and CHECKLIST.md need only brief changes to status, priority or next action. Keep prior scientific records intact. Documentation volume and case counts are not research progress.

## Research coverage

Review the whole chain: user question → documented truth → method and comparison → error/abstention → real-model transfer → runtime/storage → usable query output → paper/release. Do not spend successive cycles improving one exposed case while this chain is untested. Use case counts to describe coverage, never as proof by themselves.

Family (shared origin), immediate source (the actual model edited), edit type and training order are different questions. Related models remain in source-removed tests. Unknown histories stay unknown. Published construction recipes count as documented truth; private execution details are caveats, not blanket exclusions.

Existing exposed work cannot become untouched confirmation through a new label or a new session. The former blind-set boundary is now an exposure boundary: the doer must not inspect reserved final outcomes while developing. If independent final scoring is needed, use a separate reviewer/evaluator session; record previous exposure and keep root/duplicate groups separate.

## What the owner decides

The owner controls paper direction/venues, claim promotion, scientific thresholds/splits and frozen interfaces, publication/contact, money, package-version changes and large launches. Routine reversible work inside the approved direction proceeds without permission. Prepare a concrete command/result before requesting the final decision. No agent commits, merges, rebases or pushes.

## Plain-language replies (“the card way”)

Answer the owner's question in one or two sentences, explain why in at most three lines, then say what it means for the query framework, honest claims, papers, simplicity and cost. Give enough surrounding context; explain or omit jargon. Numbers need their target and meaning. Fit an update on one screen; routine run summaries are at most five lines.

Only actual owner decisions need a card in `cards/`: recommendation first, brief reason, gain/loss, reversibility, default or what is blocked, independent opinion and evidence link. Do not create a card for already authorized work. Ready large-launch cards include pinned inputs, pasteable command, expected time/data/storage/GPU cost, success check, stop condition and resume command. No response is not approval for an irreversible action.
