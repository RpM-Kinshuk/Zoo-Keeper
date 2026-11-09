# Saved-control audit — predeclared 2026-11-08

Purpose: determine whether proposed edit marks can distinguish edits/sources,
and whether unresolved writer measurements used genuinely edited positive blocks.
This audit is necessary before proposing calibration or reading fuller matrices.
Reuse joined results, saved scalar controls and original construction metadata;
no checkpoint matrices, new fingerprints, blind data, network or GPU use.

Three checks, without choosing or changing a classifier threshold:

1. TIES/DARE share feasibility. For pooled and each available anchor, compare
   minimum merge target-entry equality share with maximum SFT/DPO/LoRA control
   share. A higher-share cutoff with all merge positives and zero control flags
   exists ONLY if min(positive) > max(control). Otherwise that one-dimensional
   rule family FAILS the stated all/zero criterion on these exposed cases.
   No cutoff is selected. Native-value shares retain their recorded dtype caveats;
   this says nothing about richer spatial/multitensor rules or unseen data.
2. Inspect the unchanged block_delta gate: status, relative update, identical
   fraction, confidence and reported rank for all saved writers and controls.
   Report resolved rank-one counts and every below-floor/low-confidence case.
   This is a gate audit, not a newly calibrated detector. Existing relative
   update floor3e-3 remains unchanged. Exact equality has zero numerical tolerance.
3. Join each sampled Heretic down_proj layer to its stored per_layer_settings.
   Weight zero is construction-inactive, not an edited positive. Nonzero recorded
   strength makes it construction-active; rounding can still erase the update.
   Count both separately. Verify classic shared-direction construction from actual
   archived metadata, not assumptions from the generic current source runner.
   Hash metadata against K0 release pins. Never relabel the graph or test outputs.

Execution PASS requires complete input hash checks, no missing writer/setting,
and full coverage of the existing8 positive cases and40 nonremoved controls.
Missing data is unavailable, never a negative. Scientific outcomes may FAIL or
remain INCONCLUSIVE even when accounting passes.

After this audit, write one concrete DEVELOPMENT-ONLY proposal: fixed control
selection, exposure/root accounting, statistics, storage/noise allowances,
nonvacuity check, and stopping rule. No approval-to-prepare card, new final
cutoff, root reassignment, or claim promotion. A bounded follow-up must name
the missing scalar/matrix information and why it could change the next verdict.
Otherwise stop. All cases remain exposed; fresh confirmation is a later issue,
not a blocker to descriptive evaluation now. Quantization/K3 remain closed.
