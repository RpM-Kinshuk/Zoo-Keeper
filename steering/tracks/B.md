# Track B: lab methods and P1 evaluation
Brief of 2026-11-08 (current focus + GPU update) read; D6 approved.

**Now:** existing-fingerprint [evaluation](../../../runs/track-b-existing-artifacts-20261108/REPORT.md)
complete:216 catalog keys,134 paired source-present/removed queries; exposed analysis.
Owner scientific-necessity rule in handoff/DIRECTIVES.md; no collection justified.
Owner checkout `worktrees/Zoo-Keeper-track-b`; nothing running; no git integration.
CPU one thread,13.53s; no weights/network/GPU. A=0/B≤2/C≤2, cap4; no reservation.

**Results since the last steward run:**
- Source present:119/134 useful family answers,15 abstentions,0 known-wrong.
- Source removed:112/134 useful family answers,21 abstentions,1 unavailable.
  Family matches do not identify source; generic parent rule remains unconfigured.
- All15 present-source known-positive misses are quantizations; threshold unchanged.
  Certified independent controls:0/180 hits;523 finite unknown pairs stay unknown.
- K5 [joined table](../../../runs/track-b-k5-joined-20261108-v2/REPORT.md):90/112 scoped
  measurements;24 adapter sets retain nonparents,8 sampled ranks resolve;
  refusal writer ranks unresolved; TIES/DARE shares overlap controls.
- Quantization exact checks and18 known-source coefficients remain closed.
- C baselines included; complete real-model labels/controls still missing.

**Next:** use existing exact marks/noise controls to propose discriminating source
rules; do not expand reads unless they can resolve a named failure. Nov21 retained.
K4/K14 bounded; K12 separates approved conclusions and open questions (D6).

**Blocked or needs:** new scientific rules require the [draft](../cards/draft-B-k5-detector-rules.md)
and untouched confirmation cases; no new cutoff, split, truth or claim applied.
Class1: existing scorer/grouping reused; inputs/outcomes independently verified.
Class2: [A handoff](B-to-A-existing-evaluation.md) recommends no collection or storage action.
