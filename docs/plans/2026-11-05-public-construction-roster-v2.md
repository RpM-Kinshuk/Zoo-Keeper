# Formal public construction roster v2 — approved amendment

Status: OWNER APPROVED and ACTIVE, 2026-11-05. Exact activation: `runs/track-b-pivot-unblock-a-20261105/active-roster-v2/`; complete A request: `A-unblock-request-active.json` at the parent run.

The owner approved the E6 pivot on2026-11-05: report how tree-like the measured
sketch geometry is and focus P1 on supervised edge scoring. That decision does
not change the lab roster. This separate amendment resolves A's exact remaining
input blocker without converting any functional failure into a pass.

## Proposed decision

Replace the eight unavailable original permutation/rotation slots with their
already generated, safely reloaded, measured `-drift-v2` constructions in a new,
versioned166-row roster. Count these as verified public construction examples,
with `functional_equivalence=FAILED` explicitly retained. Original models/edges,
recipes, thresholds, failed results and all old reports remain immutable.

This changes the roster and public-construction counting policy. It does not
claim functional preservation or decide G1/G0. Public constructions are
16/16 under the approved policy. Original Task8 completion remains8/16 (four
rescalings pass; four targeted perturbations are exempt), with all eight failed
original permutation/rotation attempts retained. C's separate
blind contribution remains unknown to B and any combined G1 gate still belongs
to the owner. Do not inspect blind files or infer their count.

| Original unavailable ID | Proposed separately named measured construction |
|---|---|
| olmo-1b-laundering-mlp_permutation | olmo-1b-laundering-mlp_permutation-drift-v2 |
| olmo-1b-laundering-value_rotation | olmo-1b-laundering-value_rotation-drift-v2 |
| q25-05-laundering-mlp_permutation | q25-05-laundering-mlp_permutation-drift-v2 |
| q25-05-laundering-value_rotation | q25-05-laundering-value_rotation-drift-v2 |
| q3-06-laundering-mlp_permutation | q3-06-laundering-mlp_permutation-drift-v2 |
| q3-06-laundering-value_rotation | q3-06-laundering-value_rotation-drift-v2 |
| smol-360-laundering-mlp_permutation | smol-360-laundering-mlp_permutation-drift-v2 |
| smol-360-laundering-value_rotation | smol-360-laundering-value_rotation-drift-v2 |

## Exact reviewable artifacts

Under `$ZOO_DATA/runs/track-b-pivot-unblock-a-20261105/roster-preview/`:

- `replacement-map8.json`: exact old/new IDs, new content keys, parent IDs,
  primary manifest/fingerprint paths and hashes. All8primary manifest,
  full-weight and spectrum file hashes rechecked against A's measured map.
- `models.csv`, `edges.csv`:166model rows and unchanged edge cardinality.
  The158 nonreplacement model rows are unchanged; parent incidence, edge types,
  licenses and root families remain identical after reversing the ID substitution.
  Shared CSV schema stays version1; a separate roster ID versions the amendment.
- `prospective-artifact-map166.json`:158 previously formal measurements plus8
  measured constructions, each under its actual content key. All are prospective;
  this does not update A's current158map or grant credit before approval.
- `verification.json`: PASS_PREVIEW_ONLY, with original/preview table hashes and
  exact substitution-map hash. Frozen inputs and held-out allocation stay fixed.

Proposed roster ID: `formal166-public-constructions-v2`. CSV paths are logical
lab identifiers; A's explicit primary manifest/fingerprint mappings are the
source of truth for numerical calibration. Raw checkpoints retain their existing
root bindings, including13temporary checkpoints. Their23.8GB durable-copy issue
is separate: A's manifests, full spectra and sketches already live on primary.

## Action authorized only after approval

Activate the versioned tables/map in a new data-root directory, preserving the
preview and all originals. Record approval scope and hashes in the activation
manifest. Verify all166identities and complete requested parent/independent
pairs with A's existing readiness machinery. If ready, A may run its existing
calibration and calibrated-R6 pipeline into new output paths, using unchanged
measurement settings, statistic, independent controls and prior held-outs.
No partial-cohort calibration, original equivalence-threshold change, new
method tuning, extrapolated gate claim, blind access, deletion, upload or push.

Suggested implementation: invocation-only overlay and request; leave default
`lab/models.csv`, `lab/edges.csv`, original inventory and previous A maps intact.
A may consume the activated overlay explicitly. If the existing readiness
validator rejects it, report the specific mismatch rather than relaxing guards.

If rejected, A remains158/166 under the original roster; the eight constructions
remain auxiliary. Regenerating identical failed recipes cannot remove that
scientific blocker. No approval is inferred from elapsed time or the E6 pivot.

## Activation verification — 2026-11-05

Owner explicitly replied “approved.” Original reviewed bytes are retained as
`approved-proposal.md` in the run; approval.json binds their SHA256. The active
166mapping scores all180independent and157parent pairs through the unchanged
k64median-cosine scorer. Original tables/default inventory/maps remain intact.
No calibration, G0/G1 gate decision or C blind access occurred.

Preflight exposed one separate supported-layout bug in Qwen3's embedded PEFT
head:56LoRA-product views collided with parent dense-matrix names. A's narrow
view-planner repair follows extractor Task3's existing canonical-name/[lora]
contract, preserves physical tensor sources and comparison guards, and leaves
checkpoint identity unchanged. New measurement and exact superseded pointers
are sealed in activation.json; original numerical artifacts remain intact.
A's repaired data are at `runs/track-a-embedded-peft-repair-20261105/`.
The reviewed source fix is now merged locally into Model-Zoo main at `7c3c356`.
Historical measurement freezes still record its source state at execution time.
