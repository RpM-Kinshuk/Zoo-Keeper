# Draft: authorize the prepared remainder of K0 storage
Status: decided — owner approved the prepared remainder on 2026-11-07 ("authorized. go ahead.").
Due: now | If no answer: preserve all local copies; do not upload the remainder.

Recommendation: approve (confidence: high). The first temporary delivery is already durable; the remaining local inventories passed hash checks. This is the existing K0/D1 scope, but automatic permission review rejected it as beyond the payload-specific approval for the first batch.

Prepared payload: 126 remaining constructed checkpoints, 168,915,002,858 bytes; lab-only evidence 2,848,505,527 bytes; filtered database 8,654,697 bytes; small lineage/recipe/table metadata. Destinations: RpM-Kinshuk/zoo-lab, zoo-evidence, zoo-db. All remain public/manual gated. No third-party weights, raw blocks, parent caches or gated-family fingerprints. No local deletion.

Big-picture effect:
- Claims: no change; auxiliary and failed construction statuses preserved.
- Papers and dates: protects existing lab data for P1.
- Real-world use: repeatable restoration of lab artifacts.
- Cost: free storage under D1; transfer time and owner approval only.
- Risk: transfer of unpublished lab artifacts; manual gating remains required.

Options: approve the prepared remainder; or stop after the verified temporary batch.
Can it be undone? Copies can be removed later only with separate authorization; disclosed bytes cannot be undisclosed.
Uncertainty: the automatic reviewer treated the prior approval as specific to 13 checkpoints.
Second opinion (skeptic): pending steward screening; draft only.
Details: `$ZOO_DATA/runs/track-a-k0-20261107/{RULES.md,inventory*.json,remote-verification.json,upload-primary.py}`.

Execution: approved upload-primary.py completed. All batches and combined remote lists/sizes/SHA256 checks PASS. Final revisions are in release-manifest.json; all local copies retained.
