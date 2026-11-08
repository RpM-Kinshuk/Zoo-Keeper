# L1: Test family matching as the search grows
Status: draft BLOCKED | Plan: K8 | Ready-card target: Nov 21

Purpose: measure useful answers, mistakes and costs as the search grows. Preparation is tested; the large weight run has not started.

Ready inputs: 5,000 pinned, distinct header/resource-eligible candidates plus 154 reserve; 70 exclusions retained. The 216-query catalog includes only two inherited held-out keys. Planned compatible counts range 0–252 per query; 12 have zero. These header estimates precede value-dependent view removal, self/duplicate removal and numeric checks; they are not bounds on final score eligibility or 5,000 independent distractors per query.

Still blocked:
- B/steward must review whether this pool and the exposed query catalog answer the scientific question. No new truth, split, threshold or scientific freeze is applied.
- Persistent free space must reach 52,977,254,828 bytes (about 52.98 GB). The budget snapshot records 26.25 GB free, a 26.72 GB shortfall. No deletion, extra upload or purchase is authorized. See [storage draft](draft-k8-storage.md).
If no answer: retain data and keep this card blocked; use existing evidence for the November review.

Prepared command, OWNER ONLY after this card becomes ready, from the workspace root:
```bash
set -a; . ./.env; set +a
bash "$ZOO_DATA/runs/track-a-k8-launch-20261108/collect.sh"
```
The verifier checks pinned inputs/source and current space before invoking the existing collector: CPU, four workers, one numerical thread each, k64/row256, sampled experts, no parents or parent cache. No large launch is authorized by this draft.

Expected cost: 0 GPU-hours; 2.567 TB planned tensor requests before retries; 19.70 GB fingerprint estimate / 40.66 GB planning reserve. Dense-anchor scaling predicts 112.44 serial elapsed hours, or 28.11 ideal four-worker wall hours. Historical mixed-host bins predict 505.91 serial hours, or 126.48 ideal wall hours, omitting 91 models without a historical bin. Neither is a measured CPU time, runtime guarantee or uncertainty interval; file/request overhead and concurrency can dominate. The hard timeout is seven days. Peak process RAM and fresh-model end-to-end latency remain unmeasured.

Success: all intended distinct identities have valid, correctly pinned artifacts; then B's reviewed analysis reports usable/compatible counts, label/test exposure, known-present recall, false-hit bounds including unknowns, abstention and cost at each pool size. Missing size or truth stays a miss, not a successful benchmark.
Stop if free space approaches 2 GiB, memory threatens the host, access is denied or systematic errors appear. Preserve completed artifacts/logs and wait for all workers to exit before rerunning the same command. Completed-artifact resume was tested on one real 8B model with zero new reader calls; interrupted models may reread data. [Resume details](../../../runs/track-a-k8-launch-20261108/RESUME.md).

Second opinion: independent preparation review found a header-boundary guard gap, reproduced and fixed with RED/GREEN regression; original review and lead disposition preserved. Storage-card skeptic agrees to retain data and hold launch. Neither substitutes for B's pending scientific review.
Details: [resource report](../../../runs/track-a-k8-headers-20261108/REPORT.md), [per-query planned counts](../../../runs/track-a-k8-protocol-20261108/geometry-coverage.csv), [proposal](../../../runs/track-a-k8-protocol-20261108/PROPOSAL.md), [paired costs](../../../runs/track-a-k8-cost-20261108/REPORT.md).
After a ready launch finishes: tell the steward; no publication or deletion follows automatically.
