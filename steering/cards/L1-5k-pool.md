# L1: Measure family false matches as the candidate pool grows
Status: draft BLOCKED | Why now: [K8](../PLAN.md), launch card due November 21.

Recommendation: keep this draft blocked until the candidate and query inputs are pinned. No overseer launch is requested yet. If no answer: do not launch; K8 remains open. The pool-size curve is still NOT MEASURED and C1 does not advance.

Command (from workspace root): **withheld**. No frozen 5,000-model candidate roster was supplied or identified in the inspected pilot inputs; a paste-ready launch would imply missing inputs exist. The existing pilot runner points to its historical run through a temporary marker and must not be reused as a 5k launcher.

Expected cost: GPU-hours **unknown** · wall-clock hours **unknown** · storage written **unknown**. **Prediction only:** 2.613 TB requested tensor payload and 492.27 serial model-hours for 5,000 draws with exactly the completed pilot's mixture, hardware and policy, excluding gaps/retries. This is not a 5k launch budget; requested tensor payload is not measured network traffic.

Measured extraction cost: nine existing header-verified 7.5–8.5B examples: median **641,728,512 bytes / 849.98 seconds**. Comparable standalone **70B cost NOT MEASURED**. Costs of searching the growing pool and end-to-end check latency also remain unmeasured.

To become ready: freeze >=5,000 distinct readable candidate revisions and manifests; freeze the labeled query roster, nested pool sizes 100–5,000, splits and threshold (>=200 queries before the P1 scale check); pin source hashes and fill the aggregate bytes/storage/GPU/wall budget. Then supply and validate an exact command using existing `evidence.cli batch --measurement_mode row_blocks`, with explicit workers, device, sketch size and bounded caches. Existing artifact-aware resume is available; demonstrate it on these inputs before marking ready. The frozen hash record here covers only reused pilot evidence, not a future 5k roster.

You will know it worked when: all 5,000 candidates have recorded outcomes and explicit gaps, and a report gives false matches per labeled query at every frozen pool size plus actual costs.
Stop it if: access/rate-limit handling or cache bounds fail, input hashes change, or costs exceed the approved budget. Safe to stop and resume: **not yet verified for this future run**; the ready card must include the same exact command and artifact verification procedure.
After it finishes: tell the steward. No other action is needed from you.

Second opinion (skeptic): pending; this blocked draft is not ready for the overseer's launch batch.
Details for agents: [cost evidence and exclusions](../../../runs/track-a-k8-20261107/REPORT.md), [frozen rules](../../../runs/track-a-k8-20261107/PASS_FAIL.md), [one input hash record](../../../runs/track-a-k8-20261107/INPUTS.sha256.json). No 5k job was launched.
