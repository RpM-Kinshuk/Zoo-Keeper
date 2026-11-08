# D1: Set up the approved lab storage
Status: decided | Decision: public, manually gated Hugging Face storage (overseer, Nov 7).
Setup complete; approved archive verified in the Nov 7 remote-check report.
All 139 prepared constructed checkpoints are durable; 1,430 artifact files
and 404 safetensors hashes match recorded remote inventories. Local copies remain.
Source: `runs/track-a-k0-20261107/final-remote-check.json`; pinned revisions in DATA.md.
The setup steps below are fulfilled reference instructions, not a pending ask.

The approved plan replaces the earlier private-storage and PRO proposal; no purchase is requested.

Setup for the overseer:
1. Create dataset repositories `zoo-lab`, `zoo-evidence`, `zoo-db` under your account: public with manual access approval.
2. Add a fine-grained write token for those three only as `HF_UPLOAD_TOKEN` in `.env`; tell A only the account name and that setup is ready.
3. A uploads the at-risk checkpoints first, verifies remote file lists, sizes and safetensors hashes, and records remote revisions.

Big-picture effect:
- Real-world use: preserves the benchmark and later zoo releases.
- Claims: no change; storage is not scientific verification.
- Papers and dates: protects evidence needed for P1.
- Simplicity: three repositories and the existing resumable upload path.
- Cost: the approved plan uses public storage; no PRO purchase authorized or needed by this plan.
- Risk: data loss remains possible until remote copies are verified; licensing exclusions in DATA.md still apply.

If setup is delayed: A prepares the inventory and a local stopgap where safe, without deleting unique files; a same-server copy is not durable backup.
Can it be undone? Access settings can change only under the existing publication rules; deleting a sole copy requires the overseer.
Second opinion (skeptic, original card): agreed that the at-risk data should be copied off-server immediately rather than wait for a paid plan. This was a review of the original proposal, not verification of an upload.
Details for agents: [DATA.md](../DATA.md), PLAN.md K0. Measured inventories and remote revisions now appear in DATA.md and RESULTS.md.
