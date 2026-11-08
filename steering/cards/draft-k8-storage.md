# Draft: Provide persistent space for the larger family-search test
Status: draft, class 3 | Ask: choose persistent capacity or defer the large run
Due: before L1 becomes ready | If no answer: keep L1 blocked; retain existing data

Recommendation: provide at least 52.98 GB free at the existing data root before launch (confidence: medium).
Why: the selected pool's fingerprints are estimated at 19.70 GB, with a conservative planning reserve of 40.66 GB. Including the proposed margin requires 52.98 GB free; the recorded filesystem has 26.25 GB, a shortfall of 26.72 GB. These are estimates and current availability, not a reservation or quota guarantee.

Big-picture effect:
- Real-world use: enables the larger candidate search and useful-error/cost evaluation.
- Claims: no change; header eligibility and a mixed pool do not establish accuracy or independent hard negatives.
- Papers and dates: existing targets unchanged; storage and B's scientific review block launch readiness.
- Simplicity: use the existing collector and data root; no new storage or archive framework.
- Cost: capacity price is unknown; no purchase is authorized. Collection plans 2.567 TB of tensor requests before retries, CPU only. Storage capacity alone does not approve the large run.
- Safety/legal: preserve all local checkpoint copies and publication gates. No deletion, extra upload or repository publication is authorized; gated-family fingerprints stay local.

Options:
- A: allocate persistent capacity to meet the free-space gate. Gain: unblocks the space prerequisite. Lose: capacity cost and availability must be confirmed.
- B: defer the large run and use existing evidence for the November review. Gain: no storage action or spend. Lose: the larger-search result stays unmeasured.

Can it be undone? Capacity allocation depends on its terms; no data deletion or spending is delegated. The default is to defer.
What we are unsure about: final output size, retry/runtime overhead, concurrent filesystem use and available persistent-capacity terms. /tmp free space is not a durable-storage plan.
Second opinion (independent preparation skeptic): Agree: retain all existing data and hold the large launch until at least 52,977,254,828 bytes of persistent free space are available and B/steward scientific review is complete. This is an estimated planning gate, not a storage guarantee; recheck free space immediately before launch.
Details for agents: [budget](../../../runs/track-a-k8-headers-20261108/budget.json), [resource rules](../../../runs/track-a-k8-headers-20261108/PREFLIGHT_RULES.md), [launch/resume draft](../../../runs/track-a-k8-launch-20261108/RESUME.md). Exact planning gate: 52,977,254,828 bytes = ceil(reserve × 1.25) + 2 GiB; recheck immediately before launch. Steward assigns the card number and decides when to bring it forward.
