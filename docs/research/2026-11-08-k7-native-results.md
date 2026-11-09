# Real-model source-neighborhood comparison — 2026-11-08

Practical finding: modelDNA0.1.0's native four-signal comparison supports
same-family consistency but makes no definite immediate-source answer among
these selected real relatives, with or without the declared-source snapshot.
This reproduces the lab ambiguity for Qwen refusal/posttraining and Tulu
SFT/DPO/RLVR pipelines. It is not an independent-negative or verified-parent test.

| Source neighborhood | Pinned/native snapshots | Scored/planned pairs | Native pool verdicts | Status |
| --- | --- | --- | --- | --- |
| Qwen2.5-7B base,Instruct,two refusal-edited children |4/4|9/9|6/6|Complete descriptive neighborhood|
| Llama3.1/Tulu3-8B base,SFT,DPO,RLVR |4/4|9/9|6/6|Complete descriptive neighborhood|
| OLMo2-1124-7B base,SFT,DPO,Instruct |0/4|0/9|0/6|Not run: SFT/DPO pins are bin-only|
| Total fixed selection, including explicit extension |8/12|18/27|12/18|Partial selected-roster coverage; original completeness target NOT passed|

The initial8-snapshot Qwen/OLMo plan remains incomplete. Tulu is an explicitly
predeclared additional case from the EXISTING K7 selection, not a hidden
replacement or a rewritten successful original experiment. It supplies another
architecture and ordinary full-posttraining branch, distinguishing the original
Qwen sparse-edit setting from a different training pipeline. No broader Hub
search, different revision, conversion or unpickling used to rescue OLMo.
All12 planned identities and9 query pools remain in protocol/results. Both
remaining OLMo payloads were skipped because they cannot complete that case.

## Evidence and supported answers
All six usable queries return SAME_FAMILY_UNRESOLVED with the declared-source
snapshot present, and all six return the same class when that snapshot is
removed. Zero definite immediate-source answers across the six usable queries.
Do not report a zero source-error rate: that denominator has no definite answers.
Missing OLMo inputs are NOT_RUN, not successful abstentions. All18 measured
pairs have all four aggregate features; no signal silently omitted.

The pool contains every OTHER member of each four-node group: three candidates
per query. Source-present/removed are candidate-pool controls, not independent
ancestry negatives. The nominal same-family grouping is supplied to the native
verdict engine as its normal reference-family label, not inferred globally here.
There is no AUROC/FPR, unknown-family recall, large-search guarantee, type/order
verdict or comparison proving Zoo's detector better. Those require matched
independent controls and B's agreed tests.

All relationship labels remain PUBLISHER DECLARATIONS. Each child and available
source snapshot is exactly pinned, but historical construction-input revisions
remain UNKNOWN. No construction edge is promoted to verified, no exact-mark
claim follows, and no source-misdeclaration accusation is supported. Full source
cards/metadata/hashes remain in the inherited K7 handoff. Detailed18 scores and
12 native verdicts are in results.json, with the nine OLMo pair gaps and six
OLMo verdict gaps explicit. Native bootstrap calibrator/Thresholds/fast seed
unchanged; no fitting on these already selected/exposed cases.

## Admission, implementation and cost
Metadata and authenticated HEAD checks admitted the eight safe Qwen/Tulu inputs;
OLMo's selected SFT/DPO snapshots expose only .bin weights. Those remain metadata-
only under project rules. Gated Llama access used the existing granted credential;
no license acceptance/access request/upload occurred. Native fingerprints stay
local, including samples from gated-family checkpoints; no raw row-block files.

Published installed modeldna0.1.0 extractor/comparator/judge unchanged. A small
ModelSource adapter reuses Model-Zoo HubReader for authenticated, bounded,
redirect-safe ranges. The native HubSource fallback that can accept full-body
responses was not used. Offline byte-contract/prefetch integration tests match
all four native local-reader signal families exactly. Core review's deadline
hole was reproduced RED and fixed; queued work halts after a source-read failure.
Full baseline package31passed after the separate Hill validation repair.
All source pins are tied to metadata and advertised file SHA256/size checks,
not a claim that full weight-file hashes were re-downloaded and recomputed.

The successful comparison reused Qwen base/Instruct's two earlier fingerprints;
no weight rereads for them. Cumulative native source bytes for the eight unique
fingerprints: 2,261,064,484. New v2 source bytes:
1,743,026,409. Initial acquisition elapsed 198.377763s;
v2 acquisition/comparison elapsed 863.483332s. Metadata API wire
bytes were not measured; do not call them zero or claim these are complete
end-to-end index/search costs. Individual source-byte counts include headers,
config/tokenizer and sampled weights. CPU8threads/network8workers, shared host
eniac.cs.dartmouth.edu; no admitted or reserved GPUs. Each source below1GiB;
combined measured phases within the authorized≤100models/≤2hour boundary.
These shared-host wall times are not isolated hardware benchmarks.

## Next useful dependency
B can reuse the exact pinned neighborhoods and all-other-member/source-removed
pools for a scientifically agreed matched comparison. Native fingerprints do
not convert into Zoo row sketches; do not invent missing outputs. No more weight
reads are justified merely to fill cells. OLMo needs safe upstream weights at
traceable revisions to become usable here; historical producer inputs and
independent decoder controls are still required for stronger verdicts. B has
AMEND/deferred the correlated adapter witness contrast: no new witness scoring.
Nov14 now has two usable declared-tier real neighborhoods plus an explicit format
gap. Dec1–5 full-tier targets remain; the verified/control-complete tier is open.
No truth, split, scientific threshold, claim, commit, merge or push changed.
