# Real 7B source-neighborhood comparison
Declared before metadata admission/weight extraction/scoring,2026-11-08.

Question: does the immediate-source ambiguity observed for modelDNA in the lab
also occur within real published 7B neighborhoods? This advances K7 beyond an
isolated block probe; it does not test a newly fitted detector or recover history.
Native samples cannot be recovered from existing Zoo sketches. Reuse the selected
and pinned K7 handoff; no expanding Hub search, model training or fresh roster.

Two fixed neighborhoods:
- OLMo-2-1124-7B base,SFT,DPO,Instruct (the base alias already pinned in handoff).
- Qwen2.5-7B base,Instruct,sci4ai Abliterated,knoveleng Uncensored.
Use exact child/available-parent snapshots from candidate-pairs.json. Six declared
child queries; every other group member is a candidate,three per query. Score18
pairs and native source-present/source-absent verdicts for each query. Keep
construction-input revisions unknown. Same-family alternatives are controls for
choosing among relatives, not independent ancestry negatives; do not report FPR,
AUROC or verified immediate-parent accuracy. No generic contradiction accusation.

Published modeldna0.1.0 native fast mode/default seed/Thresholds/bootstrap
calibrator, no fitting; actual shared family label per group. Native four-signal
extractor/comparator/verdict package unchanged. Save all features/unsupported notes
and unresolved answers. Native family probability is not calibrated exact-parent
probability. No C2/noise/edge-type/direction inference. B owns those tests.
If model incompatible/inaccessible/not run, record it and report missing complete
neighborhood; never substitute another repo or drop its denominator.

Operational PASS iff all8 fingerprints,18 comparisons and12 query-pool verdicts
are accounted for, pinned revisions admitted and file ranges bounded; feature gaps
explicit. Full-family comparison missing data is not_run, never zero error.
No scientific accuracy threshold. Outcomes: usable/partial cases, unresolved
vs definite answers, declared-parent agreement only where meaningful, actual
read/metadata bytes and wall time. No truth/split/threshold changes.

Use existing Model-Zoo HubReader (auth/redirect/range/rate-limit safeguards) through
a minimal native ModelSource adapter; avoid native HubSource's full-response
fallback and retry policy. Model metadata/listing via authenticated HfApi only
for these8pins. Native header parsing is bounded; existing reader bounds each
range against file size. Tests before adapter: exact offset/bytes preservation,
size/deadline/byte-budget violations before reads, parallel accounting and one
resolve per file. No copied auth logic or new resource manager. Fake readers for
offline tests, one core review before scientific interpretation.

CPU8threads,8 network workers; no useful/admitted GPU. Hard deadline110min;
8models (below100), expected<60min, per-model1GiB source-byte cap, total8GiB.
If admission or progress makes those limits implausible, stop and report a budget
gap; do not evade limits by splitting into more runs. No full model downloads.
Only small native fingerprints/scalars/metadata retained locally; no raw row
blocks. No uploads or publication. Redacted error categories only, no tokens or
signed URLs in reports/logs. Credentials inherited; never open/print .env.
Stop when the fixed neighborhoods are measured or their concrete blocker is known.

## Explicit case extension and format admission, before further payloads
The initial8-snapshot plan is NOT complete: OLMo SFT/DPO are .bin-only at the
selected pins. Do not unpickle, convert, choose other revisions, or read the
other two OLMo models just to fill cells. Keep all original OLMo cases not_run.
Initial Qwen base/Instruct fingerprints are valid and reused exactly; no repeat
reads. Initial generic stop treated format incompatibility as transport failure;
correct this operational distinction before continuing the supported cases.

Add the already-selected Tulu3 four-node base/SFT/DPO/RLVR neighborhood as an
ADDITIONAL case, not a replacement or a rewritten successful original test.
Use its exact three child/available-parent snapshot pairs in the same K7 handoff;
parent construction revisions remain unknown. Total fixed selection is now12
snapshots,9queries,27candidate comparisons,18pool verdicts. Report all denominators,
including the four OLMo snapshots/nine OLMo comparisons/six OLMo verdict gaps.
Question unchanged: does family-vs-immediate-source ambiguity transfer across
usable real families? The extra family is needed because the original second
family is unsupported. No further family search/expansion in this run.

Before payloads, admit whole neighborhoods using file lists and authenticated
HEAD access to one safetensors shard at each exact pin. A missing/denied/unsupported
member blocks that complete neighborhood; do not waste its other payload reads.
Metadata-only .bin models stay explicit no_safetensors. No gated-access requests
or license acceptance. If Tulu is blocked, finish Qwen and report the gap.
Same modelDNA defaults and same all-other-members/source-present/absent pools,
using one shared tulu3 family label. No independent ancestry negatives claimed.
Byte caps apply per model; source-byte/elapsed reports separate reused acquisition
from new reads and retain original failed-format preparation. All≤12models and
combined expected<60min; per-run110min ceiling is not permission to exceed the
project's120min total-job boundary. No larger run is being split or launched.
