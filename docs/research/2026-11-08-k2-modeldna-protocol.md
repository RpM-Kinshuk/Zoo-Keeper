# modelDNA matched lab comparison
Declared before native extraction/scoring, 2026-11-08.

Use published PyPI modeldna0.1.0, installed --no-deps with no existing package
changes; downloaded GitHub source stays inert. Defaults fast mode/seed20260704,
native features, default bootstrap calibrator and Thresholds unchanged.
Freeze B's12 quantization queries,126 candidate comparisons,46 distinct models
from runs/track-b-k5-parent-out-20261108/protocol.json. Verify every safetensors
hash against B's input ledger, preserve source recipes; no truth/split changes.

Use local native sources and extract_fingerprint, compare_fingerprints, judge.
Pass the actual shared anchor family to judge (siblings are not different
families). Judge both source-present and source-absent pools. Save complete
native verdicts/evidence including missing features and abstentions. Report
exact immediate-parent success only where the method makes an unambiguous
answer; family consistency and its bootstrap probability are not a validated
parent guarantee. No calibrator training or threshold fitting on these cases.
These are all relatives: ancestry AUROC/FPR unsupported without independent
same-architecture negatives. Names retain construction truth; metadata is not
new verified historical evidence.

Execution completeness PASS iff46 fingerprints and126 evidence records plus
12 present/12 absent verdicts are accounted for, pins match and no unrecorded
missing feature/case. Unavailable inputs become not_run, never zero error.
This is an operational pass; no new scientific accuracy threshold is set.
Report numeric read bytes using source.bytes_read, extraction and comparison
wall time, separate full-file verification scan bytes. CPU8threads, no GPU.
Expected <1hour, stop at110minutes;≤46 models, no network model reads. Local
native sample fingerprints stay private under ZOO_DATA; never upload them.
Existing local reader handles only safetensors; never execute model code.
