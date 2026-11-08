# K7 bounded real Hub writer comparison, 2026-11-08

Measured a pinned Qwen 7B instruction model against one declared refusal-removed
candidate using existing extractor manifest, view, range-read and row-bound helpers.
No raw blocks were saved; no new method implementation or full spectra were built.
This is a selected-block diagnostic, not a complete fingerprint or verified edge.

Parent snapshot: `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`.
Candidate: `knoveleng/Qwen2.5-7B-Instruct-Uncensored@888673633f34f493a71f7594eab1bb047e8ad353`.
Producer-named parent, not independently verified construction input.
Protocol written before weight reads in the run's `PROTOCOL.md`.

| Matrix (zero-based layer) | Relative update | Rank-one residual / delta norm | Cast-noise bound / delta norm | Stated consistency rule |
| --- | ---: | ---: | ---: | --- |
| model.layers.16.self_attn.o_proj.weight | 0.02159903 | 0.07264634 | 1.44681544 | passes, non-discriminating |
| model.layers.16.mlp.down_proj.weight | 0.02052196 | 0.07645242 | 1.52268724 | passes, non-discriminating |
| model.layers.17.self_attn.o_proj.weight | 0.02031807 | 0.07713000 | 1.53799720 | passes, non-discriminating |
| model.layers.17.mlp.down_proj.weight | 0.01928200 | 0.08121047 | 1.62060041 | passes, non-discriminating |

All samples use rows [1664,1920), selected by the existing ROW_BLOCK rule.
Updates exceed the prewritten 3e-3 floor. Both storage dtypes are bf16.
The rank-one approximation leaves a small measured residual. However, the
conservative bound exceeds the entire delta norm in every case: even the
zero-rank approximation would pass. **The consistency test is uninformative
for rank identification here**, not evidence that C2's exact-mark test passed.
No negative controls, all-layer check, shared-direction comparison, behavior
measurement or calibration was run. Construction tier remains declared.

v1 is invalid for its consistency verdict: it used the extractor's decoded
float32 dtype instead of the original bf16 storage dtype. v2 fixes that
implementation error; no thresholds were tuned. v1 remains archived.

Successful v2 weight payload: 46,137,344 bytes; metadata/header payload: 107,782 bytes; elapsed: 4.805406s. These are reader payload counters, not total physical network traffic. v1 repeated the same bounded reads and is excluded from v2 cost.

Run: `$ZOO_DATA/runs/track-c-k7-writer-diagnostic-20261108-v2/`. Input manifest SHA256 `3bd8f429736d8906cc35625610e80df271127c2cc441bddcc4b0a3e3b94dfbec`; protocol/script/manifests checked. Only scalar diagnostics retained.

Next for B: use scalar residuals as exploratory evidence; set an informative noise control before any rank-one detection verdict, and keep this pair outside frozen verified truth until construction provenance is resolved.
