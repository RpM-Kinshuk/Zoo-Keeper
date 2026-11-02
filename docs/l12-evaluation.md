# Real L12 delivery evaluation, 2026-11-02

The new report is `$ZOO_DATA/runs/track-b-l12-evaluation-v2/REPORT.md` with
machine-readable `report.json`, `pre-score-freeze.json`, training predictions,
control metrics/universes and tree distances. It consumes committed Model-Zoo
`48a70fc` and A's `track-a-l12-20261102.mnbqfy_y` delivery. Original reports,
held-out assignments, trained checkpoints and extraction artifacts are preserved.

The evaluated union has36 ready models/30 construction edges: the prior33 plus
three new quantized DPO endpoints. A's six LoRA fingerprints are consumed, so
the training report now covers12 models/11 edges. Independent Pythia seeds1/2
remain development roots and seeds3/4 remain held-out test roots. Comparisons
stay inside each partition; no cross-architecture negatives are introduced.
The0.75 threshold is unchanged, with no fitting on the held-out data.

Held-out same-root AUROC is1.00 at both T1 and T2. Related-pair cosine Spearman
is0.937063 over24 ordered related pairs; all-pair Spearman is0.959511 over60
within-partition pairs. Results for T1 blocks128/256/512 and k64/k128 prefixes
are retained separately. These are small-scope relatedness diagnostics, not a
calibration guarantee or a full G0/P1 result. Parent and ancestor inference
still abstain; the12-model training universe has132 ordered pairs,11 parent
positives and13 ancestor positives, with deployed direction TPR/coverage0.

## Real chains and tree diagnostics

`runs/track-b-chain-cohort-v1` contains three verified depth-three paths:
Smol135→SFTs0→DPOs0→q4/q8, and Smol135→SFTs1→DPOs1→q4. The five inherited
measurements are reused; only three new endpoints were extracted. This is an
extra scoped experiment and earns no formal chain/G1 credit. The prescribed
formal anchors remain Qwen2.5-0.5B, Qwen3-0.6B and Smol360.

New generation took162.925s CPU and wrote807,181,656 safe weight bytes.
Standalone new T1/k64 selected435,290,112 tensor bytes in26.6895s. New T2/k128,
full spectra and free T1 blocks took182.871s on one GPU, selecting2,342,338,560
bytes including parent reads. GPU admission passed two checks at least5s apart
with utilization<80%, occupied memory<80% and24GiB free; all jobs ended.
Together with A's43 inputs,46 unique checkpoints now have T2; all36 ready
cohort models have T1 and T2.

Every family/mode uses one identical intersection of view names, shapes,
geometry and row coordinates across every node. Distances are the mean squared
Frobenius sketch difference, subtracting in float64 before squaring. Settings
(seed, version, tier, block, native size) are validated before use.

At the frozen1e-6 tolerance, expanded Smol135 violates the four-point condition
on4193/4845 quartets at T1 and4201/4845 at T2 (about86.5%/86.7%). Original
17-node Smol results are1847/2380 and1878/2380; the four-node OLMo star violates
its single quartet in both modes. Predeclared1e-8/1e-10 sensitivity results are
reported without relaxing the primary tolerance. Both T1 and T2 flag the tree
assumption on these scoped transform/training cases. Formal multi-anchor tree
coverage remains incomplete; a person retains the pivot decision. nRF is null
because no validated asymmetric assembly was supplied. Truth only labels
metrics and never orients predictions.

## Reproduction and remaining work

```bash
python -m lab.scripts.evaluate_delivery --data-root "$ZOO_DATA" \
  --delivery "$ZOO_DATA/runs/track-a-l12-20261102.mnbqfy_y" \
  --chain-run "$ZOO_DATA/runs/track-b-chain-cohort-v1" \
  --out "$ZOO_DATA/runs/a-new-evaluation-directory"
```

Use the established CPU/offline environment and import paths. New output
folders are mandatory. The initial v1 attempt stopped before metrics when its
explicit input index lacked a retrieval index; it is retained, and v2 builds
retrieval from exactly the12 bound manifests/fingerprint files. A fresh core
review also caught mixed measurement settings and an invalid relatedness edge
label; regressions reject those cases and publication uses `unknown` edge type.

No further A code change is needed for this analysis. B owns formal generated
checkpoints and verified source/tokenizer launch inputs. Full E9 also needs the
larger reference pool, matched baselines and validated asymmetric direction.
Artifact sizes are not complete extraction/verification/repeated-read IO.
Free T1/k128 files share T2-pass counters and do not establish standalone T1
byte savings. Release staging remains deferred. Changes are uncommitted.


## First formal chain completed

`runs/track-b-formal-smol360-launches-v1/execution/REPORT.md` records the first
original-inventory chain: Smol360→SFTs0→DPOs0→q8. All three checkpoints are
verified and have new T1/k64 measurements; the existing base measurement is
reused. This earns1/3 formal chains,1/8SFT,1/8DPO and brings quantization to4/12.
There are18 verified/measured formal IDs; full T2 coverage stays15/166 pending
incremental extraction. New T2 input/argv is frozen in the bundle's
`incremental-A-l12-launch.json`; no A code change is needed. The first formal
chain also violates its single quartet at the frozen tolerances; formal E6
needs the other anchors before a person decides the pivot.

Source preparation promoted the exact Smol360 weights and prepared11 native
training configs with official IDs/seeds, actual data/license/token hashes,
representative safe model-load preflights and a48GiB-free GPU admission wrapper.
Two training jobs completed;9 remain (8ready/1parent-dependent). The q8 terminal
is additional. The tracked completion pointer is
`lab/configs/launches/track-b-formal-smol360-v1.json`. No old manifest was edited.


A subsequently completed the three formal Smol360 T2 outputs, preserving all
old evaluations. The exact-key T2/HDF5/freeT1 delivery is
`runs/track-b-formal-smol360-launches-v1/incremental-A-t2/artifact-index3.csv`;
full formal T2 coverage is now18/166. No repeat extraction is needed.
