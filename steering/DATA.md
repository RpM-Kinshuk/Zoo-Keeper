# Data: storage and publication on Hugging Face

**Decision D1 (2026-11-07):** public Hugging Face storage. The repositories
stay **gated with manual approval** until the P1 submission. Gated
repositories are public, so the storage is free, but files download only
after the overseer approves a request. That keeps unreleased results and
ablated models from circulating before the paper. If the overseer says "fully
open", gating is skipped.

## Access without old agent context

Start with [DATA-INDEX.json](DATA-INDEX.json). It records the actual workspace data root, 18 checked entrypoints, model/evidence maps, archive inventory locations and recovery paths. Old `track-a/b/c` names in retained run directories are permanent experiment identifiers, not agent dependencies. Original manifests and frozen scientific rules remain authoritative.

Current workspace/data root: `/jumbo/yaoqingyang/kinshuk/zoo`. `$ZOO_DATA` was unset during this audit; set it explicitly to that root for local reuse. Nothing requires reading `.env` or reviving a worktree. From the workspace in conda `esd_ind`:

```bash
export ZOO_DATA="$PWD"
export PYTHONPATH="$PWD/Model-Research:$PWD/Model-Zoo:$PWD/Model-Zoo/esd_experiment/src${PYTHONPATH:+:$PYTHONPATH}"
cd Zoo-Keeper
python -m zoo_keeper.cli find --zoo "$ZOO_DATA/runs/track-a-expanded-r6-20261105/zoo" --data_root "$ZOO_DATA" --text smol --limit 1 --json
```

That read-only query succeeded in the Nov10 audit, without weights, credentials or GPUs. For an owner-approved public-scope lab database, use `runs/track-a-k0-20261107/stage-db-with-retrieval/lab`; this does not authorize a new release.

| Need | Starting point, relative to the data root |
| --- | --- |
| Shared real-model complete-matrix comparison | `runs/real-comparison-20261113/{RESEARCH.md,inputs.json,runtime.json,results.json,feature-equality.json}`;11pickle-free private profile NPZs/receipts/model records adjacent.24scores/16rankings replay without weights/network. Exact MLP feature collisions for two attention-only LoRA controls; no calibrated source/general accuracy claim.651.20s/3.30GBremote bodies; independent review accepted scope. |
| Common benchmark coverage, checked subset | `runs/benchmark-coverage-20261113/{RESEARCH.md,coverage.json,assemble.py}`; exact row/pool/label joins,72source hashes, exposure and input contracts.60directed ancestry pairs/4seed roots of one architecture;12source queries/4root families;3documented real pins. Adjacent `real-comparison-inputs.json` fixes the next11-checkpoint scope; prospective metadata/range review. Its follow-up measurements are in the shared real-model comparison above; retain the frozen original map. No complete benchmark claim. |
| Stopped fixed parallel task probe | `runs/task-probe-20261113/RESEARCH.md`; prospective inputs/packets, frozen executed source, setup failure, campaign/logs and partial probes retained. No choices/held-out results or retry. Loading-info JSON bug fixed offline;5focused checks pass. Do not reuse as untouched confirmation. |
| Audited public task comparison | `runs/task-evidence-20261113/RESEARCH.md`;12pinned small source files,400paired records, declared600-example training reconstruction, context/results and two unbound score-evidence.csv rows. Exact report counts reproduce; evaluated weight revisions remain unknown, no catalog/verification upgrade. Ten-intent workflow only; no new inference. |
| Public LoRA+merged construction check | `runs/documented-lora-20261112/RESEARCH.md`; adjacent inputs/results bind the public factors, reference/three relatives,169sampled raw bands and native comparison. Offline `scripts.documented_lora --score-only` reproduces scoring. Extra-factor compatibility, not general source detection; raw reconstructive evidence stays private. |
| Recorded weight-file budget query over703entries | `runs/catalog-budget-20261112/{REPORT.md,query-answer.json,inventory.csv,eligible-keys.json,freeze.json}`;95eligible selected-file inventories at≤2GB under fixed recorded license/access/format predicates, top10allunscored, first-two diff has0shared views. Sizes are not RAM/VRAM, full-package or loadability guarantees. CLI adapter: `Zoo-Keeper/scripts/find_weight_budget.py`; review/replay adjacent. |
| Existing independent-seed decoder comparisons: do not repeat | `runs/track-c-k2-ancestry-controls-20261109/{REPORT.md,results.json,inputs.json,cross-method-input-audit.json,label-semantics-audit.json}`; all four comparators already score the exposed four-root Pythia14m/quantized-child panel.60ordered dependent pairs per method, no broad decoder-scale/FPR/parent guarantee. Native developmental negatives abstain despite the SAME_FAMILY_UNRESOLVED string. |
| Complete formal166 measurement mapping | `runs/track-b-pivot-unblock-a-20261105/active-roster-v2/artifact-map166.json`; adjacent models/edges tables |
| Current joined edit measurements | `runs/track-b-k5-joined-20261108-v2/shared-evidence.csv`; adjacent results/protocol/verification |
| Existing family catalog and input hashes | `runs/track-b-existing-artifacts-20261108/{results,inputs,protocol,groups}.json` |
| Current findings and their exact reports | [RESULTS.md](RESULTS.md); paths beginning `runs/` are relative to the data root |
| Saved real-model/comparator measurements | `real_model_cases` and `matched_baselines` entries in DATA-INDEX.json |
| Official FP8 documented-source query | `runs/documented-fp8-20261112/{REPORT.md,query-answer.json,results.json,verification.json}`;196raw private band inputs/hash index in `bands.json`, frozen source/relative pins, conditional replay mismatches and native decoded-input comparison. Documentation names the source repository; measured immediate source stays unresolved. CSV/replay/reviews adjacent; no raw-band publication. |
| Two-available-model task workflow and reusable score rows | `runs/task-choice-available-20261112/{REPORT.md,results.json,workflow-scores.csv,freeze.json,verification.json}`;32probe/256validation per task, strict8-token exact A/B, invalid counts/protocol pins; independent/doer replay. Not official benchmark/catalog Raw scores or general selection benefit. |
| Stopped three-model acquisition and preserved inputs | `runs/task-choice-20261112/{REPORT.md,attempt1/runtime.json,transfer-check.json,prepared.json}`; original49-source archive, incomplete7B prefixes, two complete pinned cached checkpoints; NO_BENEFIT transfer check, no task answers in this panel. Literature context/primary snapshots adjacent. |
| Score-covered candidate evidence and choice baseline | `runs/model-choice-20261112/{REPORT.md,candidate-evidence.json,freeze.json,results.json}`;25evaluations/22identities, original Raw rows/K10 joins/published context,13eligible candidates and fixed choices; CSV exports/reproduction adjacent; nominal metadata only, old catalog unchanged |
| Fresh acquisition-to-query example | `runs/query-journey-20261112/{REPORT.md,freeze.json,results.json,outer-timing.json}`; queryable `catalog` symlink, frame, safe metadata, fresh row fingerprint, exact score subset and7CLIoutputs;3model scope, no performance recommendation |
| Saved-lab merge-source search and its limit | `runs/merge-source-search-20261112/{REPORT.md,freeze.json,results.json}`; three center merges, six SFT/DPO controls, fixed retained-relative/removal pools, all fits and independent reproduction; source precision remains unmeasured |
| Documented real SLERP and missing-source comparison | `runs/documented-slerp-20261112/{REPORT.md,results.json,freeze.json,review-verification.json}`; five identity-bound current row fingerprints and three fresh/two reused native artifacts; recipe truth, historical-input caveats and lab-control exposure retained |
| Checkpoint/evidence/database archives and local stages | `runs/track-a-k0-20261107/release-manifest.json`, `inventory*.json`, `final-remote-check.json`; stage roots in DATA-INDEX.json |
| Pinned same-architecture candidate pools | `runs/scaling-pools-20261111-v2/{q25-05,q3-06}-pool.csv`; summary/outcomes/manifests/review adjacent;2,000 header-compatible distinct inventories each, not completed weight measurements |
| Current small-model extraction cost | `runs/scaling-cost-probe-20261111/{freeze,verification}.json`, `row_blocks/summary.json`; two fresh F32 fingerprints, exact payload/range checks, no broad lineage claim |
| Rank3 synthetic arithmetic and its limit | `runs/heretic-rank3-noise-20261111/{results.json,REPORT.md,proof-review.md}`; known-factor bound, not pair-only scoring |
| Draft large-launch material | `runs/scaling-launch-20261111/{launch-plan.json,README.md,COMMAND.txt}`;4,000 pinned inputs/metadata budget; scientific scoring not ready, no large job launched |
| Historical collector data and baseline sources | `Model-Zoo/analysis_runs/` and `Model-Research/`; preserve per-run provenance and exploratory limits |

**Verified locally Nov10 (independently checked against the access index):** all 498 manifest/fingerprint/full-spectrum references for 166 mapped models exist and can be opened; 1,349 staged archive files match their recorded sizes. This is an access/metadata check, not a fresh scientific score, full payload-hash audit or remote backup verification.

**Recovery status (Nov10):** the original 13-checkpoint temporary upload stage is absent. Four Heretic `model.safetensors` files remain under `lab-restored-heretic/models/` and size-match the archive inventory. Their 24 companion files (config/recipe/lineage/edit records/directions) are now restored under `lab-restored-archive/models/`; all 1,297,366 bytes match original inventory sizes and SHA256 hashes, independently checked. Use these two explicit roots together; no weight files were copied or freshly hash-verified. Proof: `runs/heretic-companion-recovery-20261110/verification.json`; recipe/saved-scalar audit: adjacent `saved-rank3-diagnostics.json` and `REPORT.md`. Nine other temporary-origin checkpoint directories remain absent at the known local paths. This audit did not establish when the temporary files disappeared.

The checkpoint dataset/revision below is the recorded recovery source for all 13. DATA-INDEX.json records restored companions, remaining missing models and exact download patterns; it does not imply all remote files were rechecked today. Companion recovery is complete; item1 still needs its recipe-specific arithmetic/noise validation before scoring. Restore older full checkpoints only when needed; do not regenerate them or treat `/tmp` as durable.

From the workspace, with the approved Hub read credential already inherited (never open/print it), restore just the missing companion files using the existing library:

```python
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download

index = json.loads(Path("Zoo-Keeper/steering/DATA-INDEX.json").read_text())
recovery = index["checkpoint_recovery"]
credential = os.environ.get("HF_TOKEN") or os.environ.get("HF_UPLOAD") or os.environ.get("HF_UPLOAD_TOKEN")
if not credential:
    raise RuntimeError("Owner must set an approved Hub credential in the environment; do not paste it into chat.")
snapshot_download(repo_id=recovery["repo"], repo_type=recovery["repo_type"],
                  revision=recovery["revision"], token=credential,
                  local_dir=recovery["suggested_restore_root"],
                  allow_patterns=recovery["restore_companions_allow_patterns"])
```

Run it in `esd_ind` on a workspace needing recovery; the Nov10 continuation performed this companion-only download and verification. Validate restored file sizes/hashes against `inventory.json` before reuse. For a needed older checkpoint, select its recorded `restore_missing_models_allow_patterns` entry instead. Keep original tables/protocols unchanged; use the explicit restore-root alias in a new run config. The evidence/database archive layout is described below.

**Backup boundary:** the Nov7 verified archives cover 139 constructed checkpoints and 112 permitted lab-evidence identities. This does not establish offsite backup of every later run or Hub/restricted-family measurement. Those local assets remain preserved; a broader backup has not been verified. Access index and findings are files in the shared checkout, currently uncommitted like the other migration changes; an owner commit/backup is needed for durability beyond this filesystem.

When new reusable data is produced or moved, update its entrypoint here/in DATA-INDEX.json together with RESULTS.md and the original run manifest. Do not create a per-agent log or duplicate per-model inventories. Before reuse, check pinned identity, file presence, scientific scope and existing exposure; never infer blindness from an agent reset.

## How much storage, and why

Measured inventory on 2026-11-07, replacing the earlier storage estimates:

| Constructed local artifacts | Models | Files | Bytes | Remote safety |
| --- | ---: | ---: | ---: | --- |
| Temporary delivery (Heretic, decision head, drift-v2) | 13 | 81 | 23,803,007,860 | verified durable |
| Primary training/adapter batch, including merged adapter weights | 45 | 338 | 66,935,465,778 | verified durable |
| Other formal constructed outputs | 54 | 345 | 95,996,397,722 | verified durable |
| Auxiliary outputs outside the current formal roster | 27 | 196 | 5,983,139,358 | verified durable |
| Permitted lab-only evidence | 112 model identities | 448 | 2,848,505,527 | verified durable |
| Filtered lab-only database snapshot | 112 model identities | 18 | 8,654,697 | verified durable |

All constructed checkpoints total **192,718,010,718 bytes**. The priority
training/adapter batch includes full merged LoRA weights as well as adapters,
so its size is not an estimate of irrepeatable training bytes alone.
Third-party bases/references are excluded. Auxiliary output status and failed
drift-equivalence findings are preserved. Local copies were retained at the Nov7 archive verification; the Nov10 access check above records current temporary-origin gaps. No bit-exact reconstruction assumption authorized discarding scientific copies.
The database retains all 112 permitted lab models; retrieval indexes contain
111 because the existing builder omits the singleton q25-15 decision-head
architecture group (no comparable peer in this lab-only snapshot).
Inventory hashes and outcomes: `$ZOO_DATA/runs/track-a-k0-20261107/`.

## Repositories (public, gated until the P1 submission)

| Repository (dataset)   | Holds                                                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `<owner>/zoo-lab`      | Lab checkpoints in `models/<model_id>/`, each with its `lineage.json`; `tables/models.csv`; `tables/edges.csv`; the recipes; a README |
| `<owner>/zoo-evidence` | Manifests, fingerprints, full-read spectra, run reports, and frozen evaluation inputs                                                 |
| `<owner>/zoo-db`       | Zoo database snapshots                                                                                                                |

`<owner>` is the overseer's Hugging Face account.

## Never uploaded

- **`.env` and tokens.**
- **Raw row blocks and the parent-block cache:** pieces of real weights.
- **Third-party weights.**
- **Gated-family fingerprints** (the pilot's Llama, Gemma and similar
  models): these stay out of the public repositories until the overseer reads
  that family's terms. If they need a copy, keep them in a private repository;
  they are small, and the free private plan is enough.

## Rules

- **Agents may upload** to these three repositories. This is already authorized because
  the overseer approved public, gated storage. Record every upload in
  `RESULTS.md`: what, the file count, the size, and the remote revision.
- **Removing the gate, or making any other repository public, needs an owner decision.**
  - The lab bases are all Apache-2.0, so redistribution is fine.
  - Ablated (refusal-removed) checkpoints stay gated even after release,
    unless a decision card says otherwise.
  - Before removing the gate, check the institution's policy on releasing
    code and data.
- **The upload token** is a fine-grained token, `HF_UPLOAD_TOKEN` in `.env`,
  with write access to these three repositories only. Reads keep using
  `HF_TOKEN`. Neither token is ever printed.
- **Use `huggingface_hub`'s `upload_large_folder`,** which is resumable.
- **A copy counts as durable** only when the remote file list and sizes match
  the local inventory, and the safetensors sha256 values match.
- **Deleting a local copy after a verified upload:**
  - temporary copies (for example in `/tmp`): doer may remove only redundant scratch after verification;
  - anything that may be the only other copy: owner decision.
- **A new server restores the lab** with `snapshot_download` into
  `$ZOO_DATA/lab`.

## Measured storage — 2026-11-07 (Track A)

The temporary-root delivery contains **13 constructed checkpoints, 81 files,
23,803,007,860 bytes** (including 17 safetensors files). All files and sizes,
and every safetensors SHA256, match dataset `RpM-Kinshuk/zoo-lab` at revision
`61b01d947a27a197cc8e63627a6285558d66ecde`. This batch was verified remotely on Nov7; current local recovery gaps are recorded above. Source: `$ZOO_DATA/runs/track-a-k0-20261107/`.
The three designated repositories verified as public with manual gating on Nov7; this audit did not recheck remote access.
Owner: `RpM-Kinshuk`; the Nov7 upload session used the existing `HF_UPLOAD` environment credential directly. That statement does not describe credentials in a fresh session.
All prepared checkpoint, lab-only evidence and filtered database inventories above had verified remote copies on Nov7. Current local gaps and backup-scope limits are recorded in the access section above.

## Verified archive revisions — 2026-11-07

| Dataset | Pinned revision | Uploaded files | Uploaded bytes |
| --- | --- | ---: | ---: |
| RpM-Kinshuk/zoo-lab | `a69530b2930700dba1aa551ddd0fafcc050711f8` | 964 | 192,718,456,977 |
| RpM-Kinshuk/zoo-evidence | `498853257da38151b473beea9fc05acda612aaf6` | 448 | 2,848,505,527 |
| RpM-Kinshuk/zoo-db | `a40c3da43710587ed27f22fcad58b05189cb723a` | 18 | 8,654,697 |

Final remote lists and sizes match exactly, excluding only the repositories'
standard `.gitattributes` files. All 404 safetensors SHA256 values match.
The checkpoint archive covers 139 constructed models, including 27 auxiliary
models whose nonformal status is preserved. Evidence and database archives
cover the 112 permitted formal lab model identities. The mixed Hub snapshot
and restricted/gated-family fingerprints were not uploaded.
At the Nov7 verification, all repositories were public/manual gated and local copies were retained. This is historical verification; use the current access/recovery findings above for local readiness.
Proof and one archive manifest: `$ZOO_DATA/runs/track-a-k0-20261107/`.

Restore layout: the checkpoint dataset stores `models/<model_id>/`; map this
subtree to the desired native lab root when restoring. The original tables
are preserved verbatim, so archived storage paths are documented in the
repo README. The database references evidence relative to
`runs/track-a-k0-20261107/stage-evidence/`; download the evidence dataset into
that location under the new `$ZOO_DATA` to preserve those relative links.
The database payload itself is under `lab/` in zoo-db. Use authenticated
`snapshot_download` with the pinned dataset revisions; no gated access
request or license acceptance is automated.
