# Data: storage and publication on Hugging Face

**Decision D1 (2026-11-07):** public Hugging Face storage. The repositories
stay **gated with manual approval** until the P1 submission. Gated
repositories are public, so the storage is free, but files download only
after the overseer approves a request. That keeps unreleased results and
ablated models from circulating before the paper. If the overseer says "fully
open", gating is skipped.

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
drift-equivalence findings are preserved. All local copies remain; no
bit-exact reconstruction assumption was used to discard anything.
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

- **Agents may upload** to these three repositories. That is class 1, because
  the overseer approved public, gated storage. Record every upload in
  `RESULTS.md`: what, the file count, the size, and the remote revision.
- **Removing the gate, or making any other repository public, is class 3.**
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
  - temporary copies (for example in `/tmp`): class 2;
  - anything that may be the only other copy: class 3.
- **A new server restores the lab** with `snapshot_download` into
  `$ZOO_DATA/lab`.

## Order of work (`PLAN.md`, K0)

1. **The overseer**, about 10 minutes, no cost:
   - create the three dataset repositories;
   - set each to public, with gated access and manual approval;
   - create the fine-grained upload token, and add it to `.env`;
   - tell track A the account name.
2. **Track A** uploads the 13 checkpoints that are only in `/tmp` (23.8 GB).
3. Then the other irreplaceable models, then everything else built in the lab,
   then the evidence, then a database snapshot.
4. Track A records the remote revisions, and the steward marks the lab as safe on the board.

## Measured storage — 2026-11-07 (Track A)

The temporary-root delivery contains **13 constructed checkpoints, 81 files,
23,803,007,860 bytes** (including 17 safetensors files). All files and sizes,
and every safetensors SHA256, match dataset `RpM-Kinshuk/zoo-lab` at revision
`61b01d947a27a197cc8e63627a6285558d66ecde`. This batch is durable; all local
copies remain. Source: `$ZOO_DATA/runs/track-a-k0-20261107/`.
The three designated repositories now verify as public with manual gating.
Owner: `RpM-Kinshuk`; this session uses the existing `HF_UPLOAD` environment
credential directly (the planned `HF_UPLOAD_TOKEN` name is unset).
All prepared checkpoint, lab-only evidence and filtered database inventories above now have verified remote copies.

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
All repositories remain public/manual gated, and all local copies remain.
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
