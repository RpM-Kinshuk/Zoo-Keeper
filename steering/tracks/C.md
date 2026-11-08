# Track C: baselines and truth sets
Brief of 2026-11-08 read; D3/D5 approved; no largest/first-search-size claim.

**Now:** requested local integration complete; K7 real Hub diagnostic recorded.
**Results since the last steward run:**
- Prior seven-file C batch merged into main as 51aec90; merged suite 668 passed.
- Fixed test environment: loopback allowed; orphan fsspec metadata quarantined,
  valid installed 2026.2.0 code/metadata unchanged; backup retained in /tmp.
- Four selected writer-block deltas measured on pinned Qwen 7B parent/candidate.
- Rank-one residuals small, but cast-noise bound exceeds whole delta in every case:
  consistency check uninformative; no C2 pass or verified construction edge.
- v2 payload: 46,137,344 weight bytes + 107,782 metadata/header bytes.
- Report: `Zoo-Keeper/docs/research/2026-11-08-k7-writer-diagnostic.md`.
- Earlier K7: 18/20 repository names pinned; 13/14 declared edges have parent snapshots.
- One mergekit recipe; two refusal-removal candidates; official GGUF build gaps retained.
**Next:** informative noise controls with B; complete fingerprints and construction
provenance; more merge recipes. Then remaining K2 priority methods.
**Blocked or needs:** producer experiment→public weight mapping; GGUF build inputs.
B evidence: `tracks/C-to-B-k7-evidence.md`; existing tier-loader issue resolves on main.
Class 1: bounded two-model row reads, no raw blocks retained, truth stays declared.
Diagnostic/integration notes and C result updates included in requested follow-up merge.
Inherited heritability edits and unrelated main changes preserved; nothing running.
