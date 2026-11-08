# Track C local integration, 2026-11-08

User explicitly requested merge into main and continued work.
Track C scope commit: `45ba50b` (seven new comparison/evidence files).
Uncommitted inherited heritability edits/files were excluded and preserved.
Merged into main without conflicts as `51aec90`. Final merged-tree suite:
**668 passed**, 16 warnings, 152.72s. The test process exited successfully.
No push or branch/worktree deletion requested or performed.

Initial merged suite: 651 passed, 17 failed. Four distributed cases failed
because sandbox loopback networking was unavailable; loopback-enabled rerun
passed the distributed test file. Thirteen Heretic tests failed importing
datasets due to orphan distribution metadata, independently reproduced.
The unchanged installed fsspec module and valid metadata both say 2026.2.0;
`fsspec-2026.3.0.dist-info` contained only `direct_url.json` and `REQUESTED`,
without METADATA or package code. Under approved filesystem escalation,
that orphan was moved reversibly to `/tmp/track-c-fsspec-orphan-20261108`.
No package code/version changed, no package was installed, and no dependency
pin changed. Fresh metadata lookup reports 2026.2.0; datasets import passes.
To undo only this repair, move that backup directory back to the active
esd_ind site-packages as `fsspec-2026.3.0.dist-info` (restores the broken state).

Final verification command, from Zoo-Keeper with esd_ind active:
`PYTHONPATH=../Model-Research:../Model-Zoo GLOO_SOCKET_IFNAME=lo python -m pytest tests -q --tb=short`
Runs with local networking allowed; no tests loosened or skipped.

Continued K7 with bounded two-model writer diagnostics, protocol before reads.
See `2026-11-08-k7-writer-diagnostic.md`. These follow-up diagnostic and
integration notes are included in the subsequent user-requested local merge.
