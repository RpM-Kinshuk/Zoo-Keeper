# Full workflow migration

Owner instruction: replace A/B/C with one doer and independent review immediately, without a trial; remove redundant material while retaining research goals, observations, useful work and reproducibility.

- [x] Audit branch ancestry, changed/untracked files, ignored files, running processes and environment imports. All four track heads are ancestors of their main checkout; no merge is required.
- [x] Preserve the two unique research notes in the main checkout, byte for byte before annotation. Other untracked research scripts/protocols/tests match main exactly; old directive additions are already present, and old track status is superseded. Ignored worktree files contain only Python/pytest caches.
- [x] Replace track coordination and duplicated rules with one ordered research checklist and one doer/reviewer workflow; preserve scientific thresholds and owner decisions.
- [x] Remove obsolete handoff/plans/track files, audited worktrees and disposable caches; retain source needed to reproduce evidence.
- [x] Check links, package imports, relevant offline tests and an independent review; results below.

Baseline revisions: Zoo-Keeper `170c068e02024378e914f8ed9bc99db7bc584f52`; Model-Zoo `e2c5c0b64ef23d2043def2bffe8bd21ae64ea971`. No commits, merges, model experiments, publication or package changes are part of this migration. `.env` is never read.

Preserved unique notes: `docs/steering/K5-applicability-validation.md` and `docs/research/2026-11-09-real-history-applicability.md`. Their old admission/stop instructions are historical and superseded by documented-case policy D8.

Environment audit: `esd_ind` imports `evidence` and `zoo` from the main Model-Zoo checkout; Zoo-Keeper modules run from its checkout. No live model jobs were visible in this session process namespace; worktree cache inventories contain no weights or run outputs.

Removed 43 obsolete coordination/plan/helper files (plus the parked workshop draft relocated as a methods/audit reference). Entire old handoff, permanent track/brief/card directories and completed implementation-plan directories are retired. The isolated macOS access helper had no callers or request queue; scientific reproduction drivers remain.

All four audited track worktrees and their four merged local branches were removed; no unique commits, code, scientific outputs or checkpoints were discarded. No source import or live symlink depended on these checkouts. Historical absolute worktree strings in raw records remain provenance, not current commands.
docs/steering/K5-applicability-validation.md original SHA256 c245d04f4c3181d7400721a60752939596fcb5d1cf3274a4fc559a440db4d162
docs/research/2026-11-09-real-history-applicability.md original SHA256 5c636a733c21c369f4947f4184027591e7f96f83a0c125b2e6ab88dbe473f2e9

Independent review: no critical findings. Important findings fixed: restored permissive/gated lab-base restrictions; corrected main-checkout/import paths and retired lab handoff filename while retaining reference labels; removed stale C ownership; clarified genuine-idle GPU admission overrides historical80% figures. Minor findings fixed: retained parked/cut scope and distinguished measurement modes from scientific evidence tiers.

Initial focused test invocation:100 passed,3 failed because the preserved Model-Research baseline package was absent from PYTHONPATH. Trace source imports explicitly require that sibling; no code was changed. The current main-checkout runbook now includes the dependency, and the same checks are rerun with it.

Verification complete:103 focused Zoo-Keeper query/lineage/baseline tests passed in5.05s with the required sibling source paths;14 Model-Zoo evidence/zoo compatibility/schema tests passed in3.92s. GPU visibility was disabled and offline flags set. No model measurements, downloads or package changes were run. Current workflow/reference Markdown links resolve, git diff --check passes in both repositories, only main worktrees remain, and the old handoff/track/brief/card/plan directories are absent. These focused checks are not a claim that every project test or scientific hypothesis passed.

Cache cleanup: removed 30 disposable Python/pytest cache directories (291 files; 4491208 bytes), excluding datasets, scientific run directories, model/download caches and Git. No scientific logs/checkpoints were deleted: the retained records support current observations and reproduction.

Next research action: CHECKLIST.md item1, the four existing realistic directional edits and normal-training controls. This migration itself produces no new research result or claim promotion. Changes remain uncommitted for owner review.

Nov10 data-continuity follow-up: DATA.md and DATA-INDEX.json now provide18 checked access entrypoints, explicit ZOO_DATA setup,498 readable formal measurement references and pinned recovery instructions. A read-only database query succeeded. All1349 remaining archive-stage files size-match; four Heretic weight files are local but24 companion files need recovery, and nine other temporary-origin checkpoints were not found at known localpaths. Historical Nov7 archive verification remains distinct from current local readiness; no new remote check/download/model scoring occurred. Independent review matched all18 entries and10 small-file hashes, the inventory/recovery counts and credential-safe instructions; two stale “local copies kept/current remote” phrases were corrected. All shared changes remain uncommitted.
