# Technical and scientific references

Start with [the current direction](../steering/DIRECTION.md), [one checklist](../steering/CHECKLIST.md) and [directives](../steering/DIRECTIVES.md). This folder is reference material, not another work queue.

- [reference/CONTRACTS.md](reference/CONTRACTS.md): current interface names, compatibility, measurement conventions and known traps.
- `operations/`: usable commands and technical methods. Dated examples/completion counts describe their original runs; current observations are in steering/RESULTS.md.
- `steering/`: retained scientific protocols/development notes and the active K12 evaluation draft. Original protocols stay pinned; old private-record/stop instructions are historical and superseded by D8 and the current checklist.
- `research/`: primary-source checks, matched comparison results, applicability audits and parked study methods. Dates identify evidence, not future task promises.
- `training-launches.md` and `l12-evaluation.md`: original training/evaluation provenance; not pending launch queues. Reuse completed artifacts rather than replaying commands by habit.

Finished implementation plans and superseded source plans/handoffs were removed during the owner-approved full migration. Their original text remains in Git at Zoo-Keeper170c068 and Model-Zooe2c5c0b. References inside frozen records describe their original layout; use that revision when reproducing them. Raw run reports, construction records, pinned code/protocol hashes and original failed results remain outside Git under `$ZOO_DATA`.

Existing experiment drivers in `scripts/` are retained where they reproduce scientific evidence; their task-like names are historical identifiers, not current priorities. The obsolete macOS access-request helper had no queue or callers and is removed; access remains manual through the owner.
