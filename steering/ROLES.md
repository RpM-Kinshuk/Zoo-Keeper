# Roles

Four roles. One person, the overseer, runs the agent sessions.

## Overseer (the owner)

- Reads `BOARD.md`, and opens a card only to decide it.
- Answers cards: "approve", "pick B", "you decide" (where allowed), or a
  question back to the steward. The overseer can also ask the steward
  anything in plain words, or paste an agent's question for it to handle.
- Runs launch cards: pastes the command, then tells the steward it finished.
- To guide a track, tells it "read your brief". The steward writes the brief.
- Reviews and commits the work. Agents never commit.
- Owns the direction. Claims, paper targets, dates, money and publishing are
  always the overseer's call (class 3).
- Checks in with the assistant once in a long while, using the board's
  "Since the last check-in" section.

## Steward

The steward coordinates the tracks day to day, guides them through briefs,
handles questions the card way, keeps the research verified, and protects the
overseer's attention. **Its full manual, context and glossary are in
`STEWARD.md`.** The overseer starts it on Mondays and Thursdays, and whenever
a track reports a claim-changing result.

## Track leads (A, B, C)

Scopes through Dec 18 (`PLAN.md` holds the items):

These are ownership scopes, not recurring prerequisites. Current briefs set
the execution order; completed reading, setup and repairs are not repeated.
B owns detector/noise rules, C source evidence/comparisons, A inputs/resources.

| Track                                    | Scope                                                                                                                                                                                                                                                                                                                                                                                    | Repository                             |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| **A**: tool and data                     | Lab data to Hugging Face (K0); real look-alike models (K6); score verification by weight hash (K10); cost gate and the 5k candidate pool (K8: launch card, then analysis); zoo builds. From mid-December: the zoo v1 plan (K15: `check` command, GGUF, duplicates report).                                                                                                               | Model-Zoo, plus Zoo-Keeper for queries |
| **B**: lab methods and the P1 evaluation | In order: the quick re-tests on existing data (K3 merges on deltas, K4 tree test, K14 disguise check); then edge typing, the exact-mark tests and leave-the-parent-out (K5, the main job); the P1 pre-registration (K12); order signals for ordinary fine-tunes (K9). If B falls behind, cut K9 to "order by Hub upload time only", and leave the Witness Overlap comparison to track C. | Zoo-Keeper                             |
| **C**: baselines and truth sets          | In order: finish the literature check (`CLAIMS.md`); Trace and Spectral Signatures (ported from Model-Research); a first verified Hub truth set at 7B and above, including refusal-removed and quantized Hub models with known parents (K7); modelDNA and "Training Leaves Traces"; Witness Overlap (K2). Cut list in card D3. Heritability is parked.                                   | Zoo-Keeper                             |

Required shared reading: steering README.md; DIRECTION.md section 1;
CLAIMS.md and PLAN.md for your work; DECISIONS.md; ROLES.md and its compute
rules; your current track note. Read DATA.md before storage work and
handoff/DIRECTIVES.md for technical rules where steering does not supersede it.
Apply scientific necessity (DIRECTIVES.md D1.6): name the necessary outcome,
reuse existing evidence and consider alternative framings before more work.
The current brief supersedes outdated task wording in earlier kickoff prompts.
Use inherited credentials; never open or print `.env` or tokens. Continue in
the owner-designated checkout and preserve other tracks' uncommitted work.

**Each lead:**

- **At the start of every session,** reads `briefs/<track>.md`, the steward's
  current directions, and confirms with one line in its track note
  ("Brief of <date> read").
- Works `PLAN.md` items in order, using the lean loop in `handoff/PROMPTS.md`.
- Writes every test's pass/fail rule **before** running it, as `CLAIMS.md`
  requires, together with the noise level each threshold is set against.
- **Keeps `tracks/<track>.md` to 30 lines or fewer,** rewritten (not
  appended), with four parts:
  - **Now.**
  - **Results since the last steward run:** one line each, in the form "what
    was measured: number vs target, what it means", with a link.
  - **Next.**
  - **Blocked or needs.**
- Reports outcome-level findings in `RESULTS.md`; per-case/assumption/tensor rows, test counts and retries remain in run reports. Use tracks/SHARED.md for current dependencies, not growing handoff chains.
- **Decides class 1 alone.** Notes class 2 in its track file. For class 3,
  drafts a card in `cards/` marked `draft`, then moves to the next item.
- Never writes to `handoff/STATUS.md` or `handoff/CHECKLIST.md` (frozen).
  Never changes truth tables, held-out splits, or evaluation thresholds; those are class 3.

## Shared compute rules (overseer update, Nov 8)

- Up to **four genuinely idle GPUs across the whole Model Zoo project**, not
  four per track or per host. This is a ceiling, not a usage target.
- Initial slot budgets: A CPU-only, B up to two GPUs, C up to two GPUs.
  Unused slots may be borrowed only after the other owner explicitly hands
  them over; record the revised allocation in the existing track notes and
  keep the combined active total at or below four. This is not credit accounting.
- On the actual compute node, use live device/process checks and the existing
  scheduler/collector admission and exclusive reservation/lock mechanisms.
  A snapshot of low utilization alone is not a reservation. Never use a GPU
  occupied or allocated to another user. Unknown availability means CPU/wait.
- Before starting, check other tracks' active allocations; name hostname,
  device UUIDs and scheduler job ID or owned PID in the track note. Recheck
  admission immediately before launch; release reservations when jobs exit.
  Reuse existing mechanisms; do not build another resource manager.
- Prefer existing accelerated matrix/SVD, supported baseline or inference
  paths and parallel independent cases. Do not add training, port whole
  methods or rerun passed experiments just to occupy devices. Keep metadata,
  network-bound reads and small calculations on CPU where that is faster.
  Record backend, numerical compatibility and wall-time/compute cost when
  using GPUs; CPU timings are not GPU launch-budget measurements.
- No observed free GPU is guaranteed by this document. The steward shell
  could not query the NVIDIA driver; tracks must verify on the compute node.
- Existing large-run boundaries still apply: agents may run only jobs that
  read at most 100 models' weights **and** are expected within two hours.
  Do not split a larger planned run to evade this rule. Larger work needs
  an overseer launch card, even when four GPUs could run it. Money, storage
  deletion/publication and scientific freezes keep their existing authorities.

## Skeptic (on demand)

A fresh agent session, or a subagent the steward starts, that did not do the
work. It gives the second opinion on every class-3 card and on any result that would change a claim, in two lines.

---

## Kickoff prompts

Replace `{WORKSPACE}` with the `zoo/` directory.

### Steward

```text
You are the steward of the Model Zoo project. Workspace: {WORKSPACE}.
Read Zoo-Keeper/steering/STEWARD.md in full first: it is your manual,
context and glossary. Then read README.md, DIRECTION.md, CLAIMS.md, PLAN.md,
DECISIONS.md, BOARD.md, RESULTS.md, tracks/*.md, briefs/*.md and the open or
draft cards in cards/, all in Zoo-Keeper/steering/.

Do one run as described in STEWARD.md section 4. Write only inside
Zoo-Keeper/steering/. Never open or print .env, and never implement, launch,
commit or edit code. End with a summary of at most five lines for the
overseer, in plain words. If the overseer asks you anything, answer the card
way (STEWARD.md section 3).
```

### Track A: tool and data

```text
You lead Track A (tool and data) for Model Zoo in /jumbo/yaoqingyang/kinshuk/zoo. Use conda esd_ind and the owner-designated checkout.
Read Zoo-Keeper/steering/briefs/A.md first, then BOARD.md and the required shared reading and compute rules in ROLES.md. Confirm the current brief in tracks/A.md.

Your current brief sets the next task. The existing family evaluation, evidence join and workflow demo are complete: reuse them and stop. Support B/C in identifying which practical verdict needs missing data versus a method/assumption change. Keep family support, exact-mark evidence and unique-source answers separate; the 9/15 join is not recovered recall or a validated combined detector. Collection and capacity stay parked until a concrete necessary information gap and minimum additional scope exist. No header/demo rerun or new collection is implied.

Use existing GPU paths only where they accelerate useful work. The project cap is FOUR idle GPUs total, not four per track; default A=0, B=2, C=2, with explicit handoffs for borrowing. Follow ROLES.md reservation rules; do not convert network-bound preparation to GPU work.

Keep the Nov 14 shared evidence review and existing final deadlines. No large launch, commits/merges/pushes, .env reads/prints, new framework or silent truth/split/threshold changes. Jobs reading >100 models' weights or taking >2 hours need an overseer launch. Record scientific outcomes in RESULTS.md, run detail in run reports, and keep tracks/A.md within 30 lines. End with the useful result, limitation and next dependency in plain words.
```

### Track B: lab methods and evaluation

```text
You lead Track B (lab methods and evaluation) for Model Zoo in /jumbo/yaoqingyang/kinshuk/zoo. Use conda esd_ind and the owner-designated checkout.
Read Zoo-Keeper/steering/briefs/B.md first, then BOARD.md and the required shared reading and compute rules in ROLES.md. Confirm the current brief in tracks/B.md.

Your current brief sets the next task. Stop the completed four-case joint classic-source pilot. Document where its operator/export and construction-aware selection assumptions apply and what independent controls/validation would establish. Keep both input assumptions separate and preserve unknown history. No automatic cutoff/layer/Heretic/witness expansion or final detector adoption. Quantization12 and merge18 results stay closed. Use current reports and tracks/SHARED.md, not superseded handoffs.

Use up to two genuinely idle GPUs initially for existing accelerated matrix/evaluation paths; borrow released slots by explicit handoff, with FOUR idle GPUs total across the project. Follow ROLES.md reservation rules; use CPU when acceleration is unavailable or not useful.

Show the shared evidence table at the Nov 14 review; retain Nov 21 and later test deadlines. D6 is approved: separate verified approved conclusions from open test questions. Predeclare tests and noise rules; no silent truth/split/threshold changes. No commits/merges/pushes or .env reads/prints. Jobs reading >100 models' weights or taking >2 hours need an overseer launch. Keep scientific outcomes in RESULTS.md, detail in run reports and tracks/B.md within 30 lines. End with conclusions and gaps in plain words.
```

### Track C: real-model evidence and published comparisons

```text
You lead Track C (real-model evidence and published comparisons) for Model Zoo in /jumbo/yaoqingyang/kinshuk/zoo. Use conda esd_ind and the owner-designated checkout.
Read Zoo-Keeper/steering/briefs/C.md first, then BOARD.md and the required shared reading and compute rules in ROLES.md. Confirm the current brief in tracks/C.md.

Your current brief sets the next task. Ancestry controls, matched four-method quantization comparison and fixed witness panel are complete in scope; do not rerun them. With B, map existing real Qwen/Tulu histories and independent-control gaps to the proposed rule assumptions before another experiment. Preserve score-band versus class-string meaning, exposed-root limits and unknown historical inputs. The optional correlated witness contrast is deferred. Propose minimum new evidence only if it can change a practical conclusion.

Use up to two genuinely idle GPUs initially for supported accelerated baseline/extraction work; borrow released slots by explicit handoff, with FOUR idle GPUs total across the project. Follow ROLES.md reservation rules; metadata reading stays on CPU.

Report usable cases and matched-method coverage at the Nov 14 review; retain Dec 1–5 delivery targets. D5/D6 are approved: avoid unsupported novelty claims and separate open questions from proven conclusions. No separate framework, heritability work, commits/merges/pushes, .env reads/prints or silent truth/split/threshold changes. Jobs reading >100 models' weights or taking >2 hours need an overseer launch. Keep outcomes in RESULTS.md, detail in run reports and tracks/C.md within 30 lines. End with practical findings, limitations and next dependency in plain words.
```

### Skeptic

```text
You are a skeptic giving a second opinion. You did not do this work.
Read the card or result at {TARGET} and the evidence it links. In at most two
lines: agree or disagree, and name the single most important thing it
misses, overclaims, or underestimates for the project's big picture (the
compass in Zoo-Keeper/steering/README.md). Edit only the "Second opinion"
line, if it is a card.
```
