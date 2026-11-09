# Track B: lab methods and P1 evaluation
Brief of 2026-11-08 scientific necessity review read; D6 approved.

**Now:** joint source development test completed; [report](../../../runs/track-b-joint-source-20261109-v3/REPORT.md).
No final rule adopted; B work integrated into main, uncommitted.
No jobs/GPU reservations remain; esd_ind CPU1thread on voyager.

**Results since the last steward run:**
- All72/72 fixed candidate pairs evaluated:33 inherited,39 new;4 exposed queries.
- fp32_input source-present: useful4, wrong0, abstain0 /4.
- fp32_input source-removed: useful0, wrong0, abstain4 /4.
- bf16_input source-present: useful4, wrong0, abstain0 /4.
- bf16_input source-removed: useful0, wrong0, abstain4 /4.
- These conditional results do not establish general precision or unknown-history
  real-model transfer. Heretic positive-query and other edit detectors stay open.
- Original v2 incomplete record preserved; exact4 missing checkpoints restored.
- GPU test:0.257s CPU vs3.737s GPU on fixed SVD; released GPU, used CPU.
- Runtime1439.30s; SVD3.50s; hashing1431.93s dominates.
- Quantization12 and coefficient18 successes remain closed; no collection justified.

**Next:** review applicability and independent source/control validation;
no automatic panel/cutoff expansion. Nov14/Nov21/later deadlines retained.

**Blocked or needs:** final noise/detector approval and generalization evidence.
Class1: main checks694pass;2 known SFT failures retained; new B tests pass.
Class2: distinguish12 closed source cases from4 conditional development cases.
No truth/split/threshold changes, commits, merges or pushes.
