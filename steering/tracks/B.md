# Track B: lab methods and P1 evaluation
Brief of 2026-11-08 read; K5 first. No claim advanced.

**Now:** integrated as uncommitted main files per owner; nothing running.
Class 1: no Git history changes; track-b originals preserved.

**Results since the last steward run:**
- K5 quantization: 12/12 grids, 0/30 control flags, 12/12 bit-exact parent replays.
  [Report](../../../runs/track-b-k5-quantization-20261108/REPORT.md).
- K5 quantization parent-out: 12/12 correct present, all absent cases abstain.
  Pooled error 0/12 answers vs ≤0.05; present coverage100% vs ≥50%.
  [Report](../../../runs/track-b-k5-parent-out-20261108/REPORT.md).
  Limited exact-mark subset; no guarantee, all-edge calibration or Hub claim.
  Effective coverage/ties checked; all original weight pins match.
- K3 coefficient tests pass18 cases; broader parent/direction claims untested.
- K4 repeats exact; additivity violations do not establish non-tree ancestry.
- K14 behavior failures unchanged; full-network robustness remains untested.

**Next:** K5 adapter/refusal/TIES/DARE marks and edge typing.
Bounded K4/K14 gaps follow without displacing K5; K8 freeze review with A.
K12 final freeze awaits Nov25 decisions; K9 stays exploratory.

**Blocked or needs:** no K5 external blocker. Integration664 passed; four Gloo
cases pass in19-test rerun outside socket sandbox. /tmp metadata/path workaround
used for tests; orphan fsspec metadata persists in installed environment.
Source work now in main uncommitted; no packages changed; artifacts external.
