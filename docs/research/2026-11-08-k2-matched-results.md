# Published comparisons on B's quantization cases — 2026-11-08

The two remaining weight-based baselines now ran on exactly B's fixed twelve
quantization queries and candidate pools, covering four lab anchor families.
This closes a comparison gap for that edit group; it is not the whole benchmark.

| Method | Matched comparisons | Source present | Source absent | Meaning |
| --- | --- | --- | --- | --- |
| Training Leaves Traces, full canonical MLP path |126/126|True parent uniquely ranks first7/12|Rankings of remaining relatives saved; no missing-parent rule|Shared ancestry score, no calibrated parent/type/direction verdict|
| modelDNA0.1.0, native fast mode |126/126|12/12 same-family-unresolved|12/12 same-family-unresolved|No definite parent answers in either pool; all four aggregate signals present|
| B exact quantizer replay, prior result |Same pools|12/12 correct answers|12/12 abstentions|Construction-specific lab quantizer result; does not imply general quantizer/Hub transfer|
| Trace/Spectral Signatures, earlier native adaptations |Not run on these pools|not_run|not_run|Earlier small L12 results are not matched coverage here|
| Witness Overlap |not_run|Ordinary-fine-tune direction needs witness triples|not_run|Next priority; quantization pools alone do not test ordinary-fine-tune order|

There are no independent ancestry negatives in these pools. B's nonquantized
controls remain relatives, so ancestry AUROC/FPR cannot be computed fairly.
No truth, split or threshold changed, and no claim advanced.

Costs, complete rankings/verdicts, protocols and input hashes:
- `runs/track-c-k2-residual-20261108/REPORT.md` (195.695743s;28,003,270,656 logical
  weight bytes plus76,403,189,056 verification bytes).
- `runs/track-c-k2-modeldna-20261108/REPORT.md` (124.358342s;8,680,734,831 native
  source bytes plus76,442,069,366 verification bytes).
Both CPU8threads; they overlapped, so timings are not isolated speed comparisons.
No GPU was admitted: driver access unavailable on this host. Algebra/package
checks17passed; one core review and a reproduced zero-gate fix recorded in run.
modeldna0.1.0 installed from published wheel with --no-deps; dependency pinned.

For Nov14: these lab comparisons are usable now. The existing K7 Hub sources
remain declarations with unknown construction-input revisions, not verified
truth. No new Hub weight result or completed positive/negative control set is
claimed. Multi-family Hub fingerprints, complete controls and matching results
remain open for Dec1–5. Historical uncertainty permits descriptive comparison;
it restricts causal and false-parent claims. B's current detection handoff leaves
matched writer controls/scientific cutoff open; C will not fit a new detector.
Next dependency: C supplies native matched cases for remaining edit groups and
witness triples; B supplies the agreed complete real-model control/test selection.
