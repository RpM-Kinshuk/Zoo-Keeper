# Four matched relatedness methods: quantization neighborhoods
Scientific purpose: compare existing published-method adaptations on exactly B's
quantization queries/pools; distinguish family similarity from immediate source.
The cached spectral run adds no weights and does not repeat the old L12 study.

| Method | Scored/planned pairs | Complete candidate pools | Unique true-source highest rank |
| --- | --- | --- | --- |
| Trace-native |126/126|12/12|3/12|
| Spectral-Signatures-native, corrected guard |126/126|12/12|0/12|
| Training Leaves Traces canonical MLP, earlier matched run |126/126|12/12|7/12|
| modelDNA0.1.0 native fast, earlier matched run |126/126|12/12|No definite parent answer; all source-present/removed pools unresolved|

All46 source identities match B's weight-file hashes through the catalog's
manifest; full-weight fingerprint repo/revision headers and paired spectrum
basenames match. SVD/full storage required. All existing native module mappings
and previously exposed lab identities retained. No split or score threshold fitted.

These are ancestry/relatedness methods, not calibrated parent detectors. All
candidate pools contain relatives; ancestry AUROC/FPR and calibrated source/type/
direction decisions are unsupported. Ranking errors are not false-parent error
rates. Spectral Signatures has the true source tied for top in one pool; it is
not uniquely first. Four pools have tied top candidates. The result demonstrates
limited source discrimination here, not a general ancestry-method failure.
B's known-quantizer replay12/12 result is a different construction-specific test;
it does not establish general quantizer or real-model transfer superiority.

## Guard correction and original result
The retained initial run scored only91/126 spectral pairs, with3/12 complete
pools. Three two-bit profiles had a few exact zero singular-energy eigenvalues
below a strictly positive fixed-half tail threshold. Strict positivity of the
ENTIRE spectrum was an erroneous input guard in our adapter/test specification.
The fixed-half formula never logs those lower zeros. Independent core review
confirmed the correction. This is not a claim that the upstream method cannot
handle quantization, and no undefined score was silently changed into zero.

The corrected implementation rejects negatives/nonfinite values, keeps every
sample (including zeros), keeps n and k=floor(n/2), and requires the SAME
selected threshold values[-k-1] positive. Zero selected threshold and flat tails
still abstain. No tail trimming, fallback or fitted threshold. Existing unit
example [0,1,2,3] was explicitly corrected to1+2/log(6), and the genuine invalid
zero-threshold example [0,0,1,2] retained as rejection. New regression
[0,0,1,2,4]=>1+2/log(8) verifies zeros are not trimmed. Both new tests RED,
then GREEN; entire baseline package31passed. This is a class1 mathematical
implementation/test-spec repair, not a loosened scientific acceptance target.
Legacy module/test/driver snapshots,91-pair results and original protocol
text remain under runs/track-c-k2-matched-spectra-20261108. The corrected run
is separate; Trace was not rescored. No original result was overwritten.

## Cost and limits
Cached spectral payload 32,808,960bytes per run;
initial source verification scan 60,357,261bytes;
corrected verification scan 60,358,723bytes.
Initial two-method analysis 5.016714s;
corrected spectral-only analysis 14.590953s, CPU8threads.
Zero new model-weight bytes. Historical extraction cost is UNKNOWN, not zero;
these cached timings cannot be compared as fresh-weight cost against modelDNA.
Shared-host observations, not isolated hardware speed claims.

Run records: runs/track-c-k2-matched-spectra-20261108/ and
runs/track-c-k2-matched-spectra-corrected-20261108/ (results, input hashes,
profile audit, every score, source-present/removed rankings and explicit gaps).
This covers one edit group, not the whole lab or verified Hub benchmark.
No claim promoted. Next use is the common evidence table, not another same-data
method sweep. B's witness review defers the correlated q/v contrast; no new
witness weights or scores are justified. Real-model comparisons proceed separately.
