# K5/C2 quantization exact mark — declared before tests, 2026-11-08

Scope: all twelve official quantization_sim children. Controls: all sixteen
SFT/DPO children plus the first fourteen LoRA children in sorted official-model
order, thirty total. Freeze IDs before tensor inspection. Known construction
bits select the per-case limit; controls are tested at the most permissive 8-bit
limit. Do not infer a configured bit count from truth to improve detection.
This tests the value-count necessary mark, not a sufficient proof of quantization.

Detector: on every row of every floating 2D tensor, count distinct STORED values;
flag as grid-compatible iff every row has <=2^bits distinct values (256 for
controls), and at least one matrix was checked. Stop a negative scan at its
first counterexample, recording tensor/row/count. Whole-row blocks of 128 keep
scratch bounded. No sampling, no approximate equality, no noise-fitted tolerance.
BF16 casting cannot create extra distinct values from a quantizer level.

Reproduction: use existing lab.quantization.simulate_quantization on the pinned
parent row blocks and cast to bf16 exactly as the writer did. Require bit-exact
stored child agreement for every floating tensor, respecting documented ties
and unchanged legacy masks. Nonfloating tensors require exact equality.
No rank or update-size claim. Quantizer input pins are checked against hashes.

PASS C2 quantization lab test iff 12/12 grid-compatible, 0/30 controls flagged,
and 12/12 parent reproductions match exactly. A miss is FAIL; missing files or
unresolved aliases are NOT MEASURED, never passed or silently omitted.
Record matrix/row coverage. Parent presence is verified construction evidence;
this does not evaluate source retrieval or its leave-parent-out calibration.

At most 47 distinct models, CPU only, expected <30 minutes, stop by110 minutes.
No Hub downloads, no blind data, no threshold/split/truth changes. Practical
claims still need K7. New outputs only; leave originals and production unchanged.
