# K5 quantization parent-out protocol — 2026-11-08, before execution

Apply the previously proposed unique exact-replay rule, without fitting.
Twelve quantization queries; freeze IDs from the completed quantization run.
Candidates for each query: that run's SFT/DPO/LoRA controls from the same anchor,
the other quantized children in the anchor, and its native base. Include the
verified immediate parent; require pinned identity, full tensor coverage and
compatible shapes, allowing only documented tied aliases. No candidate inferred
from a score; no names/timestamps used to select the answer.

Replay the query's specified quantizer on each candidate using existing
simulate_quantization; require bit-exact whole-child agreement, bf16 cast and
unchanged tensors included. First mismatch may reject early. Unsupported or
unreadable candidates make that query incomplete; never count them as negatives.
Freeze candidate lists before reading weights. No truth/split/threshold changes.

Answer iff exactly one complete candidate reproduces the child. Zero or multiple
matches abstain. Score present, then remove the true parent and reuse the same
candidate results. Report each setting's answers/errors, pooled errors per
answered query, present coverage, ambiguous groups, and incomplete queries.
PASS only if all queries complete, pooled false parents per answer <=0.05,
and present coverage >=0.50; additionally report errors separately for each
setting. Zero answers cannot meet the coverage target. This is only the C3
quantization subset, not all-edge calibration or independent statistical evidence.
No order or immediate-parent proof follows from replay alone.

Noise allowance: ZERO, stored-bit equality; rounding collisions cause ambiguity
or wrong answers, not threshold adjustment. At most 46 distinct models,
CPU only, expected <30 min, abort after110 min. No spectrum/rank estimation.
Artifacts outside git; originals unchanged. No practical claim without K7.
