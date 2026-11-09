# D9: Test scaling with pools of same-shape models, not a random mixed roster

Status: decided (overseer, Nov 9). The run itself still needs a launch card
(L1), which track A prepares by Nov 12 and the overseer launches.

**The question.** As the number of same-architecture candidates grows (10 →
100 → 1,000+):
- do family answers stay correct?
- how often is a relative wrongly named as the parent?

This is claim C1's "at scale", and the field's main open question.

**Why the old plan failed.** The 5,000-model roster was a random mix of shapes.
Our fingerprints only compare models of the same shape, so each query had only
0–252 real rivals.

**The design:**
1. **Choose the bases (metadata only).** For each lab base (Qwen2.5-0.5B,
   Qwen3-0.6B, SmolLM2-360M, OLMo-2-1B), count the public models with the
   same architecture: declared derivatives plus same-shape models that
   declare nothing. Use the hub-stats dump and configs. Pick the **two** bases
   with the most.
2. **Build the pools.** All readable same-architecture models per base,
   capped at about 2,000 each. GGUF-only, GPTQ/AWQ and pickle-only models are
   counted as coverage gaps, not silently dropped.
3. **Queries:** our lab models for those bases (exact truth), plus the
   documented real cases from D8.
4. **Measure** false family matches and wrong-parent answers per query, at
   pool sizes 10, 100, 1,000 and "all", with several random subsamples each.
   - Bonus: the share of same-shape models that look derived but declare no
     base (a C5 measured-versus-declared number).
5. **Stop** when two bases are done, or when the curve has clearly flattened.

**Cost estimate,** for track A to confirm in the launch card: a partial read
of a 0.5B model is about 0.1 GB. So 2 bases × about 2,000 models is about
200–400 GB read over a few hours, and roughly 10 GB of fingerprints.

**Big-picture effect:**
- Claims: decides whether C1 can say "at scale".
- Papers and dates: the scaling figure for P1. Results by about Dec 10.
- Real-world use: compliance checks meet exactly this many look-alike
  siblings.
- Cost: bandwidth and a few CPU hours; no GPU needed.
- Risk: fewer public same-shape models than hoped. Then report the largest
  pool reached.

**Can it be undone?** Yes. The cost is only bandwidth and time.
