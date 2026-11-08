# K7 first pinned Hub evidence, 2026-11-08

Metadata collection tested; K7 verified truth/fingerprints remain incomplete.
Protocol: `2026-11-08-k7-protocol.md`, written before collection. No frozen truth/split/threshold changed.

Inspected 20 selected repository names: 18 pinned, 2 not found. The pins include
OLMo and Tülu post-training stages at 7B/8B, Qwen 7B parent/instruction models,
two refusal-removal candidates, the official Qwen GGUF release, and a Beagle merge.
Names are not independent models: the old/new OLMo base names resolve to the same commit.

Producer cards give 14 declared edges; 13 have separately pinned current parent
snapshots. **Those snapshots are not the construction-time input revision.**
Verified construction edges: 0; fingerprints and exact-mark tests: not run.
The first verified-set target (Dec 1) is still pending, not counted as passing.

**Useful evidence for B:**
- OLMo and Tülu cards publish SFT→DPO→RLVR stage links. Pinned training metadata
  supplies experiment/dataset identifiers, but some parent fields are internal names
  or local paths. Their mapping to public checkpoint bytes remains unresolved.
- sci4ai/Qwen2.5-7B-Instruct-Abliterated describes per-layer directions in o_proj/down_proj.
- knoveleng/Qwen2.5-7B-Instruct-Uncensored describes a shared selected direction and
  additional embedding/head targets. Its evaluation JSON is archived as producer evidence,
  not reproduced behavior. Its referenced model is Qwen/Qwen2.5-7B-Instruct.
- Qwen/Qwen2.5-7B-Instruct-GGUF names its parent; pinned build/environment/calibration
  evidence is missing. Bit-exact reconstruction remains not run; current GGUF
  dequantization support is also outside this collection.
- mlabonne/Beagle14-7B ships a SLERP mergekit config with two named parents and
  layer/type-dependent t schedules. NeuralBeagle14 is a downstream DPO, not that merge.
  This is 1 recipe against K7’s target of at least 20. Construction parent commits
  and mergekit revision are still missing. No downloaded repository code was run.

**Candidate parent pairs (declared, not frozen evaluation truth):**

| Child at pinned revision | Producer-named parent snapshot | Proposed edit |
| --- | --- | --- |
| `allenai/OLMo-2-1124-7B-SFT@1de02c0175118a9de5854aec80a1f970e701e928` | `allenai/OLMo2-7B-1124@7df9a82518afdecae4e8c026b27adccc8c1f0032` | sft |
| `allenai/OLMo-2-1124-7B-DPO@e34ea60adff2e575f4fe7569eaffd1b28509b6fd` | `allenai/OLMo-2-1124-7B-SFT@1de02c0175118a9de5854aec80a1f970e701e928` | dpo |
| `allenai/OLMo-2-1124-7B-Instruct@470b1fba1ae01581f270116362ee4aa1b97f4c84` | `allenai/OLMo-2-1124-7B-DPO@e34ea60adff2e575f4fe7569eaffd1b28509b6fd` | posttraining |
| `allenai/Llama-3.1-Tulu-3-8B-SFT@f2a0b46b0cfda21003c6141b1ff837b7e165524d` | `meta-llama/Llama-3.1-8B@d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` | sft |
| `allenai/Llama-3.1-Tulu-3-8B-DPO@a7beb67e33ffd01cc87ac3b46cadc1000985b8db` | `allenai/Llama-3.1-Tulu-3-8B-SFT@f2a0b46b0cfda21003c6141b1ff837b7e165524d` | dpo |
| `allenai/Llama-3.1-Tulu-3-8B@666943798adbde0b1aff34626007e26986a3c107` | `allenai/Llama-3.1-Tulu-3-8B-DPO@a7beb67e33ffd01cc87ac3b46cadc1000985b8db` | rlvr |
| `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28` | `Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796` | posttraining_unspecified |
| `Qwen/Qwen2.5-7B-Instruct-GGUF@bb5d59e06d9551d752d08b292a50eb208b07ab1f` | `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28` | quantization |
| `sci4ai/Qwen2.5-7B-Instruct-Abliterated@a8360abe1ddd6b162843c3932779d20e5d739e55` | `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28` | refusal_removal |
| `knoveleng/Qwen2.5-7B-Instruct-Uncensored@888673633f34f493a71f7594eab1bb047e8ad353` | `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28` | refusal_removal |
| `mlabonne/NeuralBeagle14-7B@1567ad618a0998139654cb355738bb9bc018ca64` | `mlabonne/Beagle14-7B@c96dba5a911e1ba51c0b1e9e6d2eefeb474c9297` | dpo |
| `mlabonne/Beagle14-7B@c96dba5a911e1ba51c0b1e9e6d2eefeb474c9297` | `fblgit/UNA-TheBeagle-7b-v1@866d3ee19f983728e21a624f8a27574960073f27` | merge |
| `mlabonne/Beagle14-7B@c96dba5a911e1ba51c0b1e9e6d2eefeb474c9297` | `argilla/distilabeled-Marcoro14-7B-slerp@68b376fbe47749d296970595078f311eb9233043` | merge |
| `argilla/distilabeled-Marcoro14-7B-slerp@68b376fbe47749d296970595078f311eb9233043` | `mlabonne/Marcoro14-7B-slerp (unresolved)` | merge_parent |

All source URLs, README/config/training-metadata/recipe SHA256 values, inventories
and explicit gaps are in `$ZOO_DATA/runs/track-c-k7-handoff-20261108/candidate-pairs.json`.
Each pair points to producer source snapshots under `$ZOO_DATA`.
One transitive source-hash record: `runs/track-c-k7-handoff-20261108/inputs.json`;
SHA256 `dcf530195acfd8110c0ee016ad3af3017cb7f8e15280e70c19022367544d2703`.

Collection: 40 small source files, 234,927 payload bytes; new weight bytes 0.
These are metadata payload counts, not total network traffic or fingerprint cost.
All downloaded source hashes checked. Initial serialization failure used the wrong
installed card-data API; v2 corrected it with to_dict; no package changes.
Not-found cases retained: huihui-ai/Qwen2.5-7B-Instruct-abliterated and
mlabonne/NeuralDaredevil-8B. No access workaround attempted. Independent producer
candidates were inspected instead. No weight access or license acceptance attempted.

Next: B can use these pins as candidate tests; resolve producer experiment-to-input
weight mapping, run bounded fingerprints/marks and collect more merge configs.
Any admission into frozen verified truth requires the class-3 review; this file
makes no such admission. No release, contact, upload or commit performed.
