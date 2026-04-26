# SPRINT-004 Phase 3 — perplexity comparison, Qwen3.5-4B

Date: 2026-04-26
Branch: rotorquant
Image: rq-vllm:latest (overlay over vllm/vllm-openai:v0.19.1)
Script: `scripts/eval_perplexity.py`
Method: POST each text in TEXTS to `/v1/completions` with
`echo=True, prompt_logprobs=1, max_tokens=1`, sum -logprob over the
prompt tokens that vLLM scored, average to get mean_nll, exponentiate.

## Corpus

Thirteen paragraphs of plain English (≈ 4 KB / 698 scored tokens).
Topics: US presidents, oceans, Paris, Fibonacci recursion, doubling
arithmetic, Mount Everest, binary search, photosynthesis, Shakespeare,
boiling point, speed of light, producer-consumer queues, Roman Empire.
Style-neutral (no `<think>` tags, no system prompt, no chat template)
so the same texts can later be scored against a llama.cpp planar3 build
for cross-substrate parity.

## Per-paragraph results

| idx | tokens | preview                                        | fp16 ppl | rq3 ppl | (rq3 − fp16) / fp16 |
|----:|-------:|------------------------------------------------|---------:|--------:|--------------------:|
|   0 |     44 | "George Washington was the first president…"   |   1.8688 |  1.7880 |             −4.32 % |
|   1 |     45 | "The Pacific Ocean is the largest ocean…"      |   1.8156 |  1.8373 |             +1.20 % |
|   2 |     57 | "Paris is the capital of France…"              |   1.4611 |  1.4815 |             +1.40 % |
|   3 |     48 | "A recursive Fibonacci function…"              |   1.9372 |  1.8419 |             −4.92 % |
|   4 |     42 | "Two plus two equals four…"                    |   2.2329 |  2.2744 |             +1.86 % |
|   5 |     47 | "Mount Everest is the highest mountain…"       |   1.8191 |  1.8086 |             −0.58 % |
|   6 |     60 | "A binary search algorithm finds…"             |   1.8192 |  1.8441 |             +1.37 % |
|   7 |     54 | "Photosynthesis is the process by which…"      |   1.5253 |  1.5215 |             −0.25 % |
|   8 |     62 | "William Shakespeare was an English…"          |   1.6975 |  1.7077 |             +0.60 % |
|   9 |     55 | "Water boils at one hundred degrees…"          |   1.8726 |  1.8822 |             +0.51 % |
|  10 |     56 | "The speed of light in a vacuum…"              |   1.7367 |  1.8869 |             +8.65 % |
|  11 |     60 | "A common pattern in software engineering…"    |   2.5037 |  2.4697 |             −1.36 % |
|  12 |     68 | "The Roman Empire reached its greatest…"       |   1.9103 |  2.0398 |             +6.78 % |

## Aggregate

| metric    | fp16        | rq3 (rotorquant_planar3) |
|-----------|-------------|--------------------------|
| tokens    | 698         | 698                      |
| mean_nll  | 0.609704    | 0.619837                 |
| **ppl**   | **1.8399**  | **1.8586**               |

`(ppl_rq3 − ppl_fp16) / ppl_fp16 = +1.02 %`. Per-paragraph relative diff
is bimodal: 11 of 13 paragraphs sit within ±2 % of fp16, while two
(speed-of-light dates and Roman Empire chronology) push to +6.8 % and
+8.7 %. Both stress the model's date / number recall — exactly the
kind of fact-tied attention that's most sensitive to KV perturbation.

## Run history

| Corpus size | fp16 ppl | rq3 ppl | Δppl    |
|------------:|---------:|--------:|--------:|
|  236 tokens | 1.8209   | 1.8002  | −1.14 % |
|  698 tokens | 1.8399   | 1.8586  | +1.02 % |

Sign flipped between runs — the true gap on this kind of plain English
is ≲1 % and the prior 236-token measurement was inside the noise floor.

## Sprint-gate interpretation

The Sprint 004 hard gate is `|Δppl| ≤ 0.05 %` *vs the llama.cpp
planar3 baseline* (cross-substrate parity), not vs vLLM fp16. Hitting
that requires re-running the same corpus through
`johndpope/llama-cpp-turboquant`. That is the next concrete step.

The vs-fp16 number above (≈ 1 % at 698 tokens, ±~7 % per outlier
paragraph) is a *correctness* check, not the gate measurement: it
confirms the kv-write integration isn't silently corrupting state.
The Qwen3-4B case would have produced a >> 100 % gap; we land at 1 %.
Combined with the byte-level quality battery
(`SPRINT-004-PHASE2C-QUALITY-PROBE.md` — most prompts match fp16
token-for-token for tens of tokens) Phase 2c integration on Qwen3.5
is in good shape.

## Caveats

* 698 tokens is still a small corpus for tight ppl conclusions. A
  10× larger eval (wikitext-2 has ~245k tokens) would shrink the
  noise band well below 0.5 %.
* Tested only Qwen3.5-4B. Qwen3.5-9B (cached as `unsloth/Qwen3.5-9B`)
  passed the kernel cos-sim probe but has not been end-to-end ppl-tested.
* No comparison against llama.cpp's planar3 yet. The eval script
  points at any OpenAI-compatible endpoint, so a llama.cpp server
  with RotorQuant planar3 KV cache will produce a directly comparable
  number.
* The two outlier paragraphs (speed-of-light, Roman Empire) suggest
  date / numeric-fact recall is the most sensitive failure mode — a
  domain-targeted ppl run on Wikipedia date-rich pages would be more
  conservative than mixed prose.
