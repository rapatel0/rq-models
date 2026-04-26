# SPRINT-004 Phase 3 — perplexity comparison, Qwen3.5-4B

Date: 2026-04-26
Branch: rotorquant
Image: rq-vllm:latest (overlay over vllm/vllm-openai:v0.19.1)
Script: `scripts/eval_perplexity.py`
Method: POST each text in TEXTS to `/v1/completions` with
`echo=True, prompt_logprobs=1, max_tokens=1`, sum -logprob over the
prompt tokens that vLLM scored, average to get mean_nll, exponentiate.

## Corpus

Five paragraphs of plain English (≈ 1500 chars / 236 scored tokens).
Topics: US presidents, oceans, Paris, Fibonacci recursion, doubling
arithmetic. Deliberately style-neutral — no `<think>` tags, no system
prompt, no chat template — so the same texts can later be scored
against a llama.cpp planar3 build for cross-substrate parity.

## Per-paragraph results

| idx | tokens | text                                           | fp16 ppl | rq3 ppl | (rq3 - fp16) / fp16 |
|----:|-------:|------------------------------------------------|---------:|--------:|--------------------:|
|   0 |     44 | "George Washington was the first president…"   |   1.8688 |  1.7867 |               -4.39 % |
|   1 |     45 | "The Pacific Ocean is the largest ocean…"      |   1.8156 |  1.8350 |               +1.07 % |
|   2 |     57 | "Paris is the capital of France…"              |   1.4611 |  1.4830 |               +1.50 % |
|   3 |     48 | "A recursive Fibonacci function…"              |   1.9372 |  1.8433 |               -4.85 % |
|   4 |     42 | "Two plus two equals four…"                    |   2.2329 |  2.2509 |               +0.81 % |

## Aggregate

| metric    | fp16        | rq3 (rotorquant_planar3) |
|-----------|-------------|--------------------------|
| tokens    | 236         | 236                      |
| mean_nll  | 0.599353    | 0.587906                 |
| **ppl**   | **1.8209**  | **1.8002**               |

`(ppl_rq3 - ppl_fp16) / ppl_fp16 = -1.14 %` — rq3 is fractionally
*lower* than fp16 on this corpus. Per-paragraph noise is ±5 %, so the
aggregate gap sits inside the corpus's noise floor; treat as
statistical parity, not as evidence that 3-bit KV outperforms fp16.

## Sprint-gate interpretation

The Sprint 004 hard gate is `|Δppl| ≤ 0.05 %` *vs the llama.cpp
planar3 baseline*, not vs vLLM fp16. Hitting that requires a separate
cross-substrate run on the same corpus through `johndpope/llama-cpp-turboquant`.
That is the next concrete step.

The vs-fp16 number above (1.14 % aggregate, ±5 % per-paragraph) is
useful as a *correctness* check: it confirms the kv-write integration
isn't silently corrupting state, which the heavily degraded Qwen3-4B
output would have suggested. Combined with the byte-level quality
battery (artifacts/SPRINT-004-PHASE2C-QUALITY-PROBE.md) — most prompts
match fp16 token-for-token for tens of tokens — Phase 2c integration
on Qwen3.5 is in good shape.

## Caveats

* 236 tokens is a small corpus; the per-paragraph spread is large
  enough that a different draw of texts could produce a Δppl with a
  flipped sign. Re-run on a 5–10× larger corpus (wikitext-2) before
  drawing tight conclusions about 0.05 % parity.
* Tested only Qwen3.5-4B. Qwen3.5-9B (cached as unsloth/Qwen3.5-9B)
  passed the kernel cos-sim probe but has not been end-to-end ppl-tested.
* No comparison against llama.cpp's planar3 yet. The same script can
  point at any OpenAI-compatible endpoint, so a llama.cpp server with
  RotorQuant planar3 KV cache will produce a directly comparable
  number.
