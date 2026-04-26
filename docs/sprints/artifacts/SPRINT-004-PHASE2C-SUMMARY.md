# SPRINT-004 Phase 2c — Qwen3.5 baseline serve, end-to-end characterization

Date range: 2026-04-26
Branch: `rq-models@rotorquant` (overlay) / `rq-vllm@feature/rotorquant`
Image: `rq-vllm:latest` (overlay over `vllm/vllm-openai:v0.19.1`,
transformers 5.5.4, CUDA 12.9.86, RTX 4090)
Scope: Qwen3.5-4B (multimodal, hybrid linear/full attention,
head_dim=256, 8 full-attn layers in the 32-layer text trunk) and
unsloth/Qwen3.5-9B (same arch, ~19 GB bf16).

This document is a single-page consolidation of the side-by-side
fp16 vs `rotorquant_planar3` evidence collected across thirteen
investigation dimensions inside vLLM. Detailed per-test artifacts are
in this directory under `SPRINT-004-PHASE2C-*` and
`SPRINT-004-PHASE3-*`.

## Top-line outcome

| Pillar                                              | Result                                                   |
|-----------------------------------------------------|----------------------------------------------------------|
| Phase 2c integration correctness                    | ✅ end-to-end serve works on Qwen3.5-4B and Qwen3.5-9B   |
| Generated-token quality vs fp16 baseline            | ✅ tens of tokens byte-identical, factually equivalent   |
| Perplexity gap vs fp16 (Qwen3.5-4B / 9B)            | +1.02 % / +1.70 % over 698-token plain-English corpus    |
| Long-context retrieval (needle-in-haystack)         | ✅ 7 / 8 PASS up to 12 493-token input depth             |
| Determinism at T=0                                  | ✅ 5 / 5 prompts byte-stable across runs                 |
| CUDAGraph compatibility                             | ✅ kernel graph-safe out of the box                      |
| Decode tok/s gap (cudagraph, single-request)        | −3 % vs fp16 (89.0 vs 91.85)                             |
| Concurrent throughput gap (cudagraph, N=16)         | −8.1 % vs fp16 (985 vs 1072 tok/s)                       |
| Prefill tok/s gap                                   | within ±1 %                                              |

Net: Phase 2c integration on Qwen3.5 is correctness-clean and
production-shape-stable. Quality is fp16-equivalent within typical
quantization noise, and the perf cost (4–8 % decode, 0 % prefill) is
explained by per-decoded-token planar3 kernel-launch overhead — a
structural cost that Phase 2.5 (single pack-and-scatter call instead
of the pack→unpack pair) is exactly designed to remove.

## Why Qwen3 was bad and Qwen3.5 is good

The planar3 KV kernel is byte-for-byte from the
`johndpope/llama-cpp-turboquant` build, calibrated for K vectors that
look like whitened Gaussians. Qwen3 introduced a per-head k_norm with
trainable γ; Layer-0 γ at small inputs blows per-element σ from 0.09
to 13.4 with strong anisotropy, and the fixed-Givens + Lloyd-Max
codebook can't represent that distribution. Cosine similarity drops
from the synthetic ceiling 0.983 to 0.666 on Qwen3-4B's L00 K, which
is enough to destroy attention (`"2+2=" → "222222222222"`).

Qwen3.5 reverted to a well-conditioned post-k_norm K (σ ≈ 1.3–1.6
across layers, all of them within range of the codebook), so the same
kernel hits cos-sim 0.95–0.98 across all probed full-attention layers
in both 4B and 9B sizes. Verified by `scripts/probe_kv_quality.py`
(see `SPRINT-004-PHASE2C-QUALITY-PROBE.md`).

Decision: target Qwen3.5 / Qwen3.6 as the supported model family on
the rq-models vLLM substrate. Qwen3 (and earlier with similar k_norm
γ pathologies) is documented as a known incompatibility.

## Dimension-by-dimension evidence

| # | Dimension                       | Test                                                                             | rq3 outcome                                       | Artifact                                          |
|---|---------------------------------|----------------------------------------------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| 1 | Kernel cos-sim (real K)         | `scripts/probe_kv_quality.py` hooks `attn.k_proj` and `attn.k_norm` post-forward | 0.95 – 0.98 mean across all probed full-attn layers | SPRINT-004-PHASE2C-QUALITY-PROBE.md               |
| 2 | Quality battery 4B (24–128 tok) | 7 prompts, T=0; compare token-by-token to fp16                                   | most spans byte-identical (e.g. all 10 US presidents) | SPRINT-004-PHASE2C-QBATTERY-{FP16,RQ3}.txt        |
| 3 | Quality battery 9B              | same battery on unsloth/Qwen3.5-9B                                               | both modes match incl. Chinese-mode "2+2=" output | SPRINT-004-PHASE2C-QBATTERY-9B-{FP16,RQ3}.txt     |
| 4 | Perplexity 4B (236 tok)         | `scripts/eval_perplexity.py` echo + prompt_logprobs                              | Δppl = −1.14 % (inside corpus noise)              | SPRINT-004-PHASE3-PPL.md                          |
| 5 | Perplexity 4B (698 tok)         | same script, expanded 13-paragraph corpus                                        | Δppl = +1.02 % (≤2 % per paragraph except 2 outliers) | SPRINT-004-PHASE3-PPL-{FP16,RQ3}.txt              |
| 6 | Perplexity 9B (698 tok)         | same script on unsloth/Qwen3.5-9B                                                | Δppl = +1.70 %                                    | SPRINT-004-PHASE3-PPL-9B-{FP16,RQ3}.txt           |
| 7 | Long context (700 in + 512 out) | `scripts/longctx_test.py` summary + 500-word story decode                        | parallel coherent output, no tail decay           | SPRINT-004-PHASE2C-LONGCTX-{FP16,RQ3}.txt         |
| 8 | Single-request perf (eager)     | `scripts/perf_bench.py` SSE TTFT + decode tok/s                                  | TTFT 46 ms, decode 76.6 tok/s (−4 % vs fp16)      | SPRINT-004-PHASE2C-PERF-{FP16,RQ3}.txt            |
| 9 | Concurrent thru (eager)         | `scripts/concurrent_bench.py` N=1, 4, 8, 16                                      | 859 tok/s aggregate at N=16 (−16 % vs fp16)       | SPRINT-004-PHASE2C-CONC-{FP16,RQ3}.txt            |
| 10 | Prefill thru                   | `scripts/prefill_bench.py` 41–1133 tok prompts                                   | 14 267 tok/s at 1133-tok prompt (within 1 % of fp16) | SPRINT-004-PHASE2C-PREFILL-{FP16,RQ3}.txt         |
| 11 | CUDAGraph compat + perf        | drop `--enforce-eager`, capture graphs                                           | kernel graph-safe; decode 89.0 tok/s, N=16 = 985 (−8 % vs fp16) | SPRINT-004-PHASE2C-{PERF,CONC}-{FP16,RQ3}-CG.txt |
| 12 | Needle (≤ 2 534 tok input)     | `scripts/needle_test.py` haystack at 25/50/75 % positions                        | 6 / 6 PASS (fp16 5 / 6)                           | SPRINT-004-PHASE2C-NEEDLE-{FP16,RQ3}.txt          |
| 13 | Determinism at T=0             | `scripts/det_test.py` 5 prompts × 3 runs, SHA-1                                  | 5 / 5 byte-stable (fp16 also 5 / 5)               | SPRINT-004-PHASE2C-DET-{FP16,RQ3}.txt             |
| 14 | Deep needle (≤ 12 493 tok in)  | scaled needle test at 16 k context                                               | 7 / 8 PASS                                        | SPRINT-004-PHASE2C-NEEDLE-DEEP-RQ3.txt            |

(Numbering is for cross-reference; tests 1–14 were run in order over a
single ~2-hour session 2026-04-26.)

## Two perf-cost insights worth keeping

1. **Decode kernel-launch overhead, not throughput.** Prefill is within
   1 % of fp16; decode is up to 16 % slower at N=16. The cost shape
   identifies it as launch overhead from the per-decoded-token
   pack→unpack call pair, not throughput-bound math. CUDAGraph capture
   amortizes the launches and halves the gap (−16 % → −8 % at N=16),
   confirming the diagnosis.

2. **Phase 2.5 closes both pieces structurally.** The single
   pack-and-scatter call (instead of pack→unpack) drops half the
   per-token launches and removes the temporary fp16 K/V buffer; the
   uint8 packed cache layout then delivers the 5.12 × cache-byte
   savings. Expected outcome: rq3 decode pulls level with fp16 (or
   ahead, since prefill amortization works for it too) AND the engine
   fits ≈ 5 × more concurrent sequences in the same VRAM budget — the
   actual sprint payoff that Phase 2c lossy-passthrough does not yet
   deliver.

## What's still open

- **Sprint hard gate**: |Δppl| ≤ 0.05 % vs the llama.cpp planar3
  baseline (cross-substrate parity). Not yet measured. The eval
  scorer (`scripts/eval_perplexity.py`) hits any OpenAI-compatible
  endpoint, so a llama.cpp-turboquant server running the same
  698-token corpus produces a directly comparable number. Setup is
  the gating concern (does llama.cpp-turboquant support Qwen3.5's
  hybrid linear/full attention?).
- **Phase 2.5 wire-in**: kernels are shipped (rq-vllm `d5f060e64`,
  overlay mirror `7b80e9c`); FlashAttention forward not yet wired to
  call `pack_and_scatter_planar3` on write or `gather_and_unpack_planar3`
  on read. The dtype map and cache-shape glue need to flip to uint8.
- **Per-model PPL on a 10 × larger corpus** (e.g., wikitext-2's
  ~245 k tokens) to shrink the per-paragraph noise band well below
  1 %. Useful only if cross-substrate parity becomes the goal.
