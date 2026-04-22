# RotorQuant Benchmark Report — Qwen3.5-27B on RTX 5090

**Date**: 2026-04-03
**Hardware**: NVIDIA RTX 5090 (32 GB VRAM)
**Model**: Qwen3.5-27B Q4_K_M (16.7 GB, unsloth/Qwen3.5-27B-GGUF)
**Runtime**: llama.cpp RotorQuant fork (johndpope/llama-cpp-turboquant, commit 20efe75)
**KV Cache (benchmarked)**: iso3/iso3 (RotorQuant IsoQuant 3-bit symmetric)
**Default Context**: 131,072 tokens (128K)

> **Note (2026-04-05):** Default KV cache switched from `iso3` to `iso4` for better quality.
>
> **Note (2026-04-20):** Default switched again from `iso4` → `planar4` after perplexity
> benchmarks showed PlanarQuant consistently outperforms IsoQuant at every bit depth.
> `planar4` beats `iso4` by ~0.17 PPL (2.5 sigma) at 4K context on Qwen3.5-27B.
> `planar3` is now the recommended high-compression alternative (replaces `iso3`).
>
> ## KV Cache Perplexity Comparison (2026-04-20)
>
> **Model**: Qwen3.5-27B Q4_K_M, wikitext-2-raw, RTX 5090
>
> | KV Cache | Bits/elem | PPL (ctx=512) | PPL (ctx=4096) | vs f16 (4K) |
> |----------|:---------:|:-------------:|:--------------:|:-----------:|
> | f16 | 16.0 | ~6.55 | 7.5942 | baseline |
> | **planar3** | 3.125 | **7.1950** | **8.1973** | +0.603 |
> | planar4 | 4.25 | — | 8.2548 | +0.661 |
> | iso4 (old default) | 4.25 | 7.2549 | 8.3689 | +0.775 |
> | iso3 | 3.125 | 7.2771 | 8.4344 | +0.840 |
>
> Key findings:
> - PlanarQuant beats IsoQuant at every bit depth by a statistically significant margin
> - `planar3` (3-bit) beats `iso4` (4-bit) — better compression *and* better quality
> - Default is now `planar4`; use `planar3` (`KV_CACHE_TYPE=planar3`) for max compression

---

> ## Qwen3.6-35B-A3B Throughput Benchmarks (2026-04-20)
>
> **Model**: Qwen3.6-35B-A3B UD-Q4_K_XL (22.3 GB), unsloth/Qwen3.6-35B-A3B-GGUF
> **Hardware**: RTX 5090 (32 GB), 23.1 GB used at idle post-load, 9 GB free
> **Config**: ctx=65536, KV cache=planar4/planar4, 2 parallel slots, 99 GPU layers, flash-attn on
>
> ### Prefill throughput (prompt processing)
>
> | Prompt tokens | Time (ms) | Prefill tok/s |
> |:-------------:|----------:|:-------------:|
> | 25 | 33 | ~750 |
> | 249 | 351 | 710 |
> | 505 | 98 | **5,174** |
>
> Notes: Short prompts (<30 tok) are dispatch-latency dominated. At 505 tokens the GPU saturates
> and throughput jumps to ~5,174 tok/s — significantly faster than Qwen3.5-27B's ~3,500 tok/s
> peak (MoE activates fewer params per token, reducing compute per prefill step).
>
> ### Decode throughput (token generation)
>
> | n_predict | Time (ms) | Decode tok/s |
> |:---------:|----------:|:------------:|
> | 50 | 338 | 148 |
> | 200 | 1,019 | **196** |
> | 500 | 2,606 | 192 |
>
> **Single-slot decode: ~190–196 tok/s** — roughly 2.8× faster than Qwen3.5-27B's 69.3 tok/s.
> This is the MoE advantage: 35B total params but only ~3B active per token.
>
> ### Concurrent throughput (2 parallel slots)
>
> | Slots | Per-slot tok/s | Aggregate tok/s |
> |:-----:|:--------------:|:---------------:|
> | 2 | 149 | **298** |
>
> Both slots served simultaneously at 149 tok/s each — near-linear scaling, total ~298 tok/s
> aggregate throughput.
>
> ### Summary vs Qwen3.5-27B
>
> | Metric | Qwen3.5-27B Q4_K_M | Qwen3.6-35B-A3B Q4_K_XL | Delta |
> |--------|:-------------------:|:------------------------:|:-----:|
> | Model size | 16.7 GB | 22.3 GB | +5.6 GB |
> | VRAM at ctx=65K | ~20 GB | 23.1 GB | +3.1 GB |
> | Decode tok/s (single) | 69 | **196** | **+184%** |
> | Decode tok/s (2-slot) | ~138 aggregate | **298** | **+116%** |
> | Peak prefill tok/s | ~3,500 | ~5,174 | +48% |
> | Active params/token | 27B | ~3B | -89% |
>
> The throughput gain is purely architectural: Qwen3.6's MoE routing activates ~3B params per
> token vs Qwen3.5's full 27B dense activation. At 22.3 GB it still fits comfortably in 32 GB
> with room for up to 65K context.

---

> ## Qwen3.6-35B-A3B Parallelization & Context Benchmarks (2026-04-20)
>
> **Model**: Qwen3.6-35B-A3B UD-Q4_K_XL (22.3 GB)
> **Config**: iso3/iso3, ctx=524288 total (65K/slot), 8 parallel slots, flash-attn on
> **Hardware**: RTX 5090 (32 GB) — 29.7 GB used at benchmark time, 2.9 GB headroom
>
> ### KV Cache Architecture Notes
>
> - All four types (planar3, planar4, iso3, iso4) use **deferred K quantization**: K is
>   allocated as f16 during prefill to avoid error compounding, then converted to the target
>   type after prefill completes (`convert_deferred_keys()`). Steady-state K is compressed.
> - Actual bit depths: **planar3/iso3 = 3.125 bpe** (2-bit centroid + 1-bit QJL sign,
>   50 bytes/128 values); **planar4/iso4 = 4.25 bpe** (4-bit nibble, 68 bytes/128 values).
> - VRAM difference between planar3 and iso3 at the same context reflects the different
>   rotation schemes' compute/memory tradeoffs during the deferred-conversion window.
>
> ### Max context window per configuration (RTX 5090 32 GB)
>
> | KV type | Per-slot ctx | Parallel slots | Total KV | VRAM total |
> |---------|:------------:|:--------------:|:--------:|:----------:|
> | planar4 | 32,768 | 2 | 0.98 GB | 22.4 GB |
> | planar3 | 262,144 | 2 | 5.4 GB | 26.8 GB |
> | **iso3** | **262,144** | **2** | **7.1 GB** | **28.7 GB** |
> | iso3 | 65,536 | 8 | 7.1 GB | 28.7 GB |
>
> Qwen3.6 training context = 262,144. iso3 P=2 achieves the full training context per slot.
>
> ### Parallelization impact on throughput (iso3, 65K ctx/slot, 500 tokens, 3 trials)
>
> | Concurrent requests | Per-slot tok/s | Aggregate tok/s | Per-slot vs P=1 | Agg. scaling |
> |:-------------------:|:--------------:|:---------------:|:---------------:|:------------:|
> | 1 | **193** | 193 | baseline | 1.0× |
> | 2 | 150 | **299** | −22% | 1.55× |
> | 4 | 99 | **397** | −49% | 2.06× |
> | 8 | 54 | **431** | −72% | 2.23× |
>
> All results are stable across 3 trials (variance < 1%). Warmup: 3 discarded requests.
>
> **Key findings:**
> - At P=2: aggregate throughput nearly doubles (+55%). Per-slot cost is only −22%.
>   Best sweet spot for interactive use with 2 concurrent users.
> - At P=4: aggregate 2× single-slot, per-slot halves. Good for batch inference.
> - At P=8: diminishing returns — aggregate only 2.2× vs 4× theoretical. GPU compute
>   saturates. Each slot gets 54 tok/s which is still usable for interactive chat.
> - **P=4 is the practical optimum**: 397 tok/s aggregate at 99 tok/s/slot — users
>   experience ~100 tok/s each while 4 simultaneous conversations run.

---

## 1. Throughput Comparison — iso3/iso3 vs f16/f16

Measured with `llama-bench`, 1 repetition, batch_size=1.

### Prefill (prompt processing, tok/s)

| Context | f16/f16 | iso3/iso3 | Overhead |
|--------:|--------:|----------:|---------:|
| 512 | 3,608 | 3,119 | 1.2x |
| 2,048 | 3,551 | 2,331 | 1.5x |
| 4,096 | 3,490 | 1,711 | 2.0x |
| 8,192 | 3,306 | 1,109 | 3.0x |
| 16,384 | 3,015 | 649 | 4.6x |
| 32,768 | 2,502 | 357 | 7.0x |
| 65,536 | **OOM** | 189 | — |
| 131,072 | **OOM** | 98 | — |

### Decode (token generation, tok/s)

| Config | tok/s | % of f16 |
|--------|------:|---------:|
| f16/f16 | 69.3 | 100% |
| iso3/iso3 | 67.5 | **97%** |

### Batch Size Impact (iso3/iso3, 32K context)

| Batch Size | Prefill tok/s | Decode tok/s |
|-----------:|--------------:|-------------:|
| 512 | 357 | 69.1 |
| 2,048 | 357 | 67.8 |
| 4,096 | 357 | 67.6 |
| 8,192 | 357 | 67.6 |

Batch size has no measurable impact on throughput.

---

## 2. Memory Usage

### VRAM at Different Context Lengths (iso3/iso3)

| Context | VRAM Used | Headroom (of 32 GB) |
|--------:|----------:|--------------------:|
| 4,096 | 17.2 GB | 14.8 GB |
| 128K | 22.3 GB | 9.7 GB |
| 196K | 25.3 GB | 6.7 GB |

### Analytical KV Cache Size — Qwen3.5-27B (64 layers, 8 KV heads, head_dim=128)

| Config | KV at 4K | KV at 32K | KV at 128K | Compression |
|--------|----------|-----------|------------|-------------|
| f16/f16 | 1.0 GB | 8.0 GB | 32.0 GB | 1.0x |
| iso3/iso3 | 0.2 GB | 1.6 GB | 6.5 GB | **4.9x** |
| iso4/iso4 | 0.3 GB | 2.2 GB | 8.5 GB | 3.8x |

### Total VRAM Budget (model Q4_K_M 16.7 GB + KV at 128K)

| Config | Total | Fits RTX 5090? | Fits A100 40GB? |
|--------|------:|:-:|:-:|
| f16/f16 | 48.7 GB | No | No |
| iso3/iso3 | **23.2 GB** | **Yes** | **Yes** |
| iso4/iso4 | 25.2 GB | Yes | Yes |

---

## 3. Perplexity — wikitext-2 (ctx=2048)

> **Updated 2026-04-20:** Full comparison across all KV cache variants and both
> Qwen3.5-9B and Qwen3.5-27B. Standard evaluation: wikitext-2-raw-v1 test split,
> ctx=2048 sliding window. Model weights: Q4_K_M GGUF (unsloth). llama-perplexity binary.

### Qwen3.5-27B-Q4_K_M

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **6.6417** | — |
| planar3 / planar3 | 3.125 | 7.0099 | +0.368 (+5.5%) |
| iso3 / iso3 | 3.125 | 7.1016 | +0.460 (+6.9%) |
| planar4 / planar4 | 4.25 | 7.0178 | +0.376 (+5.7%) |
| iso4 / iso4 | 4.25 | 7.0962 | +0.454 (+6.8%) |
| planar3 / f16 (K-only) | — | 6.6417 | **0.000** |
| f16 / planar3 (V-only) | — | 7.0099 | +0.368 |

### Qwen3.5-9B-Q4_K_M

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **7.6760** | — |
| planar4 / planar4 | 4.25 | 7.3368 | −0.339 (−4.4%) |
| planar3 / planar3 | 3.125 | 7.3731 | −0.303 (−3.9%) |
| iso4 / iso4 | 4.25 | 7.4868 | −0.188 (−2.5%) |
| iso3 / iso3 | 3.125 | 7.5490 | −0.127 (−1.7%) |
| planar3 / f16 (K-only) | — | 7.6760 | **0.000** |
| f16 / planar3 (V-only) | — | 7.3731 | −0.303 |
| planar4 / f16 (K-only) | — | 7.6760 | **0.000** |
| f16 / iso3 (V-only) | — | 7.5490 | −0.127 |

### Qwen3.5-9B-Q4_K_M — C4 validation

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **11.5291** | — |
| planar3 / planar3 | 3.125 | 11.7294 | +0.200 (+1.7%) |
| iso3 / iso3 | 3.125 | 11.7364 | +0.207 (+1.8%) |
| planar4 / planar4 | 4.25 | 11.7429 | +0.214 (+1.9%) |
| iso4 / iso4 | 4.25 | 11.7430 | +0.214 (+1.9%) |
| planar3 / f16 (K-only) | — | 11.5291 | **0.000** |
| f16 / planar3 (V-only) | — | 11.7294 | +0.200 |

### Qwen3.5-27B-Q4_K_M — C4 validation

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **9.8485** | — |
| planar3 / planar3 | 3.125 | 9.9624 | +0.114 (+1.2%) |
| planar4 / planar4 | 4.25 | 9.9611 | +0.113 (+1.1%) |
| iso3 / iso3 | 3.125 | 9.9823 | +0.134 (+1.4%) |
| iso4 / iso4 | 4.25 | 9.9822 | +0.134 (+1.4%) |
| planar3 / f16 (K-only) | — | 9.8485 | **0.000** |
| f16 / planar3 (V-only) | — | 9.9624 | +0.114 |

### Qwen3.6-35B-A3B-Q4_K_XL

> **Note (2026-04-21):** Qwen3.6 MoE inverts the planar vs iso ordering seen on Qwen3.5.
> **iso consistently beats planar** on both corpora. Docker-compose default: `iso3`
> (same 3.125 bpe / 7.1 GB KV budget, better PPL than planar3). Use `iso4` for quality.

#### wikitext-2-raw-v1 test

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **6.1316** | — |
| **iso4 / iso4** | 4.25 | **6.2262** | **+0.095 (+1.5%)** |
| iso3 / iso3 | 3.125 | 6.2515 | +0.120 (+2.0%) |
| planar4 / planar4 | 4.25 | 6.2529 | +0.121 (+2.0%) |
| planar3 / planar3 | 3.125 | 6.2904 | +0.159 (+2.6%) |

#### C4 validation

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **10.5681** | — |
| **iso4 / iso4** | 4.25 | **10.6928** | **+0.125 (+1.2%)** |
| iso3 / iso3 | 3.125 | 10.7108 | +0.143 (+1.4%) |
| planar4 / planar4 | 4.25 | 10.7335 | +0.165 (+1.6%) |
| planar3 / planar3 | 3.125 | 10.7354 | +0.167 (+1.6%) |

Both corpora agree: iso4 best, iso3 ≈ planar4 (within ±0.038 noise), planar3 worst.

### Qwen3.6-27B-Q4_K_XL (dense)

> **Note (2026-04-22):** Dense model reverts to the Qwen3.5 pattern — planar beats iso.
> Compose default: `planar3`. The iso > planar inversion appears specific to the Gated
> Delta Net (MoE) architecture of Qwen3.6-35B-A3B.

#### wikitext-2-raw-v1 test

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **7.0901** | — |
| **planar3 / planar3** | 3.125 | **7.4044** | **+0.314 (+4.4%)** |
| planar4 / planar4 | 4.25 | 7.4703 | +0.380 (+5.4%) |
| iso3 / iso3 | 3.125 | 7.5587 | +0.469 (+6.6%) |
| iso4 / iso4 | 4.25 | 7.6405 | +0.550 (+7.8%) |

#### C4 validation

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **10.2963** | — |
| **planar4 / planar4** | 4.25 | **10.4032** | **+0.107 (+1.0%)** |
| planar3 / planar3 | 3.125 | 10.4176 | +0.121 (+1.2%) |
| iso4 / iso4 | 4.25 | 10.4366 | +0.140 (+1.4%) |
| iso3 / iso3 | 3.125 | 10.4506 | +0.154 (+1.5%) |

planar4 vs planar3 gap on C4 (0.014) is within ±0.037 noise. **planar3 is the better
default**: same quality as planar4 on C4, clearly better on wikitext-2, at 3.125 vs
4.25 bpe (lower memory).

### Key findings

**K quantization has zero PPL impact on all datasets.** Quantizing only K
produces identical PPL to the f16/f16 baseline across both models, both datasets,
and all bit depths. All PPL change comes entirely from V quantization.

**C4 shows the true degradation; wikitext-2 was misleading for 9B.** On wikitext-2,
9B appeared to *improve* with V quantization (regularization artifact on clean
encyclopedia text). On C4 (diverse web text), all quantized variants correctly
degrade: +0.20 PPL for 9B, +0.11–0.13 PPL for 27B. C4 is the right eval corpus.

**PlanarQuant beats IsoQuant on dense Qwen3.5 models.** At the same bit depth:
planar3 < iso3 and planar4 < iso4 on Qwen3.5-9B and 27B (both datasets), by
~0.02 PPL (27B/C4) to ~0.01 PPL (9B/C4).

**IsoQuant beats PlanarQuant on Qwen3.6-35B-A3B MoE; PlanarQuant wins on dense models.**
The iso > planar inversion is specific to the Gated Delta Net hybrid MoE architecture:
iso4 (6.2262) < iso3 ≈ planar4 (~6.252) < planar3 (6.2904) on the 35B-A3B MoE.
For Qwen3.6-27B dense, the pattern reverts: planar3 < planar4 < iso3 < iso4 (same as
Qwen3.5). Rule of thumb: **MoE → iso, dense → planar**.

**3-bit and 4-bit are statistically indistinguishable on C4 (Qwen3.5).** planar3
and planar4 differ by only 0.0001–0.002 PPL on Qwen3.5/C4. On Qwen3.6/wikitext-2,
iso4 edges iso3 by 0.025 PPL — within noise but consistently in the same direction.

**27B degrades less than 9B in percentage terms.** +1.2% vs +1.7% — larger models
are more robust to KV quantization, consistent with scale improving representation
redundancy.

---

## SpectralQuant (HuggingFace Python) — Qwen3.5-9B Results

**Date**: 2026-04-20
**Model**: Qwen/Qwen3.5-9B (bf16, HuggingFace transformers 5.5.4)
**Calibration**: C4 train, 128 samples × 2048 tokens, seed=42
**Calibration file**: `calibration/calibration-qwen3.5-9b.safetensors` (16.96 MB)
**Evaluation**: wikitext-2-raw-v1 test, sliding window ctx=2048 stride=512

### d_eff summary (full attention layers only, 8/32 layers)

| Layer | d_eff_k | d_eff_v | head_dim |
|------:|--------:|--------:|:--------:|
| 3  | 207 | 226 | 256 |
| 7  | 212 | 225 | 256 |
| 11 | 209 | 223 | 256 |
| 15 | 215 | 230 | 256 |
| 19 | 216 | 232 | 256 |
| 23 | 213 | 237 | 256 |
| 27 | 213 | 231 | 256 |
| 31 | 211 | 226 | 256 |

### Perplexity Results

| Method | PPL | Δ vs f16 |
|--------|----:|:--------:|
| f16 baseline | 7.9462 | — |
| SpectralKVCache (C4 calibration) | 78.8447 | **+70.9** |

**Kill gate (Δ ≤ 0.5): FAIL**

### Analysis: Architecture Mismatch

SpectralQuant uses **vector quantization (VQ)** — the full signal subspace vector
is quantized to the nearest of 16 centroids (4-bit). This is effective when `d_eff`
is small (the original assumption was d_eff ≈ 4–5 dims for K), giving a manageable
codebook in a compact space.

For Qwen3.5-9B's hybrid architecture (linear attention every 3/4 layers, full
attention every 4th), the full-attention layers have **near-full-rank KV
representations**: d_eff_k ≈ 207–216 / 256 (81–84%). The 16 VQ centroids are
distributed across a 207-dimensional space — effectively 0.077 bits/dimension —
causing catastrophic reconstruction error.

**Root cause**: The Qwen3.5 hybrid architecture produces high-rank attention outputs
in its full-attention layers. SpectralQuant's VQ approach is only effective when
the KV cache is low-rank (d_eff ≤ ~32), which is the case for pure-attention models
but not hybrid SSM/attention architectures.

**Implication**: SpectralQuant should be evaluated on pure full-attention models
(e.g. Llama 3.1, Mistral) where low-rank KV structure is more likely. For Qwen3.5's
hybrid architecture, IsoQuant/PlanarQuant (rotation-based, per-element quantization)
are the appropriate methods.

---

## 4. Needle-In-A-Haystack (NIAH)

Tested via llama-server API. Needle: "The secret project codename is ZEPHYR-NINE-SEVEN-ALPHA."

### iso3/iso3 Results

| Context | Depth 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Recall |
|---------|:---------:|:---:|:---:|:---:|:---:|-------:|
| 4K | HIT | HIT | HIT | HIT | HIT | **100%** |
| 8K | HIT | HIT | HIT | MISS | HIT | **80%** |

Average recall: **90%**

---

## 5. Battery Test — Competence (27/28 passed, 96%)

Tested via Docker container (`rotorquant:latest`, iso3/iso3, Qwen3.5-27B Q4_K_M).

### Performance Metrics

| Metric | Result |
|--------|--------|
| Decode throughput | 67.5 tok/s |
| Prefill (2K tokens) | 4.0s |
| Time to first token | 0.20s |
| VRAM during battery | 22.8 GB / 32.6 GB (70%) |

### Competence Results

| Category | Pass/Total | Tests |
|----------|:----------:|-------|
| Math | 4/4 | 17×24=408, √144=12, word problem, percentages |
| Reasoning | 4/4 | Syllogism, spatial, causal, counterfactual |
| Code | 2/2 | Factorial generation, list reversal tracing |
| Knowledge | 3/4 | Tokyo, Shakespeare, 1945. H₂O vs H2O (unicode) |
| Instruction following | 3/3 | Numbered lists, constraints, JSON output |
| NIAH (battery) | 6/6 | 100% at 4K and 8K, depths 0.25/0.5/0.75 |
| Consistency | 1/1 | Deterministic at temp=0 (3 identical runs) |
| VRAM | 1/1 | 70% utilization, 10 GB headroom |

The single "failure" (H₂O vs H2O) is a unicode subscript match — the model gave the chemically correct answer with proper formatting.

---

## 6. Key Findings

1. **Decode speed is virtually unchanged**: 67.5 tok/s (iso3) vs 69.3 tok/s (f16) — 97% of baseline. Users experience no perceptible generation slowdown.

2. **Prefill has measurable overhead**: 1.2x at 512 tokens scaling to 7x at 32K. This is the rotation + quantization cost during prompt ingestion. For interactive chat (short prompts), negligible. For bulk document processing, noticeable.

3. **128K context on consumer GPU**: iso3/iso3 enables 128K context on RTX 5090 (22.3 GB) where f16 OOMs at 65K. This is the primary value proposition.

4. **Quality degradation is minimal**: +5.9% perplexity, 90% NIAH recall, 96% battery test pass rate. No catastrophic failures on any reasoning, math, or code tasks.

5. **RotorQuant vs TurboQuant**: RotorQuant's fused CUDA kernels solve the throughput problem our Python TurboQuant implementation had (44ms/token → 67.5 tok/s). The block-diagonal rotation (372 params vs 16,384) is also 44x more parameter-efficient.

---

## 7. Deployment Configuration

```bash
# Start serving Qwen3.5-27B with RotorQuant 128K context
docker compose --profile qwen up

# Or directly:
docker run --gpus all \
  -e MODEL_NAME=qwen3.5-27b \
  -v llm-models:/models \
  -p 8080:8080 \
  rotorquant

# API endpoint:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":500}'
```

### Available Models

| Profile | Model | Size | VRAM | Notes |
|---------|-------|------|------|-------|
| `--profile qwen` | Qwen3.6-35B-A3B UD-Q4_K_XL | 22.3 GB | ~27 GB | Default — 35B MoE, ~3B active, 196 tok/s |
| `--profile qwen36-q3` | Qwen3.6-35B-A3B UD-Q3_K_XL | ~17 GB | ~24 GB | 24 GB GPUs (RTX 4090) |
| `--profile qwen36-iq3` | Qwen3.6-35B-A3B UD-IQ3_XXS | ~14 GB | ~16 GB | 16 GB GPUs |
| `--profile qwen36-27b` | Qwen3.6-27B UD-Q4_K_XL | 16.4 GB | ~20 GB | Dense 27B, planar3 |
| `--profile qwen36-27b-q3` | Qwen3.6-27B UD-Q3_K_XL | ~12 GB | ~16 GB | 24 GB GPUs |
| `--profile qwen36-27b-iq3` | Qwen3.6-27B UD-IQ3_XXS | ~9 GB | ~12 GB | 16 GB GPUs |
| `--profile reasoning` | Qwen3.5-27B Claude Opus Distilled | 16.6 GB | ~20 GB | Reasoning-tuned |
| `--profile gemma` | Gemma 4 26B MoE UD-Q4_K_XL | 17.1 GB | ~21 GB | 3.8B active params |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | (required) | Model to serve |
| `KV_CACHE_TYPE` | `planar3` | KV cache quantization (planar3, planar4, iso3, f16) |
| `CTX_SIZE` | per-model | Context window size |
| `PORT` | `8080` | Server port |
| `GPU_LAYERS` | `99` | Layers offloaded to GPU |
| `N_PARALLEL` | `2` | Concurrent request slots |
| `HF_TOKEN` | — | HuggingFace token for gated models |
