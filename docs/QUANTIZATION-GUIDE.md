# Quantization Guide — RotorQuant KV Cache + Weight Quantization

**Tested**: 2026-04-06 (Qwen3.5-27B), 2026-04-22 (Qwen3.6-27B)
**Hardware**: RTX 5090 (32 GB)
**KV Cache defaults**: `planar3` for dense models (4.9x, 3.125 bpe); `iso3` for MoE (Qwen3.6-35B-A3B)
**Method**: llama-perplexity on wikitext-2, ctx=2048

---

## Qwen3.5-27B Dense — Perplexity by Quantization

All variants from unsloth/Qwen3.5-27B-GGUF (Unsloth Dynamic imatrix).

| Variant | Size | PPL | Delta vs IQ4 | 16 GB planar4 ctx | 16 GB planar3 ctx |
|---------|-----:|----:|-----------:|------------------:|------------------:|
| **IQ4_XS** | 13.9 GB | **6.292** | baseline | ~18K | ~24K |
| **UD-Q3_K_XL** | 13.4 GB | **6.375** | +0.083 | ~28K | ~36K |
| **UD-IQ3_XXS** | 10.7 GB | **6.622** | +0.330 | ~56K | ~74K |
| UD-IQ2_M | 9.5 GB | 6.801 | +0.509 | ~70K | ~92K |
| UD-IQ2_XXS | 8.0 GB | 7.477 | +1.185 | ~88K | ~115K |

### Quality Cliff

```
PPL
7.5 ┤ ▪ IQ2_XXS — significant degradation
    │
7.0 ┤
    │     ▪ IQ2_M — noticeable drop
6.8 ┤       ─────── quality cliff ───────
    │
6.6 ┤         ▪ IQ3_XXS — good
    │
6.4 ┤             ▪ Q3_K_XL — very good
    │
6.2 ┤                 ▪ IQ4_XS — excellent
    └──┬──┬──┬──┬──┬──┬──┬──
       8  9  10 11 12 13 14 15  Size (GB)
```

**The cliff is between IQ3_XXS (6.62) and IQ2_M (6.80).** Don't go below 3-bit.

### Key Findings — Qwen3.5-27B

1. **IQ4_XS → UD-Q3_K_XL costs only +0.08 PPL but gains 33% more context** (24K→32K).
   Best trade on a 16 GB card.

2. **UD-IQ3_XXS is the context king** at +0.33 PPL for 3x more context (24K→74K).
   Still very usable — 6.62 PPL is strong for a 27B model.

3. **2-bit is not worth it.** IQ2_M gains only 18K more context over IQ3_XXS
   but costs +0.18 PPL. IQ2_XXS is significantly degraded.

4. **Unsloth imatrix makes 3-bit viable.** UD-Q3_K_XL (6.375) is only +0.08
   from IQ4_XS (6.292).

---

## Gemma 4 26B MoE — Status

**Perplexity benchmarking is not reliable for Gemma 4.** The model's hybrid
architecture (sliding window + global attention + MoE routing) produces
nonsensical PPL values (5000-8000) with standard `llama-perplexity`. This
is a known limitation — Gemma 4 uses 1024-token sliding windows that break
the standard PPL evaluation loop.

Additionally, the RotorQuant llama.cpp fork does not yet support the `gemma4`
architecture. Gemma 4 RotorQuant KV cache support depends on the fork
rebasing to include the upstream `gemma4` model loader.

**For now, Gemma 4 quantization recommendations are based on the Unsloth
35B-A3B MoE benchmarks (similar architecture) and general MoE quantization
behavior:**

| Variant | Size | Est. Quality | 16 GB planar4 ctx | Notes |
|---------|-----:|-------------|------------------:|-------|
| **UD-IQ4_XS** | 12.5 GB | Best 4-bit | ~49K | Recommended for 16 GB |
| **UD-Q3_K_XL** | 12.0 GB | Good 3-bit | ~57K | More context headroom |
| UD-Q3_K_M | 11.7 GB | Good 3-bit | ~62K | Slightly smaller |
| UD-IQ3_XXS | 10.4 GB | Acceptable | ~83K | Max context |

> Gemma 4 profiles currently require mainline llama.cpp (not the RotorQuant
> fork). Once the fork rebases, RotorQuant KV cache will be available.

---

## Best Config per GPU Tier

### 16 GB (RTX 4060 Ti, RTX 5060, RTX 4080)

| Use Case | Model | Quant | PPL | Context (planar3) |
|----------|-------|-------|----:|------------------:|
| **Best quality** | Qwen3.5-27B | IQ4_XS | 6.29 | ~24K |
| **Recommended** | **Qwen3.5-27B** | **UD-Q3_K_XL** | **6.38** | **~36K** |
| Max context | Qwen3.5-27B | UD-IQ3_XXS | 6.62 | ~74K |
| Dense alt | Qwen3.6-27B | UD-IQ3_XXS | — | ~90K |

### 24 GB (RTX 3090, RTX 4090)

| Use Case | Model | Quant | PPL | Context (planar3) |
|----------|-------|-------|----:|------------------:|
| **Recommended** | **Qwen3.6-27B** | **UD-Q4_K_XL** | **7.09** | **~131K** |
| More context | Qwen3.6-27B | UD-Q3_K_XL | — | ~180K |
| Legacy | Qwen3.5-27B | Q4_K_M | ~6.64 | ~300K |

### 32 GB (RTX 5090)

**Use Qwen3.6-35B-A3B MoE** (`make run-qwen`) — 196 tok/s, 262K context, iso3 KV.

| Use Case | Model | Quant | PPL (f16) | KV | Context |
|----------|-------|-------|----------:|:--:|--------:|
| **Recommended** | **Qwen3.6-35B-A3B** | **UD-Q4_K_XL** | **6.13** | **iso3** | **262K** |
| Dense alt | Qwen3.6-27B | UD-Q4_K_XL | 7.09 | planar3 | 131K |

### 40 GB (A100 40GB, A6000)

| Use Case | Model | Quant | PPL (f16) | KV | Context |
|----------|-------|-------|----------:|:--:|--------:|
| **Recommended** | **Qwen3.6-35B-A3B** | **UD-Q4_K_XL** | **6.13** | **iso3** | **~400K** |
| Dense alt | Qwen3.6-27B | UD-Q4_K_XL | 7.09 | planar3 | ~375K |

---

## Draft Model VRAM Cost (Sprint 004 — DFlash speculative)

The two new DFlash profiles add a draft model alongside the target. VRAM
accounting therefore picks up two new line items: draft weights and draft
KV cache. The Phase 1 source-spike confirmed (BENCHMARK-REPORT.md §10) that
**checkpoint snapshots live in host pageable RAM, not VRAM** — the per-verify
recurrent-state byte-copy goes to `std::vector<uint8_t>`, never to a device
buffer. Draft VRAM is therefore exactly draft weights + draft KV with no
snapshot term. The fork's optional `vram_seq_checkpoint` shadow buffer adds
~157 MB (27B) or ~66 MB (35B) of device memory if enabled, but that is a
one-time allocation, not a per-checkpoint cost.

### Draft GGUF sizes (pinned in `docker/entrypoint.sh:60-62`)

| Profile draft key | HF repo | File | Size |
|---|---|---|---:|
| `qwen3.6-27b-dflash` | `spiritbuun/Qwen3.6-27B-DFlash-GGUF` @ `5e4442a` | `dflash-draft-3.6-q4_k_m.gguf` | 0.96 GB |
| `qwen3.6-35b-dflash` | `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test` @ `3813f31` | `Qwen3.6-35B-A3B-DFlash-q8_0.gguf` | 0.48 GB |

Sizes from the HF API at the pinned SHAs. Both drafts are small relative to
their targets — 4–6% of target weight size — so draft weight VRAM is not
the binding constraint.

### Draft KV at typical contexts

The DFlash draft is itself a hybrid Qwen3.6 model and inherits per-layer
recurrent-state size from the target (see BENCHMARK-REPORT.md §10 Phase 2
table — partial state is per-model-fixed, context-independent). Full-attention
KV scales with context and is RotorQuant-quantized at the same factor as the
target.

| Profile | Draft KV type | Recurrent (fixed) | Full-attn KV @ 65K | Full-attn KV @ 131K |
|---------|:-------------:|------------------:|-------------------:|--------------------:|
| `qwen36-27b-dflash` | planar3 | ~157 MB | TBD | TBD |
| `qwen36-dflash` | iso3 | ~66 MB | TBD (gate is 65K) | n/a (capped at 65K) |

Full-attention KV cells are TBD pending an actual profile boot with a working
draft GGUF — measurement blocked on the same Phase 3 / Phase 5 issue as the
L2/L3/L4 gates.

### Per-profile RTX 5090 (32 GB) accounting

| Profile | Target weights | Draft weights | Target KV | Draft KV | Snapshot (host RAM, off-VRAM) | VRAM headroom (32 GB) |
|---------|---------------:|--------------:|----------:|---------:|------------------------------:|---------------------:|
| `qwen36-27b-dflash` (planar3, 131K) | 16.4 GB | 0.96 GB | TBD | TBD | ~157 MB (host) | TBD |
| `qwen36-dflash` (iso3, 65K) | 20.8 GB | 0.48 GB | TBD | TBD | ~66 MB (host) | TBD |

The 65K context cap on `qwen36-dflash` is conservative pending Phase 5
measurement; Phase 1's spike originally suggested no VRAM headroom impact
from snapshots, which Phase 2 confirmed. Production cap may rise once draft
KV at 131K is measured.
