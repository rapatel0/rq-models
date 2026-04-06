# Quantization Guide — RotorQuant KV Cache + Weight Quantization

**Tested**: 2026-04-06
**Hardware**: RTX 5090 (32 GB)
**KV Cache**: RotorQuant iso4 (default, 3.8x compression)
**Method**: llama-perplexity on wikitext-2, ctx=2048

---

## Qwen3.5-27B Dense — Perplexity by Quantization

All variants from unsloth/Qwen3.5-27B-GGUF (Unsloth Dynamic imatrix).

| Variant | Size | PPL | Delta vs IQ4 | 16 GB iso4 ctx | 16 GB iso3 ctx |
|---------|-----:|----:|-----------:|---------------:|---------------:|
| **IQ4_XS** | 13.9 GB | **6.292** | baseline | ~24K | ~31K |
| **UD-Q3_K_XL** | 13.4 GB | **6.375** | +0.083 | ~32K | ~42K |
| **UD-IQ3_XXS** | 10.7 GB | **6.622** | +0.330 | ~74K | ~96K |
| UD-IQ2_M | 9.5 GB | 6.801 | +0.509 | ~92K | ~120K |
| UD-IQ2_XXS | 8.0 GB | 7.477 | +1.185 | ~115K | ~150K |

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

| Variant | Size | Est. Quality | 16 GB iso4 ctx | Notes |
|---------|-----:|-------------|---------------:|-------|
| **UD-IQ4_XS** | 12.5 GB | Best 4-bit | ~49K | Recommended for 16 GB |
| **UD-Q3_K_XL** | 12.0 GB | Good 3-bit | ~57K | More context headroom |
| UD-Q3_K_M | 11.7 GB | Good 3-bit | ~62K | Slightly smaller |
| UD-IQ3_XXS | 10.4 GB | Acceptable | ~83K | Max context |

> Gemma 4 profiles currently require mainline llama.cpp (not the RotorQuant
> fork). Once the fork rebases, RotorQuant KV cache will be available.

---

## Best Config per GPU Tier

### 16 GB (RTX 4060 Ti, RTX 5060, RTX 4080)

| Use Case | Model | Quant | PPL | Context (iso4) |
|----------|-------|-------|----:|---------------:|
| **Best quality** | Qwen 27B | IQ4_XS | 6.29 | ~24K |
| **Recommended** | **Qwen 27B** | **UD-Q3_K_XL** | **6.38** | **~32K** |
| Max context | Qwen 27B | UD-IQ3_XXS | 6.62 | ~74K |

### 24 GB (RTX 3090, RTX 4090)

| Use Case | Model | Quant | PPL | Context (iso4) |
|----------|-------|-------|----:|---------------:|
| **Recommended** | **Qwen 27B** | **Q4_K_M** | **~6.27** | **~131K** |
| More context | Qwen 27B | IQ4_XS | 6.29 | ~155K |

### 32 GB (RTX 5090)

| Use Case | Model | Quant | PPL | Context (iso4) |
|----------|-------|-------|----:|---------------:|
| **Recommended** | **Qwen 27B** | **Q4_K_M** | **~6.27** | **~252K** |
| Best quality | Qwen 27B | UD-Q4_K_XL | ~6.25 | ~240K |

### 40 GB (A100 40GB, A6000)

| Use Case | Model | Quant | PPL | Context (iso4) |
|----------|-------|-------|----:|---------------:|
| **Recommended** | **Qwen 27B** | **Q4_K_M** | **~6.27** | **~375K** |
| Best quality | Qwen 27B | Q5_K_M | ~6.22 | ~334K |
