# Qwen3.5-27B Dense — Quantization Guide for 16 GB GPUs

**Tested**: 2026-04-06
**Model**: Qwen3.5-27B (dense, 27B params)
**Source**: unsloth/Qwen3.5-27B-GGUF (Unsloth Dynamic imatrix variants)
**Method**: llama-perplexity on wikitext-2, ctx=2048
**Hardware**: RTX 5090 (all variants fit in VRAM for testing)

---

## Perplexity Results

| Variant | Size | PPL | Delta vs IQ4 | 16 GB iso4 ctx | 16 GB iso3 ctx |
|---------|-----:|----:|-----------:|---------------:|---------------:|
| **IQ4_XS** | 13.9 GB | **6.292** | baseline | ~24K | ~31K |
| **UD-Q3_K_XL** | 13.4 GB | **6.375** | +0.083 | ~32K | ~42K |
| **UD-IQ3_XXS** | 10.7 GB | **6.622** | +0.330 | ~74K | ~96K |
| UD-IQ2_M | 9.5 GB | 6.801 | +0.509 | ~92K | ~120K |
| UD-IQ2_XXS | 8.0 GB | 7.477 | +1.185 | ~115K | ~150K |

## Analysis

### The Quality Cliff

```
PPL
7.5 ┤ ▪ IQ2_XXS (8.0G) — significant degradation
    │
7.0 ┤
    │     ▪ IQ2_M (9.5G) — noticeable drop
6.8 ┤
    │
6.6 ┤         ▪ IQ3_XXS (10.7G) — good
    │
6.4 ┤             ▪ Q3_K_XL (13.4G) — very good
    │
6.2 ┤                 ▪ IQ4_XS (13.9G) — excellent
    └──┬──┬──┬──┬──┬──┬──┬──
       8  9  10 11 12 13 14 15  Size (GB)
```

**The cliff is between IQ3_XXS (6.62) and IQ2_M (6.80).** Above IQ3_XXS, quality
degrades gracefully. Below it, PPL jumps sharply — 2-bit loses too much information
in the dense 27B architecture.

### Key Findings

1. **IQ4_XS → UD-Q3_K_XL costs only +0.08 PPL but gains 33% more context** (24K→32K).
   This is the best trade on a 16 GB card.

2. **UD-IQ3_XXS is the context king** at +0.33 PPL for **3x more context** (24K→74K).
   Still very usable — 6.62 PPL is strong for a 27B model.

3. **2-bit is not worth it.** IQ2_M gains only 18K more context over IQ3_XXS
   but costs +0.18 PPL. IQ2_XXS is significantly degraded.

4. **Unsloth imatrix makes 3-bit viable.** UD-Q3_K_XL (6.375) is only +0.08
   from IQ4_XS (6.292) — the importance matrix preserves the critical weights.

## Recommendation for 16 GB GPUs

| Use Case | Pick | PPL | Context (iso4) |
|----------|------|----:|---------------:|
| **Coding / reasoning** | IQ4_XS | 6.29 | ~24K |
| **General assistant** | UD-Q3_K_XL | 6.38 | ~32K |
| **Long document / RAG** | UD-IQ3_XXS | 6.62 | ~74K |

**Default recommendation: UD-Q3_K_XL** — best balance of intelligence and context
for a 16 GB card. Only +1.3% PPL over the 4-bit variant, with 33% more context.

## Comparison: 16 GB vs 24 GB vs 32 GB

| GPU | Best Quant | PPL | Max iso4 ctx | Experience |
|-----|-----------|----:|-------------:|------------|
| 16 GB | UD-Q3_K_XL | 6.38 | ~32K | Good assistant, moderate context |
| 16 GB | UD-IQ3_XXS | 6.62 | ~74K | Slightly less sharp, huge context |
| 24 GB | Q4_K_M | ~6.27 | ~104K | Full quality, 100K+ context |
| 32 GB | Q4_K_M | ~6.27 | ~228K | Full quality, massive context |
