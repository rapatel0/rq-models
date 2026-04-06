# Sprint 001 Critique: Claude Draft vs Gemini Draft

**Reviewer**: Codex (Claude Opus 4.6)
**Date**: 2026-03-24
**Inputs**: SPRINT-001-INTENT.md, SPRINT-001-CLAUDE-DRAFT.md, SPRINT-001-GEMINI-DRAFT.md

---

## 1. Strengths of Each Draft

### Claude Draft

- **Thoroughness**: At 306 lines with 6 phases, 12 DoD items, and 7 risks, this is a production-grade sprint document. Every algorithm has inline pseudocode (lines 70–101) that a developer can translate directly into PyTorch.
- **Architecture diagram** (lines 42–58): The ASCII component tree clearly shows the data flow from attention layer → OutlierChannelSplitter → packed storage, including concrete tensor shapes and byte counts per head per token.
- **GQA awareness** (risk table, line 277): Explicitly calls out that quantization must happen at `num_key_value_heads` granularity (8 heads), not `num_query_heads` (32). This is a critical correctness detail neither the intent nor Gemini mentions.
- **Multi-dimension codebooks** (line 154): Correctly identifies that outlier splitting requires codebooks at d=32 and d=96 in addition to d=128 — an easy-to-miss detail.
- **Rotation matrix scope** (line 116): Correctly specifies that Π operates on the partition dimension (32 or 96), not the full 128.
- **Sharing Π/S across layers** (open question 5, line 305): Proposes sharing random matrices across all 64 layers, saving ~64× memory. This is valid because TurboQuant is data-oblivious by design.
- **Version pinning** (line 289): Pins `transformers >= 4.40.0` and acknowledges API instability of `DynamicCache` across versions (risk table, line 278).

### Gemini Draft

- **Conciseness**: At 99 lines, the draft is readable in one pass. Phase structure is clean and each phase has a clear goal sentence.
- **Correct 3.5-bit split** (line 57): Uses 64 channels @ 4-bit + 64 channels @ 3-bit, which is arithmetically correct: (64×4 + 64×3)/128 = 3.5. This is the *only* draft that gets this right (see §3 below).
- **Method signatures** (lines 42–51): Provides concrete Python signatures for both `TurboQuantMSE` and `TurboQuantProd`, including return types, making the API contract unambiguous.
- **Broader cache superclass** (line 22): Subclasses `transformers.Cache` rather than `DynamicCache`, which is more resilient if Qwen3.5-27B uses a non-standard cache.
- **torch.compile question** (line 98): Raises whether `torch.compile` can optimize the matrix multiplications, relevant for the <1ms latency target.

---

## 2. Algorithm Correctness Issues

### 2.1 QJL Projection Matrix Scaling — Gemini line 16

The intent document (line 59) specifies S_{i,j} ~ N(0, 1). Gemini (line 16) changes this to N(0, 1/d). This is not a cosmetic difference — it changes the magnitude of the dequantized QJL residual.

With S ~ N(0, 1), each entry of S^T · qjl (where qjl ∈ {±1}^d) has variance d, so ‖S^T · qjl‖ ~ d. The dequant formula (√(π/2) / d) · γ · S^T · qjl then produces an O(γ) correction, matching the residual norm. With S ~ N(0, 1/d), ‖S^T · qjl‖ ~ √d, and the same dequant formula produces a correction that is √d too small.

**Verdict**: Gemini must either use N(0, 1) as stated in the paper, or adjust the dequant scaling to (√(π/2) / √d). The draft as written will produce incorrect (attenuated) residual corrections.

### 2.2 TurboQuantMSE vs TurboQuantProd in the Outlier Splitter — Claude lines 49–51

Claude's architecture diagram (lines 49–51) shows:
```
OutlierChannelSplitter
├── TurboQuantMSE (outlier channels, 3-bit)
└── TurboQuantMSE (non-outlier channels, 2-bit)
```

But the QuantizedKVStore (lines 53–54) includes `qjl_bits` and `gamma` fields — these are TurboQuantProd outputs that TurboQuantMSE does not produce. Furthermore, the Phase 3 OutlierSplitter (line 182) "holds two TurboQuantMSE instances," yet the memory layout stores Prod-specific metadata.

This is internally inconsistent. For KV cache quantization, attention computes Q·K^T (an inner product), so the unbiased estimator from TurboQuantProd is preferred for K (and arguably V, since softmax(scores)·V is also a dot product over the sequence dimension). The draft must clarify:
- Whether the OutlierSplitter wraps TurboQuantProd or TurboQuantMSE instances.
- If TurboQuantProd: the bit accounting changes — a "3-bit Prod" uses 2-bit MSE + 1-bit QJL internally.
- If TurboQuantMSE: remove qjl_bits and gamma from the storage layout and accept biased inner products.

### 2.3 Neither Draft Specifies MSE vs Prod Usage Per Cache

The intent defines both algorithms but never prescribes which to use for K vs V. Neither draft makes this decision explicit. At minimum: K cache should use TurboQuantProd (attention scores = Q·K^T), and V cache could use either (V retrieval = attn_weights · V, also a dot product). This choice affects storage layout, bit budget, and DoD test design.

---

## 3. Outlier Channel Splitting — Arithmetic Bug

### 2.5-bit Split

The intent (line 78) claims: (32×3 + 96×2)/128 = 2.5. This is arithmetically wrong:

> 32×3 + 96×2 = 96 + 192 = 288; 288/128 = **2.25**, not 2.5.

To achieve exactly 2.5 bits with 3-bit outlier / 2-bit regular:
> x×3 + (128−x)×2 = 2.5×128 = 320 → x = **64**, not 32.

Both drafts copy this error without verification:
- **Claude** (line 108): "(32×3 + 96×2) / 128 = 2.5" — wrong.
- **Gemini** (line 56): "32 channels (3-bit) + 96 channels (2-bit)" — doesn't recompute, inherits the error.

### 3.5-bit Split

- **Claude** (line 113): "(32×4 + 96×3) / 128 = 3.5" → actual value is (128+288)/128 = **3.25**. Wrong.
- **Gemini** (line 57): "(64×4 + 64×3) / 128" → (256+192)/128 = **3.5**. Correct.

### Recommendation

The correct split for both configurations, assuming a 1-bit gap between outlier and regular precision, is **64/64**, not 32/96:

| Config | Outlier channels | Outlier bits | Regular channels | Regular bits | Effective |
|--------|-----------------|--------------|-----------------|--------------|-----------|
| 2.5-bit | 64 | 3 | 64 | 2 | 2.50 |
| 3.5-bit | 64 | 4 | 64 | 3 | 3.50 |

This must be reconciled with the paper. If the paper truly uses 32/96, the effective bits are 2.25 and 3.25, and the naming should be updated accordingly. Either way, the current arithmetic in the intent and both drafts is self-contradictory.

---

## 4. HuggingFace Integration Strategy

### Claude's Approach (lines 119–139)

- Subclasses `DynamicCache` and overrides `update()` / property accessors.
- Patches `model.generate()` to inject the custom cache.
- **Strength**: Minimal invasiveness — doesn't touch attention forward().
- **Risk not addressed**: `DynamicCache.update()` returns `(key_states, value_states)` which the attention layer uses *in the same forward pass*. If `update()` quantizes on write, the returned values must be either (a) the original unquantized tensors (for the current token's attention) or (b) dequantized approximations. Neither draft clarifies this. Using dequantized values for the current token's own attention score would introduce unnecessary error; the current token's KV should be used at full precision for the current step and only quantized for future steps. This is a subtle but important correctness point.
- **Missing**: No discussion of how `key_cache[layer_idx]` and `value_cache[layer_idx]` are accessed during the attention computation. The standard `DynamicCache` stores a list of tensors; the custom cache must intercept reads to dequantize the packed representation into a full `[batch, heads, seq, head_dim]` tensor. The draft mentions this conceptually but doesn't address the performance implication of dequantizing the entire sequence every step (O(seq_len × d) per layer per token).

### Gemini's Approach (lines 63–64)

- Mentions replacing `DynamicCache` with `TurboKVCache` but provides no implementation detail.
- Subclasses `transformers.Cache` (line 22) — this is the abstract base, not the concrete class. This is potentially more robust but requires implementing more methods.
- **Gap**: No discussion of the `update()` return value semantics, streaming append, or dequantization-on-read strategy.

### Shared Gap: Qwen3.5-27B Attention Implementation

Neither draft investigates whether Qwen3.5-27B uses Flash Attention, SDPA, or a custom attention kernel. Flash Attention typically manages its own KV cache layout and may bypass the `DynamicCache` interface entirely. If Qwen3.5-27B defaults to `torch.nn.functional.scaled_dot_product_attention` (SDPA), the cache hook approach works. If it uses a custom CUDA kernel, the model patching strategy fails silently. This should be a Phase 0 / blocking investigation.

---

## 5. Definition of Done Completeness

| Criterion | Claude | Gemini | Gap |
|-----------|--------|--------|-----|
| MSE distortion bound | ✅ 10% tolerance, b=2,3,4, d=128 (line 255) | ✅ 5% tolerance, b=2,3 (line 75) | Gemini's 5% is tighter but only tests b=2,3; misses b=4 |
| Inner product bias | ✅ <1% relative bias (line 256) | ✅ "zero mean bias" (line 76) | Gemini doesn't specify a numeric threshold |
| Inner product distortion | ✅ with bound formula (line 257) | ❌ not mentioned | Gemini omits this entirely |
| Outlier bit calculation | ✅ (line 258) | ❌ | Gemini has no DoD item for the splitter |
| KV cache roundtrip error | ✅ (line 259) | ❌ | Gemini has no DoD for cache-level distortion |
| E2E generation | ✅ 2.5-bit and 3.5-bit (line 260) | ✅ (line 77) | Comparable |
| NaN/Inf check | ✅ during 1000-token gen (line 261) | ❌ | Gemini omits |
| Perplexity | ✅ wikitext-2, ≤ baseline + 0.1 nats (line 262) | ❌ | Gemini omits perplexity entirely |
| NIAH | ✅ ≥ 0.99 at 32k (line 263) | ✅ ≥ 0.99 at 32k (line 78) | Comparable |
| Memory reduction | ❌ no specific DoD item | ✅ ≥ 4.5× (line 79) | Claude mentions it as a Phase 5 task but not DoD |
| Latency | ✅ < 1ms/token (line 264) | ✅ < 2ms/token (line 80) | Gemini's 2ms is looser than the intent's 1ms target |
| Tests pass | ✅ (line 265) | ❌ not stated | Gemini assumes it implicitly |
| requirements.txt | ✅ (line 266) | ❌ | Gemini omits |

**Summary**: Claude's DoD is substantially more complete (12 items vs 6). Gemini's DoD is missing distortion bounds for Prod, cache roundtrip verification, NaN checks, perplexity, and the requirements file. However, Gemini uniquely includes a memory reduction criterion.

### Missing from Both DoDs

1. **Batch > 1 correctness**: No DoD item verifies that quantized KV cache works correctly with batch_size > 1 and variable-length sequences.
2. **bfloat16 input handling**: No DoD item checks that bfloat16 model outputs are correctly upcasted before rotation and quantization, and that results are cast back.
3. **Deterministic reproducibility**: No DoD item verifies that the same seed produces bit-identical quantized outputs across runs.
4. **Streaming token-by-token correctness**: No DoD item compares a KV cache built incrementally (1 token at a time, as in `generate()`) against one built from the full sequence at once.

---

## 6. Missing Edge Cases

### 6.1 Batch > 1

- **Claude** (line 199): Mentions "Handle batch_size > 1 and variable sequence lengths" as a Phase 4 task, but provides no detail on how variable-length sequences interact with the packed storage format. When sequences in a batch have different lengths, the KV cache must either pad to max_seq_len (wasting memory) or use a ragged representation (complicating dequant). Neither approach is discussed.
- **Gemini**: Does not mention batch > 1 at all.

### 6.2 GQA Head Layout

- **Claude** (line 277): Correctly notes that K/V have 8 heads while Q has 32. Quantization operates on K/V heads; dequantized K/V must be broadcast/repeated to match Q heads during attention. This broadcast logic is not in the KV cache phase tasks.
- **Gemini**: Does not mention GQA. With `num_key_value_heads=8` and `num_query_heads=32`, each KV head serves 4 query heads. The KV cache stores 8 heads, but attention computation needs the head dimension aligned. Missing this will cause shape errors.

### 6.3 bfloat16

- **Claude** (open question 3, line 301): Raises the question and proposes float32 for rotation (risk line 275). But this is left as an open question — it should be a concrete design decision with a test.
- **Gemini** (line 85): Mentions "numerical stability" of QR decomposition but doesn't discuss the bfloat16→float32 casting needed for rotation. bfloat16 has only 8 mantissa bits; multiplying by a 128×128 orthogonal matrix in bfloat16 accumulates significant rounding error that could violate distortion bounds.
- **Neither draft** includes a DoD item that distortion bounds hold for bfloat16 inputs (the most common dtype for Qwen3.5-27B).

### 6.4 Streaming Generation

- **Claude** (line 198, line 303–304): Acknowledges streaming append and raises pre-allocation vs. list-append as an open question. This is the right concern — Python list append of small tensors creates GC pressure and fragmentation at high token counts.
- **Gemini**: Does not discuss streaming storage strategy at all.
- **Neither draft** addresses: what happens when context exceeds a pre-allocated buffer? Does the cache grow dynamically? Is there a maximum context length?

### 6.5 Prefill vs Decode Phase

Neither draft distinguishes the prefill phase (processing the full prompt in one forward pass, producing many KV vectors at once) from the decode phase (one token at a time). During prefill, the input to `update()` has shape `[batch, heads, prompt_len, head_dim]` — the quantization path must handle arbitrary `seq_len`, not just `seq_len=1`. This is especially relevant for efficiency: batch-quantizing the full prompt at once is faster than looping token-by-token.

---

## 7. Risk Gaps

### Risks Missing from Claude

1. **Flash Attention / SDPA bypass**: If Qwen3.5-27B's attention implementation uses Flash Attention (which manages its own KV memory), the `DynamicCache` subclass approach may be silently ignored. This is higher-risk than the "custom KV cache" item (line 272) because Flash Attention is the *default* in modern transformers.
2. **Dequantization cost at long contexts**: Every generated token requires dequantizing the *entire* K and V cache (all past tokens) to compute attention. At 32k tokens, this is 32k × 8 heads × 128 dims × 2 (K+V) = ~65M values dequantized per layer, per generated token. At 64 layers, that's ~4B dequant operations per token. This may dominate the <1ms budget.
3. **Codebook quantization error at d=32**: The distortion bounds are asymptotic in d. At d=32 (the outlier partition), the Beta((d-1)/2, (d-1)/2) distribution is less concentrated than at d=128, so the codebook MSE cost may exceed the theoretical bound by more than 10%. No risk or test addresses this.

### Risks Missing from Gemini

1. **All of the above**, plus:
2. **GQA head mismatch**: Not mentioned at all.
3. **Batch > 1 variable-length handling**: Not mentioned.
4. **bfloat16 numerical degradation**: Not specifically addressed.
5. **Outlier split arithmetic error**: The 2.5-bit split is inherited from the intent without verification.
6. **transformers version sensitivity**: Pins `>= 4.38.0` but doesn't flag DynamicCache API instability.
7. **No risk for the Prod vs MSE algorithm choice**: The draft defines both but never discusses the implications of choosing one over the other for K and V.

### Risk Missing from Both

- **No rollback strategy**: If the quantized cache produces quality degradation that's only visible at long contexts (>8k tokens), there's no mechanism to fall back to full-precision mid-generation. A hybrid approach (quantize only tokens beyond a threshold) would be a useful mitigation.

---

## 8. Summary Recommendations

| # | Issue | Severity | Action |
|---|-------|----------|--------|
| 1 | 2.5-bit split arithmetic (32/96 → 2.25, not 2.5) | **Blocking** | Reconcile with paper. If 2.5 is the target, use 64/64 split. |
| 2 | Claude 3.5-bit split arithmetic (32/96 → 3.25, not 3.5) | **Blocking** | Fix to 64/64 (as Gemini does) or adjust effective-bit label. |
| 3 | Gemini S ~ N(0, 1/d) scaling | **Blocking** | Revert to N(0, 1) per paper, or adjust dequant formula. |
| 4 | MSE vs Prod usage for K and V caches unspecified | **High** | Decide and document: Prod for K (inner product), MSE or Prod for V. |
| 5 | Claude: MSE/Prod inconsistency in architecture diagram vs storage layout | **High** | Align OutlierSplitter to use Prod instances; update diagram. |
| 6 | No Flash Attention / SDPA investigation | **High** | Add Phase 0 task: verify Qwen3.5-27B attention path on target transformers version. |
| 7 | Dequantization cost at long context not analyzed | **Medium** | Add throughput benchmark at 4k, 16k, 32k tokens; revise latency DoD. |
| 8 | No batch > 1 DoD criterion | **Medium** | Add test: batch_size=4, variable seq lengths, verify correctness. |
| 9 | No bfloat16 distortion DoD criterion | **Medium** | Add test: run distortion bounds with bfloat16 inputs after float32 rotation. |
| 10 | Gemini DoD missing 6 of Claude's 12 criteria | **Medium** | Adopt Claude's DoD as baseline; add memory reduction from Gemini. |
| 11 | Prefill vs decode path not distinguished | **Medium** | Ensure `update()` handles seq_len > 1 efficiently during prefill. |
| 12 | Codebook distortion at d=32 may exceed bounds | **Low** | Add distortion test specifically at d=32; widen tolerance if needed. |
| 13 | Gemini latency target (2ms) looser than intent (1ms) | **Low** | Align to intent: < 1ms/token on A100. |

### Overall Verdict

**Use the Claude draft as the base**, incorporating these fixes:
1. Adopt Gemini's 64/64 channel split (or verify against the paper).
2. Fix the MSE vs Prod inconsistency — the OutlierSplitter should wrap TurboQuantProd for K cache at minimum.
3. Add Gemini's memory reduction DoD criterion.
4. Resolve S matrix scaling to N(0, 1).
5. Add blocking investigation of Qwen3.5-27B's attention backend (Flash Attention / SDPA) before Phase 4.
6. Add DoD items for batch > 1, bfloat16, and streaming token-by-token consistency.
