# Sprint 001: TurboQuant KV Cache Quantization for Qwen3.5-27B

**Status**: Planning
**Created**: 2026-03-25
**Model**: Qwen3.5-27B (HuggingFace transformers)
**Target**: 2.5-bit and 3.5-bit KV cache compression, quality-neutral generation

---

## Overview

TurboQuant (arXiv:2504.19874) is an online, data-oblivious vector quantizer that achieves
near-optimal distortion by (1) randomly rotating input vectors to induce near-independence
across coordinates and (2) applying precomputed optimal scalar quantizers (Lloyd-Max on
Beta distribution) per coordinate. A second variant, TurboQuant_prod, adds a 1-bit
Quantized Johnson-Lindenstrauss (QJL) pass on the MSE residual to produce **unbiased**
inner product estimates — the exact operation that attention scores require.

This sprint implements TurboQuant as a drop-in KV cache compressor for Qwen3.5-27B.
K cache uses TurboQuant_prod for unbiased attention scores; V cache uses TurboQuant_mse
for simpler, lower-overhead reconstruction. An outlier channel split (fixed first-N
channels at higher bit-width) achieves 2.5-bit and 3.5-bit effective precision without
any calibration data. The algorithm is fully online: each token's KV vectors are
quantized immediately on insertion and dequantized on retrieval.

At 3.5-bit, the paper reports quality-neutral performance on Llama-3.1-8B (NIAH recall
0.997, matching full precision) while compressing the KV cache by >4.5×. This sprint
validates those claims for Qwen3.5-27B and produces a reusable `turboquant` Python package.

---

## Use Cases

1. **Long-context single-GPU inference**: At 3.5-bit, a 128k-token Qwen3.5-27B context
   fits in ~40 GB VRAM instead of ~180 GB (fp16), enabling A100 deployment.

2. **Throughput-sensitive batch serving**: Compressed KV cache reduces HBM bandwidth
   during autoregressive decoding — the primary serving bottleneck for large models.

3. **Quality-critical RAG pipelines**: TurboQuant_prod's unbiased inner product ensures
   attention distributions are not systematically shifted, unlike KIVI which shows
   recall degradation on needle-in-a-haystack tasks.

4. **Drop-in compression for other GQA models**: The architecture-agnostic design
   (operates per head_dim vector) is directly reusable for Llama, Mistral, and other
   models with head_dim=128 and any num_key_value_heads.

---

## Architecture

### Component Hierarchy

```
Qwen3.5-27B (HuggingFace transformers)
│
└── attention layer × 64
    ├── Q projection          [unchanged]
    └── KV projection ──────► TurboKVCache
                               │
                     ┌─────────┴──────────┐
                     │ K path             │ V path
                     │ TurboQuantProd     │ TurboQuantMSE
                     │   Outlier 32ch     │   Outlier 32ch
                     │   @ (b-1) MSE      │   @ b-bit
                     │   + 1-bit QJL      │
                     │   Regular 96ch     │   Regular 96ch
                     │   @ (b-2) MSE      │   @ (b-1)-bit
                     │   + 1-bit QJL      │
                     └────────────────────┘
                               │
                     QuantizedKVStore (per layer)
                     K: k_out_idx, k_out_qjl, k_out_gamma
                        k_reg_idx, k_reg_qjl, k_reg_gamma
                     V: v_out_idx
                        v_reg_idx
```

### Mathematical Primitives

#### Lloyd-Max Codebook (`turboquant/codebook.py`)

The KV vector coordinates after random rotation follow the Beta distribution:
```
f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)   for x ∈ [-1, 1]
```
In high dimensions (d=128) this converges to N(0, 1/d). The optimal b-bit scalar
quantizer for this distribution is found by Lloyd-Max iteration:

```python
# Initialize 2^b uniform centroids in [-1, 1]
# Iterate:
#   boundaries[i] = (centroids[i] + centroids[i+1]) / 2
#   centroids[i] = E[X | X ∈ (boundaries[i-1], boundaries[i])]  # quadrature
# Until convergence (Δ < 1e-8)
```

Precomputed for d ∈ {32, 96, 128} × b ∈ {1, 2, 3, 4, 5} and stored as tensors.
The codebook is symmetric (c_i = -c_{2^b - 1 - i}), so only the positive half is stored.

#### TurboQuantMSE (`turboquant/core.py`, Algorithm 1)

```
Global setup (per partition dimension d_p, bit-width b):
  Π ∈ R^(d_p × d_p)  ← qr(randn(d_p, d_p))[0]   # random orthogonal
  codebook ← load_codebook(d_p, b)                  # 2^b centroids ∈ [-1, 1]
  # Π and codebook are shared across all layers (data-oblivious)

QUANT_mse(x: Tensor[..., d_p]) → idx: Tensor[..., d_p]:
  norm = ‖x‖₂  per vector (preserve for rescaling)
  x_unit = x / norm                              # project to unit sphere
  y = x_unit @ Π.T                              # rotate: [..., d_p]   (float32)
  idx = argmin_k |y[..., j] - codebook[k]|      # broadcast: [..., d_p] (uint8)
  return idx, norm

DEQUANT_mse(idx: [..., d_p], norm: [...]) → x̃: Tensor[..., d_p]:
  ỹ = codebook[idx]                             # lookup: [..., d_p]
  x̃_unit = ỹ @ Π                               # inverse rotate: [..., d_p]
  return x̃_unit * norm[..., None]               # rescale to original magnitude
```

**Note on float32 upcasting**: All rotation and quantization operations must be
performed in float32, even when the model operates in bfloat16. Cast inputs to
float32 before QUANT, cast outputs back to the model dtype after DEQUANT.

#### TurboQuantProd (`turboquant/core.py`, Algorithm 2)

```
Global setup (per partition dimension d_p, bit-width b):
  mse_quant ← TurboQuantMSE(d_p, b-1)          # one less bit for MSE stage
  S ∈ R^(d_p × d_p)  ← randn(d_p, d_p)        # N(0,1) — NOT N(0,1/d)
  # S is shared across all layers (data-oblivious)

QUANT_prod(x: Tensor[..., d_p]) → (idx, qjl, γ):
  idx, norm = mse_quant.QUANT(x)
  x̃_mse = mse_quant.DEQUANT(idx, norm)
  r = x - x̃_mse                                # residual (original magnitude)
  γ = ‖r‖₂  per vector                         # scalar, float16
  qjl = sign(S @ r.T).T                        # {-1, +1}^(d_p), stored as int8
  return idx, qjl, γ

DEQUANT_prod(idx, qjl, γ, original_norm) → x̃:
  x̃_mse = mse_quant.DEQUANT(idx, original_norm)
  x̃_qjl = (√(π/2) / d_p) · γ[..., None] · (S.T @ qjl.T).T
  return x̃_mse + x̃_qjl
```

**Unbiasedness**: E[⟨y, x̃⟩] = ⟨y, x⟩ for any query vector y (proved in paper Theorem 2)

#### OutlierSplitter (`turboquant/outlier.py`)

```python
@dataclass
class BitConfig:
    label: str                   # e.g. "2.5-bit"
    k_effective_bits: float      # e.g. 2.5
    v_effective_bits: float      # e.g. 2.5
    outlier_count: int           # e.g. 32  (always first-N channels)
    outlier_k_bits: int          # bit-width for outlier K channels
    outlier_v_bits: int          # bit-width for outlier V channels
    regular_k_bits: int          # bit-width for regular K channels
    regular_v_bits: int          # bit-width for regular V channels

PRESET_2_5BIT = BitConfig(
    label="2.5-bit",
    k_effective_bits=2.25,  # (32×3 + 96×2) / 128
    v_effective_bits=2.25,
    outlier_count=32,
    outlier_k_bits=3, outlier_v_bits=3,
    regular_k_bits=2, regular_v_bits=2,
)
PRESET_3_5BIT = BitConfig(
    label="3.5-bit",
    k_effective_bits=3.5,   # (64×4 + 64×3) / 128
    v_effective_bits=3.5,
    outlier_count=64,
    outlier_k_bits=4, outlier_v_bits=4,
    regular_k_bits=3, regular_v_bits=3,
)
```

Outlier channel selection: **fixed first-N indices** — fully data-oblivious. Since Π
randomizes the input, "first 32 post-rotation channels" are statistically equivalent
to any other 32, matching the paper's theoretical guarantees.

#### TurboKVCache (`turboquant/kv_cache.py`)

Subclasses `transformers.cache_utils.DynamicCache`. Overrides `update()` to quantize
on write; overrides `key_cache` and `value_cache` properties to dequantize on read.

```python
class TurboKVCache(DynamicCache):
    def update(
        self,
        key_states: Tensor,    # [batch, num_kv_heads, 1, head_dim]
        value_states: Tensor,  # [batch, num_kv_heads, 1, head_dim]
        layer_idx: int,
        cache_kwargs: dict,
    ) -> tuple[Tensor, Tensor]:
        # Quantize and append to layer storage
        # Return dequantized K (full sequence), V (full sequence)
        ...

    @property
    def key_cache(self) -> list[Tensor]:
        # Dequantize all stored K tokens per layer
        ...

    @property
    def value_cache(self) -> list[Tensor]:
        # Dequantize all stored V tokens per layer
        ...
```

Storage per layer (2.5-bit config, d=128):
```
K storage (TurboQuantProd path):
  k_out_idx:   list[Tensor[batch, kv_heads, seq, 32]]  uint8  (3-bit packed in 8-bit)
  k_out_qjl:   list[Tensor[batch, kv_heads, seq, 32]]  int8   (±1)
  k_out_gamma: list[Tensor[batch, kv_heads, seq]]       float16
  k_reg_idx:   list[Tensor[batch, kv_heads, seq, 96]]  uint8  (2-bit packed in 8-bit)
  k_reg_qjl:   list[Tensor[batch, kv_heads, seq, 96]]  int8
  k_reg_gamma: list[Tensor[batch, kv_heads, seq]]       float16

V storage (TurboQuantMSE path):
  v_out_idx:   list[Tensor[batch, kv_heads, seq, 32]]  uint8
  v_reg_idx:   list[Tensor[batch, kv_heads, seq, 96]]  uint8
```

**GQA handling**: KV states have shape `[batch, num_kv_heads=8, seq, head_dim=128]`.
Quantize and store at `num_kv_heads` granularity. HuggingFace's attention module
handles broadcasting to `num_query_heads=32` internally.

---

## Implementation

### Phase 1: Offline Codebook Generation (Days 1–2, ~15% effort)

**Goal**: Precompute Lloyd-Max codebooks for all required (d, b) combinations.

**Files:**
- `scripts/generate_codebooks.py`
- `turboquant/codebook.py`
- `codebooks/d{d}_b{b}.pt` for d ∈ {32, 96, 128}, b ∈ {1, 2, 3, 4, 5}

**Tasks:**
- [ ] Implement Lloyd-Max iteration: `scipy.integrate.quad` for centroid update;
      convergence threshold 1e-8; max 500 iterations
- [ ] Validate symmetry: `sum(codebook) ≈ 0`, `codebook[-i-1] ≈ -codebook[i]`
- [ ] Validate MSE cost C(f_X, b) within 1% of paper values: b=1→0.36/d, b=2→0.117/d, b=3→0.03/d, b=4→0.009/d
- [ ] Generate for d ∈ {32, 96, 128}, b ∈ {1, 2, 3, 4, 5}; save as float32 tensors
- [ ] `codebook.py`: module-level dict cache; `load_codebook(d, b) → Tensor`
- [ ] Test: verify codebook MSE ≤ (√3π/2)·(1/4^b)·1.1 for d=128, b=1..4 (Monte Carlo, 10k Beta samples)

### Phase 2: Core Algorithm Implementation (Days 2–5, ~25% effort)

**Goal**: Implement TurboQuantMSE and TurboQuantProd with verified distortion bounds.

**Files:**
- `turboquant/core.py`
- `turboquant/config.py`
- `tests/test_codebook.py`
- `tests/test_core.py`

**Tasks:**
- [ ] `TurboQuantMSE.__init__(d: int, b: int, seed: int = 42)`:
      generates Π via `torch.linalg.qr(torch.randn(d, d))[0]` in float32; stores codebook
- [ ] `TurboQuantMSE.quantize(x: Tensor) → (idx: Tensor, norm: Tensor)`:
      float32 upcast → normalize → rotate → nearest-centroid lookup (vectorized broadcast)
- [ ] `TurboQuantMSE.dequantize(idx: Tensor, norm: Tensor) → Tensor`:
      lookup → inverse rotate → rescale → cast to input dtype
- [ ] Unit test: rotation correctness `‖Πᵀ·Π - I‖_F < 1e-6`
- [ ] Unit test: MSE D_mse ≤ (√3π/2)·(1/4^b)·1.1 for b=2,3,4 (10k unit vectors, d=128)
- [ ] `TurboQuantProd.__init__(d: int, b: int, seed: int = 42)`:
      stores `mse_quant = TurboQuantMSE(d, b-1)`; generates S ∈ R^(d×d) with S_{i,j} ~ N(0,1)
- [ ] `TurboQuantProd.quantize(x) → (idx, qjl, gamma)`:
      MSE quantize → compute residual → QJL sign projection → residual norm
- [ ] `TurboQuantProd.dequantize(idx, qjl, gamma, norm) → Tensor`:
      MSE dequantize + QJL correction term (√(π/2) / d) · γ · Sᵀ · qjl
- [ ] Unit test: bias |E[⟨y, x̃⟩] − ⟨y, x⟩| / |⟨y, x⟩| < 0.01 for b=3 (1k paired samples)
- [ ] Unit test: D_prod ≤ (√3π²‖y‖²/d)·(1/4^b)·1.1 for b=2,3,4

### Phase 3: Outlier Channel Splitter (Days 5–7, ~15% effort)

**Goal**: Implement configurable channel partitioning with two quantizers per head.

**Files:**
- `turboquant/outlier.py`
- `turboquant/config.py` (BitConfig, presets)
- `tests/test_outlier.py`

**Tasks:**
- [ ] Define `BitConfig` dataclass with presets: `PRESET_2_5BIT`, `PRESET_3_5BIT`
- [ ] `OutlierSplitter.__init__(config: BitConfig, head_dim: int = 128, seed: int = 42)`:
      creates `k_out_quant = TurboQuantProd(outlier_count, outlier_k_bits)`
      and `k_reg_quant = TurboQuantProd(head_dim - outlier_count, regular_k_bits)`;
      same for V with TurboQuantMSE
- [ ] `OutlierSplitter.quantize_k(x: Tensor[..., d]) → KQuantized` (namedtuple):
      split at `[..., :outlier_count]` and `[..., outlier_count:]`; quantize each partition
- [ ] `OutlierSplitter.dequantize_k(qk: KQuantized) → Tensor[..., d]`:
      dequantize both partitions; concatenate on channel dim
- [ ] `OutlierSplitter.quantize_v` / `dequantize_v`: same with TurboQuantMSE
- [ ] Test: 2.5-bit and 3.5-bit configs produce correct `k_effective_bits` values
- [ ] Test: roundtrip K MSE ≤ sum of per-partition distortion bounds (within 1.1×)
- [ ] Test: `batch_size=4` roundtrip identical to `batch_size=1`

### Phase 4: KV Cache Integration (Days 7–10, ~25% effort)

**Goal**: Implement `TurboKVCache` as a `DynamicCache` subclass for Qwen3.5-27B.

**Files:**
- `turboquant/kv_cache.py`
- `turboquant/model.py`
- `tests/test_kv_cache.py`

**Tasks:**
- [ ] Inspect Qwen3.5-27B source: confirm `DynamicCache` is used, locate `update()` call site
- [ ] Confirm Qwen3.5-27B shapes: `num_key_value_heads=8`, `head_dim=128`, `num_hidden_layers=64`
- [ ] `TurboKVCache.__init__(config: BitConfig, model_config)`:
      instantiate one `OutlierSplitter` per layer (64 total); use shared Π/S across layers
- [ ] Override `update(key_states, value_states, layer_idx, cache_kwargs)`:
      - input shapes: `[batch, num_kv_heads, 1, head_dim]` (one new token)
      - quantize k and v via OutlierSplitter
      - append quantized tensors to per-layer storage lists
      - return (dequantized K full sequence, dequantized V full sequence)
- [ ] Override `key_cache` property: dequantize all stored K tokens per layer on demand
- [ ] Override `value_cache` property: same for V
- [ ] Verify: returned shapes always `[batch, num_kv_heads, seq_len, head_dim]` (not query heads)
- [ ] `patch_qwen_model(model, config: BitConfig) → None`:
      wrap `model.generate()` to inject `TurboKVCache` as `past_key_values`
- [ ] Test: 100-token generation produces identical output shapes to baseline `DynamicCache`
- [ ] Test: no NaN/Inf in attention scores at step 1 and step 100
- [ ] Test: K retrieval shape `[batch, 8, seq, 128]` at all seq_lens

### Phase 5: Smoke Test & Qwen3.5-27B Verification (Day 10–11, ~5% effort)

**Goal**: End-to-end inference at 2.5-bit and 3.5-bit without errors.

**Files:**
- `scripts/run_inference.py`

**Tasks:**
- [ ] Load Qwen3.5-27B in bfloat16 with `device_map="auto"` on A100
- [ ] Run 256-token generation at 2.5-bit with `TurboKVCache`; confirm coherent text output
- [ ] Run 256-token generation at 3.5-bit; confirm coherent text output
- [ ] Log KV cache memory at token 256: baseline (MB) vs TurboQuant 2.5-bit vs 3.5-bit
- [ ] Verify memory reduction ≥ 3× vs fp16 baseline at token 256

### Phase 6: Evaluation (Days 11–15, ~15% effort)

**Goal**: Validate quality and performance against the paper's claims.

**Files:**
- `scripts/eval_perplexity.py`
- `scripts/eval_niah.py`
- `scripts/eval_longbench.py`
- `scripts/benchmark_throughput.py`

**Tasks:**
- [ ] **Perplexity** (wikitext-2): sliding window (stride=512, max_length=2048), compare
      3.5-bit vs baseline; target: Δppl ≤ 0.1 nats
- [ ] **NIAH**: needle-in-haystack at context lengths 4k, 8k, 16k, 32k;
      target recall ≥ 0.99 at 3.5-bit
- [ ] **LongBench-E**: SingleQA and Summarization tasks at 2.5-bit and 3.5-bit;
      target: within 1 point of full-precision average
- [ ] **Throughput benchmark**: `torch.cuda.Event` timing, 1000 calls,
      measure separately: (a) prefill KV quantization, (b) decode per-token KV quantization;
      target: decode ≤ 1ms/token
- [ ] **Memory profiling**: peak VRAM at 1k and 4k tokens for each config vs baseline

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `turboquant/__init__.py` | Create | Package exports: TurboQuantMSE, TurboQuantProd, OutlierSplitter, TurboKVCache, patch_qwen_model, PRESET_2_5BIT, PRESET_3_5BIT |
| `turboquant/config.py` | Create | BitConfig dataclass, PRESET_2_5BIT, PRESET_3_5BIT |
| `turboquant/codebook.py` | Create | Lloyd-Max solver, codebook loader/cache (module-level dict) |
| `turboquant/core.py` | Create | TurboQuantMSE, TurboQuantProd (float32 rotation, batched) |
| `turboquant/outlier.py` | Create | OutlierSplitter, KQuantized, VQuantized namedtuples |
| `turboquant/kv_cache.py` | Create | TurboKVCache (DynamicCache subclass) |
| `turboquant/model.py` | Create | patch_qwen_model() |
| `tests/__init__.py` | Create | Empty |
| `tests/test_codebook.py` | Create | Lloyd-Max correctness, distortion cost validation |
| `tests/test_core.py` | Create | MSE/Prod distortion bound Monte Carlo tests |
| `tests/test_outlier.py` | Create | Splitter roundtrip, bit accounting, batch correctness |
| `tests/test_kv_cache.py` | Create | Cache shapes, no NaN, DynamicCache compatibility |
| `scripts/generate_codebooks.py` | Create | Offline codebook generation for all (d, b) |
| `scripts/run_inference.py` | Create | Smoke test: 256-token Qwen3.5-27B generation |
| `scripts/eval_perplexity.py` | Create | wikitext-2 sliding window perplexity |
| `scripts/eval_niah.py` | Create | Needle-in-haystack benchmark 4k–32k |
| `scripts/eval_longbench.py` | Create | LongBench-E SingleQA + Summarization |
| `scripts/benchmark_throughput.py` | Create | Per-token quantization timing + VRAM profiling |
| `codebooks/d32_b{1..5}.pt` | Generate | Precomputed codebooks for outlier partition (32-dim) |
| `codebooks/d96_b{1..5}.pt` | Generate | Precomputed codebooks for regular partition (96-dim) |
| `codebooks/d128_b{1..5}.pt` | Generate | Precomputed codebooks for full head (128-dim) |
| `requirements.txt` | Create | Package dependencies |
| `pyproject.toml` | Create | Package metadata |

---

## Definition of Done

### Algorithm Correctness
- [ ] `generate_codebooks.py` produces C(f_X, b) within 1% of paper values for d=128, b=1..4
- [ ] Rotation orthogonality: `‖Πᵀ·Π - I‖_F < 1e-6` for d=32, 96, 128
- [ ] `TurboQuantMSE` MSE D_mse ≤ (√3π/2)·(1/4^b)·1.1 for b=2,3,4 at d=128 (10k random unit vectors)
- [ ] `TurboQuantProd` bias |E[⟨y, x̃⟩] − ⟨y, x⟩| / |⟨y, x⟩| < 1% for b=3 (1k sample pairs, d=128)
- [ ] `TurboQuantProd` D_prod ≤ (√3π²‖y‖²/d)·(1/4^b)·1.1 for b=2,3,4

### Integration
- [ ] `TurboKVCache.update()` called on every token (not skipped); verified by insertion counter
- [ ] Returned K shape: `[batch, 8, seq, 128]` at seq_len=1, 100, 1000 (never 32 heads)
- [ ] No NaN or Inf in attention logits at any step during 1000-token generation
- [ ] `batch_size=4` generation produces same K/V reconstruction MSE as `batch_size=1`
- [ ] Smoke test: 256-token Qwen3.5-27B generation at 2.5-bit and 3.5-bit completes without error

### Quality (Qwen3.5-27B on A100)
- [ ] wikitext-2 perplexity at 3.5-bit: Δppl ≤ 0.1 nats vs full-precision baseline
- [ ] NIAH recall ≥ 0.99 at 3.5-bit, 32k context
- [ ] LongBench-E average at 3.5-bit: within 1 point of full-precision score

### Performance
- [ ] Decode-phase KV quantization per token < 1ms on A100 (averaged over 1000 calls)
- [ ] Prefill KV quantization per token < 0.1ms on A100
- [ ] Peak VRAM at 4k tokens: ≤ 30% of full-precision baseline (≥ 3.3× compression)
- [ ] Separate latency measurements for prefill and decode phases reported

### Code Quality
- [ ] `pytest tests/` passes with 0 failures
- [ ] All public functions have type annotations
- [ ] Fixed seeds (seed=42) for Π and S; documented in docstrings
- [ ] `pip install -r requirements.txt && pip install -e .` succeeds from clean env
- [ ] `scripts/run_inference.py --help` documents usage

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Qwen3.5-27B uses custom cache (not DynamicCache) | Medium | High | Inspect source in Phase 4 first; fallback: monkey-patch attention `forward()` directly |
| GQA shape mismatch: KV stored as [batch, 32, seq, 128] by HF internally | Low | High | Assert shape == `[batch, num_kv_heads, ...]` in `update()`; fail loud, not silently |
| bfloat16 rotation error: Π multiplication loses precision | Medium | Medium | Always upcast to float32 before rotation; cast back after dequant. Unit test validates Π orthogonality |
| QJL sign matrix S (d×d) accumulates float error at d=32 | Low | Medium | S is float32 throughout; only QJL *output* is int8 |
| O(N²) dequant at 32k+ context causes OOM | Medium | Low | Accepted for Sprint 001. Profile at 8k and 32k; defer incremental dequant to Sprint 002 |
| `torch.cat` fragmentation grows KV storage list per token | Medium | Medium | Accepted for Sprint 001; pre-allocated buffer deferred to Sprint 002 |
| Outlier fixed-N (first 32 post-rotation) performs worse than expected | Low | Medium | Evaluate 2.5-bit vs 3.5-bit quality; if DoD misses, investigate per-layer calibrated split in Sprint 002 |
| Codex/Gemini centroid value examples (±0.27, ±0.82) incorrect for d=128 | Low | Low | Codebook generated from Lloyd-Max numerics, not from memory; validated against paper's MSE bounds |
| transformers version API break (DynamicCache interface) | Low | Medium | Pin `transformers >= 4.40.0, < 5.0` in requirements; add integration test with version assertion |
| A100 not available during development | Low | High | Unit tests and codebook generation runnable on CPU; KV cache integration testable with small mock model (2L, 2H, d=64) |

---

## Security Considerations

- All operations are pure local computation; no network calls, no external data downloads
- Precomputed codebooks are deterministic from (d, b); no third-party artifacts fetched
- Fixed seeds (seed=42) for Π and S documented and reproducible; security is not affected
  by seed predictability since TurboQuant provides compression guarantees, not cryptographic ones
- Model weights loaded from HuggingFace Hub — standard practice; use `revision=` pin in production

---

## Dependencies

```
# requirements.txt
torch>=2.2.0                  # batched matmul, torch.linalg.qr
transformers>=4.40.0,<5.0     # DynamicCache API stability
scipy>=1.10.0                 # optimize, integrate for Lloyd-Max
numpy>=1.24.0
datasets>=2.14.0              # wikitext-2 for perplexity eval
pytest>=7.0.0
pytest-cov>=4.0.0
```

**Hardware requirement**: NVIDIA A100 (40GB+) for full Qwen3.5-27B evaluation.
CPU is sufficient for unit tests and codebook generation.

---

## Open Questions

1. **Qwen3.5-27B architecture verification**: Must confirm `num_key_value_heads=8`,
   `head_dim=128`, `num_hidden_layers=64` by inspecting `AutoConfig.from_pretrained()`.
   If head_dim ≠ 128, codebook dimensions need adjustment.

2. **2.5-bit arithmetic discrepancy**: The paper states `(32×3 + 96×2)/128 = 2.5`
   but this equals 2.25. We implement the paper's exact channel counts and call the
   config "2.5-bit" to match paper terminology; the actual bit savings are 2.25 bits/coord.

3. **V cache quantizer choice**: We use TurboQuantMSE for V. If LongBench-E results
   show V-path degradation, upgrading to TurboQuantProd for V is a clean Sprint 002 option.

4. **Shared Π/S across layers**: Since TurboQuant is data-oblivious, using one Π
   and one S matrix across all 64 layers is valid. If this introduces unexpected
   correlations, per-layer instances are the fallback (64× memory cost for Π/S).

5. **TurboQuantProd bit budget for K**: With b=3 for K, the MSE stage uses b-1=2 bits
   plus 1-bit QJL, totaling 3 bits per coordinate. The outlier/regular split (3-bit Prod
   for outliers + 2-bit Prod for regulars) means 2-bit MSE+1-bit QJL and 1-bit MSE+1-bit
   QJL respectively. At 1-bit MSE (b-1=1) the codebook has just 2 centroids — verify
   this doesn't collapse reconstruction quality.
