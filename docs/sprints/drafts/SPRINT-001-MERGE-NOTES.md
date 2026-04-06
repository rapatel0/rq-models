# Sprint 001 Merge Notes

## Draft Strengths Accepted

### Claude Draft
- Correct Beta distribution parameters: Beta((d-1)/2, (d-1)/2), not Beta(d/2, d/2)
- Rotation matrix Π operates on the *partition dimension* (32 or 96), not full head_dim=128
- Codebooks needed at d=32, d=96, d=128 — missed by Codex and Gemini
- GQA: quantize at num_key_value_heads=8 granularity, broadcast to query heads
- Share Π and S across all 64 layers (data-oblivious, saves 64× memory) — valid per algorithm design
- Pin transformers version; DynamicCache API instability is a real risk
- 6-phase structure with concrete file paths and task checklists

### Codex Draft
- 15-day phased execution timeline is realistic for a solo implementer
- Numerical stability DoD: verify ‖Πᵀ·Π - I‖_F < 1e-6
- Reproducibility criterion: fixed seeds for Π and S
- Smoke test scheduled early (Day 10) before full eval
- Broader DoD including code quality (type hints, PEP 8)

### Gemini Draft
- Only draft to correctly specify 3.5-bit split arithmetic: 64@4 + 64@3 = 3.5 ✓
- Broader cache superclass tip (transformers.Cache)
- torch.compile question: valid latency consideration
- Separate prefill vs decode latency measurement in DoD

## Valid Critiques Accepted

| Critique | Finding | Action |
|----------|---------|--------|
| Claude critique of Codex | `__setitem__`/`__getitem__` wrong — must override `update()` | Use `update()` in final sprint |
| Claude critique of both | S_{i,j} must be N(0,1) not N(0,1/d) — scaling error breaks unbiasedness | Specify N(0,1) explicitly |
| Claude critique of both | K cache inner product → TurboQuantProd; V cache → TurboQuantMSE (user confirmed) | Separate K/V quantizer types |
| Claude critique of Codex | Codebook centroids for b=2 wrong — Beta(63.5,63.5) is near-Gaussian, centroids ≈ ±0.018, ±0.054 not ±0.27, ±0.82 | Validate codebook via Lloyd-Max, not from memory |
| Gemini critique of Claude | bfloat16 rotation instability — must upcast to float32 | Add float32 upcast requirement |
| Gemini critique of both | O(N²) dequant at long context | Noted in Risks; user accepted for Sprint 001, defer optimization to Sprint 002 |
| Codex critique of Claude | Claude draft inconsistency: showed TurboQuantMSE in OutlierSplitter but stored qjl_bits+gamma (Prod fields) | K uses TurboQuantProd instances, V uses TurboQuantMSE instances |
| Gemini critique of both | GQA: dequant must return [batch, 8, seq, 128] not [batch, 32, seq, 128] | Explicit shape requirement in DoD |

## Critiques Rejected

| Critique | Claim | Why Rejected |
|----------|-------|-------------|
| Codex critique of Gemini | Gemini's 3.5-bit split (64@4+64@3) is wrong | Actually correct: (64×4+64×3)/128=3.5. Claude's 32@4+96@3 = (128+288)/128=3.25, which is wrong |
| Gemini critique: DoD 5% threshold | Tighten to 5% distortion from paper bounds | Intent specifies 10% (Monte Carlo variance); 5% causes spurious failures |
| Codex: magnitude-sorted outliers | Better quality via activation stats | User explicitly chose data-oblivious fixed-N to match paper's algorithm design |

## Interview Decisions Applied

1. **K: TurboQuantProd(b), V: TurboQuantMSE(b)** — separate storage layouts for K and V
2. **Full evaluation in Sprint 001**: NIAH, wikitext-2 perplexity, LongBench-E, throughput benchmark
3. **Outlier channels**: Fixed first-N (indices [0..31] for 2.5-bit / [0..63] for 3.5-bit), fully data-oblivious
4. **Dequant strategy**: Full dequant on each forward pass; O(N²) accepted for Sprint 001; optimized cache deferred to Sprint 002

## Bit Split Arithmetic (Verified)

| Config | Outlier | Regular | Formula | Effective bits |
|--------|---------|---------|---------|---------------|
| 2.5-bit | 32 ch @ 3-bit | 96 ch @ 2-bit | (32×3+96×2)/128 | 2.25 (paper claims 2.5 — minor text error) |
| 3.5-bit | 64 ch @ 4-bit | 64 ch @ 3-bit | (64×4+64×3)/128 | 3.5 ✓ |

**Note**: Paper's 2.5-bit arithmetic (32×3+96×2)/128 = 2.25, not 2.5. We implement the paper's exact channel counts (32 outlier, 96 regular at b=3,2 respectively) and call the config "2.5-bit" to match the paper's labeling.
