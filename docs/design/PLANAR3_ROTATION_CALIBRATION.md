# Design — planar3 KV quantization: rotation tables and codebook calibration

Date: 2026-04-27
Status: descriptive (documents the as-shipped behavior of the
`planar3` KV kernel ported from `johndpope/llama-cpp-turboquant @
feature/planarquant-kv-cache`); informs future work on per-model
calibration.
Audience: engineers working on rq-models KV quantization, anyone
debugging quality issues with a specific model family.

## TL;DR

The planar3 KV kernel applies a per-block L2 normalization, a fixed
sequence of 64 random 2D Givens rotations, and a per-element 3-bit
Lloyd-Max codebook. **The rotation tables are not data-calibrated** —
they are random samples from the unit circle drawn from a fixed
seed-42 PRNG at compile time. **Only the 3-bit codebook is
calibrated** — its 8 centroids are Lloyd-Max optimal for a Gaussian
with variance 1/128 (the per-element variance of a 128-element vector
normalized to unit L2 norm).

The design assumes the input K is approximately isotropic Gaussian
after per-block L2 normalization. When that assumption holds (LLaMA /
Mistral / Qwen2 / Qwen2.5 / Qwen3.5 / Qwen3.6), a random rotation is
sufficient because Gaussians are rotationally invariant, and the
codebook fit is preserved. When it doesn't hold (Qwen3 L0 post-`k_norm`
with anisotropic per-dim variance from large γ), no fixed rotation
can whiten the distribution and the codebook loses precision on
direction.

## The pipeline

Each 128-element K block goes through:

```
K  →  L2-normalize  →  64 random 2D Givens rotations  →  3-bit Lloyd-Max codebook
       (per-block)         (fixed table, seed=42)            (8 centroids)

storage:  fp16 norm scalar + 32 bytes qs (2 bits per element)
       + 16 bytes signs (1 bit per element) = 50 bytes / 128 elements
       = 3.125 bits per element
```

### Stage 1 — L2 normalization

```
norm    = sqrt( sum_{i=0..127} K[i]^2 )
K_hat   = K / norm
```

This pulls all K vectors onto the unit sphere. A per-block fp16 norm
scalar is stored alongside the quantized values so dequantization can
recover the original magnitude. After normalization, if K was
*approximately Gaussian*, then K_hat has approximately uniform
distribution on the unit sphere, with per-element variance ≈ 1/128
(since `mean(K_hat[i]^2) = 1/128` by construction).

### Stage 2 — 64 random 2D Givens rotations

```
for p in 0..63:
    c = cos_table[p]
    s = sin_table[p]
    K_rot[2p]   = c * K_hat[2p] - s * K_hat[2p+1]
    K_rot[2p+1] = s * K_hat[2p] + c * K_hat[2p+1]
```

Each rotation acts on an adjacent pair `(2p, 2p+1)`, mixing the two
elements by an angle `θ_p` whose cosine and sine are stored at
compile time. There are 64 rotations covering all 128 elements.

The mathematical role: **decorrelate adjacent elements** so each can
be quantized independently without one element's value leaking
into another's codebook bucket. The rotation is *not* there to
improve the codebook fit — Gaussians are rotationally invariant, so
the post-rotation distribution has the same shape as the
pre-rotation distribution.

### Stage 3 — 3-bit Lloyd-Max codebook

```
centroids = {-0.190685, -0.117832, -0.065717, -0.021460,
              0.021460,  0.065717,  0.117832,  0.190685}

for j in 0..127:
    idx[j] = argmin_k |K_rot[j] - centroids[k]|     # 0..7
    qs[j/4]    |= (idx[j] & 0b011) << (2 * (j % 4)) # 2-bit lower
    if idx[j] & 0b100:
        signs[j/8] |= 1 << (j % 8)                  # 1-bit upper
```

Eight quantization levels (3 bits per element) chosen so the integral

$$\int_{-\infty}^{+\infty} \min_k (x - c_k)^2 \, p_{\mathcal{N}(0, 1/128)}(x) \, dx$$

is minimized — the standard Lloyd-Max optimal scalar quantizer for a
Gaussian source. The variance 1/128 is the per-element variance after
the L2 normalization step, *assuming* K was approximately isotropic.

### Stage 4 — variance-corrected norm

```
recon_sum_sq = sum_j centroids[idx[j]]^2
recon_norm   = sqrt(recon_sum_sq)
norm_stored  = norm / recon_norm        # corrects for codebook bias
```

The reconstruction has slightly different L2 norm than the unit-norm
input due to quantization rounding. We store a corrected norm so
`unpack(pack(K))` matches the original `K` in magnitude exactly,
even though direction has noise.

## What's actually calibrated, and what's not

| Stage              | Calibrated against data?           | Tunable?                     |
|--------------------|------------------------------------|------------------------------|
| L2 normalization   | n/a (pure math)                    | no                           |
| Rotation tables    | **no** — random PRNG, seed=42      | **yes, with offline fit**    |
| 3-bit codebook     | yes — Lloyd-Max for N(0, 1/128)    | yes, with offline fit        |
| Variance-corrected norm | n/a (algebraic correction)    | no                           |

**The rotation tables encode no information about any specific model
or distribution.** They are 64 angles uniformly sampled from `[0, 2π)`
at compile time from a fixed seed. The choice of seed=42 has no
significance — any seed gives a valid set of rotations.

The codebook is the only calibrated component. It assumes the input
distribution to the codebook lookup is N(0, 1/128). That assumption
is correct *if and only if* K was isotropic Gaussian to start with —
the rotation step preserves Gaussian-ness but cannot create it.

## When the design works: isotropic K

LLaMA, Mistral, Qwen2, Qwen2.5, and (empirically) Qwen3.5 and Qwen3.6
all produce K vectors that, after per-block L2 normalization, look
approximately like uniform draws from the unit sphere — i.e. each
element is approximately N(0, 1/128) and elements are weakly
correlated.

In that regime:

* The random rotation maps the input to another approximately
  isotropic distribution.
* The 3-bit codebook fits the post-rotation per-element distribution
  well (because it was tuned for exactly that distribution).
* The variance-corrected norm makes the magnitude reconstruction
  exact.

Empirically, cos-sim of the round-tripped K vs the original K lands
at 0.95–0.98 across all probed full-attention layers in this regime
— compared to the synthetic Gaussian ceiling of 0.983. See
`docs/MODEL_COMPATIBILITY.md` for per-family numbers.

## When the design breaks: anisotropic K (Qwen3 L0)

Qwen3 introduced `k_norm`: an RMS-norm with learned per-dim scale γ
applied to K before attention.

```
K_norm[i] = γ[i] * K[i] / sqrt( mean(K^2) )
```

If γ has high variance across dims (some γ_i ≫ others — empirically
the case in Qwen3-4B's Layer 0, where γ values span roughly ~2 → ~25),
then:

* Per-element σ varies by ~10× across the 128 dims after k_norm.
* L2 normalization brings the *whole vector* to unit norm but doesn't
  equalize per-dim variance.
* No fixed orthogonal rotation can whiten the distribution. A
  rotation is a unitary mixing of input dims — it preserves the
  collective L2 energy but redistributes it across output dims. If
  the input has, say, three high-energy dims and 125 low-energy ones,
  the rotation produces 128 output dims that each combine bits of all
  the input dims; the high-energy components leak into many outputs
  but do not get equalized.
* The codebook, tuned for a single per-element variance of 1/128,
  loses precision on the components that carry the bulk of the energy.

Empirical Qwen3-4B L0 result: cos-sim mean **0.666**, vs the synthetic
ceiling of **0.983**. The kernel preserves L2 norm tightly (rel_l2
≤ 0.0004), so magnitude recovery is correct — direction is what's
lost. Qwen3-4B `"2+2=" → "222222222222"` follows directly.

Deeper Qwen3 layers (L17, L35) have less extreme γ values, σ ≈ 2–6,
and recover cos-sim ≈ 0.83–0.91. The L0 case is the worst.

## Cross-substrate inconsistency in upstream

`johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` ships
**two different rotation tables**:

| Source                                          | Rotation table[0] (cos) | Generated by                  |
|-------------------------------------------------|------------------------:|-------------------------------|
| `ggml/src/ggml-planar-quant.c` (CPU path)       |        +0.7386546135    | custom LCG, seed=42           |
| `ggml/src/ggml-cuda/planar-iso-constants.cuh` (CUDA path) | −0.9095053397 | `torch.manual_seed(42); torch.rand(64) * 2π` |

Same nominal seed, different PRNG implementations → different
rotations. Both are mathematically valid (any random orthogonal
rotation works for isotropic-Gaussian input), but they are *different*
random rotations — KV produced on upstream's CPU path will not
round-trip through upstream's CUDA path.

Our rq-vllm CUDA kernel ports the **CUDA-path** table byte-for-byte
(verified bit-identical via `scripts/cross_substrate_parity.py` on
both synthetic Gaussian and Qwen3-4B L0 K — see
`docs/MODEL_COMPATIBILITY.md`). Since our integration only ever reads
and writes through the same CUDA kernel, we don't inherit upstream's
inconsistency: every K we write is decoded with the same table.

## Future work — per-model rotation calibration

The cheapest fix for currently-broken models (Qwen3 family) is to
*replace* the random rotation with one tuned to actually whiten the
target model's K distribution.

### Approach

1. **Capture a calibration set**: feed a representative corpus through
   the target model, hook the K tensors at the same point our kernel
   would (post-`k_norm`, pre-FlashAttention), accumulate samples
   across many tokens × all full-attention layers.
2. **Per-layer covariance**: compute `Σ_l = E[K_l K_l^T]` for each
   full-attention layer l (one 128×128 matrix per layer).
3. **Eigendecompose**: `Σ_l = V_l Λ_l V_l^T`. The columns of `V_l` are
   the principal axes of K's distribution at layer l.
4. **Use V_l as the rotation**: replace the random Givens table with
   the rotation that maps K into its eigenbasis. Post-rotation, K's
   covariance is diagonal — components are decorrelated.
5. **Refit the codebook per layer**: post-rotation, the per-dim
   variance is `Λ_l[i]`, not 1/128. The Lloyd-Max-optimal codebook
   for that distribution is *different* per dim.

### Implementation cost

~half-day of code:

* `scripts/calibrate_planar3.py` — capture K, compute Σ_l per layer,
  Lloyd-Max refit.
* New kernel variant `rotorquant_planar3_perlayer` that takes a
  per-layer rotation matrix and a per-layer codebook as run-time
  arguments instead of compile-time `__constant__`.
* Per-model packed-constants file (`models/<model_id>/planar3.bin`),
  loaded at engine init.

The trade-off: the rotation is now a 128×128 matrix per layer instead
of 64 cos/sin pairs (~32 KB per layer), and the codebook is
per-layer instead of global (~256 bytes per layer). Negligible
storage; one extra matmul per attention layer in the encode/decode
hot path. A fused implementation can keep the per-token cost
comparable to the current random-rotation kernel.

### Alternative: pre-norm hook

Even cheaper: capture and quantize K *before* k_norm, where the
distribution is well-conditioned even on Qwen3-4B (cos-sim 0.86 at L0
→ 0.94 mean per existing probes). The kernel stays as-is; we just
move the integration point upstream. Requires a small model-side
edit in vLLM (redirect the kv-write hook from
FlashAttentionBackend's `do_kv_cache_update` to the model's K
projection output).

This avoids the calibration entirely but only helps if the model
keeps `k_norm` outside our cache write path — i.e. the cache stores
pre-norm K and the model re-applies k_norm on read. Not how vLLM
currently structures the attention forward, so it's a slightly more
invasive integration change.

### Most invasive: asymmetric per-dim quantization

Replace the kernel entirely with a per-dim absmax + 4-bit asymmetric
quant. Handles arbitrary K distributions but at higher
bytes-per-element (4 bpe vs 3.125), and breaks compatibility with
the upstream rq-models GGUF format. Different product.

## Empirical evidence

* `docs/MODEL_COMPATIBILITY.md` — per-family worst-layer cos-sim with
  the random rotation table.
* `docs/sprints/artifacts/SPRINT-004-PHASE2C-QUALITY-PROBE.md` —
  Qwen3-4B vs Qwen3.5-4B layer-by-layer cos-sim with σ values.
* `docs/sprints/artifacts/SPRINT-004-PHASE2C-QWEN36-PROBE.md` —
  Qwen3.6-27B and Qwen3.6-35B-A3B kernel probe.
* `scripts/cross_substrate_ref/README.md` — bit-parity proof that
  the rq-vllm port matches the upstream CUDA-path math exactly,
  including on the broken Qwen3 distribution.
