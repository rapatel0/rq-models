# rq-models — model compatibility list

Running list of model families and their compatibility with the
shipped RotorQuant KV-cache quantization kernels (`planar3`, and by
extension `iso3` / `planar4` / `iso4` once they land).

The kernels are calibrated for K vectors that look like a whitened
Gaussian — zero-mean, isotropic, σ ≈ 1 after per-block L2 normalization.
Models whose K diverges from that assumption (anisotropic per-dim
scale, large γ blow-up after RMS-norm, distribution shift due to
training data) lose direction information at quantization. The kernel
preserves magnitude (L2 norm) tightly in every case (rel_l2 ≤ 0.0004
on real K) — when a model is "bad," the failure mode is **direction
drift, not magnitude error**.

How we measure compatibility:

  cos-sim of round-tripped K vs original K, averaged per 128-element
  block, captured directly from each model layer via
  `scripts/probe_kv_quality.py`. The synthetic-Gaussian ceiling is
  cos-sim mean ≈ 0.983; well-conditioned models land within 1
  cos-sim point of that ceiling.

| Cos-sim mean | Verdict                                      |
|--------------|----------------------------------------------|
| 0.95 – 0.98  | clean — Phase 2c rq3 produces fp16-equivalent output |
| 0.85 – 0.95  | borderline — slight quality drop; OK for non-recall tasks |
| < 0.85       | **broken** — attention scores degrade, expect gibberish |

## Compatibility table

Updated as we test new models. "Predicted" rows are based on the
architecture's QK-norm presence and γ training pattern, **not**
verified empirically. "Verified" rows have a real-K probe artifact
under `docs/sprints/artifacts/`.

| Model family              | Has QK-norm? | Worst-layer cos-sim | Status        | Verdict | Notes |
|---------------------------|:------------:|--------------------:|---------------|---------|-------|
| LLaMA 1 / 2 / 3           |      no      |                  —  | predicted     | clean   | Original calibration target. No QK-norm γ. |
| Mistral 7B / Mixtral      |      no      |                  —  | predicted     | clean   | No QK-norm. |
| Qwen2 / Qwen2.5           |      no      |                  —  | predicted     | clean   | Pre-Qwen3 — no QK-norm in attention. |
| **Qwen3-4B**              |     yes      |          **0.666**  | **verified**  | **broken** | Layer-0 `k_norm` γ blows σ from 0.09 (pre) to 13.4 (post) with strong anisotropy. Smoke output: `"2+2=" → "222222222222"`. Don't ship rq3 on this. |
| Qwen3-8B / 14B / 32B      |     yes      |               (TBD) | predicted     | likely broken | Same model class as Qwen3-4B; expect the same γ pathology unless the ablation differs by size. Probe before relying on. |
| **Qwen3.5-4B**            |     yes      |             0.9499  | **verified**  | clean   | k_norm γ well-conditioned (σ ≈ 1.3–1.6 across layers). End-to-end serve fp16-equivalent. |
| **Qwen3.5-9B (unsloth)**  |     yes      |             0.9491  | **verified**  | clean   | Same calibration as 4B. |
| **Qwen3.6-27B (dense)**   |     yes      |             0.9472  | **verified**  | clean   | Inherits Qwen3.5 model class; serve via bnb 4-bit + rq3 validated end-to-end. |
| **Qwen3.6-35B-A3B (MoE)** |     yes      |             0.9359  | **verified**  | clean   | MoE FFN doesn't perturb K. End-to-end serve gated by an upstream vLLM bnb+MoE bug, not rq3. |
| Gemma / Gemma 2 / Gemma 3 |   yes (newer)|               (TBD) | predicted     | unknown | Newer Gemma variants ship QK-norm with model-specific γ; needs probing. |
| DeepSeek V2 / V3          |   yes (V3)   |               (TBD) | predicted     | unknown | V3 introduced QK-norm; probe before relying on. |

To probe a new model, run:

```bash
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/scripts/probe_kv_quality.py:/probe.py \
    --entrypoint python3 rq-vllm:latest /probe.py <hf_model_id>
```

and append a row above with the model id, worst-layer cos-sim, and the
serve-time outcome.

## Why models fail: the calibration assumption

The planar3 kernel applies a fixed sequence of 64 Givens rotations to
each 128-element K block, then maps each rotated element through a
3-bit Lloyd-Max codebook tuned for unit-variance Gaussian inputs.
Specifically:

- **The rotations are data-independent** (compile-time constants from
  the original llama.cpp planar3 design). They were chosen to
  approximately diagonalize the covariance of "typical" K vectors —
  i.e., LLaMA-class K with no QK-norm. They do not adapt to the K
  distribution they receive at runtime.
- **The 3-bit codebook is Lloyd-Max optimal for Gaussian σ=1**. Each
  block's L2 norm is stored separately (in the per-block `norm`
  field), so absolute scale is handled — but the per-element relative
  scale across the 128 dims is assumed to be roughly uniform after
  rotation. If the K has strongly anisotropic per-dim variance (some
  γ_i ≫ others), no fixed rotation can whiten it, and the codebook
  loses direction.

That's exactly the failure mode in Qwen3-4B's L0: post-`k_norm` K has
per-dim σ varying by 10×+ across the 128 dims, so the rotation can't
get it isotropic and the codebook represents direction poorly.

Every other model we've probed (LLaMA-class behavior, no QK-norm) and
every Qwen3.5 / 3.6 variant has K within the codebook's design range,
and the kernel works as intended.

## Possible future work to extend coverage to "bad" models

These are out-of-scope for the current sprint but worth noting:

1. **Per-model recalibration** — refit the planar3 rotation matrix
   and codebook from a sample of real K captured from the target
   model. Needs an offline calibration pipeline + per-model packed
   constants in the kernel.
2. **Pre-norm quantization** — capture and quantize K *before* the
   `k_norm` RMS-norm step, where the distribution is well-conditioned
   on Qwen3-4B (cos-sim 0.86 at L0 → 0.94 mean). Requires hooking the
   model code in vLLM rather than the FlashAttention KV-write
   boundary.
3. **Asymmetric per-dim scale** — replace the fixed Givens + Lloyd-Max
   scheme with a per-dim absmax + 4-bit asymmetric quant. Different
   kernel entirely; would handle anisotropic K but at a higher
   bytes-per-element.

Of the three, (2) is the cheapest experiment — same kernel, different
hook point. (1) is the rigorous answer. (3) is a different product.
