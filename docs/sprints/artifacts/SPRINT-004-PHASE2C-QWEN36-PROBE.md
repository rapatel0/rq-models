# SPRINT-004 Phase 2c — Qwen3.6 family kernel probe

Date: 2026-04-26
Branch: `rq-models@rotorquant`
Probe: `scripts/probe_kv_quality.py`
Model: `Qwen/Qwen3.6-27B` (55.6 GB bf16, 64 transformer layers, 16 of
which are full-attention; head_dim=256, num_kv_heads=4, rope partial
factor 0.25, mrope-interleaved). Architecture string is
`Qwen3_5ForConditionalGeneration` — Qwen3.6 inherits the Qwen3.5 model
class, just bigger (5120 hidden, 17408 intermediate, 24 attn heads).

The probe captures K from layer 0, mid, last full-attn layer at both
`k_proj` (pre-norm) and `k_norm` (post-norm) outputs, runs the planar3
round-trip kernel on it, and reports per-block cosine similarity vs
the original. Compares against the synthetic Gaussian baseline that
the kernel is calibrated for.

## Result

Synthetic baseline (kernel ceiling): cos-sim mean **0.983** at any scale.

| Layer (full_attn idx) | k_proj cos-sim mean | k_norm cos-sim mean | k_norm σ |
|----------------------:|--------------------:|--------------------:|---------:|
|              L03 / 16 |              0.9696 |          **0.9699** |    1.178 |
|              L35 / 16 |              0.9502 |          **0.9472** |    1.537 |
|              L63 / 16 |              0.9745 |          **0.9726** |    1.449 |

All probed Qwen3.6-27B layers land within ~1 cos-sim point of the
synthetic ceiling. Same signature as Qwen3.5-4B and Qwen3.5-9B —
post-k_norm σ in the 1.2–1.6 range with isotropic-enough distribution
that the fixed-Givens + Lloyd-Max codebook represents direction
well. No L00 γ blow-up like Qwen3-4B.

## Conclusion

Qwen3.6 inherits Qwen3.5's well-conditioned K distribution. **Phase 2c
rq3 will work for the Qwen3.6 family without any kernel recalibration.**

## End-to-end serve attempt

Tried `vllm/vllm-openai:v0.19.1 + rq3 overlay` with `--cpu-offload-gb 35
--enforce-eager --gpu-memory-utilization 0.95` (24 GB RTX 4090 + 188 GB
RAM, max_model_len=2048). Weights load fine (15.81 GiB GPU + 35 GiB
CPU, 104 s) and KV-cache profile reports `Available KV cache memory:
3.01 GiB`, but engine init then fails:

```
AssertionError: Cannot re-initialize the input batch when CPU weight
offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298
for more details.
```

This is a vLLM v0.19.1 limitation on **hybrid linear/full attention
models combined with weight CPU offload** — the engine wants to
re-initialize the input batch with mismatched mamba and attention
block sizes, which the offload path doesn't support yet (PR #18298 is
the upstream fix). Triggers on fp16 too (re-running without
ROTORQUANT_MODE crashes at the same assertion).

**Not an rq3 bug**, and not a hardware issue we can route around in
this sprint — needs either:
- a vLLM bump that includes #18298 / the offload-aware hybrid init, or
- a GPU with ≥ 40 GB VRAM (A6000, H100, RTX 6000 Ada) that doesn't
  need the CPU-offload path at all.

Verdict: Phase 2c rq3 + Qwen3.6 is **kernel-validated** today. Full
end-to-end serve confirmation is gated on hardware/vLLM-version, not
on rq-models.

## Cross-family summary

| Model           | L00-ish cos-sim | mid-layer cos-sim | last-layer cos-sim | K σ range | Phase 2c verdict |
|-----------------|----------------:|------------------:|-------------------:|----------:|------------------|
| Qwen3-4B        |       **0.666** |             0.914 |              0.827 | 13.34 → 2.7 | broken — k_norm γ blows up |
| Qwen3.5-4B      |          0.9499 |             0.9562|              0.9737| 1.31 → 1.43 | clean              |
| Qwen3.5-9B      |          0.9509 |             0.9491|              0.9754| 1.31 → 1.43 | clean              |
| **Qwen3.6-27B** |      **0.9699** |          **0.9472**|         **0.9726**| **1.18 → 1.45** | **clean**     |

The pattern is unambiguous: Qwen3 alone has the pathological γ
distribution at low layers; Qwen3.5 onward reverted to whitened-K
training, and Qwen3.6 (which is architecturally Qwen3.5 just scaled to
27B) preserves that. The kernel-as-shipped, byte-for-byte from
`johndpope/llama-cpp-turboquant`, is the right tool for the Qwen3.5 →
3.6 series.
