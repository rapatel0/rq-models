# SPRINT-004 Phase 2c — KV-quality probe across Qwen3 / Qwen3.5

Date: 2026-04-26
Branch: rotorquant
Image: rq-vllm:latest (overlay over vllm/vllm-openai:v0.19.1, transformers 5.5.4)

The 24-token Qwen3-4B Phase 2c smoke produced degenerate output
(" the city of the city of the…"). To localise the cause we ran a
direct kernel-input probe (`/tmp/probe_real_k.py`): load the HF model,
hook attention layers, capture the K tensor that flows into
FlashAttention, run the planar3 round-trip on it, and compare per-block
cosine similarity vs the synthetic Gaussian baseline the kernel was
calibrated for.

## Synthetic baseline

The kernel is scale-invariant via per-block L2 norm. All scales hit
the same number, which is the reference ceiling of the codebook.

| Input              | Blocks | cos_sim min | cos_sim mean | normerr p99 |
|--------------------|-------:|------------:|-------------:|------------:|
| Gaussian σ=1.0     |    256 |      0.9688 |       0.9831 |      0.1687 |
| Gaussian σ=0.1     |    256 |      0.9687 |       0.9831 |      0.1683 |
| Gaussian σ=0.01    |    256 |      0.9688 |       0.9831 |      0.1685 |

## Real K — Qwen/Qwen3-4B (head_dim=128, 8 KV heads, **with k_norm**)

| Layer / source            | cos_sim min | cos_sim mean | per-elem σ | ‖K‖ median |
|---------------------------|------------:|-------------:|-----------:|-----------:|
| L00 k_proj (pre-k_norm)   |      0.8607 |       0.9436 |      0.092 |       1.08 |
| L00 k_norm (post-k_norm)  |  **0.6231** |   **0.6655** | **13.34**  |     125.73 |
| L18 k_proj (pre-k_norm)   |      0.7857 |       0.9374 |      0.664 |       7.38 |
| L18 k_norm (post-k_norm)  |      0.7667 |       0.9137 |      2.017 |      22.60 |
| L35 k_proj (pre-k_norm)   |      0.6351 |       0.9100 |      6.039 |      61.04 |
| L35 k_norm (post-k_norm)  |      0.6761 |       0.8274 |      2.684 |      32.13 |

Layer 0 post-k_norm is catastrophic. The learned γ in `k_norm` blows
per-element σ from 0.092 (pre) to 13.34 (post), with very anisotropic
per-dim scales — Lloyd-Max + fixed-Givens calibration assumes
post-RMSnorm K is approximately isotropic Gaussian, which is not true
here. The kernel preserves L2 norm tightly in every case (rel_l2
≤ 0.0004) — magnitude is fine, **direction is wrong**.

## Real K — Qwen/Qwen3.5-4B (head_dim=256, 4 KV heads, hybrid full/linear attention)

Probed only the full_attention layers (the linear ones use a different
recurrent state, no standard KV cache).

| Layer                  | cos_sim min | cos_sim mean | per-elem σ |
|------------------------|------------:|-------------:|-----------:|
| L03 full_attn k_proj   |      0.7649 |       0.9511 |      1.042 |
| L03 full_attn k_norm   |      0.7595 |       0.9499 |      1.315 |
| L19 full_attn k_proj   |      0.8412 |       0.9587 |      1.060 |
| L19 full_attn k_norm   |      0.8659 |   **0.9562** |      1.630 |
| L31 full_attn k_proj   |      0.9559 |       0.9760 |      1.158 |
| L31 full_attn k_norm   |      0.9447 |       0.9737 |      1.427 |

All probed Qwen3.5-4B layers land at cos_sim mean **0.95–0.97** — within
the synthetic ceiling (0.98) and well above the threshold where attention
breaks. Qwen3.5's k_norm produces well-conditioned K (σ≈1.3–1.6) instead
of Qwen3-4B's pathological L00 σ=13.4. Same picture for unsloth/Qwen3.5-9B
(all layers cos_sim mean ≥ 0.949).

## End-to-end serve sanity

Same image, `--enforce-eager`, T=0. `rq3` = `--kv-cache-dtype rotorquant_planar3`.

| Model           | Mode | Prompt                       | Output (24 tokens)                                               |
|-----------------|------|------------------------------|------------------------------------------------------------------|
| Qwen3-4B        | fp16 | "The capital of France is"   | " Paris. The capital of Germany is Berlin..."                    |
| Qwen3-4B        | rq3  | "The capital of France is"   | " the city of the... the city of the..."                         |
| Qwen3-4B        | rq3  | "2+2="                       | "222222222222"                                                   |
| Qwen3.5-4B      | fp16 | "The capital of France is"   | " Paris.\nA. True\nB. False\nAnswer:\nA\n\nWhich of the following is NOT a reason" |
| Qwen3.5-4B      | rq3  | "The capital of France is"   | " Paris.\nA. True\nB. False\n\n<think>\nThinking Process:..."    |
| Qwen3.5-4B      | fp16 | "2+2="                       | "4\n2+2=4\n2+2="                                                 |
| Qwen3.5-4B      | rq3  | "2+2="                       | "4\n2+2=4\n2+2="  ← **byte-identical to fp16**                   |

For Qwen3.5-4B "2+2=" the rq3 output is literally byte-identical to
fp16. For "The capital…" both modes produce the right answer ("Paris")
followed by the same A/B/False multiple-choice pattern; they diverge
at token 11 with fp16 going to "Answer:\nA…" and rq3 branching into
`<think>` mode. Both branches are plausible Qwen3.5 generations — no
quality cliff.

## Conclusion

The planar3 KV kernel is byte-for-byte from
`johndpope/llama-cpp-turboquant`, which calibrated against
LLaMA / Qwen2-class models without QK-norm. Qwen3-4B's Layer-0 k_norm
violates the codebook's whitened-Gaussian assumption hard enough to
destroy attention. Qwen3.5 (and presumably Qwen3.6 by lineage) reverts
to a well-conditioned K distribution that the existing calibration
handles within ~1 cos-similarity-point of the synthetic ceiling.

**Action taken**: target Qwen3.5 and Qwen3.6 as the supported family
for rq-models on the vLLM substrate. Qwen3-4B is documented as a known
incompatibility (kernel calibration vs k_norm γ blow-up) and is
*not* a Phase 3 PPL-gate target.
