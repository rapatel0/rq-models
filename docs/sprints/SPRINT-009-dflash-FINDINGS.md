# Sprint 009-dflash Findings

**Date**: 2026-05-05
**Status**: COMPLETE — Q8_0 ships as opt-in; default unchanged.

## Pivot from original plan

Sprint 009 was originally scoped as "build imatrix calibration tooling for
DFlash drafts (F-027), then quantize Q4 with imatrix." The user pointed at
[`spiritbuun/Qwen3.6-27B-DFlash-GGUF`](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF)
which already publishes Q4_K_M and Q8_0 GGUFs of z-lab's draft, with a
detailed README explaining why **Q8_0, not Q4, is the right choice**:

> Unlike the 3.5 drafter (all full-attention, Q4-robust), the 3.6 drafter
> introduces **causal sliding-window attention layers** (pattern `[S,S,S,S,F]`,
> window = 2048). Those SWA layers are Q4-fragile — Q4_K_M collapses
> acceptance from ~43% → ~28% on the same workload. Q8_0 is the smallest
> quant that preserves F16 quality.

This explained our Sprint 008 Q4-plain Hamlet collapse to 0% acceptance
without us needing to build imatrix infra. **Pivoted Sprint 009 to test
Q8_0 instead** — much simpler, no fork rebase, no new tooling.

## What we did

1. Quantized our locally-converted `Qwen3.6-27B-DFlash-bf16.gguf` (3300 MiB)
   → `Qwen3.6-27B-DFlash-Q8_0.gguf` (1753 MiB) using `llama-quantize Q8_0`
   in our existing fork. Size matches spiritbuun's published Q8 file
   exactly (1.75 GB).
2. Registered `qwen3.6-27b-dflash-q8` model key in `docker/entrypoint.sh`.
3. Benched at N=2 and N=4 vs Sprint 008 BF16 baseline.

## Numbers

| Config | Quicksort× | Pythagorean× | DC× | Hamlet× | SQL× | **Median** | **Worst** |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 N=2 (current default) | 1.41 | 1.23 | 1.03 | **0.91** | 1.21 | **1.21** | 0.91 |
| BF16 N=4 | 2.22 | 1.15 | 0.60 | 0.53 | 1.02 | 1.02 | 0.53 |
| **Q8 N=2** | 1.49 | 1.27 | 1.03 | 0.87 | 1.24 | **1.24** | 0.87 |
| **Q8 N=4** | **2.53** | 1.20 | 0.65 | 0.54 | 1.02 | 1.02 | 0.54 |
| Q4-plain N=2 (Sprint 008) | 1.53 | 1.30 | 1.02 | **0.57** † | 1.28 | 1.28 | 0.57 |

†Q4-plain Hamlet had 2/3 trials at 0% draft acceptance (catastrophic).
Q8_0 has no such collapse — acceptance stable across all 3 trials per
prompt.

**Headlines**:
- **Quicksort 2.53× at Q8 N=4** — exceeds PR #22105's published 1.5–2× range
  by ~25%. Best single-prompt result of the entire DFlash track.
- **Median 1.24× at Q8 N=2** — marginal +0.03 vs BF16. Within noise.
- **Worst case 0.87× at Q8 N=2** — slight regression vs BF16's 0.91×.

## Hypothesis verdicts

| Hypothesis | Verdict |
|---|---|
| Q8_0 preserves acceptance (vs Q4 collapse) | ✅ Confirmed: Hamlet 63% acc Q8 vs 0%/22% Q4-plain |
| Quantizing draft → much faster wallclock | ❌ My math was too pessimistic. GPU draft-gen with BF16 already mostly hides in HBM; Q8 only shaves ~5-10%. Verify-decode (target Q4_K_XL forward) dominates wallclock. |
| Default flips to Q8 | ❌ Median win is marginal, worst case slightly worse. BF16 stays default; Q8 ships opt-in. |
| ≥1.3× median hard gate clears | ❌ 1.24 (Q8 N=2) — closer but still short. Draft acceptance is the gate, not quantization. |

## What spiritbuun's data tells us

Their published numbers (RTX 3090, target Q4_K_M, code prompt):
- Q8_0: 87 tok/s raw, 97 tok/s chat
- F16: 80 tok/s raw, 93 tok/s chat
- Q4_K_M: 73 tok/s raw, 70 tok/s chat (the SWA collapse)

Q8 vs F16: +9% raw, +4% chat. **Same relative magnitude as our +5-7%.**
This isn't a stack-specific issue with our fork — it's the architectural
ceiling of small-draft block-diffusion speculative decoding.

Note: spiritbuun's [`buun-llama-cpp`](https://github.com/spiritbuun/buun-llama-cpp)
fork has SWA support in the DFlash decoder (commit `b9d01582b` / SD-073)
and a different KV cache quant scheme. Our fork (PR #22105 cherry-pick)
has neither. Even with full SWA + their KV quant, their published Q8
delta over F16 is still single-digit percent on the prompts they tested.
The architectural ceiling argument holds.

## Decision: don't flip the default

Q8_0 is registered as `qwen3.6-27b-dflash-q8` for opt-in. The general
docker-compose default stays at BF16 N=2 because:
- Median improvement (1.21 → 1.24) is within noise.
- Worst-case regresses (0.91 → 0.87).
- BF16 has known-stable behavior across all trial counts.

Operators who want the **2.53× quicksort peak** can use:
```bash
DRAFT_MODEL_NAME=qwen3.6-27b-dflash-q8 DRAFT_N_MAX=4 make run-qwen
```

Operators who want **smaller draft VRAM footprint** (1.75 GB vs 3.47 GB):
```bash
DRAFT_MODEL_NAME=qwen3.6-27b-dflash-q8 make run-qwen
```

## Sprint 010 recommendation

The DFlash median ceiling has been demonstrated across:
- Sprint 005: BF16 N=16 host, median 0.67
- Sprint 008: BF16 N=2 VRAM, median 1.21
- Sprint 009: Q8 N=2 VRAM, median 1.24

The remaining gap to ≥1.3× is draft acceptance on entropic prose
prompts (Hamlet, DC trip). **Three viable Sprint 010 directions**:

1. **Distill a smaller domain-tuned DFlash draft** — z-lab's 1.7B is
   undertrained on prose. A 0.5B or domain-tuned variant could push
   prose acceptance from 60-70% to 85%+, which would close the gap.
   ~1-2 weeks. High risk (training run); high upside if it works.

2. **Productionize EAGLE3** — already in fork from Sprint 004 cherry-pick.
   Architecturally cheaper draft (reuses target hidden states), so the
   draft-gen overhead we hit isn't an issue. Requires Qwen3.6-EAGLE3
   draft weights — survey HF for them, or train one. ~1 week if drafts
   exist.

3. **Cherry-pick spiritbuun's SWA + KV quant from buun-llama-cpp** —
   limited upside based on their own published delta vs F16, but might
   compound with #1 or #2 if undertaken alongside. ~3 days.

**Recommend (1)** based on Sprint 008/009 data: pipeline is solved, draft
quality is the bottleneck.

## Definition of Done

- ✅ spiritbuun GGUF repo evaluated; Q8_0 finding adopted instead of building
  imatrix tooling.
- ✅ Q8_0 GGUF produced locally (1753 MiB; matches spiritbuun's 1.75 GB).
- ✅ Q8 registered as `qwen3.6-27b-dflash-q8` model key.
- ✅ Q8 benched at N=2 and N=4. Side-by-side table written.
- ✅ Sprint 010 recommendation made.
- ⏭ docker-compose default unchanged (BF16 N=2 stays).
