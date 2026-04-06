# Sprint 001 Deferred Items

Items discussed, proposed in drafts, or raised in critiques but explicitly scoped
out of Sprint 001.

---

## D-001: Incremental / Pre-Allocated KV Cache

**What**: Replace the dynamic `torch.cat`-per-token storage with a pre-allocated
`uint8` tensor of shape `[max_seq_len, batch, kv_heads, d]`. Use a sequence pointer
instead of growing lists. Eliminates CUDA memory fragmentation and makes throughput
predictable at long contexts.

**Why deferred**: User chose "full dequant each forward pass" in interview; adds
implementation complexity without changing correctness. O(N²) dequant is acceptable
at ≤8k context for Sprint 001.

**Target sprint**: Sprint 002

**Prerequisites**: Sprint 001 TurboKVCache must be functional and correctness-validated first.
Profile at 8k and 32k context to quantify the regression before implementing.

**Files**: `turboquant/kv_cache.py` (replace list-based storage with pre-allocated buffer)

---

## D-002: Triton Kernel for Rotation + Codebook Lookup

**What**: Fused Triton kernel that performs `Π @ x` (random rotation) and nearest-centroid
lookup in a single GPU kernel pass. Eliminates intermediate float32 tensor allocation for
the rotated vector; targets ≤0.1ms/token quantization overhead.

**Why deferred**: Pure PyTorch is sufficient for Sprint 001 (<1ms target is achievable
without custom kernels at d=128). Triton adds implementation complexity and is optional.

**Target sprint**: Sprint 002 (after profiling confirms PyTorch is the bottleneck)

**Prerequisites**: D-001 (pre-allocated buffer reduces fragmentation overhead first)

**Files**: `turboquant/kernels/rotate_quant.py` (new), `turboquant/core.py` (use kernel optionally)

---

## D-003: TurboQuantProd for V Cache

**What**: Upgrade V cache from TurboQuantMSE to TurboQuantProd. Value aggregation
`softmax(scores) · V` is also a dot product, so TurboQuantProd's unbiased inner product
guarantee applies to V retrieval as well.

**Why deferred**: TurboQuantMSE for V is simpler and the bias diminishes at b≥3. The
paper's primary claim is K-path unbiasedness (attention scores). If Sprint 001 evaluation
shows V-path degradation on LongBench-E or perplexity, upgrade V in Sprint 002.

**Target sprint**: Sprint 002 (conditioned on Sprint 001 quality results)

**Prerequisites**: Sprint 001 evaluation results. Only worth doing if V-path is a bottleneck.

**Files**: `turboquant/outlier.py` (OutlierSplitter.quantize_v, dequantize_v),
`turboquant/kv_cache.py` (V storage format: add qjl and gamma fields)

---

## D-004: Magnitude-Sorted Outlier Channel Detection

**What**: Instead of always using fixed first-N post-rotation channels as outliers,
run a short calibration pass (first 128 input tokens) to identify the channels with
highest L2 norm after rotation. These are then treated as outlier channels for the
remainder of generation.

**Why deferred**: Breaks TurboQuant's data-oblivious guarantee (a core paper claim).
User explicitly chose fixed-N to match the paper's algorithm. If fixed-N fails to
achieve the quality targets (NIAH recall, perplexity), revisit as a pragma option.

**Target sprint**: Sprint 003 (if Sprint 001/002 quality results show fixed-N is insufficient)

**Prerequisites**: Sprint 001 evaluation results; quality gap analysis

**Files**: `turboquant/outlier.py` (add `calibrate()` method), `turboquant/config.py`
(add `outlier_mode: Literal["fixed", "calibrated"]` field to BitConfig)

---

## D-005: LongBench-E Full Suite

**What**: Full LongBench-E evaluation across all 21 tasks (SingleQA, MultiQA,
Summarization, Few-shot, Synthetic, Code). Sprint 001 covers only SingleQA and
Summarization (the two most relevant for KV cache compression).

**Why deferred**: Scope — running all 21 tasks requires significant compute time and
dataset setup. The subset covers the paper's reported tasks.

**Target sprint**: Sprint 002

**Prerequisites**: Sprint 001 LongBench-E subset must pass DoD

**Files**: `scripts/eval_longbench.py` (extend task list)

---

## D-006: Nearest-Neighbor Search Application

**What**: Apply TurboQuant as a product quantization alternative for approximate nearest
neighbor (ANN) search on high-dimensional vector databases (paper §4.4). Benchmarked
against Product Quantization and RabitQ on DBpedia Entities dataset.

**Why deferred**: Out of scope for this sprint (KV cache only). Different use case,
different evaluation infrastructure.

**Target sprint**: Future (separate sprint if ANN search becomes a project goal)

**Prerequisites**: None (independent of KV cache work)

**Files**: `turboquant/ann.py` (new module), `scripts/eval_ann.py`

---

## D-007: Weight Quantization

**What**: Apply TurboQuant to compress model weights (linear layer weight matrices)
in addition to KV cache. This would enable serving Qwen3.5-27B in lower VRAM for
weights as well as KV.

**Why deferred**: Explicitly out of scope. Weight quantization requires different
application logic (one-time quantization of static matrices, not streaming vectors).

**Target sprint**: Future (separate sprint if weight quantization becomes a goal)

**Prerequisites**: None

**Files**: `turboquant/weight_quant.py` (new module)

---

## D-008: torch.compile Optimization

**What**: Wrap TurboQuantMSE and TurboQuantProd under `@torch.compile` to allow
the compiler to fuse the matmul + argmin pipeline. Could reduce quantization overhead
below the 1ms target without custom Triton kernels.

**Why deferred**: Interaction with `torch.compile` and dynamic shapes (variable seq_len)
needs careful testing. Deferred to avoid blocking Sprint 001 evaluation.

**Target sprint**: Sprint 002

**Prerequisites**: Sprint 001 throughput benchmark showing latency regression; D-001 (stable buffer)

**Files**: `turboquant/core.py` (add `torch.compile` decorator behind a config flag)

---

## D-009: Per-Layer Distinct Π / S Matrices

**What**: Use a unique Π and S matrix per layer (64 per model) instead of sharing one
across all layers. Eliminates any theoretical concern about shared randomness introducing
layer-to-layer correlations.

**Why deferred**: Data-oblivious design means sharing is valid (paper's algorithm doesn't
require per-layer uniqueness). Memory cost of 64 unique Π matrices at d=128, float32:
64 × 128 × 128 × 4 bytes = 4.2 MB — small but unnecessary per current analysis.

**Target sprint**: Future (only if Sprint 001 shows unexpected correlation artifacts)

**Prerequisites**: Sprint 001 quality evaluation must show a problem first

**Files**: `turboquant/core.py`, `turboquant/outlier.py`

---

## Summary Table

| Item | Description | Target Sprint | Blocker |
|------|-------------|---------------|---------|
| D-001 | Pre-allocated KV cache buffer | Sprint 002 | Sprint 001 correctness + profiling |
| D-002 | Triton rotation+quant kernel | Sprint 002 | D-001, profiling confirms bottleneck |
| D-003 | TurboQuantProd for V cache | Sprint 002 | Sprint 001 quality results |
| D-004 | Calibrated outlier detection | Sprint 003 | Sprint 001/002 quality gap analysis |
| D-005 | LongBench-E full suite (21 tasks) | Sprint 002 | Sprint 001 subset passes |
| D-006 | ANN search application | Future | Separate project goal |
| D-007 | Weight quantization | Future | Separate project goal |
| D-008 | torch.compile integration | Sprint 002 | Sprint 001 latency profiling |
| D-009 | Per-layer distinct Π/S matrices | Future | Quality regression observed |
