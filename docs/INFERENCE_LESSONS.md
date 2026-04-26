# Inference Engineering Lessons (carried over from vortex)

**Audience**: future engineer working on rq-models — RotorQuant /
SpectralQuant kernel work, llama.cpp / vLLM substrate decisions, perf
investigations on consumer GPUs.

**Source of evidence**: 24 sprints of profile-driven optimization on
vortex (Rust LLM inference engine, archived 2026-04-25). Hardware was
RTX 4090 + Qwen 27B Q4_K_M; conclusions about structure are durable,
specific numbers need recalibration on RTX 5090 + the Qwen 3.6 model
class. Full evidence base in
[vortex/docs/RETROSPECTIVE.md](https://github.com/rapatel0/vortex/blob/main/docs/RETROSPECTIVE.md)
and
[vortex/docs/sprints/artifacts/profile-s024/](https://github.com/rapatel0/vortex/tree/main/docs/sprints/artifacts/profile-s024/).

---

## TL;DR for rq-models

1. **At N=1-4 on consumer GPUs, the FFN GEMV family is bandwidth-bound,
   not compute-bound.** Don't aim optimizations at compute (tensor cores,
   MMA, IMMA) when memory is the wall.
2. **The decode token has four orthogonal bottleneck classes**: VRAM
   bandwidth, occupancy starvation, launch overhead, and fp32 intermediate
   bandwidth. Plans that target only one leave 75% of headroom on the
   table.
3. **RotorQuant KV compression is the only optimization that helps at
   *every* regime** by shrinking the largest growing tensor. iso/planar
   variants buy 4.9× compression at 3 bpe with 97% decode speed retention
   — your existing measurements confirm this.
4. **Speculative decoding (dflash/Eagle-3) is ecosystem-immature for
   hybrid models** (Qwen 3.5/3.6). Wait for upstream rollback +
   AMD-platform fixes before investing.
5. **Single-developer kernel work loses to vLLM / llama.cpp / FlashAttention
   community work.** RotorQuant is the differentiator; let the engines
   handle the engine. This is why we're moving to vLLM substrate (Sprint
   004+).

---

## What this means for the rq-models roadmap

### Sprint 003 (SpectralQuant Python prototype) — keep going

The SpectralQuant approach (asymmetric K/V quantization based on PCA-
identified signal subspaces) is exactly the kind of *quantization
research* that leverages where bytes-per-weight matters most: K cache
read traffic at long contexts is bandwidth-amplified across every
attention call. Reducing bytes-per-K-vector beats any kernel-level
trick.

The key gating question (does spectral4 beat planar3's PPL=8.20?) is
the right validation gate. Don't over-engineer until that closes.

### Sprint 004 (after SpectralQuant Python validation)

Two parallel decisions need answers in this sprint:

1. **Engine substrate**: stay on llama.cpp fork
   (`johndpope/llama-cpp-turboquant`) or move to vLLM fork.
2. **CUDA kernel implementation language**: C++ inside the chosen
   engine's kernel directory.

**Recommendation**: fork upstream `vllm-project/vllm` directly (latest
stable tag), under the rq-models org. Don't fork
`mitkox/vllm-turboquant` — it's a 4-commit snapshot, single maintainer,
and inheriting it locks rq-models out of upstream vLLM updates. Use
mitkox's TurboQuant patches as *reference / inspiration* for where the
quantization plug-in points are in vLLM, then port the rq-models
RotorQuant kernels (iso3, planar3, planar4, iso4) into the fork
following vLLM's existing GPTQ/AWQ pattern.

### Sprint 005+ (after vLLM substrate works)

- SpectralQuant CUDA kernel in vLLM (assuming Sprint 003 validated it).
- Long-context throughput sweep on vLLM with rq-models KV variants.
- Speculative decoding evaluation (likely *not* dflash yet — see below).

---

## Profile-first culture

Every performance sprint must start with `ncu` + `nsys` evidence. The
reason: vortex Sprint 023's "MMA-first" plan was misframed for 4 sprints
because no one ran ncu before planning. A 1-day profile pass would have
killed the wrong direction immediately.

The decision tree for "is this kernel worth optimizing":

```
Is the kernel >85% DRAM utilization?
├── YES → bandwidth-bound. Optimize via:
│         (a) Reduce bytes read (quantization, KV compression)
│         (b) Reduce reads per kernel (fusion → less round-tripping)
│         (c) Increase batch size to amortize weight reads
│         DO NOT: add tensor cores, change compute dtype, tune occupancy
│
├── Is the kernel >85% SM utilization?
│   YES → compute-bound. Optimize via:
│         (a) Tensor cores (MMA/IMMA) for matmul work — only here
│         (b) Higher-throughput math (e.g., dp4a → IMMA gives 2× on Ada)
│         DO NOT: shrink memory bandwidth (it's not the bottleneck)
│
└── Both <50%? → occupancy-bound or launch-overhead-bound. Diagnose:
    - Few blocks (<n_SMs) at peak? → grid is too small. Restructure
    - Lots of small kernels in sequence? → CUDA Graphs / fusion / merge
    - Idle gaps in nsys timeline between kernels? → CPU launch overhead
```

For RotorQuant / SpectralQuant kernel work specifically: the K-read and
V-read kernels in attention are bandwidth-amplified across all KV
positions, so they are the highest-leverage targets. The matmul-side
(QKV projection, FFN gate/up/down) is less affected by KV compression
since it reads weights, not KV.

---

## What works at each regime (verdicts)

### Won't help at gate regime (N=1-4 on consumer GPUs, Q4_K_M)

| Technique | Why not |
|---|---|
| MMA / tensor cores for matmul | Adds compute where compute isn't the bottleneck. dp4a is already saturating DRAM at 93% utilization. |
| Quant scheme micro-optimization (Q4_K vs Q4_0 vs Q4_K_M) | Bandwidth scales with bytes-per-weight; variant differences are <5%. |
| Per-kernel occupancy tuning for bandwidth-bound kernels | If at 90% DRAM, raising occupancy from 50% to 90% just makes more SMs wait. |
| Larger thread blocks for bandwidth-bound kernels | Reduces grid parallelism without removing the bottleneck. |
| MMQ (compute-dense matmul) at low N | Compute-dense (89.8% SM) but reduces grid parallelism → 1.7-2× wall-clock slower. |
| CPU-side overhead reduction without CUDA Graphs | Individual launch overhead is irreducible at 5-10 µs without graph capture. |

### Will help at gate regime

| Technique | Magnitude (estimated) | Risk |
|---|---|---|
| **KV cache compression** (RotorQuant iso4/iso3, SpectralQuant) | +5-15% throughput at long ctx; primary value is fitting larger ctx | ppl impact must be validated per model — iso beats planar on MoE, planar beats iso on dense |
| bf16 intermediate buffers (with f32 reductions) | +3-5% standalone | Δppl validation required |
| CUDA Graphs replay | +5-10% at gate regime | Graph capture invalidates on shape change → admission churn defeats it; need ≥95% hit rate |
| Fusion (silu_mul→ffn_down, fused_add_rmsnorm) | +1-3% per fusion | Custom kernels = higher maintenance |
| Larger batch sizes (when application allows) | +30-100% per-card aggregate | Latency increase per request |
| Speculative decoding *with verified high acceptance* | +50-200% if accept ≥0.7; -10-30% if accept <0.4 | Acceptance is workload-specific; admission churn poisons it |

### Conditional / overrated

| Technique | When useful | When not |
|---|---|---|
| Weight streaming (PCIe → GPU) | Model >> VRAM | Q4_K_M makes 27B models fit in 24-32 GB cards. Premise gone for consumer hardware. |
| n_parallel slot scheduling | Multi-tenant serving | Single-user batch — set n_parallel=1, simpler. |
| DeltaNet-specific optimizations | Models that use DeltaNet (some Gemma variants) | Pure-attention models (Llama, Qwen 27B). |
| Custom paged attention | Real differentiator over vLLM's | Otherwise, vLLM's PagedAttention is years ahead — use it. |

### Speculative decoding specifically (dflash, Eagle-3, lookahead)

State of the art as of 2026-04-25:
- **dflash** (`z-lab/dflash`, 2,284 stars) is real research traction.
- **Stability concerns**: can crash and take down hosting servers.
- **AMD platforms broken** as of vLLM #40632.
- **Hybrid-attention models** (Qwen 3.5/3.6 — your target!) have weak
  acceptance because recurrent linear-attention state requires custom
  cache rollback that isn't yet upstream.

**Verdict for rq-models**: don't invest in dflash now. Revisit in 6
months when AMD support lands and Qwen-hybrid rollback is upstream.
The published 2.5×-EAGLE-3 numbers are paper projections; on the
recurrent-hybrid path they'll be lower in practice.

---

## Empirical numbers to remember (RTX 4090, Qwen 27B Q4_K_M)

These came from vortex Sprint 024's profile pass. **Recalibrate on RTX
5090 before citing.** The structure of the conclusions is durable; the
specific constants are not.

- DRAM peak: 1008 GB/s (GDDR6X 4090). RTX 5090 is ~1792 GB/s (GDDR7).
- DRAM effective at well-optimized kernel: ~93% of peak.
- PCIe 4.0 x16 sustained: 25-27 GB/s. RTX 5090 has PCIe 5.0 x16 ≈ 50-55
  GB/s sustained.
- ρ_N (compute scaling vs N=1) at FFN-down on Qwen 27B: 1.00 / 1.72 /
  2.95 / 6.81 / 14.3 for N = 1 / 2 / 4 / 8 / 16.
- Crossover N (streaming vs resident throughput): ~13.76 for Qwen 27B
  on PCIe 4.0 x16. Higher on PCIe 5.0.
- Gate regime (N=1-4) FFN-down: 89-93% DRAM, 20-44% SM utilization.
  Bandwidth-bound, not compute-bound.
- N=8+ Q6_K: 90% SM utilization. Compute-bound at this slice only.

### Per-token time breakdown (Qwen 27B iso4 N=8, ~42 ms wall-clock)

| Component | Share | Optimizable? |
|---|---|---|
| FFN GEMV (gate + up + down × 32 layers) | ~80% | bandwidth at gate, partial compute headroom at N≥8 |
| Attention (paged GQA + KV write/read + QKV) | ~10% | memory at long ctx |
| Elementwise (rmsnorm + silu_mul + add) | ~5% | launch-overhead-bound; CUDA Graphs |
| LM head | ~3% | bandwidth-bound |
| Embed + sample | <1% | not worth optimizing |
| Launch overhead (gaps in nsys) | ~7-12% | CUDA Graphs target |

---

## Process learnings (apply directly)

1. **Profile before you plan.** Every perf sprint starts with `ncu` /
   `nsys`. The output is the *input* to the planning doc, not the
   validation.

2. **Gates must be inside the achievable region.** If you can't prove
   a gate is achievable from a back-of-envelope of bandwidth and
   compute, it's not a gate, it's a hope. The vortex "3.0× at N=4 vs
   llama.cpp" gate was structurally impossible on RTX 4090 and shaped
   four sprints of misdirected work.

3. **Validation harnesses precede the work that needs them.** The Δppl
   ≤ 0.5% gate on bf16 in vortex was blocked because the perplexity
   harness was being built *during* the bf16 sprint. rq-models's
   `eval_perplexity.py` is correctly already in place — don't lose
   that property when adding new substrates.

4. **Avoid cudarc small-CudaSlice churn** if you ever build Rust-side
   inference work. Vortex hit reproducible segfaults after ~14k
   cumulative tokens of fresh `clone_htod` cycles
   ([FU-5 details](https://github.com/rapatel0/vortex/blob/main/docs/sprints/SPRINT-024-FOLLOWUPS.md#fu-5)).
   Pre-allocate scratch slices once and reuse with `copy_into`.

5. **Single-developer kernel work loses to community engines.** Novel
   research contributions land *as plug-ins to vLLM / llama.cpp*, not
   as bespoke engines. The reach is 100× higher: your novel idea
   ships via vLLM's installed base, not by convincing users to switch
   engines.

---

## Companion docs in this repo

- [`THROUGHPUT_CONFIGURATION_MODEL.md`](THROUGHPUT_CONFIGURATION_MODEL.md)
  — formal regime equation, ρ_N tables, crossover N, configuration
  picker logic. Recalibrate constants for RTX 5090.
- [`scripts/bench_n_parallel.py`](../scripts/bench_n_parallel.py) —
  cross-substrate aggregate throughput benchmark. Targets any
  OpenAI-API-compatible server. Use this to compare llama.cpp vs vLLM
  under matched workloads in Sprint 004+.
- [`scripts/eval_perplexity.py`](../scripts/eval_perplexity.py) —
  existing rq-models perplexity harness (HF transformers-based). Keep
  using this; it's more sophisticated than vortex's HTTP-API version.

---

*Written 2026-04-25 as part of the vortex → rq-models pivot. Ground
truth for current rq-models hardware (RTX 5090) hasn't been measured
yet; the calibration constants in this doc and
THROUGHPUT_CONFIGURATION_MODEL.md need a refresh once you stand up an
ncu pass on the new substrate.*
