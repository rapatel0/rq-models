# Throughput Configuration Model

A model for predicting LLM inference throughput given hardware + model + workload parameters, and for choosing the best (resident-vs-streaming × N-parallel × context-size) configuration for a target metric.

**Intended use**: reference logic for an automatic configuration profiler. Given `(model, card, workload_goal)`, the profiler picks a config.

**Origin**: derived from RTX 4090 + Qwen 27B iso4 measurements during the
vortex Sprint 024 profile investigation (see
[vortex `RETROSPECTIVE.md`](https://github.com/rapatel0/vortex/blob/main/docs/RETROSPECTIVE.md)
and [`profile-s024/SPRINT-024-INVESTIGATION-SYNTHESIS.md`](https://github.com/rapatel0/vortex/blob/main/docs/sprints/artifacts/profile-s024/SPRINT-024-INVESTIGATION-SYNTHESIS.md)).
The structure (regime equation, ρ_N scaling, crossover N) is hardware-
agnostic; only the calibration constants change.

**Calibration constants for rq-models hardware (RTX 5090, 32 GB)**: TODO —
rerun the ncu profile pass on RTX 5090 to refresh `β_vram_effective`,
`β_pcie` (likely PCIe 5.0 x16 ≈ 50-55 GB/s sustained), and the ρ_N table
on the target Qwen3.6 / Qwen3.5 model shapes. Until then, treat the
4090 numbers as illustrative, not authoritative.

---

## TL;DR

The throughput ceiling for a decode forward pass is:

```
t_forward_pass = max(t_compute(N), t_weight_delivery)
```

where:
- `t_compute(N)` = total kernel compute time for one forward pass at batch size N
- `t_weight_delivery` = `model_weight_bytes / min(vram_bandwidth_effective, weight_source_bandwidth)`

For **resident weights**, the weight source is GDDR6X/HBM (~1 TB/s on a 4090). For **streamed weights**, the weight source is PCIe (~27 GB/s on PCIe 4.0 x16).

Streaming hurts throughput **iff** `t_compute(N) < t_pcie_delivery`. The crossover N is set by the **model-weight-byte-to-compute-cycle ratio** — it's a property of the model × hardware combination.

Above crossover, streaming frees VRAM for larger N / larger ctx / larger models without a throughput hit. Below crossover, streaming is a throughput regression.

---

## The model

### Inputs

| Symbol | Meaning | Units | Typical source |
|---|---|---|---|
| `W` | Total weight bytes in the model | bytes | GGUF metadata |
| `L` | Number of transformer layers | int | model config |
| `W_L` | Average weight bytes per layer | bytes | `W / L` |
| `C_1` | Compute time per forward pass at N=1 | seconds | measured (microbench or live decode) |
| `ρ_N` | Compute scaling factor: `C_N / C_1` | dimensionless | measured across N sweep |
| `β_vram` | Effective VRAM→SM bandwidth | bytes/s | measured (ncu DRAM Throughput × peak) |
| `β_pcie` | Effective PCIe host→device bandwidth | bytes/s | measured (cuMemcpyHtoD bulk) |
| `M_total` | Total VRAM budget | bytes | GPU spec |
| `M_kv_per_token` | KV cache bytes per context token per slot | bytes | KV type (iso4 / iso3 / f16) |
| `ctx` | Context window length | tokens | workload parameter |
| `N` | Concurrent decode slots | int | workload parameter |
| `goal` | `aggregate_throughput` \| `per_session_latency` \| `max_context` | enum | user choice |

### Core equations

**Compute time per forward pass**:
```
t_compute(N) = C_1 × ρ_N
```

Measured `ρ_N` is sub-linear in N due to weight reuse (L1 sharing across warps within a block, L2 sharing across blocks within a wave). For Qwen 27B iso4 on 4090 at FFN-down shape:

| N | ρ_N (measured) | notes |
|---|---|---|
| 1 | 1.00 | baseline |
| 2 | 1.72 | |
| 4 | 2.95 | |
| 8 | 6.81 | |
| 16 | 14.3 | extrapolated |
| 32 | ~28 | extrapolated — L1 saturates here |
| 64 | ~54 | compute-bound regime |

Note `ρ_N / N < 1` means weight reuse delivers sub-linear scaling; the kernel gets more efficient per-token as N grows.

**Weight delivery time**:
```
t_delivery_resident = W / β_vram_effective
t_delivery_stream   = W / β_pcie
```

`β_vram_effective` is typically **90-95% of peak** VRAM bandwidth when kernels are well-optimized and DRAM-bound. For cold kernels or overhead-dominated kernels it can be 40-70%. We observed **93% peak on 4090 GDDR6X at Q4_K FFN-down** under ncu profile, or ~940 GB/s effective out of 1008 GB/s peak.

`β_pcie` is the **sustained** bulk DMA throughput. PCIe 4.0 x16 theoretical is 32 GB/s; practical sustained with pinned host memory and aligned transfers is **25-27 GB/s**. PCIe 5.0 x16 sustained is ~50-55 GB/s.

**Forward pass latency**:
```
t_forward_pass = max(t_compute(N), t_delivery)
```

This is the key equation. Since every weight must be delivered and every weight-read kernel must run, neither side can "skip" work. Double-buffering can hide the shorter of the two behind the longer, but cannot shrink the longer one.

**Throughput** (tokens per second, aggregate):
```
tok_s_aggregate(N) = N / t_forward_pass(N)
```

**Single-session throughput**:
```
tok_s_per_session = 1 / t_forward_pass(N)   # in a batch of N sessions each get the same latency
```

### Crossover N

Streaming matches resident latency when:
```
t_compute(N*) = t_delivery_stream
C_1 × ρ_N* = W / β_pcie
ρ_N* = W / (C_1 × β_pcie)
```

For Qwen 27B iso4 on 4090 (C_1 = 42 ms, W = 15.6 GB, β_pcie = 27 GB/s):
```
ρ_N* = 15.6 GB / (0.042 s × 27 GB/s) = 13.76
```

Interpolating the measured `ρ_N` table, `ρ = 13.76` happens around **N ≈ 14-16**.

**Below N=14**: streaming is slower than resident (PCIe is the bottleneck; compute waits on weights).

**Above N=14**: streaming matches or beats resident *per-pass-latency* equivalently, AND frees VRAM for higher N / larger ctx.

### VRAM budget for a config

```
M_used = M_weights_resident + M_kv + M_workspace + M_streaming_buffers
M_kv   = N × ctx × M_kv_per_token
M_weights_resident = W                if resident
                   = 2 × W_L          if streaming with double-buffer
M_workspace ~= 1 GB (fixed)
M_streaming_buffers ~= 2 × W_L if streaming
```

Must satisfy `M_used ≤ M_total`.

Measured constants for Qwen 27B iso4 on 24 GB 4090:
- `W = 15.6 GB`
- `W_L = 244 MB` (average across 64 layers)
- `M_kv_per_token = 16.5 KB` (iso4) | 66 KB (f16)
- `M_workspace ≈ 1 GB`

### Configurations that fit (Qwen 27B iso4, 24 GB 4090)

| config | weights | KV | workspace | total | fits |
|---|---|---|---|---|---|
| resident, N=8, ctx=4096 | 15.6 | 4.0 | 1 | 20.6 | ✅ (current) |
| resident, N=16, ctx=4096 | 15.6 | 8.0 | 1 | 24.6 | ❌ |
| resident, N=8, ctx=8192 | 15.6 | 8.0 | 1 | 24.6 | ❌ |
| streaming, N=32, ctx=2048 | 0.5 | 1.1 | 1 | 2.6 | ✅ |
| streaming, N=32, ctx=4096 | 0.5 | 2.2 | 1 | 3.7 | ✅ |
| streaming, N=64, ctx=4096 | 0.5 | 4.3 | 1 | 5.8 | ✅ |
| streaming, N=128, ctx=4096 | 0.5 | 8.7 | 1 | 10.2 | ✅ |
| streaming, N=64, ctx=16384 | 0.5 | 17.3 | 1 | 18.8 | ✅ |

---

## Regime map

For a given (model, card) pair, three regimes:

### Regime A — "Memory-bound, resident wins"
- `N < N*` (below streaming crossover)
- `t_compute(N) < t_delivery_stream`
- **Use resident weights**. Streaming adds PCIe stall.
- Current Sprint 023 state at Qwen 27B N=1-8.

### Regime B — "Crossover"
- `N ≈ N*`
- `t_compute(N) ≈ t_delivery_stream`
- Either approach works; resident has slight edge for latency, streaming frees VRAM.

### Regime C — "Compute-bound, streaming wins on aggregate"
- `N > N*`
- `t_compute(N) > t_delivery_stream`
- **Use streaming** if the freed VRAM enables higher aggregate N than resident allows.
- At Qwen 27B iso4 on 4090: resident caps N=8, streaming unlocks N=32-128.

---

## Configuration-selection algorithm (pseudocode)

```python
def select_config(model, card, goal):
    """Return (weight_policy, N, ctx) that maximizes the goal."""
    W = model.weight_bytes
    C_1 = model.measured_c_1  # or estimate from model_size * 0.04 s/GB
    rho = model.measured_rho_table  # or default sub-linear curve
    beta_vram = card.vram_bandwidth * 0.90  # effective
    beta_pcie = card.pcie_bandwidth  # sustained
    M = card.vram_bytes

    # Crossover N where streaming ties resident on latency
    rho_star = W / (C_1 * beta_pcie)
    N_star   = invert(rho, rho_star)

    candidates = []
    for ctx in sorted_ctx_options(model):
        # Resident: find max N that fits
        for N in range(1, MAX_N):
            if W + N*ctx*model.kv_bytes_per_token + WORKSPACE <= M:
                candidates.append(('resident', N, ctx))
            else:
                break
        # Streaming: find max N that fits
        for N in range(1, MAX_N):
            if 2*model.weight_bytes_per_layer + N*ctx*model.kv_bytes_per_token + WORKSPACE <= M:
                candidates.append(('streaming', N, ctx))

    if goal == 'per_session_latency':
        # Pick lowest t_forward_pass with N=1 preferred
        return min(candidates, key=lambda c: t_forward_pass(c))
    elif goal == 'aggregate_throughput':
        # Pick highest N / t_forward_pass
        return max(candidates, key=lambda c: c.N / t_forward_pass(c))
    elif goal == 'max_context':
        # Pick largest ctx, tiebreak by throughput
        return max(candidates, key=lambda c: (c.ctx, c.N / t_forward_pass(c)))


def t_forward_pass(c):
    policy, N, ctx = c.policy, c.N, c.ctx
    t_comp = C_1 * interpolate(rho_table, N)
    if policy == 'resident':
        t_deliv = W / beta_vram_effective
    else:
        t_deliv = W / beta_pcie
    return max(t_comp, t_deliv)
```

---

## Measured constants (Qwen 27B iso4 on RTX 4090, as of 2026-04-22)

```
model:
  W = 15.6 GB
  L = 64
  W_L = 244 MB
  kv_bytes_per_token (iso4) = 16.5 KB
  kv_bytes_per_token (f16)  = 66 KB
  C_1 = 42 ms                         # total forward-pass compute at N=1
  rho_table = {1: 1.0, 2: 1.72, 4: 2.95, 8: 6.81, 16: ~14.3, 32: ~28}

hardware (RTX 4090):
  vram_peak = 1008 GB/s                # GDDR6X
  vram_effective = 940 GB/s            # ≈93% on well-optimized dp4a kernel
  pcie_peak = 32 GB/s                  # PCIe 4.0 x16
  pcie_sustained = 27 GB/s             # pinned host memory, bulk transfer
  vram_bytes = 24 GB
  sm_count = 128
  ada_fp32_tflops = 82.6
  ada_int8_dp4a_peak = ~600 GOPs/cycle/SM   # rough
  ada_int8_mma_peak = ~660 GOPs/cycle/SM

predicted:
  rho_star = 13.76   # crossover where streaming matches resident
  N_star = 14-16     # interpolated N for the crossover
  streaming_floor_N1 = 578 ms/forward_pass = 1.73 agg tok/s
  resident_n8_measured = 39.4 agg tok/s
  streaming_n32_predicted = ~55 agg tok/s
  streaming_n64_predicted = ~67 agg tok/s
```

---

## Important caveats and what's not yet measured

### `ρ_N` is kernel-architecture-dependent, not just N-dependent

The measured `ρ_N` values are for the **current weight-shared Q4_K kernel**. Other kernel variants produce different scaling:
- Pretile variant (pre-Sprint-023): slightly better `ρ_N` for Q4/Q5 at N=8-16 (13-28% faster wall-clock).
- MMQ variant (Sprint 022): ~2× worse `ρ_N`.
- MMA (tensor-core): unknown (scaffold was broken). Expected better at N=16+.

An auto-config tool must **benchmark the specific kernel** under use, not inherit this table blindly.

### `β_vram_effective` varies by quant and shape

Q4_K at Qwen 27B FFN-down hit 93% DRAM. Q5_K hit 93.5%. Q6_K hit 88%. Smaller shapes and smaller models likely vary.

### `β_pcie` is sensitive to host setup

- Pinned host memory: 25-27 GB/s sustained.
- Pageable host memory: 10-15 GB/s (depends on OS page cache state).
- IOMMU / virtualization: can cut bandwidth 30-50%.
- PCIe 5.0: ~55 GB/s sustained. Shifts crossover to N ≈ 7-8.

### N not arbitrarily scalable

`ρ_N` is measured only up to N=16. Beyond that:
- L1 per-SM capacity caps how many warps can fit in a block sharing weights.
- Occupancy caps (max warps per SM).
- Scheduler / admission queue limits.

Need dedicated experiments at N=32/64/128 to validate the extrapolations.

### Hardware outside Ada sm_89

The model generalizes, but specific constants change:
- Hopper (H100): HBM3 ~3.35 TB/s, PCIe 5.0 ~55 GB/s. Crossover shifts.
- Ada sm_86 (RTX 3090, 4080): different VRAM bandwidth, dp4a peak.
- AMD MI300: different architecture entirely (IMMA vs MMA, different SM layout).

Auto-config should probe the target hardware at startup.

### Non-FFN time is fixed cost

The `C_1 = 42 ms` includes attention, DeltaNet, rmsnorm, rope, embed, copy, etc. These don't scale with N the same way FFN does. Attention at iso4 paged is 1.6% of GPU time per nsys. These scale more like O(N × ctx) for attention, O(N) for rmsnorm. At high N the non-FFN fraction may grow.

---

## How this feeds auto-config

A runtime configuration profiler should:

1. **Probe hardware** at startup:
   - `β_vram_effective`: micro-kernel that reads a large buffer, measure throughput.
   - `β_pcie`: pinned-host→device bulk transfer, ≥1 GB.
   - `sm_count`, `vram_bytes`, compute capability.

2. **Profile the model's kernels** at startup (or cached from previous runs):
   - `C_1`: run one decode step, measure wall-clock.
   - `ρ_N`: sweep N over {2, 4, 8, 16}, measure.
   - `kv_bytes_per_token`: from KV type and model config.

3. **Compute `N*`, regime boundaries, VRAM budget**:
   - Apply equations above.

4. **Present options to the user** or pick automatically based on goal:
   - `aggregate_throughput` → streaming + largest fitting N ≥ N*.
   - `per_session_latency` → resident + smallest N that meets compute floor.
   - `max_context` → streaming + ctx maximizing under VRAM budget.
   - `most_concurrent_sessions` → streaming + largest N fitting.

5. **Emit rationale** so operators can audit:
   ```
   Selected config: streaming, N=32, ctx=4096, iso4
   Rationale:
     - Goal: aggregate_throughput
     - Crossover at N*=14-16 (measured); N=32 > N*
     - Fits VRAM: 2.6 / 24 GB used
     - Predicted aggregate: ~55 tok/s (vs resident N=8 measured 39.4)
   ```

6. **Validate prediction** via short bench after config selection; re-tune if observed ≠ predicted.

---

## Known tension: predicted vs measured

The streaming predictions above are **derived from a bandwidth-vs-compute model**, not measured. They assume:
- PCIe can run at 27 GB/s sustained without stalls.
- Double-buffering hides PCIe latency perfectly when compute ≥ PCIe.
- Kernel weight reuse (`ρ_N`) continues the measured sub-linear pattern past N=16.

Actual streaming performance will likely be somewhat worse due to:
- Non-overlapping periods (startup, sync, kernel launches with no concurrent prefetch).
- PCIe contention with KV writes or other host-device traffic.
- Host memory allocator fragmentation under multi-hour operation.

**Rule of thumb**: discount predicted streaming throughput by 15-25% to get a conservative estimate. Validate with a short live bench.

---

## Open questions for future work

1. **How does `ρ_N` behave at N=32, 64, 128?** Need direct measurement, extrapolation risky.
2. **What's the actual PCIe bandwidth under concurrent KV writes?** PCIe traffic is shared between weight streaming and host-side tokenizer/output.
3. **How does the model generalize to MoE?** Expert-selection changes which weights are "hot" per token — reduces effective weight-demand per token.
4. **Smaller models (7B, 13B)**: `W` shrinks, making streaming more viable at lower N. A 7B model streamable fully at N=3-4. Auto-config should surface this.
5. **Multi-GPU tensor parallel**: if we split weights across 2 GPUs, `W_per_gpu` halves, reducing streaming cost. Different regime.
6. **Smart resident partial cache**: keep embed + lm_head + attention QKV/O resident (smaller, hot weights); stream only FFN (~80% of weights). Reduces `W_streamed` to ~12.5 GB, shifts crossover to lower N.

---

## Provenance

- Profile measurements: Sprint 024 investigation, branch `profile-s024-investigation`, commit `f61507c`.
- ncu reports: `docs/sprints/artifacts/profile-s024/sweep/` (binary, gitignored) + `docs/sprints/artifacts/profile-s024/sweep-csv/` (CSVs, committed).
- Analysis: `docs/sprints/artifacts/profile-s024/SPRINT-024-PROFILE-ANALYSIS.md`.
- This document: user-driven bandwidth discussion, 2026-04-22.
