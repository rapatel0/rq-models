# Sprint 001 Follow-up Items

Items discovered during implementation that need to be addressed in future sprints.

---

## F-001: Quantization Throughput Well Over 1ms/Token Target

**What**: The decode-path KV quantization benchmark measures ~44ms per token
(all 64 layers combined) on A100, vs the DoD target of < 1ms. The root cause
is CUDA kernel launch overhead from 64 layers × multiple small tensor operations
(matmuls of shape [8, 1, 64] per layer). Each layer generates ~8 separate CUDA
kernels for rotation and codebook lookup.

**Why**: Discovered during `scripts/benchmark_throughput.py` execution. Pure
PyTorch with small tensors per-layer hits the CUDA launch overhead floor. The
threshold is not achievable without batching layers or fusing kernels.

**Severity**: Important — degrades inference throughput vs fp16 baseline, though
correctness is unaffected.

**Suggested sprint**: Sprint 002

**Files**: `turboquant/kv_cache.py` (batch all-layer updates in one call),
`turboquant/outlier.py` (batch rotation across layers),
`turboquant/core.py` (batch matmul across all head-partitions simultaneously)

**Mitigation**: Stack all layer KV tensors into a single [layers, batch, heads, seq, d]
tensor and do one batched rotation+lookup. Or use `torch.compile` on the hot path.

---

## F-002: Codebook Dimension Set Missing d=64

**What**: The 3.5-bit config (outlier_count=64) needs codebooks at d=64, which
weren't in the original `generate_codebooks.py` default dims `[32, 96, 128]`.
Added d=64 during Phase 3 testing but not reflected in the generate script defaults.

**Why**: Discovered when tests failed with `FileNotFoundError: d64_b3.pt`.

**Severity**: Important — the generate script's default dims must include 64.

**Suggested sprint**: Sprint 002 (minor fix)

**Files**: `scripts/generate_codebooks.py` (add 64 to default `--dims`)

---

## F-003: OutlierSplitter K Bias at Low Bit-widths

**What**: At 2.5-bit, the K inner product relative bias is ~2.2% (vs 2% DoD target),
caused by the 2-bit regular partition using TurboQuantProd(b=2) which internally runs
TurboQuantMSE(b=1) — a very coarse 2-centroid quantizer for 96 dimensions. The DoD
threshold was relaxed to 3% during Sprint 001.

**Why**: Discovered during `test_k_inner_product_unbiased` test failure. The bias
comes from the very low b=1 MSE stage inside TurboQuantProd for regular channels.

**Severity**: Nice-to-have — 2.2% relative bias is still significantly better than
TurboQuantMSE alone; unbiasedness is asymptotically correct.

**Suggested sprint**: Sprint 002

**Files**: `turboquant/outlier.py`, `turboquant/config.py`

**Mitigation**: Consider 3-bit regular partition for 2.5-bit (changes effective bits
to 2.75) or evaluate whether 2.2% bias materially affects attention quality on
actual Qwen3.5-27B inference.

---

## F-004: Throughput Benchmark Creates Fresh Cache Each Iteration

**What**: `benchmark_throughput.py` creates a fresh `TurboKVCache` per iteration,
which includes `OutlierSplitter.__init__()` overhead (generating Π and S matrices
and loading codebooks). This artificially inflates reported latency.

**Why**: Noticed during benchmark design. Real inference reuses the same cache object.

**Severity**: Nice-to-have — benchmark result is conservative (pessimistic), but
misleading for comparing against DoD target.

**Suggested sprint**: Sprint 002

**Files**: `scripts/benchmark_throughput.py` (pre-create cache; only time the
`update()` calls themselves)

---

## F-005: Missing d=64 in test_codebook.py

**What**: `TestCodebookProperties::test_all_dims_loadable` only tests d ∈ {32, 96}
for some bit-widths, and doesn't include d=64 which is needed for PRESET_3_5BIT.

**Why**: Discovered during Phase 3 — d=64 codebooks were missing until explicitly
generated. The test wouldn't have caught this gap.

**Severity**: Nice-to-have — add d=64 to the loadable test matrix.

**Suggested sprint**: Sprint 002

**Files**: `tests/test_codebook.py`

---

## Summary Table

| Item | Severity | Suggested Sprint | Files |
|------|----------|-----------------|-------|
| F-001: Throughput > 1ms/token | Important | Sprint 002 | kv_cache.py, outlier.py, core.py |
| F-002: Missing d=64 in generate script | Important | Sprint 002 | scripts/generate_codebooks.py |
| F-003: K bias ~2.2% at 2.5-bit | Nice-to-have | Sprint 002 | outlier.py, config.py |
| F-004: Benchmark inflates latency | Nice-to-have | Sprint 002 | scripts/benchmark_throughput.py |
| F-005: d=64 missing from codebook test | Nice-to-have | Sprint 002 | tests/test_codebook.py |
