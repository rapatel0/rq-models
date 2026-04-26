# Sprint 004 Phase 2 — Kernel port notes

**Source repo**: `johndpope/llama-cpp-turboquant` branch
`feature/planarquant-kv-cache`. Latest commit on the branch as of
2026-04-26: `fc3d1b6` (newer than the `20efe75` originally documented
in SPRINT-004; either should be a valid port basis — confirm against
rq-models's own ppl baselines before pinning).

This doc is the explicit port catalog. Phase 2 deliverable is to
replicate the planar3 KV-write/read math inside `rapatel0/rq-vllm`,
replacing the Phase 1 passthrough stubs in
`vllm/v1/attention/ops/rotorquant_kv.py`.

## The packed block layout (must match exactly)

From `ggml/src/ggml-common.h:339-347`:

```c
#define QK_PLANAR3 128                       // 128 elements per block
typedef struct {
    ggml_half  norm;                         //  2 bytes (fp16 norm scalar)
    uint8_t    qs[QK_PLANAR3 / 4];           // 32 bytes (2-bit indices, packed 4 per byte)
    uint8_t    signs[QK_PLANAR3 / 8];        // 16 bytes (1-bit signs, packed 8 per byte)
} block_planar3_0;
// Total: 2 + 32 + 16 = 50 bytes per 128 elements
//      = 3.125 bits per element (matches 3 bpe target)
```

In vLLM the equivalent is a uint8 tensor of shape `(num_blocks,
block_size, n_kv_heads, packed_bytes_per_token)` where
`packed_bytes_per_token = 50 * head_size / 128` (assumes head_size is
a multiple of 128, which is universal for Qwen / Llama).

## The math (rotation + Lloyd-Max codebook)

From `ggml/src/ggml-cuda/cpy-planar-iso.cu:73+`:

```
For each 128-element segment of fp16 input:
1. Read 128 fp16 values, convert to fp32.
2. Apply 64 successive 2D Givens rotations using the precomputed
   cos/sin tables (PI_COS[64], PI_SIN[64] — seeded from torch
   manual_seed(42)). This is the "planar" rotation.
3. Compute norm = max(|rotated|) (fp16, stored in block.norm).
4. Normalize: each rotated value / norm.
5. Quantize each normalized value to one of 8 centroids via the
   d_mid_3bit[7] thresholds (Lloyd-Max boundary midpoints):
     -0.154259, -0.091775, -0.043589, 0.0,
      0.043589,  0.091775,  0.154259
   Index ∈ {0..7}.
6. Pack: qs stores the 2 LSBs of each 3-bit index (4 indices per
   byte); signs stores the MSB of each 3-bit index (8 indices per
   byte). Reconstruction: index = (qs_2bit | (signs_1bit << 2)).
```

The constants are baked into `__constant__` device memory at compile
time. See `planar-iso-constants.cuh` for the exact float values
(64 cos, 64 sin, 32 quaternions for iso variants).

The dequant path inverts steps 6→1:
```
For each block:
1. Reconstruct 3-bit indices from qs + signs.
2. Lookup d_centroids_3bit[index] → normalized value in [-1, 1].
3. Multiply by block.norm (fp16) to scale back.
4. Apply inverse Givens rotation (same cos/sin tables, transposed
   rotation order).
5. Output 128 fp16 values.
```

## Files to read in the reference repo

```
ggml/src/ggml-cuda/
├── planar-iso-constants.cuh      [5 KB]  ← rotation + codebook constants
├── cpy-planar-iso.cuh             [435 B] ← public interface (4 funcs)
├── cpy-planar-iso.cu              [320 lines] ← pack kernels (planar3/4, iso3/4)
├── set-rows-planar-iso.cuh        [?]     ← per-row write (incremental decode)
├── set-rows.cu                    [?]     ← integration into ggml's set-rows
├── dequantize.cuh                 [?]     ← unpack helpers used by attention
├── fattn-common.cuh               [?]     ← FlashAttention KV read integration
├── fattn.cu                       [?]     ← FlashAttention dispatch
└── template-instances/             [20 files] ← per-(K-dtype × V-dtype) attn templates
    ├── fattn-vec-instance-f16-planar3_0.cu
    ├── fattn-vec-instance-planar3_0-f16.cu
    ├── fattn-vec-instance-q8_0-planar3_0.cu
    ├── fattn-vec-instance-planar3_0-q8_0.cu
    ├── fattn-vec-instance-planar3_0-planar3_0.cu  ← same-dtype K&V (Sprint 004 target)
    └── ... iso variants (Sprint 005)
```

## What to port for Phase 2 (planar3 only, Sprint 004 scope)

1. **Constants** (5 KB, mechanical copy):
   - Source: `planar-iso-constants.cuh`
   - Target: `rq-vllm/vllm/csrc/attention/rotorquant/constants.cuh`
   - Just the `__constant__` arrays for planar3 (cos, sin, centroids,
     midpoints). Skip iso quaternions for Sprint 004.

2. **Pack kernel** (~80 lines including helpers):
   - Source: `cpy-planar-iso.cu` lines 73-310 (`kernel_cpy_f16_planar3`
     + `quantize_3bit` helper)
   - Target: `rq-vllm/vllm/csrc/attention/rotorquant/planar3_kv.cu`
     (add `kernel_pack_f16_to_planar3`, callable from a vLLM op)
   - Refactor: vLLM's KV write takes per-token rather than per-block
     input. Wrap the existing block kernel in a per-token launch that
     writes into the paged-KV's slot_mapping-resolved offset.

3. **Unpack kernel** (~60 lines):
   - Source: `dequantize.cuh` (the dequantize_block_planar3 inline
     function — needs grep to find exact location).
   - Target: same `planar3_kv.cu` file as `kernel_unpack_planar3_to_f16`.
   - Read-side runs inside paged_attention's K, V load path.

4. **Pybind / torch.ops registration**:
   - Two custom ops: `rotorquant_kv_write_planar3`,
     `rotorquant_kv_read_planar3`.
   - Wire into `setup.py`'s extension build alongside vLLM's existing
     CUDA kernels.
   - Replace the passthrough stubs in
     `vllm/v1/attention/ops/rotorquant_kv.py` with `torch.ops`
     dispatches into the new ops.

5. **Wire-in at FlashAttention call site**:
   - Source: `vllm/v1/attention/backends/flash_attn.py` around line 817
     (the `reshape_and_cache_flash` call).
   - Add a one-line branch:
     ```python
     if self.kv_cache_dtype.startswith("rotorquant_"):
         from vllm.v1.attention.ops.rotorquant_kv import rotorquant_kv_write
         rotorquant_kv_write(key, value, key_cache, value_cache,
                             slot_mapping, self.kv_cache_dtype,
                             layer._k_scale, layer._v_scale)
     else:
         reshape_and_cache_flash(...)  # existing
     ```
   - The Phase 1 passthrough stub now becomes the real CUDA op
     dispatcher.

6. **Update `STR_DTYPE_TO_TORCH_DTYPE`**:
   - `vllm/utils/torch_utils.py:46` — change
     `"rotorquant_planar3": torch.float16` →
     `"rotorquant_planar3": torch.uint8` (packed-byte storage).

7. **Update `get_kv_cache_shape`**:
   - `vllm/v1/attention/backends/flash_attn.py:120-130` — when
     `cache_dtype_str.startswith("rotorquant_")`, return packed shape:
     ```python
     packed_bytes = (head_size // 128) * 50  # 50 bytes per 128 elements
     return (2, num_blocks, block_size, num_kv_heads, packed_bytes)
     ```
   - Validate that head_size is a multiple of 128 in
     `supports_head_size`.

## Validation gates for Phase 2

**Hard gate (Δppl)**: WikiText-2 PPL within Δppl ≤ 0.05% of the
llama.cpp planar3 baseline (8.20 on Qwen3.5-27B). Run via
`scripts/eval_perplexity.py` against the docker-served vLLM with
`ROTORQUANT_MODE=planar3`.

**Per-kernel parity test** (recommended): write a unit test that:
1. Generates random fp16 K, V tensors.
2. Packs via the new kernel → unpacks back to fp16.
3. Compares against the same round-trip on llama.cpp's planar3.
4. Asserts `rel_err <= 5e-3` per element.

This catches subtle math bugs (rotation order, sign handling, midpoint
boundaries) before the end-to-end ppl run, which can't easily isolate
which kernel went wrong.

## Reference repo locally cloned

```
~/repos/rq-llama-cpp-ref/   (88 MB, shallow clone, branch
                             feature/planarquant-kv-cache @ fc3d1b6)
```

Don't push this directory anywhere; it's a read-only reference for
the port. Drop it after Phase 2 ships.

## Estimated effort

- Constants port: 30 min
- Pack kernel + helpers: 1-2 days (most of the math)
- Unpack kernel: 1 day
- Pybind / torch.ops registration: 1 day
- FlashAttention integration: 1 day
- Per-kernel parity tests: 1 day
- Δppl validation runs (Qwen3.6-27B + Qwen3.5-27B): half day
- Debug iteration buffer: 2-3 days

**Total: ~1-1.5 weeks** for planar3 only on a single-engineer pace,
assuming the Phase 1 passthrough wiring is correct (Step 4 of
PHASE1-VALIDATION.md is green).
