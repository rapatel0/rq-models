# Sprint 004 Phase 2b + 2c — integration steps (run on GPU box)

**Status before this doc**: Phase 2a complete. The planar3 pack/unpack
CUDA kernels are committed at `rapatel0/rq-vllm@feature/rotorquant`
(commit `0393317e1`) under
`csrc/attention/rotorquant/{planar3_kv.cu, torch_bindings.cpp}` along
with a parity test scaffold at
`tests/kernels/test_rotorquant_planar3_kv.py`.

The kernels are NOT yet built or wired into vLLM's import path.

## Step 0: validate the kernel math (before integration)

Run the standalone harness on the GPU box. This decouples kernel
correctness from the vLLM integration so you don't debug both at once.

```bash
cd ~/repos/rq-models && git pull
git -C ~/repos/rq-vllm pull origin feature/rotorquant
python3 scripts/sprint004_phase2_kernel_test.py
```

Expected output ends with:
```
ALL PASS — planar3 KV pack/unpack kernels are correctness-validated.
```

If this fails, fix the kernel math BEFORE attempting integration. Failure
modes to watch for:
- L2 norm drift > 1e-2: the `corrected = grp_norm / recon_norm` scaling
  in pack is wrong. Check the recon_sq accumulation in the loop.
- Per-element p95 > 0.10: 3-bit codebook spacing or rotation order
  inverted. Compare `quantize_3bit` thresholds against
  `d_centroids_3bit` and verify `unpack_3bit` decode matches.
- NaN on zero input: `inv_norm` guard is missing or wrong epsilon.

If Step 0 passes, Steps 1-3 below are the wire-in.

## Step 1: build wiring (Phase 2b)

vLLM uses scikit-build-core + CMake. The planar3 sources need to be
added to the kernel-extension build target.

### Option A: extend vllm._C with our sources (preferred, simplest)

Edit `~/repos/rq-vllm/CMakeLists.txt`. Find the existing
`add_library(_C ...)` or equivalent (search for the cu source list).
Add:

```cmake
list(APPEND VLLM_EXT_SRC
    "csrc/attention/rotorquant/planar3_kv.cu"
    "csrc/attention/rotorquant/torch_bindings.cpp"
)
```

before `define_gpu_extension_target(_C ...)` is called. The torch_bindings.cpp
PYBIND11_MODULE block at the bottom will register the ops under the
`_C` extension's namespace, so they become available as
`torch.ops._C.rotorquant_planar3_{pack,unpack}`.

Note: if vLLM's existing TORCH_LIBRARY_FRAGMENT pattern is used elsewhere,
swap the PYBIND11_MODULE in torch_bindings.cpp for an equivalent
TORCH_LIBRARY_FRAGMENT registration to keep style consistent. Both work.

### Option B: separate JIT extension (if Option A's CMake gets messy)

Keep the kernels as a separate JIT-loaded extension. In
`vllm/v1/attention/ops/rotorquant_kv.py`, replace the passthrough
stubs with:

```python
import os
from pathlib import Path
import torch
from torch.utils import cpp_extension

_THIS_DIR = Path(__file__).resolve()
_VLLM_ROOT = _THIS_DIR.parents[5]  # adjust if dir depth differs
_RQ_CSRC = _VLLM_ROOT / "csrc" / "attention" / "rotorquant"

_ext = cpp_extension.load(
    name="rq_models_rotorquant",
    sources=[str(_RQ_CSRC / "planar3_kv.cu"),
             str(_RQ_CSRC / "torch_bindings.cpp")],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3", "-std=c++17", "--use_fast_math",
        # Architectures the kernel is verified to compile for. The actual
        # rotorquant_kv.py module does runtime detection via nvcc
        # --list-gpu-arch and drops anything the local toolkit doesn't
        # support (CUDA 13.x silently drops sm_70 / sm_75).
        # Verified compile-clean with nvcc 12.6 (2026-04-26):
        "-gencode=arch=compute_70,code=sm_70",   # V100 (Volta)
        "-gencode=arch=compute_75,code=sm_75",   # T4 (Turing)
        "-gencode=arch=compute_80,code=sm_80",   # A100 (Ampere)
        "-gencode=arch=compute_86,code=sm_86",   # A10G / RTX 3090
        "-gencode=arch=compute_89,code=sm_89",   # RTX 4090 (Ada)
        "-gencode=arch=compute_90,code=sm_90",   # H100 (Hopper)
        "-gencode=arch=compute_120,code=sm_120", # RTX 5090 (Blackwell)
    ],
)
_pack = _ext.rotorquant_planar3_pack
_unpack = _ext.rotorquant_planar3_unpack
```

This builds on first import (slow first call, fast afterwards). Useful
for development iteration; switch to Option A for production builds.

### Verify Phase 2b wiring

After Option A or B, this should work:

```bash
docker run --rm --entrypoint /bin/sh rq-vllm:phase2 -c '
python3 -c "
import torch
torch.zeros(128, device=\"cuda\", dtype=torch.float16)  # init CUDA
from torch.ops._C import rotorquant_planar3_pack, rotorquant_planar3_unpack
print(\"phase 2b ops registered: OK\")
"'
```

## Step 2: replace passthrough stubs (Phase 2c part 1)

In `vllm/v1/attention/ops/rotorquant_kv.py` (the file we wrote in Phase 1
as passthrough), replace the body of `rotorquant_kv_write` and
`rotorquant_kv_read` with calls into the new ops.

The challenging part: vLLM's KV cache is shaped
`(2, num_blocks, block_size, num_kv_heads, head_size)` for fp16. For
RotorQuant, the `head_size` axis becomes a packed-byte axis — each
128-element segment of the original head_size gets packed to 50 bytes.
The KV cache shape for rotorquant_planar3 is therefore
`(2, num_blocks, block_size, num_kv_heads, head_size * 50 / 128)`.

Concrete write op:

```python
def rotorquant_kv_write(
    key, value, key_cache, value_cache, slot_mapping,
    kv_cache_dtype, k_scale=None, v_scale=None,
):
    assert kv_cache_dtype == "rotorquant_planar3"
    # key shape: [num_tokens, num_kv_heads, head_size]
    # We pack each head's head_size-dim vector as ceil(head_size/128) blocks.
    num_tokens, num_kv_heads, head_size = key.shape
    assert head_size % 128 == 0, f"head_size {head_size} must be multiple of 128"
    blocks_per_head = head_size // 128
    n_blocks = num_tokens * num_kv_heads * blocks_per_head

    key_flat = key.contiguous().view(-1)  # [num_tokens * num_kv_heads * head_size]
    value_flat = value.contiguous().view(-1)

    # We need scratch packed buffers, then a scatter into key_cache/value_cache
    # at slot_mapping locations. For Phase 2c step 1 we allocate scratch and do
    # the scatter in Python; later we should fuse into a single CUDA kernel.
    PACKED_BYTES_PER_HEAD = blocks_per_head * 50
    packed_k = torch.empty(num_tokens * num_kv_heads * PACKED_BYTES_PER_HEAD,
                           device=key.device, dtype=torch.uint8)
    packed_v = torch.empty_like(packed_k)
    torch.ops._C.rotorquant_planar3_pack(key_flat, packed_k, n_blocks)
    torch.ops._C.rotorquant_planar3_pack(value_flat, packed_v, n_blocks)

    # Scatter packed_k into key_cache at the slot_mapping offsets.
    # slot_mapping[i] gives the linear slot index in (num_blocks * block_size)
    # space; multiply by (num_kv_heads * PACKED_BYTES_PER_HEAD) to get the byte
    # offset in key_cache's flat view.
    # (Implementation as a separate scatter op or via a fused write kernel is
    # left for the engineer doing the actual integration — see notes below.)
    raise NotImplementedError(
        "Phase 2c step 1 scaffold: scatter-into-paged-block needed here. "
        "Either (a) write a fused pack-and-scatter CUDA op, or (b) do the "
        "pack into scratch then index-copy into key_cache using slot_mapping."
    )
```

This is where the engineer integrating Phase 2 has to make a design call:

**(A) Fused pack-and-scatter kernel** — slowest to implement but fastest
   at runtime. Add a new kernel
   `kernel_planar3_kv_write(key, key_cache, slot_mapping, ...)` that
   does pack+scatter in one launch. Cleaner integration into vLLM's
   existing `reshape_and_cache_flash` pattern.

**(B) Separate pack + index_copy** — fastest to write. Allocates a
   per-step scratch buffer for the packed bytes, then uses
   `torch.index_copy_` or a small scatter kernel to write into the
   right paged-block offsets. Slower at runtime but easier to debug.

Recommend (B) for the FIRST integration to validate end-to-end
correctness, then optimize to (A) once Phase 3 PPL gate is green.

## Step 3: branch in FlashAttention KV-write (Phase 2c part 2)

In `vllm/v1/attention/backends/flash_attn.py` around line 817 (the
`reshape_and_cache_flash` call site — verified location in v0.19.1):

```python
# BEFORE (existing line):
reshape_and_cache_flash(
    key, value, key_cache, value_cache, slot_mapping,
    self.kv_cache_dtype, layer._k_scale, layer._v_scale,
)

# AFTER:
if self.kv_cache_dtype.startswith("rotorquant_"):
    from vllm.v1.attention.ops.rotorquant_kv import rotorquant_kv_write
    rotorquant_kv_write(
        key, value, key_cache, value_cache, slot_mapping,
        self.kv_cache_dtype, layer._k_scale, layer._v_scale,
    )
else:
    reshape_and_cache_flash(
        key, value, key_cache, value_cache, slot_mapping,
        self.kv_cache_dtype, layer._k_scale, layer._v_scale,
    )
```

There's an analogous read-side branch in the paged_attention call (look
for `kv_cache_dtype.startswith("fp8")` in the same file around the
attention forward pass) — add a `rotorquant_` branch that calls the
unpack op before passing K/V to the attention math.

## Step 4: update `STR_DTYPE_TO_TORCH_DTYPE` and `get_kv_cache_shape`

When real packed storage is used (Phase 2c, after the integration above
works), update:

- `vllm/utils/torch_utils.py:46`: change
  `"rotorquant_planar3": torch.float16` → `"rotorquant_planar3": torch.uint8`
- `vllm/v1/attention/backends/flash_attn.py:120-130`: when `cache_dtype_str
  .startswith("rotorquant_")`, return packed shape:
  ```python
  PACKED_BYTES_PER_HEAD = (head_size // 128) * 50
  return (2, num_blocks, block_size, num_kv_heads, PACKED_BYTES_PER_HEAD)
  ```

This is the shape change that makes the cache actually 5× smaller.
**Don't make this change until Step 1-3 work end-to-end** — otherwise the
shape mismatch will cascade into many vLLM internal validators.

## Step 5: validation gate (Phase 3 hard gate)

Once Steps 0-4 are green, run the perplexity comparison:

```bash
# llama.cpp baseline
docker compose --profile qwen36-27b up   # planar3 KV
python3 scripts/eval_perplexity.py \
    --model Qwen/Qwen3.6-27B \
    --bits 3.125 \
    --eval-dataset wikitext-2-raw-v1 \
    --server http://localhost:8080 \
    --output /tmp/llamacpp-planar3-ppl.json
docker compose down

# vLLM baseline
docker run -d --name rq-vllm-phase3 --gpus all -p 8080:8080 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e MODEL=Qwen/Qwen3.6-27B \
    -e ROTORQUANT_MODE=planar3 \
    rq-vllm:phase2
python3 scripts/eval_perplexity.py \
    --model Qwen/Qwen3.6-27B \
    --bits 3.125 \
    --eval-dataset wikitext-2-raw-v1 \
    --server http://localhost:8080 \
    --output /tmp/vllm-planar3-ppl.json
docker rm -f rq-vllm-phase3

# diff
python3 -c "
import json
a = json.load(open('/tmp/llamacpp-planar3-ppl.json'))
b = json.load(open('/tmp/vllm-planar3-ppl.json'))
print(f'llama.cpp ppl: {a[\"perplexity\"]}')
print(f'vLLM ppl:      {b[\"perplexity\"]}')
delta = abs(b['perplexity'] - a['perplexity']) / a['perplexity'] * 100
print(f'Δppl: {delta:.4f}%')
print('PASS' if delta <= 0.05 else 'FAIL — Phase 2 has a kernel or integration bug')
"
```

**Hard gate: Δppl ≤ 0.05%.** If above, debug:
- compare per-block packed bytes between substrates (math sanity)
- compare KV cache contents byte-for-byte after the same prompt
- compare per-token attention outputs between substrates

## Estimated effort summary

- Step 0 (kernel parity test): 30 min
- Step 1 (build wiring): half day
- Step 2 (Python ops, scaffold-then-fused): 1-2 days
- Step 3 (FlashAttention branches): 1 day
- Step 4 (storage shape switch): half day, but needs careful testing
- Step 5 (Δppl validation): half day plus iterations

**Total Phase 2b+2c+3: ~1 week** on a stable RTX 5090 box.
