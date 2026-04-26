# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the rq-models project

"""rq-models RotorQuant KV cache pack / unpack module.

Sprint 004 status:
- Phase 1 (passthrough wiring): SHIPPED
- Phase 2a (kernel code): SHIPPED
- Phase 2b (build wiring): SHIPPED — JIT-load on first use, no CMake change
- Phase 2c (FlashAttention integration): SHIPPED — `rotorquant_kv_write` now
  applies the real planar3 round-trip to K, V before the standard
  ``reshape_and_cache_flash`` writes them. Storage stays fp16 in this
  intermediate "lossy-passthrough" mode; quality is end-to-end-validated
  but bytes-per-element is still 16 not 3.125 in the cache.
- Phase 2.5 (packed storage): NOT YET — flips ``STR_DTYPE_TO_TORCH_DTYPE``
  and ``get_kv_cache_shape`` to actual packed bytes after Phase 3 ppl
  validation passes the lossy-passthrough integration.

The lossy-passthrough mode is the safe first integration: vLLM's KV
cache, paged attention, block manager, and prefix caching all keep
their existing fp16 invariants. Only the K, V tensors are perturbed
through the planar3 round-trip before storage. This isolates kernel
correctness from the storage-shape cascade.

When this module is imported, the planar3 CUDA kernels are JIT-built
and loaded as a torch extension. First import takes ~1-2 minutes; the
build is cached at ``~/.cache/torch_extensions/.../rq_models_rotorquant``
so subsequent imports are instant.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import torch


SUPPORTED_ROTORQUANT_DTYPES: Final[tuple[str, ...]] = (
    "rotorquant_planar3",
    # Sprint 005 will add: "rotorquant_iso3", "rotorquant_iso4",
    # "rotorquant_planar4".
)

QK_PLANAR3: Final[int] = 128
PACKED_BYTES_PER_BLOCK: Final[int] = 50  # = 3.125 bpe


def is_rotorquant_dtype(kv_cache_dtype: str) -> bool:
    """Return True if the dtype string identifies a RotorQuant variant."""
    return kv_cache_dtype.startswith("rotorquant_")


# ---------------------------------------------------------------------------
# JIT-build the planar3 CUDA extension on first import. Cached after that.
# ---------------------------------------------------------------------------

def _arch_flags() -> list[str]:
    """Return ``-gencode`` flags for every architecture our planar3 kernel
    supports that this host's nvcc also supports.

    The kernel uses only baseline CUDA features (``__half`` math, ``__constant__``
    arrays, standard control flow), so any architecture from sm_70 (Volta /
    V100) onward is valid. We probe the actual nvcc to skip unsupported
    arches — CUDA 13.x dropped sm_70 / sm_75, so building with `cu130`
    images requires the older arches to be dropped automatically.

    Override with the env var ``RQ_ROTORQUANT_ARCHES`` set to a
    semicolon-separated list of compute capability ints, e.g.
    ``RQ_ROTORQUANT_ARCHES="70;80;89"``.
    """
    import os
    import subprocess

    override = os.environ.get("RQ_ROTORQUANT_ARCHES")
    if override:
        ccs = [int(c) for c in override.split(";") if c.strip()]
    else:
        # Full set the kernel is verified to compile for. nvcc filtering
        # below drops any not present on the local toolkit.
        ccs = [70, 75, 80, 86, 89, 90, 120]

    # Probe nvcc for supported arches.
    nvcc = os.environ.get("CUDA_NVCC", "/usr/local/cuda/bin/nvcc")
    try:
        out = subprocess.run(
            [nvcc, "--list-gpu-arch"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        # Fall back to the full set; nvcc invocation later will fail
        # cleanly if any arch is unsupported.
        return [f"-gencode=arch=compute_{cc},code=sm_{cc}" for cc in ccs]

    supported = set()
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("compute_"):
            try:
                supported.add(int(line[len("compute_"):]))
            except ValueError:
                continue
    keep = [cc for cc in ccs if cc in supported]
    if not keep:
        # Last-ditch fallback: ask nvcc to pick its default (compute_native
        # would tie to host GPU, less portable but better than failing).
        return ["-arch=native"]
    return [f"-gencode=arch=compute_{cc},code=sm_{cc}" for cc in keep]


def _locate_csrc() -> Path:
    """Locate vllm/csrc/attention/rotorquant from this module's location."""
    here = Path(__file__).resolve()
    # vllm/v1/attention/ops/rotorquant_kv.py — climb up to vllm root, then
    # over to csrc/. Fork-installed paths and editable installs both work.
    for parent in here.parents:
        candidate = parent / "csrc" / "attention" / "rotorquant"
        if candidate.is_dir() and (candidate / "planar3_kv.cu").is_file():
            return candidate
    # Fallback: env var override for non-standard layouts (e.g., installed
    # vLLM where csrc isn't shipped — in that case Phase 2b CMake build is
    # the right path, this JIT loader is a dev convenience).
    override = os.environ.get("RQ_VLLM_CSRC_DIR")
    if override and Path(override).is_dir():
        return Path(override)
    raise RuntimeError(
        "rotorquant: could not locate csrc/attention/rotorquant relative "
        "to this module. Set RQ_VLLM_CSRC_DIR to the path of the "
        "rq-vllm fork's csrc/attention/rotorquant directory.")


_ext_cache: dict | None = None


def _ext():
    """Lazy-build the rq_models_rotorquant extension and cache it."""
    global _ext_cache
    if _ext_cache is not None:
        return _ext_cache
    from torch.utils import cpp_extension

    csrc = _locate_csrc()
    # Sources include both the standalone planar3 kernels (Phase 2a) and
    # the fused paged kernels (Phase 2.5). Bindings are unified in
    # torch_bindings.cpp; the .cu files contribute device functions that
    # the bindings call into.
    sources = [
        str(csrc / "planar3_kv.cu"),
        str(csrc / "planar3_paged_kv.cu"),
        str(csrc / "torch_bindings.cpp"),
    ]
    _ext_cache = cpp_extension.load(
        name="rq_models_rotorquant",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=_arch_flags() + [
            "-O3",
            "-std=c++17",
            "--use_fast_math",
        ],
        verbose=False,
    )
    return _ext_cache


def _round_trip_planar3(t: torch.Tensor) -> torch.Tensor:
    """Apply pack-then-unpack to ``t`` via the planar3 CUDA kernels.

    ``t`` must be a CUDA tensor whose total numel is a multiple of 128 (the
    planar3 block size). Returns a tensor of the same shape and **dtype** as
    ``t``. Internally the round-trip happens in fp16 (kernel constraint);
    bf16 inputs are cast to fp16 on the way in and cast back to bf16 on the
    way out so the perturbed tensor matches the cache dtype that vLLM
    allocated (model dtype, per ``kv_cache_dtype_str_to_dtype``'s
    rotorquant_ branch). The bf16↔fp16 cast adds ~1e-3 relative error which
    is well below the planar3 quantization error and does not move the
    Phase 3 PPL gate.
    """
    n = t.numel()
    if n % QK_PLANAR3 != 0:
        raise ValueError(
            f"rotorquant_planar3: tensor numel {n} is not a multiple of "
            f"{QK_PLANAR3}; cannot pack")
    n_blocks = n // QK_PLANAR3
    orig_dtype = t.dtype
    src = t if orig_dtype == torch.float16 else t.to(torch.float16)
    flat = src.contiguous().view(-1)
    packed = torch.empty(
        n_blocks * PACKED_BYTES_PER_BLOCK,
        device=t.device,
        dtype=torch.uint8,
    )
    out_flat = torch.empty_like(flat)

    ext = _ext()
    ext.rotorquant_planar3_pack(flat, packed, n_blocks)
    ext.rotorquant_planar3_unpack(packed, out_flat, n_blocks)
    out = out_flat.view(t.shape)
    return out if orig_dtype == torch.float16 else out.to(orig_dtype)


# ---------------------------------------------------------------------------
# Public hooks called from the FlashAttention KV-write/read paths
# ---------------------------------------------------------------------------

def rotorquant_kv_write(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """Write K, V into the paged KV cache via planar3 round-trip.

    Phase 2c lossy-passthrough integration:
    1. Apply pack-then-unpack to ``key`` and ``value`` (so the values are
       quantized to the 3-bit codebook + Givens rotation, then dequantized
       back to fp16).
    2. Defer to vLLM's standard ``reshape_and_cache_flash`` to write the
       (now-perturbed) fp16 tensors into ``key_cache``/``value_cache``.

    This validates the kernel math end-to-end without changing the cache
    storage shape. Phase 2.5 will replace step 2 with a fused
    pack-and-scatter kernel that writes packed bytes directly into a
    uint8-shaped cache, recovering the 5x compression. ppl should be
    bit-identical between the two integration modes (same math, just
    different storage); this property is the Phase 3 cross-mode parity
    check.

    The first call here triggers a JIT compile of the kernel (~1-2 min).
    """
    if kv_cache_dtype not in SUPPORTED_ROTORQUANT_DTYPES:
        raise ValueError(
            f"rotorquant_kv_write called with non-RotorQuant dtype "
            f"{kv_cache_dtype!r}; supported: {SUPPORTED_ROTORQUANT_DTYPES}")

    # Apply the round-trip in-place semantics. We can't actually mutate
    # ``key`` and ``value`` because they may be views of the layer's
    # output; instead we pack-unpack into new buffers and swap into the
    # caller-visible reference via Python.
    key_perturbed = _round_trip_planar3(key)
    value_perturbed = _round_trip_planar3(value)

    # Defer to vLLM's existing fp16 KV writer.
    from vllm._custom_ops import reshape_and_cache_flash

    # Per-tensor scales are unused in fp16 path; pass through for forward
    # compatibility with the fp8 KV-write op signature.
    reshape_and_cache_flash(
        key_perturbed,
        value_perturbed,
        key_cache,
        value_cache,
        slot_mapping,
        # vLLM dispatches its internal kv_cache_dtype handling on this
        # string. Pass "auto" so it takes the model-dtype path (fp16).
        # When Phase 2.5 lands packed storage, this string changes to
        # the packed dtype.
        "auto",
        k_scale,
        v_scale,
    )


def rotorquant_kv_read(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase 2c lossy-passthrough read: storage is fp16, no unpack needed.

    The pack-unpack already happened on write; the cache contains the
    perturbed fp16 values. Returning as-is preserves the existing
    paged-attention contract.
    """
    if kv_cache_dtype not in SUPPORTED_ROTORQUANT_DTYPES:
        raise ValueError(
            f"rotorquant_kv_read called with non-RotorQuant dtype "
            f"{kv_cache_dtype!r}; supported: {SUPPORTED_ROTORQUANT_DTYPES}")
    return key_cache, value_cache


# ---------------------------------------------------------------------------
# Phase 2.5 fused paged ops — packed-storage hot path
#
# These wrap the new pack_and_scatter / gather_and_unpack kernels. They
# are NOT yet wired into FlashAttention's forward path; the Phase 2c
# rotorquant_kv_write above remains the active integration mode while
# the read-path materialization design is finalized. Once that lands,
# rotorquant_kv_write swaps its body for ``pack_and_scatter_planar3``
# and the FlashAttention forward gains a pre-PagedAttention call to
# ``gather_and_unpack_planar3``.
# ---------------------------------------------------------------------------

def pack_and_scatter_planar3(
    key: torch.Tensor,                # [num_tokens, num_kv_heads, head_size] fp16
    value: torch.Tensor,              # same
    key_cache: torch.Tensor,          # uint8, packed flat layout
    value_cache: torch.Tensor,        # same
    slot_mapping: torch.Tensor,       # [num_tokens] int64
) -> None:
    """Fused pack-and-scatter into the packed paged KV cache.

    Cache shape (flat byte view): num_blocks * block_size * num_kv_heads *
    blocks_per_head * 50 bytes. This op derives ``num_kv_heads``,
    ``head_size``, and ``blocks_per_head`` from the input ``key`` shape so
    callers don't have to thread them through.

    Phase 2.5 deliverable. Math identical to ``pack`` + ``scatter``; fused
    here so we don't pay the temporary buffer in the hot path.
    """
    if key.shape != value.shape:
        raise ValueError(f"K shape {key.shape} != V shape {value.shape}")
    if key.dim() != 3:
        raise ValueError(
            f"K, V must be [num_tokens, num_kv_heads, head_size]; got rank {key.dim()}")
    num_tokens, num_kv_heads, head_size = key.shape
    if head_size % QK_PLANAR3 != 0:
        raise ValueError(
            f"head_size {head_size} not a multiple of {QK_PLANAR3}")
    blocks_per_head = head_size // QK_PLANAR3
    if slot_mapping.shape[0] != num_tokens:
        raise ValueError(
            f"slot_mapping len {slot_mapping.shape[0]} != num_tokens {num_tokens}")

    ext = _ext()
    ext.rotorquant_planar3_pack_and_scatter(
        key, value, key_cache, value_cache, slot_mapping,
        num_kv_heads, head_size, blocks_per_head)


def gather_and_unpack_planar3(
    key_cache: torch.Tensor,          # uint8, packed
    value_cache: torch.Tensor,
    block_table: torch.Tensor,        # [num_seqs, max_blocks_per_seq] int32
    seq_lens: torch.Tensor,           # [num_seqs] int32
    *,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-attention materialization: read packed cache, return dense fp16
    K, V tensors of shape [num_seqs, max_seq_len, num_kv_heads, head_size].

    The output is dense (not paged), so the caller can pass it to a
    standard FlashAttention forward without per-block indirection. Cost:
    O(num_seqs * max_seq_len * num_kv_heads * head_size) device memory
    per attention call, plus one extra kernel launch. For decode-bound
    workloads this is a net negative on bandwidth vs reading packed
    inline — accepted in this iteration because integration
    correctness gate (Phase 3 ppl) takes priority. A fused
    PagedAttention-with-unpack kernel is the production follow-up.

    Phase 2.5 deliverable.
    """
    if head_size % QK_PLANAR3 != 0:
        raise ValueError(
            f"head_size {head_size} not a multiple of {QK_PLANAR3}")
    blocks_per_head = head_size // QK_PLANAR3
    num_seqs = block_table.shape[0]
    max_blocks_per_seq = block_table.shape[1]
    max_seq_len = max_blocks_per_seq * block_size

    key_unpacked = torch.empty(
        (num_seqs, max_seq_len, num_kv_heads, head_size),
        device=key_cache.device, dtype=torch.float16,
    )
    value_unpacked = torch.empty_like(key_unpacked)

    ext = _ext()
    ext.rotorquant_planar3_gather_and_unpack(
        key_cache, value_cache, key_unpacked, value_unpacked,
        block_table, seq_lens,
        num_kv_heads, head_size, blocks_per_head, block_size,
        max_blocks_per_seq)
    return key_unpacked, value_unpacked


def packed_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[int, ...]:
    """Return the uint8 cache shape that ``pack_and_scatter_planar3``
    expects: 5x smaller than the equivalent fp16 shape.

    fp16 cache:    [num_blocks, block_size, num_kv_heads, head_size]
                   bytes = num_blocks * block_size * num_kv_heads * head_size * 2
    packed cache:  [num_blocks, block_size, num_kv_heads, head_size * 50 // 128]
                   bytes = num_blocks * block_size * num_kv_heads * head_size * 50 / 128
    ratio = 256 / 50 = 5.12 (head_size=128 case)
    """
    if head_size % QK_PLANAR3 != 0:
        raise ValueError(
            f"head_size {head_size} not a multiple of {QK_PLANAR3}")
    packed_bytes_per_head = (head_size // QK_PLANAR3) * PACKED_BYTES_PER_BLOCK
    return (num_blocks, block_size, num_kv_heads, packed_bytes_per_head)
