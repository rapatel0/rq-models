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
    sources = [str(csrc / "planar3_kv.cu"), str(csrc / "torch_bindings.cpp")]
    _ext_cache = cpp_extension.load(
        name="rq_models_rotorquant",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            # nvcc 13.2 verified compile-clean for these arches:
            "-gencode=arch=compute_89,code=sm_89",   # RTX 4090 (Ada)
            "-gencode=arch=compute_90,code=sm_90",   # H100 (Hopper)
            "-gencode=arch=compute_120,code=sm_120", # RTX 5090 (Blackwell)
        ],
        verbose=False,
    )
    return _ext_cache


def _round_trip_planar3(t: torch.Tensor) -> torch.Tensor:
    """Apply pack-then-unpack to ``t`` via the planar3 CUDA kernels.

    ``t`` must be an fp16 CUDA tensor whose total numel is a multiple of 128
    (the planar3 block size). Returns a fresh fp16 tensor of the same shape.
    """
    n = t.numel()
    if n % QK_PLANAR3 != 0:
        raise ValueError(
            f"rotorquant_planar3: tensor numel {n} is not a multiple of "
            f"{QK_PLANAR3}; cannot pack")
    n_blocks = n // QK_PLANAR3
    flat = t.contiguous().view(-1)
    packed = torch.empty(
        n_blocks * PACKED_BYTES_PER_BLOCK,
        device=t.device,
        dtype=torch.uint8,
    )
    out_flat = torch.empty_like(flat)

    ext = _ext()
    ext.rotorquant_planar3_pack(flat, packed, n_blocks)
    ext.rotorquant_planar3_unpack(packed, out_flat, n_blocks)
    return out_flat.view(t.shape)


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
