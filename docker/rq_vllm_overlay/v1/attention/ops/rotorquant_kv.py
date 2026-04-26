# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the rq-models project

"""rq-models RotorQuant KV cache pack / unpack module.

Phase 1 of Sprint 004 (current): this module is the passthrough stub
that lets `--kv-cache-dtype rotorquant_planar3` flow through vLLM's
plumbing without crashing. The actual storage path falls through to
fp16 via vllm.utils.torch_utils.STR_DTYPE_TO_TORCH_DTYPE which maps
``rotorquant_planar3 -> torch.float16``. Tokens emitted with
``rotorquant_planar3`` should be bit-identical to those emitted with
``float16``.

Phase 2 of Sprint 004 (next): swap the passthrough functions below for
calls into the CUDA kernels at ``vllm/csrc/attention/rotorquant/``,
porting the planar3 rotation + Lloyd-Max codebook from
``johndpope/llama-cpp-turboquant@20efe75``. At that point:
- ``STR_DTYPE_TO_TORCH_DTYPE['rotorquant_planar3']`` becomes
  ``torch.uint8`` (or appropriate packed-byte type)
- ``FlashAttentionBackend.get_kv_cache_shape`` for rotorquant dtypes
  returns a packed shape (3 bpe instead of 16 bpe per element)
- The K/V write path in ``flash_attn.py`` (around the
  ``reshape_and_cache_flash`` call) gains a branch that calls
  ``rotorquant_kv_write_planar3`` instead of the fp16/fp8 op.
- The paged-attention read path gains a corresponding unpack branch.

See rq-models ``docs/sprints/SPRINT-004.md`` for the full plan and the
code-anchor table.
"""

from __future__ import annotations

from typing import Final

import torch


SUPPORTED_ROTORQUANT_DTYPES: Final[tuple[str, ...]] = (
    "rotorquant_planar3",
    # Sprint 005 will add: "rotorquant_iso3", "rotorquant_iso4",
    # "rotorquant_planar4".
)


def is_rotorquant_dtype(kv_cache_dtype: str) -> bool:
    """Return True if the dtype string identifies a RotorQuant variant."""
    return kv_cache_dtype.startswith("rotorquant_")


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
    """KV pack stub.

    Phase 1 passthrough: do nothing here; the caller continues into the
    standard ``reshape_and_cache_flash`` path which writes fp16 K, V into
    the (rotorquant_planar3 → fp16-mapped) cache tensor unchanged. Phase
    2 will replace this stub with a CUDA dispatch.

    Args mirror ``reshape_and_cache_flash`` for forward compatibility so
    the eventual production path can swap signatures with no caller-side
    changes.
    """
    if kv_cache_dtype not in SUPPORTED_ROTORQUANT_DTYPES:
        raise ValueError(
            f"rotorquant_kv_write called with non-RotorQuant dtype "
            f"{kv_cache_dtype!r}; supported: {SUPPORTED_ROTORQUANT_DTYPES}"
        )
    # Phase 1: explicit no-op. The real fp16 store happens via the
    # caller's reshape_and_cache_flash invocation; we only got here as
    # a future hook point. Keeping this empty function in place lets
    # Phase 2 land as a one-line edit at the call site.
    return


def rotorquant_kv_read(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KV unpack stub.

    Phase 1 passthrough: returns the cache tensors as-is (fp16 storage).
    Phase 2 will dispatch to a CUDA kernel that dequantizes the packed
    3-bpe layout back to fp16 for the attention matmul.
    """
    if kv_cache_dtype not in SUPPORTED_ROTORQUANT_DTYPES:
        raise ValueError(
            f"rotorquant_kv_read called with non-RotorQuant dtype "
            f"{kv_cache_dtype!r}; supported: {SUPPORTED_ROTORQUANT_DTYPES}"
        )
    return key_cache, value_cache
