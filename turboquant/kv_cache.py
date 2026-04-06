"""
TurboKVCache: DynamicCache subclass with TurboQuant KV compression.

Overrides update() to quantize new K/V tokens on insertion.
Overrides key_cache/value_cache properties to dequantize on retrieval.

Design choices (Sprint 001):
- Full dequantize on every retrieval (O(N) per call, O(N²) total at long context).
  Pre-allocated buffer with incremental dequant is deferred to Sprint 002.
- K path: TurboQuantProd (unbiased inner product for Q @ Kᵀ attention scores).
- V path: TurboQuantMSE  (MSE reconstruction for softmax @ V value aggregation).
- GQA: quantize at num_key_value_heads granularity; HF handles broadcasting.
"""

from __future__ import annotations

from typing import Optional, Any

import torch
from torch import Tensor

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    from transformers import DynamicCache  # older transformers

from turboquant.config import BitConfig
from turboquant.outlier import OutlierSplitter, KQuantized, VQuantized


class TurboKVCache(DynamicCache):
    """
    KV cache that compresses keys and values via TurboQuant.

    Usage:
        cache = TurboKVCache(PRESET_3_5BIT)
        outputs = model.generate(..., past_key_values=cache)

    The cache is a drop-in replacement for DynamicCache. It stores quantized
    representations internally and dequantizes transparently on retrieval.

    Shape conventions (GQA):
        key_states:   [batch, num_kv_heads, 1, head_dim]
        value_states: [batch, num_kv_heads, 1, head_dim]
        returned K:   [batch, num_kv_heads, seq_len, head_dim]
        returned V:   [batch, num_kv_heads, seq_len, head_dim]

    The caller (HF attention) handles broadcasting from num_kv_heads to
    num_query_heads via repeat_kv / expand — we never return query-heads shapes.
    """

    def __init__(self, config: BitConfig, seed: int = 42) -> None:
        super().__init__()
        self.config = config

        # One OutlierSplitter shared across all layers (data-oblivious).
        # Sharing Π/S across 64 layers saves 64× memory and is valid because
        # TurboQuant is data-independent.
        self._splitter = OutlierSplitter(config, seed=seed)

        # Quantized storage per layer.
        # Each element is a list of per-token tensors (grown dynamically).
        # _k_store[layer_idx] = KQuantized with [..., seq_len, ...]
        self._k_store: list[Optional[KQuantized]] = []
        self._v_store: list[Optional[VQuantized]] = []

        # Track the model's native dtype for output casting
        self._model_dtype: torch.dtype = torch.float32

    # ------------------------------------------------------------------
    # Core override: update() — called for each new token, each layer
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: Tensor,    # [batch, num_kv_heads, 1, head_dim]
        value_states: Tensor,  # [batch, num_kv_heads, 1, head_dim]
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Quantize and store new K/V tokens; return full dequantized sequence.

        This is the main entry point called by HuggingFace attention modules.
        """
        self._model_dtype = key_states.dtype

        # Quantize the new token
        qk_new = self._splitter.quantize_k(key_states)
        qv_new = self._splitter.quantize_v(value_states)

        # Grow the layer-indexed stores
        while len(self._k_store) <= layer_idx:
            self._k_store.append(None)
            self._v_store.append(None)

        # Append to existing sequence or start fresh
        if self._k_store[layer_idx] is None:
            self._k_store[layer_idx] = qk_new
            self._v_store[layer_idx] = qv_new
        else:
            self._k_store[layer_idx] = _concat_kquantized(
                self._k_store[layer_idx], qk_new
            )
            self._v_store[layer_idx] = _concat_vquantized(
                self._v_store[layer_idx], qv_new
            )

        # Dequantize full sequence and return
        k_full = self._splitter.dequantize_k(
            self._k_store[layer_idx], out_dtype=self._model_dtype
        )
        v_full = self._splitter.dequantize_v(
            self._v_store[layer_idx], out_dtype=self._model_dtype
        )
        return k_full, v_full

    # ------------------------------------------------------------------
    # Property overrides — used when callers access the cache directly
    # ------------------------------------------------------------------

    @property
    def key_cache(self) -> list[Tensor]:
        """Dequantize all stored K sequences (one tensor per layer)."""
        result = []
        for i, qk in enumerate(self._k_store):
            if qk is None:
                result.append(torch.zeros(1))
            else:
                result.append(
                    self._splitter.dequantize_k(qk, out_dtype=self._model_dtype)
                )
        return result

    @property
    def value_cache(self) -> list[Tensor]:
        """Dequantize all stored V sequences (one tensor per layer)."""
        result = []
        for i, qv in enumerate(self._v_store):
            if qv is None:
                result.append(torch.zeros(1))
            else:
                result.append(
                    self._splitter.dequantize_v(qv, out_dtype=self._model_dtype)
                )
        return result

    # ------------------------------------------------------------------
    # DynamicCache compatibility
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the current sequence length for the given layer."""
        if layer_idx >= len(self._k_store) or self._k_store[layer_idx] is None:
            return 0
        # Infer seq_len from the indices shape [..., seq, d_out]
        return self._k_store[layer_idx].out_prod.indices.shape[-2]

    def get_max_length(self) -> Optional[int]:
        """TurboKVCache has no fixed maximum length."""
        return None

    def __len__(self) -> int:
        return len(self._k_store)

    def __repr__(self) -> str:
        n_layers = len(self._k_store)
        seq = self.get_seq_length(0) if n_layers > 0 else 0
        return (
            f"TurboKVCache({self.config.label}, "
            f"layers={n_layers}, seq_len={seq})"
        )

    def memory_stats(self) -> dict:
        """Return approximate memory usage statistics."""
        k_bytes = 0
        v_bytes = 0
        for qk in self._k_store:
            if qk is not None:
                k_bytes += _kquantized_bytes(qk)
        for qv in self._v_store:
            if qv is not None:
                v_bytes += _vquantized_bytes(qv)
        return {
            "k_bytes": k_bytes,
            "v_bytes": v_bytes,
            "total_mb": (k_bytes + v_bytes) / 1024**2,
        }


# ---------------------------------------------------------------------------
# Helpers: concatenate quantized tensors along the sequence dimension
# ---------------------------------------------------------------------------

def _concat_kquantized(a: KQuantized, b: KQuantized) -> KQuantized:
    """Concatenate two KQuantized objects along the seq dimension (dim=-2)."""
    out_prod = _concat_prod(a.out_prod, b.out_prod) if a.out_prod is not None else None
    out_norm = torch.cat([a.out_norm, b.out_norm], dim=-1) if a.out_norm is not None else None
    return KQuantized(
        out_prod=out_prod,
        out_norm=out_norm,
        reg_prod=_concat_prod(a.reg_prod, b.reg_prod),
        reg_norm=torch.cat([a.reg_norm, b.reg_norm], dim=-1),
    )


def _concat_vquantized(a: VQuantized, b: VQuantized) -> VQuantized:
    """Concatenate two VQuantized objects along the seq dimension (dim=-2)."""
    from turboquant.core import MSEQuantized
    if a.out_mse is not None:
        out_mse = MSEQuantized(
            indices=torch.cat([a.out_mse.indices, b.out_mse.indices], dim=-2),
            norm=torch.cat([a.out_mse.norm, b.out_mse.norm], dim=-1),
        )
    else:
        out_mse = None
    return VQuantized(
        out_mse=out_mse,
        reg_mse=MSEQuantized(
            indices=torch.cat([a.reg_mse.indices, b.reg_mse.indices], dim=-2),
            norm=torch.cat([a.reg_mse.norm, b.reg_mse.norm], dim=-1),
        ),
    )


def _concat_prod(a, b):
    from turboquant.core import ProdQuantized
    return ProdQuantized(
        indices=torch.cat([a.indices, b.indices], dim=-2),
        qjl=torch.cat([a.qjl, b.qjl], dim=-2),
        gamma=torch.cat([a.gamma, b.gamma], dim=-1),
    )


def _kquantized_bytes(qk: KQuantized) -> int:
    """Approximate bytes used by a KQuantized tensor."""
    def prod_bytes(p, norm):
        return (
            p.indices.numel()          # uint8 → 1 byte each
            + p.qjl.numel()            # int8  → 1 byte each
            + p.gamma.numel() * 2      # float16 → 2 bytes
            + norm.numel() * 4         # float32 → 4 bytes
        )
    return prod_bytes(qk.out_prod, qk.out_norm) + prod_bytes(qk.reg_prod, qk.reg_norm)


def _vquantized_bytes(qv: VQuantized) -> int:
    def mse_bytes(m):
        return m.indices.numel() + m.norm.numel() * 4
    return mse_bytes(qv.out_mse) + mse_bytes(qv.reg_mse)
