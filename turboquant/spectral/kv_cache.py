"""
SpectralKVCache: DynamicCache subclass using SpectralQuant compression.

Drop-in replacement for TurboKVCache and HuggingFace DynamicCache.
Attention layers use SpectralQuantizer for KV storage.
SSM/linear-attention layers pass through via standard DynamicCache machinery.

Usage:
    from turboquant.spectral import SpectralKVCache, load_calibration
    cal = load_calibration("calibration/calibration-qwen3.5-9b.safetensors")
    cache = SpectralKVCache(cal, config=model.config)
    outputs = model.generate(..., past_key_values=cache)

For hybrid models (Qwen3.5, Mamba, etc.) pass model.config so that the
internal layers list is initialized with the correct LinearAttentionLayer /
DynamicLayer types. Without config, lazy DynamicLayer init is used (works
for pure full-attention models).
"""

from __future__ import annotations

from typing import Optional, Any

import torch
from torch import Tensor

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    from transformers import DynamicCache

from turboquant.spectral.calibrator import LayerCalibration
from turboquant.spectral.quantizer import SpectralQuantizer, QuantizedKV
from turboquant.spectral.store import CalibrationNotFoundError


class SpectralKVCache(DynamicCache):
    """
    KV cache with SpectralQuant compression for attention layers.

    Attention layers (those present in calibration data) are compressed.
    Linear-attention / SSM layers use the standard DynamicCache machinery
    (LinearAttentionLayer objects handle their own conv/recurrent state).

    Args:
        calibration: dict[layer_idx → LayerCalibration] from CalibrationStore.load()
        config:      optional HuggingFace model config. Required for hybrid models
                     (Qwen3.5, Jamba, etc.) so linear-attention layers get proper
                     LinearAttentionLayer instances in self.layers.
        device:      device for calibration tensors (defaults to first call device)
    """

    def __init__(
        self,
        calibration: dict[int, LayerCalibration],
        config=None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(config=config)
        self._quantizer = SpectralQuantizer(calibration, device=device)
        self._attn_layers: set[int] = set(calibration.keys())

        # Spectral storage: dict layer_idx → list of QuantizedKV tokens
        self._k_spectral: dict[int, list[QuantizedKV]] = {}
        self._v_spectral: dict[int, list[QuantizedKV]] = {}

    def update(
        self,
        key_states: Tensor,     # [batch, n_kv_heads, seq, head_dim]
        value_states: Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Store new K/V tokens and return full K/V sequences for attention.

        For attention layers in calibration: quantize → store → decode.
        For linear-attention / SSM layers: delegate to parent (DynamicCache).
        """
        if layer_idx not in self._attn_layers:
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        device = key_states.device

        if self._quantizer._device is None:
            self._quantizer._device = device

        if layer_idx not in self._k_spectral:
            self._k_spectral[layer_idx] = []
            self._v_spectral[layer_idx] = []

        seq_len = key_states.shape[-2]
        if seq_len > 1:
            # Prefill: encode token by token
            for t in range(seq_len):
                k_tok = key_states[:, :, t:t+1, :]
                v_tok = value_states[:, :, t:t+1, :]
                self._k_spectral[layer_idx].append(self._quantizer.encode_k(k_tok, layer_idx))
                self._v_spectral[layer_idx].append(self._quantizer.encode_v(v_tok, layer_idx))
        else:
            self._k_spectral[layer_idx].append(self._quantizer.encode_k(key_states, layer_idx))
            self._v_spectral[layer_idx].append(self._quantizer.encode_v(value_states, layer_idx))

        k_full = self._quantizer.decode_k(self._k_spectral[layer_idx], layer_idx)
        v_full = self._quantizer.decode_v(self._v_spectral[layer_idx], layer_idx)

        dtype = key_states.dtype
        return k_full.to(dtype=dtype, device=device), v_full.to(dtype=dtype, device=device)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for the given layer."""
        if layer_idx in self._attn_layers:
            return len(self._k_spectral.get(layer_idx, []))
        return super().get_seq_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        return None

    @property
    def seen_tokens(self) -> int:
        """Number of tokens processed (for compatibility with HF generate)."""
        for tokens in self._k_spectral.values():
            if tokens:
                return len(tokens)
        return super().seen_tokens if hasattr(super(), "seen_tokens") else 0

    def compression_ratio(self) -> dict:
        """
        Estimate compression ratio vs f16 baseline.
        Returns stats dict.
        """
        spectral_bytes = 0
        f16_bytes = 0
        total_tokens = 0

        for layer_idx, tokens in self._k_spectral.items():
            if not tokens:
                continue
            cal = self._quantizer._cal[layer_idx]
            n_heads = tokens[0].signal_indices.shape[0]
            head_dim = cal.eigvec_k.shape[1]
            n_tokens = len(tokens)
            total_tokens = max(total_tokens, n_tokens)

            if tokens[0].raw is not None:
                spectral_bytes += n_tokens * n_heads * head_dim * 2
            else:
                spectral_bytes += n_tokens * n_heads * 2  # 1-byte signal + 1-byte noise idx per head
            f16_bytes += n_tokens * n_heads * head_dim * 2

        for layer_idx, tokens in self._v_spectral.items():
            if not tokens:
                continue
            cal = self._quantizer._cal[layer_idx]
            n_heads = tokens[0].signal_indices.shape[0]
            head_dim = cal.eigvec_v.shape[1]
            n_tokens = len(tokens)
            if tokens[0].raw is None:
                spectral_bytes += n_tokens * n_heads * 2
            else:
                spectral_bytes += n_tokens * n_heads * head_dim * 2
            f16_bytes += n_tokens * n_heads * head_dim * 2

        ratio = f16_bytes / spectral_bytes if spectral_bytes > 0 else 0
        return {
            "total_tokens": total_tokens,
            "spectral_bytes": spectral_bytes,
            "f16_bytes": f16_bytes,
            "compression_ratio": ratio,
        }
