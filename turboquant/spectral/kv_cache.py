"""
SpectralKVCache: DynamicCache subclass using SpectralQuant compression.

Drop-in replacement for TurboKVCache and HuggingFace DynamicCache.
Attention layers use SpectralQuantizer for KV storage.
SSM/recurrent layers pass through uncompressed.

Usage:
    from turboquant.spectral import SpectralKVCache, load_calibration
    cal = load_calibration("calibration/calibration-qwen3.5-9b.safetensors")
    cache = SpectralKVCache(cal)
    outputs = model.generate(..., past_key_values=cache)
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
    SSM/recurrent layers (not in calibration data) pass through as raw f16.

    Args:
        calibration: dict[layer_idx → LayerCalibration] from CalibrationStore.load()
        device: device for calibration tensors (defaults to first call device)
    """

    def __init__(
        self,
        calibration: dict[int, LayerCalibration],
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self._quantizer = SpectralQuantizer(calibration, device=device)
        self._attn_layers: set[int] = set(calibration.keys())

        # Spectral storage: list per layer of QuantizedKV tokens
        self._k_spectral: list[list[QuantizedKV]] = []
        self._v_spectral: list[list[QuantizedKV]] = []

        # Fallback storage for non-attention layers (standard DynamicCache behavior)
        # DynamicCache already has key_cache / value_cache lists for this

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self._k_spectral) <= layer_idx:
            self._k_spectral.append([])
            self._v_spectral.append([])

    def update(
        self,
        key_states: Tensor,     # [batch, n_kv_heads, seq=1, head_dim]
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Store new K/V tokens and return full K/V sequences for attention.

        For attention layers: quantize → store → decode on retrieval.
        For SSM layers: delegate to DynamicCache (no quantization).
        """
        if layer_idx not in self._attn_layers:
            # SSM/recurrent layer — use parent DynamicCache behavior
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        self._ensure_layer(layer_idx)
        device = key_states.device

        # Move quantizer calibration to this device on first use
        if self._quantizer._device is None:
            self._quantizer._device = device

        # Handle empty cache first-token edge case
        if key_states.shape[-2] > 1:
            # Prefill: process multiple tokens at once
            for t in range(key_states.shape[-2]):
                k_tok = key_states[:, :, t:t+1, :]
                v_tok = value_states[:, :, t:t+1, :]
                self._k_spectral[layer_idx].append(
                    self._quantizer.encode_k(k_tok, layer_idx)
                )
                self._v_spectral[layer_idx].append(
                    self._quantizer.encode_v(v_tok, layer_idx)
                )
        else:
            # Decode: single new token
            self._k_spectral[layer_idx].append(
                self._quantizer.encode_k(key_states, layer_idx)
            )
            self._v_spectral[layer_idx].append(
                self._quantizer.encode_v(value_states, layer_idx)
            )

        # Decode full sequence for attention
        k_full = self._quantizer.decode_k(self._k_spectral[layer_idx], layer_idx)
        v_full = self._quantizer.decode_v(self._v_spectral[layer_idx], layer_idx)

        # Ensure correct dtype
        dtype = key_states.dtype
        return k_full.to(dtype=dtype, device=device), v_full.to(dtype=dtype, device=device)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return number of cached tokens for the given layer."""
        if layer_idx in self._attn_layers and layer_idx < len(self._k_spectral):
            return len(self._k_spectral[layer_idx])
        return super().get_seq_length(layer_idx)

    def get_max_length(self) -> Optional[int]:
        return None

    @property
    def seen_tokens(self) -> int:
        """Number of tokens processed (for compatibility with HF generate)."""
        if self._k_spectral:
            for layer_tokens in self._k_spectral:
                if layer_tokens:
                    return len(layer_tokens)
        return super().seen_tokens if hasattr(super(), "seen_tokens") else 0

    def compression_ratio(self) -> dict:
        """
        Estimate compression ratio vs f16 baseline.
        Returns stats dict.
        """
        total_tokens = 0
        spectral_bytes = 0
        f16_bytes = 0

        for layer_idx, tokens in enumerate(self._k_spectral):
            if not tokens or layer_idx not in self._attn_layers:
                continue
            cal = self._quantizer._cal[layer_idx]
            n_heads = tokens[0].signal_indices.shape[0]
            head_dim = cal.eigvec_k.shape[1]
            n_tokens = len(tokens)
            total_tokens = max(total_tokens, n_tokens)

            if tokens[0].raw is not None:
                # Fallback: f16 storage
                spectral_bytes += n_tokens * n_heads * head_dim * 2
            else:
                # Signal: 4-bit (int16 stores index) → 0.5 bytes per element
                # Noise: 2-bit (int8 stores index)  → 0.25 bytes per element
                # (index storage is 1 byte each, but represents compressed data)
                d_eff_k = cal.d_eff_k
                d_eff_v = cal.d_eff_v
                # K: 1 byte (signal idx) + 1 byte (noise idx) per head per token
                spectral_bytes += n_tokens * n_heads * 2

            f16_bytes += n_tokens * n_heads * head_dim * 2  # K f16

        for layer_idx, tokens in enumerate(self._v_spectral):
            if not tokens or layer_idx not in self._attn_layers:
                continue
            cal = self._quantizer._cal[layer_idx]
            n_heads = tokens[0].signal_indices.shape[0]
            head_dim = cal.eigvec_v.shape[1]
            n_tokens = len(tokens)
            if tokens[0].raw is None:
                spectral_bytes += n_tokens * n_heads * 2  # V: same as K
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
