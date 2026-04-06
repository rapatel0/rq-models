"""
OutlierSplitter: channel partitioning for TurboQuant KV cache.

Splits each head vector into outlier (first-N) and regular (remaining) channels,
applying separate TurboQuant instances to each partition.

K channels use TurboQuantProd (unbiased inner product).
V channels use TurboQuantMSE (MSE reconstruction).
"""

from __future__ import annotations
from typing import NamedTuple, Optional

import torch
from torch import Tensor

from turboquant.config import BitConfig
from turboquant.core import TurboQuantMSE, TurboQuantProd, MSEQuantized, ProdQuantized


# ---------------------------------------------------------------------------
# Quantized storage types
# ---------------------------------------------------------------------------

class KQuantized(NamedTuple):
    """Quantized K cache for one batch of vectors (one token or a sequence)."""
    # Outlier partition (TurboQuantProd)
    out_prod: ProdQuantized   # indices, qjl, gamma — outlier channels
    out_norm: Tensor          # MSE sub-quantizer norm for outlier channels

    # Regular partition (TurboQuantProd)
    reg_prod: ProdQuantized   # indices, qjl, gamma — regular channels
    reg_norm: Tensor          # MSE sub-quantizer norm for regular channels


class VQuantized(NamedTuple):
    """Quantized V cache for one batch of vectors."""
    # Outlier partition (TurboQuantMSE)
    out_mse: MSEQuantized     # indices, norm — outlier channels

    # Regular partition (TurboQuantMSE)
    reg_mse: MSEQuantized     # indices, norm — regular channels


# ---------------------------------------------------------------------------
# OutlierSplitter
# ---------------------------------------------------------------------------

class OutlierSplitter:
    """
    Partitions head vectors into outlier + regular channels and applies
    separate TurboQuant instances to each partition.

    K path: TurboQuantProd (unbiased inner product for Q @ Kᵀ attention)
    V path: TurboQuantMSE  (MSE reconstruction for softmax @ V)

    Outlier selection: fixed first-N channels (data-oblivious).
    The rotation matrix Π randomizes the representation, so "first N
    post-rotation channels" are statistically equivalent to any other N.

    Rotation matrices (Π, S) are shared across all layers — valid because
    the algorithm is data-oblivious and saves 64× memory.
    """

    def __init__(self, config: BitConfig, seed: int = 42) -> None:
        self.config = config
        d_out = config.outlier_count
        d_reg = config.regular_count

        # K path: TurboQuantProd for each partition (skip if partition is empty)
        self.k_out_quant = TurboQuantProd(d_out, config.outlier_k_bits, seed=seed) if d_out > 0 else None
        self.k_reg_quant = TurboQuantProd(d_reg, config.regular_k_bits, seed=seed + 1) if d_reg > 0 else None

        # V path: TurboQuantMSE for each partition
        self.v_out_quant = TurboQuantMSE(d_out, config.outlier_v_bits, seed=seed + 2) if d_out > 0 else None
        self.v_reg_quant = TurboQuantMSE(d_reg, config.regular_v_bits, seed=seed + 3) if d_reg > 0 else None

    # ------------------------------------------------------------------
    # K path (TurboQuantProd)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def quantize_k(self, x: Tensor) -> KQuantized:
        """
        Quantize K vectors using TurboQuantProd.

        Args:
            x: [..., head_dim] tensor (any dtype, will be cast to float32)

        Returns:
            KQuantized namedtuple with all fields needed for dequantization.
        """
        out_slice = x[..., : self.config.outlier_count]
        reg_slice = x[..., self.config.outlier_count :]

        if self.k_out_quant is not None:
            out_prod, out_norm = self.k_out_quant.quantize_and_store(out_slice)
        else:
            out_prod, out_norm = None, None

        reg_prod, reg_norm = self.k_reg_quant.quantize_and_store(reg_slice)

        return KQuantized(
            out_prod=out_prod, out_norm=out_norm,
            reg_prod=reg_prod, reg_norm=reg_norm,
        )

    @torch.no_grad()
    def dequantize_k(self, qk: KQuantized, out_dtype: torch.dtype = torch.float32) -> Tensor:
        """
        Reconstruct K vectors from KQuantized representation.

        Returns: [..., head_dim] tensor in out_dtype.
        """
        parts = []
        if self.k_out_quant is not None:
            parts.append(self.k_out_quant.dequantize(qk.out_prod, qk.out_norm, out_dtype=out_dtype))
        parts.append(self.k_reg_quant.dequantize(qk.reg_prod, qk.reg_norm, out_dtype=out_dtype))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    # ------------------------------------------------------------------
    # V path (TurboQuantMSE)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def quantize_v(self, x: Tensor) -> VQuantized:
        """
        Quantize V vectors using TurboQuantMSE.

        Args:
            x: [..., head_dim] tensor.

        Returns:
            VQuantized namedtuple.
        """
        out_slice = x[..., : self.config.outlier_count]
        reg_slice = x[..., self.config.outlier_count :]

        out_mse = self.v_out_quant.quantize(out_slice) if self.v_out_quant is not None else None
        reg_mse = self.v_reg_quant.quantize(reg_slice)

        return VQuantized(out_mse=out_mse, reg_mse=reg_mse)

    @torch.no_grad()
    def dequantize_v(self, qv: VQuantized, out_dtype: torch.dtype = torch.float32) -> Tensor:
        """
        Reconstruct V vectors from VQuantized representation.

        Returns: [..., head_dim] tensor in out_dtype.
        """
        parts = []
        if self.v_out_quant is not None:
            parts.append(self.v_out_quant.dequantize(qv.out_mse, out_dtype=out_dtype))
        parts.append(self.v_reg_quant.dequantize(qv.reg_mse, out_dtype=out_dtype))
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    # ------------------------------------------------------------------
    # Convenience: quantize both K and V together
    # ------------------------------------------------------------------

    @torch.no_grad()
    def quantize_kv(
        self, key: Tensor, value: Tensor
    ) -> tuple[KQuantized, VQuantized]:
        """Quantize K and V vectors in one call."""
        return self.quantize_k(key), self.quantize_v(value)

    @torch.no_grad()
    def dequantize_kv(
        self,
        qk: KQuantized,
        qv: VQuantized,
        out_dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor]:
        """Reconstruct K and V from their quantized representations."""
        return self.dequantize_k(qk, out_dtype), self.dequantize_v(qv, out_dtype)
