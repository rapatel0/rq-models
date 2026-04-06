"""
TurboQuant core algorithms (Algorithms 1 & 2 from arXiv:2504.19874).

TurboQuantMSE  — minimizes mean-squared error (Algorithm 1)
TurboQuantProd — unbiased inner-product estimator (Algorithm 2)

Both operate on batched float tensors of shape [..., d] and perform all
rotation/quantization in float32 for numerical stability, regardless of the
input dtype.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from turboquant.codebook import load_codebook


# ---------------------------------------------------------------------------
# Named output types
# ---------------------------------------------------------------------------

class MSEQuantized(NamedTuple):
    """Output of TurboQuantMSE.quantize()."""
    indices: Tensor   # uint8 [..., d], values in [0, 2^b)
    norm: Tensor      # float32 [...], L2 norm of original vector


class ProdQuantized(NamedTuple):
    """Output of TurboQuantProd.quantize()."""
    indices: Tensor   # uint8 [..., d], MSE indices at bit-width b-1
    qjl: Tensor       # int8  [..., d], sign of S @ residual, values ±1
    gamma: Tensor     # float16 [...], L2 norm of residual


# ---------------------------------------------------------------------------
# TurboQuantMSE — Algorithm 1
# ---------------------------------------------------------------------------

class TurboQuantMSE:
    """
    MSE-optimal vector quantizer via random rotation + Lloyd-Max codebook.

    Setup (data-oblivious, shared across layers):
      Π ∈ R^(d×d) — random orthogonal matrix (QR of Gaussian)
      codebook ∈ R^(2^b) — Lloyd-Max centroids for Beta distribution

    QUANT(x):
      1. Normalize: x_unit = x / ‖x‖₂
      2. Rotate:    y = Π @ x_unit  (each coord ~ Beta → near-Gaussian)
      3. Lookup:    idx_j = argmin_k |y_j - codebook[k]|

    DEQUANT(idx, norm):
      1. Lookup:    ỹ_j = codebook[idx_j]
      2. Unrotate:  x̃_unit = Πᵀ @ ỹ
      3. Rescale:   x̃ = x̃_unit * norm
    """

    def __init__(self, d: int, b: int, seed: int = 42) -> None:
        """
        Args:
            d:    Vector dimension (partition size).
            b:    Bit-width per coordinate (1–5).
            seed: RNG seed for reproducibility.
        """
        if b < 1 or b > 5:
            raise ValueError(f"bit-width b must be in [1, 5], got {b}")

        self.d = d
        self.b = b

        # Generate random orthogonal matrix in float32
        gen = torch.Generator()
        gen.manual_seed(seed)
        raw = torch.randn(d, d, generator=gen, dtype=torch.float32)
        self.Pi: Tensor = torch.linalg.qr(raw)[0]  # [d, d], float32

        # Load precomputed codebook
        self.codebook: Tensor = load_codebook(d, b).float()  # [2^b], float32

    @torch.no_grad()
    def quantize(self, x: Tensor) -> MSEQuantized:
        """
        Quantize vectors to b-bit indices.

        Args:
            x: Input tensor of shape [..., d], any dtype.

        Returns:
            MSEQuantized(indices=[..., d] uint8, norm=[...] float32)
        """
        orig_dtype = x.dtype
        device = x.device
        x = x.float()  # upcast to float32

        # Move rotation matrix to device if needed
        Pi = self.Pi.to(device)
        cb = self.codebook.to(device)

        # Normalize to unit sphere
        norm = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=1e-12)
        norm_scalar = norm.squeeze(-1)  # [...]
        x_unit = x / norm              # [..., d]

        # Rotate: y = x_unit @ Πᵀ  (equivalent to Π @ x_unit per-vector)
        y = x_unit @ Pi.T             # [..., d]

        # Nearest-centroid lookup via broadcast
        # y: [..., d, 1], cb: [2^b]  -> dist: [..., d, 2^b]
        dist = (y.unsqueeze(-1) - cb).abs()
        indices = dist.argmin(dim=-1).to(torch.uint8)  # [..., d]

        return MSEQuantized(indices=indices, norm=norm_scalar.float())

    @torch.no_grad()
    def dequantize(self, quantized: MSEQuantized, out_dtype: torch.dtype = torch.float32) -> Tensor:
        """
        Reconstruct vectors from quantized representation.

        Args:
            quantized: MSEQuantized namedtuple from quantize().
            out_dtype: Output dtype (default float32; use bfloat16 for model integration).

        Returns:
            Reconstructed tensor [..., d] in out_dtype.
        """
        device = quantized.indices.device
        Pi = self.Pi.to(device)
        cb = self.codebook.to(device)

        # Codebook lookup
        y_hat = cb[quantized.indices.long()]  # [..., d], float32

        # Inverse rotation: x̃_unit = y_hat @ Π
        x_unit = y_hat @ Pi                   # [..., d]

        # Rescale to original magnitude
        x_hat = x_unit * quantized.norm.unsqueeze(-1)

        return x_hat.to(out_dtype)

    def orthogonality_error(self) -> float:
        """‖Πᵀ·Π - I‖_F — should be < 1e-5 for a valid rotation matrix."""
        eye = torch.eye(self.d, dtype=torch.float32)
        return (self.Pi.T @ self.Pi - eye).norm(p="fro").item()


# ---------------------------------------------------------------------------
# TurboQuantProd — Algorithm 2
# ---------------------------------------------------------------------------

class TurboQuantProd:
    """
    Unbiased inner-product quantizer via TurboQuantMSE + 1-bit QJL residual.

    Two-stage process:
      1. Apply TurboQuantMSE at bit-width (b-1) to get MSE-optimal approximation.
      2. Compute residual r = x - x̃_mse.
      3. Apply QJL: qjl = sign(S @ r), γ = ‖r‖₂.

    Dequantization:
      x̃ = x̃_mse + (√(π/2) / d) · γ · Sᵀ · qjl

    This produces an unbiased inner product estimator:
      E[⟨y, x̃⟩] = ⟨y, x⟩  for any query vector y.

    Note: S entries are drawn from N(0,1), NOT N(0,1/d). The 1/d scaling
    is applied in the dequantization formula explicitly.
    """

    _SQRT_PI_OVER_2 = math.sqrt(math.pi / 2)

    def __init__(self, d: int, b: int, seed: int = 42) -> None:
        """
        Args:
            d:    Vector dimension.
            b:    Target bit-width (total: (b-1) bits MSE + 1 bit QJL).
            seed: RNG seed. MSE sub-quantizer uses seed, S uses seed+1.
        """
        if b < 2:
            raise ValueError(f"TurboQuantProd requires b >= 2, got {b}")

        self.d = d
        self.b = b
        self._scale = self._SQRT_PI_OVER_2 / d

        # MSE sub-quantizer at bit-width b-1
        self.mse_quant = TurboQuantMSE(d, b - 1, seed=seed)

        # QJL projection matrix: S_{i,j} ~ N(0, 1)
        gen = torch.Generator()
        gen.manual_seed(seed + 1)
        self.S: Tensor = torch.randn(d, d, generator=gen, dtype=torch.float32)

    @torch.no_grad()
    def quantize(self, x: Tensor) -> ProdQuantized:
        """
        Quantize vectors for unbiased inner product estimation.

        Args:
            x: Input tensor [..., d], any dtype.

        Returns:
            ProdQuantized(indices, qjl, gamma)
        """
        device = x.device
        x_f32 = x.float()

        # Stage 1: MSE quantization at (b-1) bits
        mse_q = self.mse_quant.quantize(x_f32)
        x_hat_mse = self.mse_quant.dequantize(mse_q, out_dtype=torch.float32)

        # Residual (in original scale, not unit-normalized)
        residual = x_f32 - x_hat_mse  # [..., d]

        # Stage 2: QJL on residual
        S = self.S.to(device)
        # proj: [..., d] = residual @ Sᵀ  (sign applied entry-wise)
        proj = residual @ S.T          # [..., d]
        qjl = proj.sign().to(torch.int8)  # [..., d], values ±1 (0 → +1 for stability)
        qjl = torch.where(qjl == 0, torch.ones_like(qjl), qjl)

        # Residual norm
        gamma = torch.linalg.norm(residual, dim=-1).to(torch.float16)  # [...]

        return ProdQuantized(indices=mse_q.indices, qjl=qjl, gamma=gamma)

    @torch.no_grad()
    def dequantize(
        self,
        quantized: ProdQuantized,
        mse_norm: Tensor,
        out_dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """
        Reconstruct vectors from ProdQuantized representation.

        Args:
            quantized: ProdQuantized from quantize().
            mse_norm:  The MSEQuantized.norm from the MSE stage (stored separately).
            out_dtype: Output dtype.

        Returns:
            Reconstructed tensor [..., d] in out_dtype.
        """
        device = quantized.indices.device
        S = self.S.to(device)

        # Reconstruct MSE part
        mse_q = MSEQuantized(indices=quantized.indices, norm=mse_norm)
        x_hat_mse = self.mse_quant.dequantize(mse_q, out_dtype=torch.float32)

        # QJL correction: (√(π/2) / d) · γ · Sᵀ · qjl
        # qjl: [..., d] int8, S: [d, d]
        gamma_f32 = quantized.gamma.float()                          # [...]
        qjl_f32 = quantized.qjl.float()                             # [..., d]
        correction = self._scale * gamma_f32.unsqueeze(-1) * (qjl_f32 @ S)  # [..., d]

        return (x_hat_mse + correction).to(out_dtype)

    @torch.no_grad()
    def quantize_and_store(self, x: Tensor):
        """
        Convenience method: quantize and return all fields needed for dequantize.
        Returns (ProdQuantized, mse_norm) tuple.
        """
        device = x.device
        x_f32 = x.float()

        mse_q = self.mse_quant.quantize(x_f32)
        x_hat_mse = self.mse_quant.dequantize(mse_q, out_dtype=torch.float32)

        residual = x_f32 - x_hat_mse
        S = self.S.to(device)
        proj = residual @ S.T
        qjl = proj.sign().to(torch.int8)
        qjl = torch.where(qjl == 0, torch.ones_like(qjl), qjl)
        gamma = torch.linalg.norm(residual, dim=-1).to(torch.float16)

        prod_q = ProdQuantized(indices=mse_q.indices, qjl=qjl, gamma=gamma)
        return prod_q, mse_q.norm
