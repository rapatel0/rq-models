"""
SpectralQuantizer: online encode/decode for SpectralQuant KV cache.

For each token's K or V vector (per head):
  Encode:
    1. Project onto PCA eigenvectors: x_pca = x @ eigvec
    2. Split: signal = x_pca[:, :d_eff], noise = x_pca[:, d_eff:]
    3. Lloyd-Max nearest-centroid for signal (4-bit) and noise (2-bit)
    4. Return (signal_indices, noise_indices)

  Decode:
    1. Lookup centroids: signal = cb_signal[signal_idx], noise = cb_noise[noise_idx]
    2. Recombine: x_pca_hat = cat(signal, noise)
    3. Inverse project: x_hat = x_pca_hat @ eigvec.T

Edge cases handled:
  - empty cache (first token): returns immediately without projection
  - context shorter than d_eff: falls back to full-dim with zero noise
  - layer without calibration: raises CalibrationNotFoundError
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from turboquant.spectral.calibrator import LayerCalibration
from turboquant.spectral.store import CalibrationNotFoundError


class QuantizedKV(NamedTuple):
    """Compact representation of one token's K or V vectors."""
    signal_indices: Tensor   # [n_kv_heads, 1]  int16 (0-15 for 4-bit)
    noise_indices:  Tensor   # [n_kv_heads, 1]  int8  (0-3  for 2-bit)
    # Store originals for layers using fallback (flat eigenspectrum)
    raw: Tensor | None       # [n_kv_heads, 1, head_dim] float16, or None


class SpectralQuantizer:
    """
    Per-layer encoder/decoder using pre-computed calibration.

    Args:
        calibration: dict[layer_idx → LayerCalibration] from CalibrationStore.load()
        device: torch device (moves calibration tensors to device on first use)
    """

    def __init__(
        self,
        calibration: dict[int, LayerCalibration],
        device: torch.device | None = None,
    ) -> None:
        self._cal = calibration
        self._device = device
        self._moved: set[int] = set()  # track which layers have been moved to device

    def _get_cal(self, layer_idx: int) -> LayerCalibration:
        if layer_idx not in self._cal:
            raise CalibrationNotFoundError(
                f"No calibration for layer {layer_idx}. "
                f"Available layers: {sorted(self._cal.keys())}"
            )
        cal = self._cal[layer_idx]
        # Lazy move to device on first access
        if self._device is not None and layer_idx not in self._moved:
            cal.eigvec_k.data         = cal.eigvec_k.to(self._device)
            cal.eigvec_v.data         = cal.eigvec_v.to(self._device)
            cal.codebook_k_signal.data = cal.codebook_k_signal.to(self._device)
            cal.codebook_k_noise.data  = cal.codebook_k_noise.to(self._device)
            cal.codebook_v_signal.data = cal.codebook_v_signal.to(self._device)
            cal.codebook_v_noise.data  = cal.codebook_v_noise.to(self._device)
            self._moved.add(layer_idx)
        return cal

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode_k(self, k: Tensor, layer_idx: int) -> QuantizedKV:
        """
        Encode K vectors for one new token.

        Args:
            k: [batch=1, n_kv_heads, 1, head_dim] float16/float32

        Returns:
            QuantizedKV with indices for storage
        """
        return self._encode(k, layer_idx, use_k=True)

    def encode_v(self, v: Tensor, layer_idx: int) -> QuantizedKV:
        return self._encode(v, layer_idx, use_k=False)

    def _encode(self, x: Tensor, layer_idx: int, use_k: bool) -> QuantizedKV:
        cal = self._get_cal(layer_idx)
        # x: [batch, n_kv_heads, seq=1, head_dim]
        x = x.squeeze(0).squeeze(-2).float()  # [n_kv_heads, head_dim]

        if use_k:
            eigvec   = cal.eigvec_k                  # [n_kv_heads, head_dim, head_dim]
            d_eff    = cal.d_eff_k
            cb_sig   = cal.codebook_k_signal         # [n_kv_heads, 16, d_eff]
            cb_noi   = cal.codebook_k_noise          # [n_kv_heads, 4,  head_dim-d_eff]
            fallback = cal.fallback_k
        else:
            eigvec   = cal.eigvec_v
            d_eff    = cal.d_eff_v
            cb_sig   = cal.codebook_v_signal
            cb_noi   = cal.codebook_v_noise
            fallback = cal.fallback_v

        eigvec = eigvec.to(x.device)
        cb_sig = cb_sig.to(x.device)
        cb_noi = cb_noi.to(x.device)

        head_dim = x.shape[-1]
        n_kv_heads = x.shape[0]

        # Fallback: flat eigenspectrum — store raw f16 without projection
        if fallback or d_eff >= head_dim:
            return QuantizedKV(
                signal_indices=torch.zeros(n_kv_heads, 1, dtype=torch.int16, device=x.device),
                noise_indices=torch.zeros(n_kv_heads, 1, dtype=torch.int8, device=x.device),
                raw=x.half().unsqueeze(1),
            )

        # Project: x_pca[h, :] = x[h, :] @ eigvec[h]  (columns are eigenvectors)
        # einsum: [H, D], [H, D, D] -> [H, D]
        x_pca = torch.einsum("hd,hde->he", x, eigvec)  # [n_kv_heads, head_dim]

        signal = x_pca[:, :d_eff]              # [H, d_eff]
        noise  = x_pca[:, d_eff:]              # [H, head_dim - d_eff]

        sig_idx = _nearest_centroid(signal, cb_sig)  # [H]
        noi_idx = _nearest_centroid(noise,  cb_noi)  # [H]

        return QuantizedKV(
            signal_indices=sig_idx.to(torch.int16).unsqueeze(1),
            noise_indices=noi_idx.to(torch.int8).unsqueeze(1),
            raw=None,
        )

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode_k(self, stored: list[QuantizedKV], layer_idx: int) -> Tensor:
        """
        Decode all stored K tokens for a layer.

        Args:
            stored: list of QuantizedKV (one per cached token)

        Returns:
            [1, n_kv_heads, seq_len, head_dim] float16
        """
        return self._decode(stored, layer_idx, use_k=True)

    def decode_v(self, stored: list[QuantizedKV], layer_idx: int) -> Tensor:
        return self._decode(stored, layer_idx, use_k=False)

    def _decode(self, stored: list[QuantizedKV], layer_idx: int, use_k: bool) -> Tensor:
        if not stored:
            return torch.empty(1, 0, 0, 0)

        cal = self._get_cal(layer_idx)

        if use_k:
            eigvec = cal.eigvec_k
            d_eff  = cal.d_eff_k
            cb_sig = cal.codebook_k_signal
            cb_noi = cal.codebook_k_noise
        else:
            eigvec = cal.eigvec_v
            d_eff  = cal.d_eff_v
            cb_sig = cal.codebook_v_signal
            cb_noi = cal.codebook_v_noise

        device = stored[0].signal_indices.device
        eigvec = eigvec.to(device)
        cb_sig = cb_sig.to(device)
        cb_noi = cb_noi.to(device)

        head_dim = eigvec.shape[1]
        n_kv_heads = stored[0].signal_indices.shape[0]
        seq_len = len(stored)

        # Handle fallback tokens (raw storage)
        if stored[0].raw is not None:
            # All tokens for this layer are stored raw (fallback path)
            raw = torch.cat([s.raw for s in stored], dim=1)  # [H, T, head_dim]
            return raw.float().unsqueeze(0)                   # [1, H, T, head_dim]

        # Gather all indices: [H, T]
        sig_indices = torch.cat([s.signal_indices for s in stored], dim=1)  # [H, T]
        noi_indices = torch.cat([s.noise_indices  for s in stored], dim=1)  # [H, T]

        # Lookup centroids
        # cb_sig: [H, 16, d_eff] → gather [H, T, d_eff]
        sig = cb_sig[
            torch.arange(n_kv_heads, device=device).unsqueeze(1),
            sig_indices.long(),
        ]  # [H, T, d_eff]

        noi = cb_noi[
            torch.arange(n_kv_heads, device=device).unsqueeze(1),
            noi_indices.long(),
        ]  # [H, T, head_dim - d_eff]

        # Recombine in PCA basis
        x_pca = torch.cat([sig, noi], dim=-1)  # [H, T, head_dim]

        # Inverse project: x_hat[h, t, :] = x_pca[h, t, :] @ eigvec[h].T
        x_hat = torch.einsum("hte,hde->htd", x_pca, eigvec)  # [H, T, head_dim]

        return x_hat.half().unsqueeze(0)  # [1, H, T, head_dim]


def _nearest_centroid(x: Tensor, codebook: Tensor) -> Tensor:
    """
    Find nearest centroid for each row of x.

    Args:
        x:        [H, d]    — query vectors per head
        codebook: [H, K, d] — K centroids per head

    Returns:
        indices: [H] int64 — index of nearest centroid per head
    """
    # Squared distances: ||x - c||^2 = ||x||^2 - 2*x@c.T + ||c||^2
    # x: [H, d], codebook: [H, K, d]
    # Using broadcasting: expand x to [H, 1, d]
    x_exp = x.unsqueeze(1)                                      # [H, 1, d]
    diffs = x_exp - codebook                                    # [H, K, d]
    dists = (diffs * diffs).sum(-1)                             # [H, K]
    return dists.argmin(-1)                                     # [H]
