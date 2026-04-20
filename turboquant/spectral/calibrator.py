"""
SpectralCalibrator: PCA-based calibration for SpectralQuant KV cache.

For each attention layer, computes:
  - PCA eigenvectors for K and V separately (at f32 for numerical stability)
  - d_eff_k, d_eff_v: number of signal dimensions explaining >= variance_threshold
  - Eigenvalue gap fallback: if no clear gap, d_eff = head_dim (full-dim quantization)
  - Lloyd-Max codebooks: 4-bit (16 centroids) for signal dims, 2-bit (4 centroids) for noise

Usage:
    from turboquant.spectral.calibrator import SpectralCalibrator
    cal = SpectralCalibrator(model, variance_threshold=0.99)
    data = cal.fit(calibration_prompts, tokenizer)
    # data is a CalibrationData dict ready for CalibrationStore.save()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Number of Lloyd-Max centroids
SIGNAL_CENTROIDS = 16   # 4-bit
NOISE_CENTROIDS  = 4    # 2-bit

# Minimum tokens required to fit PCA; fall back to full-dim if fewer
MIN_TOKENS_FOR_PCA = 64


@dataclass
class LayerCalibration:
    """Calibration artifacts for one attention layer."""
    layer_idx: int
    d_eff_k: int                  # signal dims for K (e.g. 4-5)
    d_eff_v: int                  # signal dims for V (e.g. 40-55)

    eigvec_k: Tensor              # [n_kv_heads, head_dim, head_dim] f32, columns=eigenvectors
    eigvec_v: Tensor              # [n_kv_heads, head_dim, head_dim] f32

    codebook_k_signal: Tensor     # [n_kv_heads, SIGNAL_CENTROIDS, d_eff_k] f32
    codebook_k_noise:  Tensor     # [n_kv_heads, NOISE_CENTROIDS,  head_dim - d_eff_k] f32
    codebook_v_signal: Tensor     # [n_kv_heads, SIGNAL_CENTROIDS, d_eff_v] f32
    codebook_v_noise:  Tensor     # [n_kv_heads, NOISE_CENTROIDS,  head_dim - d_eff_v] f32

    # Eigenvalues (stored for inspection / d_eff sweep analysis)
    eigenvalues_k: Tensor         # [n_kv_heads, head_dim] f32, descending
    eigenvalues_v: Tensor         # [n_kv_heads, head_dim] f32, descending

    # Whether d_eff fallback was used (no clear eigenvalue gap)
    fallback_k: bool = False
    fallback_v: bool = False


def _lloyd_max_fit(data: Tensor, n_centroids: int, n_iter: int = 100) -> Tensor:
    """
    Fit Lloyd-Max codebook via k-means.

    Args:
        data: [N, d] float32 — training vectors
        n_centroids: number of centroids (4 or 16)
        n_iter: k-means iterations

    Returns:
        centroids: [n_centroids, d] float32
    """
    from scipy.cluster.vq import kmeans
    import numpy as np

    data_np = data.float().cpu().numpy()
    if len(data_np) < n_centroids:
        # Not enough data — use uniform quantile initialization
        quantiles = np.linspace(0, 100, n_centroids)
        centroids = np.percentile(data_np, quantiles, axis=0)
        return torch.from_numpy(centroids.astype(np.float32))

    # k-means with uniform quantile init
    init_idx = np.linspace(0, len(data_np) - 1, n_centroids, dtype=int)
    init = data_np[init_idx]
    try:
        centroids, _ = kmeans(data_np, init, iter=n_iter, check_finite=False)
    except Exception:
        # Fall back to quantile init if k-means fails
        quantiles = np.linspace(0, 100, n_centroids)
        centroids = np.percentile(data_np, quantiles, axis=0)

    return torch.from_numpy(centroids.astype(np.float32))


def _select_d_eff(eigenvalues: Tensor, variance_threshold: float) -> tuple[int, bool]:
    """
    Select d_eff as the minimum number of top eigenvectors explaining
    >= variance_threshold of total variance.

    Returns (d_eff, fallback_used).
    fallback_used=True if eigenspectrum is flat (no clear gap) → d_eff = head_dim.
    """
    total = eigenvalues.sum()
    if total <= 0:
        return eigenvalues.shape[-1], True

    cumvar = eigenvalues.cumsum(-1) / total
    # Find first index where cumulative variance >= threshold
    exceeded = (cumvar >= variance_threshold).nonzero(as_tuple=True)[-1]
    if len(exceeded) == 0:
        return eigenvalues.shape[-1], True

    d_eff = int(exceeded[0].item()) + 1  # +1: 0-indexed → count

    # Flat spectrum check: if top eigenvalue explains < 2× the mean, no clear gap
    mean_eigenval = total / eigenvalues.shape[-1]
    if eigenvalues[0] < 2.0 * mean_eigenval:
        logger.debug("Flat eigenspectrum detected — using full-dim quantization")
        return eigenvalues.shape[-1], True

    return d_eff, False


class SpectralCalibrator:
    """
    Runs a calibration pass through the model to compute per-layer PCA bases
    and Lloyd-Max codebooks for K and V separately.

    Args:
        model: HuggingFace model (must expose attention layers with k_proj/v_proj)
        variance_threshold: cumulative variance fraction to determine d_eff (default 0.99)
        signal_centroids: Lloyd-Max centroids for signal subspace (4-bit = 16)
        noise_centroids: Lloyd-Max centroids for noise subspace (2-bit = 4)
    """

    def __init__(
        self,
        model: nn.Module,
        variance_threshold: float = 0.99,
        signal_centroids: int = SIGNAL_CENTROIDS,
        noise_centroids: int = NOISE_CENTROIDS,
    ) -> None:
        self.model = model
        self.variance_threshold = variance_threshold
        self.signal_centroids = signal_centroids
        self.noise_centroids = noise_centroids
        self._hooks: list = []
        self._activations: dict[int, dict[str, list[Tensor]]] = {}

    def _register_hooks(self, attention_layer_indices: list[int]) -> None:
        """Register forward hooks on attention layers to collect KV activations."""
        self._activations = {idx: {"k": [], "v": []} for idx in attention_layer_indices}

        # Walk model layers and hook k_proj / v_proj outputs
        # Works for Qwen3.5 and most HF attention implementations
        layers = self._get_attention_layers(attention_layer_indices)
        for layer_idx, (k_proj, v_proj) in layers.items():
            def make_k_hook(idx):
                def hook(module, inp, out):
                    self._activations[idx]["k"].append(out.detach().float().cpu())
                return hook

            def make_v_hook(idx):
                def hook(module, inp, out):
                    self._activations[idx]["v"].append(out.detach().float().cpu())
                return hook

            self._hooks.append(k_proj.register_forward_hook(make_k_hook(layer_idx)))
            self._hooks.append(v_proj.register_forward_hook(make_v_hook(layer_idx)))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _get_attention_layers(
        self, attention_layer_indices: list[int]
    ) -> dict[int, tuple[nn.Module, nn.Module]]:
        """
        Return {layer_idx: (k_proj, v_proj)} for each attention layer.
        Handles Qwen3.5/Qwen3.6 layer naming conventions.
        """
        result = {}
        # Try standard HF naming: model.layers[i].self_attn.{k,v}_proj
        layers_attr = getattr(self.model, "model", self.model)
        raw_layers = getattr(layers_attr, "layers", None)
        if raw_layers is None:
            raise RuntimeError("Cannot find model.layers — unsupported architecture")

        for idx in attention_layer_indices:
            layer = raw_layers[idx]
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            k_proj = getattr(attn, "k_proj", None)
            v_proj = getattr(attn, "v_proj", None)
            if k_proj is not None and v_proj is not None:
                result[idx] = (k_proj, v_proj)

        return result

    def _find_attention_layers(self) -> list[int]:
        """
        Detect which layers are attention layers (have k_proj/v_proj).
        SSM/recurrent layers lack these projections and are skipped.
        """
        layers_attr = getattr(self.model, "model", self.model)
        raw_layers = getattr(layers_attr, "layers", [])
        attn_indices = []
        for i, layer in enumerate(raw_layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None and hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
                attn_indices.append(i)
        logger.info(f"Found {len(attn_indices)} attention layers: {attn_indices}")
        return attn_indices

    def _collect_activations(
        self,
        prompts: list[str],
        tokenizer,
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        """Run calibration prompts through model to populate self._activations."""
        if device is None:
            device = next(self.model.parameters()).device

        self.model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                try:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    ).to(device)
                    self.model(**inputs, use_cache=False)
                    if (i + 1) % 8 == 0:
                        logger.info(f"  Calibration: {i+1}/{len(prompts)} prompts processed")
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM on prompt {i}, skipping")
                    torch.cuda.empty_cache()

    def _pca_for_layer(
        self, activations: list[Tensor], n_kv_heads: int, head_dim: int
    ) -> tuple[Tensor, Tensor]:
        """
        Compute PCA (eigenvectors + eigenvalues) from collected activation tensors.

        Args:
            activations: list of [batch, n_kv_heads * head_dim, seq_len] or
                         [batch, seq_len, n_kv_heads * head_dim] tensors
                         (output of k_proj / v_proj)

        Returns:
            eigvecs: [n_kv_heads, head_dim, head_dim] — columns are eigenvectors, f32
            eigenvalues: [n_kv_heads, head_dim] — descending, f32
        """
        # Stack activations: collect [N_total_tokens, n_kv_heads, head_dim]
        token_vecs_per_head = [[] for _ in range(n_kv_heads)]

        for act in activations:
            # act shape: [batch, seq_len, n_kv_heads * head_dim] (most HF models)
            if act.dim() == 3 and act.shape[-1] == n_kv_heads * head_dim:
                act = act.reshape(-1, n_kv_heads, head_dim)  # [B*T, H, D]
            elif act.dim() == 3 and act.shape[1] == n_kv_heads * head_dim:
                act = act.permute(0, 2, 1).reshape(-1, n_kv_heads, head_dim)
            elif act.dim() == 4:
                # [batch, n_kv_heads, seq_len, head_dim]
                act = act.permute(0, 2, 1, 3).reshape(-1, n_kv_heads, head_dim)
            else:
                logger.warning(f"Unexpected activation shape {act.shape}, skipping")
                continue

            for h in range(n_kv_heads):
                token_vecs_per_head[h].append(act[:, h, :])

        eigvecs_list = []
        eigenvalues_list = []

        for h in range(n_kv_heads):
            vecs = torch.cat(token_vecs_per_head[h], dim=0).float()  # [N_tokens, head_dim]
            N = vecs.shape[0]

            if N < MIN_TOKENS_FOR_PCA:
                logger.warning(f"Head {h}: only {N} tokens, using identity")
                eigvecs_list.append(torch.eye(head_dim))
                eigenvalues_list.append(torch.ones(head_dim))
                continue

            # Zero-mean
            mean = vecs.mean(0, keepdim=True)
            vecs = vecs - mean

            # Covariance and eigendecomposition (f32 for stability)
            # torch.linalg.eigh returns eigenvalues in ascending order
            cov = (vecs.T @ vecs) / (N - 1)
            eigenvalues, eigvecs = torch.linalg.eigh(cov)

            # Flip to descending order
            eigenvalues = eigenvalues.flip(-1).clamp(min=0)
            eigvecs = eigvecs.flip(-1)  # [head_dim, head_dim], columns=eigenvectors

            eigvecs_list.append(eigvecs)
            eigenvalues_list.append(eigenvalues)

        return (
            torch.stack(eigvecs_list),      # [n_kv_heads, head_dim, head_dim]
            torch.stack(eigenvalues_list),  # [n_kv_heads, head_dim]
        )

    def _fit_codebooks(
        self,
        activations: list[Tensor],
        eigvecs: Tensor,
        d_eff: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> tuple[Tensor, Tensor]:
        """
        Project activations onto eigenvectors and fit Lloyd-Max codebooks for
        signal (top d_eff dims) and noise (remaining dims).

        Returns:
            codebook_signal: [n_kv_heads, SIGNAL_CENTROIDS, d_eff]
            codebook_noise:  [n_kv_heads, NOISE_CENTROIDS,  head_dim - d_eff]
        """
        projected_signal = [[] for _ in range(n_kv_heads)]
        projected_noise  = [[] for _ in range(n_kv_heads)]

        for act in activations:
            if act.dim() == 3 and act.shape[-1] == n_kv_heads * head_dim:
                act = act.reshape(-1, n_kv_heads, head_dim).float()
            elif act.dim() == 4:
                act = act.permute(0, 2, 1, 3).reshape(-1, n_kv_heads, head_dim).float()
            else:
                continue

            for h in range(n_kv_heads):
                vecs = act[:, h, :]                         # [N, head_dim]
                V = eigvecs[h]                              # [head_dim, head_dim]
                projected = vecs @ V                        # [N, head_dim] in PCA basis
                projected_signal[h].append(projected[:, :d_eff])
                projected_noise[h].append(projected[:, d_eff:])

        cb_signal_list = []
        cb_noise_list  = []

        for h in range(n_kv_heads):
            sig = torch.cat(projected_signal[h], dim=0)   # [N, d_eff]
            noi = torch.cat(projected_noise[h], dim=0)    # [N, head_dim - d_eff]

            cb_signal_list.append(_lloyd_max_fit(sig, self.signal_centroids))
            cb_noise_list.append(_lloyd_max_fit(noi, self.noise_centroids))

        return (
            torch.stack(cb_signal_list),  # [n_kv_heads, SIGNAL_CENTROIDS, d_eff]
            torch.stack(cb_noise_list),   # [n_kv_heads, NOISE_CENTROIDS,  head_dim - d_eff]
        )

    def fit(
        self,
        prompts: list[str],
        tokenizer,
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> dict[int, LayerCalibration]:
        """
        Run calibration and return per-layer calibration data.

        Args:
            prompts: list of calibration text strings
            tokenizer: HF tokenizer
            max_length: max token length per prompt
            device: inference device (defaults to model device)

        Returns:
            dict mapping layer_idx → LayerCalibration
        """
        import time

        if device is None:
            device = next(self.model.parameters()).device

        attn_indices = self._find_attention_layers()
        if not attn_indices:
            raise RuntimeError("No attention layers found — is this a transformer model?")

        logger.info(f"Calibrating {len(attn_indices)} attention layers on {len(prompts)} prompts")

        self._register_hooks(attn_indices)
        t0 = time.time()
        try:
            self._collect_activations(prompts, tokenizer, max_length=max_length, device=device)
        finally:
            self._remove_hooks()

        elapsed = time.time() - t0
        logger.info(f"Activation collection: {elapsed:.1f}s")

        # Get model head config
        cfg = getattr(self.model, "config", None)
        n_kv_heads = getattr(cfg, "num_key_value_heads", 1)
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            hidden = getattr(cfg, "hidden_size", None)
            n_heads = getattr(cfg, "num_attention_heads", 1)
            head_dim = hidden // n_heads if hidden else 128

        logger.info(f"n_kv_heads={n_kv_heads}, head_dim={head_dim}")

        result = {}

        for layer_idx in attn_indices:
            acts = self._activations[layer_idx]
            k_acts = acts["k"]
            v_acts = acts["v"]

            if not k_acts:
                logger.warning(f"Layer {layer_idx}: no activations collected, skipping")
                continue

            t1 = time.time()

            # PCA for K and V
            eigvec_k, eigenvalues_k = self._pca_for_layer(k_acts, n_kv_heads, head_dim)
            eigvec_v, eigenvalues_v = self._pca_for_layer(v_acts, n_kv_heads, head_dim)

            # d_eff selection per head — use median across heads
            d_eff_k_per_head = []
            d_eff_v_per_head = []
            fallback_k = False
            fallback_v = False

            for h in range(n_kv_heads):
                dk, fb_k = _select_d_eff(eigenvalues_k[h], self.variance_threshold)
                dv, fb_v = _select_d_eff(eigenvalues_v[h], self.variance_threshold)
                d_eff_k_per_head.append(dk)
                d_eff_v_per_head.append(dv)
                if fb_k: fallback_k = True
                if fb_v: fallback_v = True

            # Use min across heads so all heads can share the same d_eff
            d_eff_k = min(d_eff_k_per_head)
            d_eff_v = min(d_eff_v_per_head)

            logger.info(
                f"Layer {layer_idx}: d_eff_k={d_eff_k} (heads={d_eff_k_per_head}), "
                f"d_eff_v={d_eff_v} (heads={d_eff_v_per_head})"
                + (" [K-fallback]" if fallback_k else "")
                + (" [V-fallback]" if fallback_v else "")
            )

            # Fit codebooks
            cb_k_sig, cb_k_noi = self._fit_codebooks(
                k_acts, eigvec_k, d_eff_k, n_kv_heads, head_dim
            )
            cb_v_sig, cb_v_noi = self._fit_codebooks(
                v_acts, eigvec_v, d_eff_v, n_kv_heads, head_dim
            )

            result[layer_idx] = LayerCalibration(
                layer_idx=layer_idx,
                d_eff_k=d_eff_k,
                d_eff_v=d_eff_v,
                eigvec_k=eigvec_k,
                eigvec_v=eigvec_v,
                codebook_k_signal=cb_k_sig,
                codebook_k_noise=cb_k_noi,
                codebook_v_signal=cb_v_sig,
                codebook_v_noise=cb_v_noi,
                eigenvalues_k=eigenvalues_k,
                eigenvalues_v=eigenvalues_v,
                fallback_k=fallback_k,
                fallback_v=fallback_v,
            )

            logger.debug(f"Layer {layer_idx} calibrated in {time.time() - t1:.1f}s")

        # Free activation memory
        self._activations.clear()

        total = time.time() - t0
        logger.info(f"Calibration complete: {len(result)} layers in {total:.1f}s")
        return result
