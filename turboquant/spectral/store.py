"""
CalibrationStore: serialize/deserialize SpectralQuant calibration data.

Format: safetensors sidecar file (no pickle, built-in integrity validation).
One file per model, stored in calibration/ directory.

Key naming convention:
    layer_{n}_eigvec_k       float32 [n_kv_heads, head_dim, head_dim]
    layer_{n}_eigvec_v       float32 [n_kv_heads, head_dim, head_dim]
    layer_{n}_eigenvalues_k  float32 [n_kv_heads, head_dim]
    layer_{n}_eigenvalues_v  float32 [n_kv_heads, head_dim]
    layer_{n}_cb_k_signal    float32 [n_kv_heads, 16, d_eff_k]
    layer_{n}_cb_k_noise     float32 [n_kv_heads, 4,  head_dim - d_eff_k]
    layer_{n}_cb_v_signal    float32 [n_kv_heads, 16, d_eff_v]
    layer_{n}_cb_v_noise     float32 [n_kv_heads, 4,  head_dim - d_eff_v]
    layer_{n}_d_eff          int32   scalar [d_eff_k, d_eff_v]
    meta_layer_indices       int32   [n_layers]
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file, load_file

from turboquant.spectral.calibrator import LayerCalibration


class CalibrationNotFoundError(FileNotFoundError):
    pass


class CalibrationStore:
    """
    Save and load SpectralQuant calibration sidecars.

    Usage:
        # Save
        CalibrationStore.save(calibration_data, "calibration/calibration-qwen3.5-9b.safetensors")

        # Load
        data = CalibrationStore.load("calibration/calibration-qwen3.5-9b.safetensors")
    """

    @staticmethod
    def save(
        calibration: dict[int, LayerCalibration],
        path: str | Path,
    ) -> None:
        """
        Save calibration data to a safetensors file.

        Validates tensor shapes before saving to catch misconfigured data early.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        tensors: dict[str, torch.Tensor] = {}

        layer_indices = sorted(calibration.keys())
        tensors["meta_layer_indices"] = torch.tensor(layer_indices, dtype=torch.int32)

        for layer_idx, cal in calibration.items():
            n = layer_idx
            tensors[f"layer_{n}_eigvec_k"]      = cal.eigvec_k.float().contiguous()
            tensors[f"layer_{n}_eigvec_v"]      = cal.eigvec_v.float().contiguous()
            tensors[f"layer_{n}_eigenvalues_k"] = cal.eigenvalues_k.float().contiguous()
            tensors[f"layer_{n}_eigenvalues_v"] = cal.eigenvalues_v.float().contiguous()
            tensors[f"layer_{n}_cb_k_signal"]   = cal.codebook_k_signal.float().contiguous()
            tensors[f"layer_{n}_cb_k_noise"]    = cal.codebook_k_noise.float().contiguous()
            tensors[f"layer_{n}_cb_v_signal"]   = cal.codebook_v_signal.float().contiguous()
            tensors[f"layer_{n}_cb_v_noise"]    = cal.codebook_v_noise.float().contiguous()
            tensors[f"layer_{n}_d_eff"]         = torch.tensor(
                [cal.d_eff_k, cal.d_eff_v], dtype=torch.int32
            )

        save_file(tensors, str(path))

    @staticmethod
    def load(path: str | Path) -> dict[int, LayerCalibration]:
        """
        Load calibration from a safetensors file.

        Validates tensor shapes against each other on load.

        Raises:
            CalibrationNotFoundError: if path does not exist
            ValueError: if tensors have unexpected shapes
        """
        path = Path(path)
        if not path.exists():
            raise CalibrationNotFoundError(
                f"Calibration sidecar not found: {path}\n"
                f"Generate it with: python scripts/calibrate_spectral.py --model <name> --output calibration/"
            )

        tensors = load_file(str(path))
        layer_indices = tensors["meta_layer_indices"].tolist()

        result: dict[int, LayerCalibration] = {}

        for layer_idx in layer_indices:
            n = layer_idx
            eigvec_k      = tensors[f"layer_{n}_eigvec_k"]
            eigvec_v      = tensors[f"layer_{n}_eigvec_v"]
            eigenvalues_k = tensors[f"layer_{n}_eigenvalues_k"]
            eigenvalues_v = tensors[f"layer_{n}_eigenvalues_v"]
            cb_k_signal   = tensors[f"layer_{n}_cb_k_signal"]
            cb_k_noise    = tensors[f"layer_{n}_cb_k_noise"]
            cb_v_signal   = tensors[f"layer_{n}_cb_v_signal"]
            cb_v_noise    = tensors[f"layer_{n}_cb_v_noise"]
            d_eff         = tensors[f"layer_{n}_d_eff"].tolist()
            d_eff_k, d_eff_v = d_eff[0], d_eff[1]

            # Shape validation
            n_kv_heads, head_dim, _ = eigvec_k.shape
            if eigvec_k.shape != (n_kv_heads, head_dim, head_dim):
                raise ValueError(f"layer_{n}_eigvec_k: expected [{n_kv_heads},{head_dim},{head_dim}], got {eigvec_k.shape}")
            if cb_k_signal.shape[-1] != d_eff_k:
                raise ValueError(f"layer_{n}_cb_k_signal d_eff mismatch: {cb_k_signal.shape[-1]} != {d_eff_k}")
            if cb_v_signal.shape[-1] != d_eff_v:
                raise ValueError(f"layer_{n}_cb_v_signal d_eff mismatch: {cb_v_signal.shape[-1]} != {d_eff_v}")

            result[layer_idx] = LayerCalibration(
                layer_idx=layer_idx,
                d_eff_k=d_eff_k,
                d_eff_v=d_eff_v,
                eigvec_k=eigvec_k,
                eigvec_v=eigvec_v,
                codebook_k_signal=cb_k_signal,
                codebook_k_noise=cb_k_noise,
                codebook_v_signal=cb_v_signal,
                codebook_v_noise=cb_v_noise,
                eigenvalues_k=eigenvalues_k,
                eigenvalues_v=eigenvalues_v,
            )

        return result

    @staticmethod
    def sidecar_path(model_name: str, calibration_dir: str = "calibration") -> Path:
        """
        Canonical path for a model's calibration sidecar.
        Sanitizes model_name to prevent directory traversal.
        """
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "-", model_name).strip("-")
        return Path(calibration_dir) / f"calibration-{safe_name}.safetensors"
