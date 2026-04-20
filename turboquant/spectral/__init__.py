"""
turboquant.spectral — SpectralQuant KV cache compression.

Public API:
    SpectralKVCache     - DynamicCache subclass, drop-in for model.generate()
    SpectralCalibrator  - offline PCA calibration + Lloyd-Max codebook fitting
    CalibrationStore    - safetensors sidecar serialization
    load_calibration    - convenience loader with path sanitization

Example:
    from turboquant.spectral import SpectralKVCache, load_calibration

    cal = load_calibration("calibration/calibration-qwen3.5-9b.safetensors")
    cache = SpectralKVCache(cal)
    outputs = model.generate(input_ids, past_key_values=cache, max_new_tokens=200)
"""

from turboquant.spectral.calibrator import SpectralCalibrator, LayerCalibration
from turboquant.spectral.store import CalibrationStore, CalibrationNotFoundError
from turboquant.spectral.quantizer import SpectralQuantizer
from turboquant.spectral.kv_cache import SpectralKVCache


def load_calibration(path: str, device=None) -> dict:
    """
    Load calibration sidecar and optionally move tensors to device.

    Args:
        path: path to .safetensors calibration file
        device: torch device (optional; quantizer moves tensors lazily on first use)

    Returns:
        dict[layer_idx → LayerCalibration]

    Raises:
        CalibrationNotFoundError: if path does not exist
    """
    return CalibrationStore.load(path)


__all__ = [
    "SpectralKVCache",
    "SpectralCalibrator",
    "CalibrationStore",
    "CalibrationNotFoundError",
    "SpectralQuantizer",
    "LayerCalibration",
    "load_calibration",
]
