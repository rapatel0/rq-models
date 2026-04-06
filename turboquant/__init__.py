"""
turboquant — TurboQuant KV cache quantization for HuggingFace transformers.

Quick start:
    from turboquant import patch_model, PRESET_3_5BIT
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B", torch_dtype="bfloat16")
    patch_model(model, PRESET_3_5BIT)
    outputs = model.generate(input_ids, max_new_tokens=200)

Reference: TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
           arXiv:2504.19874 (ICLR 2026)
"""

from turboquant.config import BitConfig, PRESET_2_5BIT, PRESET_3_5BIT, PRESET_4BIT
from turboquant.core import TurboQuantMSE, TurboQuantProd, MSEQuantized, ProdQuantized
from turboquant.outlier import OutlierSplitter, KQuantized, VQuantized
from turboquant.kv_cache import TurboKVCache
from turboquant.model import patch_model, unpatch_model, get_config

__all__ = [
    "BitConfig", "PRESET_2_5BIT", "PRESET_3_5BIT", "PRESET_4BIT",
    "TurboQuantMSE", "TurboQuantProd", "MSEQuantized", "ProdQuantized",
    "OutlierSplitter", "KQuantized", "VQuantized",
    "TurboKVCache",
    "patch_model", "unpatch_model", "get_config",
]
