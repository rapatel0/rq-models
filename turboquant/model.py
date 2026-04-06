"""
Qwen3.5-27B model patching: inject TurboKVCache into HuggingFace generate().

Usage:
    from turboquant import patch_model, PRESET_3_5BIT
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-27B", ...)
    patch_model(model, PRESET_3_5BIT)
    outputs = model.generate(input_ids, max_new_tokens=200)
"""

from __future__ import annotations

from functools import wraps
from typing import Optional

from turboquant.config import BitConfig
from turboquant.kv_cache import TurboKVCache


def patch_model(model, config: BitConfig, seed: int = 42) -> None:
    """
    Wrap model.generate() to automatically inject TurboKVCache.

    This is non-destructive: the original generate() is preserved and
    the patch can be removed by calling unpatch_model(model).

    Args:
        model:  A HuggingFace CausalLM (e.g. Qwen3.5-27B loaded with AutoModel).
        config: BitConfig preset (e.g. PRESET_2_5BIT or PRESET_3_5BIT).
        seed:   RNG seed for TurboKVCache (Π and S matrices).
    """
    if hasattr(model, "_turboquant_original_generate"):
        raise RuntimeError(
            "Model is already patched. Call unpatch_model() first."
        )

    original_generate = model.generate

    @wraps(original_generate)
    def patched_generate(*args, **kwargs):
        # Only inject if the caller hasn't provided their own past_key_values
        if kwargs.get("past_key_values") is None and "past_key_values" not in kwargs:
            kwargs["past_key_values"] = TurboKVCache(config, seed=seed)
        elif "past_key_values" not in kwargs:
            pass  # positional arg — don't override
        return original_generate(*args, **kwargs)

    model.generate = patched_generate
    model._turboquant_original_generate = original_generate
    model._turboquant_config = config


def unpatch_model(model) -> None:
    """Remove TurboQuant patch, restoring the original generate()."""
    if not hasattr(model, "_turboquant_original_generate"):
        return
    model.generate = model._turboquant_original_generate
    del model._turboquant_original_generate
    del model._turboquant_config


def get_config(model) -> Optional[BitConfig]:
    """Return the BitConfig currently applied to a patched model, or None."""
    return getattr(model, "_turboquant_config", None)
