"""
Tests for turboquant/kv_cache.py and turboquant/model.py.

Uses a lightweight mock model (2 layers, 2 KV heads, head_dim=128) to test
the TurboKVCache interface without requiring the full Qwen3.5-27B weights.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch as mock_patch

from turboquant.config import PRESET_2_5BIT, PRESET_3_5BIT
from turboquant.kv_cache import TurboKVCache
from turboquant.model import patch_model, unpatch_model, get_config

torch.manual_seed(0)

# Dimensions matching Qwen3.5-27B KV head layout
BATCH = 2
NUM_KV_HEADS = 8
HEAD_DIM = 128


def make_kv(seq: int = 1, batch: int = BATCH):
    """Generate random K/V tensors for one or more new tokens."""
    k = torch.randn(batch, NUM_KV_HEADS, seq, HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(batch, NUM_KV_HEADS, seq, HEAD_DIM, dtype=torch.bfloat16)
    return k, v


# ---------------------------------------------------------------------------
# TurboKVCache unit tests
# ---------------------------------------------------------------------------

class TestTurboKVCache:
    def test_update_returns_correct_shapes(self):
        """update() must return [batch, kv_heads, seq, head_dim]."""
        cache = TurboKVCache(PRESET_3_5BIT)
        k, v = make_kv(seq=1)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape == (BATCH, NUM_KV_HEADS, 1, HEAD_DIM)
        assert v_out.shape == (BATCH, NUM_KV_HEADS, 1, HEAD_DIM)

    def test_seq_grows_with_tokens(self):
        """Sequence length must grow by 1 for each token inserted."""
        cache = TurboKVCache(PRESET_3_5BIT)
        for step in range(1, 6):
            k, v = make_kv(seq=1)
            k_out, v_out = cache.update(k, v, layer_idx=0)
            assert k_out.shape[-2] == step, f"seq_len={k_out.shape[-2]} != {step} at step {step}"
        assert cache.get_seq_length(0) == 5

    def test_output_dtype_matches_input(self):
        """Dequantized output dtype must match input dtype (bfloat16)."""
        cache = TurboKVCache(PRESET_3_5BIT)
        k, v = make_kv(seq=1)
        assert k.dtype == torch.bfloat16
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.dtype == torch.bfloat16, f"K dtype: {k_out.dtype}"
        assert v_out.dtype == torch.bfloat16, f"V dtype: {v_out.dtype}"

    def test_no_nan_or_inf(self):
        """No NaN or Inf in output after 20 tokens."""
        cache = TurboKVCache(PRESET_3_5BIT)
        for _ in range(20):
            k, v = make_kv()
            k_out, v_out = cache.update(k, v, layer_idx=0)
        assert torch.isfinite(k_out).all(), "NaN/Inf in K output"
        assert torch.isfinite(v_out).all(), "NaN/Inf in V output"

    def test_kv_not_query_heads(self):
        """Returned shapes must be num_kv_heads (8), never num_query_heads (32)."""
        cache = TurboKVCache(PRESET_3_5BIT)
        k, v = make_kv()
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape[1] == NUM_KV_HEADS, (
            f"Got {k_out.shape[1]} heads; expected {NUM_KV_HEADS} (KV heads, not query heads)"
        )

    def test_multi_layer_independent(self):
        """Updates to different layers must not interfere."""
        cache = TurboKVCache(PRESET_3_5BIT)
        for layer in range(4):
            k, v = make_kv()
            cache.update(k, v, layer_idx=layer)
        assert cache.get_seq_length(0) == 1
        assert cache.get_seq_length(3) == 1
        assert len(cache) == 4

    def test_key_cache_property(self):
        """key_cache property must return list of dequantized tensors."""
        cache = TurboKVCache(PRESET_2_5BIT)
        k, v = make_kv()
        cache.update(k, v, layer_idx=0)
        kc = cache.key_cache
        assert isinstance(kc, list)
        assert kc[0].shape == (BATCH, NUM_KV_HEADS, 1, HEAD_DIM)

    def test_reconstruction_reasonable_mse(self):
        """K/V reconstruction MSE should be within expected distortion bounds.

        V uses TurboQuantMSE. For randn(d=128) vectors, E[‖x‖²]≈128.
        With 3.5-bit V: outlier(64ch@4b) ≈ 0.58 + regular(64ch@3b) ≈ 2.18 → ~2.8.
        Allow up to 4.0 to account for variance.
        """
        cache = TurboKVCache(PRESET_3_5BIT)
        k_orig = torch.randn(1, 1, 1, HEAD_DIM)
        v_orig = torch.randn(1, 1, 1, HEAD_DIM)
        k_out, v_out = cache.update(k_orig, v_orig, layer_idx=0)

        v_mse = ((v_orig.float() - v_out.float()) ** 2).sum(-1).mean().item()
        assert v_mse < 4.0, f"V reconstruction MSE too high: {v_mse:.4f}"

    def test_memory_stats(self):
        """memory_stats() must return positive values after populating cache."""
        cache = TurboKVCache(PRESET_3_5BIT)
        for layer in range(3):
            k, v = make_kv()
            cache.update(k, v, layer_idx=layer)
        stats = cache.memory_stats()
        assert stats["total_mb"] > 0
        assert stats["k_bytes"] > 0
        assert stats["v_bytes"] > 0

    @pytest.mark.parametrize("config", [PRESET_2_5BIT, PRESET_3_5BIT])
    def test_both_configs(self, config):
        """Both 2.5-bit and 3.5-bit configs must produce valid outputs."""
        cache = TurboKVCache(config)
        for _ in range(5):
            k, v = make_kv()
            k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape == (BATCH, NUM_KV_HEADS, 5, HEAD_DIM)
        assert torch.isfinite(k_out).all()

    def test_batch_size_4(self):
        """Cache must work correctly with batch_size=4."""
        cache = TurboKVCache(PRESET_3_5BIT)
        k, v = make_kv(batch=4)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape[0] == 4

    def test_repr(self):
        cache = TurboKVCache(PRESET_3_5BIT)
        k, v = make_kv()
        cache.update(k, v, layer_idx=0)
        r = repr(cache)
        assert "TurboKVCache" in r
        assert "3.5-bit" in r


# ---------------------------------------------------------------------------
# model.py tests (mock model)
# ---------------------------------------------------------------------------

class TestModelPatch:
    def _make_mock_model(self):
        model = MagicMock()
        model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        # Remove any pre-existing patch marker
        if hasattr(model, "_turboquant_original_generate"):
            del model._turboquant_original_generate
        return model

    def test_patch_injects_cache(self):
        """patch_model() must inject TurboKVCache into generate() kwargs."""
        model = self._make_mock_model()
        patch_model(model, PRESET_3_5BIT)

        input_ids = torch.tensor([[1, 2, 3]])
        model.generate(input_ids, max_new_tokens=10)

        call_kwargs = model._turboquant_original_generate.call_args[1]
        assert "past_key_values" in call_kwargs
        assert isinstance(call_kwargs["past_key_values"], TurboKVCache)

    def test_patch_does_not_override_explicit_cache(self):
        """If caller provides past_key_values, it must not be overridden."""
        model = self._make_mock_model()
        patch_model(model, PRESET_3_5BIT)

        my_cache = TurboKVCache(PRESET_2_5BIT)
        input_ids = torch.tensor([[1, 2, 3]])
        model.generate(input_ids, past_key_values=my_cache)

        call_kwargs = model._turboquant_original_generate.call_args[1]
        assert call_kwargs["past_key_values"] is my_cache

    def test_get_config(self):
        model = self._make_mock_model()
        # MagicMock intercepts getattr, so test after patch only
        patch_model(model, PRESET_3_5BIT)
        assert get_config(model) is PRESET_3_5BIT
        unpatch_model(model)
        # After unpatch, _turboquant_config is deleted from the model object
        assert not hasattr(model, "_turboquant_config")

    def test_double_patch_raises(self):
        model = self._make_mock_model()
        patch_model(model, PRESET_3_5BIT)
        with pytest.raises(RuntimeError, match="already patched"):
            patch_model(model, PRESET_2_5BIT)

    def test_unpatch(self):
        model = self._make_mock_model()
        original_gen = model.generate
        patch_model(model, PRESET_3_5BIT)
        assert model.generate is not original_gen
        unpatch_model(model)
        assert get_config(model) is None
        # Double unpatch is safe
        unpatch_model(model)
