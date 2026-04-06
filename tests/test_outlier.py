"""Tests for turboquant/outlier.py and turboquant/config.py."""

import pytest
import torch

from turboquant.config import BitConfig, PRESET_2_5BIT, PRESET_3_5BIT
from turboquant.outlier import OutlierSplitter

torch.manual_seed(0)


def rand_kv(batch: int = 4, heads: int = 8, seq: int = 16, d: int = 128):
    """Generate random KV tensors with non-unit norms (realistic)."""
    k = torch.randn(batch, heads, seq, d) * 0.5
    v = torch.randn(batch, heads, seq, d) * 0.5
    return k, v


class TestBitConfig:
    def test_2_5bit_effective_bits(self):
        assert PRESET_2_5BIT.k_effective_bits == pytest.approx(2.25, abs=0.01)
        assert PRESET_2_5BIT.v_effective_bits == pytest.approx(2.25, abs=0.01)

    def test_3_5bit_effective_bits(self):
        assert PRESET_3_5BIT.k_effective_bits == pytest.approx(3.5, abs=0.01)
        assert PRESET_3_5BIT.v_effective_bits == pytest.approx(3.5, abs=0.01)

    def test_regular_count(self):
        assert PRESET_2_5BIT.regular_count == 96
        assert PRESET_3_5BIT.regular_count == 64

    def test_str(self):
        s = str(PRESET_2_5BIT)
        assert "2.5-bit" in s
        assert "K=" in s


class TestOutlierSplitter:
    @pytest.mark.parametrize("config", [PRESET_2_5BIT, PRESET_3_5BIT])
    def test_k_roundtrip_shape(self, config):
        """K roundtrip preserves shape."""
        splitter = OutlierSplitter(config, seed=42)
        k, _ = rand_kv()
        qk = splitter.quantize_k(k)
        k_hat = splitter.dequantize_k(qk)
        assert k_hat.shape == k.shape

    @pytest.mark.parametrize("config", [PRESET_2_5BIT, PRESET_3_5BIT])
    def test_v_roundtrip_shape(self, config):
        """V roundtrip preserves shape."""
        splitter = OutlierSplitter(config, seed=42)
        _, v = rand_kv()
        qv = splitter.quantize_v(v)
        v_hat = splitter.dequantize_v(qv)
        assert v_hat.shape == v.shape

    def test_k_reconstruction_mse(self):
        """
        K MSE at 3.5-bit should be within expected range.
        Vectors have std=0.5, so E[‖x‖²] ≈ 32. TurboQuantProd optimizes inner
        product (not L2), so L2 error is higher than TurboQuantMSE. Expected MSE
        ≈ 32 × D_mse(b_internal) ≈ 32 × 0.117 ≈ 3.7. Allow up to 6.0.
        """
        splitter = OutlierSplitter(PRESET_3_5BIT, seed=0)
        k = torch.randn(1000, 128) * 0.5
        qk = splitter.quantize_k(k)
        k_hat = splitter.dequantize_k(qk)
        mse = ((k - k_hat) ** 2).sum(-1).mean().item()
        assert mse < 6.0, f"K MSE too high: {mse:.4f}"

    def test_v_reconstruction_mse(self):
        """
        V MSE at 3.5-bit. V uses TurboQuantMSE (tight L2 reconstruction).
        E[‖x‖²] ≈ 32. Expected V MSE ≈ 32 × 0.034 ≈ 1.1. Allow up to 1.5.
        """
        splitter = OutlierSplitter(PRESET_3_5BIT, seed=0)
        v = torch.randn(1000, 128) * 0.5
        qv = splitter.quantize_v(v)
        v_hat = splitter.dequantize_v(qv)
        mse = ((v - v_hat) ** 2).sum(-1).mean().item()
        assert mse < 1.5, f"V MSE too high: {mse:.4f}"

    def test_k_inner_product_unbiased(self):
        """K dequantization must give unbiased inner product estimates."""
        splitter = OutlierSplitter(PRESET_3_5BIT, seed=0)
        x = torch.randn(2000, 128) * 0.5
        y = torch.randn(2000, 128) * 0.5
        qk = splitter.quantize_k(x)
        x_hat = splitter.dequantize_k(qk)

        true_ip = (y * x).sum(-1)
        est_ip = (y * x_hat).sum(-1)
        bias = (est_ip - true_ip).mean().abs().item()
        scale = true_ip.abs().mean().item() + 1e-8
        assert bias / scale < 0.03, f"K relative bias={bias/scale:.4f} > 3%"

    def test_batch_size_1_matches_batch_4(self):
        """Batched quantization must match single-item result."""
        splitter = OutlierSplitter(PRESET_3_5BIT, seed=5)
        k = torch.randn(4, 128)
        qk_batch = splitter.quantize_k(k)
        k_hat_batch = splitter.dequantize_k(qk_batch)
        for i in range(4):
            qk_i = splitter.quantize_k(k[i : i + 1])
            k_hat_i = splitter.dequantize_k(qk_i)
            assert torch.allclose(k_hat_batch[i], k_hat_i[0], atol=1e-4), (
                f"Batch vs single mismatch at i={i}"
            )

    def test_quantize_kv_convenience(self):
        """quantize_kv / dequantize_kv must match separate calls."""
        splitter = OutlierSplitter(PRESET_2_5BIT, seed=7)
        k, v = rand_kv(batch=2, seq=4)
        qk_sep = splitter.quantize_k(k)
        qv_sep = splitter.quantize_v(v)
        k_hat_sep = splitter.dequantize_k(qk_sep)
        v_hat_sep = splitter.dequantize_v(qv_sep)

        qk_j, qv_j = splitter.quantize_kv(k, v)
        k_hat_j, v_hat_j = splitter.dequantize_kv(qk_j, qv_j)

        assert torch.allclose(k_hat_sep, k_hat_j, atol=1e-5)
        assert torch.allclose(v_hat_sep, v_hat_j, atol=1e-5)

    def test_no_nan_or_inf(self):
        """Quantization must not produce NaN or Inf."""
        splitter = OutlierSplitter(PRESET_2_5BIT, seed=11)
        k = torch.randn(8, 8, 32, 128)
        v = torch.randn(8, 8, 32, 128)
        qk, qv = splitter.quantize_kv(k, v)
        k_hat, v_hat = splitter.dequantize_kv(qk, qv)
        assert torch.isfinite(k_hat).all(), "NaN/Inf in K reconstruction"
        assert torch.isfinite(v_hat).all(), "NaN/Inf in V reconstruction"

    @pytest.mark.parametrize("config", [PRESET_2_5BIT, PRESET_3_5BIT])
    def test_4d_kv_shape(self, config):
        """Handles 4D tensors [batch, heads, seq, dim] correctly."""
        splitter = OutlierSplitter(config, seed=99)
        k, v = rand_kv(batch=2, heads=8, seq=10, d=128)
        qk, qv = splitter.quantize_kv(k, v)
        k_hat, v_hat = splitter.dequantize_kv(qk, qv)
        assert k_hat.shape == (2, 8, 10, 128)
        assert v_hat.shape == (2, 8, 10, 128)
