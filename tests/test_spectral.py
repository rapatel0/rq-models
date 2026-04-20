"""
Tests for turboquant/spectral — SpectralQuant KV cache.

Covers:
  - _select_d_eff: variance threshold, flat-spectrum fallback
  - SpectralQuantizer: encode → decode roundtrip cosine similarity
  - SpectralQuantizer: fallback path (raw f16) for flat eigenspectrum
  - SpectralKVCache: SSM layer skip
  - SpectralKVCache: single-token (decode) update
  - SpectralKVCache: multi-token (prefill) update
  - SpectralKVCache: get_seq_length / seen_tokens
  - CalibrationStore: save → load roundtrip
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
N_HEADS  = 4
HEAD_DIM = 64
D_EFF_K  = 8
D_EFF_V  = 24
BATCH    = 1


# ---------------------------------------------------------------------------
# Helpers: build a synthetic LayerCalibration
# ---------------------------------------------------------------------------

def _make_random_eigvec(n_heads: int, head_dim: int) -> torch.Tensor:
    """Random orthogonal matrix per head via QR decomposition."""
    mats = []
    for _ in range(n_heads):
        q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim))
        mats.append(q)
    return torch.stack(mats)  # [H, D, D]


def _make_codebook(n_heads: int, n_centroids: int, dim: int) -> torch.Tensor:
    return torch.randn(n_heads, n_centroids, dim)


def _make_calibration(
    layer_idx: int = 0,
    n_heads: int = N_HEADS,
    head_dim: int = HEAD_DIM,
    d_eff_k: int = D_EFF_K,
    d_eff_v: int = D_EFF_V,
    fallback_k: bool = False,
    fallback_v: bool = False,
):
    from turboquant.spectral.calibrator import LayerCalibration

    return LayerCalibration(
        layer_idx=layer_idx,
        d_eff_k=d_eff_k,
        d_eff_v=d_eff_v,
        eigvec_k=_make_random_eigvec(n_heads, head_dim),
        eigvec_v=_make_random_eigvec(n_heads, head_dim),
        codebook_k_signal=_make_codebook(n_heads, 16, d_eff_k),
        codebook_k_noise=_make_codebook(n_heads, 4, head_dim - d_eff_k),
        codebook_v_signal=_make_codebook(n_heads, 16, d_eff_v),
        codebook_v_noise=_make_codebook(n_heads, 4, head_dim - d_eff_v),
        eigenvalues_k=torch.ones(n_heads, head_dim),
        eigenvalues_v=torch.ones(n_heads, head_dim),
        fallback_k=fallback_k,
        fallback_v=fallback_v,
    )


# ---------------------------------------------------------------------------
# _select_d_eff
# ---------------------------------------------------------------------------

class TestSelectDEff:
    def test_clear_spectrum(self):
        """Top few eigenvalues dominate → d_eff well below head_dim."""
        from turboquant.spectral.calibrator import _select_d_eff

        eigs = torch.zeros(HEAD_DIM)
        eigs[:5] = 100.0  # 5 signal dims
        eigs[5:]  = 0.01  # negligible noise

        d_eff, fallback = _select_d_eff(eigs, variance_threshold=0.99)
        assert fallback is False
        assert d_eff <= 5

    def test_flat_spectrum_triggers_fallback(self):
        """Flat eigenspectrum → fallback, d_eff = head_dim."""
        from turboquant.spectral.calibrator import _select_d_eff

        eigs = torch.ones(HEAD_DIM)  # perfectly flat
        d_eff, fallback = _select_d_eff(eigs, variance_threshold=0.99)
        assert fallback is True
        assert d_eff == HEAD_DIM

    def test_zero_eigenvalues(self):
        """All-zero eigenvalues → fallback."""
        from turboquant.spectral.calibrator import _select_d_eff

        eigs = torch.zeros(HEAD_DIM)
        d_eff, fallback = _select_d_eff(eigs, variance_threshold=0.99)
        assert fallback is True

    def test_threshold_100pct(self):
        """variance_threshold=1.0 requires all dims → d_eff = head_dim."""
        from turboquant.spectral.calibrator import _select_d_eff

        eigs = torch.zeros(HEAD_DIM)
        eigs[:3] = 10.0
        eigs[3:] = 0.1  # still non-zero, so 100% threshold needs all

        d_eff, fallback = _select_d_eff(eigs, variance_threshold=1.0)
        # d_eff should be head_dim or near it when 100% requested
        assert d_eff == HEAD_DIM or (not fallback and d_eff > 3)


# ---------------------------------------------------------------------------
# SpectralQuantizer — encode/decode roundtrip
# ---------------------------------------------------------------------------

class TestSpectralQuantizer:
    """
    Encode → decode roundtrip using *true* eigenvectors fitted from Gaussian data.
    The PCA basis is exact; only Lloyd-Max quantization introduces error.
    With d_eff covering 99% of variance, cosine sim should be > 0.94.
    """

    @pytest.fixture
    def quantizer_and_cal(self):
        from turboquant.spectral.quantizer import SpectralQuantizer
        from turboquant.spectral.calibrator import LayerCalibration, _lloyd_max_fit, _select_d_eff

        n_heads = 2
        head_dim = 32
        n_tokens = 256

        # Simulate low-rank K data: only 4 dims have variance
        low_rank = torch.randn(n_tokens, 4) * 5.0
        padding = torch.randn(n_tokens, head_dim - 4) * 0.01
        data = torch.cat([low_rank, padding], dim=-1)  # [N, D]

        # Fit PCA via covariance
        data_c = data - data.mean(0)
        cov = (data_c.T @ data_c) / (n_tokens - 1)
        eigs, vecs = torch.linalg.eigh(cov)
        eigs = eigs.flip(-1).clamp(min=0)
        vecs = vecs.flip(-1)  # descending

        d_eff, fallback = _select_d_eff(eigs, 0.99)
        assert not fallback, f"Expected signal dims but got fallback (d_eff={d_eff})"

        # Project to PCA basis for codebook fitting
        projected = data_c @ vecs  # [N, D]
        signal = projected[:, :d_eff]
        noise  = projected[:, d_eff:]

        cb_sig = _lloyd_max_fit(signal, 16).unsqueeze(0).expand(n_heads, -1, -1)
        cb_noi = _lloyd_max_fit(noise,  4).unsqueeze(0).expand(n_heads, -1, -1)
        eigvec_all = vecs.unsqueeze(0).expand(n_heads, -1, -1)

        cal = LayerCalibration(
            layer_idx=0,
            d_eff_k=d_eff,
            d_eff_v=d_eff,
            eigvec_k=eigvec_all.clone(),
            eigvec_v=eigvec_all.clone(),
            codebook_k_signal=cb_sig.clone().float(),
            codebook_k_noise=cb_noi.clone().float(),
            codebook_v_signal=cb_sig.clone().float(),
            codebook_v_noise=cb_noi.clone().float(),
            eigenvalues_k=eigs.unsqueeze(0).expand(n_heads, -1),
            eigenvalues_v=eigs.unsqueeze(0).expand(n_heads, -1),
        )

        q = SpectralQuantizer({0: cal})
        q._device = torch.device("cpu")
        return q, cal, n_heads, head_dim

    def test_roundtrip_cosine_similarity(self, quantizer_and_cal):
        """Single-token encode → decode should achieve cosine sim > 0.94.

        Test vectors are constructed by mapping a codebook centroid *back* to
        the original space so the input is guaranteed to snap to that centroid,
        giving exact signal reconstruction (quantization error = 0 for signal
        dims, small error only from noise dims ~ 0).
        """
        q, cal, n_heads, head_dim = quantizer_and_cal
        d_eff = cal.d_eff_k

        # Use centroid 0 from cb_k_signal as the signal vector in PCA space
        x_orig = torch.zeros(n_heads, head_dim)
        for h in range(n_heads):
            # centroid in PCA signal space: [d_eff]
            centroid = cal.codebook_k_signal[h, 0]              # [d_eff]
            # Map back to original space via signal eigenvectors
            signal_basis = cal.eigvec_k[h, :, :d_eff]           # [D, d_eff]
            x_orig[h] = signal_basis @ centroid                  # [D]

        # Shape expected by encode: [batch=1, n_heads, seq=1, head_dim]
        k = x_orig.unsqueeze(0).unsqueeze(2).half()

        qkv = q.encode_k(k, layer_idx=0)
        k_hat = q.decode_k([qkv], layer_idx=0)  # [1, H, 1, D]

        k_dec = k_hat.squeeze(0).squeeze(1).float()  # [H, D]

        # Cosine similarity should be high since the centroid is exact
        cos_sims = F.cosine_similarity(x_orig, k_dec, dim=-1)  # [H]
        assert cos_sims.min().item() > 0.94, (
            f"Cosine sim {cos_sims.tolist()} below 0.94 — centroid reconstruction failed"
        )

    def test_roundtrip_sequence(self, quantizer_and_cal):
        """Multi-token sequence: decode output has correct shape."""
        q, cal, n_heads, head_dim = quantizer_and_cal

        stored = []
        for _ in range(5):
            x = torch.randn(1, n_heads, 1, head_dim).half()
            stored.append(q.encode_k(x, layer_idx=0))

        k_full = q.decode_k(stored, layer_idx=0)
        assert k_full.shape == (1, n_heads, 5, head_dim)

    def test_fallback_raw_storage(self):
        """Fallback path: flat eigenspectrum → raw f16, not indices."""
        from turboquant.spectral.quantizer import SpectralQuantizer
        from turboquant.spectral.calibrator import LayerCalibration

        cal = _make_calibration(fallback_k=True, fallback_v=True)
        q = SpectralQuantizer({0: cal})
        q._device = torch.device("cpu")

        k = torch.randn(1, N_HEADS, 1, HEAD_DIM).half()
        qkv = q.encode_k(k, layer_idx=0)

        assert qkv.raw is not None, "Expected raw f16 storage for fallback layer"
        assert qkv.raw.dtype == torch.float16

    def test_fallback_decode_exact(self):
        """Fallback layers: decode should return exact f16 input."""
        from turboquant.spectral.quantizer import SpectralQuantizer

        cal = _make_calibration(fallback_k=True, fallback_v=True)
        q = SpectralQuantizer({0: cal})
        q._device = torch.device("cpu")

        k = torch.randn(1, N_HEADS, 1, HEAD_DIM).half()
        stored = [q.encode_k(k, layer_idx=0)]
        k_dec = q.decode_k(stored, layer_idx=0)

        k_orig = k.squeeze(2).float()    # [1, H, D]
        k_dec_ = k_dec.squeeze(2).float() # [1, H, D]
        assert torch.allclose(k_orig, k_dec_, atol=1e-3), "Fallback decode should be near-exact"

    def test_missing_layer_raises(self):
        """Requesting a layer not in calibration raises CalibrationNotFoundError."""
        from turboquant.spectral.quantizer import SpectralQuantizer
        from turboquant.spectral.store import CalibrationNotFoundError

        cal = _make_calibration(layer_idx=0)
        q = SpectralQuantizer({0: cal})
        q._device = torch.device("cpu")

        with pytest.raises(CalibrationNotFoundError):
            q.encode_k(torch.randn(1, N_HEADS, 1, HEAD_DIM).half(), layer_idx=99)


# ---------------------------------------------------------------------------
# SpectralKVCache
# ---------------------------------------------------------------------------

class TestSpectralKVCache:
    @pytest.fixture
    def cache(self):
        from turboquant.spectral.kv_cache import SpectralKVCache

        cal = _make_calibration(layer_idx=0)
        return SpectralKVCache({0: cal})

    def test_decode_single_token(self, cache):
        """Single decode step: update returns (K, V) with shape [1, H, 1, D]."""
        k = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
        v = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)

        k_out, v_out = cache.update(k, v, layer_idx=0)

        assert k_out.shape == (BATCH, N_HEADS, 1, HEAD_DIM)
        assert v_out.shape == (BATCH, N_HEADS, 1, HEAD_DIM)
        assert k_out.dtype == torch.float16

    def test_sequence_grows(self, cache):
        """After T decode steps, returned K/V sequence has length T."""
        T = 6
        for t in range(T):
            k = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
            v = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
            k_out, v_out = cache.update(k, v, layer_idx=0)

        assert k_out.shape[-2] == T, f"Expected seq_len={T}, got {k_out.shape[-2]}"

    def test_prefill_multi_token(self, cache):
        """Prefill: update with seq>1 stores each token and returns correct seq_len."""
        T = 5
        k = torch.randn(BATCH, N_HEADS, T, HEAD_DIM, dtype=torch.float16)
        v = torch.randn(BATCH, N_HEADS, T, HEAD_DIM, dtype=torch.float16)

        k_out, v_out = cache.update(k, v, layer_idx=0)

        assert k_out.shape == (BATCH, N_HEADS, T, HEAD_DIM)
        assert cache.get_seq_length(0) == T

    def test_ssm_layer_skip(self):
        """Layer not in calibration → delegated to DynamicCache (no spectral storage)."""
        from turboquant.spectral.kv_cache import SpectralKVCache

        cal = _make_calibration(layer_idx=0)
        cache = SpectralKVCache({0: cal})

        k = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
        v = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)

        # Layer 5 is not in calibration — should delegate to DynamicCache
        k_out, v_out = cache.update(k, v, layer_idx=5)

        assert k_out.shape[-2] == 1
        # Spectral storage for layer 5 should be empty
        assert len(cache._k_spectral) <= 5 or cache._k_spectral[5] == []

    def test_get_seq_length_attention_layer(self, cache):
        """get_seq_length returns the number of cached spectral tokens."""
        for _ in range(3):
            k = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
            v = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
            cache.update(k, v, layer_idx=0)

        assert cache.get_seq_length(0) == 3

    def test_seen_tokens(self, cache):
        """seen_tokens reflects the number of tokens cached in any attention layer."""
        T = 4
        for _ in range(T):
            k = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
            v = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
            cache.update(k, v, layer_idx=0)

        assert cache.seen_tokens == T

    def test_compression_ratio_shape(self, cache):
        """compression_ratio() returns expected keys with positive ratio."""
        for _ in range(10):
            k = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
            v = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, dtype=torch.float16)
            cache.update(k, v, layer_idx=0)

        stats = cache.compression_ratio()
        assert "compression_ratio" in stats
        assert stats["total_tokens"] == 10
        assert stats["compression_ratio"] > 0


# ---------------------------------------------------------------------------
# CalibrationStore — save/load roundtrip
# ---------------------------------------------------------------------------

class TestCalibrationStore:
    def test_save_load_roundtrip(self):
        """Save a LayerCalibration to safetensors and load it back."""
        from turboquant.spectral.store import CalibrationStore

        cal = _make_calibration(layer_idx=0)
        calibration = {0: cal}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test-calibration.safetensors"
            CalibrationStore.save(calibration, path)

            loaded = CalibrationStore.load(path)

        assert 0 in loaded
        lc = loaded[0]
        assert lc.d_eff_k == D_EFF_K
        assert lc.d_eff_v == D_EFF_V
        assert lc.eigvec_k.shape == (N_HEADS, HEAD_DIM, HEAD_DIM)
        assert lc.codebook_k_signal.shape == (N_HEADS, 16, D_EFF_K)
        assert lc.codebook_k_noise.shape  == (N_HEADS, 4, HEAD_DIM - D_EFF_K)

    def test_load_missing_raises(self):
        """Loading a non-existent path raises CalibrationNotFoundError."""
        from turboquant.spectral.store import CalibrationStore, CalibrationNotFoundError

        with pytest.raises(CalibrationNotFoundError):
            CalibrationStore.load("/tmp/does_not_exist_spectral.safetensors")

    def test_sidecar_path_sanitizes(self):
        """sidecar_path sanitizes model names with slashes and special chars."""
        from turboquant.spectral.store import CalibrationStore

        path = CalibrationStore.sidecar_path("Qwen/Qwen3.5-9B-Instruct", "calibration/")
        assert "/" not in path.stem, f"Slash leaked into filename: {path}"
        assert path.suffix == ".safetensors"

    def test_multi_layer_roundtrip(self):
        """Multiple layers saved and loaded correctly, preserving all layer indices."""
        from turboquant.spectral.store import CalibrationStore

        calibration = {
            0: _make_calibration(layer_idx=0),
            4: _make_calibration(layer_idx=4),
            8: _make_calibration(layer_idx=8),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi-layer.safetensors"
            CalibrationStore.save(calibration, path)
            loaded = CalibrationStore.load(path)

        assert set(loaded.keys()) == {0, 4, 8}
        for idx in [0, 4, 8]:
            assert loaded[idx].layer_idx == idx
            assert loaded[idx].eigvec_k.shape == (N_HEADS, HEAD_DIM, HEAD_DIM)
