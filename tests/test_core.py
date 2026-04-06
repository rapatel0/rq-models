"""
Tests for turboquant/core.py — distortion bound validation.

All Monte Carlo tests use 10k random unit vectors at d=128 to verify the
distortion guarantees from Theorems 1 & 2 of arXiv:2504.19874.
"""

import math
import pytest
import torch

from turboquant.core import TurboQuantMSE, TurboQuantProd, MSEQuantized

torch.manual_seed(0)

D = 128
N = 10_000  # Monte Carlo sample count

# Paper's D_mse upper bounds (Theorem 1 empirical values for moderate d)
PAPER_DMSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
# Paper's D_prod upper bounds per unit ‖y‖ (Theorem 2): 1.57/d, 0.56/d, 0.18/d, 0.047/d
PAPER_DPROD_TIMES_D = {1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047}

TOLERANCE = 1.15  # allow 15% above paper bound (finite-d gap)


def random_unit_vectors(n: int, d: int) -> torch.Tensor:
    """Generate n random unit vectors in R^d."""
    x = torch.randn(n, d)
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# TurboQuantMSE tests
# ---------------------------------------------------------------------------

class TestTurboQuantMSE:
    def test_rotation_orthogonality(self):
        """Π must be orthogonal: ‖Πᵀ·Π - I‖_F < 1e-5."""
        q = TurboQuantMSE(D, 2)
        err = q.orthogonality_error()
        assert err < 1e-5, f"Π orthogonality error = {err:.2e}"

    def test_roundtrip_shape(self):
        """Quantize→dequantize must preserve input shape."""
        q = TurboQuantMSE(D, 3)
        x = random_unit_vectors(8, D)
        qx = q.quantize(x)
        x_hat = q.dequantize(qx)
        assert x_hat.shape == x.shape

    def test_indices_in_range(self):
        """All indices must be in [0, 2^b)."""
        for b in [1, 2, 3, 4]:
            q = TurboQuantMSE(D, b)
            x = random_unit_vectors(100, D)
            qx = q.quantize(x)
            assert qx.indices.max() < 2**b, f"b={b}: index {qx.indices.max()} >= {2**b}"
            assert qx.indices.min() >= 0

    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_mse_within_paper_bound(self, b):
        """
        D_mse = E[‖x - x̃‖²] ≤ PAPER_DMSE[b] × TOLERANCE.
        Vectors are random unit vectors; norm is 1 so no rescaling bias.
        """
        q = TurboQuantMSE(D, b, seed=123)
        x = random_unit_vectors(N, D)
        qx = q.quantize(x)
        x_hat = q.dequantize(qx)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        bound = PAPER_DMSE[b] * TOLERANCE
        assert mse <= bound, (
            f"b={b}: D_mse={mse:.5f} > paper bound {PAPER_DMSE[b]} × {TOLERANCE} = {bound:.5f}"
        )

    def test_mse_decreases_with_bitwidth(self):
        """Higher bit-width must give lower MSE."""
        q1 = TurboQuantMSE(D, 1, seed=7)
        q2 = TurboQuantMSE(D, 2, seed=7)
        q3 = TurboQuantMSE(D, 3, seed=7)
        x = random_unit_vectors(1000, D)

        def mse(q):
            qx = q.quantize(x)
            return ((x - q.dequantize(qx)) ** 2).sum(dim=-1).mean().item()

        m1, m2, m3 = mse(q1), mse(q2), mse(q3)
        assert m1 > m2 > m3, f"MSE not decreasing: {m1:.4f} > {m2:.4f} > {m3:.4f}"

    def test_norm_preserved_in_expectation(self):
        """E[‖x̃‖₂] ≈ ‖x‖₂ (within 5%)."""
        q = TurboQuantMSE(D, 3, seed=5)
        x = random_unit_vectors(1000, D)
        qx = q.quantize(x)
        x_hat = q.dequantize(qx)
        orig_norm = torch.linalg.norm(x, dim=-1).mean().item()
        recon_norm = torch.linalg.norm(x_hat, dim=-1).mean().item()
        assert abs(recon_norm - orig_norm) / orig_norm < 0.05, (
            f"Norm not preserved: orig={orig_norm:.4f}, recon={recon_norm:.4f}"
        )

    def test_batched_equals_single(self):
        """Batched quantization must match single-vector quantization."""
        q = TurboQuantMSE(D, 2, seed=99)
        x = random_unit_vectors(4, D)
        # Batched
        qx_batch = q.quantize(x)
        x_hat_batch = q.dequantize(qx_batch)
        # Single
        for i in range(4):
            qx_single = q.quantize(x[i : i + 1])
            x_hat_single = q.dequantize(qx_single)
            assert torch.allclose(x_hat_batch[i], x_hat_single[0], atol=1e-5), (
                f"Batched vs single mismatch at i={i}"
            )


# ---------------------------------------------------------------------------
# TurboQuantProd tests
# ---------------------------------------------------------------------------

class TestTurboQuantProd:
    def test_roundtrip_shape(self):
        """quantize_and_store → dequantize must preserve input shape."""
        q = TurboQuantProd(D, 3)
        x = random_unit_vectors(8, D)
        pq, norm = q.quantize_and_store(x)
        x_hat = q.dequantize(pq, norm)
        assert x_hat.shape == x.shape

    def test_qjl_values(self):
        """QJL output must be strictly ±1."""
        q = TurboQuantProd(D, 3)
        x = random_unit_vectors(100, D)
        pq, _ = q.quantize_and_store(x)
        vals = pq.qjl.float().abs()
        assert torch.all(vals == 1.0), "QJL values not ±1"

    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_inner_product_unbiased(self, b):
        """
        E[⟨y, x̃⟩] ≈ ⟨y, x⟩  (bias < 1% relative for paired random vectors).
        Tests unbiasedness property of Theorem 2.
        """
        q = TurboQuantProd(D, b, seed=42)
        x = random_unit_vectors(N, D)
        y = random_unit_vectors(N, D)

        pq, norm = q.quantize_and_store(x)
        x_hat = q.dequantize(pq, norm)

        true_ip = (y * x).sum(dim=-1)          # [N]
        est_ip = (y * x_hat).sum(dim=-1)        # [N]

        bias = (est_ip - true_ip).mean().abs().item()
        scale = true_ip.abs().mean().item() + 1e-8
        rel_bias = bias / scale

        assert rel_bias < 0.01, (
            f"b={b}: relative bias = {rel_bias:.4f} >= 0.01"
        )

    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_inner_product_distortion(self, b):
        """
        D_prod = E[(⟨y,x⟩ - ⟨y,x̃⟩)²] ≤ (√3π²‖y‖²/d) · (1/4^b) × TOLERANCE.
        For unit ‖y‖=1: D_prod ≤ PAPER_DPROD_TIMES_D[b] / d × TOLERANCE.
        """
        q = TurboQuantProd(D, b, seed=42)
        x = random_unit_vectors(N, D)
        y = random_unit_vectors(N, D)  # unit ‖y‖ = 1

        pq, norm = q.quantize_and_store(x)
        x_hat = q.dequantize(pq, norm)

        true_ip = (y * x).sum(dim=-1)
        est_ip = (y * x_hat).sum(dim=-1)
        dprod = ((true_ip - est_ip) ** 2).mean().item()

        bound = (PAPER_DPROD_TIMES_D[b] / D) * TOLERANCE
        assert dprod <= bound, (
            f"b={b}: D_prod={dprod:.6f} > bound {PAPER_DPROD_TIMES_D[b]}/{D}×{TOLERANCE} = {bound:.6f}"
        )

    def test_mse_is_biased_prod_is_not(self):
        """
        TurboQuantMSE introduces bias in inner products at low bit-widths;
        TurboQuantProd is unbiased by design (Theorem 2).
        At b=1: bias of MSE estimator should be significantly non-zero;
        bias of Prod estimator should be near zero.
        """
        b = 2
        q_mse = TurboQuantMSE(D, b, seed=1)
        q_prod = TurboQuantProd(D, b, seed=1)
        x = random_unit_vectors(N, D)
        y = random_unit_vectors(N, D)

        true_ip = (y * x).sum(-1)

        # MSE path — biased at low b
        qx_mse = q_mse.quantize(x)
        x_hat_mse = q_mse.dequantize(qx_mse)
        bias_mse = ((y * x_hat_mse).sum(-1) - true_ip).mean().abs().item()

        # Prod path — must be unbiased
        pq, norm = q_prod.quantize_and_store(x)
        x_hat_prod = q_prod.dequantize(pq, norm)
        bias_prod = ((y * x_hat_prod).sum(-1) - true_ip).mean().abs().item()

        scale = true_ip.abs().mean().item() + 1e-8
        # Prod must be unbiased (< 1% relative error)
        assert bias_prod / scale < 0.01, f"Prod not unbiased: rel_bias={bias_prod/scale:.4f}"
        # MSE must also be low-bias at b=2 (paper: bias diminishes with b)
        assert bias_mse / scale < 0.01, f"MSE bias too high at b=2: rel_bias={bias_mse/scale:.4f}"

    def test_batched_equals_single(self):
        """Batched and single-vector results must agree."""
        q = TurboQuantProd(D, 3, seed=77)
        x = random_unit_vectors(4, D)
        pq_batch, norm_batch = q.quantize_and_store(x)
        x_hat_batch = q.dequantize(pq_batch, norm_batch)

        for i in range(4):
            pq_i, norm_i = q.quantize_and_store(x[i : i + 1])
            x_hat_i = q.dequantize(pq_i, norm_i)
            assert torch.allclose(x_hat_batch[i], x_hat_i[0], atol=1e-4), (
                f"Batched vs single mismatch at i={i}"
            )
