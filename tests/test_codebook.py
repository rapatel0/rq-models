"""Tests for turboquant/codebook.py — codebook validity and distortion."""

import math
import numpy as np
import pytest
import torch

from turboquant.codebook import (
    load_codebook,
    codebook_mse_cost,
    _beta_pdf,
    _centroid,
)

# Paper's reported D_mse values (cost * d) for b=1..4 at moderate-to-large d.
# These are empirical upper bounds from Theorem 1 (paper §1.3).
PAPER_DMse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

# Tolerance: 15% above paper value (finite-d Beta vs Gaussian gap)
TOLERANCE = 1.15


class TestCodebookProperties:
    @pytest.mark.parametrize("d,b", [(128, b) for b in [1, 2, 3, 4]])
    def test_symmetry(self, d, b):
        """Codebook must be symmetric: c[i] == -c[2^b - 1 - i]."""
        cb = load_codebook(d, b).numpy()
        assert np.allclose(cb, -cb[::-1], atol=1e-5), f"Codebook d={d},b={b} not symmetric"

    @pytest.mark.parametrize("d,b", [(128, b) for b in [1, 2, 3, 4]])
    def test_size(self, d, b):
        """Codebook must have exactly 2^b centroids."""
        cb = load_codebook(d, b)
        assert len(cb) == 2**b, f"Expected {2**b} centroids, got {len(cb)}"

    @pytest.mark.parametrize("d,b", [(128, b) for b in [1, 2, 3, 4]])
    def test_sorted_ascending(self, d, b):
        """Centroids must be sorted in ascending order."""
        cb = load_codebook(d, b).numpy()
        assert np.all(np.diff(cb) > 0), f"Codebook d={d},b={b} not sorted"

    @pytest.mark.parametrize("d,b", [(128, b) for b in [1, 2, 3, 4]])
    def test_within_unit_interval(self, d, b):
        """All centroids must lie strictly in (-1, 1)."""
        cb = load_codebook(d, b).numpy()
        assert np.all(np.abs(cb) < 1.0), f"Codebook d={d},b={b} has centroid outside (-1,1)"

    @pytest.mark.parametrize("d,b", [(32, b) for b in [1, 2, 3, 4]] + [(96, b) for b in [1, 2]])
    def test_all_dims_loadable(self, d, b):
        """All required (d, b) combinations must be loadable."""
        cb = load_codebook(d, b)
        assert len(cb) == 2**b


class TestCodebookDistortion:
    @pytest.mark.parametrize("b", [1, 2, 3, 4])
    def test_mse_cost_within_paper_bound_d128(self, b):
        """
        For d=128, D_mse = d * C(f_X, b) must be ≤ PAPER_DMse[b] * TOLERANCE.
        Small overshoot is expected because d=128 Beta is not perfectly Gaussian.
        """
        cost = codebook_mse_cost(128, b)
        dmse = 128 * cost
        bound = PAPER_DMse[b] * TOLERANCE
        assert dmse <= bound, (
            f"b={b}: D_mse={dmse:.4f} exceeds paper bound {PAPER_DMse[b]} × {TOLERANCE} = {bound:.4f}"
        )

    @pytest.mark.parametrize("b", [1, 2])
    def test_mse_cost_within_paper_bound_d32(self, b):
        """d=32 has heavier-tailed Beta; allow 20% over paper value."""
        cost = codebook_mse_cost(32, b)
        dmse = 32 * cost
        bound = PAPER_DMse[b] * 1.25
        assert dmse <= bound, f"b={b}: D_mse(d=32)={dmse:.4f} > {bound:.4f}"


class TestBetaPdf:
    def test_integrates_to_one(self):
        """Beta PDF must integrate to 1 over [-1, 1]."""
        from scipy import integrate
        for d in [32, 96, 128]:
            area, _ = integrate.quad(
                lambda x: _beta_pdf(np.array([x]), d)[0], -1, 1
            )
            assert abs(area - 1.0) < 1e-4, f"PDF integral={area} ≠ 1 for d={d}"

    def test_symmetric(self):
        """PDF must be symmetric: f(x) == f(-x)."""
        xs = np.linspace(-0.9, 0.9, 20)
        for d in [32, 128]:
            assert np.allclose(
                _beta_pdf(xs, d), _beta_pdf(-xs, d), atol=1e-6
            ), f"PDF not symmetric for d={d}"
