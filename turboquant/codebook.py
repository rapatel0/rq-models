"""
Lloyd-Max codebook generation and loading for TurboQuant.

Codebooks are precomputed optimal scalar quantizers for the Beta distribution
that arises from random projections onto the unit sphere in R^d:

    f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)  for x ∈ [-1, 1]

In high dimensions (d ≥ 64) this converges to N(0, 1/d).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from scipy import integrate

# Module-level cache: (d, b) -> codebook tensor
_CODEBOOK_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}

CODEBOOK_DIR = Path(__file__).parent.parent / "codebooks"


# ---------------------------------------------------------------------------
# Beta distribution helpers
# ---------------------------------------------------------------------------

def _beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """PDF of a single coordinate of a uniform random point on S^{d-1}."""
    alpha = (d - 1) / 2
    # f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
    log_norm = (
        math.lgamma(d / 2)
        - 0.5 * math.log(math.pi)
        - math.lgamma(alpha)
    )
    with np.errstate(invalid="ignore"):
        val = np.where(
            np.abs(x) < 1.0,
            np.exp(log_norm + ((d - 3) / 2) * np.log(1.0 - x**2 + 1e-300)),
            0.0,
        )
    return val


def _centroid(lo: float, hi: float, d: int) -> float:
    """E[X | X ∈ (lo, hi)] under the Beta-on-sphere distribution."""
    num, _ = integrate.quad(lambda x: x * _beta_pdf(np.array([x]), d)[0], lo, hi)
    den, _ = integrate.quad(lambda x: _beta_pdf(np.array([x]), d)[0], lo, hi)
    if den < 1e-300:
        return (lo + hi) / 2
    return num / den


def _mse_cost(centroids: np.ndarray, boundaries: np.ndarray, d: int) -> float:
    """Total MSE cost C(f_X, b) = Σ_i ∫_{b_i}^{b_{i+1}} (x - c_i)² f(x) dx."""
    cost = 0.0
    for i, c in enumerate(centroids):
        lo, hi = boundaries[i], boundaries[i + 1]
        v, _ = integrate.quad(
            lambda x, ci=c: (x - ci) ** 2 * _beta_pdf(np.array([x]), d)[0],
            lo,
            hi,
        )
        cost += v
    return cost


# ---------------------------------------------------------------------------
# Lloyd-Max iteration
# ---------------------------------------------------------------------------

def compute_codebook(d: int, b: int, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    """
    Compute the optimal b-bit scalar quantizer (Lloyd-Max) for the Beta
    distribution of a single coordinate of a uniform point on S^{d-1}.

    Returns centroids array of shape [2^b], sorted ascending, in [-1, 1].
    """
    n = 2**b
    # Initialize centroids uniformly in (-1, 1), symmetric
    centroids = np.linspace(-1 + 2 / (n + 1), 1 - 2 / (n + 1), n)

    for _ in range(max_iter):
        # Step 1: Update boundaries (Voronoi: midpoints between centroids)
        boundaries = np.empty(n + 1)
        boundaries[0] = -1.0
        boundaries[-1] = 1.0
        for i in range(1, n):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

        # Step 2: Update centroids (conditional expectation in each cell)
        new_centroids = np.array([
            _centroid(boundaries[i], boundaries[i + 1], d) for i in range(n)
        ])

        delta = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if delta < tol:
            break

    return centroids


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_codebook(centroids: np.ndarray, d: int, b: int) -> Path:
    """Save codebook tensor to CODEBOOK_DIR/d{d}_b{b}.pt."""
    CODEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = CODEBOOK_DIR / f"d{d}_b{b}.pt"
    torch.save(torch.tensor(centroids, dtype=torch.float32), path)
    return path


def load_codebook(d: int, b: int) -> torch.Tensor:
    """
    Load codebook for dimension d and bit-width b.
    Returns float32 tensor of shape [2^b], sorted ascending.
    Caches results in _CODEBOOK_CACHE.
    """
    key = (d, b)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]

    path = CODEBOOK_DIR / f"d{d}_b{b}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Codebook not found: {path}. "
            f"Run scripts/generate_codebooks.py first."
        )

    cb = torch.load(path, map_location="cpu", weights_only=True)
    _CODEBOOK_CACHE[key] = cb
    return cb


def codebook_mse_cost(d: int, b: int) -> float:
    """Return C(f_X, b) for the precomputed codebook at (d, b)."""
    cb = load_codebook(d, b).numpy()
    n = len(cb)
    boundaries = np.empty(n + 1)
    boundaries[0] = -1.0
    boundaries[-1] = 1.0
    for i in range(1, n):
        boundaries[i] = (cb[i - 1] + cb[i]) / 2
    return _mse_cost(cb, boundaries, d)
