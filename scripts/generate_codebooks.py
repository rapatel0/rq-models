#!/usr/bin/env python3
"""
Offline precomputation of Lloyd-Max codebooks for TurboQuant.

Generates codebooks for all required (d, b) combinations and saves them
to the codebooks/ directory as PyTorch tensors.

Usage:
    python scripts/generate_codebooks.py
    python scripts/generate_codebooks.py --dims 32 96 128 --bits 1 2 3 4 5
"""

import argparse
import math
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from turboquant.codebook import compute_codebook, save_codebook, _mse_cost, _beta_pdf
import numpy as np

# Paper's theoretical upper bounds on C(f_X, b) * d for b = 1..4
# From Theorem 1: D_mse = d * C(f_X, b) ≤ (√3π/2) · (1/4^b)
# So C(f_X, b) ≤ (√3π/2) / (d · 4^b)
# The empirical values for moderate d are: 0.36/d, 0.117/d, 0.03/d, 0.009/d
PAPER_COST_TIMES_D = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}


def main():
    parser = argparse.ArgumentParser(description="Generate TurboQuant codebooks")
    parser.add_argument(
        "--dims", nargs="+", type=int, default=[32, 96, 128],
        help="Head partition dimensions to generate codebooks for"
    )
    parser.add_argument(
        "--bits", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Bit-widths to generate codebooks for"
    )
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="Validate MSE cost against paper bounds"
    )
    args = parser.parse_args()

    total = len(args.dims) * len(args.bits)
    done = 0

    for d in args.dims:
        for b in args.bits:
            done += 1
            print(f"[{done}/{total}] Computing codebook d={d}, b={b} ({2**b} centroids)...", end=" ", flush=True)
            centroids = compute_codebook(d, b)

            # Compute boundaries for MSE cost
            n = 2**b
            boundaries = np.empty(n + 1)
            boundaries[0] = -1.0
            boundaries[-1] = 1.0
            for i in range(1, n):
                boundaries[i] = (centroids[i - 1] + centroids[i]) / 2

            cost = _mse_cost(centroids, boundaries, d)
            cost_times_d = cost * d

            path = save_codebook(centroids, d, b)

            status = "✓"
            if args.validate and b in PAPER_COST_TIMES_D:
                paper_val = PAPER_COST_TIMES_D[b]
                upper_bound = math.sqrt(3 * math.pi) / 2 / (4**b)
                if cost_times_d > upper_bound * 1.01:
                    status = f"⚠ EXCEEDS BOUND (cost·d={cost_times_d:.4f} > {upper_bound:.4f})"
                else:
                    status = f"✓ cost·d={cost_times_d:.4f} (paper≈{paper_val:.4f})"

            print(f"{status} → {path.name}")

            # Validate symmetry
            if not np.allclose(centroids, -centroids[::-1], atol=1e-6):
                print(f"  ⚠ WARNING: codebook is not symmetric for d={d}, b={b}")

    print(f"\nDone. Codebooks written to: {path.parent}/")


if __name__ == "__main__":
    main()
