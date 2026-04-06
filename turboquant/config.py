"""
BitConfig: quantization configuration for TurboQuant KV cache.

Defines the outlier/regular channel split and per-partition bit-widths
for K (TurboQuantProd) and V (TurboQuantMSE) cache paths.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BitConfig:
    """
    Configuration for a TurboQuant KV cache compression regime.

    K cache uses TurboQuantProd (unbiased inner product for attention scores).
    V cache uses TurboQuantMSE (MSE reconstruction for value aggregation).

    Outlier channels: fixed first-N indices (data-oblivious, no calibration).
    """
    label: str              # Human-readable label, e.g. "2.5-bit"
    head_dim: int           # Total head dimension (e.g. 128)
    outlier_count: int      # Number of outlier channels (first-N)

    # K cache (TurboQuantProd): total bits = (outlier_k_bits-1) MSE + 1 QJL
    outlier_k_bits: int     # Total bit-width for outlier K partition
    regular_k_bits: int     # Total bit-width for regular K partition

    # V cache (TurboQuantMSE): straightforward scalar quantization
    outlier_v_bits: int     # Bit-width for outlier V partition
    regular_v_bits: int     # Bit-width for regular V partition

    @property
    def regular_count(self) -> int:
        return self.head_dim - self.outlier_count

    @property
    def k_effective_bits(self) -> float:
        """Effective bits per K coordinate (including both partitions)."""
        total = self.outlier_count * self.outlier_k_bits + self.regular_count * self.regular_k_bits
        return total / self.head_dim

    @property
    def v_effective_bits(self) -> float:
        """Effective bits per V coordinate."""
        total = self.outlier_count * self.outlier_v_bits + self.regular_count * self.regular_v_bits
        return total / self.head_dim

    def __str__(self) -> str:
        return (
            f"BitConfig({self.label}: "
            f"K={self.k_effective_bits:.2f}bit "
            f"[out={self.outlier_count}ch@{self.outlier_k_bits}b + "
            f"reg={self.regular_count}ch@{self.regular_k_bits}b], "
            f"V={self.v_effective_bits:.2f}bit "
            f"[out={self.outlier_count}ch@{self.outlier_v_bits}b + "
            f"reg={self.regular_count}ch@{self.regular_v_bits}b])"
        )


# ---------------------------------------------------------------------------
# Standard presets (head_dim=128)
# ---------------------------------------------------------------------------

PRESET_2_5BIT = BitConfig(
    label="2.5-bit",
    head_dim=128,
    outlier_count=32,       # first 32 channels
    # K: 3-bit Prod = 2-bit MSE + 1-bit QJL
    outlier_k_bits=3,
    regular_k_bits=2,       # 2-bit Prod = 1-bit MSE + 1-bit QJL
    # V: MSE only
    outlier_v_bits=3,
    regular_v_bits=2,
)
# Actual effective bits: (32×3 + 96×2) / 128 = 2.25 (paper labels as "2.5-bit")

PRESET_3_5BIT = BitConfig(
    label="3.5-bit",
    head_dim=128,
    outlier_count=64,       # first 64 channels
    # K: 4-bit Prod = 3-bit MSE + 1-bit QJL
    outlier_k_bits=4,
    regular_k_bits=3,       # 3-bit Prod = 2-bit MSE + 1-bit QJL
    # V: MSE only
    outlier_v_bits=4,
    regular_v_bits=3,
)
# Actual effective bits: (64×4 + 64×3) / 128 = 3.5 ✓

PRESET_4BIT = BitConfig(
    label="4-bit",
    head_dim=128,
    outlier_count=0,        # no split — all 128 channels at uniform 4-bit
    # K: 4-bit Prod = 3-bit MSE + 1-bit QJL (unbiased attention scores)
    outlier_k_bits=4,       # unused (outlier_count=0)
    regular_k_bits=4,
    # V: 4-bit MSE (tight L2 reconstruction)
    outlier_v_bits=4,       # unused
    regular_v_bits=4,
)
# Effective bits: 4.0 exactly. No splitting overhead.
# D_mse ≈ 0.009 (paper: b=4). Near fp16 quality at 4× compression vs fp16.
