# Sprint 001: TurboQuant KV Cache Quantization for Qwen3.5-27B

## Overview

Implement TurboQuant (arXiv:2504.19874) as a drop-in KV cache compressor for
Qwen3.5-27B on HuggingFace transformers. TurboQuant is a two-stage, online,
data-oblivious vector quantizer:

1. **TurboQuant_mse** (Algorithm 1): Random rotation projects input onto the unit
   hypersphere, inducing Beta-distributed coordinates. Precomputed Lloyd-Max
   codebooks then optimally scalar-quantize each coordinate independently.
   Achieves near-optimal MSE: D_mse <= (sqrt(3)*pi/2) * (1/4^b).

2. **TurboQuant_prod** (Algorithm 2): Composes TurboQuant_mse at (b-1) bits with
   1-bit QJL (Quantized Johnson-Lindenstrauss) on the residual. The QJL step
   produces an **unbiased** inner-product estimator — the property that attention
   scores depend on — with distortion D_prod <= (sqrt(3)*pi^2 * ||y||^2) / (d * 4^b).

Target effective bit-widths: **2.5-bit** and **3.5-bit** KV cache via outlier
channel splitting. At 3.5-bit the paper achieves quality-neutral generation on
LongBench-V1 and 0.997 NIAH recall (matching full-precision 0.997).

No calibration data is required for the quantizer itself (rotation, codebook, and
projection are all data-oblivious). Outlier channel indices require a lightweight
one-time calibration pass (~128 tokens).

## Use Cases

1. **Long-context inference on a single GPU**: At 3.5-bit, KV cache is compressed
   >4.5x. For Qwen3.5-27B at 32k context, KV memory drops from ~16 GB (fp16) to
   ~3.5 GB, freeing headroom for longer contexts or larger batches on A100-80GB.

2. **Quality-critical compression**: TurboQuant_prod's **unbiased** inner product
   estimator preserves attention score distributions without systematic shift,
   unlike KIVI and other MSE-only quantizers that introduce multiplicative bias
   at low bit-widths (paper §3.2 shows bias of 2/pi at b=1).

3. **Streaming / online**: Quantize-on-arrival with zero preprocessing. Each new
   token's KV is compressed independently — no need for calibration data, no
   second pass, no batch-level statistics.

4. **Architecture-agnostic**: Operates per-head on `head_dim`-sized vectors.
   Same implementation works for any GQA model with head_dim=128 (Llama 3,
   Mistral, Qwen2.5, etc.).

## Architecture

### Component Dependency Graph

```
scripts/generate_codebooks.py
    └─> turboquant/codebook.py       (offline Lloyd-Max solver)
            │
            v
        turboquant/core.py           (TurboQuantMSE, TurboQuantProd)
            │
            v
        turboquant/outlier.py        (outlier channel detection + splitting)
            │
            v
        turboquant/packing.py        (bit-packing/unpacking for indices + signs)
            │
            v
        turboquant/kv_cache.py       (QuantizedKVCache replacing DynamicCache)
            │
            v
        turboquant/model.py          (Qwen3.5-27B model patching + calibration)
            │
            v
        scripts/eval_*.py            (NIAH, LongBench, perplexity benchmarks)
```

### Data Flow (per attention layer, per token)

```
key_states: [batch, n_kv_heads, 1, head_dim=128]
    │
    ├─ normalize: k_hat = k / ||k||, store ||k|| as float16
    │
    ├─ split by outlier_indices / non_outlier_indices (per-layer, fixed after calibration)
    │     │                                 │
    │     v                                 v
    │  outlier_sub [.., d_out=32]         non_outlier_sub [.., d_non=96]
    │     │                                 │
    │     v                                 v
    │  TurboQuantProd(b=3)               TurboQuantProd(b=2)
    │     │                                 │
    │     v                                 v
    │  (idx_o[32], sgn_o[32], γ_o)       (idx_n[96], sgn_n[96], γ_n)
    │     │                                 │
    │     └────────────┬────────────────────┘
    │                  v
    │     bit-pack + append to QuantizedKVCache storage
    │
    ├─ on attention read (every decode step):
    │     unpack all → dequant outlier + dequant non-outlier
    │     → reassemble full [batch, heads, seq_total, 128] → rescale by ||k||
    v
value_states: (identical pipeline, separate quantizer instances per layer)
```

### Bit-Budget Accounting

Each quantized vector stored by TurboQuant_prod at b bits on d_sub dimensions:

| Component           | Bits                  | Notes                               |
|---------------------|-----------------------|-------------------------------------|
| MSE indices         | (b-1) * d_sub         | Each coord gets (b-1)-bit index     |
| QJL signs           | 1 * d_sub             | 1-bit sign per coord                |
| Residual norm γ     | 16                    | float16, 1 per sub-vector           |
| **Sub-total**       | **b * d_sub + 16**    |                                     |

Plus 1 shared input norm ||x|| (16 bits, amortized across both sub-groups).

**2.5-bit configuration** (head_dim=128, 32 outlier channels):
- Outlier: 3 * 32 + 16 = 112 bits
- Non-outlier: 2 * 96 + 16 = 208 bits
- Input norm: 16 bits
- **Total: 336 bits** per (head, token, K-or-V)
- **Effective: 336/128 = 2.625 bits/channel**
- The paper's stated "2.5 bits" counts only the b-rate per coordinate (ignoring
  the ~0.125 bits/channel norm overhead). Both accountings are valid; the norms
  are O(1/d) per coordinate and negligible at large d.

**3.5-bit configuration** (head_dim=128, 32 outlier @ 4-bit, 96 non-outlier @ 3-bit):
- Outlier: 4 * 32 + 16 = 144 bits
- Non-outlier: 3 * 96 + 16 = 304 bits
- Input norm: 16 bits
- **Total: 464 bits → 3.625 bits/channel** (paper nominal: 3.5)

**Compression ratios** (vs fp16 = 16 bits/channel):
- 2.5-bit nominal: 16/2.625 ≈ **6.1x**
- 3.5-bit nominal: 16/3.625 ≈ **4.4x**

## Implementation

### Phase 1: Core Quantization Algorithms

**Goal**: Implement TurboQuant_mse and TurboQuant_prod with verified distortion
bounds. No model integration — pure math + unit tests on CPU.

#### 1a. Codebook Generation — `turboquant/codebook.py`

```python
# turboquant/codebook.py

import numpy as np
from scipy import integrate
from scipy.special import gamma as gamma_fn
from typing import Tuple

def beta_pdf(x: float, d: int) -> float:
    """
    PDF of a single coordinate of a point uniformly distributed on S^{d-1}.

    From Lemma 1 of the paper:
        f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)

    Defined on x in [-1, 1]. Symmetric around 0. For d=128 this closely
    approximates N(0, 1/128) with std ≈ 0.0884.

    Implementation note: use log-gamma for numerical stability at large d.
    """
    ...

def lloyd_max(d: int, b: int, max_iter: int = 200, tol: float = 1e-12
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute optimal Lloyd-Max centroids for b-bit quantization.

    Solves the continuous 1D k-means problem (Eq. 4 in paper):
        C(f_X, b) = min over c_1 <= ... <= c_{2^b}
                     sum_i integral_{t_{i-1}}^{t_i} |x - c_i|^2 * f_X(x) dx

    where t_i = (c_i + c_{i+1})/2 are Voronoi boundaries.

    Algorithm:
        1. Initialize: 2^b equally-spaced quantiles of the Beta distribution.
           Use scipy.stats.beta(a=(d-1)/2, b=(d-1)/2) on [0,1], map to [-1,1].
        2. Iterate until convergence:
           a. boundaries[i] = (centroids[i-1] + centroids[i]) / 2
           b. centroids[i] = integral(x * f_X(x), lo, hi) / integral(f_X(x), lo, hi)
              computed via scipy.integrate.quad
        3. Convergence: max |c_new - c_old| < tol

    Returns:
        centroids: shape [2^b], sorted ascending, symmetric (c_i ≈ -c_{2^b-1-i})
        boundaries: shape [2^b + 1], with boundaries[0]=-1, boundaries[-1]=1

    Expected centroids for d=128:
        b=1: [-0.0886, +0.0886]                (≈ ±sqrt(2/(pi*d)))
        b=2: [-0.1336, -0.0400, +0.0400, +0.1336] (≈ ±1.51/sqrt(d), ±0.453/sqrt(d))
        b=3: 8 centroids (computed numerically)
        b=4: 16 centroids (computed numerically)

    Expected MSE cost C(f_X, b) for d=128 (distortion = d * C):
        b=1: C ≈ 0.36/128 ≈ 0.00281  → D_mse ≈ 0.36
        b=2: C ≈ 0.117/128 ≈ 0.000914 → D_mse ≈ 0.117
        b=3: C ≈ 0.03/128 ≈ 0.000234  → D_mse ≈ 0.03
        b=4: C ≈ 0.009/128 ≈ 0.0000703 → D_mse ≈ 0.009
    """
    ...

def compute_mse_cost(centroids: np.ndarray, d: int) -> float:
    """
    Compute the MSE cost C(f_X, b) for given centroids.
    C = sum_i integral_{t_{i-1}}^{t_i} (x - c_i)^2 * f_X(x) dx
    Total MSE distortion D_mse = d * C.
    """
    ...

def generate_all_codebooks(
    dims: list[int] = [32, 64, 96, 128],
    bit_widths: list[int] = [1, 2, 3, 4],
    output_path: str = "codebooks/default.json",
) -> dict:
    """Precompute and serialize codebooks as JSON."""
    ...
```

**Numerical stability**: For `d=128`, the PDF `(1 - x^2)^62.5` underflows to 0 for
`|x| > ~0.3`. Use `log_gamma` and compute the log-PDF, exponentiating only inside
the integrand. Initialize centroids within the effective support `[-3*sigma, 3*sigma]`
where `sigma = 1/sqrt(d) ≈ 0.0884`.

#### 1b. Core Quantizers — `turboquant/core.py`

```python
# turboquant/core.py

import math
import torch
from torch import Tensor
from typing import Tuple, Optional

class TurboQuantMSE:
    """
    Algorithm 1 from the paper: MSE-optimal vector quantizer.

    Attributes:
        d (int): vector dimension
        b (int): bit-width per coordinate
        rotation (Tensor): [d, d] orthogonal matrix Π (float32)
        centroids (Tensor): [2^b] sorted centroid values (float32)
        boundaries (Tensor): [2^b - 1] Voronoi midpoints for bucketize

    Usage:
        quant = TurboQuantMSE(d=128, b=2)
        idx = quant.quantize(x)        # x: [..., 128] → idx: [..., 128] uint8
        x_hat = quant.dequantize(idx)   # idx: [..., 128] → x_hat: [..., 128]
    """

    def __init__(self, d: int, b: int, codebook_path: Optional[str] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 seed: int = 42):
        self.d = d
        self.b = b

        # 1. Random orthogonal rotation matrix via QR decomposition
        gen = torch.Generator(device="cpu").manual_seed(seed)
        A = torch.randn(d, d, generator=gen)
        Q, R = torch.linalg.qr(A)
        # Haar measure correction: Q @ diag(sign(diag(R)))
        Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)
        self.rotation = Q.to(device=device, dtype=dtype)   # [d, d]

        # 2. Load precomputed codebook
        centroids = self._load_codebook(d, b, codebook_path)
        self.centroids = torch.tensor(centroids, device=device, dtype=dtype)
        # Voronoi boundaries = midpoints between consecutive centroids
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2

    def quantize(self, x: Tensor) -> Tensor:
        """
        x: [..., d] — assumed unit-norm (caller normalizes)
        returns: [..., d] uint8 indices in [0, 2^b - 1]

        Steps:
            1. y = x @ Π^T  (rotate into codebook space)
            2. idx_j = bucketize(y_j, boundaries)  (nearest centroid)
        """
        y = x.float() @ self.rotation.T                        # [..., d] float32
        idx = torch.bucketize(y, self.boundaries)              # [..., d] int64
        return idx.to(torch.uint8)

    def dequantize(self, idx: Tensor) -> Tensor:
        """
        idx: [..., d] uint8
        returns: [..., d] float (reconstructed vector)

        Steps:
            1. y_hat_j = centroids[idx_j]  (centroid lookup)
            2. x_hat = y_hat @ Π           (inverse rotation: Π^T^T = Π for orth.)
        """
        y_hat = self.centroids[idx.long()]                     # [..., d]
        x_hat = y_hat @ self.rotation                          # [..., d]
        return x_hat


class TurboQuantProd:
    """
    Algorithm 2 from the paper: inner-product-optimal vector quantizer.

    Composes TurboQuant_mse at (b-1) bits with 1-bit QJL on the residual.

    Key property: E[<y, dequant(quant(x))>] = <y, x>  (unbiased)

    Attributes:
        d (int): vector dimension
        b (int): total bit-width (MSE uses b-1, QJL uses 1)
        mse (TurboQuantMSE): inner MSE quantizer at (b-1) bits
        S (Tensor): [d, d] random Gaussian projection matrix for QJL

    Storage per vector:
        - (b-1)*d bits for MSE indices
        - d bits for QJL signs
        - 16 bits for residual norm γ (float16)
        Total: b*d + 16 bits
    """

    def __init__(self, d: int, b: int, codebook_path: Optional[str] = None,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 mse_seed: int = 42, qjl_seed: int = 137):
        assert b >= 2, "TurboQuantProd requires b >= 2 (MSE part uses b-1 >= 1)"
        self.d = d
        self.b = b

        # MSE quantizer at (b-1) bits
        self.mse = TurboQuantMSE(d, b - 1, codebook_path, device, dtype,
                                 seed=mse_seed)

        # QJL projection matrix: S_{i,j} ~ N(0,1), NOT orthogonal
        gen = torch.Generator(device="cpu").manual_seed(qjl_seed)
        self.S = torch.randn(d, d, generator=gen, dtype=dtype).to(device)

    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x: [..., d] — assumed unit-norm

        Returns:
            idx:   [..., d] uint8 — MSE codebook indices (b-1 bits used)
            signs: [..., d] bool  — QJL sign bits
            gamma: [...]   float  — residual L2 norm

        Steps (Algorithm 2, lines 4-8):
            1. idx = QUANT_mse(x)
            2. r = x - DEQUANT_mse(idx)           # residual
            3. qjl = sign(S @ r)                   # {-1, +1}^d
            4. γ = ||r||_2
        """
        idx = self.mse.quantize(x)                              # [..., d]
        x_mse = self.mse.dequantize(idx)                        # [..., d]
        r = x.float() - x_mse                                   # [..., d] residual
        gamma = torch.linalg.norm(r, dim=-1)                    # [...]

        # QJL: sign(S @ r) — batch matmul
        Sr = r @ self.S.T                                        # [..., d]
        signs = Sr >= 0                                          # [..., d] bool

        return idx, signs, gamma

    def dequantize(self, idx: Tensor, signs: Tensor, gamma: Tensor) -> Tensor:
        """
        Returns: [..., d] float — reconstructed vector

        Steps (Algorithm 2, lines 9-12):
            1. x_mse = DEQUANT_mse(idx)
            2. qjl = 2*signs - 1                   # bool → {-1, +1}
            3. x_qjl = sqrt(pi/2)/d * γ * S^T @ qjl
            4. return x_mse + x_qjl
        """
        x_mse = self.mse.dequantize(idx)                        # [..., d]
        qjl = 2.0 * signs.float() - 1.0                         # [..., d] {-1,+1}

        # QJL inverse: (sqrt(pi/2) / d) * gamma * S^T @ qjl
        scale = math.sqrt(math.pi / 2) / self.d                 # scalar
        St_qjl = qjl @ self.S                                   # [..., d] (S^T @ qjl)
        x_qjl = scale * gamma.unsqueeze(-1) * St_qjl            # [..., d]

        return x_mse + x_qjl
```

**Critical implementation details**:

1. **`torch.bucketize` semantics**: Given sorted boundaries `[b0, b1, ..., b_{n-1}]`,
   `bucketize(y, boundaries)` returns index `i` such that `boundaries[i-1] < y <= boundaries[i]`.
   With `2^b` centroids, we have `2^b - 1` boundaries (midpoints), producing
   indices in `{0, 1, ..., 2^b - 1}`. This is the vectorized equivalent of
   `argmin_k |y_j - c_k|` since Voronoi regions for sorted centroids are exactly
   the half-open intervals between midpoints.

2. **Rotation matrix generation**: The sign correction `Q * sign(diag(R))` ensures
   Haar-uniform distribution. However, for TurboQuant's guarantees, the only
   requirement is that `Π` is orthogonal (so `Π · x` is uniform on S^{d-1} for
   any fixed unit x). The Haar correction is mathematically nice but not strictly
   necessary for correctness.

3. **QJL sign convention**: `sign(0) = 0` in PyTorch, but QJL requires {-1, +1}.
   Using `Sr >= 0` maps zero to True (+1). Since `Sr` is a sum of 128 Gaussian
   products, P(Sr = 0) = 0 in theory. In float32, exact zero is astronomically
   unlikely but we handle it defensively.

4. **Precision**: Quantize/dequantize internals run in float32 for numerical
   stability. The caller casts the output back to bfloat16/float16 as needed.

#### 1c. Unit Tests — `tests/test_core.py`

```python
class TestTurboQuantMSE:
    """
    Theorem 1 verification:
        D_mse := E[||x - dequant(quant(x))||^2] <= (sqrt(3)*pi/2) * (1/4^b)

    Refined per-b bounds (paper, d=128):
        b=1: 0.36,  b=2: 0.117,  b=3: 0.03,  b=4: 0.009
    """

    @pytest.mark.parametrize("b,bound", [(1,0.36), (2,0.117), (3,0.03), (4,0.009)])
    def test_mse_distortion_bound(self, b, bound):
        """Monte Carlo: 10k random unit vectors on S^127, measure mean ||x - x_hat||^2.
        Assert measured <= bound * 1.10 (10% tolerance for finite-sample variance)."""

    def test_codebook_symmetry(self):
        """centroids should satisfy c_i ≈ -c_{2^b - 1 - i}. |sum(centroids)| < 1e-10."""

    def test_rotation_orthogonality(self):
        """||Π @ Π^T - I||_F < 1e-5."""

    def test_quantize_dequantize_shapes(self):
        """Input [4, 8, 128] → idx [4, 8, 128] uint8 → output [4, 8, 128] float."""

    def test_indices_in_range(self):
        """All idx values in [0, 2^b - 1]."""


class TestTurboQuantProd:
    """
    Theorem 2 verification:
        E[<y, dequant(quant(x))>] = <y, x>     (unbiasedness)
        D_prod <= (sqrt(3)*pi^2*||y||^2/d) * (1/4^b)

    Refined per-b bounds (paper, d=128):
        b=2: 0.56/128,  b=3: 0.18/128,  b=4: 0.047/128
    """

    @pytest.mark.parametrize("b,bound_num", [(2,0.56), (3,0.18), (4,0.047)])
    def test_inner_product_distortion(self, b, bound_num):
        """10k random unit-vector pairs. D_prod = mean(|<y,x> - <y,x_hat>|^2).
        Assert D_prod <= (bound_num / 128) * 1.10."""

    @pytest.mark.parametrize("b", [2, 3, 4])
    def test_unbiasedness(self, b):
        """Mean signed error: |mean(<y,x_hat> - <y,x>)| < 0.005 over 10k trials."""

    def test_residual_norm_positive(self):
        """gamma = ||r|| >= 0 always."""

    def test_qjl_signs_binary(self):
        """signs tensor is dtype=bool."""

    def test_batched_quantize(self):
        """Input [batch=4, seq=16, d=128] works correctly."""
```

#### 1d. Codebook Script — `scripts/generate_codebooks.py`

Precomputes Lloyd-Max codebooks for all needed (d, b) pairs and writes to
`codebooks/default.json`. Also validates MSE cost against paper.

### Phase 2: KV Cache Wrapper with Bit-Packed Storage

**Goal**: Build `QuantizedKVCache` with proper bit-packing and outlier channel
splitting, verified on synthetic tensors (no model yet).

#### 2a. Outlier Channel Detection — `turboquant/outlier.py`

```python
# turboquant/outlier.py

from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class ChannelSplit:
    """Describes outlier/non-outlier channel partitioning for one layer."""
    outlier_indices: Tensor      # [n_outlier] long — channel indices
    non_outlier_indices: Tensor  # [n_non_outlier] long
    b_outlier: int               # bit-width for outlier channels
    b_non_outlier: int           # bit-width for non-outlier channels

    @property
    def n_outlier(self) -> int:
        return self.outlier_indices.shape[0]

    @property
    def n_non_outlier(self) -> int:
        return self.non_outlier_indices.shape[0]

    @property
    def effective_bits_nominal(self) -> float:
        """Paper's nominal accounting (index+sign bits only, norms amortized)."""
        return (self.n_outlier * self.b_outlier +
                self.n_non_outlier * self.b_non_outlier) / (
                self.n_outlier + self.n_non_outlier)


def detect_outlier_channels(
    calibration_kvs: Tensor,
    n_outlier: int,
) -> Tensor:
    """
    Identify outlier channel indices from calibration KV activations.

    Args:
        calibration_kvs: [n_samples, head_dim] — KV vectors from calibration pass
        n_outlier: number of channels to designate as outliers

    Returns:
        outlier_indices: [n_outlier] long — sorted ascending

    Method: per-channel RMS, top-n_outlier by magnitude.
    """
    rms = torch.sqrt(torch.mean(calibration_kvs ** 2, dim=0))  # [head_dim]
    _, top_idx = torch.topk(rms, n_outlier)
    return top_idx.sort().values


# Predefined split configs (paper §4.3)
SPLIT_2_5_BIT = dict(n_outlier=32, b_outlier=3, b_non_outlier=2)
# Nominal: (32*3 + 96*2)/128 = 288/128 = 2.25 (index+sign bits)
# With norms: (32*3 + 96*2 + 3*16)/128 = 336/128 = 2.625 bits/channel
# Paper labels this "2.5-bit" using per-coordinate bit rate only

SPLIT_3_5_BIT = dict(n_outlier=32, b_outlier=4, b_non_outlier=3)
# Nominal: (32*4 + 96*3)/128 = 416/128 = 3.25 (index+sign bits)
# With norms: (32*4 + 96*3 + 3*16)/128 = 464/128 = 3.625 bits/channel
# Paper labels this "3.5-bit"
```

**On the paper's bit accounting**: The paper states `(32*3 + 96*2)/128 = 2.5` (page
18). This arithmetic yields 2.25, not 2.5. The discrepancy arises because the
paper counts the **total effective rate** including amortized norm storage: with
3 fp16 norms (input norm, outlier residual norm, non-outlier residual norm) adding
48 bits, the total is `288 + 48 = 336` bits. At `336/128 = 2.625`, this is still
not exactly 2.5. The paper likely uses a slightly different norm accounting (e.g.,
2 norms instead of 3, or amortizes norms differently). For our implementation, we
use honest per-component accounting and note that the effective rate is ~2.6-2.7
bits, which still achieves >6x compression over fp16.

#### 2b. Bit-Packed Storage — `turboquant/packing.py`

```python
# turboquant/packing.py

import torch
from torch import Tensor

def pack_indices(indices: Tensor, bits: int) -> Tensor:
    """
    Pack b-bit integer indices into uint8 bytes (GPU-friendly bitwise ops).

    Args:
        indices: [..., d] uint8, values in [0, 2^bits - 1]
        bits: 1, 2, 3, or 4

    Returns:
        packed: [..., packed_dim] uint8

    Packing (little-endian within byte):
        bits=1: 8 per byte → packed_dim = d/8    (d=128 → 16 bytes)
        bits=2: 4 per byte → packed_dim = d/4    (d=128 → 32 bytes)
        bits=3: 2 per byte → packed_dim = d/2    (d=128 → 64 bytes, 2 wasted bits)
        bits=4: 2 per byte → packed_dim = d/2    (d=128 → 64 bytes, exact)

    Implementation uses torch.bitwise_left_shift and torch.bitwise_or on
    reshaped tensors. No Python loops.
    """
    ...

def unpack_indices(packed: Tensor, bits: int, d: int) -> Tensor:
    """Inverse of pack_indices. Returns [..., d] uint8."""
    ...

def pack_signs(signs: Tensor) -> Tensor:
    """Pack bool[..., d] → uint8[..., d//8]. 8 signs per byte."""
    ...

def unpack_signs(packed: Tensor, d: int) -> Tensor:
    """Inverse. Returns [..., d] bool."""
    ...
```

**Memory savings from packing** (per KV vector, head_dim=128, 2.5-bit config):

| Component              | Unpacked (bytes) | Packed (bytes) |
|------------------------|------------------|----------------|
| Outlier MSE idx (2-bit, 32 dims)  | 32 | 8   |
| Outlier QJL signs (32 dims)       | 32 | 4   |
| Outlier γ (float16)               | 2  | 2   |
| Non-outlier MSE idx (1-bit, 96 dims) | 96 | 12 |
| Non-outlier QJL signs (96 dims)   | 96 | 12  |
| Non-outlier γ (float16)           | 2  | 2   |
| Input norm (float16)              | 2  | 2   |
| **Total**                         | **262** | **42** |
| vs fp16 KV (128 * 2 bytes)       | **256** | **256** |
| **Compression ratio**            | **~1x** | **~6.1x** |

Without packing there is virtually no savings. **Packing is essential, not optional.**

#### 2c. Quantized KV Cache — `turboquant/kv_cache.py`

```python
# turboquant/kv_cache.py

import torch
from torch import Tensor
from typing import Optional, Tuple, List
from transformers import DynamicCache
from turboquant.core import TurboQuantProd
from turboquant.outlier import ChannelSplit
from turboquant.packing import pack_indices, unpack_indices, pack_signs, unpack_signs

class QuantizedKVCache(DynamicCache):
    """
    Drop-in replacement for HuggingFace DynamicCache.

    Stores KV states in TurboQuant-compressed form with bit-packing.
    Dequantizes on access for attention computation.

    Memory layout per layer (stored in self._quantized_storage[layer_idx]):
        For each of K and V:
            packed_idx_outlier:     [batch, heads, seq, packed_dim_o]  uint8
            packed_sgn_outlier:     [batch, heads, seq, d_out//8]     uint8
            gamma_outlier:          [batch, heads, seq]               float16
            packed_idx_non_outlier: [batch, heads, seq, packed_dim_n] uint8
            packed_sgn_non_outlier: [batch, heads, seq, d_non//8]    uint8
            gamma_non_outlier:      [batch, heads, seq]               float16
            input_norm:             [batch, heads, seq]               float16

    Dequantize-all strategy:
        On each update() call, we dequantize ALL past tokens and concatenate with
        the newly quantized (also dequantized) tokens. This is required because
        HuggingFace attention expects the full KV sequence.

        Optimization: cache dequantized tensors from previous steps and only
        dequantize the new token(s). Invalidate cache on any operation that
        modifies past storage (e.g., reorder_cache for beam search).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        channel_splits: List[ChannelSplit],
        device: torch.device,
        compute_dtype: torch.dtype = torch.float32,
        storage_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.channel_splits = channel_splits
        self.compute_dtype = compute_dtype
        self.storage_dtype = storage_dtype

        # Create quantizer instances per layer (one pair per layer for K, one for V)
        # Both K and V share the same ChannelSplit but have independent
        # rotation/projection matrices
        self._quantizers_k: List[Tuple[TurboQuantProd, TurboQuantProd]] = []
        self._quantizers_v: List[Tuple[TurboQuantProd, TurboQuantProd]] = []
        for i in range(num_layers):
            split = channel_splits[i]
            d_o, d_n = split.n_outlier, split.n_non_outlier
            b_o, b_n = split.b_outlier, split.b_non_outlier
            self._quantizers_k.append((
                TurboQuantProd(d_o, b_o, device=device, mse_seed=1000+i, qjl_seed=2000+i),
                TurboQuantProd(d_n, b_n, device=device, mse_seed=3000+i, qjl_seed=4000+i),
            ))
            self._quantizers_v.append((
                TurboQuantProd(d_o, b_o, device=device, mse_seed=5000+i, qjl_seed=6000+i),
                TurboQuantProd(d_n, b_n, device=device, mse_seed=7000+i, qjl_seed=8000+i),
            ))

        # Quantized storage: initialized lazily on first update
        self._q_storage = [None] * num_layers

        # Dequantized cache (optimization: avoid redundant dequantization)
        self._dequant_cache_k = [None] * num_layers
        self._dequant_cache_v = [None] * num_layers

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Quantize new KV, append to storage, return full dequantized sequence.

        key_states:   [batch, n_kv_heads, seq_new, head_dim] (post-RoPE)
        value_states: [batch, n_kv_heads, seq_new, head_dim]

        Returns: (full_keys, full_values) both [batch, n_kv_heads, seq_total, head_dim]
        """
        ...

    def get_seq_length(self, layer_idx: int = 0) -> int:
        ...

    def _quantize_and_pack(self, x: Tensor, layer_idx: int,
                           is_key: bool) -> dict:
        """Normalize → split → quantize → pack. Returns storage dict."""
        ...

    def _dequantize_all(self, layer_idx: int, is_key: bool) -> Tensor:
        """Unpack → dequantize → reassemble → rescale. Returns [b, h, seq, d]."""
        ...
```

**Dequantize-all optimization**: Naively dequantizing all past tokens on every
`update()` is O(seq_len * d^2) per layer (due to the rotation matmul). At seq_len=32k
with d=128, that's 32k * 128^2 ≈ 524M FLOPs per layer per update, or ~34G FLOPs
across 64 layers. On A100 (312 TFLOPS fp32), this is ~0.1ms — acceptable.

However, we still implement incremental dequantization (cache previous results,
only dequantize new tokens) as a robustness measure. The cache is invalidated on
`reorder_cache()` (beam search).

### Phase 3: Qwen3.5-27B Model Integration

**Goal**: Patch Qwen3.5-27B to use `QuantizedKVCache` with calibration-based
outlier detection. End-to-end `model.generate()` works.

#### 3a. Model Patching — `turboquant/model.py`

```python
# turboquant/model.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from turboquant.kv_cache import QuantizedKVCache
from turboquant.outlier import ChannelSplit, detect_outlier_channels

def patch_model_for_quantized_kv(
    model_name: str = "Qwen/Qwen3.5-27B",
    bit_config: str = "3.5",
    calibration_text: Optional[str] = None,
    calibration_tokens: int = 128,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple:
    """
    Load model, calibrate outlier channels, return (model, tokenizer, cache_factory).

    Steps:
        1. Load model + tokenizer
        2. Extract config: num_hidden_layers, num_key_value_heads, head_dim
        3. Run calibration forward pass to detect outlier channels per layer
        4. Build QuantizedKVCache factory function
        5. Monkey-patch model.generate() to inject the quantized cache

    Returns:
        model: the loaded model (unmodified weights)
        tokenizer: the tokenizer
        make_cache: callable() → QuantizedKVCache (new cache for each generation)
    """
    ...


def _run_calibration(
    model, tokenizer, text: Optional[str], n_tokens: int, n_outlier: int
) -> list[ChannelSplit]:
    """
    Capture KV activations via forward hooks, detect outlier channels.

    For each Qwen2DecoderLayer (or equivalent):
        1. Register a forward hook on self_attn that captures key_states
           AFTER RoPE application but BEFORE cache.update()
        2. Run a single forward pass on n_tokens of text
        3. Compute per-channel RMS and select top-n_outlier channels
        4. Return List[ChannelSplit] with per-layer outlier indices

    If calibration_text is None, use a generic English paragraph.
    The outlier indices should be stable across different calibration texts
    (>80% overlap) since outlier channels are a structural property of the model.
    """
    ...
```

**HuggingFace integration considerations**:

1. **Cache API**: Pin `transformers>=4.40`. The `DynamicCache.update()` signature
   is `update(key_states, value_states, layer_idx, cache_kwargs) -> (keys, values)`.
   Our subclass must return the same types.

2. **Qwen model cache usage**: In `Qwen2Attention.forward()`, the model calls
   `past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)`
   after applying RoPE to keys. Our `update()` receives post-RoPE tensors, which
   is correct (RoPE is a rotation that preserves norms).

3. **GQA head layout**: Qwen3.5-27B has `num_key_value_heads=8` (fewer than query
   heads). KV cache stores 8-head tensors. Query attention uses `repeat_kv()` to
   expand KV heads before the attention matmul. Our quantizer operates on the
   8-head KV tensors directly; expansion happens after dequantization.

4. **bfloat16 handling**: Model outputs key_states in bfloat16. Our quantizer
   casts to float32 internally, returns float32. The attention code handles the
   cast via PyTorch's automatic type promotion, or we cast back to bfloat16 in
   `update()`.

5. **If Qwen3.5-27B is unavailable**: Prototype and test against `Qwen/Qwen2.5-7B`
   (same architecture family, head_dim=128, 28 layers, 4 KV heads). Swap to
   Qwen3.5-27B when available.

#### 3b. Tests — `tests/test_integration.py`

```python
class TestQwenIntegration:
    """End-to-end tests. Require GPU; skip if unavailable."""

    def test_generate_no_errors(self):
        """256-token prompt → 32 new tokens. No NaN/Inf in output."""

    def test_output_shapes(self):
        """Cache returns [batch, 8, seq_total, 128] for K and V."""

    def test_batch_inference(self):
        """batch_size=4, variable prompt lengths. All outputs valid."""

    def test_cache_seq_length_grows(self):
        """After N generate steps, get_seq_length() == prompt_len + N."""

    def test_calibration_stability(self):
        """Two calibration runs with different texts: >80% outlier index overlap."""
```

### Phase 4: Evaluation and Benchmarks

**Goal**: Validate quality and performance against paper's reported results.

| Script | Metric | Target |
|--------|--------|--------|
| `scripts/eval_perplexity.py` | WikiText-2 PPL (stride=512, max_len=2048) | 3.5-bit ≤ baseline + 0.1 nats |
| `scripts/eval_niah.py` | NIAH recall (4k-32k context) | ≥ 0.99 at 3.5-bit |
| `scripts/eval_longbench.py` | LongBench-E average score | 3.5-bit ≥ 49.0 (paper: 50.06) |
| `scripts/bench_throughput.py` | Per-token quant+dequant latency | < 1ms on A100 |

## Files Summary

| File | Purpose | Phase | Lines (est.) |
|------|---------|-------|-------------|
| `pyproject.toml` | Project metadata, dependencies | 1 | 40 |
| `turboquant/__init__.py` | Public API: TurboQuantMSE, TurboQuantProd, QuantizedKVCache | 1 | 20 |
| `turboquant/codebook.py` | `beta_pdf()`, `lloyd_max()`, `generate_all_codebooks()` | 1 | 150 |
| `turboquant/core.py` | `TurboQuantMSE`, `TurboQuantProd` classes | 1 | 200 |
| `turboquant/outlier.py` | `ChannelSplit`, `detect_outlier_channels()`, split configs | 2 | 80 |
| `turboquant/packing.py` | `pack_indices()`, `unpack_indices()`, `pack_signs()`, `unpack_signs()` | 2 | 120 |
| `turboquant/kv_cache.py` | `QuantizedKVCache(DynamicCache)` | 2 | 300 |
| `turboquant/model.py` | `patch_model_for_quantized_kv()`, `_run_calibration()` | 3 | 150 |
| `tests/test_codebook.py` | Lloyd-Max convergence, symmetry, MSE cost validation | 1 | 80 |
| `tests/test_core.py` | Distortion bounds (Theorem 1 & 2), unbiasedness, shapes | 1 | 150 |
| `tests/test_packing.py` | Pack/unpack round-trip for all bit-widths and dims | 2 | 60 |
| `tests/test_kv_cache.py` | Cache update, dequant correctness, memory, shapes | 2 | 120 |
| `tests/test_integration.py` | Qwen3.5-27B end-to-end smoke tests (GPU) | 3 | 80 |
| `scripts/generate_codebooks.py` | Offline Lloyd-Max codebook precomputation | 1 | 60 |
| `scripts/eval_perplexity.py` | WikiText-2 sliding-window perplexity | 4 | 80 |
| `scripts/eval_niah.py` | Needle-In-A-Haystack evaluation | 4 | 100 |
| `scripts/eval_longbench.py` | LongBench-E evaluation | 4 | 100 |
| `scripts/bench_throughput.py` | Quantize/dequantize latency benchmark | 4 | 60 |
| `codebooks/default.json` | Precomputed centroids for d∈{32,64,96,128}, b∈{1..4} | 1 | — |

## Definition of Done

### Phase 1 — Core Algorithms
- [ ] `codebook.py`: `lloyd_max(d=128, b)` converges for b=1,2,3,4
- [ ] `codebook.py`: Centroids are symmetric: `|c_i + c_{2^b-1-i}| < 1e-10` for all i
- [ ] `codebook.py`: `beta_pdf()` integrates to 1.0±1e-8 over [-1, 1]
- [ ] `codebook.py`: MSE cost `C(f_X, b)` matches paper values within 5% for b=1..4
- [ ] `core.py`: `TurboQuantMSE.rotation` is orthogonal: `||Π Π^T - I||_F < 1e-5`
- [ ] `core.py`: MSE distortion (10k random unit vectors, d=128) within 10% of paper bounds for b=1,2,3,4
- [ ] `core.py`: Inner-product distortion (10k random pairs, d=128) within 10% of paper bounds for b=2,3,4
- [ ] `core.py`: Inner-product estimator unbiased: `|mean(<y,x_hat> - <y,x>)| < 0.005` for b=2,3,4
- [ ] `core.py`: Both quantizers handle batched input `[batch, seq, d]` correctly
- [ ] `core.py`: All indices in `[0, 2^b - 1]`; all signs are bool; all gammas ≥ 0
- [ ] `scripts/generate_codebooks.py` produces `codebooks/default.json`
- [ ] `pytest tests/test_codebook.py tests/test_core.py` — all pass

### Phase 2 — KV Cache Wrapper
- [ ] `packing.py`: `unpack(pack(x, bits), bits, d) == x` for bits=1,2,3,4 and d=32,64,96,128
- [ ] `packing.py`: `unpack_signs(pack_signs(s), d) == s` for d=32,64,96,128
- [ ] `packing.py`: Packing operates on GPU tensors without CPU round-trip
- [ ] `outlier.py`: `detect_outlier_channels` returns correct top-K on synthetic data
- [ ] `kv_cache.py`: `update()` returns `[batch, heads, seq_total, head_dim]` — correct shape
- [ ] `kv_cache.py`: Dequantized output from cache matches direct quantizer output (no packing-induced error)
- [ ] `kv_cache.py`: Sequential `update()` calls grow `get_seq_length()` correctly
- [ ] `kv_cache.py`: Quantized storage uses < 40% memory of fp16 DynamicCache at seq_len=4096 (2.5-bit)
- [ ] `kv_cache.py`: `reorder_cache()` for beam search works (or raises NotImplementedError)
- [ ] `pytest tests/test_packing.py tests/test_kv_cache.py` — all pass

### Phase 3 — Model Integration
- [ ] `model.py`: Successfully loads Qwen3.5-27B (or fallback Qwen2.5-7B) and extracts config
- [ ] `model.py`: Calibration completes in < 30s on A100; outlier indices are per-layer
- [ ] `model.py`: Outlier indices stable across calibration runs (>80% overlap)
- [ ] `test_integration.py`: Generate 32 tokens from 256-token prompt — no NaN/Inf
- [ ] `test_integration.py`: Batch inference (batch_size=4) correct output shapes
- [ ] `test_integration.py`: Cache seq_length matches prompt_len + generated_len
- [ ] `pytest tests/test_integration.py` — all pass (GPU required)

### Phase 4 — Evaluation
- [ ] WikiText-2 perplexity at 3.5-bit ≤ baseline + 0.1 nats
- [ ] WikiText-2 perplexity at 2.5-bit ≤ baseline + 0.5 nats
- [ ] NIAH recall ≥ 0.99 at 3.5-bit, context lengths 4k-32k
- [ ] Quantize+dequantize latency < 1ms per token on A100 (median, 1000 iterations)
- [ ] All eval scripts run end-to-end without error

## Risks

### High Risk

1. **HuggingFace cache interface mismatch**: Qwen3.5-27B may use a non-standard
   cache class or override internal cache methods beyond `update()`.
   - **Likelihood**: Medium (Qwen2.5 uses standard DynamicCache)
   - **Impact**: High (blocks all integration)
   - **Mitigation**: Read `Qwen2Attention.forward()` source before implementing.
     Fallback: monkey-patch `forward()` to intercept KV before/after cache ops.

2. **Dequantize-all latency at long sequences**: Every `update()` call dequantizes
   the entire past sequence. At 32k tokens, 64 layers, this is 32k × 64 × 8 heads
   × 128-dim matmuls.
   - **Likelihood**: Medium
   - **Impact**: High (throughput regression)
   - **Mitigation**: Cache dequantized tensors incrementally. Only dequantize new
     tokens; prepend cached past. Invalidate on `reorder_cache()`.

### Medium Risk

3. **Numerical precision in bfloat16↔float32 boundary**: Rotation `Π @ x` for
   bfloat16 input loses precision. The rotated coordinates must fall within the
   codebook support (±0.3 for d=128); bfloat16 has sufficient precision in this
   range but accumulation of 128 products may drift.
   - **Mitigation**: Cast to float32 before rotation. Clamp rotated values to
     `[centroids[0] - margin, centroids[-1] + margin]`.

4. **Outlier channel instability**: If outlier channels differ significantly
   between calibration runs or between K and V, the fixed split degrades quality.
   - **Mitigation**: Validate stability in Phase 3 (>80% overlap test). Fallback:
     use uniform quantization (single bit-width, no split) if unstable.

5. **Rotation/projection matrix memory**: Each `TurboQuantProd` instance stores
   `Π` (d×d) and `S` (d×d). With 64 layers × 2 (K,V) × 2 (outlier+non-outlier) =
   256 instances, each storing two d×d float32 matrices:
   256 × 2 × 128 × 128 × 4 bytes = **32 MB**. Negligible vs model weights (~54 GB).
   - **Mitigation**: None needed. Could share across layers (valid since
     data-oblivious) to reduce to ~0.5 MB if memory is tight.

### Low Risk

6. **Lloyd-Max convergence**: Well-conditioned for smooth, unimodal Beta PDF.
   Quantile initialization ensures convergence within ~50 iterations.

7. **PyTorch `bucketize` edge cases**: If a rotated coordinate exactly equals a
   boundary, `bucketize` assigns it to the right bucket. This is consistent and
   does not affect distortion bounds.

## Security Considerations

- **No external data**: Codebooks computed from mathematical distributions. No
  network calls during quantization/dequantization.
- **No model modification**: TurboQuant wraps KV cache only. Model weights are
  read-only. No weight tampering or gradient manipulation.
- **Deterministic seeds**: Rotation (Π) and projection (S) matrices generated from
  fixed seeds for reproducibility. For adversarial robustness, use
  `torch.Generator` with `os.urandom()` seed. Default: fixed seed=42 for
  reproducibility.
- **Numerical safety**: Division by vector norm clamped to `min=1e-8`. Rotated
  values clamped to codebook range to prevent out-of-bounds indexing.

## Dependencies

### Python Packages
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.1 | Core tensor ops, QR decomposition, GPU compute |
| `transformers` | >=4.40 | Qwen3.5-27B model, DynamicCache, tokenizer |
| `scipy` | >=1.11 | `integrate.quad` for Lloyd-Max, `stats.beta` for init |
| `numpy` | >=1.24 | Codebook computation, JSON serialization |
| `pytest` | >=7.0 | Unit and integration tests |
| `datasets` | >=2.14 | WikiText-2 and LongBench-E data loading |

### Hardware
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | A100-40GB | A100-80GB |
| System RAM | 64 GB | 128 GB |
| Disk | 60 GB (model weights) | 100 GB |

### External
- HuggingFace Hub access for `Qwen/Qwen3.5-27B` model download
- All inference runs offline after initial download

## Open Questions

1. **Qwen3.5-27B availability and architecture**: The model may not yet be public.
   Need to verify `head_dim`, `num_key_value_heads`, and `num_hidden_layers` at
   implementation time. If unavailable, prototype against `Qwen/Qwen2.5-7B` (same
   architecture family: head_dim=128, 28 layers, 4 KV heads) and upgrade later.

2. **Outlier detection — calibration vs fixed**: The paper's "data-oblivious"
   claim covers the quantizer (rotation, codebook, projection — no calibration
   needed). The outlier channel split is a separate concern. The paper cites
   RotateKV [51] and QJL [63] for outlier treatment, which use calibration-based
   detection. Our approach: lightweight calibration (128 tokens) to identify
   outlier indices per layer, then fix them for all subsequent inference. This is
   consistent with the literature and adds negligible startup cost.

3. **Dequantize-all vs incremental attention**: The `update()` returning full
   dequantized KV is the simplest integration but means all past tokens are
   decompressed every step. Alternatives:
   - **(A) Incremental dequant cache** (recommended for this sprint): cache
     dequantized past, only decompress new tokens. Memory: 2x (quantized + cached
     dequantized), but attention compute gets full fp16 KV.
   - **(B) Quantized attention kernel**: compute Q @ K^T directly on quantized K.
     Requires custom Triton kernel. Out of scope — Sprint 002 candidate.

4. **bfloat16 vs float32 for quantizer internals**: bfloat16 has 8 mantissa bits
   (precision ~0.8%). For d=128, codebook centroids are O(0.01-0.13). Bfloat16 can
   represent these but accumulation error in the 128-dim rotation matmul may be
   significant. Recommend float32 for quantize/dequantize internals. Benchmark
   bfloat16 as a potential optimization if float32 meets latency target.

5. **Norm storage precision**: Float16 (2 bytes, ~3 decimal digits, range [6e-8,
   65504]) is sufficient for KV vector norms (typically O(1-10)) and residual norms
   (typically O(0.01-0.1)). Use float16 to minimize storage overhead.

6. **Sharing quantizer instances across layers**: The algorithm is data-oblivious,
   so sharing Π and S across layers is mathematically valid and saves 255/256 of
   the matrix memory. Tradeoff: shared randomness means quantization errors are
   correlated across layers, which could theoretically amplify distortion through
   the transformer stack. Start with per-layer instances (32 MB total — negligible)
   and investigate sharing in Sprint 002 if memory becomes a concern.
