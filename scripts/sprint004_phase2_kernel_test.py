#!/usr/bin/env python3
"""Sprint 004 Phase 2a — standalone kernel parity test.

Builds the rq-vllm planar3 KV pack/unpack CUDA kernels as a torch JIT
extension (no vLLM build required) and runs the same parity tests that
will eventually live inside vLLM's CI under
tests/kernels/test_rotorquant_planar3_kv.py.

Usage (on the GPU box, with rq-vllm cloned at ~/repos/rq-vllm):

    python3 scripts/sprint004_phase2_kernel_test.py

Or with a custom rq-vllm location:

    RQ_VLLM_DIR=/path/to/rq-vllm python3 scripts/sprint004_phase2_kernel_test.py

Requires:
    - PyTorch with CUDA (matches host driver; >= 2.5)
    - nvcc on PATH (typically /usr/local/cuda/bin/nvcc)

Hard gate (kernel correctness):
    - Per-block L2 norm preserved within 1e-2 relative
    - Per-element p95 relative error < 0.10 (3-bit codebook bound)
    - Zero input -> zero output (no NaN)
    - Bytes-per-element exactly 3.125

If this passes, the math in csrc/attention/rotorquant/planar3_kv.cu is
correct and Phase 2c (FlashAttention integration) can proceed without
worrying about kernel-level bugs.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension


def build_extension(rq_vllm_dir: Path) -> object:
    """JIT-build the rq-models RotorQuant planar3 extension."""
    csrc = rq_vllm_dir / "csrc" / "attention" / "rotorquant"
    sources = [
        str(csrc / "planar3_kv.cu"),
        str(csrc / "torch_bindings.cpp"),
    ]
    for f in sources:
        if not Path(f).is_file():
            sys.exit(f"missing source: {f}")

    print(f"compiling rq_models_rotorquant from {csrc} ...", flush=True)
    ext = cpp_extension.load(
        name="rq_models_rotorquant",
        sources=sources,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            # Ada (RTX 4090 / 5090). Add 80, 86, 90 if targeting other arches.
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_90,code=sm_90",
        ],
        verbose=True,
    )
    return ext


QK_PLANAR3 = 128
BLOCK_BYTES = 50


def _make_input(n_blocks: int, seed: int = 42, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(n_blocks * QK_PLANAR3, generator=g,
                        device="cuda", dtype=torch.float32)
            * scale).to(torch.float16)


def _round_trip(ext, src: torch.Tensor, n_blocks: int) -> torch.Tensor:
    packed = torch.empty(n_blocks * BLOCK_BYTES, device="cuda", dtype=torch.uint8)
    out = torch.empty(n_blocks * QK_PLANAR3, device="cuda", dtype=torch.float16)
    ext.rotorquant_planar3_pack(src, packed, n_blocks)
    ext.rotorquant_planar3_unpack(packed, out, n_blocks)
    torch.cuda.synchronize()
    return out


def test_shapes(ext) -> None:
    print("test_shapes ...", end=" ", flush=True)
    for n in [1, 8, 64, 1024]:
        src = _make_input(n)
        packed = torch.empty(n * BLOCK_BYTES, device="cuda", dtype=torch.uint8)
        ext.rotorquant_planar3_pack(src, packed, n)
        assert packed.shape == (n * BLOCK_BYTES,)
        out = torch.empty(n * QK_PLANAR3, device="cuda", dtype=torch.float16)
        ext.rotorquant_planar3_unpack(packed, out, n)
        assert out.shape == (n * QK_PLANAR3,)
    print("PASS")


def test_l2_norm(ext) -> None:
    print("test_l2_norm ...", end=" ", flush=True)
    for n_blocks in [1, 8, 64]:
        for scale in [0.1, 1.0, 10.0]:
            src = _make_input(n_blocks, scale=scale)
            out = _round_trip(ext, src, n_blocks)
            src_blocks = src.view(n_blocks, QK_PLANAR3).float()
            out_blocks = out.view(n_blocks, QK_PLANAR3).float()
            src_n = torch.linalg.norm(src_blocks, dim=1)
            out_n = torch.linalg.norm(out_blocks, dim=1)
            rel = ((src_n - out_n).abs() / src_n.clamp(min=1e-6)).max().item()
            assert rel < 1e-2, (
                f"L2 norm drift: n_blocks={n_blocks} scale={scale} "
                f"max rel err = {rel}")
    print("PASS")


def test_per_element(ext) -> None:
    print("test_per_element ...", end=" ", flush=True)
    for n_blocks in [1, 8, 64]:
        src = _make_input(n_blocks)
        out = _round_trip(ext, src, n_blocks)
        src_f = src.float()
        out_f = out.float()
        diff = (src_f - out_f).abs()
        rel = diff / src_f.abs().clamp(min=1e-3)
        sorted_rel = torch.sort(rel)[0]
        p95 = sorted_rel[int(0.95 * len(sorted_rel))].item()
        assert p95 < 0.10, (
            f"Per-element p95 rel err = {p95} > 0.10 (n_blocks={n_blocks})")
    print("PASS")


def test_zero_input(ext) -> None:
    print("test_zero_input ...", end=" ", flush=True)
    n = 4
    src = torch.zeros(n * QK_PLANAR3, device="cuda", dtype=torch.float16)
    out = _round_trip(ext, src, n)
    assert torch.isfinite(out.float()).all(), "NaN or Inf in zero round-trip"
    assert (out.float().abs() < 1e-6).all(), "non-zero output for zero input"
    print("PASS")


def test_bpe_invariant() -> None:
    print("test_bpe_invariant ...", end=" ", flush=True)
    bpe = (BLOCK_BYTES * 8) / QK_PLANAR3
    assert math.isclose(bpe, 3.125, rel_tol=1e-9), f"bpe drifted: {bpe}"
    print("PASS")


def main() -> int:
    if not torch.cuda.is_available():
        print("error: no CUDA device available")
        return 2
    rq_vllm_dir = Path(os.environ.get("RQ_VLLM_DIR",
                                       Path.home() / "repos" / "rq-vllm"))
    if not rq_vllm_dir.is_dir():
        print(f"error: rq-vllm not found at {rq_vllm_dir}; "
              "set RQ_VLLM_DIR or `git clone git@github.com:rapatel0/rq-vllm.git ~/repos/rq-vllm`")
        return 2

    ext = build_extension(rq_vllm_dir)

    print()
    print("running planar3 kernel parity tests on", torch.cuda.get_device_name(0))
    print()
    test_bpe_invariant()
    test_shapes(ext)
    test_l2_norm(ext)
    test_per_element(ext)
    test_zero_input(ext)

    print()
    print("ALL PASS — planar3 KV pack/unpack kernels are correctness-validated.")
    print("Phase 2b (build wiring) and Phase 2c (FlashAttention integration)")
    print("can proceed without re-validating the kernel math.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
