#!/usr/bin/env python3
"""
Quantization throughput benchmark for TurboKVCache.

Measures per-token latency for quantize+store+dequantize on an A100.
Reports separately for prefill (batch of tokens) and decode (single token).

Usage:
    python scripts/benchmark_throughput.py
    python scripts/benchmark_throughput.py --bits 2.5 --warmup 100 --iters 1000
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from turboquant import PRESET_2_5BIT, PRESET_3_5BIT, TurboKVCache

PRESETS = {"2.5": PRESET_2_5BIT, "3.5": PRESET_3_5BIT}

# Qwen3.5-27B shapes
BATCH = 1
NUM_KV_HEADS = 8
HEAD_DIM = 128
NUM_LAYERS = 64


def benchmark_decode(config, iters: int = 1000, warmup: int = 100, device: str = "cuda"):
    """Benchmark single-token decode KV quantization."""
    cache = TurboKVCache(config)
    k = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(BATCH, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(warmup):
        for layer in range(4):  # just 4 layers for warmup
            cache.update(k, v, layer_idx=layer)
    cache = TurboKVCache(config)

    # Benchmark: simulate one full forward pass across all layers
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        cache = TurboKVCache(config)  # fresh cache per iter for fairness
        t0 = time.perf_counter()
        for layer in range(NUM_LAYERS):
            cache.update(k, v, layer_idx=layer)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    mean = sum(times) / len(times)
    return mean, p50, p95


def benchmark_prefill(config, seq_len: int = 512, iters: int = 50, device: str = "cuda"):
    """Benchmark prefill KV quantization (batch of tokens)."""
    k = torch.randn(BATCH, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(BATCH, NUM_KV_HEADS, seq_len, HEAD_DIM, dtype=torch.bfloat16, device=device)

    from turboquant.outlier import OutlierSplitter
    splitter = OutlierSplitter(config)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        qk, qv = splitter.quantize_kv(k, v)
        _, _ = splitter.dequantize_kv(qk, qv)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000 / seq_len)  # ms per token

    times.sort()
    return sum(times) / len(times), times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", choices=["2.5", "3.5"], default="3.5")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PRESETS[args.bits]

    print(f"TurboKVCache throughput benchmark ({args.bits}-bit, {device})")
    print(f"Model shape: {NUM_LAYERS} layers × {NUM_KV_HEADS} KV heads × {HEAD_DIM} head_dim\n")

    print(f"Decode benchmark ({args.iters} iters, 1 token per call)...")
    mean_dec, p50_dec, p95_dec = benchmark_decode(config, args.iters, args.warmup, device)
    print(f"  Full forward pass (all {NUM_LAYERS} layers):")
    print(f"    Mean: {mean_dec:.3f} ms  |  p50: {p50_dec:.3f} ms  |  p95: {p95_dec:.3f} ms")
    per_token = mean_dec
    print(f"  Per token (= per full forward): {per_token:.3f} ms", end="")
    if per_token < 1.0:
        print(f"  ✓ PASS (< 1ms target)")
    else:
        print(f"  ✗ FAIL (> 1ms target)")

    print(f"\nPrefill benchmark ({args.seq_len}-token batch)...")
    mean_pre, p50_pre = benchmark_prefill(config, args.seq_len, 50, device)
    print(f"  Per-token latency: mean={mean_pre:.4f} ms  |  p50={p50_pre:.4f} ms")


if __name__ == "__main__":
    main()
