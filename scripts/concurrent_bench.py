"""Concurrent throughput bench against a vLLM endpoint.

Submits N parallel /v1/completions requests (non-streaming, T=0), each
asking for `max_tokens` tokens. Measures aggregate tokens/sec — that
is, total completion tokens across all N requests divided by wall time
from first request submitted to last response received.

This is the metric vLLM is built to optimize: continuous batching means
serving N concurrent requests should produce a higher aggregate
throughput than N serial requests (up to the engine's max-concurrent
batch size).

Usage:
    python3 concurrent_bench.py <url> <model> <label> [N=8]
"""

from __future__ import annotations
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Write three sentences about Paris.\n",
    "List the first five US presidents:\n1.",
    "def fibonacci(n):\n    if n",
    "Photosynthesis is",
    "The four largest oceans are",
    "William Shakespeare was",
    "A binary search algorithm",
    "Mount Everest is the",
    "Water boils at",
    "The Roman Empire reached",
    "The speed of light is",
    "George Washington was",
    "A producer-consumer queue",
    "Recursion in software is",
]
MAX_TOKENS = 128


def hit(url: str, model: str, prompt: str) -> int:
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": MAX_TOKENS, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        resp = json.loads(r.read())
    return resp["usage"]["completion_tokens"]


def run_concurrent(url: str, model: str, n: int) -> tuple[float, int]:
    """Run n concurrent requests; return (wall_seconds, total_tokens)."""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
    t_start = time.perf_counter()
    total_tokens = 0
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(hit, url, model, p) for p in prompts]
        for fut in as_completed(futures):
            total_tokens += fut.result()
    wall = time.perf_counter() - t_start
    return wall, total_tokens


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit("usage: concurrent_bench.py <url> <model> <label> [N=8]")
    url, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    sweep = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 4, 8, 16]

    print(f"# {label}  url={url}  model={model}\n")
    # Warm up.
    print("warm-up ...", flush=True)
    hit(url, model, "Hello.")

    print()
    print(f"{'N':>4}  {'wall_s':>7}  {'tokens':>7}  {'agg_tps':>9}  {'per_req_tps':>11}")
    for n in sweep:
        wall, tokens = run_concurrent(url, model, n)
        agg = tokens / wall if wall > 0 else float("inf")
        per_req = agg / n if n > 0 else 0
        print(f"{n:>4}  {wall:>7.3f}  {tokens:>7d}  {agg:>9.2f}  {per_req:>11.2f}")


if __name__ == "__main__":
    main()
