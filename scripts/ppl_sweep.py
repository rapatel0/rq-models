#!/usr/bin/env python3
"""Sprint 004 L1 PPL regression sweep.

Runs llama-perplexity for each (model, KV type, corpus) cell and
compares to BENCHMARK-REPORT.md baselines within ±0.05 PPL tolerance.

Outputs JSON with per-cell PPL, delta vs baseline, pass/fail.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

LLAMA_PPL = Path("/home/ravi/repos/llama-cpp-turboquant/build/bin/llama-perplexity")
WIKI = Path("/home/ravi/repos/turbo/models/wikitext2-test.txt")

MODELS = {
    "qwen3.6-27b": "/home/ravi/models/Qwen3.6-27B-UD-Q4_K_XL.gguf",
    "qwen3.6-35b": "/home/ravi/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
}

KV_TYPES = ["f16", "iso3", "iso4", "planar3", "planar4"]

# From BENCHMARK-REPORT.md §1.5–1.8 (pre-rebase baselines)
BASELINES = {
    ("qwen3.6-35b", "wikitext-2", "f16"):     6.1316,
    ("qwen3.6-35b", "wikitext-2", "iso4"):    6.2262,
    ("qwen3.6-35b", "wikitext-2", "iso3"):    6.2515,
    ("qwen3.6-35b", "wikitext-2", "planar4"): 6.2529,
    ("qwen3.6-35b", "wikitext-2", "planar3"): 6.2904,
    ("qwen3.6-27b", "wikitext-2", "f16"):     7.0901,
    ("qwen3.6-27b", "wikitext-2", "planar3"): 7.4044,
    ("qwen3.6-27b", "wikitext-2", "planar4"): 7.4703,
    ("qwen3.6-27b", "wikitext-2", "iso3"):    7.5587,
    ("qwen3.6-27b", "wikitext-2", "iso4"):    7.6405,
}

CORPORA = {"wikitext-2": WIKI}

TOLERANCE = 0.05


def run_ppl(model_path: Path, kv_type: str, corpus_path: Path, ctx: int = 2048) -> tuple[float, str]:
    """Run llama-perplexity and parse the Final estimate."""
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    cmd = [
        str(LLAMA_PPL),
        "-m", str(model_path),
        "-f", str(corpus_path),
        "-c", str(ctx),
        "-ngl", "99",
        "-fa", "1",
        "--cache-type-k", kv_type,
        "--cache-type-v", kv_type,
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return float("nan"), f"exit={proc.returncode}\n{proc.stderr[-1000:]}"

    # Parse "Final estimate: PPL = N.NNNN +/- M.MMMM" from stderr
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", proc.stderr)
    if not m:
        return float("nan"), f"no Final estimate in output\n{proc.stderr[-1000:]}"

    ppl = float(m.group(1))
    return ppl, f"elapsed={elapsed:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--kv-types", nargs="+", default=KV_TYPES)
    ap.add_argument("--corpora", nargs="+", default=list(CORPORA.keys()))
    ap.add_argument("--output", default="/home/ravi/repos/turbo/docs/sprints/SPRINT-004-L1-results.json")
    ap.add_argument("--ctx", type=int, default=2048)
    args = ap.parse_args()

    results = {"meta": {"timestamp": time.time(), "ctx": args.ctx, "tolerance": TOLERANCE}, "cells": []}
    pass_count = 0
    fail_count = 0
    total = len(args.models) * len(args.kv_types) * len(args.corpora)
    cell_idx = 0

    for model_name in args.models:
        model_path = Path(MODELS[model_name])
        if not model_path.exists():
            print(f"SKIP {model_name}: file not found at {model_path}", flush=True)
            continue
        for corpus_name in args.corpora:
            if corpus_name not in CORPORA:
                continue
            for kv_type in args.kv_types:
                cell_idx += 1
                key = (model_name, corpus_name, kv_type)
                baseline = BASELINES.get(key)
                print(f"[{cell_idx}/{total}] {model_name} / {corpus_name} / {kv_type} (baseline={baseline})", flush=True)
                ppl, info = run_ppl(model_path, kv_type, CORPORA[corpus_name], ctx=args.ctx)

                cell = {
                    "model": model_name,
                    "corpus": corpus_name,
                    "kv_type": kv_type,
                    "ppl": ppl,
                    "baseline": baseline,
                    "delta": (ppl - baseline) if baseline is not None and ppl == ppl else None,
                    "pass": (
                        baseline is not None
                        and ppl == ppl  # not NaN
                        and abs(ppl - baseline) <= TOLERANCE
                    ),
                    "info": info,
                }
                if cell["pass"]:
                    pass_count += 1
                    print(f"    PASS: ppl={ppl:.4f} delta={cell['delta']:+.4f} ({info})", flush=True)
                else:
                    fail_count += 1
                    print(f"    FAIL: ppl={ppl} baseline={baseline} ({info})", flush=True)

                results["cells"].append(cell)
                # Save incrementally so we can interrupt and inspect
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)

    results["summary"] = {"pass": pass_count, "fail": fail_count, "total": pass_count + fail_count}
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Summary: {pass_count} pass, {fail_count} fail ===", flush=True)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
