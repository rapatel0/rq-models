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

LLAMA_PPL_NEW = Path("/home/ravi/repos/llama-cpp-turboquant/build/bin/llama-perplexity")
WIKI = Path("/home/ravi/repos/turbo/models/wikitext2-raw-test.txt")

MODELS = {
    "qwen3.6-27b": "/home/ravi/models/Qwen3.6-27B-UD-Q4_K_XL.gguf",
    "qwen3.6-35b": "/home/ravi/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
}

KV_TYPES = ["f16", "iso3", "iso4", "planar3", "planar4"]

# Note: BENCHMARK-REPORT.md §1.5–1.8 baselines (e.g. 7.0901 for 27B/f16) were
# measured on a different wikitext-2 file than the canonical wikitext-2-raw-v1
# test split. The OLD fork (commit 20efe75) on the canonical dataset gives
# PPL = 8.0491 for 27B/f16, not 7.0901. We therefore treat the OLD fork (run
# via docker rotorquant:latest) as our canonical baseline for the L1 gate.
BASELINES = {}

CORPORA = {"wikitext-2": WIKI}

TOLERANCE = 0.05


def run_ppl_native(model_path: Path, kv_type: str, corpus_path: Path, ctx: int = 2048) -> tuple[float, str]:
    """Run rebased-fork llama-perplexity and parse the Final estimate."""
    env = os.environ.copy()
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    cmd = [
        str(LLAMA_PPL_NEW),
        "-m", str(model_path),
        "-f", str(corpus_path),
        "-c", str(ctx),
        "-ngl", "99",
        "-fa", "1",
        "--cache-type-k", kv_type,
        "--cache-type-v", kv_type,
    ]
    print(f"  [new] $ {LLAMA_PPL_NEW.name} (fork {kv_type})", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return float("nan"), f"exit={proc.returncode}\n{proc.stderr[-1000:]}"

    out = proc.stderr + proc.stdout
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", out)
    if not m:
        return float("nan"), f"no Final estimate\n{out[-800:]}"

    return float(m.group(1)), f"elapsed={elapsed:.0f}s"


def run_ppl_old(model_name: str, kv_type: str, ctx: int = 2048) -> tuple[float, str]:
    """Run OLD fork (rotorquant:latest docker image) llama-perplexity for baseline."""
    model_file = {
        "qwen3.6-27b": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        "qwen3.6-35b": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    }[model_name]

    cmd = [
        "docker", "run", "--rm",
        "--entrypoint", "bash",
        "-v", "llm-models:/models",
        "-v", "/home/ravi/repos/turbo/models:/host_models",
        "--gpus", "all",
        "rotorquant:latest", "-c",
        f"/app/bin/llama-perplexity -m /models/{model_file} -f /host_models/wikitext2-raw-test.txt "
        f"-c {ctx} -ngl 99 -fa 1 --cache-type-k {kv_type} --cache-type-v {kv_type}",
    ]
    print(f"  [old] $ docker rotorquant:latest (fork {kv_type})", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0

    out = proc.stderr + proc.stdout
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", out)
    if not m:
        return float("nan"), f"exit={proc.returncode}, no Final estimate\n{out[-800:]}"

    return float(m.group(1)), f"elapsed={elapsed:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--kv-types", nargs="+", default=KV_TYPES)
    ap.add_argument("--corpora", nargs="+", default=list(CORPORA.keys()))
    ap.add_argument("--output", default="/home/ravi/repos/turbo/docs/sprints/SPRINT-004-L1-results.json")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--mode", choices=["pair", "new", "old"], default="pair",
                    help="pair: run both forks per cell (slow but produces deltas); new: rebased only; old: docker baseline only")
    args = ap.parse_args()

    results = {
        "meta": {"timestamp": time.time(), "ctx": args.ctx, "tolerance": TOLERANCE, "mode": args.mode},
        "cells": [],
    }
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
                print(f"[{cell_idx}/{total}] {model_name} / {corpus_name} / {kv_type}", flush=True)

                ppl_new = info_new = ppl_old = info_old = None

                if args.mode in ("pair", "new"):
                    ppl_new, info_new = run_ppl_native(model_path, kv_type, CORPORA[corpus_name], ctx=args.ctx)
                    print(f"    new: ppl={ppl_new} ({info_new})", flush=True)
                if args.mode in ("pair", "old"):
                    ppl_old, info_old = run_ppl_old(model_name, kv_type, ctx=args.ctx)
                    print(f"    old: ppl={ppl_old} ({info_old})", flush=True)

                if args.mode == "pair":
                    delta = (ppl_new - ppl_old) if (ppl_new == ppl_new and ppl_old == ppl_old) else None
                    # L1 gate: rebased fork must not regress. Improvement (negative delta) passes;
                    # regression beyond tolerance fails. NaN fails.
                    passed = delta is not None and delta <= TOLERANCE
                    if passed:
                        pass_count += 1
                        verdict = "improvement" if delta < -TOLERANCE else "match"
                        print(f"    PASS ({verdict}): delta={delta:+.4f}", flush=True)
                    else:
                        fail_count += 1
                        print(f"    FAIL: delta={delta} (regression > +{TOLERANCE})", flush=True)
                else:
                    delta = None
                    passed = None

                cell = {
                    "model": model_name,
                    "corpus": corpus_name,
                    "kv_type": kv_type,
                    "ppl_new": ppl_new,
                    "info_new": info_new,
                    "ppl_old": ppl_old,
                    "info_old": info_old,
                    "delta": delta,
                    "pass": passed,
                }
                results["cells"].append(cell)
                with open(args.output, "w") as f:
                    json.dump(results, f, indent=2)

    if args.mode == "pair":
        results["summary"] = {"pass": pass_count, "fail": fail_count, "total": pass_count + fail_count}
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n=== Summary: {pass_count} pass, {fail_count} fail ===", flush=True)
        return 0 if fail_count == 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
