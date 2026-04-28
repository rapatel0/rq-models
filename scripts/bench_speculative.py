#!/usr/bin/env python3
"""Sprint 005 L4 speedup benchmark.

3-way decode-tok/s comparison: target-only vs target+autoregressive-draft
vs target+DFlash. Caller is responsible for starting the right compose
profile before each leg; the script just hits the OpenAI-compat endpoint
and parses the `usage` block.

The 3 legs are run sequentially against the same `--base-url`. Each leg
expects the operator to have started the corresponding compose profile
(or pointed at three distinct ports). Without a working DFlash draft
GGUF the third leg will return server errors — the harness records and
moves on; gate evaluation only fires once all three legs have data.

L4 hard gate: median of per-prompt tok/s ratios (DFlash / target-only)
across the 5 prompts is ≥ 1.3× on Qwen3.6-27B + DFlash (`qwen`) with
greedy sampling and seed=42.

Headline (not gated): quicksort prompt ≥ 1.5× on Qwen3.6-27B.

Usage:
  # Start the target-only profile, run leg 1, stop, repeat for the others.
  make run-qwen36-27b-bg
  python3 scripts/bench_speculative.py --leg target-only
  make stop
  make run-qwen36-27b-bg SPECULATIVE_MODE=autoregressive DRAFT_MODEL_NAME=...
  python3 scripts/bench_speculative.py --leg autoregressive
  make stop
  make run-qwen36-27b-dflash-bg
  python3 scripts/bench_speculative.py --leg dflash
  python3 scripts/bench_speculative.py --finalize  # write summary table
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = [
    "Write a quicksort algorithm in Python. Write code only.",
    "Explain the Pythagorean theorem.",
    "Plan a 1 day trip to DC.",
    "Summarize the plot of Hamlet in 3 paragraphs.",
    "Write a SQL query to find the top 5 customers by revenue.",
]

LEGS = ["target-only", "autoregressive", "dflash"]


def default_output_path(profile: str) -> Path:
    return Path(
        f"/home/ravi/repos/turbo/docs/sprints/SPRINT-005-L4-results-{profile}.json",
    )


def default_md_path(profile: str) -> Path:
    return Path(
        f"/home/ravi/repos/turbo/docs/sprints/SPRINT-005-L4-summary-{profile}.md",
    )


def post_completion(base_url: str, prompt: str, max_tokens: int, seed: int,
                    temperature: float, top_k: int, timeout_s: int) -> dict:
    payload = {
        "model": "rotorquant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "stream": False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode())
    body["_wallclock_s"] = time.perf_counter() - t0
    return body


def wait_health(base_url: str, timeout_s: int = 5) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    return False


def parse_decode_tps(resp: dict) -> float:
    """Per-leg decode tok/s. Prefer llama-server's `timings.predicted_per_second`,
    fall back to wallclock + completion_tokens."""
    timings = resp.get("timings") or {}
    if timings.get("predicted_per_second"):
        return float(timings["predicted_per_second"])
    usage = resp.get("usage") or {}
    completion = usage.get("completion_tokens") or 0
    wallclock = resp.get("_wallclock_s") or 0
    if completion and wallclock > 0:
        return completion / wallclock
    return float("nan")


def parse_draft_stats(resp: dict) -> dict:
    timings = resp.get("timings") or {}
    usage = resp.get("usage") or {}

    def _to_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _to_float(value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    draft_n = _to_int(timings.get("draft_n"))
    if draft_n is None:
        draft_n = _to_int(usage.get("draft_n"))

    draft_n_accepted = _to_int(timings.get("draft_n_accepted"))
    if draft_n_accepted is None:
        draft_n_accepted = _to_int(usage.get("draft_n_accepted"))

    acceptance_rate = _to_float(usage.get("acceptance_rate"))
    if acceptance_rate is None and draft_n and draft_n_accepted is not None and draft_n > 0:
        acceptance_rate = draft_n_accepted / draft_n

    return {
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "acceptance_rate": acceptance_rate,
    }


def run_leg(args, leg: str) -> dict:
    if not wait_health(args.base_url):
        return {"error": f"server at {args.base_url}/health not responding"}

    cells = []
    for idx, p in enumerate(PROMPTS, 1):
        trials = []
        for trial in range(args.trials):
            last_error = None
            for attempt in range(args.request_retries + 1):
                if not wait_health(args.base_url, timeout_s=args.health_timeout_s):
                    last_error = "server not healthy before request"
                    continue
                try:
                    resp = post_completion(
                        args.base_url, p, args.tokens, args.seed,
                        args.temp, args.top_k, args.request_timeout_s,
                    )
                    tps = parse_decode_tps(resp)
                    draft_stats = parse_draft_stats(resp)
                    trials.append({
                        "trial": trial,
                        "attempt": attempt,
                        "tps": tps,
                        "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
                        "wallclock_s": resp.get("_wallclock_s"),
                        "draft_n": draft_stats["draft_n"],
                        "draft_n_accepted": draft_stats["draft_n_accepted"],
                        "acceptance_rate": draft_stats["acceptance_rate"],
                    })
                    acc = draft_stats["acceptance_rate"]
                    if acc is None:
                        print(
                            f"  [{leg}][{idx}/{len(PROMPTS)}][trial {trial}] "
                            f"{tps:.2f} tok/s",
                            flush=True,
                        )
                    else:
                        print(
                            f"  [{leg}][{idx}/{len(PROMPTS)}][trial {trial}] "
                            f"{tps:.2f} tok/s acc={acc * 100.0:.2f}%",
                            flush=True,
                        )
                    last_error = None
                    break
                except urllib.error.HTTPError as e:
                    body = e.read()
                    last_error = f"http {e.code}: {body!r}"
                    if e.code < 500:
                        break
                    if attempt < args.request_retries:
                        print(
                            f"  [{leg}][{idx}/{len(PROMPTS)}][trial {trial}] "
                            f"retry after HTTP {e.code} (attempt {attempt + 1}/{args.request_retries})",
                            flush=True,
                        )
                        time.sleep(1.0)
                except (urllib.error.URLError, OSError) as e:
                    last_error = str(e)
                    if attempt < args.request_retries:
                        print(
                            f"  [{leg}][{idx}/{len(PROMPTS)}][trial {trial}] "
                            f"retry after transport error (attempt {attempt + 1}/{args.request_retries})",
                            flush=True,
                        )
                        time.sleep(1.0)
            if last_error is not None:
                trials.append({"trial": trial, "error": last_error})

        valid = [t["tps"] for t in trials if isinstance(t.get("tps"), float)
                 and t["tps"] == t["tps"]]
        draft_n = sum(
            int(t["draft_n"]) for t in trials
            if isinstance(t.get("draft_n"), int) and t["draft_n"] > 0
        )
        draft_n_accepted = sum(
            int(t["draft_n_accepted"]) for t in trials
            if isinstance(t.get("draft_n"), int)
            and t["draft_n"] > 0
            and isinstance(t.get("draft_n_accepted"), int)
        )
        median = statistics.median(valid) if valid else float("nan")
        acceptance_rate = (
            draft_n_accepted / draft_n
            if draft_n > 0
            else float("nan")
        )
        cells.append({
            "prompt_idx": idx,
            "prompt": p,
            "trials": trials,
            "median_tps": median,
            "sum_draft_n": draft_n,
            "sum_draft_n_accepted": draft_n_accepted,
            "acceptance_rate": acceptance_rate,
        })

    return {
        "leg": leg,
        "meta": {
            "timestamp": time.time(),
            "base_url": args.base_url,
            "tokens": args.tokens,
            "trials": args.trials,
            "seed": args.seed,
            "temp": args.temp,
            "top_k": args.top_k,
        },
        "cells": cells,
    }


def load_results(path: Path) -> dict:
    if not path.exists():
        return {"legs": {}}
    with open(path) as f:
        return json.load(f)


def save_results(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def finalize(out: Path, md: Path, profile: str) -> int:
    data = load_results(out)
    legs = data.get("legs", {})
    missing = [leg for leg in LEGS if leg not in legs]
    if missing:
        print(f"ERROR: missing legs: {missing}", flush=True)
        return 2

    rows = []
    target_only = legs["target-only"]["cells"]
    autoreg = legs["autoregressive"]["cells"]
    dflash = legs["dflash"]["cells"]
    ratios_dflash = []
    ratios_autoreg = []
    for i, cell in enumerate(target_only):
        base = cell["median_tps"]
        ar_tps = autoreg[i]["median_tps"]
        df_tps = dflash[i]["median_tps"]
        r_ar = ar_tps / base if base and base == base else float("nan")
        r_df = df_tps / base if base and base == base else float("nan")
        ratios_autoreg.append(r_ar)
        ratios_dflash.append(r_df)
        rows.append({
            "prompt": cell["prompt"],
            "target_only_tps": base,
            "autoregressive_tps": ar_tps,
            "dflash_tps": df_tps,
            "ratio_autoreg": r_ar,
            "ratio_dflash": r_df,
            "target_only_acceptance_rate": target_only[i].get("acceptance_rate"),
            "autoregressive_acceptance_rate": autoreg[i].get("acceptance_rate"),
            "dflash_acceptance_rate": dflash[i].get("acceptance_rate"),
        })

    valid_df = [r for r in ratios_dflash if r == r]
    median_ratio = statistics.median(valid_df) if valid_df else float("nan")
    quicksort_ratio = ratios_dflash[0] if ratios_dflash else float("nan")
    gate_pass = median_ratio >= 1.3
    headline_pass = quicksort_ratio >= 1.5
    leg_acceptance = {}
    for leg in LEGS:
        leg_cells = legs[leg]["cells"]
        total_draft_n = sum(int(c.get("sum_draft_n", 0)) for c in leg_cells)
        total_draft_n_accepted = sum(int(c.get("sum_draft_n_accepted", 0)) for c in leg_cells)
        leg_acceptance[leg] = (
            total_draft_n_accepted / total_draft_n
            if total_draft_n > 0
            else float("nan")
        )

    data["summary"] = {
        "profile": profile,
        "median_ratio_dflash": median_ratio,
        "quicksort_ratio_dflash": quicksort_ratio,
        "gate_pass_median_ge_1_3": gate_pass,
        "headline_quicksort_ge_1_5": headline_pass,
        "acceptance_rate_by_leg": leg_acceptance,
        "rows": rows,
    }
    save_results(out, data)

    md.parent.mkdir(parents=True, exist_ok=True)
    def fmt(v: float) -> str:
        return f"{v:.2f}" if isinstance(v, float) and v == v else "n/a"

    with open(md, "w") as f:
        f.write(f"# Sprint 005 L4 — 3-way decode tok/s ({profile})\n\n")
        f.write("| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['prompt'][:48]!r} | {r['target_only_tps']:.2f} | "
                f"{r['autoregressive_tps']:.2f} | {r['dflash_tps']:.2f} | "
                f"{r['ratio_autoreg']:.2f} | {r['ratio_dflash']:.2f} | "
                f"{fmt(r['autoregressive_acceptance_rate'] * 100.0) if isinstance(r['autoregressive_acceptance_rate'], float) and r['autoregressive_acceptance_rate'] == r['autoregressive_acceptance_rate'] else 'n/a'}% | "
                f"{fmt(r['dflash_acceptance_rate'] * 100.0) if isinstance(r['dflash_acceptance_rate'], float) and r['dflash_acceptance_rate'] == r['dflash_acceptance_rate'] else 'n/a'}% |\n"
            )
        f.write(f"\n**Median DFlash×**: {median_ratio:.2f} "
                f"(gate ≥1.3 → {'PASS' if gate_pass else 'FAIL'})\n\n")
        f.write(f"**Quicksort headline DFlash×**: {quicksort_ratio:.2f} "
                f"(headline ≥1.5 → {'PASS' if headline_pass else 'FAIL'})\n")
        f.write(
            "\n**Leg acceptance rates**: "
            f"target-only={fmt(leg_acceptance['target-only'] * 100.0)}%, "
            f"autoregressive={fmt(leg_acceptance['autoregressive'] * 100.0)}%, "
            f"dflash={fmt(leg_acceptance['dflash'] * 100.0)}%\n"
        )
    print(f"\nWrote {out}\nWrote {md}", flush=True)
    return 0 if gate_pass else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=LEGS,
                    help="run a single leg against the currently-running server")
    ap.add_argument("--finalize", action="store_true",
                    help="compute ratios + emit summary table from existing legs")
    ap.add_argument("--base-url",
                    default=os.environ.get("BASE_URL", "http://localhost:8080"))
    ap.add_argument("--profile", choices=["qwen", "qwen36"],
                    default=os.environ.get("PROFILE", "qwen"))
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--request-retries", type=int, default=3,
                    help="retries per completion request after transport/5xx errors")
    ap.add_argument("--health-timeout-s", type=int, default=30,
                    help="health poll timeout before each request")
    ap.add_argument("--request-timeout-s", type=int, default=120,
                    help="timeout for each completion request")
    ap.add_argument("--output", default=None)
    ap.add_argument("--md-output", default=None)
    args = ap.parse_args()

    out = Path(args.output) if args.output else default_output_path(args.profile)
    md = Path(args.md_output) if args.md_output else default_md_path(args.profile)

    if args.finalize:
        return finalize(out, md, args.profile)

    if not args.leg:
        ap.error("--leg or --finalize required")

    leg_data = run_leg(args, args.leg)
    if "error" in leg_data:
        print(f"ERROR: {leg_data['error']}", flush=True)
        return 2

    data = load_results(out)
    data.setdefault("legs", {})[args.leg] = leg_data
    save_results(out, data)
    print(f"\nWrote leg '{args.leg}' to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
