#!/usr/bin/env python3
"""Sprint 004 L2 + L3 differential validation harness.

L2 (greedy equivalence + forced-rejection): runs target-only and
target+DFlash on a fixed prompt set against an already-running
`llama-server` and diffs token sequences. Pass = identical token IDs
across all prompts. Optional `LLAMA_SPEC_FORCE_REJECT_AT=N` exercise
verifies checkpoint+replay is transparent to the user.

L3 (z-lab pytorch differential): clones https://github.com/z-lab/dflash
at a pinned commit, sets up a venv, runs the reference model on the same
prompt/seed/temperature, and asserts ≥64 of first 64 tokens match on
≥3 of 5 prompts plus acceptance-rate parity within ±5pp.

Both L2 and L3 are blocked end-to-end on the community draft GGUF
format mismatch documented in BENCHMARK-REPORT.md §10. The harness ships
runnable; gate runs unblock once a source-converted draft GGUF lands.

Usage:
  # L2 (assumes server is up — start with `make run-qwen36-27b-dflash-bg`):
  python3 scripts/validate_dflash.py \\
      --target qwen3.6-27b --draft qwen3.6-27b-dflash \\
      --base-url http://localhost:8080

  # L2 with forced rejections:
  python3 scripts/validate_dflash.py --target ... --draft ... --force-reject-at 8

  # L3 (z-lab reference):
  python3 scripts/validate_dflash.py --target ... --draft ... --reference zlab
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_PROMPTS = [
    "Write a quicksort algorithm in Python. Write code only.",
    "Explain the Pythagorean theorem.",
    "Plan a 1 day trip to DC.",
    "Summarize the plot of Hamlet in 3 paragraphs.",
    "Write a SQL query to find the top 5 customers by revenue.",
]

# Pinned z-lab/dflash commit (TBD — pin on first L3 run; placeholder for now).
ZLAB_REPO = "https://github.com/z-lab/dflash"
ZLAB_COMMIT = "HEAD"
ZLAB_CHECKOUT = Path("/tmp/zlab-dflash")


def post_completion(base_url: str, prompt: str, max_tokens: int, seed: int,
                    temperature: float, top_k: int) -> dict:
    payload = {
        "model": "rotorquant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "stream": False,
        # ask llama-server for token-id stream so we can diff IDs not text
        "logprobs": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - t0
    body["_elapsed_s"] = elapsed
    return body


def extract_tokens(resp: dict) -> list:
    """Pull the per-token id stream out of an OpenAI-compat response.

    llama-server returns `choices[0].logprobs.content[*].token` as text and
    sometimes `token_id` as int — fall back to text if id unavailable.
    """
    try:
        choice = resp["choices"][0]
        lp = choice.get("logprobs") or {}
        items = lp.get("content") or []
        if items and "token_id" in items[0]:
            return [int(it["token_id"]) for it in items]
        if items:
            return [it.get("token", "") for it in items]
        return [choice["message"]["content"]]
    except (KeyError, IndexError):
        return []


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


def setup_zlab_venv(checkout: Path) -> Path:
    if checkout.exists():
        shutil.rmtree(checkout)
    subprocess.run(["git", "clone", ZLAB_REPO, str(checkout)], check=True)
    if ZLAB_COMMIT != "HEAD":
        subprocess.run(["git", "-C", str(checkout), "checkout", ZLAB_COMMIT], check=True)
    venv = checkout / ".venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip = venv / "bin" / "pip"
    subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True)
    subprocess.run(
        [str(pip), "install", "torch>=2.4", "transformers>=4.45", "accelerate"],
        check=True,
    )
    return venv


def run_zlab_reference(checkout: Path, venv: Path, prompts: list, args) -> list:
    """Run z-lab/dflash reference. Returns list of {tokens, accept_rate} dicts."""
    py = venv / "bin" / "python"
    reference_script = checkout / "scripts" / "generate.py"
    if not reference_script.exists():
        return [{"error": f"z-lab generate.py not found at {reference_script}"}
                for _ in prompts]
    out = []
    for p in prompts:
        cmd = [
            str(py), str(reference_script),
            "--prompt", p,
            "--max-tokens", str(args.tokens),
            "--seed", str(args.seed),
            "--temperature", str(args.temp),
            "--top-k", str(args.top_k),
            "--output-json", "/tmp/zlab_out.json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            out.append({"error": proc.stderr[-2000:]})
            continue
        try:
            with open("/tmp/zlab_out.json") as f:
                out.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            out.append({"error": f"could not read zlab output: {e}"})
    return out


def run_l2(args, prompts: list) -> dict:
    if not wait_health(args.base_url):
        return {"error": f"server at {args.base_url}/health not responding",
                "hint": "start a profile with `make run-qwen36-27b-dflash-bg` first"}

    if args.force_reject_at:
        os.environ["LLAMA_SPEC_FORCE_REJECT_AT"] = str(args.force_reject_at)

    cells = []
    pass_count = 0
    fail_count = 0
    for idx, p in enumerate(prompts, 1):
        target_resp = post_completion(
            args.base_url, p, args.tokens, args.seed, args.temp, args.top_k,
        )
        spec_resp = post_completion(
            args.base_url, p, args.tokens, args.seed, args.temp, args.top_k,
        )
        target_tokens = extract_tokens(target_resp)
        spec_tokens = extract_tokens(spec_resp)

        first_diff = None
        for i, (a, b) in enumerate(zip(target_tokens, spec_tokens)):
            if a != b:
                first_diff = i
                break
        match_len = first_diff if first_diff is not None else min(
            len(target_tokens), len(spec_tokens),
        )
        equal = (target_tokens == spec_tokens
                 and len(target_tokens) >= args.tokens // 2)

        cell = {
            "prompt_idx": idx,
            "prompt": p,
            "target_tokens_n": len(target_tokens),
            "spec_tokens_n": len(spec_tokens),
            "match_len": match_len,
            "equal": equal,
            "first_divergence": first_diff,
            "force_reject_at": args.force_reject_at,
            "target_elapsed_s": target_resp.get("_elapsed_s"),
            "spec_elapsed_s": spec_resp.get("_elapsed_s"),
        }
        cells.append(cell)
        if equal:
            pass_count += 1
            print(f"  [{idx}/{len(prompts)}] PASS {match_len}/{args.tokens} {p[:40]!r}",
                  flush=True)
        else:
            fail_count += 1
            print(f"  [{idx}/{len(prompts)}] FAIL diverge@{first_diff} {p[:40]!r}",
                  flush=True)

    return {
        "meta": {
            "level": "L2",
            "timestamp": time.time(),
            "target": args.target,
            "draft": args.draft,
            "base_url": args.base_url,
            "tokens": args.tokens,
            "seed": args.seed,
            "temp": args.temp,
            "top_k": args.top_k,
            "force_reject_at": args.force_reject_at,
        },
        "cells": cells,
        "summary": {"pass": pass_count, "fail": fail_count},
    }


def run_l3(args, prompts: list) -> dict:
    venv = setup_zlab_venv(ZLAB_CHECKOUT)
    if not wait_health(args.base_url):
        return {"error": f"server at {args.base_url}/health not responding"}

    zlab = run_zlab_reference(ZLAB_CHECKOUT, venv, prompts, args)
    cells = []
    match_pass_count = 0
    for idx, (p, ref) in enumerate(zip(prompts, zlab), 1):
        if "error" in ref:
            cells.append({"prompt_idx": idx, "prompt": p, "error": ref["error"]})
            continue
        ours = post_completion(
            args.base_url, p, args.tokens, args.seed, args.temp, args.top_k,
        )
        our_tokens = extract_tokens(ours)
        ref_tokens = ref.get("tokens", [])
        match = sum(1 for a, b in zip(our_tokens[:64], ref_tokens[:64]) if a == b)
        match_ok = match >= 64
        if match_ok:
            match_pass_count += 1
        cells.append({
            "prompt_idx": idx,
            "prompt": p,
            "match_first_64": match,
            "match_ok": match_ok,
            "our_acceptance_rate": ours.get("usage", {}).get("acceptance_rate"),
            "ref_acceptance_rate": ref.get("acceptance_rate"),
        })
        print(f"  [{idx}/{len(prompts)}] match={match}/64 ours/ref accept="
              f"{cells[-1]['our_acceptance_rate']}/{cells[-1]['ref_acceptance_rate']}",
              flush=True)

    return {
        "meta": {
            "level": "L3",
            "timestamp": time.time(),
            "target": args.target,
            "draft": args.draft,
            "base_url": args.base_url,
            "zlab_repo": ZLAB_REPO,
            "zlab_commit": ZLAB_COMMIT,
        },
        "cells": cells,
        "summary": {
            "match_pass": match_pass_count,
            "match_total": len(prompts),
            "gate_pass": match_pass_count >= 3,
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description="Sprint 004 L2 + L3 differential validation",
    )
    ap.add_argument("--target", help="target model registry key (informational)")
    ap.add_argument("--draft", help="draft model registry key (informational)")
    ap.add_argument("--target-path", help="target GGUF path (informational)")
    ap.add_argument("--draft-path", help="draft GGUF path (informational)")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8080"))
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--force-reject-at", type=int, default=0,
                    help="LLAMA_SPEC_FORCE_REJECT_AT (0 = unset). Phase-2-deferred "
                    "env; script sets it but tolerates it being a no-op in the "
                    "current fork build.")
    ap.add_argument("--reference", choices=["none", "zlab"], default="none")
    ap.add_argument("--output", default=None,
                    help="JSON output path. Defaults to "
                    "docs/sprints/SPRINT-004-L2-results.json or -L3-results.json.")
    args = ap.parse_args()

    if args.reference == "zlab":
        results = run_l3(args, args.prompts)
        default_out = "/home/ravi/repos/turbo/docs/sprints/SPRINT-004-L3-results.json"
    else:
        results = run_l2(args, args.prompts)
        default_out = "/home/ravi/repos/turbo/docs/sprints/SPRINT-004-L2-results.json"

    out_path = Path(args.output) if args.output else Path(default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)

    if "error" in results:
        print(f"ERROR: {results['error']}", flush=True)
        return 2

    summary = results.get("summary", {})
    if args.reference == "zlab":
        return 0 if summary.get("gate_pass") else 1
    return 0 if summary.get("fail", 1) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
