#!/usr/bin/env python3
"""Sprint 005 DFlash experiment sweep driver.

Runs one-prompt (quicksort) x 3-trial measurements on the `qwen` profile for:
- target quant (Q4_K_XL / Q5_K_M / Q8_0)
- draft KV cache type (planar3 / iso3 / Q8_0 / f16)
- draft-max (8 / 16 / 24)
- N_PARALLEL (1 / 2 / 4)

Results are resumable and written to docs/sprints/SPRINT-005-experiments.json.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = "Write a quicksort algorithm in Python. Write code only."

DEFAULT_OUTPUT = Path("/home/ravi/repos/turbo/docs/sprints/SPRINT-005-experiments.json")

# Ordered definitions for reproducible output.
SWEEPS = {
    "target_weight_quant": {
        "q4_k_xl": {},
        "q5_k_m": {
            "EXTRA_ARGS": "--model /models/Qwen3.6-27B-UD-Q5_K_M.gguf",
        },
        "q8_0": {
            "EXTRA_ARGS": "--model /models/Qwen3.6-27B-UD-Q8_0.gguf",
        },
    },
    "draft_kv_cache_type": {
        "planar3": {"DRAFT_KV_CACHE_TYPE": "planar3"},
        "iso3": {"DRAFT_KV_CACHE_TYPE": "iso3"},
        "q8_0": {"DRAFT_KV_CACHE_TYPE": "q8_0"},
        "f16": {"DRAFT_KV_CACHE_TYPE": "f16"},
    },
    "draft_max": {
        "8": {"DRAFT_N_MAX": "8"},
        "16": {"DRAFT_N_MAX": "16"},
        "24": {"DRAFT_N_MAX": "24"},
    },
    "n_parallel": {
        "1": {"N_PARALLEL": "1"},
        "2": {"N_PARALLEL": "2"},
        "4": {"N_PARALLEL": "4"},
    },
}

BASE_OVERRIDES = {
    "SPECULATIVE_MODE": "dflash",
    "DRAFT_MODEL_NAME": "qwen3.6-27b-dflash",
    "KV_CACHE_TYPE": "planar3",
    "DRAFT_KV_CACHE_TYPE": "planar3",
    "DRAFT_N_MAX": "16",
    "N_PARALLEL": "1",
    "EXTRA_ARGS": "",
}


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, text=True, capture_output=True, check=check)


def wait_health(base_url: str, timeout_s: int = 240) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    return False


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


def parse_decode_tps(resp: dict) -> float:
    timings = resp.get("timings") or {}
    if timings.get("predicted_per_second"):
        return float(timings["predicted_per_second"])
    usage = resp.get("usage") or {}
    completion = usage.get("completion_tokens") or 0
    wallclock = resp.get("_wallclock_s") or 0
    if completion and wallclock > 0:
        return completion / wallclock
    return float("nan")


def parse_draft_stats(resp: dict) -> tuple[int | None, int | None, float | None]:
    timings = resp.get("timings") or {}
    usage = resp.get("usage") or {}

    def _to_int(v):
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _to_float(v):
        if v is None:
            return None
        try:
            return float(v)
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

    return draft_n, draft_n_accepted, acceptance_rate


def load_results(path: Path) -> dict:
    if not path.exists():
        return {
            "meta": {},
            "sweeps": {},
        }
    with open(path) as f:
        return json.load(f)


def save_results(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_one_level(args, sweep_name: str, level_name: str, level_overrides: dict[str, str]) -> dict:
    env = os.environ.copy()
    env.update(BASE_OVERRIDES)
    env.update(level_overrides)
    env["PORT"] = str(args.port)

    level_result = {
        "sweep": sweep_name,
        "level": level_name,
        "env_overrides": level_overrides,
        "trials": [],
        "timestamp": time.time(),
    }

    up_ok = False
    try:
        run_cmd(["make", "stop"], check=False)
        up = run_cmd(["docker", "compose", "--profile", "qwen", "up", "-d"], env=env, check=False)
        if up.returncode != 0:
            level_result["error"] = "docker compose up failed"
            level_result["stderr_tail"] = up.stderr[-4000:]
            return level_result
        up_ok = True

        if not wait_health(args.base_url):
            level_result["error"] = f"server at {args.base_url}/health not ready"
            return level_result

        for trial in range(args.trials):
            last_error = None
            for attempt in range(args.request_retries + 1):
                if not wait_health(args.base_url, timeout_s=args.health_timeout_s):
                    last_error = "server not healthy before request"
                    continue
                try:
                    resp = post_completion(
                        args.base_url,
                        PROMPT,
                        args.tokens,
                        args.seed,
                        args.temp,
                        args.top_k,
                        args.request_timeout_s,
                    )
                    tps = parse_decode_tps(resp)
                    draft_n, draft_n_accepted, acceptance_rate = parse_draft_stats(resp)
                    trial_row = {
                        "trial": trial,
                        "attempt": attempt,
                        "tps": tps,
                        "draft_n": draft_n,
                        "draft_n_accepted": draft_n_accepted,
                        "acceptance_rate": acceptance_rate,
                        "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
                        "wallclock_s": resp.get("_wallclock_s"),
                    }
                    level_result["trials"].append(trial_row)
                    if acceptance_rate is None:
                        print(f"[{sweep_name}/{level_name}][trial {trial}] {tps:.2f} tok/s", flush=True)
                    else:
                        print(
                            f"[{sweep_name}/{level_name}][trial {trial}] "
                            f"{tps:.2f} tok/s acc={acceptance_rate * 100.0:.2f}%",
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
                            f"[{sweep_name}/{level_name}][trial {trial}] "
                            f"retry after HTTP {e.code} (attempt {attempt + 1}/{args.request_retries})",
                            flush=True,
                        )
                        time.sleep(1.0)
                except (urllib.error.URLError, OSError) as e:
                    last_error = str(e)
                    if attempt < args.request_retries:
                        print(
                            f"[{sweep_name}/{level_name}][trial {trial}] "
                            f"retry after transport error (attempt {attempt + 1}/{args.request_retries})",
                            flush=True,
                        )
                        time.sleep(1.0)
            if last_error is not None:
                level_result["trials"].append({"trial": trial, "error": last_error})

        valid_tps = [
            t["tps"] for t in level_result["trials"]
            if isinstance(t.get("tps"), float) and t["tps"] == t["tps"]
        ]
        level_result["median_tps"] = statistics.median(valid_tps) if valid_tps else float("nan")
        level_result["mean_tps"] = statistics.mean(valid_tps) if valid_tps else float("nan")

        total_draft_n = sum(
            int(t["draft_n"]) for t in level_result["trials"]
            if isinstance(t.get("draft_n"), int) and t["draft_n"] > 0
        )
        total_draft_n_accepted = sum(
            int(t["draft_n_accepted"]) for t in level_result["trials"]
            if isinstance(t.get("draft_n"), int)
            and t["draft_n"] > 0
            and isinstance(t.get("draft_n_accepted"), int)
        )
        level_result["draft_n_total"] = total_draft_n
        level_result["draft_n_accepted_total"] = total_draft_n_accepted
        level_result["acceptance_rate"] = (
            total_draft_n_accepted / total_draft_n
            if total_draft_n > 0
            else float("nan")
        )

        return level_result
    finally:
        if up_ok:
            run_cmd(["make", "stop"], check=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8080"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
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
    ap.add_argument("--only-sweep", choices=list(SWEEPS.keys()), default=None)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    data = load_results(out)
    data["meta"] = {
        "timestamp": time.time(),
        "base_url": args.base_url,
        "port": args.port,
        "prompt": PROMPT,
        "trials": args.trials,
        "tokens": args.tokens,
        "seed": args.seed,
        "temp": args.temp,
        "top_k": args.top_k,
        "profile": "qwen",
        "mode": "dflash",
    }
    data.setdefault("sweeps", {})

    sweep_names = [args.only_sweep] if args.only_sweep else list(SWEEPS.keys())

    for sweep_name in sweep_names:
        data["sweeps"].setdefault(sweep_name, {})
        for level_name, overrides in SWEEPS[sweep_name].items():
            if level_name in data["sweeps"][sweep_name] and not args.rerun:
                print(f"[skip] {sweep_name}/{level_name} already recorded", flush=True)
                continue
            print(f"[run]  {sweep_name}/{level_name}", flush=True)
            result = run_one_level(args, sweep_name, level_name, overrides)
            data["sweeps"][sweep_name][level_name] = result
            save_results(out, data)

    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
