#!/usr/bin/env python3
"""Benchmark aggregate throughput across N concurrent chat requests.

Targets any OpenAI-compatible chat completion server (llama.cpp's `--api`,
vLLM's OpenAI server, oMLX, etc.). Drives concurrent decode lanes, reports
aggregate and per-session tok/s, and snapshots `/health` before and after
each N lane for state diagnostics.

Originally written for vortex Sprint 015 multi-tenant slot validation;
ported into rq-models for cross-substrate benchmarking. Use this to
compare:
- llama.cpp (current rq-models substrate) at N=1, 2, 4, 8
- vLLM (Sprint 004+ substrate) at N=1, 2, 4, 8, 16
under matched workload.

Original behaviors preserved:
- optional per-session warmup (for cached-prefix decode lanes)
- concurrent request fanout for N in {1,2,4,...}
- aggregate token/s + per-request token/s
- health snapshots before and after each lane
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class RequestResult:
    session_id: str
    ok: bool
    elapsed_s: float
    completion_tokens: int
    tok_s: float
    error: str | None


def _http_json(url: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _health(url: str, timeout_s: int) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _chat_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int | None,
    session_id: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": False,
        "session_id": session_id,
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


def _run_one(
    chat_url: str,
    timeout_s: int,
    payload: dict[str, Any],
    session_id: str,
) -> RequestResult:
    t0 = time.time()
    try:
        out = _http_json(chat_url, payload, timeout_s)
        dt = max(0.0, time.time() - t0)
        usage = out.get("usage", {}) if isinstance(out, dict) else {}
        completion_tokens = int(usage.get("completion_tokens", 0))
        tok_s = completion_tokens / dt if dt > 0.0 else 0.0
        return RequestResult(
            session_id=session_id,
            ok=True,
            elapsed_s=dt,
            completion_tokens=completion_tokens,
            tok_s=tok_s,
            error=None,
        )
    except urllib.error.HTTPError as e:
        dt = max(0.0, time.time() - t0)
        err = f"http_{e.code}"
        try:
            err_body = e.read().decode("utf-8")
            if err_body:
                err = f"{err}: {err_body[:300]}"
        except Exception:
            pass
        return RequestResult(
            session_id=session_id,
            ok=False,
            elapsed_s=dt,
            completion_tokens=0,
            tok_s=0.0,
            error=err,
        )
    except Exception as e:  # noqa: BLE001
        dt = max(0.0, time.time() - t0)
        return RequestResult(
            session_id=session_id,
            ok=False,
            elapsed_s=dt,
            completion_tokens=0,
            tok_s=0.0,
            error=str(e),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chat-url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--health-url", default="http://127.0.0.1:8080/health")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-path", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup-tokens", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--lanes", default="1,2")
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = Path(args.prompt_path).read_text(encoding="utf-8")
    lane_values = [int(x.strip()) for x in args.lanes.split(",") if x.strip()]
    lane_values = [n for n in lane_values if n >= 1]
    if not lane_values:
        raise SystemExit("no valid lane values")

    artifact: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": vars(args),
        "health_before_all": _health(args.health_url, timeout_s=5),
        "lanes": {},
    }

    for n in lane_values:
        lane_key = str(n)
        sessions = [f"s15-n{n}-slot{i}" for i in range(n)]
        lane_entry: dict[str, Any] = {
            "sessions": sessions,
            "warmup_tokens": args.warmup_tokens,
            "repeats": args.repeats,
            "health_before_lane": _health(args.health_url, timeout_s=5),
            "runs": [],
        }

        # Warm each slot session once.
        if args.warmup_tokens > 0:
            for sid in sessions:
                payload = _chat_payload(
                    args.model,
                    prompt,
                    args.warmup_tokens,
                    args.temperature,
                    args.seed,
                    sid,
                )
                lane_entry.setdefault("warmups", []).append(
                    asdict(_run_one(args.chat_url, args.timeout_s, payload, sid))
                )

        for rep in range(args.repeats):
            run_payloads = {
                sid: _chat_payload(
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    args.seed,
                    sid,
                )
                for sid in sessions
            }
            barrier = threading.Barrier(n)
            results: list[RequestResult | None] = [None] * n

            def worker(idx: int, sid: str) -> None:
                barrier.wait()
                results[idx] = _run_one(
                    args.chat_url, args.timeout_s, run_payloads[sid], sid
                )

            threads = [
                threading.Thread(target=worker, args=(i, sid), daemon=True)
                for i, sid in enumerate(sessions)
            ]
            t0 = time.time()
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            wall_s = max(0.0, time.time() - t0)
            rr = [r for r in results if r is not None]
            ok = all(r.ok for r in rr)
            aggregate_tokens = sum(r.completion_tokens for r in rr)
            aggregate_tok_s = aggregate_tokens / wall_s if wall_s > 0.0 else 0.0
            lane_entry["runs"].append(
                {
                    "rep": rep + 1,
                    "ok": ok,
                    "wall_s": wall_s,
                    "aggregate_tokens": aggregate_tokens,
                    "aggregate_tok_s": aggregate_tok_s,
                    "requests": [asdict(r) for r in rr],
                }
            )

        lane_entry["health_after_lane"] = _health(args.health_url, timeout_s=5)
        valid = [r["aggregate_tok_s"] for r in lane_entry["runs"] if r["ok"]]
        lane_entry["mean_aggregate_tok_s"] = (
            sum(valid) / len(valid) if valid else 0.0
        )
        artifact["lanes"][lane_key] = lane_entry

    artifact["health_after_all"] = _health(args.health_url, timeout_s=5)
    if "1" in artifact["lanes"] and "2" in artifact["lanes"]:
        n1 = artifact["lanes"]["1"]["mean_aggregate_tok_s"]
        n2 = artifact["lanes"]["2"]["mean_aggregate_tok_s"]
        artifact["n2_over_n1"] = (n2 / n1) if n1 > 0 else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(out_path), "n2_over_n1": artifact.get("n2_over_n1")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
