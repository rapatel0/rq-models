#!/usr/bin/env python3
"""Probe llama-server MTP draft acceptance and optional A/B speedup."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


DEFAULT_PROMPT = (
    "Write a concise technical explanation of why speculative decoding can "
    "improve autoregressive transformer throughput. Include three numbered "
    "points."
)


def post_completion(base_url: str, prompt: str, n_predict: int, temperature: float) -> dict:
    url = base_url.rstrip("/") + "/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.9,
        "cache_prompt": False,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to reach {url}: {exc}") from exc


def summarize(label: str, result: dict) -> dict:
    timings = result.get("timings") or {}
    predicted_tps = float(timings.get("predicted_per_second") or 0.0)
    predicted_n = int(timings.get("predicted_n") or result.get("tokens_predicted") or 0)
    draft_n = int(timings.get("draft_n") or 0)
    draft_n_accepted = int(timings.get("draft_n_accepted") or 0)
    acceptance = (draft_n_accepted / draft_n) if draft_n else 0.0

    spec_type = (
        (result.get("generation_settings") or {}).get("speculative.types")
        or "unknown"
    )

    print(
        f"{label}: {predicted_tps:.2f} tok/s, predicted={predicted_n}, "
        f"spec={spec_type}, drafts={draft_n_accepted}/{draft_n} "
        f"({acceptance:.1%})"
    )

    return {
        "predicted_tps": predicted_tps,
        "predicted_n": predicted_n,
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "acceptance": acceptance,
        "spec_type": spec_type,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a deterministic completion to llama-server and fail if MTP "
            "is not producing accepted draft tokens. Optionally compare to a "
            "non-MTP baseline URL."
        )
    )
    parser.add_argument("--mtp-url", default="http://localhost:8080")
    parser.add_argument("--base-url", help="optional non-MTP control server URL")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", help="read prompt text from this file")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--min-acceptance", type=float, default=0.50)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the final summary as JSON in addition to human output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            prompt = fh.read()

    try:
        mtp_result = post_completion(args.mtp_url, prompt, args.tokens, args.temperature)
        mtp = summarize("mtp", mtp_result)

        base = None
        speedup = None
        if args.base_url:
            base_result = post_completion(args.base_url, prompt, args.tokens, args.temperature)
            base = summarize("base", base_result)
            if base["predicted_tps"] > 0:
                speedup = mtp["predicted_tps"] / base["predicted_tps"]
                print(f"speedup: {speedup:.2f}x")

        failures = []
        if mtp["draft_n"] <= 0:
            failures.append("MTP generated zero draft tokens")
        if mtp["draft_n_accepted"] <= 0:
            failures.append("MTP accepted zero draft tokens")
        if mtp["acceptance"] < args.min_acceptance:
            failures.append(
                f"MTP acceptance {mtp['acceptance']:.1%} < {args.min_acceptance:.1%}"
            )
        if speedup is not None and speedup < args.min_speedup:
            failures.append(f"MTP speedup {speedup:.2f}x < {args.min_speedup:.2f}x")

        summary = {
            "mtp": mtp,
            "base": base,
            "speedup": speedup,
            "ok": not failures,
            "failures": failures,
        }
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))

        if failures:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
            return 1

        print("PASS: MTP is generating and accepting draft tokens")
        return 0
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
