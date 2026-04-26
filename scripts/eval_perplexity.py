"""Perplexity eval against a vLLM /v1/completions endpoint.

Submits each text in TEXTS with ``echo=True, prompt_logprobs=1,
max_tokens=1`` and reads the per-token logprobs vLLM returns for the
prompt. From those we compute the average negative log-likelihood and
report perplexity = exp(mean_nll).

Phase 3 sprint gate: |ppl_rq3 - ppl_fp16| / ppl_fp16 ≤ 0.05%. Run this
script once per container and diff the numbers.

Usage:
    python3 eval_perplexity.py <url> <model> [<label>]
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request


# Five paragraphs of plain English from public-domain sources (US presidents,
# basic geography, simple math, code commentary). Total ≈ 1500 chars / a
# few hundred tokens — enough to be representative without taking forever
# to score. We deliberately avoid model-specific output styles
# (no "<think>", no system prompts) to make the result comparable to a
# llama.cpp planar3 run later.
TEXTS = [
    "George Washington was the first president of the United States. "
    "He was inaugurated in seventeen eighty-nine and served two terms. "
    "John Adams succeeded him in seventeen ninety-seven, followed by "
    "Thomas Jefferson, James Madison, and James Monroe.",

    "The Pacific Ocean is the largest ocean on Earth, covering about "
    "one hundred sixty-five million square kilometers. The Atlantic "
    "Ocean is the second largest, followed by the Indian Ocean. The "
    "Arctic Ocean is the smallest of the four traditional oceans.",

    "Paris is the capital of France and one of the most visited cities "
    "in the world. It is famous for the Eiffel Tower, the Louvre Museum, "
    "and Notre Dame Cathedral. The Seine River runs through the heart "
    "of the city, dividing it into the Right Bank and the Left Bank.",

    "A recursive Fibonacci function returns zero when n is zero, returns "
    "one when n is one, and otherwise returns the sum of the previous two "
    "Fibonacci numbers. The naive recursive implementation has exponential "
    "time complexity, so memoization or an iterative loop is preferred.",

    "Two plus two equals four. Four plus four equals eight. Eight plus "
    "eight equals sixteen. Each step doubles the previous value. This "
    "simple sequence illustrates how repeated addition can be used to "
    "express multiplication by powers of two.",
]


def score(url: str, model: str, text: str) -> tuple[float, int]:
    """POST text and read prompt_logprobs. Returns (sum_nll, n_tokens)."""
    body = json.dumps({
        "model": model, "prompt": text,
        "max_tokens": 1, "temperature": 0.0,
        "echo": True, "prompt_logprobs": 1, "logprobs": 1,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    choice = resp["choices"][0]
    pl = choice.get("prompt_logprobs")
    if pl is None:
        raise RuntimeError(f"server did not return prompt_logprobs; resp keys: {list(choice)}")
    sum_nll = 0.0
    n = 0
    # First entry is null (no logprob for the very first token). Each
    # entry is a dict {token_id_str: {"logprob": float, "rank": ..., "decoded_token": ...}}
    # We sum the logprob for the actual chosen token at each position.
    for entry in pl:
        if entry is None:
            continue
        # entry is a dict, possibly with multiple top candidates; the
        # actual prompt token is the one with rank 1 (or the only entry).
        chosen = None
        for tid, info in entry.items():
            if isinstance(info, dict) and info.get("rank") == 1:
                chosen = info
                break
        if chosen is None:
            # Fallback: take first entry.
            _, info = next(iter(entry.items()))
            chosen = info if isinstance(info, dict) else {"logprob": info}
        lp = chosen["logprob"]
        if lp is None or not math.isfinite(lp):
            continue
        sum_nll += -lp
        n += 1
    return sum_nll, n


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit("usage: eval_perplexity.py <url> <model> [<label>]")
    url, model = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else "(no label)"

    total_nll = 0.0
    total_n = 0
    print(f"# {label}  url={url}  model={model}\n")
    print(f"{'idx':>3}  {'tokens':>6}  {'nll/tok':>8}  {'ppl':>10}  text_preview")
    for i, t in enumerate(TEXTS):
        s, n = score(url, model, t)
        if n == 0:
            print(f"{i:>3}  {n:>6}  {'-':>8}  {'-':>10}  (no scored tokens)")
            continue
        nll = s / n
        ppl = math.exp(nll)
        prev = t[:50].replace("\n", " ") + ("..." if len(t) > 50 else "")
        print(f"{i:>3}  {n:>6}  {nll:>8.4f}  {ppl:>10.4f}  {prev}")
        total_nll += s
        total_n += n

    if total_n == 0:
        print("\nno tokens scored — does this server return prompt_logprobs?")
        return 1
    mean_nll = total_nll / total_n
    ppl = math.exp(mean_nll)
    print()
    print(f"# {label}: tokens={total_n}  mean_nll={mean_nll:.6f}  ppl={ppl:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
