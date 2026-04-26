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


# Plain English paragraphs covering common topics (history, geography,
# basic science, math, code commentary, literature, trivia). We deliberately
# avoid model-specific output styles (no "<think>", no system prompts, no
# chat template) so the same TEXTS can be re-scored against a llama.cpp
# planar3 run later for cross-substrate parity. Target: ~1500 scored
# tokens at 256-token block boundaries (small enough to stay under the
# 2048-token context window and large enough that per-paragraph noise
# averages out).
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

    "Mount Everest is the highest mountain above sea level on Earth, "
    "rising about eight thousand eight hundred and forty-nine meters. "
    "It sits on the border between Nepal and the Tibet Autonomous Region "
    "of China. The first confirmed ascent was in nineteen fifty-three.",

    "A binary search algorithm finds the position of a target value within "
    "a sorted array. It compares the target to the middle element and "
    "discards half of the search space at each step. The running time is "
    "logarithmic in the size of the array, which is much faster than a "
    "linear scan for large inputs.",

    "Photosynthesis is the process by which green plants convert light "
    "energy into chemical energy stored in sugars. Carbon dioxide from "
    "the air and water from the soil are combined inside chloroplasts "
    "to produce glucose, releasing oxygen as a by-product. This process "
    "supports nearly all life on Earth.",

    "William Shakespeare was an English playwright and poet, widely "
    "regarded as the greatest writer in the English language. He was "
    "born in Stratford-upon-Avon in fifteen sixty-four and produced "
    "around thirty-nine plays, including Hamlet, Macbeth, and Romeo and "
    "Juliet, along with one hundred and fifty-four sonnets.",

    "Water boils at one hundred degrees Celsius at standard atmospheric "
    "pressure, which is equivalent to two hundred and twelve degrees "
    "Fahrenheit. At higher altitudes, where atmospheric pressure is lower, "
    "water boils at a lower temperature, which is why cooking times often "
    "need to be adjusted in mountainous regions.",

    "The speed of light in a vacuum is approximately two hundred and "
    "ninety-nine million seven hundred and ninety-two thousand four "
    "hundred and fifty-eight meters per second. This constant, denoted "
    "by the letter c, plays a fundamental role in modern physics, "
    "including Einstein's theory of special relativity.",

    "A common pattern in software engineering is the producer-consumer "
    "queue. One or more producer threads enqueue work items while one or "
    "more consumer threads dequeue and process them. The queue itself is "
    "responsible for synchronization, typically using a mutex and a "
    "condition variable, so the producers and consumers do not need to "
    "coordinate directly.",

    "The Roman Empire reached its greatest territorial extent under the "
    "emperor Trajan in the early second century. After his reign, the "
    "empire gradually lost territory and split into a Western and Eastern "
    "half. The Western Roman Empire fell in four hundred and seventy-six, "
    "while the Eastern Roman Empire, also known as the Byzantine Empire, "
    "endured until fourteen fifty-three.",
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
