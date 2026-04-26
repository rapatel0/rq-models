"""Prefill-throughput bench. TTFT under varying prompt lengths.

For each target prompt length we send a request that asks for exactly
1 completion token. The wall time from POST start to response received
is dominated by prefill (compute attention over N input tokens) plus a
single decode step. Plotting wall time vs prompt length gives the
prefill cost per token.

Usage:
    python3 prefill_bench.py <url> <model> <label>
"""

from __future__ import annotations
import json, sys, time, urllib.request

# A long passage we can slice to reach target lengths. Aim for plain
# tokenizable English; tokens are roughly 0.75 words for Qwen.
SOURCE = (
    "The history of computing dates back to ancient times when humans "
    "first began using simple tools to assist with calculation. The "
    "abacus, used in Mesopotamia and later in China and Greece, is one "
    "of the earliest known computing devices. Mechanical calculators "
    "appeared in the seventeenth century with Pascal's Pascaline and "
    "Leibniz's Stepped Reckoner. Charles Babbage designed the "
    "Analytical Engine in the eighteen thirties, a general-purpose "
    "mechanical computer that was never fully built in his lifetime. "
    "Ada Lovelace, a mathematician and writer, recognized that the "
    "Analytical Engine could be used for more than just calculation, "
    "and she wrote the first algorithm intended to be processed by a "
    "machine, making her widely regarded as the first computer "
    "programmer. The twentieth century saw the development of "
    "electromechanical and then fully electronic computers. Alan "
    "Turing's theoretical work on computability laid the foundation "
    "for modern computer science. The first electronic general-purpose "
    "computer, ENIAC, was completed in nineteen forty-six. Vacuum tubes "
    "gave way to transistors in the nineteen fifties, and integrated "
    "circuits arrived in the nineteen sixties. The microprocessor, "
    "introduced in the nineteen seventies, made personal computing "
    "possible. The internet emerged from research projects in the "
    "nineteen sixties and seventies and became widely used in the "
    "nineteen nineties. Mobile computing, cloud computing, and "
    "artificial intelligence have all reshaped how computers are used "
    "in the twenty-first century. "
) * 6  # plenty to slice from


# Approximate target prompt lengths in characters; the actual token count
# is reported in the response.
LENGTHS = [200, 800, 2000, 4000, 6000]


def hit(url: str, model: str, prompt: str) -> tuple[float, int]:
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": 1, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    wall = time.perf_counter() - t0
    return wall, resp["usage"]["prompt_tokens"]


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit("usage: prefill_bench.py <url> <model> <label>")
    url, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"# {label}  url={url}  model={model}\n")

    # Warm up.
    print("warm-up ...", flush=True)
    hit(url, model, "Hello.")

    print()
    print(f"{'chars':>6}  {'tokens':>7}  {'wall_ms':>8}  {'prefill_tps':>12}")
    for L in LENGTHS:
        prompt = SOURCE[:L]
        # Run twice and take the second (warmer cache, more stable).
        hit(url, model, prompt)
        wall, tokens = hit(url, model, prompt)
        prefill_tps = tokens / wall if wall > 0 else float("inf")
        print(f"{L:>6}  {tokens:>7}  {wall*1000:>8.1f}  {prefill_tps:>12.1f}")


if __name__ == "__main__":
    main()
