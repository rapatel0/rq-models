"""Time-to-first-token and decode throughput against a vLLM endpoint.

We can't get TTFT from a non-streaming POST, so we use the streaming
SSE response (`stream=true`). For each request we record:

  ttft     — wall time from POST start to first non-empty token chunk
  decode   — wall time from first token to last token
  n_tokens — number of completion tokens
  tps      — n_tokens / decode  (decode throughput, tokens/sec)

Run a few prompts and report the median + range.

Usage:
    python3 perf_bench.py <url> <model> <label>
"""

from __future__ import annotations
import json, sys, time, urllib.request

PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Write three sentences about Paris.\n",
    "List the first five US presidents:\n1.",
    "def fibonacci(n):\n    if n",
]
MAX_TOKENS = 128


def stream_one(url: str, model: str, prompt: str) -> tuple[float, float, int]:
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": MAX_TOKENS, "temperature": 0.0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t_start = time.perf_counter()
    t_first: float | None = None
    n_tokens = 0
    last_text: list[str] = []
    with urllib.request.urlopen(req, timeout=180) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            tok = choices[0].get("text") or ""
            if tok:
                if t_first is None:
                    t_first = time.perf_counter()
                last_text.append(tok)
                n_tokens += 1
    t_end = time.perf_counter()
    if t_first is None:
        return (0.0, 0.0, 0)
    return (t_first - t_start, t_end - t_first, n_tokens)


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit("usage: perf_bench.py <url> <model> <label>")
    url, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"# {label}  url={url}  model={model}\n")

    # Warm-up so the JIT compiles, caches load, etc.
    print("warm-up ...", flush=True)
    stream_one(url, model, "Hello.")
    print()

    print(f"{'idx':>3}  {'ttft_s':>7}  {'decode_s':>9}  {'n':>4}  {'tps':>7}  prompt")
    ttfts: list[float] = []
    tpss: list[float] = []
    for i, p in enumerate(PROMPTS):
        ttft, decode, n = stream_one(url, model, p)
        if n == 0:
            print(f"{i:>3}  ERROR")
            continue
        tps = n / decode if decode > 0 else float("inf")
        ttfts.append(ttft)
        tpss.append(tps)
        prev = p[:30].replace("\n", " ")
        print(f"{i:>3}  {ttft:>7.3f}  {decode:>9.3f}  {n:>4d}  {tps:>7.2f}  {prev!r}")

    if ttfts:
        ttfts.sort()
        tpss.sort()
        med_ttft = ttfts[len(ttfts) // 2]
        med_tps = tpss[len(tpss) // 2]
        print(f"\n# {label}: median ttft = {med_ttft*1000:.1f} ms, "
              f"median tps = {med_tps:.2f} tok/s   "
              f"(ttft range {min(ttfts)*1000:.1f}–{max(ttfts)*1000:.1f} ms, "
              f"tps range {min(tpss):.2f}–{max(tpss):.2f})")


if __name__ == "__main__":
    main()
