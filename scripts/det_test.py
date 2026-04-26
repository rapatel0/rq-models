"""Determinism check. Submit each prompt N times at T=0 and verify all
N completions are byte-identical."""

from __future__ import annotations
import json, sys, urllib.request, hashlib

PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "def fibonacci(n):\n    if n",
    "List the first five US presidents:\n1.",
    "Photosynthesis is",
]
RUNS = 3
MAX_TOKENS = 64


def hit(url: str, model: str, prompt: str) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": MAX_TOKENS, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["text"]


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit("usage: det_test.py <url> <model> <label>")
    url, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"# {label}  url={url}  model={model}\n")

    print(f"{'idx':>3}  match?  hashes (run 0..{RUNS-1})")
    overall_pass = 0
    overall_fail = 0
    for i, p in enumerate(PROMPTS):
        outs = [hit(url, model, p) for _ in range(RUNS)]
        hashes = [hashlib.sha1(o.encode()).hexdigest()[:8] for o in outs]
        all_same = len(set(outs)) == 1
        marker = "PASS" if all_same else "FAIL"
        if all_same:
            overall_pass += 1
        else:
            overall_fail += 1
        print(f"{i:>3}  {marker}    {' '.join(hashes)}    prompt={p[:30]!r}")

    print(f"\n# {label}: {overall_pass}/{len(PROMPTS)} prompts deterministic across {RUNS} runs each")


if __name__ == "__main__":
    main()
