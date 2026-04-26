"""Hit a running vLLM server with a battery of prompts and dump
side-by-side outputs. Designed to be run twice (once against the
rotorquant container, once against fp16) so the dumps can be diffed.
"""

import json
import sys
import urllib.request
import urllib.error

# Mix of factual, code, math, and longer-horizon prompts so we exercise
# more of the KV cache than the trivial 24-token smoke.
PROMPTS: list[tuple[str, str, int]] = [
    ("paris_short",   "The capital of France is",                                 24),
    ("math_basic",    "2+2=",                                                      12),
    ("math_chain",    "If a train leaves Boston at 8 AM going 60 mph and another", 80),
    ("code_python",   "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n    if n",  64),
    ("list_recall",   "List the first ten US presidents:\n1.",                     128),
    ("essay_open",    "Write three sentences explaining why Paris is famous.\n",   128),
    ("longer_ctx",    "The four largest oceans on Earth, ranked by surface area, are: 1.",  64),
]


def hit(url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return data["choices"][0]["text"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
        return f"<<error: {exc}>>"


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit("usage: quality_battery.py <url> <model> <label>")
    url, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"# {label}  url={url}  model={model}\n")
    for name, prompt, max_tokens in PROMPTS:
        out = hit(url, model, prompt, max_tokens)
        # Truncate output for readability but keep enough to spot drift.
        view = out.replace("\n", "\\n")
        if len(view) > 240:
            view = view[:237] + "..."
        print(f"## {name}  ({max_tokens} tok)")
        print(f"prompt:  {prompt!r}")
        print(f"output:  {view!r}")
        print()


if __name__ == "__main__":
    main()
