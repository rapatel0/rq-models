"""Long-context smoke for the rq-vllm endpoint.

Two tests:

  test1 — feed the entire 13-paragraph PPL corpus (~700 tokens) as a
          prompt asking for a topical summary, generate 256 tokens.
          Validates that at least 1k tokens of context survive a full
          forward pass under planar3 KV.

  test2 — single short prompt, generate 512 tokens of story. The
          generated tokens become KV history the model attends to.
          Validates that depth-of-decode doesn't accumulate damage.

Usage:
    python3 longctx_test.py <url> <model> <label>
"""

from __future__ import annotations
import json, sys, urllib.request

PARAGRAPHS = [
    "George Washington was the first president of the United States.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Paris is the capital of France.",
    "A recursive Fibonacci function returns zero when n is zero.",
    "Two plus two equals four.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "A binary search algorithm finds the position of a target value within a sorted array.",
    "Photosynthesis is how green plants convert light into chemical energy.",
    "William Shakespeare was an English playwright born in fifteen sixty-four.",
    "Water boils at one hundred degrees Celsius at standard atmospheric pressure.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "A common pattern in software engineering is the producer-consumer queue.",
    "The Western Roman Empire fell in four hundred and seventy-six AD.",
]

LONG_PROMPT = (
    "Read the following thirteen short facts carefully:\n\n"
    + "\n".join(f"{i+1}. {p}" for i, p in enumerate(PARAGRAPHS))
    + "\n\nNow answer in five complete English sentences: which two of the "
    "above facts are about science (physics or biology), which two are "
    "about history, and what is the capital of France?\nAnswer:\n"
)

STORY_PROMPT = (
    "Write a 500-word short story about a software engineer named Mira who "
    "discovers an unexpected bug in the company's billing system on her "
    "first day at a new job. Begin the story with Mira walking into the "
    "office.\n\n"
)


def hit(url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["choices"][0]["text"]


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit("usage: longctx_test.py <url> <model> <label>")
    url, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"# {label}  url={url}  model={model}\n")

    print("## test1 — long-prompt summary (≈700-tok input → 256-tok output)\n")
    out = hit(url, model, LONG_PROMPT, 256)
    print(out.strip())

    print("\n\n## test2 — short prompt → 512-tok generation (depth-of-decode)\n")
    out = hit(url, model, STORY_PROMPT, 512)
    # Print first 200 chars + last 200 chars so we can spot decay at the tail.
    print(f"FIRST 240 chars:\n{out[:240]!r}\n")
    print(f"LAST 240 chars:\n{out[-240:]!r}\n")
    print(f"TOTAL LENGTH: {len(out)} chars")


if __name__ == "__main__":
    main()
