"""Needle-in-haystack test for KV cache quality.

Build a long context out of factual filler and embed a unique 'needle'
sentence at a controllable position. Ask the model to retrieve the
needle. Run at multiple haystack lengths to see if rq3 holds at depth.

Usage:
    python3 needle_test.py <url> <model> <label>
"""

from __future__ import annotations
import json, sys, urllib.request


# Filler sentences. Each is short and factual; concatenating many gives
# us tokens cheaply without anything that could be confused with the
# needle.
FILLER = [
    "Cats sleep around twelve to sixteen hours per day.",
    "The Eiffel Tower was completed in eighteen eighty-nine.",
    "Honey never spoils when stored properly.",
    "A group of crows is called a murder.",
    "Octopuses have three hearts.",
    "Bananas are technically berries while strawberries are not.",
    "The Great Wall of China is over thirteen thousand miles long.",
    "Sound travels faster in water than in air.",
    "A bolt of lightning is hotter than the surface of the sun.",
    "The shortest war in history lasted thirty-eight minutes.",
    "Sloths can hold their breath for forty minutes underwater.",
    "Polar bears have black skin under their white fur.",
    "Wombat droppings are cube-shaped.",
    "Sharks have existed for over four hundred million years.",
    "The dot over the letter 'i' is called a tittle.",
    "A snail can sleep for three years.",
    "There are more stars in the universe than grains of sand on Earth.",
    "Penguins propose to their mates with pebbles.",
    "Hot water freezes faster than cold water under some conditions.",
    "Bees can recognize human faces.",
]


def build_haystack(target_chars: int, needle: str, needle_pos: float = 0.5) -> str:
    """Make a long passage with `needle` inserted at `needle_pos` fraction."""
    parts: list[str] = []
    char_count = 0
    inserted = False
    cycle_idx = 0
    while char_count < target_chars:
        if not inserted and char_count >= target_chars * needle_pos:
            parts.append(needle)
            char_count += len(needle) + 1
            inserted = True
            continue
        s = FILLER[cycle_idx % len(FILLER)]
        cycle_idx += 1
        parts.append(s)
        char_count += len(s) + 1
    if not inserted:
        parts.append(needle)
    return " ".join(parts)


def hit(url: str, model: str, prompt: str, max_tokens: int) -> tuple[str, int]:
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        resp = json.loads(r.read())
    return resp["choices"][0]["text"], resp["usage"]["prompt_tokens"]


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit("usage: needle_test.py <url> <model> <label>")
    url, model, label = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f"# {label}  url={url}  model={model}\n")

    NEEDLE = ("The secret passphrase for this conversation is "
              "PURPLE-OCTOPUS-SEVENTEEN.")

    print(f"{'target_chars':>13}  {'tokens':>7}  {'needle_pos':>11}  hit?  output_preview")
    for target_chars, pos in [
        (1000, 0.50),
        (3000, 0.50),
        (6000, 0.25),
        (6000, 0.50),
        (6000, 0.75),
        (12000, 0.50),
    ]:
        haystack = build_haystack(target_chars, NEEDLE, pos)
        prompt = (
            "Read the following passage carefully:\n\n" + haystack +
            "\n\nQuestion: What is the secret passphrase mentioned in the "
            "passage above? Answer with just the passphrase, nothing else.\n"
            "Answer:"
        )
        out, prompt_tokens = hit(url, model, prompt, max_tokens=500)
        match = "PURPLE-OCTOPUS-SEVENTEEN" in out.upper()
        marker = "PASS" if match else "FAIL"
        # Find what the model actually said about the passphrase, even if
        # buried in <think>. Look for any text after the last "PURPLE" or
        # the last informative-looking line.
        prev = out.replace("\n", "\\n")[:200]
        print(f"{target_chars:>13}  {prompt_tokens:>7}  {pos:>11.2f}  {marker}  {prev!r}")


if __name__ == "__main__":
    main()
