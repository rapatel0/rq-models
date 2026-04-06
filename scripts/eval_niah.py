#!/usr/bin/env python3
"""
Needle-In-A-Haystack (NIAH) evaluation for TurboQuant KV cache.

Tests recall of a hidden "needle" sentence embedded in a long document,
across context lengths 4k–32k.

Usage:
    python scripts/eval_niah.py --model Qwen/Qwen3.5-27B --bits 3.5
    python scripts/eval_niah.py --model Qwen/Qwen3.5-27B --bits 3.5 --context-lengths 4096 8192 16384
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant import PRESET_2_5BIT, PRESET_3_5BIT, TurboKVCache

PRESETS = {"2.5": PRESET_2_5BIT, "3.5": PRESET_3_5BIT, "none": None}

NEEDLE = "The special magic number is 73849-ALPHA-ZETA-99."
QUESTION = "What is the special magic number mentioned in the document?"
ANSWER_KEY = "73849-ALPHA-ZETA-99"

# Filler text to build the haystack
HAYSTACK_SENTENCE = (
    "The advancements in large language models have opened new frontiers in natural "
    "language processing research and applications across many different domains. "
)


def build_haystack(target_tokens: int, tokenizer, needle_position: float = 0.5) -> str:
    """Build a haystack of approximately target_tokens tokens with needle embedded."""
    filler = HAYSTACK_SENTENCE * 1000
    filler_ids = tokenizer.encode(filler, add_special_tokens=False)

    needle_ids = tokenizer.encode(f"\n{NEEDLE}\n", add_special_tokens=False)
    needle_pos = int(target_tokens * needle_position)

    haystack_ids = (
        filler_ids[:needle_pos]
        + needle_ids
        + filler_ids[needle_pos : target_tokens - len(needle_ids)]
    )
    return tokenizer.decode(haystack_ids)


def evaluate_niah(model, tokenizer, context_length: int, config, n_trials: int = 5) -> float:
    """Return recall score (fraction of trials where model found the needle)."""
    hits = 0
    depths = [i / (n_trials - 1) for i in range(n_trials)]  # 0.0 to 1.0

    for depth in depths:
        haystack = build_haystack(context_length - 100, tokenizer, needle_position=depth)
        prompt = (
            f"Document:\n{haystack}\n\nQuestion: {QUESTION}\nAnswer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=context_length).to(model.device)

        past_kv = TurboKVCache(config) if config is not None else None

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                past_key_values=past_kv,
                max_new_tokens=30,
                do_sample=False,
            )

        new_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True)

        if ANSWER_KEY in answer:
            hits += 1

    return hits / n_trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--bits", choices=["2.5", "3.5", "none"], default="3.5")
    parser.add_argument(
        "--context-lengths", nargs="+", type=int,
        default=[4096, 8192, 16384, 32768]
    )
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    config = PRESETS[args.bits]
    print(f"\nNIAH evaluation ({args.bits}-bit TurboQuant):")
    print(f"{'Context':>12}  {'Recall':>8}  {'Status':>8}")
    print("-" * 36)

    results = {}
    for ctx in args.context_lengths:
        recall = evaluate_niah(model, tokenizer, ctx, config, n_trials=args.trials)
        results[ctx] = recall
        status = "✓ PASS" if recall >= 0.99 else "✗ FAIL"
        print(f"{ctx:>12,}  {recall:>8.3f}  {status}")

    avg = sum(results.values()) / len(results)
    print(f"\nAverage recall: {avg:.3f} (target ≥ 0.99)")
    if avg >= 0.99:
        print("✓ PASS: NIAH DoD criterion met")
    else:
        print("✗ FAIL: NIAH DoD criterion not met")


if __name__ == "__main__":
    main()
