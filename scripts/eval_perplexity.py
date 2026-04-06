#!/usr/bin/env python3
"""
wikitext-2 sliding window perplexity evaluation.

Measures perplexity of Qwen3.5-27B with TurboQuant KV cache vs baseline.

Usage:
    python scripts/eval_perplexity.py --model Qwen/Qwen3.5-27B --bits 3.5
    python scripts/eval_perplexity.py --model Qwen/Qwen3.5-27B --bits 3.5 --max-length 1024
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from turboquant import PRESET_2_5BIT, PRESET_3_5BIT, TurboKVCache

PRESETS = {"2.5": PRESET_2_5BIT, "3.5": PRESET_3_5BIT, "none": None}


def compute_perplexity(
    model, tokenizer, texts: str,
    max_length: int = 2048,
    stride: int = 512,
    config=None,
) -> float:
    """
    Sliding-window perplexity on a string.
    Matches the standard HuggingFace perplexity evaluation methodology.
    """
    encodings = tokenizer(texts, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    loss_fn = CrossEntropyLoss(reduction="sum")

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        window_ids = input_ids[:, begin_loc:end_loc]
        target_ids = window_ids.clone()
        target_ids[:, :-trg_len] = -100  # mask prefix tokens

        past_kv = TurboKVCache(config) if config is not None else None

        with torch.no_grad():
            outputs = model(
                window_ids,
                past_key_values=past_kv,
                use_cache=(past_kv is not None),
            )
        logits = outputs.logits  # [1, seq, vocab]

        # Shift: predict token t from logits at position t-1
        shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        shift_labels = target_ids[:, 1:].contiguous().view(-1)
        nll = loss_fn(shift_logits, shift_labels)
        nlls.append(nll.item())

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = math.exp(sum(nlls) / (seq_len - 1))
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--bits", choices=["2.5", "3.5", "none"], default="3.5")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    print("Loading wikitext-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    config = PRESETS[args.bits]

    print(f"\nEvaluating {args.bits}-bit TurboQuant perplexity...")
    ppl_turbo = compute_perplexity(
        model, tokenizer, text,
        max_length=args.max_length, stride=args.stride, config=config
    )
    print(f"  Perplexity ({args.bits}-bit): {ppl_turbo:.3f}")

    if args.compare_baseline:
        print("\nEvaluating baseline (fp16)...")
        ppl_base = compute_perplexity(
            model, tokenizer, text,
            max_length=args.max_length, stride=args.stride, config=None
        )
        print(f"  Perplexity (baseline): {ppl_base:.3f}")
        delta = ppl_turbo - ppl_base
        print(f"  Δppl = {delta:+.3f} (target: ≤ 0.1 nats)")
        if delta <= 0.1:
            print("  ✓ PASS: within 0.1 nats of baseline")
        else:
            print("  ✗ FAIL: exceeds 0.1 nats threshold")


if __name__ == "__main__":
    main()
