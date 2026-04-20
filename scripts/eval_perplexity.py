#!/usr/bin/env python3
"""
Sliding-window perplexity evaluation.

Measures perplexity of TurboQuant / SpectralQuant KV cache vs baseline.
Evaluation standard: wikitext-2-raw-v1 test split (industry standard).

Usage:
    # TurboKVCache (original)
    python scripts/eval_perplexity.py --model Qwen/Qwen3.5-27B --bits 3.5

    # SpectralKVCache
    python scripts/eval_perplexity.py \
        --model Qwen/Qwen3.5-9B-Instruct \
        --cache spectral \
        --calibration calibration/calibration-qwen3.5-9b-instruct.safetensors

    # Both caches vs baseline in one run
    python scripts/eval_perplexity.py --model ... --compare-all

    # Use C4 validation instead of wikitext-2
    python scripts/eval_perplexity.py --model ... --eval-dataset allenai/c4
"""

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant import PRESET_2_5BIT, PRESET_3_5BIT, TurboKVCache
from turboquant.corpus import load_eval_text, EVAL_DATASET, EVAL_CONFIG, EVAL_SPLIT

PRESETS = {"2.5": PRESET_2_5BIT, "3.5": PRESET_3_5BIT, "none": None}


def compute_perplexity(
    model, tokenizer, text: str,
    max_length: int = 2048,
    stride: int = 512,
    cache_factory=None,
) -> float:
    """
    Sliding-window perplexity on a string.
    Matches the standard HuggingFace perplexity evaluation methodology.

    Args:
        model:         HF causal LM
        tokenizer:     HF tokenizer
        text:          evaluation text (single concatenated string)
        max_length:    context window per step
        stride:        stride between windows
        cache_factory: callable → past_key_values object, or None for f16 baseline
    """
    encodings = tokenizer(text, return_tensors="pt")
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

        past_kv = cache_factory() if cache_factory is not None else None

        with torch.no_grad():
            outputs = model(
                window_ids,
                past_key_values=past_kv,
                use_cache=(past_kv is not None),
            )
        logits = outputs.logits  # [1, seq, vocab]

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
    parser.add_argument(
        "--cache", choices=["turbo", "spectral", "none"], default="turbo",
        help="KV cache type to evaluate (default: turbo)"
    )
    parser.add_argument(
        "--bits", choices=["2.5", "3.5", "none"], default="3.5",
        help="TurboKVCache preset (only used when --cache turbo)"
    )
    parser.add_argument(
        "--calibration", default=None,
        help="Path to SpectralQuant .safetensors calibration file"
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument(
        "--eval-dataset", default=EVAL_DATASET,
        help=f"Evaluation dataset (default: {EVAL_DATASET} — wikitext-2 test)"
    )
    parser.add_argument(
        "--eval-config", default=EVAL_CONFIG,
        help=f"Dataset config/subset (default: {EVAL_CONFIG})"
    )
    parser.add_argument(
        "--eval-split", default=EVAL_SPLIT,
        help=f"Dataset split (default: {EVAL_SPLIT})"
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Also run f16 baseline and print delta"
    )
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    print(f"Loading eval corpus: {args.eval_dataset}/{args.eval_config} {args.eval_split}...")
    text = load_eval_text(
        dataset=args.eval_dataset,
        config=args.eval_config,
        split=args.eval_split,
    )

    # Build cache factory
    cache_factory = None
    cache_label = "f16 baseline"

    if args.cache == "turbo":
        config = PRESETS[args.bits]
        if config is not None:
            cache_factory = lambda: TurboKVCache(config)
            cache_label = f"TurboKVCache {args.bits}-bit"

    elif args.cache == "spectral":
        if not args.calibration:
            print("ERROR: --calibration required when --cache spectral", file=sys.stderr)
            sys.exit(1)
        from turboquant.spectral import SpectralKVCache, load_calibration
        calibration = load_calibration(args.calibration)
        cache_factory = lambda: SpectralKVCache(calibration, config=model.config)
        cache_label = f"SpectralKVCache ({Path(args.calibration).stem})"

    print(f"\nEvaluating {cache_label}...")
    ppl = compute_perplexity(
        model, tokenizer, text,
        max_length=args.max_length,
        stride=args.stride,
        cache_factory=cache_factory,
    )
    print(f"  Perplexity: {ppl:.4f}")

    if args.compare_baseline and cache_factory is not None:
        print("\nEvaluating f16 baseline...")
        ppl_base = compute_perplexity(
            model, tokenizer, text,
            max_length=args.max_length,
            stride=args.stride,
            cache_factory=None,
        )
        print(f"  Perplexity (f16): {ppl_base:.4f}")
        delta = ppl - ppl_base
        print(f"  Δppl = {delta:+.4f}")
        gate = 0.5
        status = "PASS" if delta <= gate else "FAIL"
        print(f"  Kill gate (Δ ≤ {gate}): {status}")


if __name__ == "__main__":
    main()
