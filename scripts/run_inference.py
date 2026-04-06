#!/usr/bin/env python3
"""
Smoke test: end-to-end generation with TurboKVCache on Qwen3.5-27B.

Usage:
    python scripts/run_inference.py --model Qwen/Qwen3.5-27B --bits 3.5
    python scripts/run_inference.py --model Qwen/Qwen3.5-27B --bits 2.5 --max-new-tokens 50

Requires A100 (40GB+) and Qwen/Qwen3.5-27B weights.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant import PRESET_2_5BIT, PRESET_3_5BIT, TurboKVCache, patch_model


PRESETS = {"2.5": PRESET_2_5BIT, "3.5": PRESET_3_5BIT}

PROMPT = (
    "The transformer architecture is a neural network design that relies entirely "
    "on attention mechanisms. Explain its key components and why it replaced RNNs "
    "for sequence modeling tasks:"
)


def main():
    parser = argparse.ArgumentParser(description="TurboQuant inference smoke test")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B", help="HuggingFace model ID")
    parser.add_argument("--bits", choices=["2.5", "3.5"], default="3.5")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--baseline", action="store_true", help="Also run baseline (no compression)")
    args = parser.parse_args()

    config = PRESETS[args.bits]
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    # Verify architecture
    cfg = model.config
    print(f"\nModel config:")
    print(f"  num_hidden_layers:   {cfg.num_hidden_layers}")
    print(f"  num_key_value_heads: {cfg.num_key_value_heads}")
    print(f"  head_dim:            {getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)}")
    print(f"  hidden_size:         {cfg.hidden_size}")

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    print(f"\nInput tokens: {inputs['input_ids'].shape[-1]}")

    # --- TurboQuant run ---
    cache_turbo = TurboKVCache(config)
    print(f"\nRunning TurboQuant {args.bits}-bit...")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs_turbo = model.generate(
            **inputs,
            past_key_values=cache_turbo,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    t1 = time.perf_counter()

    turbo_text = tokenizer.decode(outputs_turbo[0], skip_special_tokens=True)
    stats = cache_turbo.memory_stats()
    new_tokens = outputs_turbo.shape[-1] - inputs["input_ids"].shape[-1]

    print(f"  Generated {new_tokens} tokens in {t1-t0:.2f}s ({new_tokens/(t1-t0):.1f} tok/s)")
    print(f"  KV cache: {stats['total_mb']:.1f} MB ({stats['k_bytes']/1024:.1f} KB K, {stats['v_bytes']/1024:.1f} KB V)")
    print(f"\nOutput:\n{turbo_text[:500]}...")

    # --- Baseline run (optional) ---
    if args.baseline:
        print(f"\nRunning baseline (fp16 KV cache)...")
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs_base = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        t1 = time.perf_counter()
        print(f"  Generated {new_tokens} tokens in {t1-t0:.2f}s ({new_tokens/(t1-t0):.1f} tok/s)")

    print("\n✓ Smoke test passed — no errors during generation")


if __name__ == "__main__":
    main()
