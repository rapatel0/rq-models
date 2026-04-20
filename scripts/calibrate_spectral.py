#!/usr/bin/env python3
"""
Offline calibration script for SpectralQuant KV cache.

Runs calibration prompts through the model, computes PCA bases per attention
layer, fits Lloyd-Max codebooks, and saves as a safetensors sidecar.

Calibration standard (matches GPTQ / AWQ):
  Dataset:  allenai/c4, English, train split
  Samples:  128 sequences × 2048 tokens
  Seed:     42

Usage:
    python scripts/calibrate_spectral.py \
        --model Qwen/Qwen3.5-9B-Instruct \
        --output calibration/

    # Different dataset or sample count:
    python scripts/calibrate_spectral.py --model ... --dataset wikitext --n-samples 64

    # With HF token for gated models:
    HF_TOKEN=hf_... python scripts/calibrate_spectral.py --model ...

    # Sweep d_eff to validate variance threshold heuristic:
    python scripts/calibrate_spectral.py --model ... --sweep-d-eff
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate_spectral")


def load_model_and_tokenizer(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    logger.info(f"Model loaded in {time.time() - t0:.1f}s on {device}")
    return model, tokenizer


def main():
    from turboquant.corpus import (
        CALIBRATION_DATASET, CALIBRATION_N_SAMPLES, CALIBRATION_SEQ_LEN, CALIBRATION_SEED,
        load_calibration_texts,
    )

    parser = argparse.ArgumentParser(description="SpectralQuant offline calibration")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--output", default="calibration/", help="Output directory")
    parser.add_argument(
        "--dataset", default=CALIBRATION_DATASET,
        help=f"HuggingFace dataset for calibration (default: {CALIBRATION_DATASET})"
    )
    parser.add_argument(
        "--n-samples", type=int, default=CALIBRATION_N_SAMPLES,
        help=f"Number of calibration samples (default: {CALIBRATION_N_SAMPLES})"
    )
    parser.add_argument(
        "--max-length", type=int, default=CALIBRATION_SEQ_LEN,
        help=f"Max token length per sample (default: {CALIBRATION_SEQ_LEN})"
    )
    parser.add_argument(
        "--seed", type=int, default=CALIBRATION_SEED,
        help=f"Random seed for corpus sampling (default: {CALIBRATION_SEED})"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--variance-threshold", type=float, default=0.99,
        help="Cumulative variance fraction for d_eff selection (default 0.99)"
    )
    parser.add_argument(
        "--sweep-d-eff", action="store_true",
        help="After calibration, print d_eff distribution per layer for analysis"
    )
    parser.add_argument("--name", default=None, help="Override sidecar filename (without extension)")
    args = parser.parse_args()

    # Check GPU memory before starting
    if args.device == "cuda" and torch.cuda.is_available():
        free_mib = torch.cuda.mem_get_info()[0] / 1024**2
        logger.info(f"GPU free memory: {free_mib:.0f} MiB")
        if free_mib < 4000:
            logger.warning("Less than 4 GB GPU memory free — may OOM during calibration")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    prompts = load_calibration_texts(
        n_samples=args.n_samples,
        dataset=args.dataset,
        seed=args.seed,
    )
    logger.info(f"Using {len(prompts)} calibration samples from {args.dataset} (max {args.max_length} tokens each)")

    from turboquant.spectral.calibrator import SpectralCalibrator
    from turboquant.spectral.store import CalibrationStore

    calibrator = SpectralCalibrator(
        model,
        variance_threshold=args.variance_threshold,
    )

    t0 = time.time()
    calibration = calibrator.fit(
        prompts=prompts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        device=torch.device(args.device),
    )
    elapsed = time.time() - t0

    if not calibration:
        logger.error("No attention layers found — calibration failed")
        sys.exit(1)

    logger.info(f"Calibrated {len(calibration)} attention layers in {elapsed:.1f}s")

    # Print d_eff summary
    print("\n=== d_eff Summary ===")
    print(f"{'Layer':>6}  {'d_eff_k':>8}  {'d_eff_v':>8}  {'K-fallback':>10}  {'V-fallback':>10}")
    for layer_idx, cal in sorted(calibration.items()):
        print(
            f"{layer_idx:>6}  {cal.d_eff_k:>8}  {cal.d_eff_v:>8}"
            f"  {'YES' if cal.fallback_k else 'no':>10}  {'YES' if cal.fallback_v else 'no':>10}"
        )

    if args.sweep_d_eff:
        print("\n=== Eigenvalue Analysis (top-10 K eigenvalues per layer) ===")
        for layer_idx, cal in sorted(calibration.items()):
            mean_ev_k = cal.eigenvalues_k[0]  # first head
            top10 = mean_ev_k[:10].tolist()
            top10_str = "  ".join(f"{v:.4f}" for v in top10)
            total = float(mean_ev_k.sum())
            print(f"Layer {layer_idx} K: [{top10_str}]  total={total:.4f}")

    # Determine output path
    if args.name:
        model_label = args.name
    else:
        model_label = args.model.split("/")[-1].lower()

    output_path = CalibrationStore.sidecar_path(model_label, args.output)
    CalibrationStore.save(calibration, output_path)
    size_mb = output_path.stat().st_size / 1024**2
    logger.info(f"Saved calibration: {output_path} ({size_mb:.2f} MB)")

    print(f"\nCalibration saved to: {output_path}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"\nTo use in inference:")
    print(f"  from turboquant.spectral import SpectralKVCache, load_calibration")
    print(f"  cal = load_calibration('{output_path}')")
    print(f"  cache = SpectralKVCache(cal)")
    print(f"  outputs = model.generate(input_ids, past_key_values=cache, max_new_tokens=200)")


if __name__ == "__main__":
    main()
