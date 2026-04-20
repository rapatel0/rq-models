#!/usr/bin/env python3
"""
Offline calibration script for SpectralQuant KV cache.

Runs calibration prompts through the model, computes PCA bases per attention
layer, fits Lloyd-Max codebooks, and saves as a safetensors sidecar.

Usage:
    python scripts/calibrate_spectral.py \
        --model Qwen/Qwen3.5-9B-Instruct \
        --output calibration/ \
        --n-prompts 32 \
        --max-length 512

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


WIKITEXT_PROMPTS = [
    # Diverse excerpts for calibration corpus (not from private data)
    "The history of artificial intelligence dates back to the 1950s, when pioneers such as Alan Turing and John McCarthy laid the theoretical foundations for machine cognition.",
    "Quantum mechanics describes the physical properties of nature at the scale of atoms and subatomic particles, fundamentally different from classical physics.",
    "The development of the internet transformed global communication, enabling instantaneous exchange of information across continents and reshaping commerce.",
    "Language models learn statistical patterns from vast corpora of text, enabling them to generate coherent and contextually appropriate responses.",
    "Climate change presents one of the most significant challenges of the twenty-first century, requiring coordinated international effort to address.",
    "The human genome contains approximately three billion base pairs encoding roughly twenty thousand protein-coding genes.",
    "Neural networks are computational systems inspired by biological neural networks, consisting of interconnected nodes that process information.",
    "The industrial revolution fundamentally altered patterns of work and daily life, accelerating urbanization and transforming economic structures.",
    "Protein folding refers to the physical process by which a protein chain acquires its functional three-dimensional structure.",
    "Mathematics provides the language and tools to describe patterns and relationships in the physical and abstract world.",
    "The theory of evolution by natural selection, first articulated by Charles Darwin, explains the diversity of life on Earth.",
    "Software engineering involves systematic approaches to the design, development, testing, and maintenance of software systems.",
    "The Big Bang theory describes the origin and evolution of the universe from an extremely hot and dense initial state.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose molecules.",
    "Democracy refers to a system of government in which power is vested in the people, who exercise it through elected representatives.",
    "Cryptography is the practice of securing communications through encoding, ensuring that only intended recipients can read messages.",
    "The immune system defends the body against pathogens through a complex network of cells, tissues, and organs.",
    "Renewable energy sources such as solar and wind power are increasingly cost-competitive with fossil fuels.",
    "Cognitive science is an interdisciplinary field studying the nature of mind and intelligence, drawing on psychology and neuroscience.",
    "The rules of thermodynamics govern energy transformations, establishing fundamental limits on the efficiency of engines and processes.",
    "Computer vision enables machines to interpret and understand visual information from the world, using convolutional neural networks.",
    "Economic policy involves government decisions about taxation, spending, and regulation that affect the broader economy.",
    "The structure of DNA was elucidated by Watson and Crick in 1953, revealing the double helix mechanism of genetic inheritance.",
    "Compiler design involves translating high-level programming languages into machine code through lexical analysis and code generation.",
    "Philosophy of mind examines questions about consciousness, perception, and the relationship between mental and physical states.",
    "Fluid dynamics describes the motion of liquids and gases, with applications in aerodynamics, meteorology, and engineering.",
    "The principles of object-oriented programming include encapsulation, inheritance, polymorphism, and abstraction.",
    "Neuroscience investigates the structure and function of the nervous system, from individual neurons to complex brain circuits.",
    "Statistical methods allow researchers to draw inferences about populations from samples, quantifying uncertainty in conclusions.",
    "The laws of motion formulated by Isaac Newton describe how objects move under the influence of forces.",
    "Distributed systems coordinate multiple computers to achieve common goals, requiring careful handling of consistency and fault tolerance.",
    "Ecology studies the relationships between organisms and their environments, including energy flow and nutrient cycling.",
]


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


def build_prompts(n: int) -> list[str]:
    """Return n calibration prompts, cycling through the corpus if needed."""
    prompts = []
    while len(prompts) < n:
        prompts.extend(WIKITEXT_PROMPTS)
    return prompts[:n]


def main():
    parser = argparse.ArgumentParser(description="SpectralQuant offline calibration")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--output", default="calibration/", help="Output directory")
    parser.add_argument("--n-prompts", type=int, default=32, help="Number of calibration prompts")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length per prompt")
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

    prompts = build_prompts(args.n_prompts)
    logger.info(f"Using {len(prompts)} calibration prompts (max {args.max_length} tokens each)")

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
