#!/usr/bin/env python3
"""
SpectralQuant benchmark: perplexity, cosine similarity, and throughput.

Compares SpectralKVCache against f16 baseline on a small model loaded locally.

Kill gate:
  - Cosine similarity K vectors: > 0.92
  - Cosine similarity V vectors: > 0.85
  - Perplexity delta vs f16: < 0.5 (spectral PPL <= f16 PPL + 0.5)

Usage:
    # Quick cosine sim + throughput check (no calibration file required):
    python scripts/benchmark_spectral.py \\
        --model Qwen/Qwen3.5-9B-Instruct \\
        --calibration calibration/calibration-qwen3.5-9b-instruct.safetensors \\
        --mode cosine

    # Full benchmark (cosine + PPL + throughput):
    python scripts/benchmark_spectral.py \\
        --model Qwen/Qwen3.5-9B-Instruct \\
        --calibration calibration/calibration-qwen3.5-9b-instruct.safetensors

    # PPL only:
    python scripts/benchmark_spectral.py --model ... --calibration ... --mode ppl

HF token for gated models:
    HF_TOKEN=hf_... python scripts/benchmark_spectral.py ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_spectral")


# ---------------------------------------------------------------------------
# PPL evaluation helpers
# ---------------------------------------------------------------------------

WIKITEXT_EVAL = [
    "The transformer architecture has revolutionized natural language processing, enabling models to capture long-range dependencies in text through attention mechanisms.",
    "Quantum computing leverages the principles of superposition and entanglement to perform computations that would be intractable on classical hardware.",
    "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses, forming a complex network.",
    "Climate scientists use general circulation models to simulate the dynamics of the atmosphere and ocean, projecting future temperature and precipitation patterns.",
    "The synthesis of penicillin by Alexander Fleming in 1928 marked the beginning of the antibiotic era and transformed the treatment of bacterial infections.",
    "Reinforcement learning agents learn to make decisions by interacting with an environment and receiving feedback in the form of rewards and penalties.",
    "The expansion of the universe, first observed by Edwin Hubble in 1929, indicates that galaxies are moving away from each other at velocities proportional to their distances.",
    "Cryptographic hash functions map data of arbitrary size to a fixed-length output and are fundamental to digital signatures and blockchain protocols.",
]


def compute_perplexity(model, tokenizer, texts: list[str], device, cache_cls=None, calibration=None) -> float:
    """
    Compute per-token negative log likelihood (≈ perplexity) on a list of texts.

    Args:
        model: HF causal LM
        tokenizer: HF tokenizer
        texts: evaluation texts
        device: torch device
        cache_cls: optional cache class to use (SpectralKVCache or None for f16)
        calibration: calibration dict for SpectralKVCache

    Returns:
        Perplexity (float)
    """
    from transformers import DynamicCache

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            input_ids = inputs["input_ids"]
            n_tokens = input_ids.shape[1]
            if n_tokens < 2:
                continue

            if cache_cls is not None and calibration is not None:
                past_kv = cache_cls(calibration)
            else:
                past_kv = DynamicCache()

            outputs = model(
                input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

            logits = outputs.logits  # [1, T, vocab]
            # Shift: predict token t+1 from position t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_nll += loss.item() * (n_tokens - 1)
            total_tokens += n_tokens - 1

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return float(torch.exp(torch.tensor(avg_nll)).item())


# ---------------------------------------------------------------------------
# Cosine similarity benchmark
# ---------------------------------------------------------------------------

def benchmark_cosine_similarity(model, tokenizer, calibration: dict, device) -> dict:
    """
    Run a forward pass with hooks to capture K/V activations, then compare
    spectral encode → decode output against original activations.

    Returns dict with per-layer cosine sim stats.
    """
    from turboquant.spectral.quantizer import SpectralQuantizer

    quantizer = SpectralQuantizer(calibration, device=device)

    # Capture K/V from a single forward pass
    captured_k = {}
    captured_v = {}

    def make_hooks(layer_idx, attn_module):
        def k_hook(mod, inp, out):
            captured_k[layer_idx] = out.detach().float()

        def v_hook(mod, inp, out):
            captured_v[layer_idx] = out.detach().float()

        k_h = attn_module.k_proj.register_forward_hook(k_hook)
        v_h = attn_module.v_proj.register_forward_hook(v_hook)
        return k_h, v_h

    # Find attention layers
    model_layers = model.model.layers
    hooks = []
    attn_layers = {}
    for i, layer in enumerate(model_layers):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "k_proj") and i in calibration:
            attn_layers[i] = attn
            kh, vh = make_hooks(i, attn)
            hooks.extend([kh, vh])

    text = " ".join(WIKITEXT_EVAL[:4])
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

    model.eval()
    with torch.no_grad():
        model(**inputs, use_cache=False)

    for h in hooks:
        h.remove()

    if not captured_k:
        logger.warning("No activations captured — check model architecture")
        return {}

    # Get model config
    cfg = model.config
    n_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    results = {}
    for layer_idx in sorted(captured_k.keys()):
        k_act = captured_k[layer_idx]  # [batch, seq, n_kv_heads * head_dim] or [batch, n_kv_heads, seq, head_dim]
        v_act = captured_v[layer_idx]

        # Reshape to [1, n_kv_heads, seq, head_dim]
        if k_act.dim() == 3:
            batch, seq, dim = k_act.shape
            k_act = k_act.reshape(batch, seq, n_kv_heads, head_dim).permute(0, 2, 1, 3)
            v_act = v_act.reshape(batch, seq, n_kv_heads, head_dim).permute(0, 2, 1, 3)

        seq_len = k_act.shape[2]
        cos_k_list = []
        cos_v_list = []

        for t in range(min(seq_len, 32)):  # compare first 32 tokens
            k_tok = k_act[:, :, t:t+1, :].half()  # [1, H, 1, D]
            v_tok = v_act[:, :, t:t+1, :].half()

            qkv_k = quantizer.encode_k(k_tok, layer_idx)
            qkv_v = quantizer.encode_v(v_tok, layer_idx)

            k_dec = quantizer.decode_k([qkv_k], layer_idx).squeeze(0).squeeze(1).float()  # [H, D]
            v_dec = quantizer.decode_v([qkv_v], layer_idx).squeeze(0).squeeze(1).float()

            k_orig = k_tok.squeeze(0).squeeze(1).float()  # [H, D]
            v_orig = v_tok.squeeze(0).squeeze(1).float()

            cos_k = F.cosine_similarity(k_orig, k_dec, dim=-1).mean().item()
            cos_v = F.cosine_similarity(v_orig, v_dec, dim=-1).mean().item()
            cos_k_list.append(cos_k)
            cos_v_list.append(cos_v)

        results[layer_idx] = {
            "cos_k_mean": sum(cos_k_list) / len(cos_k_list),
            "cos_k_min":  min(cos_k_list),
            "cos_v_mean": sum(cos_v_list) / len(cos_v_list),
            "cos_v_min":  min(cos_v_list),
        }

    return results


# ---------------------------------------------------------------------------
# Throughput benchmark
# ---------------------------------------------------------------------------

def benchmark_throughput(model, tokenizer, calibration: dict, device, n_tokens: int = 100) -> dict:
    """
    Measure decode throughput: tokens/second with and without SpectralKVCache.
    """
    from turboquant.spectral.kv_cache import SpectralKVCache

    prompt = "The history of quantum computing dates back to the early 1980s when Richard Feynman proposed"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    results = {}

    for label, cache_factory in [
        ("f16_baseline", lambda: None),
        ("spectral_cache", lambda: SpectralKVCache(calibration)),
    ]:
        # Warmup
        with torch.no_grad():
            model.generate(
                inputs["input_ids"],
                max_new_tokens=20,
                do_sample=False,
                past_key_values=cache_factory(),
            )

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs["input_ids"],
                max_new_tokens=n_tokens,
                do_sample=False,
                past_key_values=cache_factory(),
            )
        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed = time.time() - t0

        generated = out.shape[1] - inputs["input_ids"].shape[1]
        tok_per_s = generated / elapsed
        results[label] = {
            "generated_tokens": generated,
            "elapsed_s": elapsed,
            "tok_per_s": tok_per_s,
        }
        logger.info(f"{label}: {tok_per_s:.1f} tok/s ({generated} tokens in {elapsed:.2f}s)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SpectralQuant benchmark")
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--calibration", required=True, help="Path to .safetensors calibration file")
    parser.add_argument(
        "--mode", choices=["cosine", "ppl", "throughput", "all"], default="all",
        help="Benchmark mode (default: all)"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--throughput-tokens", type=int, default=100, help="Tokens for throughput test")
    args = parser.parse_args()

    # Load calibration
    from turboquant.spectral.store import CalibrationStore, CalibrationNotFoundError
    try:
        calibration = CalibrationStore.load(args.calibration)
    except CalibrationNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Loaded calibration: {len(calibration)} layers from {args.calibration}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    device = torch.device(args.device)

    run_cosine = args.mode in ("cosine", "all")
    run_ppl = args.mode in ("ppl", "all")
    run_throughput = args.mode in ("throughput", "all")

    # --- Cosine similarity ---
    if run_cosine:
        logger.info("Running cosine similarity benchmark...")
        cos_results = benchmark_cosine_similarity(model, tokenizer, calibration, device)

        if cos_results:
            all_cos_k = [v["cos_k_mean"] for v in cos_results.values()]
            all_cos_v = [v["cos_v_mean"] for v in cos_results.values()]
            mean_k = sum(all_cos_k) / len(all_cos_k)
            mean_v = sum(all_cos_v) / len(all_cos_v)

            print("\n=== Cosine Similarity (per layer, averaged over 32 tokens) ===")
            print(f"{'Layer':>6}  {'cos_k_mean':>11}  {'cos_k_min':>10}  {'cos_v_mean':>11}  {'cos_v_min':>10}")
            for layer_idx, r in sorted(cos_results.items()):
                print(
                    f"{layer_idx:>6}  {r['cos_k_mean']:>11.4f}  {r['cos_k_min']:>10.4f}"
                    f"  {r['cos_v_mean']:>11.4f}  {r['cos_v_min']:>10.4f}"
                )

            print(f"\nMean cos_k: {mean_k:.4f}  (kill gate: > 0.92)")
            print(f"Mean cos_v: {mean_v:.4f}  (kill gate: > 0.85)")

            k_pass = mean_k > 0.92
            v_pass = mean_v > 0.85
            print(f"\nKill gate K: {'PASS' if k_pass else 'FAIL'}")
            print(f"Kill gate V: {'PASS' if v_pass else 'FAIL'}")
        else:
            logger.warning("No cosine similarity results collected")

    # --- Perplexity ---
    if run_ppl:
        logger.info("Running perplexity benchmark...")
        from turboquant.spectral.kv_cache import SpectralKVCache

        logger.info("  Computing f16 baseline PPL...")
        ppl_f16 = compute_perplexity(model, tokenizer, WIKITEXT_EVAL, device)

        logger.info("  Computing SpectralKVCache PPL...")
        ppl_spectral = compute_perplexity(
            model, tokenizer, WIKITEXT_EVAL, device,
            cache_cls=SpectralKVCache, calibration=calibration
        )

        delta = ppl_spectral - ppl_f16
        print(f"\n=== Perplexity ===")
        print(f"f16 baseline:      {ppl_f16:.4f}")
        print(f"SpectralKVCache:   {ppl_spectral:.4f}")
        print(f"Delta:             {delta:+.4f}  (kill gate: < +0.5)")
        print(f"Kill gate PPL: {'PASS' if delta < 0.5 else 'FAIL'}")

    # --- Throughput ---
    if run_throughput:
        logger.info("Running throughput benchmark...")
        thr = benchmark_throughput(model, tokenizer, calibration, device, args.throughput_tokens)

        f16_tps  = thr.get("f16_baseline", {}).get("tok_per_s", 0)
        spec_tps = thr.get("spectral_cache", {}).get("tok_per_s", 0)
        overhead = ((f16_tps - spec_tps) / f16_tps * 100) if f16_tps > 0 else 0

        print(f"\n=== Throughput ({args.throughput_tokens} tokens) ===")
        print(f"f16 baseline:    {f16_tps:.1f} tok/s")
        print(f"SpectralKVCache: {spec_tps:.1f} tok/s")
        print(f"Overhead:        {overhead:+.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
