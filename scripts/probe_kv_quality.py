"""Capture real K from a HF model and measure planar3 round-trip error.

Targets Qwen3 / Qwen3.5 / Qwen3.6 variants. Auto-detects q_norm/k_norm
presence and, for hybrid models like Qwen3.5, only hooks full_attention
layers (skipping linear-attention / mamba-style layers that don't use a
standard KV cache).

Usage:
    python3 probe_real_k.py <hf_model_id_or_path>

If no argument is given, uses Qwen/Qwen3-4B for quick comparison.
"""

import os
import sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.v1.attention.ops.rotorquant_kv import (  # noqa: E402
    QK_PLANAR3, _round_trip_planar3,
)


def metrics(orig: torch.Tensor, perturbed: torch.Tensor, label: str) -> None:
    a = orig.float().view(-1, QK_PLANAR3)
    b = perturbed.float().view(-1, QK_PLANAR3)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
    bmax = a.abs().max(dim=1, keepdim=True).values
    ne = (a - b).abs() / bmax.clamp(min=1e-6)
    a_norm = torch.linalg.norm(a, dim=1)
    p99 = torch.sort(ne.flatten())[0][int(0.99 * ne.numel())].item()
    print(
        f"  {label:48s}  blocks={a.shape[0]:5d}  "
        f"cos[min={cos.min().item():.4f} mean={cos.mean().item():.4f}]  "
        f"normerr_p99={p99:.4f}  ||x||_med={a_norm.median().item():.3f}"
    )


def synthetic(n_blocks: int = 256) -> None:
    print("synthetic baselines (Gaussian, scale-invariant kernel)")
    for scale in [1.0, 0.1, 0.01]:
        g = torch.Generator(device="cuda").manual_seed(42)
        src = (torch.randn(n_blocks * QK_PLANAR3, generator=g, device="cuda",
                           dtype=torch.float32) * scale).to(torch.float16)
        out = _round_trip_planar3(src)
        metrics(src, out, f"gauss σ={scale}")


def find_attn_layers(model: torch.nn.Module) -> list[tuple[int, str, torch.nn.Module]]:
    """Return [(layer_idx, layer_kind, attn_module), ...] for every layer
    that has a meaningful K projection (skips mamba/linear layers)."""
    out: list[tuple[int, str, torch.nn.Module]] = []
    # Try standard locations.
    layers = None
    for path in ("model.layers", "model.model.layers"):
        cur = model
        ok = True
        for p in path.split("."):
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok:
            layers = cur
            break
    if layers is None:
        raise RuntimeError("could not locate transformer layers")

    cfg = getattr(model, "config", None)
    layer_types = None
    if cfg is not None:
        text_cfg = getattr(cfg, "text_config", cfg)
        layer_types = getattr(text_cfg, "layer_types", None)

    for i, layer in enumerate(layers):
        for attn_path in ("self_attn", "attn", "attention"):
            if hasattr(layer, attn_path):
                attn = getattr(layer, attn_path)
                if any(hasattr(attn, p) for p in ("k_proj", "kv_proj", "wkv")):
                    kind = (layer_types[i] if layer_types and i < len(layer_types)
                            else "attention")
                    out.append((i, kind, attn))
                break
    return out


def capture_real_k(model_id: str, prompt: str = "The capital of France is",
                   max_layers_to_probe: int = 4) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {model_id} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="eager", trust_remote_code=True,
        )
    except Exception as exc:
        print(f"  AutoModelForCausalLM failed: {exc}; trying AutoModel", flush=True)
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda:0",
            trust_remote_code=True,
        )

    layers = find_attn_layers(model)
    print(f"  found {len(layers)} attention layers; layer_types sample: "
          f"{[k for _, k, _ in layers[:8]]}", flush=True)

    # Filter to full_attention layers if hybrid; else take a sample.
    full_attn = [t for t in layers if "full" in t[1].lower() or t[1] == "attention"]
    if not full_attn:
        full_attn = layers
    chosen_idxs = [0, len(full_attn) // 2, len(full_attn) - 1][:max_layers_to_probe]
    chosen = [full_attn[i] for i in chosen_idxs]
    print(f"  probing layers (full_attn idx): {[c[0] for c in chosen]}", flush=True)

    captured: dict[str, torch.Tensor] = {}
    handles: list = []
    for li, kind, attn in chosen:
        prefix = f"L{li:02d} ({kind})"
        if hasattr(attn, "k_proj"):
            def k_hook(_m, _i, out, _p=prefix):
                captured[f"{_p} k_proj"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            handles.append(attn.k_proj.register_forward_hook(k_hook))
        if hasattr(attn, "k_norm"):
            def kn_hook(_m, _i, out, _p=prefix):
                captured[f"{_p} k_norm"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            handles.append(attn.k_norm.register_forward_hook(kn_hook))

    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        try:
            model(**inputs)
        except Exception as exc:
            print(f"  forward failed: {exc}", flush=True)
            raise
    for h in handles:
        h.remove()

    for k, v in captured.items():
        print(f"  {k}: shape={tuple(v.shape)}  σ≈{v.float().std().item():.4f}",
              flush=True)
    return captured


def run(model_id: str) -> None:
    print(f"\n{'=' * 80}\n{model_id}\n{'=' * 80}")
    captures = capture_real_k(model_id)
    if not captures:
        print("  no captures collected")
        return
    for label, t in captures.items():
        flat = t.contiguous().view(-1)
        n = (flat.numel() // QK_PLANAR3) * QK_PLANAR3
        if n == 0:
            print(f"  {label}: skipped (numel < 128)")
            continue
        flat = flat[:n]
        src_fp16 = flat.to(torch.float16)
        out_fp16 = _round_trip_planar3(src_fp16)
        metrics(src_fp16, out_fp16, label)


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("no CUDA")
    print("=" * 80)
    print("synthetic baseline")
    print("=" * 80)
    synthetic()

    model_ids = sys.argv[1:] or ["Qwen/Qwen3-4B"]
    for m in model_ids:
        try:
            run(m)
        except Exception as exc:
            print(f"\n!!! {m}: failed — {type(exc).__name__}: {exc}")
        # Free GPU memory between models.
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
