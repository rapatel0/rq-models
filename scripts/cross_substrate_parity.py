"""Cross-substrate bit-parity test for the planar3 KV kernel.

Captures a real K tensor from a HF model layer (or generates synthetic
input), saves it as a raw fp16 binary, runs it through the rq-vllm
CUDA kernel, and compares the round-tripped output against a reference
binary produced by an independent llama.cpp-turboquant build of the
same kernel. Two implementations of the same math should produce
bit-identical output (or at least within the floating-point determinism
guarantee of each substrate).

This is the strongest "no-bug" test we can run for the kernel: synthetic
parity tests prove the math is correct *vs. the design intent*, but a
cross-substrate bit-parity test proves our CUDA port matches the
reference C implementation for any input distribution including
pathological ones (Qwen3 L0 k_norm, etc.).

## Usage

Step 1 — capture input + run our kernel:

    python3 scripts/cross_substrate_parity.py capture \\
        --model Qwen/Qwen3-4B \\
        --layer 0 --hook k_norm \\
        --in /tmp/k_in.fp16 --out /tmp/k_out_rq.fp16

Step 2 — run llama.cpp-turboquant reference on the same input. See
    scripts/cross_substrate_ref/README.md for build + invocation:

    /tmp/rq_planar3_ref /tmp/k_in.fp16 /tmp/k_out_ref.fp16

Step 3 — diff:

    python3 scripts/cross_substrate_parity.py diff \\
        --rq /tmp/k_out_rq.fp16 --ref /tmp/k_out_ref.fp16

Or run all three in one go (requires the reference binary on $PATH):

    python3 scripts/cross_substrate_parity.py all \\
        --model Qwen/Qwen3-4B --layer 0 --hook k_norm \\
        --ref-bin /tmp/rq_planar3_ref
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
from vllm.v1.attention.ops.rotorquant_kv import (  # noqa: E402
    QK_PLANAR3, _round_trip_planar3,
)


# ---------------------------------------------------------------------------
# Binary protocol — keep this stable so any reference impl can interop.
#
#   bytes 0..3   : magic "RQK1"
#   bytes 4..7   : little-endian uint32, n_blocks
#   bytes 8..N   : n_blocks * 128 * 2 bytes of fp16, row-major
#
# A round-trip output file uses the same layout. The reference binary
# is responsible for matching it byte-for-byte.
# ---------------------------------------------------------------------------

MAGIC = b"RQK1"


def write_fp16_blob(path: Path, t: torch.Tensor) -> None:
    if t.dtype != torch.float16:
        t = t.to(torch.float16)
    flat = t.contiguous().view(-1).cpu()
    n = flat.numel()
    if n % QK_PLANAR3 != 0:
        raise ValueError(f"numel {n} not multiple of {QK_PLANAR3}")
    n_blocks = n // QK_PLANAR3
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", n_blocks))
        f.write(flat.numpy().tobytes())


def read_fp16_blob(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic {magic!r} in {path}")
        (n_blocks,) = struct.unpack("<I", f.read(4))
        n = n_blocks * QK_PLANAR3
        buf = f.read(n * 2)
        if len(buf) != n * 2:
            raise ValueError(f"truncated blob in {path}: "
                             f"expected {n * 2} bytes, got {len(buf)}")
    arr = np.frombuffer(buf, dtype=np.float16).copy()
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_k(model_id: str, layer: int, hook: str,
              prompt: str = "The capital of France is") -> torch.Tensor:
    """Pull the K tensor at the given layer and hook point."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {model_id} ...", file=sys.stderr, flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager", trust_remote_code=True,
    )

    # Locate transformer layers.
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

    attn = layers[layer].self_attn
    target = getattr(attn, hook, None)
    if target is None:
        raise RuntimeError(f"layer {layer} attn has no .{hook} module")

    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        captured["k"] = t.detach().clone()

    handle = target.register_forward_hook(hook_fn)
    inputs = tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    if "k" not in captured:
        raise RuntimeError(f"hook on .{hook} did not fire")
    k = captured["k"]
    print(f"captured {k.shape} dtype={k.dtype} σ={k.float().std().item():.4f}",
          file=sys.stderr, flush=True)
    return k


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def diff(rq_path: Path, ref_path: Path) -> int:
    a = read_fp16_blob(rq_path)
    b = read_fp16_blob(ref_path)
    if a.shape != b.shape:
        print(f"shape mismatch: rq {a.shape} vs ref {b.shape}")
        return 1
    af = a.float()
    bf = b.float()
    diff_t = (af - bf).abs()
    n_eq = int((af == bf).sum().item())
    n = af.numel()
    print(f"elements        : {n}")
    print(f"bit-identical   : {n_eq} ({100 * n_eq / n:.4f} %)")
    print(f"abs diff max    : {diff_t.max().item():.6e}")
    print(f"abs diff mean   : {diff_t.mean().item():.6e}")
    print(f"abs diff p99    : {diff_t.flatten().sort().values[int(0.99*n)].item():.6e}")
    rel = diff_t / af.abs().clamp(min=1e-6)
    print(f"rel diff max    : {rel.max().item():.6e}")
    print(f"rel diff mean   : {rel.mean().item():.6e}")
    # ULP-style diff via fp16 bit pattern.
    a_u16 = a.view(torch.uint16).int()
    b_u16 = b.view(torch.uint16).int()
    ulp = (a_u16 - b_u16).abs()
    print(f"ulp diff max    : {ulp.max().item()}")
    print(f"ulp diff mean   : {ulp.float().mean().item():.4f}")
    return 0 if diff_t.max().item() == 0.0 else 2


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def cmd_capture(args: argparse.Namespace) -> int:
    if args.synthetic:
        n_blocks = int(args.synthetic)
        g = torch.Generator(device="cuda").manual_seed(42)
        k = (torch.randn(n_blocks * QK_PLANAR3, generator=g, device="cuda",
                         dtype=torch.float32)).to(torch.float16)
    else:
        k = capture_k(args.model, args.layer, args.hook)
        # Trim to a multiple of 128, cast to fp16.
        flat = k.contiguous().view(-1)
        n = (flat.numel() // QK_PLANAR3) * QK_PLANAR3
        k = flat[:n].to(torch.float16)

    write_fp16_blob(Path(args.in_), k)
    out = _round_trip_planar3(k.cuda())
    write_fp16_blob(Path(args.out), out)
    print(f"input  → {args.in_}", file=sys.stderr)
    print(f"output → {args.out}", file=sys.stderr)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    return diff(Path(args.rq), Path(args.ref))


def cmd_all(args: argparse.Namespace) -> int:
    in_path = Path("/tmp/cross_in.fp16")
    rq_path = Path("/tmp/cross_out_rq.fp16")
    ref_path = Path("/tmp/cross_out_ref.fp16")

    sub = argparse.Namespace(
        in_=str(in_path), out=str(rq_path),
        model=args.model, layer=args.layer, hook=args.hook,
        synthetic=args.synthetic,
    )
    rc = cmd_capture(sub)
    if rc != 0:
        return rc

    print(f"\nrunning reference: {args.ref_bin} {in_path} {ref_path}",
          file=sys.stderr)
    rc = subprocess.call([args.ref_bin, str(in_path), str(ref_path)])
    if rc != 0:
        print(f"reference binary exited with {rc}", file=sys.stderr)
        return rc

    return diff(rq_path, ref_path)


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("capture", help="capture K + run our kernel")
    pc.add_argument("--model", default="Qwen/Qwen3-4B")
    pc.add_argument("--layer", type=int, default=0)
    pc.add_argument("--hook", default="k_norm",
                    help="attribute on attn module to hook (k_proj, k_norm)")
    pc.add_argument("--synthetic", default=None,
                    help="if set, skip model load and use N synthetic blocks")
    pc.add_argument("--in", dest="in_", required=True)
    pc.add_argument("--out", required=True)
    pc.set_defaults(func=cmd_capture)

    pd = sub.add_parser("diff", help="diff our output against reference output")
    pd.add_argument("--rq", required=True)
    pd.add_argument("--ref", required=True)
    pd.set_defaults(func=cmd_diff)

    pa = sub.add_parser("all", help="capture + run reference + diff")
    pa.add_argument("--model", default="Qwen/Qwen3-4B")
    pa.add_argument("--layer", type=int, default=0)
    pa.add_argument("--hook", default="k_norm")
    pa.add_argument("--synthetic", default=None)
    pa.add_argument("--ref-bin", default="/tmp/rq_planar3_ref")
    pa.set_defaults(func=cmd_all)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
