# Cross-substrate parity reference

Tiny C driver that calls llama.cpp-turboquant's `quantize_row_planar3` /
`dequantize_row_planar3` on a binary blob and writes the round-tripped
output. Companion to `scripts/cross_substrate_parity.py`.

## Why

Synthetic-Gaussian parity tests prove the kernel does what the design
intends. They don't catch porting bugs that only manifest on
distributions outside the synthetic test corpus (e.g., the Qwen3 L0
post-`k_norm` K with σ=13.4 and strong anisotropy).

Running both substrates on the same input and comparing element-wise
is the strongest "no-bug" check for the rq-vllm CUDA port: any
divergence means we shipped a port bug. Bit-identity (or fp-equivalence
within an ULP or two) means we shipped a port that matches the
reference exactly — and the Qwen3 quality cliff is purely a calibration
limit of the kernel design, not a bug we introduced.

## Build

Assumes you have `johndpope/llama-cpp-turboquant` cloned and built
locally with planar3 support (commit `fc3d1b6` /
`feature/planarquant-kv-cache` is the reference we ported from).

```bash
# Reference repo — kernel C source.
git clone git@github.com:johndpope/llama-cpp-turboquant.git ~/repos/llama-cpp-turboquant
cd ~/repos/llama-cpp-turboquant
git checkout fc3d1b6           # known-good commit; or feature/planarquant-kv-cache HEAD
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Build the cross-substrate driver against the reference's libggml.a.
cd ~/repos/rq-models
gcc -O3 \
    -I ~/repos/llama-cpp-turboquant/ggml/include \
    -I ~/repos/llama-cpp-turboquant/ggml/src \
    scripts/cross_substrate_ref/rq_planar3_ref.c \
    ~/repos/llama-cpp-turboquant/build/ggml/src/libggml.a \
    -lm -o /tmp/rq_planar3_ref

# Quick sanity — should print usage.
/tmp/rq_planar3_ref
```

If `quantize_row_planar3_reference` / `dequantize_row_planar3` are
named differently on your fork, adjust the symbol names in
`rq_planar3_ref.c`. The test will fail to link with a clear error if
the symbols are wrong.

## Run

Synthetic input first (cheapest sanity check — confirms the build
works at all):

```bash
python3 scripts/cross_substrate_parity.py capture \
    --synthetic 256 \
    --in /tmp/k_in.fp16 --out /tmp/k_out_rq.fp16
/tmp/rq_planar3_ref /tmp/k_in.fp16 /tmp/k_out_ref.fp16
python3 scripts/cross_substrate_parity.py diff \
    --rq /tmp/k_out_rq.fp16 --ref /tmp/k_out_ref.fp16
```

Expected output on a clean port:

```
elements        : 32768
bit-identical   : 32768 (100.0000 %)
abs diff max    : 0.000000e+00
abs diff mean   : 0.000000e+00
abs diff p99    : 0.000000e+00
rel diff max    : 0.000000e+00
rel diff mean   : 0.000000e+00
ulp diff max    : 0
ulp diff mean   : 0.0000
```

A nonzero diff means the CUDA port diverges from the C reference for
that input — that's a real port bug to chase down. Per-block fp16
arithmetic is associative-up-to-rounding, so a small ULP diff (1–2 ULPs
per element) is acceptable; anything larger needs investigation.

## Run on a real-K stress case

The interesting test is feeding the kernel actual model K — especially
distributions like Qwen3-4B L0 post-k_norm where the codebook has
trouble representing direction. Both substrates should fail in the
*exact same way*; if they don't, our port has a bug specific to the
distribution (e.g., overflow in the rotation, bad signed/unsigned
codebook index, etc.).

```bash
# Capture Qwen3-4B L0 post-k_norm K and round-trip through our kernel.
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/repos/rq-models/scripts:/scripts \
    --entrypoint python3 rq-vllm:latest /scripts/cross_substrate_parity.py \
    capture --model Qwen/Qwen3-4B --layer 0 --hook k_norm \
    --in /tmp/k_qwen3_in.fp16 --out /tmp/k_qwen3_rq.fp16

# Run the C reference on the same input.
/tmp/rq_planar3_ref /tmp/k_qwen3_in.fp16 /tmp/k_qwen3_ref.fp16

# Diff.
python3 scripts/cross_substrate_parity.py diff \
    --rq /tmp/k_qwen3_rq.fp16 --ref /tmp/k_qwen3_ref.fp16
```

Both substrates should emit *the same broken output* (cos-sim ≈ 0.67
vs the original input — that's the codebook hitting its calibration
limit). The bit-parity check is between rq-vllm's CUDA output and
llama.cpp-turboquant's C output, not against the original K.

## What "passing" tells you

| Diff outcome                | Interpretation                                                    |
|-----------------------------|-------------------------------------------------------------------|
| 100 % bit-identical         | rq-vllm CUDA port is exactly equivalent to the C reference. No port bug. |
| ulp diff ≤ 2 per element    | Acceptable fp-arithmetic reordering between CUDA and CPU implementations. |
| ulp diff > 2 (or nonzero p99 abs) | **Port bug.** The CUDA implementation diverges from the reference for this input distribution. Investigate. |

The Qwen3 quality story (cos-sim 0.67 on real K) is *upstream* of this
diff: it's about how well the codebook represents the input, not about
whether the two substrates agree. They agree on broken output for
broken input. That's the whole point — calibration is the limit, not
the kernel.
