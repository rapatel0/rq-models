# Cross-substrate parity reference (planar3)

Tiny C driver that emulates llama-cpp-turboquant's **CUDA-path**
planar3 round-trip: the same algorithm as `ggml-planar-quant.c` but
using the rotation constants from
`ggml-cuda/planar-iso-constants.cuh` (`PI_COS` / `PI_SIN`), which is
the exact table our rq-vllm CUDA kernel ports byte-for-byte. Companion
to `scripts/cross_substrate_parity.py`.

## Why

Synthetic-Gaussian parity tests prove the kernel does what the design
intends. They don't catch porting bugs that only manifest on
distributions outside the synthetic test corpus (e.g., the Qwen3 L0
post-`k_norm` K with σ=13.4 and strong anisotropy).

Running both substrates on the same input and comparing element-wise
is the strongest "no-bug" check for the rq-vllm CUDA port: any
divergence means we shipped a port bug. Bit-identity (or fp-equivalence
within 1–2 ULPs) means we shipped a port that matches the reference
exactly — and the Qwen3 quality cliff is purely a calibration limit
of the kernel design, not a bug we introduced.

## Important: CPU vs CUDA path inconsistency in upstream

`johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache` ships
*two different rotation tables*: one in `ggml-planar-quant.c` (CPU
path, first cos value `0.7386546135`) and one in
`ggml-cuda/planar-iso-constants.cuh` (CUDA path, first cos value
`-0.9095053397`). They're both PRNG-seeded but with different
sequences. KV cache quantized on the CPU path and decoded on the CUDA
path will not round-trip correctly within upstream itself.

Our CUDA kernel ports the **CUDA-path** constants (verified — first cos
value `-0.9095053397`). So the right oracle for cross-substrate parity
testing rq-vllm is a C emulator using the **CUDA-path** table, not the
CPU `ggml-planar-quant.c` source. That's what `rq_planar3_ref.c` here
does.

## Build

```bash
gcc -O3 -std=c11 scripts/cross_substrate_ref/rq_planar3_ref.c \
    -lm -o /tmp/rq_planar3_ref
```

No external dependencies — the rotation constants and centroids are
inlined verbatim from upstream's `planar-iso-constants.cuh`. The
algorithm matches `kernel_cpy_f16_planar3` (encode) and
`dequantize_planar3_0` (decode) from upstream's CUDA path.

## Run

Synthetic input first (sanity check that the build works):

```bash
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp:/host_tmp \
    -v $(pwd)/scripts/cross_substrate_parity.py:/probe.py \
    --entrypoint python3 rq-vllm:latest /probe.py capture \
    --synthetic 256 \
    --in /host_tmp/k_in.fp16 --out /host_tmp/k_out_rq.fp16

/tmp/rq_planar3_ref /tmp/k_in.fp16 /tmp/k_out_ref.fp16

docker run --rm --gpus all \
    -v /tmp:/host_tmp \
    -v $(pwd)/scripts/cross_substrate_parity.py:/probe.py \
    --entrypoint python3 rq-vllm:latest /probe.py diff \
    --rq /host_tmp/k_out_rq.fp16 --ref /host_tmp/k_out_ref.fp16
```

Verified on 2026-04-27 (RTX 4090, CUDA 12.9.86, vLLM v0.19.1, our
rq-vllm@feature/rotorquant build):

```
elements        : 32768
bit-identical   : 32767 (99.9969 %)
abs diff max    : 4.882812e-04   ← single 1-ULP fp16 step
abs diff mean   : 1.490116e-08
ulp diff max    : 1
ulp diff mean   : 0.0000
```

99.997 % bit-identical with a single 1-ULP discrepancy out of 32 768
elements — exactly the level of fp-arithmetic-reordering noise
expected between CUDA's FMA pipeline and our C reference's standard
arithmetic. **No port bug.**

## Run on a real-K stress case

The interesting test is feeding the kernel actual model K — especially
distributions like Qwen3-4B L0 post-`k_norm` where the codebook has
trouble representing direction. Both substrates should fail in the
*exact same way*; if they don't, our port has a bug specific to the
distribution.

```bash
docker run --rm --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /tmp:/host_tmp \
    -v $(pwd)/scripts/cross_substrate_parity.py:/probe.py \
    --entrypoint python3 rq-vllm:latest /probe.py capture \
    --model Qwen/Qwen3-4B --layer 0 --hook k_norm \
    --in /host_tmp/k_q3_in.fp16 --out /host_tmp/k_q3_rq.fp16

/tmp/rq_planar3_ref /tmp/k_q3_in.fp16 /tmp/k_q3_ref.fp16

docker run --rm --gpus all \
    -v /tmp:/host_tmp \
    -v $(pwd)/scripts/cross_substrate_parity.py:/probe.py \
    --entrypoint python3 rq-vllm:latest /probe.py diff \
    --rq /host_tmp/k_q3_rq.fp16 --ref /host_tmp/k_q3_ref.fp16
```

Verified on 2026-04-27, same setup:

```
elements        : 5120
bit-identical   : 5120 (100.0000 %)
abs diff max    : 0.000000e+00
abs diff mean   : 0.000000e+00
ulp diff max    : 0
ulp diff mean   : 0.0000
```

**100.0000 % bit-identical, 0 ULP diff** on the worst-case Qwen3 L0
distribution that the codebook itself can't represent (cos-sim
0.666 vs the original input). Both substrates produce the same broken
output for the same broken input — that's the signature of a clean
port. The Qwen3 quality cliff is exactly what the kernel's
calibration says it should be, not a bug we introduced.

## What "passing" tells you

| Diff outcome                | Interpretation                                                    |
|-----------------------------|-------------------------------------------------------------------|
| 100 % bit-identical         | rq-vllm CUDA port is exactly equivalent to the C reference. No port bug. |
| ulp diff ≤ 2 per element    | Acceptable fp-arithmetic reordering between CUDA and CPU implementations. |
| ulp diff > 2 (or nonzero p99 abs) | **Port bug.** The CUDA implementation diverges from the reference for this input distribution. Investigate. |
