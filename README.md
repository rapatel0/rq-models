# RotorQuant LLM Server
> NOTE: Gemma Models have not been tested for ppl and qualtity. YMMV

Serve Qwen3.6-35B-A3B, Qwen3.6-27B, Qwen3.5-27B, Gemma 4 26B, and other large models with **large context on a single consumer GPU** using RotorQuant KV cache compression (4.9x at 3-bit, 97% decode speed of fp16).

> **KV defaults**: `iso3` for MoE models (Qwen3.6-35B-A3B), `planar3` for dense models (Qwen3.6-27B, Qwen3.5-27B). Both are 3.125 bpe (4.9x compression). Benchmarks show rotation type matters: iso beats planar on MoE; planar beats iso on dense.


## Quick Start

```bash
# 1. Build
make build

# 2. One-time: source-convert the DFlash drafts (~5 GB total download).
#    The 27B-DFlash repo is gated; visit https://huggingface.co/z-lab/Qwen3.6-27B-DFlash
#    once to request access (the 35B repo is open). Re-run after each upstream
#    draft refresh; idempotent.
make convert-drafts

# 3. Run (first run downloads the target model ~22 GB)
#    `make run-qwen` is now Qwen3.6-35B-A3B MoE + DFlash speculative decoding
#    by default. To skip speculative decoding, use `make run-qwen-target-only`.
make run-qwen

# 4. Query
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"temperature":1.0,"top_p":0.95,"max_tokens":500}'
```

## Available Models

| Command | Model | Weights | Active Params | KV default | Speculative |
|---------|-------|---------|---------------|:----------:|:-----------:|
| `make run-qwen` | **Qwen3.6-35B-A3B** (default) | UD-Q4_K_XL (20.8 GB) | 3B active (MoE) | `iso3` | DFlash |
| `make run-qwen-target-only` | Qwen3.6-35B-A3B (multi-user throughput) | UD-Q4_K_XL (20.8 GB) | 3B active (MoE) | `iso3` | — |
| `docker compose --profile qwen36-q3 up` | Qwen3.6-35B-A3B | UD-Q3_K_XL (16.8 GB) | 3B active (MoE) | `planar4` | — |
| `docker compose --profile qwen36-iq3 up` | Qwen3.6-35B-A3B | UD-IQ3_XXS (13.2 GB) | 3B active (MoE) | `planar4` | — |
| `make run-qwen36-27b` | **Qwen3.6-27B** (dense) | UD-Q4_K_XL (16.4 GB) | 27B dense | `planar3` | — |
| `make run-qwen36-27b-dflash` | Qwen3.6-27B + DFlash (preview) | UD-Q4_K_XL + bf16 draft | 27B + 1B draft | `planar3` | DFlash (PREVIEW=1) |
| `docker compose --profile qwen36-27b-q3 up` | Qwen3.6-27B | UD-Q3_K_XL (~12 GB) | 27B dense | `planar3` | — |
| `docker compose --profile qwen36-27b-iq3 up` | Qwen3.6-27B | UD-IQ3_XXS (~9 GB) | 27B dense | `planar3` | — |
| `make run-reasoning` | Qwen3.5-27B Claude Opus Distilled | i1-Q4_K_M (16.6 GB) | 27B dense | `planar3` | — |
| `make run-gemma` | Gemma 4 26B MoE | UD-Q4_K_XL (17.1 GB) | 3.8B active | `planar4` | — |

Or use Docker Compose directly:

```bash
docker compose --profile qwen up                   # Qwen3.6-35B-A3B Q4 + DFlash (default)
docker compose --profile qwen-target-only up       # Same, but no speculative decoding
docker compose --profile qwen36-q3 up              # Qwen3.6-35B-A3B Q3 (24 GB)
docker compose --profile qwen36-iq3 up             # Qwen3.6-35B-A3B IQ3 (16 GB)
docker compose --profile qwen36-27b up             # Qwen3.6-27B dense (24 GB+), planar3
PREVIEW=1 docker compose --profile qwen36-27b-dflash up  # 27B + DFlash (preview)
docker compose --profile qwen36-27b-q3 up          # Qwen3.6-27B Q3 (16 GB)
docker compose --profile reasoning up              # Qwen3.5-27B reasoning-tuned
docker compose --profile gemma up                  # Gemma 4 MoE
```

### Recommended Sampling (Qwen3.6)

| Use Case | Temperature | Top-P | Top-K | Presence Penalty |
|----------|:-----------:|:-----:|:-----:|:----------------:|
| **General** | 1.0 | 0.95 | 20 | 1.5 |
| **Coding / precise** | 0.6 | 0.95 | 20 | 0.0 |

## Configuration

Override defaults via environment variables:

```bash
# Reduce context for lower VRAM GPUs
CTX_SIZE=32768 make run-qwen

# Disable RotorQuant (fp16 KV cache)
KV_CACHE_TYPE=f16 make run-qwen

# Custom port
PORT=9090 make run-qwen

# Gated model access
HF_TOKEN=hf_xxx make run-reasoning
```

| Variable | Default | Description |
|----------|---------|-------------|
| `KV_CACHE_TYPE` | per-profile | KV cache type: `iso3` (MoE default), `planar3` (dense default), `planar4`, `iso4`, `f16` |
| `CTX_SIZE` | per-model | Context window (e.g. 114688 for Q4_K_M on 24 GB) |
| `PORT` | `8080` | API port |
| `GPU_LAYERS` | `99` | Layers on GPU (99 = all) |
| `N_PARALLEL` | `2` | Concurrent request slots (set higher for throughput mode) |
| `CACHE_RAM` | `8192` | Prompt cache size in MiB (system RAM, not VRAM) |
| `HF_TOKEN` | — | HuggingFace token for gated models |

## Best Config by GPU

Recommended configurations per VRAM tier, based on measured perplexity
(see [docs/BENCHMARK-REPORT.md](docs/BENCHMARK-REPORT.md) and [docs/QUANTIZATION-GUIDE.md](docs/QUANTIZATION-GUIDE.md) for full data).

### 32 GB+ (RTX 5090) — Recommended Default

**Use Qwen3.6-35B-A3B** (`make run-qwen`). 196 tok/s decode, 262K context, iso3 KV (4.9x compression).

| Use Case | Quant | Size | KV | Context | Command |
|----------|-------|-----:|:--:|--------:|---------|
| **Default** | **UD-Q4_K_XL** | **20.8 GB** | **iso3** | **262K** | **`make run-qwen`** |
| Max context | UD-Q4_K_XL | 20.8 GB | iso3 | 262K × 2 slots | default |
| Max throughput | UD-Q4_K_XL | 20.8 GB | iso3 | 65K × 8 slots | `--profile qwen36-throughput` |

### 24 GB (RTX 4090, RTX 3090) — Qwen3.6-27B Dense

**Use Qwen3.6-27B** (`--profile qwen36-27b`). 131K context, planar3 KV (4.9x compression).

| Use Case | Quant | Size | KV | Context |
|----------|-------|-----:|:--:|--------:|
| **Recommended** | **UD-Q4_K_XL** | **16.4 GB** | **planar3** | **131K** |
| More context | UD-Q3_K_XL | ~12 GB | planar3 | ~180K |
| Max context | UD-IQ3_XXS | ~9 GB | planar3 | ~240K |

### 16 GB (RTX 4060 Ti, RTX 5060, RTX 4080) — Qwen3.5-27B Quants

Qwen3.5-27B Q4_K_M doesn't fit. Use Unsloth imatrix quants (Qwen3.5-27B):

| Use Case | Quant | Size | PPL | Context (planar3) |
|----------|-------|-----:|----:|------------------:|
| **Best quality** | IQ4_XS | 13.9 GB | 6.29 | ~18K |
| **Recommended** | **UD-Q3_K_XL** | **13.4 GB** | **6.38** | **~36K** |
| Max context | UD-IQ3_XXS | 10.7 GB | 6.62 | ~74K |

> UD-Q3_K_XL is the sweet spot — only +0.08 PPL over 4-bit, 2× more context.

### Quick Reference

| GPU | Model | Quant | PPL | KV | Context | Command |
|-----|-------|-------|----:|:--:|--------:|---------|
| **32 GB** | Qwen3.6-35B-A3B | UD-Q4_K_XL | 6.13 | iso3 | 262K | `make run-qwen` |
| **24 GB** | Qwen3.6-27B | UD-Q4_K_XL | 7.09 | planar3 | 131K | `--profile qwen36-27b` |
| **24 GB** | Qwen3.6-27B | UD-Q3_K_XL | — | planar3 | ~180K | `--profile qwen36-27b-q3` |
| **16 GB** | Qwen3.6-35B-A3B | UD-IQ3_XXS | — | planar4 | ~32K | `--profile qwen36-iq3` |
| **16 GB** | Qwen3.6-27B | UD-IQ3_XXS | — | planar3 | ~90K | `--profile qwen36-27b-iq3` |
| **16 GB** | Qwen3.5-27B | UD-Q3_K_XL | 6.38 | planar3 | ~36K | `--profile qwen-q3` |
| **16 GB** | Qwen3.5-27B | UD-IQ3_XXS | 6.62 | planar3 | ~74K | `--profile qwen-q3-xxs` |
| **16 GB** | Gemma4 26B | UD-Q3_K_M | — | planar4 | ~40K | `--profile gemma-q3` |

PPL values are f16 baseline (wikitext-2). All profiles auto-download the correct model on first run.

### Throughput Mode (Parallel Slots)

For serving multiple concurrent users, trade context length for parallel slots.
During decode, single-user inference is memory-bandwidth bound (mat-vec). With
N parallel slots, the weight multiplications become mat-mat and tensor cores
engage — aggregate throughput scales near-linearly.

#### Qwen3.6-35B-A3B MoE — measured on RTX 5090 (iso3, 65K ctx/slot)

| Slots | Per-slot tok/s | Aggregate tok/s | Profile |
|:-----:|:--------------:|:---------------:|---------|
| 1 | 193 | 193 | `--profile qwen` |
| 2 | 150 | **299** | `--profile qwen` (default) |
| 4 | 99 | **397** | `--profile qwen36-throughput` |
| 8 | 54 | **431** | `--profile qwen36-throughput` |

P=4 is the practical optimum: 397 tok/s aggregate at 99 tok/s per slot.

```bash
docker compose --profile qwen36-throughput up   # 8 slots × 65K ctx
```

#### Qwen3.5-27B Dense — estimated on RTX 5090 (planar4, 16K ctx/slot)

RotorQuant is the multiplier: planar4 KV at 16K = 1.06 GB/slot vs f16 = 4.0 GB/slot — **3.8× more concurrent users at the same VRAM.**

| GPU | Slots (planar4) | Slots (f16) | Est. Aggregate tok/s | Command |
|-----|-------------:|------------:|---------------------:|---------|
| **24 GB** | 6 | 1 | ~320 | `N_PARALLEL=6 docker compose --profile qwen-throughput up` |
| **32 GB** | 14 | 3 | ~660 | `docker compose --profile qwen-throughput up` |
| **40 GB** | 22 | 5 | ~880 | `N_PARALLEL=22 docker compose --profile qwen-throughput up` |
| **80 GB** | 59 | 15 | ~2,400 | `N_PARALLEL=59 docker compose --profile qwen-throughput up` |

```bash
make run-throughput   # Qwen3.5-27B, 14 slots × 16K, RTX 5090
```

## Requirements

- Docker 24+
- Docker Compose v2.20+
- NVIDIA Container Toolkit (`nvidia-ctk`)
- NVIDIA driver 570+ (CUDA 13.1 support)
- GPU with 16+ GB VRAM (RTX 5060/4060 Ti with Q3 quants, RTX 4090+, A100, H100)

## Performance

Benchmarked on RTX 5090 (32 GB). Full results in [docs/BENCHMARK-REPORT.md](docs/BENCHMARK-REPORT.md).

### Qwen3.6-35B-A3B MoE (default, `--profile qwen`)

| Metric | iso3 (default) | iso4 | planar3 | f16 |
|--------|:-:|:-:|:-:|:-:|
| Decode speed (1 slot) | **196 tok/s** | ~196 tok/s | ~196 tok/s | ~196 tok/s |
| KV compression | 4.9x | 3.8x | 4.9x | 1x |
| Max ctx (RTX 5090, 2 slots) | **262K** | ~190K | 262K | ~30K |
| PPL wikitext-2 (ctx=2048) | 6.25 | **6.23** | 6.29 | **6.13** |
| NIAH recall | — | — | — | — |

> `iso3` and `iso4` both beat `planar` on this Gated Delta Net MoE architecture. `iso3` is the default: same context as iso4 with 4.9× vs 3.8× compression.

### Qwen3.5-27B Q4_K_M (legacy dense, `--profile reasoning`)

| Metric | planar3 (default) | planar4 | iso3 | f16 |
|--------|:-:|:-:|:-:|:-:|
| Decode speed | ~67 tok/s | ~67 tok/s | 67.5 tok/s | 69.3 tok/s |
| KV compression | **4.9x** | 3.8x | 4.9x | 1x |
| Max ctx (RTX 5090) | **300K** | 236K | 300K | ~45K |
| PPL wikitext-2 (ctx=2048) | **7.01** | 7.02 | 7.10 | 6.64 |
| NIAH recall (4K) | 100% | 100% | 100% | 100% |

> `planar3` wins on all dense models: same compression as iso3 (4.9×) but lower PPL.

### Speculative Decoding

DFlash block-diffusion speculative decoding is **on by default** for the 35B
MoE — `make run-qwen` runs target+DFlash, single-slot. The dense 27B has a
preview-gated DFlash profile because its draft training is still iterating
upstream.

| Profile | Target | Draft | KV | Ctx | Status |
|---------|--------|-------|:--:|----:|:------:|
| `qwen` (default) | Qwen3.6-35B-A3B (MoE) | source-converted from `z-lab/Qwen3.6-35B-A3B-DFlash` | iso3 | 131K | shipped |
| `qwen-target-only` | Qwen3.6-35B-A3B (MoE) | — | iso3 | 524K (P=2) | shipped, opt-in |
| `qwen36-27b-dflash` | Qwen3.6-27B (dense) | source-converted from `z-lab/Qwen3.6-27B-DFlash` | planar3 | 131K | preview (`PREVIEW=1`) |

`qwen-target-only` is the escape hatch when you'd rather have 524K context
and 2 concurrent slots than DFlash's single-slot speculative gain — DFlash
serializes through one shared draft `llama_context`, so multi-user throughput
is sub-linear vs target-only multi-slot today.

The 27B-DFlash draft is preview because z-lab is still iterating; re-running
`make convert-drafts` after each upstream draft refresh is expected.

```bash
# One-time: source-convert drafts from z-lab safetensors into the llm-models volume.
# 27B repo is gated; visit https://huggingface.co/z-lab/Qwen3.6-27B-DFlash to
# request access (one-time per HF account).
make convert-drafts

# Default — MoE + DFlash
make run-qwen

# Multi-user throughput escape hatch (no speculative)
make run-qwen-target-only

# Dense 27B + DFlash (preview)
make run-qwen36-27b-dflash      # implicitly sets PREVIEW=1
```

#### Env var contract

Set on the compose service or via `docker run -e` (full table in
[docker/entrypoint.sh:18-23](docker/entrypoint.sh#L18-L23)):

| Variable | Values | Default | Notes |
|----------|--------|---------|-------|
| `SPECULATIVE_MODE` | `target-only` / `autoregressive` / `dflash` | `target-only` | Selects the verify path |
| `DRAFT_MODEL_NAME` | model key from `MODELS` registry | — | Required if mode != target-only |
| `DRAFT_KV_CACHE_TYPE` | same options as `KV_CACHE_TYPE` | inherits target | Independent draft KV quantization |
| `DRAFT_N_MAX` | int | `16` | Max draft tokens per verify round |
| `PREVIEW` | `0` / `1` | `0` | Required `=1` to enable `qwen36-27b-dflash` (z-lab drafts iterating) |

The entrypoint defaults `N_PARALLEL=1` when `SPECULATIVE_MODE != target-only` but
no longer forces it. Multi-slot speculative is functionally correct in the
cherry-picked PR #22105 — each slot has its own `common_speculative` context
([server-context.cpp:928](https://github.com/rapatel0/llama-cpp-turboquant/blob/feature/sprint-004-rebase-dflash/tools/server/server-context.cpp#L928))
— but all slots serialize through one shared draft `llama_context`
(`params_dft.n_parallel = 1` at server-context.cpp:779). The optimization to
batch all slots' draft inference together is upstream's `TAG_SERVER_SPEC_REWORK`
TODO. Throughput experiments with `N_PARALLEL > 1` are operator-overridable
but currently expected to scale sub-linearly with concurrent slots.

#### Acceptance-rate tuning

`LLAMA_SPEC_NO_THINK=1` (read in `examples/speculative-simple/speculative-simple.cpp`,
came in via PR #22105) suppresses Qwen3.x thinking-mode tokens. Per the upstream
PR, leaving thinking on drops acceptance rate by 60–80 percentage points relative
to a no-think baseline. **Sprint 004 measures with thinking on** because thinking
mode is critical to Qwen3.x quality on actual tasks; suppressing it for benchmark
optics would not reflect deployed behavior. Operators who specifically want PR
#22105's headline numbers can set `LLAMA_SPEC_NO_THINK=1` to reproduce them.

#### Validation scope

This sprint validates greedy decoding only: `--temp 0 --top-k 1 --seed 42`.
Sampling-mode (temp > 0) speculative behavior is unverified — token sequences
diverge by design under sampling, so equivalence requires distribution-level
metrics that are out of scope (see SPRINT-004-DEFERRED.md D-003). Streaming
(`stream: true`) is similarly deferred (D-006).

#### Source-converted drafts (`make convert-drafts`)

`scripts/convert_dflash_drafts.sh` downloads the z-lab safetensors plus the
target's tokenizer files (no target weights needed), runs the cherry-picked
PR #22105's `convert_hf_to_gguf.py` against them, and copies the result GGUFs
into the `llm-models` named volume so the compose profiles can find them. The
script is idempotent — re-running skips repos already on disk and only
re-publishes if the volume copy differs.

The 27B repo (`z-lab/Qwen3.6-27B-DFlash`) is gated; the operator must visit
the HF page once and request access using the same account as
`~/.cache/huggingface/token`. The 35B repo is public.

## Project Structure

```
.
├── docker-compose.yml        # Compose profiles: qwen, reasoning, gemma
├── Makefile                  # build, run-qwen, run-reasoning, run-gemma, test, bench
├── docker/
│   ├── Dockerfile            # Multi-stage CUDA 13.1 build from source
│   ├── entrypoint.sh         # Model download + config + server launch
│   └── test.sh               # Automated smoke test
├── scripts/
│   ├── battery_test.py       # Performance + competence test suite
│   ├── benchmark_rotorquant.py  # Full benchmark (throughput, PPL, NIAH)
│   └── bench_tps.sh          # Quick tok/s benchmark via llama-bench
├── turboquant/               # Python TurboQuant library (research)
│   ├── core.py               # TurboQuantMSE, TurboQuantProd
│   ├── kv_cache.py           # HuggingFace DynamicCache subclass
│   └── ...
├── tests/                    # Python unit tests (pytest)
└── docs/
    ├── BENCHMARK-REPORT.md   # Full benchmark results
    └── sprints/              # Sprint planning documents
```

## How It Works

[RotorQuant](https://github.com/scrya-com/rotorquant) replaces the dense d×d rotation matrix in [TurboQuant](https://arxiv.org/abs/2504.19874) (ICLR 2026) with Clifford algebra rotors operating on groups of 3-4 dimensions. This gives:

- **44x fewer parameters** (372 vs 16,384 for d=128)
- **10-19x faster** fused CUDA kernels (no cuBLAS GEMM, all in-register)
- **Identical attention fidelity** on real model data

The KV cache is compressed from 16-bit to 3-bit per element, reducing memory by ~5x. Decode speed is unaffected because attention computation dominates — the dequantization overhead is hidden.

## License

MIT
