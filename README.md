# RotorQuant LLM Server
> NOTE: Gemma Models have not been tested for ppl and qualtity. YMMV

Serve Qwen3.6-35B-A3B, Qwen3.6-27B, Qwen3.5-27B, Gemma 4 26B, and other large models with **large context on a single consumer GPU** using RotorQuant KV cache compression (4.9x at 3-bit, 97% decode speed of fp16). Qwen3.6 MTP profiles are included for speculative decoding on MTP-capable llama.cpp builds.

> **KV defaults**: `iso3` for MoE models (Qwen3.6-35B-A3B), `planar3` for dense models (Qwen3.6-27B, Qwen3.5-27B). Both are 3.125 bpe (4.9x compression). Benchmarks show rotation type matters: iso beats planar on MoE; planar beats iso on dense.

### Inference engineering reference

Before doing perf work or planning a new substrate, read:
- [`docs/INFERENCE_LESSONS.md`](docs/INFERENCE_LESSONS.md) — verdicts on
  optimization techniques (what works, doesn't, when), profile-first
  decision tree, and roadmap recommendations carried over from the
  vortex Sprint 024 investigation.
- [`docs/THROUGHPUT_CONFIGURATION_MODEL.md`](docs/THROUGHPUT_CONFIGURATION_MODEL.md)
  — bandwidth-vs-compute regime model for picking
  resident-vs-streaming × N-parallel × ctx configurations.

Constants in those docs were measured on RTX 4090 / Qwen 27B; recalibrate
on RTX 5090 / Qwen 3.6 before citing.


## Quick Start

```bash
# 1. Build
make build

# 2. Run (first run downloads the model ~22 GB)
make run-qwen

# Or run the MTP profile for speculative decoding (one slot)
make run-qwen-mtp

# 27B dense MTP speed profile for 24 GB GPUs
make run-qwen36-27b-mtp-speed

# 3. Query
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"temperature":1.0,"top_p":0.95,"max_tokens":500}'
```

## Available Models

| Command | Model | Weights | Active Params | KV default |
|---------|-------|---------|---------------|:----------:|
| `make run-qwen` | **Qwen3.6-35B-A3B** (default) | UD-Q4_K_XL (20.8 GB) | 3B active (MoE) | `iso3` |
| `make run-qwen-mtp` | **Qwen3.6-35B-A3B MTP** | UD-Q4_K_XL (MTP) | 3B active (MoE) | `iso3` |
| `docker compose --profile qwen36-q3 up` | Qwen3.6-35B-A3B | UD-Q3_K_XL (16.8 GB) | 3B active (MoE) | `planar4` |
| `docker compose --profile qwen36-iq3 up` | Qwen3.6-35B-A3B | UD-IQ3_XXS (13.2 GB) | 3B active (MoE) | `planar4` |
| `make run-qwen36-27b` | **Qwen3.6-27B** (dense) | UD-Q4_K_XL (16.4 GB) | 27B dense | `planar3` |
| `make run-qwen36-27b-mtp` | **Qwen3.6-27B MTP** | UD-Q4_K_XL (MTP) | 27B dense | `planar3` |
| `make run-qwen36-27b-mtp-speed` | **Qwen3.6-27B MTP speed** | UD-Q4_K_XL (MTP) | 27B dense | `q4_0` |
| `docker compose --profile qwen36-27b-q3 up` | Qwen3.6-27B | UD-Q3_K_XL (~12 GB) | 27B dense | `planar3` |
| `docker compose --profile qwen36-27b-iq3 up` | Qwen3.6-27B | UD-IQ3_XXS (~9 GB) | 27B dense | `planar3` |
| `make run-reasoning` | Qwen3.5-27B Claude Opus Distilled | i1-Q4_K_M (16.6 GB) | 27B dense | `planar3` |
| `make run-gemma` | Gemma 4 26B MoE | UD-Q4_K_XL (17.1 GB) | 3.8B active | `planar4` |

Or use Docker Compose directly:

```bash
docker compose --profile qwen up           # Qwen3.6-35B-A3B Q4 (32 GB+), iso3
docker compose --profile qwen-mtp up       # Qwen3.6-35B-A3B MTP Q4 (32 GB+), iso3
docker compose --profile qwen36-q3 up     # Qwen3.6-35B-A3B Q3 (24 GB)
docker compose --profile qwen36-iq3 up    # Qwen3.6-35B-A3B IQ3 (16 GB)
docker compose --profile qwen36-27b up    # Qwen3.6-27B dense (24 GB+), planar3
docker compose --profile qwen36-27b-mtp up # Qwen3.6-27B MTP Q4 (24 GB+), planar3
docker compose --profile qwen36-27b-mtp-speed up # 27B MTP speed path, q4_0 KV
docker compose --profile qwen36-27b-q3 up # Qwen3.6-27B Q3 (16 GB)
docker compose --profile reasoning up     # Qwen3.5-27B reasoning-tuned
docker compose --profile gemma up         # Gemma 4 MoE
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
| `KV_CACHE_TYPE` | per-profile | KV cache type: `iso3` (MoE default), `planar3` (dense default), `q4_0` (MTP speed profile), `tbq4`, `planar4`, `iso4`, `f16` |
| `CTX_SIZE` | per-model | Context window (e.g. 114688 for Q4_K_M on 24 GB) |
| `PORT` | `8080` | API port |
| `GPU_LAYERS` | `99` | Layers on GPU (99 = all) |
| `N_PARALLEL` | `2` | Concurrent request slots (set higher for throughput mode) |
| `UBATCH_SIZE` | MTP: `32` | Physical batch size. The MTP path is sensitive to this. |
| `CACHE_RAM` | `8192` | Prompt cache size in MiB (system RAM, not VRAM) |
| `MTP_SPEC_TYPE` | `auto` | MTP spec flag spelling: auto-detects `draft-mtp` vs older `mtp` builds |
| `MTP_DRAFT_N_MAX` | `4` | Draft tokens per MTP speculative decoding step. Override to `2` for conservative testing or `6` on latest upstream llama.cpp builds. |
| `MTP_DRAFT_P_MIN` | `0.75` | Minimum speculative draft probability when supported by the llama.cpp build |
| `NO_WARMUP` | MTP: `1` | Pass `--no-warmup` for MTP profiles unless set false |
| `MTP_MLOCK` | off | Pass `--mlock`; requires memlock privileges in Docker/Kubernetes |
| `HF_TOKEN` | — | HuggingFace token for gated models |

MTP profiles force `N_PARALLEL=1` by default because current llama.cpp MTP does not support multiple parallel slots. The entrypoint also normalizes cache names across builds: `iso3`/`planar3`/`tbq4` become `iso3_0`/`planar3_0`/`tbq4_0` when the compiled binary expects the suffixed names.

Unsloth's current llama.cpp guidance uses `--spec-type draft-mtp` with
`--spec-draft-n-max 6` on latest upstream builds. This stack keeps
`MTP_SPEC_TYPE=auto` and defaults to draft 4 as the conservative 4090 starting
point; set `MTP_DRAFT_N_MAX=6` when validating the upstream `draft-mtp` path.

### MTP Operational Check

Do not treat an MTP startup banner as proof of a speedup. A working run must generate and accept draft tokens.

```bash
python scripts/mtp_probe.py --mtp-url http://localhost:8080 --min-acceptance 0.50

# Optional A/B check against a non-MTP control server
python scripts/mtp_probe.py \
  --mtp-url http://localhost:8080 \
  --base-url http://localhost:8081 \
  --min-acceptance 0.50 \
  --min-speedup 1.05
```

Historical local probe on Apple M1 Max with the cached Unsloth `Qwen3.6-27B-UD-Q4_K_XL.gguf`, the previous Indras fork, `--spec-type mtp`, `--spec-draft-p-min 0.75`, and `--ubatch-size 32`: draft 4 was the fastest local setting tested, accepting 46/67 draft tokens (68.7%) at 13.74 tok/s. Re-run this probe after each upstream rebase before moving traffic.

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

**Use Qwen3.6-35B-A3B Q3 for highest 4090 throughput when quality is acceptable.**
Use `make run-qwen36-27b-mtp-speed` when you specifically want the dense 27B
MTP path and can trade some KV compression for decode speed.

| Use Case | Quant | Size | KV | Context | Command |
|----------|-------|-----:|:--:|--------:|---------|
| Highest throughput | Qwen3.6-35B-A3B UD-Q3_K_XL | 16.8 GB | iso3 | 98K-131K | `docker compose --profile qwen36-q3 up` |
| Dense MTP speed | Qwen3.6-27B MTP UD-Q4_K_XL | 16.4 GB | q4_0 | 131K | `make run-qwen36-27b-mtp-speed` |
| Dense max compression | Qwen3.6-27B MTP UD-Q4_K_XL | 16.4 GB | planar3 | 131K | `make run-qwen36-27b-mtp` |
| Dense non-MTP | Qwen3.6-27B UD-Q4_K_XL | 16.4 GB | planar3 | 131K | `make run-qwen36-27b` |
| Dense max context | Qwen3.6-27B UD-IQ3_XXS | ~9 GB | planar3 | ~240K | `docker compose --profile qwen36-27b-iq3 up` |

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

## Build Source

The Dockerfile builds from official upstream `ggml-org/llama.cpp` stable tag
`b9196` by default, then applies
[`docker/patches/llama-b9196-rotorquant.patch.gz`](docker/patches/llama-b9196-rotorquant.patch.gz)
to add the RotorQuant KV cache types. This keeps Qwen3.6 MTP on upstream's
`draft-mtp` implementation instead of the older fork-specific MTP path.

Override `LLAMA_CPP_REPO`, `LLAMA_CPP_REF`, or `ROTORQUANT_PATCH` at build time
when rebasing to a newer upstream stable.

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
