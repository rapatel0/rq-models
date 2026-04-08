# RotorQuant LLM Server
> NOTE: Gemma Models have not been tested for ppl and qualtity. YMMV

Serve Qwen3.5-27B, Gemma 4 26B, and other large models with **112K+ context on a single consumer GPU** using RotorQuant KV cache compression (3.8x compression at iso4 default, 97% decode speed of fp16).

> **Default: iso4 (4-bit)** — best balance of quality and compression. Use `iso3` for maximum compression (4.9x) if you need more context headroom.


## Quick Start

```bash
# 1. Build
make build

# 2. Run (first run downloads the model ~17 GB)
make run-qwen

# 3. Query
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":500}'
```

## Available Models

| Command | Model | Weights | Active Params |
|---------|-------|---------|---------------|
| `make run-qwen` | Qwen3.5-27B | Q4_K_M (16.7 GB) | 27B dense |
| `make run-reasoning` | Qwen3.5-27B Claude Opus Distilled | i1-Q4_K_M (16.6 GB) | 27B dense |
| `make run-gemma` | Gemma 4 26B MoE | UD-Q4_K_XL (17.1 GB) | 3.8B active |

Or use Docker Compose directly:

```bash
docker compose --profile qwen up        # Qwen3.5-27B
docker compose --profile reasoning up   # Reasoning-tuned
docker compose --profile gemma up       # Gemma 4 MoE
```

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
| `KV_CACHE_TYPE` | `iso4` | KV cache type: `iso4` (default, best quality), `iso3` (max compression), `planar3`, `f16` |
| `CTX_SIZE` | per-model | Context window (e.g. 114688 for Q4_K_M on 24 GB) |
| `PORT` | `8080` | API port |
| `GPU_LAYERS` | `99` | Layers on GPU (99 = all) |
| `N_PARALLEL` | `2` | Concurrent request slots (set higher for throughput mode) |
| `CACHE_RAM` | `8192` | Prompt cache size in MiB (system RAM, not VRAM) |
| `HF_TOKEN` | — | HuggingFace token for gated models |

## Best Config by GPU

Recommended configurations per VRAM tier, based on measured perplexity
(see [docs/QUANTIZATION-GUIDE.md](docs/QUANTIZATION-GUIDE.md) for full data).

### 16 GB (RTX 4060 Ti, RTX 5060, RTX 4080)

Q4_K_M doesn't fit. Use Unsloth imatrix quants:

| Use Case | Quant | Size | PPL | Context (iso4) | Context (iso3) |
|----------|-------|-----:|----:|---------------:|---------------:|
| **Best quality** | IQ4_XS | 13.9 GB | 6.29 | ~14K | ~18K |
| **Recommended** | **UD-Q3_K_XL** | **13.4 GB** | **6.38** | **~28K** | **~36K** |
| Max context | UD-IQ3_XXS | 10.7 GB | 6.62 | ~56K | ~74K |

> UD-Q3_K_XL is the sweet spot — only +0.08 PPL over 4-bit, 33% more context.

### 24 GB (RTX 3090, RTX 4090)

| Use Case | Quant | Size | PPL | Context (iso4) | Context (iso3) |
|----------|-------|-----:|----:|---------------:|---------------:|
| **Recommended** | **Q4_K_M** | **15.6 GB** | **~6.27** | **~112K** | **~171K** |
| More context | IQ4_XS | 13.9 GB | 6.29 | ~155K | ~203K |
| Max context | UD-Q3_K_XL | 13.4 GB | 6.38 | ~163K | ~213K |

> Q4_K_M at iso4 gives full quality with 112K context on 24 GB GPUs.
> For more context, switch to iso3: `KV_CACHE_TYPE=iso3 make run-qwen`

### 32 GB (RTX 5090)

| Use Case | Quant | Size | PPL | Context (iso4) | Context (iso3) |
|----------|-------|-----:|----:|---------------:|---------------:|
| **Recommended** | **Q4_K_M** | **15.6 GB** | **~6.27** | **~252K** | **~330K** |
| Best quality | UD-Q4_K_XL | 16.4 GB | ~6.25 | ~240K | ~314K |
| Max context | IQ4_XS | 13.9 GB | 6.29 | ~278K | ~364K |

> RTX 5090 has headroom for Q4 quality + 250K context. No compromises needed.

### 40 GB (A100 40GB, A6000)

| Use Case | Quant | Size | PPL | Context (iso4) | Context (iso3) |
|----------|-------|-----:|----:|---------------:|---------------:|
| **Recommended** | **Q4_K_M** | **15.6 GB** | **~6.27** | **~375K** | **~491K** |
| Best quality | Q5_K_M | 18.3 GB | ~6.22 | ~334K | ~437K |
| Overkill | Q6_K | 20.9 GB | ~6.20 | ~293K | ~384K |

> At 40 GB there's no reason to go below Q4. Use Q5_K_M if you want the
> absolute best quality; the extra 2.7 GB is negligible.

### Quick Reference

| GPU | Quant | PPL | Context | Command |
|-----|-------|----:|--------:|---------|
| **16 GB** | UD-Q3_K_XL | 6.38 | ~28K | `docker compose --profile qwen-q3 up` |
| **16 GB** | UD-IQ3_XXS | 6.62 | ~56K | `docker compose --profile qwen-q3-xxs up` |
| **16 GB** | IQ4_XS | 6.29 | ~14K | `docker compose --profile qwen-iq4 up` |
| **16 GB** | Gemma4 Q3 | — | ~40K | `docker compose --profile gemma-q3 up` |
| **24 GB** | Q4_K_M | ~6.27 | ~112K | `docker compose --profile qwen up` |
| **32 GB** | Q4_K_M | ~6.27 | ~252K | `docker compose --profile qwen up` |
| **40 GB** | Q4_K_M | ~6.27 | ~375K | `docker compose --profile qwen up` |

All profiles auto-download the correct model on first run.

### Throughput Mode (Parallel Slots)

For serving multiple concurrent users, trade context length for parallel slots.
During decode, single-user inference is memory-bandwidth bound (mat-vec). With
N parallel slots, the weight multiplications become mat-mat and tensor cores
engage — aggregate throughput scales near-linearly.

RotorQuant is the multiplier: iso4 KV at 16K = 1.06 GB/slot vs f16 = 4.0 GB/slot — **3.8x more concurrent users at the same VRAM.**

| GPU | Slots (iso4) | Slots (f16) | Est. Aggregate tok/s | Command |
|-----|-------------:|------------:|---------------------:|---------|
| **24 GB** | 6 | 1 | ~320 | `N_PARALLEL=6 docker compose --profile qwen-throughput up` |
| **32 GB** | 14 | 3 | ~660 | `docker compose --profile qwen-throughput up` |
| **40 GB** | 22 | 5 | ~880 | `N_PARALLEL=22 docker compose --profile qwen-throughput up` |
| **80 GB** | 59 | 15 | ~2,400 | `N_PARALLEL=59 docker compose --profile qwen-throughput up` |

```bash
# Quick start: max throughput on RTX 5090 (32 GB)
make run-throughput

# Custom slot count
N_PARALLEL=8 docker compose --profile qwen-throughput up
```

Per-user latency stays ~67 tok/s at low load; at full capacity, ~50-55 tok/s per user.

## Requirements

- Docker 24+
- Docker Compose v2.20+
- NVIDIA Container Toolkit (`nvidia-ctk`)
- NVIDIA driver 570+ (CUDA 13.1 support)
- GPU with 16+ GB VRAM (RTX 5060/4060 Ti with Q3 quants, RTX 4090+, A100, H100)

## Performance

Benchmarked on RTX 5090 (32 GB) with Qwen3.5-27B Q4_K_M:

| Metric | iso4 (default) | iso3 | f16 |
|--------|:-:|:-:|:-:|
| Decode speed | ~67 tok/s | 67.5 tok/s | 69.3 tok/s |
| KV compression | **3.8x** | **4.9x** | 1x |
| Max ctx (RTX 5090) | **236K** | **300K** | 32K |
| Max ctx (RTX 4090) | **112K** | **147K** | 16K |
| Perplexity (wikitext-2) | better than iso3 | 6.76 | 6.38 |
| NIAH recall (4K) | 100% | 100% | 100% |

See [docs/BENCHMARK-REPORT.md](docs/BENCHMARK-REPORT.md) for full results.

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
