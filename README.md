# RotorQuant LLM Server

Serve Qwen3.5-27B, Gemma 4 26B, and other large models with **128K+ context on a single consumer GPU** using RotorQuant KV cache compression (3.8x compression at iso4 default, 97% decode speed of fp16).

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
| `CTX_SIZE` | `131072` | Context window (128K default) |
| `PORT` | `8080` | API port |
| `GPU_LAYERS` | `99` | Layers on GPU (99 = all) |
| `HF_TOKEN` | — | HuggingFace token for gated models |

## GPU Compatibility

All numbers for Qwen3.5-27B Q4_K_M (16.7 GB weights).

### iso4 (default — best quality, 3.8x compression)

| GPU | VRAM | 32K ctx | 64K ctx | 128K ctx | Max ctx |
|-----|-----:|:-------:|:-------:|:--------:|--------:|
| RTX 3090 | 24 GB | 18.8 GB | 20.9 GB | OOM | ~112K |
| RTX 4090 | 24 GB | 18.8 GB | 20.9 GB | OOM | ~112K |
| RTX 5090 | 32 GB | 18.8 GB | 20.9 GB | 25.2 GB | ~236K |
| A100 40GB | 40 GB | 18.8 GB | 20.9 GB | 25.2 GB | ~236K |
| A100 80GB | 80 GB | 18.8 GB | 20.9 GB | 25.2 GB | ~236K+ |
| H100 | 80 GB | 18.8 GB | 20.9 GB | 25.2 GB | ~236K+ |

### iso3 (max compression, 4.9x)

| GPU | VRAM | 32K ctx | 64K ctx | 128K ctx | Max ctx |
|-----|-----:|:-------:|:-------:|:--------:|--------:|
| RTX 3090 | 24 GB | 18.3 GB | 19.9 GB | 23.2 GB | ~147K |
| RTX 4090 | 24 GB | 18.3 GB | 19.9 GB | 23.2 GB | ~147K |
| RTX 5090 | 32 GB | 18.3 GB | 19.9 GB | 23.2 GB | ~300K |
| A100 40GB | 40 GB | 18.3 GB | 19.9 GB | 23.2 GB | ~300K+ |

> **RTX 4090 users**: iso4 fits up to ~112K context. For 128K, switch to iso3:
> `KV_CACHE_TYPE=iso3 make run-qwen`

## Requirements

- Docker 24+
- Docker Compose v2.20+
- NVIDIA Container Toolkit (`nvidia-ctk`)
- NVIDIA driver 570+ (CUDA 13.1 support)
- GPU with 24+ GB VRAM (RTX 4090, RTX 5090, A100, H100)

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
