# Sprint 002 Intent: Dockerize RotorQuant llama.cpp Server

## Seed

Dockerize the RotorQuant llama.cpp server with multi-model support for Qwen3.5-27B,
Qwen3.5-27B-Reasoning-Distilled, and Gemma 4 26B. Models selectable at runtime,
stored on Docker volume, with RotorQuant KV cache compression enabled.

## Context

- Sprint 001 built a Python TurboQuant package (KV cache quantization for HF transformers)
- During Sprint 001 follow-up, we discovered RotorQuant — a faster variant with fused CUDA kernels and llama.cpp integration
- We built the RotorQuant llama.cpp fork (`johndpope/llama-cpp-turboquant`, branch `feature/planarquant-kv-cache`) and benchmarked it on Qwen3.5-27B Q4_K_M
- Benchmark results on RTX 5090 (32GB): 67.5 tok/s decode (97% of f16), 128K context fits in 22.3 GB, PPL 6.76 (vs 6.38 f16)
- No existing Docker infrastructure in this repo
- Sprint 001 deferred items are Python-side optimizations — not relevant here

## Models

| ID | HuggingFace Repo | File | Size | Architecture |
|----|-------------------|------|------|-------------|
| qwen3.5-27b | unsloth/Qwen3.5-27B-GGUF | Qwen3.5-27B-Q4_K_M.gguf | 16.7 GB | Qwen3.5, 64L, 8 KV heads |
| qwen3.5-27b-reasoning | mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF | Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-Q4_K_M.gguf | 16.6 GB | Qwen3.5, same arch |
| gemma4-26b | unsloth/gemma-4-26B-A4B-it-GGUF | gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf | 17.1 GB | Gemma 4 MoE, 30L, 3.8B active |

## Relevant Codebase

- `/home/ravi/repos/turbo/llama-cpp-rq/` — built RotorQuant fork (source of truth for build commands)
- `/home/ravi/repos/turbo/scripts/benchmark_rotorquant.py` — benchmark script to include in image
- No existing Dockerfile or docker-compose.yml

## Constraints

- Multi-stage Docker build (build stage + slim runtime)
- NVIDIA Container Toolkit required (`--gpus all`)
- Models on a named Docker volume (`/models`), NOT baked into image
- Single model selected at runtime via `MODEL_NAME` env var
- OpenAI-compatible API on configurable port (default 8080)
- Chat templates auto-detected from GGUF metadata by llama.cpp (no manual template needed)
- Gemma 4 recommended sampling: temperature=1.0, top_p=0.95, top_k=64

## Success Criteria

1. `docker build -t rotorquant .` completes
2. `docker run --gpus all -e MODEL_NAME=qwen3.5-27b -v llm-models:/models -p 8080:8080 rotorquant` serves correctly
3. First run downloads model, subsequent runs skip download
4. All 3 models start and generate via the API
5. RotorQuant KV cache (iso3/iso3) enabled by default, configurable via env var
6. Health check passes

## Uncertainty Assessment

- Correctness: **Low** — standard Docker patterns, tested build
- Scope: **Low** — 4 files (Dockerfile, entrypoint.sh, docker-compose.yml, model config)
- Architecture: **Low** — well-known multi-stage Docker + volume pattern
