# Sprint 002: Dockerize RotorQuant llama.cpp Server

**Status**: Planning
**Created**: 2026-04-03
**Depends on**: RotorQuant llama.cpp fork (tested in Sprint 001 follow-up)

---

## Overview

Package the RotorQuant llama.cpp server into a Docker image with multi-model support.
Three models are selectable at runtime via environment variable: Qwen3.5-27B (base),
Qwen3.5-27B Claude Reasoning Distilled, and Gemma 4 26B MoE. Models are stored on a
Docker volume (not baked into the image) and downloaded on first run. The server exposes
an OpenAI-compatible API with RotorQuant iso3/iso3 KV cache compression enabled by default.

Sprint 001 benchmarks showed: 67.5 tok/s decode (97% of f16), 128K context in 22.3 GB VRAM,
4.9x KV compression on RTX 5090.

## Use Cases

1. **One-command deployment**: `docker compose --profile qwen up` starts serving immediately
2. **Model switching**: Change profile to swap models without rebuilding
3. **Long-context serving**: 128K+ context on consumer GPUs (RTX 5090, RTX 4090) via RotorQuant
4. **Team sharing**: Anyone with Docker + NVIDIA toolkit can run the same setup
5. **Benchmarking**: Included benchmark script for validation on new hardware

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Docker Image: rotorquant                           │
│                                                     │
│  /app/bin/llama-server     (RotorQuant CUDA build)  │
│  /app/bin/llama-perplexity (for benchmarks)         │
│  /app/entrypoint.sh        (download + config + run)│
│  /app/benchmark.py         (validation script)      │
│                                                     │
│  ENV: MODEL_NAME, KV_CACHE_TYPE, CTX_SIZE, PORT     │
│                                                     │
│  Volume: /models (Docker named volume)              │
│    ├── Qwen3.5-27B-Q4_K_M.gguf                     │
│    ├── ...i1-Q4_K_M.gguf                            │
│    └── ...UD-Q4_K_XL.gguf                           │
└─────────────────────────────────────────────────────┘
         │                        │
    Port 8080              GPU (--gpus all)
    OpenAI API             NVIDIA Container Toolkit
```

### Model Registry (embedded in entrypoint.sh)

```bash
# MODEL_NAME → "HF_REPO|FILENAME|DEFAULT_CTX|EXTRA_FLAGS"
declare -A MODELS=(
  [qwen3.5-27b]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-Q4_K_M.gguf|131072|"
  [qwen3.5-27b-reasoning]="mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF|Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-Q4_K_M.gguf|131072|"
  [gemma4-26b]="unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf|131072|--temp 1.0 --top-p 0.95 --top-k 64"
)
```

### Runtime Flow

```
entrypoint.sh:
  1. Validate MODEL_NAME exists in registry
  2. Check if /models/{filename} exists
     - If missing: huggingface-cli download → /models/
  3. Build llama-server command with:
     - model path, GPU layers, context size
     - KV cache type (default: iso3/iso3)
     - flash attention, port, model-specific flags
  4. exec llama-server (PID 1 for signal handling)
```

---

## Implementation

### Phase 1: Dockerfile (~40% effort)

**File:** `docker/Dockerfile`

**Tasks:**
- [ ] Multi-stage build:
  - Build stage: `nvidia/cuda:12.8.0-devel-ubuntu24.04`, clone RotorQuant fork, cmake + build
  - Runtime stage: `nvidia/cuda:12.8.0-runtime-ubuntu24.04`, copy binaries + shared libs
- [ ] CUDA architectures: `80;86;89;90;100;120` (A100, A10G, RTX 4090, H100, B100, RTX 5090)
- [ ] Build flags: `-DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_BUILD_TYPE=Release`
- [ ] Install `huggingface-hub[cli]` in runtime stage (for model downloads)
- [ ] Create non-root user `llm` (UID 1000) — run server as non-root
- [ ] Copy entrypoint.sh, benchmark script
- [ ] HEALTHCHECK: `curl -sf http://localhost:${PORT}/health`
- [ ] EXPOSE 8080

```dockerfile
# === Build Stage ===
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y git cmake build-essential
RUN git clone --depth 50 -b feature/planarquant-kv-cache \
    https://github.com/johndpope/llama-cpp-turboquant.git /src
WORKDIR /src
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120" \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j$(nproc)

# === Runtime Stage ===
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl python3 python3-pip ca-certificates \
    && pip3 install --break-system-packages huggingface-hub[cli] \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 llm
COPY --from=build /src/build/bin/llama-server /app/bin/
COPY --from=build /src/build/bin/llama-perplexity /app/bin/
COPY --from=build /src/build/bin/lib*.so* /app/lib/
COPY docker/entrypoint.sh /app/
COPY scripts/benchmark_rotorquant.py /app/
RUN chmod +x /app/entrypoint.sh
ENV LD_LIBRARY_PATH=/app/lib
ENV PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -sf http://localhost:${PORT}/health || exit 1
USER llm
ENTRYPOINT ["/app/entrypoint.sh"]
```

### Phase 2: Entrypoint Script (~30% effort)

**File:** `docker/entrypoint.sh`

**Tasks:**
- [ ] Model registry: bash associative array mapping MODEL_NAME → config
- [ ] Parse env vars: MODEL_NAME (required), KV_CACHE_TYPE (default iso3), CTX_SIZE, PORT, GPU_LAYERS, EXTRA_ARGS
- [ ] Validate MODEL_NAME exists in registry; exit 1 with usage message if invalid
- [ ] Download logic: check `/models/${FILENAME}` exists; if not, run `huggingface-cli download`
  - Use `--local-dir /models --local-dir-use-symlinks False` (avoid symlink issues on volumes)
  - Support `HF_TOKEN` env var for gated models
  - Print download progress, handle errors gracefully
- [ ] Build server command array with all flags
- [ ] Signal handling: `trap 'kill $PID' SIGTERM SIGINT`; `exec` for PID 1
- [ ] Print startup banner with model name, config, and RotorQuant version

```bash
#!/bin/bash
set -euo pipefail

declare -A MODELS=(
  [qwen3.5-27b]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-Q4_K_M.gguf|131072|"
  [qwen3.5-27b-reasoning]="mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF|Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-Q4_K_M.gguf|131072|"
  [gemma4-26b]="unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf|131072|--temp 1.0 --top-p 0.95 --top-k 64"
)

MODEL_NAME="${MODEL_NAME:?ERROR: MODEL_NAME env var is required}"
KV_CACHE="${KV_CACHE_TYPE:-iso3}"
CTX="${CTX_SIZE:-131072}"
PORT="${PORT:-8080}"
NGL="${GPU_LAYERS:-99}"

# Parse model config
IFS='|' read -r HF_REPO FILENAME DEFAULT_CTX EXTRA <<< "${MODELS[$MODEL_NAME]:-}"
[ -z "$HF_REPO" ] && echo "Unknown MODEL_NAME=$MODEL_NAME" && exit 1
CTX="${CTX:-$DEFAULT_CTX}"

# Download if missing
MODEL_PATH="/models/${FILENAME}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading $FILENAME from $HF_REPO..."
    huggingface-cli download "$HF_REPO" "$FILENAME" \
        --local-dir /models --local-dir-use-symlinks False
fi

echo "=== RotorQuant Server ==="
echo "  Model:    $MODEL_NAME ($FILENAME)"
echo "  KV Cache: $KV_CACHE / $KV_CACHE"
echo "  Context:  $CTX"
echo "  Port:     $PORT"
echo "========================="

exec /app/bin/llama-server \
    --model "$MODEL_PATH" \
    --n-gpu-layers "$NGL" \
    --ctx-size "$CTX" \
    --cache-type-k "$KV_CACHE" \
    --cache-type-v "$KV_CACHE" \
    --flash-attn \
    --host 0.0.0.0 \
    --port "$PORT" \
    $EXTRA ${EXTRA_ARGS:-}
```

### Phase 3: Docker Compose (~15% effort)

**File:** `docker-compose.yml`

**Tasks:**
- [ ] Define base service `x-llm-base` with shared config (image, GPU, volume, health check)
- [ ] 3 profile services: `qwen`, `reasoning`, `gemma` — each sets MODEL_NAME
- [ ] Named volume `llm-models` for persistent model storage
- [ ] GPU passthrough via `deploy.resources.reservations.devices`
- [ ] Configurable overrides via environment variables

```yaml
x-llm-base: &llm-base
  image: rotorquant:latest
  build:
    context: .
    dockerfile: docker/Dockerfile
  volumes:
    - llm-models:/models
  ports:
    - "${PORT:-8080}:${PORT:-8080}"
  environment:
    KV_CACHE_TYPE: ${KV_CACHE_TYPE:-iso3}
    CTX_SIZE: ${CTX_SIZE:-131072}
    PORT: ${PORT:-8080}
    GPU_LAYERS: ${GPU_LAYERS:-99}
    HF_TOKEN: ${HF_TOKEN:-}
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  restart: unless-stopped

services:
  qwen:
    <<: *llm-base
    profiles: [qwen]
    environment:
      MODEL_NAME: qwen3.5-27b

  reasoning:
    <<: *llm-base
    profiles: [reasoning]
    environment:
      MODEL_NAME: qwen3.5-27b-reasoning

  gemma:
    <<: *llm-base
    profiles: [gemma]
    environment:
      MODEL_NAME: gemma4-26b

volumes:
  llm-models:
```

### Phase 4: Testing & Documentation (~15% effort)

**Files:** `docker/test.sh`, `README.md` (Docker section)

**Tasks:**
- [ ] `test.sh`: build image, start with each model, health check, one API call, shutdown
- [ ] README section: quick start, model list, configuration table, examples
- [ ] Verify Gemma 4 MoE works with RotorQuant KV cache (no special flags needed — MoE routing is in GGUF metadata)

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `docker/Dockerfile` | Create | Multi-stage CUDA build + slim runtime |
| `docker/entrypoint.sh` | Create | Model download, config, server launch |
| `docker-compose.yml` | Create | Compose profiles for each model |
| `docker/test.sh` | Create | Build + smoke test all 3 models |
| `.dockerignore` | Create | Exclude models/, .git, __pycache__ |

---

## Definition of Done

### Build
- [ ] `docker build -t rotorquant -f docker/Dockerfile .` completes without error
- [ ] Image size < 8 GB (runtime stage, no model weights)
- [ ] `docker images rotorquant` shows the built image

### Runtime — Qwen3.5-27B
- [ ] `docker compose --profile qwen up` starts server
- [ ] First run downloads model (~17 GB) to volume; second run skips download
- [ ] `curl localhost:8080/health` returns `{"status":"ok"}`
- [ ] `curl localhost:8080/v1/chat/completions` with a test prompt returns valid response
- [ ] Server runs as non-root user (UID 1000)

### Runtime — Qwen3.5-27B Reasoning
- [ ] `docker compose --profile reasoning up` starts server
- [ ] API responds with reasoning_content in chat completions

### Runtime — Gemma 4 26B
- [ ] `docker compose --profile gemma up` starts server
- [ ] API responds correctly (Gemma chat template auto-detected)

### Configuration
- [ ] `KV_CACHE_TYPE=f16 docker compose --profile qwen up` disables RotorQuant
- [ ] `CTX_SIZE=131072 docker compose --profile qwen up` sets 128K context
- [ ] `HF_TOKEN=hf_xxx docker compose --profile reasoning up` supports gated models
- [ ] `PORT=9090 docker compose --profile qwen up` uses custom port

### Health & Signals
- [ ] HEALTHCHECK passes after model loads
- [ ] `docker compose down` triggers graceful shutdown (SIGTERM handled)
- [ ] Container exits cleanly with code 0 on `docker stop`

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gemma 4 MoE incompatible with RotorQuant iso3 | Low | High | Test first; fall back to f16 for Gemma |
| CUDA 12.8 base image not available | Low | Medium | Fall back to 12.6; parameterize via build arg |
| Model download fails mid-way (network issues) | Medium | Low | huggingface-cli has resume support; re-run picks up |
| GGUF filename changes on HuggingFace | Low | Medium | Pin exact filenames in registry; update on breakage |
| Volume permissions: non-root can't write to /models | Medium | Medium | `chmod 777 /models` in Dockerfile or run download as root |
| Build takes 30+ minutes (CUDA compilation) | Expected | Low | Multi-arch build is one-time; cache Docker build layers |

---

## Security Considerations

- Non-root user `llm` (UID 1000) in runtime container
- `HF_TOKEN` passed via env var, never baked into image or logged
- No secrets in Dockerfile; model downloads use public repos by default
- `.dockerignore` excludes `.env`, credentials, model files from build context
- Server binds to `0.0.0.0:8080` — use Docker network isolation or firewall in production

---

## Dependencies

- Docker Engine 24+
- Docker Compose v2.20+
- NVIDIA Container Toolkit (`nvidia-ctk`)
- NVIDIA driver 550+ (for CUDA 12.8)
- ~20 GB disk per model on the Docker volume
- Internet access for first-run model download

---

## Open Questions

1. **Volume permissions**: If the Docker volume is created by root but the container runs as
   `llm` (UID 1000), the download may fail. Options: create volume with correct ownership
   in entrypoint (switch to root briefly), or use `--user root` for download only.

2. **Gemma 4 RotorQuant validation**: We've only tested iso3/iso3 on Qwen3.5-27B. Need to
   verify Gemma 4's MoE attention heads work correctly with RotorQuant KV cache types.

3. **Pinning RotorQuant fork commit**: The Dockerfile clones `HEAD` of the branch. Should we
   pin to a specific commit hash for reproducibility? (Yes — add `ARG ROTORQUANT_COMMIT=abc123`)
