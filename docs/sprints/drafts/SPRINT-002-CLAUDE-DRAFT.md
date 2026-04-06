# Sprint 002: Dockerize RotorQuant llama.cpp Server

**Status**: Draft
**Created**: 2026-04-02
**Depends on**: Sprint 001 (RotorQuant llama.cpp fork built and benchmarked)
**Target**: Multi-model Docker image with RotorQuant KV cache compression

---

## Overview

Package the RotorQuant llama.cpp fork (`johndpope/llama-cpp-turboquant`, branch
`feature/planarquant-kv-cache`) into a production-ready Docker image. The image
builds llama-server with CUDA support in a multi-stage build, and an entrypoint
script handles model download (via huggingface-cli), model config lookup, and
server launch with RotorQuant KV cache flags.

Three models are supported at launch: Qwen3.5-27B, Qwen3.5-27B-Reasoning-Distilled,
and Gemma 4 26B (MoE). Models are selected at runtime via a `MODEL_NAME` environment
variable and stored on a named Docker volume so they persist across container restarts.
First run downloads the model; subsequent runs skip the download.

The server exposes an OpenAI-compatible API (default port 8080) with iso3/iso3
RotorQuant KV cache compression enabled by default. A docker-compose.yml provides
a one-command launch experience.

---

## Use Cases

1. **One-command local LLM server**: `docker compose up` starts a GPU-accelerated
   llama.cpp server with RotorQuant compression, no manual build or model management.

2. **Model switching without rebuild**: Change `MODEL_NAME=gemma4-26b` in the
   environment and restart the container. The entrypoint downloads the new model
   if needed and adjusts server flags automatically.

3. **Persistent model storage**: Models live on a Docker volume (`llm-models`).
   Pulling a new image version or recreating the container does not re-download
   17 GB model files.

4. **Team-shareable inference**: The docker-compose.yml is a single file any
   team member can `docker compose up` with, given an NVIDIA GPU and the
   NVIDIA Container Toolkit.

5. **Benchmarking harness**: The image includes the benchmark script
   (`benchmark_rotorquant.py`) so perf comparisons between KV cache configs
   can run inside the container without additional setup.

---

## Architecture

### Container Layout

```
/app/
  llama-server          # compiled binary (from build stage)
  llama-perplexity      # compiled binary (for benchmarks)
  entrypoint.sh         # model download + config lookup + exec
  benchmark_rotorquant.py
  libllama.so*          # shared libs

/models/                # Docker volume mount point
  Qwen3.5-27B-Q4_K_M.gguf
  Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-Q4_K_M.gguf
  gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
```

### Build Flow

```
Stage 1: nvidia/cuda:12.8.1-devel-ubuntu24.04
  ├── apt: cmake, git, build-essential, python3, pip
  ├── git clone --branch feature/planarquant-kv-cache --depth 1
  ├── cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120"
  └── cmake --build build --target llama-server llama-perplexity -j$(nproc)

Stage 2: nvidia/cuda:12.8.1-runtime-ubuntu24.04
  ├── copy binaries + libs from stage 1
  ├── pip install huggingface_hub[cli] (for model downloads)
  ├── copy entrypoint.sh, benchmark script
  └── ENTRYPOINT ["/app/entrypoint.sh"]
```

### Entrypoint Flow

```
entrypoint.sh
  │
  ├── 1. Resolve MODEL_NAME → (hf_repo, filename, ctx_size, extra_flags)
  │      via embedded config map (associative array)
  │
  ├── 2. Check /models/$filename exists
  │      ├── YES → skip download
  │      └── NO  → huggingface-cli download $hf_repo $filename --local-dir /models
  │
  ├── 3. Build llama-server command line:
  │      --model /models/$filename
  │      --ctx-size $ctx_size
  │      --host 0.0.0.0 --port $PORT
  │      --n-gpu-layers 99
  │      --cache-type-k $CACHE_TYPE_K --cache-type-v $CACHE_TYPE_V
  │      --flash-attn on
  │      $extra_flags (model-specific, e.g. Gemma 4 sampling defaults)
  │
  ├── 4. Trap SIGTERM/SIGINT → forward to llama-server PID
  │
  └── 5. exec llama-server $args
```

### Model Config Map

| MODEL_NAME | HuggingFace Repo | Filename | ctx_size | extra_flags |
|---|---|---|---|---|
| `qwen3.5-27b` | `unsloth/Qwen3.5-27B-GGUF` | `Qwen3.5-27B-Q4_K_M.gguf` | 32768 | (none) |
| `qwen3.5-27b-reasoning` | `mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF` | `Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-Q4_K_M.gguf` | 32768 | (none) |
| `gemma4-26b` | `unsloth/gemma-4-26B-A4B-it-GGUF` | `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` | 32768 | `--temp 1.0 --top-p 0.95 --top-k 64` |

**Gemma 4 MoE considerations**: Gemma 4 26B is a Mixture of Experts model (30 layers,
3.8B active parameters out of 26B total). llama.cpp handles MoE architectures
natively via GGUF metadata -- the expert routing is baked into the model file and
the llama-server binary dispatches accordingly. No special `--moe-*` flags are
needed. The only model-specific adjustment is the recommended sampling parameters
(temperature=1.0, top_p=0.95, top_k=64) which are set as server-level defaults
via `extra_flags`. Users can still override these per-request via the API.

If future llama.cpp builds introduce MoE-specific flags (e.g., expert offloading
or MoE-aware KV cache partitioning), they can be added to the config map's
`extra_flags` field without changing the entrypoint logic.

---

## Implementation

### Phase 1: Dockerfile (Day 1, ~30% effort)

**Goal**: Multi-stage Docker build that compiles the RotorQuant llama.cpp fork with
CUDA and produces a slim runtime image.

**Files:**
- `docker/Dockerfile`

**Tasks:**
- [ ] Write build stage from `nvidia/cuda:12.8.1-devel-ubuntu24.04`:
  - Install build deps: `cmake`, `git`, `build-essential`, `ccache`
  - Clone `https://github.com/johndpope/llama-cpp-turboquant.git` branch `feature/planarquant-kv-cache` at pinned commit (depth 1)
  - Configure: `cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120" -DCMAKE_BUILD_TYPE=Release`
  - Build: `cmake --build build --target llama-server llama-perplexity -j$(nproc)`
- [ ] Write runtime stage from `nvidia/cuda:12.8.1-runtime-ubuntu24.04`:
  - Install runtime deps: `python3`, `python3-pip`, `curl`, `ca-certificates`
  - `pip install --no-cache-dir huggingface_hub[cli]`
  - Copy from build stage: `build/bin/llama-server`, `build/bin/llama-perplexity`, `build/bin/libllama.so*`
  - Copy `entrypoint.sh` and `benchmark_rotorquant.py` into `/app/`
  - Set `LD_LIBRARY_PATH=/app`
  - Expose port 8080
  - `HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -sf http://localhost:8080/health || exit 1`
  - `ENTRYPOINT ["/app/entrypoint.sh"]`
- [ ] Add `.dockerignore` to exclude `build/`, `.git/`, `*.o` from context
- [ ] Verify `docker build -t rotorquant .` completes without errors
- [ ] Verify image size is under 8 GB (runtime base ~3.5 GB + binaries ~500 MB + pip ~200 MB)

**Exact Dockerfile:**

```dockerfile
# ============================================================
# Stage 1: Build llama.cpp with RotorQuant + CUDA
# ============================================================
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS builder

ARG LLAMA_CPP_REPO=https://github.com/johndpope/llama-cpp-turboquant.git
ARG LLAMA_CPP_BRANCH=feature/planarquant-kv-cache
# Pin to a specific commit for reproducibility; update as needed
ARG LLAMA_CPP_COMMIT=HEAD
ARG CMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake git build-essential ccache ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --branch ${LLAMA_CPP_BRANCH} --depth 50 ${LLAMA_CPP_REPO} llama.cpp \
    && cd llama.cpp \
    && if [ "${LLAMA_CPP_COMMIT}" != "HEAD" ]; then git checkout ${LLAMA_CPP_COMMIT}; fi

WORKDIR /build/llama.cpp
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_STANDALONE=ON \
    && cmake --build build --target llama-server llama-perplexity -j$(nproc)

# ============================================================
# Stage 2: Slim runtime image
# ============================================================
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates \
    && pip install --no-cache-dir --break-system-packages huggingface_hub[cli] \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binaries and shared libraries from build stage
COPY --from=builder /build/llama.cpp/build/bin/llama-server /app/
COPY --from=builder /build/llama.cpp/build/bin/llama-perplexity /app/
COPY --from=builder /build/llama.cpp/build/bin/libllama.so* /app/
COPY --from=builder /build/llama.cpp/build/ggml/src/libggml*.so* /app/

# Copy entrypoint and benchmark script
COPY entrypoint.sh /app/entrypoint.sh
COPY benchmark_rotorquant.py /app/benchmark_rotorquant.py
RUN chmod +x /app/entrypoint.sh

ENV LD_LIBRARY_PATH=/app
ENV PORT=8080
ENV MODEL_NAME=qwen3.5-27b
ENV CACHE_TYPE_K=iso3
ENV CACHE_TYPE_V=iso3
ENV N_GPU_LAYERS=99
ENV CTX_SIZE=""
ENV EXTRA_ARGS=""

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

VOLUME /models

ENTRYPOINT ["/app/entrypoint.sh"]
```

### Phase 2: entrypoint.sh (Day 1-2, ~30% effort)

**Goal**: Shell script that resolves the model config, downloads the model if absent,
and launches llama-server with correct flags. Handles graceful shutdown.

**Files:**
- `docker/entrypoint.sh`

**Tasks:**
- [ ] Define model config map as bash associative arrays (repo, filename, ctx_size, extra_flags)
- [ ] Validate `MODEL_NAME` is a known key; exit 1 with helpful error listing valid names
- [ ] Check if `/models/$filename` exists; if not, download via `huggingface-cli download`
- [ ] Build server command from env vars: `CACHE_TYPE_K`, `CACHE_TYPE_V`, `N_GPU_LAYERS`, `CTX_SIZE`, `PORT`, `EXTRA_ARGS`
- [ ] If `CTX_SIZE` is empty, use the model's default from config map
- [ ] Trap SIGTERM and SIGINT; forward to the llama-server process for graceful shutdown
- [ ] Use `exec` to replace shell with llama-server (PID 1, proper signal handling)
- [ ] Log model name, file path, and full command line at startup

**Exact entrypoint.sh:**

```bash
#!/usr/bin/env bash
set -euo pipefail

# ── Model Configuration Map ──────────────────────────────────
# Each model: HF_REPO|FILENAME|DEFAULT_CTX_SIZE|EXTRA_FLAGS
declare -A MODEL_CONFIG
MODEL_CONFIG=(
  ["qwen3.5-27b"]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-Q4_K_M.gguf|32768|"
  ["qwen3.5-27b-reasoning"]="mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF|Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-Q4_K_M.gguf|32768|"
  ["gemma4-26b"]="unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf|32768|--temp 1.0 --top-p 0.95 --top-k 64"
)

# ── Validate MODEL_NAME ──────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-qwen3.5-27b}"

if [[ -z "${MODEL_CONFIG[$MODEL_NAME]+x}" ]]; then
  echo "ERROR: Unknown MODEL_NAME='${MODEL_NAME}'"
  echo "Valid models: ${!MODEL_CONFIG[*]}"
  exit 1
fi

# ── Parse config ─────────────────────────────────────────────
IFS='|' read -r HF_REPO FILENAME DEFAULT_CTX EXTRA_MODEL_FLAGS <<< "${MODEL_CONFIG[$MODEL_NAME]}"

MODEL_PATH="/models/${FILENAME}"
CTX_SIZE="${CTX_SIZE:-$DEFAULT_CTX}"
PORT="${PORT:-8080}"
CACHE_TYPE_K="${CACHE_TYPE_K:-iso3}"
CACHE_TYPE_V="${CACHE_TYPE_V:-iso3}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================"
echo "  RotorQuant llama.cpp Server"
echo "============================================"
echo "  Model:       ${MODEL_NAME}"
echo "  HF Repo:     ${HF_REPO}"
echo "  File:        ${FILENAME}"
echo "  Model Path:  ${MODEL_PATH}"
echo "  Context:     ${CTX_SIZE}"
echo "  KV Cache:    K=${CACHE_TYPE_K} V=${CACHE_TYPE_V}"
echo "  GPU Layers:  ${N_GPU_LAYERS}"
echo "  Port:        ${PORT}"
echo "============================================"

# ── Download model if not present ────────────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
  echo "Model not found at ${MODEL_PATH}. Downloading..."
  huggingface-cli download "${HF_REPO}" "${FILENAME}" \
    --local-dir /models \
    --local-dir-use-symlinks False
  echo "Download complete."
else
  echo "Model already exists at ${MODEL_PATH}. Skipping download."
fi

# Sanity check
if [ ! -f "${MODEL_PATH}" ]; then
  echo "ERROR: Model file not found after download attempt: ${MODEL_PATH}"
  exit 1
fi

# ── Build command ────────────────────────────────────────────
CMD_ARGS=(
  --model "${MODEL_PATH}"
  --ctx-size "${CTX_SIZE}"
  --host 0.0.0.0
  --port "${PORT}"
  --n-gpu-layers "${N_GPU_LAYERS}"
  --cache-type-k "${CACHE_TYPE_K}"
  --cache-type-v "${CACHE_TYPE_V}"
  --flash-attn on
)

# Append model-specific flags
if [ -n "${EXTRA_MODEL_FLAGS}" ]; then
  read -ra EXTRA_ARR <<< "${EXTRA_MODEL_FLAGS}"
  CMD_ARGS+=("${EXTRA_ARR[@]}")
fi

# Append user-supplied extra args
if [ -n "${EXTRA_ARGS}" ]; then
  read -ra USER_ARR <<< "${EXTRA_ARGS}"
  CMD_ARGS+=("${USER_ARR[@]}")
fi

echo "Launching: /app/llama-server ${CMD_ARGS[*]}"

# ── Graceful shutdown ────────────────────────────────────────
# exec replaces the shell with llama-server so it becomes PID 1
# and receives SIGTERM directly from Docker on `docker stop`
exec /app/llama-server "${CMD_ARGS[@]}"
```

### Phase 3: docker-compose.yml (Day 2, ~15% effort)

**Goal**: Provide a single-file launch experience for all three models.

**Files:**
- `docker/docker-compose.yml`

**Tasks:**
- [ ] Define `rotorquant` service with GPU reservation, volume mount, port mapping
- [ ] Default to `qwen3.5-27b` model
- [ ] Use named volume `llm-models` mapped to `/models`
- [ ] Include commented-out profiles/overrides for the other two models
- [ ] Verify `docker compose up` works end to end

**Exact docker-compose.yml:**

```yaml
services:
  rotorquant:
    build:
      context: .
      dockerfile: Dockerfile
    image: rotorquant:latest
    container_name: rotorquant-server
    restart: unless-stopped
    ports:
      - "${PORT:-8080}:${PORT:-8080}"
    volumes:
      - llm-models:/models
    environment:
      - MODEL_NAME=${MODEL_NAME:-qwen3.5-27b}
      - CACHE_TYPE_K=${CACHE_TYPE_K:-iso3}
      - CACHE_TYPE_V=${CACHE_TYPE_V:-iso3}
      - CTX_SIZE=${CTX_SIZE:-}
      - N_GPU_LAYERS=${N_GPU_LAYERS:-99}
      - PORT=${PORT:-8080}
      - EXTRA_ARGS=${EXTRA_ARGS:-}
      - HF_TOKEN=${HF_TOKEN:-}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # Health check is defined in Dockerfile; override here if needed:
    # healthcheck:
    #   test: ["CMD", "curl", "-sf", "http://localhost:8080/health"]
    #   interval: 30s
    #   timeout: 5s
    #   start_period: 120s
    #   retries: 3

volumes:
  llm-models:
    name: llm-models
```

**Usage examples:**

```bash
# Default: Qwen3.5-27B with RotorQuant iso3
docker compose up

# Reasoning model
MODEL_NAME=qwen3.5-27b-reasoning docker compose up

# Gemma 4 MoE
MODEL_NAME=gemma4-26b docker compose up

# Disable RotorQuant (f16 KV cache) for comparison
CACHE_TYPE_K=f16 CACHE_TYPE_V=f16 docker compose up

# Custom context size + extra args
CTX_SIZE=65536 EXTRA_ARGS="--batch-size 4096" docker compose up
```

### Phase 4: Integration Testing and Validation (Day 2-3, ~25% effort)

**Goal**: Verify all three models start, serve, and respond correctly.

**Files:**
- `docker/test_docker.sh`

**Tasks:**
- [ ] Build image: `docker build -t rotorquant docker/`
- [ ] Test Qwen3.5-27B: start container, wait for health check, send chat completion request via curl, verify JSON response
- [ ] Test Qwen3.5-27B-Reasoning: same flow
- [ ] Test Gemma 4 26B: same flow, verify MoE model loads without errors
- [ ] Test health endpoint returns `{"status":"ok"}` (or equivalent)
- [ ] Test graceful shutdown: `docker stop` completes within 30 seconds (not killed)
- [ ] Test model persistence: stop container, restart, verify no re-download (check logs for "Skipping download")
- [ ] Test CACHE_TYPE override: start with `CACHE_TYPE_K=f16 CACHE_TYPE_V=f16`, verify server logs show f16
- [ ] Test invalid MODEL_NAME: verify container exits with error listing valid names
- [ ] Test EXTRA_ARGS: pass `--batch-size 1024`, verify it appears in server startup

**Exact test_docker.sh:**

```bash
#!/usr/bin/env bash
set -euo pipefail

IMAGE="rotorquant:latest"
VOLUME="llm-models-test"
PORT=8090

cleanup() {
  echo "Cleaning up..."
  docker stop rq-test 2>/dev/null || true
  docker rm rq-test 2>/dev/null || true
}
trap cleanup EXIT

wait_for_health() {
  local max_wait=300
  local elapsed=0
  echo "Waiting for server health (up to ${max_wait}s)..."
  while [ $elapsed -lt $max_wait ]; do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
      echo "Server healthy after ${elapsed}s"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "ERROR: Server did not become healthy within ${max_wait}s"
  docker logs rq-test
  return 1
}

test_chat_completion() {
  local model_name=$1
  echo "Testing chat completion for ${model_name}..."
  response=$(curl -sf "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [{"role": "user", "content": "Say hello in exactly 5 words."}],
      "max_tokens": 32
    }')
  if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['message']['content']"; then
    echo "PASS: ${model_name} chat completion returned valid response"
  else
    echo "FAIL: ${model_name} chat completion"
    echo "$response"
    return 1
  fi
}

test_model() {
  local model_name=$1
  echo ""
  echo "========== Testing model: ${model_name} =========="
  cleanup

  docker run -d --name rq-test \
    --gpus all \
    -e MODEL_NAME="${model_name}" \
    -e PORT="${PORT}" \
    -v "${VOLUME}:/models" \
    -p "${PORT}:${PORT}" \
    "${IMAGE}"

  wait_for_health
  test_chat_completion "${model_name}"

  echo "Testing graceful shutdown..."
  time docker stop rq-test
  echo "PASS: ${model_name} graceful shutdown"
}

echo "=== Invalid model name test ==="
docker run --rm -e MODEL_NAME=nonexistent "${IMAGE}" 2>&1 | grep -q "Unknown MODEL_NAME" \
  && echo "PASS: invalid model name rejected" \
  || echo "FAIL: invalid model name not rejected"

for model in qwen3.5-27b qwen3.5-27b-reasoning gemma4-26b; do
  test_model "$model"
done

echo ""
echo "=== All tests passed ==="
```

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `docker/Dockerfile` | Create | Multi-stage build: CUDA devel (compile) -> CUDA runtime (serve) |
| `docker/entrypoint.sh` | Create | Model config lookup, HF download, server launch, signal forwarding |
| `docker/docker-compose.yml` | Create | One-command GPU-enabled server with named volume |
| `docker/.dockerignore` | Create | Exclude build artifacts, .git from Docker context |
| `docker/test_docker.sh` | Create | Integration tests for all 3 models, health, shutdown, error cases |
| `docker/benchmark_rotorquant.py` | Copy | Copied from `scripts/benchmark_rotorquant.py` for in-container benchmarks |

---

## Definition of Done

### Build
- [ ] `docker build -t rotorquant docker/` completes without errors
- [ ] Image size under 8 GB (verify with `docker images rotorquant`)
- [ ] Build uses cached layers for CUDA toolkit (only recompiles llama.cpp on code change)

### Model Download
- [ ] First run with `MODEL_NAME=qwen3.5-27b` downloads model to `/models/` volume
- [ ] Subsequent runs with same model skip download (log says "Skipping download")
- [ ] Switching to `MODEL_NAME=gemma4-26b` triggers download of only that model
- [ ] Invalid `MODEL_NAME` exits with code 1 and prints valid model names

### Server Operation
- [ ] `docker run --gpus all -e MODEL_NAME=qwen3.5-27b -v llm-models:/models -p 8080:8080 rotorquant` serves correctly
- [ ] `/health` endpoint returns HTTP 200 within 120 seconds of container start
- [ ] `/v1/chat/completions` returns valid JSON with generated text for all 3 models
- [ ] Chat templates auto-detected from GGUF metadata (no manual `--chat-template` needed)
- [ ] Server logs show `cache_type_k = iso3` and `cache_type_v = iso3` by default

### RotorQuant KV Cache
- [ ] Default KV cache is iso3/iso3 (RotorQuant); visible in server startup logs
- [ ] `CACHE_TYPE_K=f16 CACHE_TYPE_V=f16` overrides to standard f16 cache
- [ ] All RotorQuant types (`iso3`, `iso4`, `turbo2`, `turbo3`, `turbo4`, `planar3`, `planar4`) accepted

### Gemma 4 MoE
- [ ] Gemma 4 26B loads and generates without MoE-related errors
- [ ] Server startup logs show correct model architecture (MoE expert count)
- [ ] Sampling defaults (temp=1.0, top_p=0.95, top_k=64) applied as server defaults
- [ ] Per-request sampling overrides still work (e.g., `temperature: 0.5` in API call)

### Health Check and Shutdown
- [ ] Docker health check transitions from `starting` -> `healthy` after model load
- [ ] `docker stop` triggers SIGTERM -> llama-server shuts down within 30 seconds
- [ ] No zombie processes (llama-server runs as PID 1 via `exec`)

### docker-compose
- [ ] `docker compose up` with no env vars starts Qwen3.5-27B on port 8080
- [ ] `MODEL_NAME=gemma4-26b docker compose up` starts Gemma 4
- [ ] `docker compose down` + `docker compose up` reuses existing model on volume

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CUDA architecture mismatch: image built for SM 80-120 but user has older GPU | Low | High | Build ARG `CMAKE_CUDA_ARCHITECTURES` is configurable; document minimum SM 80 (A100+) requirement |
| huggingface-cli download fails (rate limit, auth needed for gated repo) | Medium | Medium | Support `HF_TOKEN` env var for authenticated downloads; entrypoint validates file exists after download |
| Gemma 4 GGUF quantization (UD-Q4_K_XL) not yet stable in this fork | Low | High | Test Gemma 4 early in Phase 4; fallback to standard Q4_K_M if needed |
| libllama.so shared lib not found at runtime | Low | Medium | Set `LD_LIBRARY_PATH=/app` in Dockerfile; copy all `libggml*.so*` too |
| Model file >20 GB fills /tmp during download | Low | Medium | huggingface-cli downloads directly to `--local-dir /models`, no /tmp staging |
| Docker build takes >30 minutes due to CUDA compilation | Medium | Low | ccache in build stage; build only needed targets (llama-server, llama-perplexity) |
| Port conflict if user runs multiple containers | Low | Low | PORT is configurable via env var; docker-compose maps `${PORT}:${PORT}` |

---

## Security Considerations

- **No secrets baked into image**: `HF_TOKEN` is passed as runtime env var, never in Dockerfile.
- **Model provenance**: Models downloaded from HuggingFace at runtime. Users should verify
  repo checksums if deploying in production. The entrypoint could be extended with SHA256
  verification per model (deferred -- not in scope for this sprint).
- **Network exposure**: Server binds to `0.0.0.0:8080` inside the container. In production,
  place behind a reverse proxy with authentication. The llama.cpp server has no built-in auth.
- **Container runs as root**: Acceptable for local/dev use. For production, add a non-root
  user and `chown` the `/models` volume. Deferred to a hardening sprint.
- **No outbound network after startup**: After model download, the server makes no external
  network calls. The container could be network-isolated after init if desired.

---

## Dependencies

**Build-time:**
- `nvidia/cuda:12.8.1-devel-ubuntu24.04` base image
- `cmake >= 3.14`, `git`, `build-essential`, `ccache`
- Source: `github.com/johndpope/llama-cpp-turboquant` branch `feature/planarquant-kv-cache`

**Runtime:**
- `nvidia/cuda:12.8.1-runtime-ubuntu24.04` base image
- NVIDIA Container Toolkit on host (`nvidia-docker2` or `nvidia-container-toolkit`)
- `python3`, `huggingface_hub[cli]` (for model downloads)
- `curl` (for health checks)
- NVIDIA GPU with compute capability >= 8.0 (A100, RTX 3090, RTX 4090, RTX 5090)

**Host requirements:**
- Docker >= 24.0 with Compose V2
- NVIDIA driver >= 535 (for CUDA 12.x runtime)
- Sufficient disk: ~20 GB per model on the Docker volume

---

## Open Questions

1. **Pin commit or track branch HEAD?** The Dockerfile clones the `feature/planarquant-kv-cache`
   branch. For reproducibility, we should pin to a specific commit hash via the `LLAMA_CPP_COMMIT`
   build arg. Which commit to pin -- latest as of sprint start, or a tagged release?

2. **Multiple GPU support**: The current config uses `--n-gpu-layers 99` which loads the
   entire model onto one GPU. For multi-GPU setups, llama.cpp supports tensor splitting
   via `--tensor-split`. Should we expose a `TENSOR_SPLIT` env var now, or defer?

3. **Gated model access**: The reasoning model
   (`mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF`) may require
   HuggingFace authentication. Does `HF_TOKEN` env var suffice, or do we need `huggingface-cli login`
   inside the entrypoint?

4. **Image registry**: Should we push the built image to a registry (GHCR, Docker Hub) for
   team use, or is local `docker build` sufficient for now?

5. **Benchmark inclusion**: The benchmark script is included in the image for convenience.
   It requires a running server instance. Should it be a separate compose service, or is
   `docker exec rotorquant-server python3 /app/benchmark_rotorquant.py` sufficient?

6. **SWA (Sliding Window Attention) flag**: Gemma 4 uses sliding window attention.
   llama.cpp has `--swa-full` to use full-size SWA cache. Should this be a default for
   Gemma 4, or does the default behavior (auto-detect from GGUF metadata) handle it correctly?
