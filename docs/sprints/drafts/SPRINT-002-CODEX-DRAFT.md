# Sprint 002: Dockerize RotorQuant llama.cpp Server with Multi-Model GPU Support

## Overview

This sprint containerizes the RotorQuant llama.cpp server (built in Sprint 001) into production-grade Docker images with multi-model support. The deliverable enables runtime model selection (Qwen3.5-27B, Qwen3.5-27B-Reasoning-Distilled, Gemma 4 26B) on NVIDIA GPUs with RotorQuant KV cache compression enabled by default. Models persist on Docker volumes for efficient reuse across container lifecycle.

**Scope**: 4 core files (Dockerfile, entrypoint.sh, docker-compose.yml, model registry)  
**Complexity**: Low (standard Docker patterns, validated build commands)  
**Duration**: 3–5 days  
**Owner**: Engineering  

---

## Use Cases

### UC-1: Local Development with GPU
Developer pulls pre-built RotorQuant image, mounts local GPU, selects a model, and runs inference tests in isolation.

### UC-2: Production Model Serving
Team deploys container with fixed model selection, volume-persisted models, and automatic health checks via load balancer.

### UC-3: Model Benchmarking Pipeline
Benchmark script (`benchmark_rotorquant.py`) runs inside container to measure token throughput, latency, and memory with identical hardware/software configuration.

### UC-4: Model Switching Without Restart
Operator changes `MODEL_NAME` env var and restarts container; entrypoint auto-downloads new model if missing, skips if present.

### UC-5: Multi-Tenant Inference (Future)
Multiple containers on same host with different models, each bound to distinct GPU device via `--gpus "device=N"`.

---

## Architecture

### Container Stack

```
┌─ Build Stage (bullseye-slim) ────────────────────────┐
│ • CUDA 12.2 + cuDNN                                  │
│ • Build tools: cmake, gcc, git                       │
│ • Python 3.11 dev headers                            │
│ • Clone & build RotorQuant fork + dependencies       │
│ • Output: /opt/llama-cpp-rq compiled binaries         │
└────────────────────────────────────────────────────┘
          │
          ▼
┌─ Runtime Stage (bullseye-slim) ──────────────────────┐
│ • CUDA 12.2 runtime (minimal)                        │
│ • Python 3.11 runtime only                           │
│ • HF transformers, torch (inference mode)            │
│ • entrypoint.sh: model registry + server bootstrap   │
│ • /models volume: model cache (named or bind mount)   │
│ • /app/config: model registry TOML/JSON              │
│ • Health check: curl /health every 30s               │
└────────────────────────────────────────────────────┘
          │
          ▼
┌─ Docker Compose Orchestration ───────────────────────┐
│ • Service: rotorquant (build: . context)             │
│ • GPU pass-through: --gpus all                       │
│ • Volumes: models (named), config (bind)             │
│ • Port mapping: 8080:8080 (OpenAI-compatible API)    │
│ • Env: MODEL_NAME, ROTORQUANT_KV_CACHE (iso3/iso3)   │
│ • Restart policy: unless-stopped                     │
└────────────────────────────────────────────────────┘
```

### Model Registry (In-Container Lookup)

**File**: `/app/config/models.toml`

```toml
[qwen3.5-27b]
repo = "unsloth/Qwen3.5-27B-GGUF"
filename = "Qwen3.5-27B-Q4_K_M.gguf"
size_gb = 16.7
llama_cpp_flags = "--n-gpu-layers 40 --threads 8"

[qwen3.5-27b-reasoning]
repo = "mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF"
filename = "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-Q4_K_M.gguf"
size_gb = 16.6
llama_cpp_flags = "--n-gpu-layers 40 --threads 8"

[gemma4-26b]
repo = "unsloth/gemma-4-26B-A4B-it-GGUF"
filename = "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
size_gb = 17.1
# Gemma 4 MoE: 30 layers, 3.8B active params per token
# Sampling: temperature=1.0, top_p=0.95, top_k=64 recommended
llama_cpp_flags = "--n-gpu-layers 30 --threads 8"
sampling_preset = "gemma4_moe"
```

### Security & Runtime Constraints

- **No HuggingFace Tokens in Image**: Models downloaded via `huggingface-cli` at runtime; user must pass `HF_TOKEN` env var if repo requires authentication
- **Non-root User**: Container runs as `llm:llm` (UID 1000, GID 1000)
- **Signal Handling**: entrypoint.sh catches SIGTERM/SIGINT, gracefully shuts down llama.cpp server
- **Health Check**: HTTP GET `/health` returns 200 + JSON `{"status": "healthy"}` within 10s
- **Volume Permissions**: `/models` mounted with mode `755`; config bind-mounted read-only

---

## Implementation

### Phase 1: Multi-Stage Dockerfile with CUDA Build

**Files**:
- `/home/ravi/repos/turbo/Dockerfile`

**Tasks**:

- [ ] Define build stage: `FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 as builder`
  - Install: cmake, gcc-11, g++-11, git, pkg-config, libssl-dev
  - Install: Python 3.11 dev, pip, python3.11-dev
  
- [ ] Clone RotorQuant fork (branch `feature/planarquant-kv-cache`)
  ```bash
  git clone --depth 1 --branch feature/planarquant-kv-cache \
    https://github.com/johndpope/llama-cpp-turboquant.git /tmp/llama-cpp-rq
  ```
  
- [ ] Build RotorQuant binary with CUDA support
  ```bash
  cd /tmp/llama-cpp-rq && \
  mkdir -p build && \
  cd build && \
  cmake .. -DCMAKE_BUILD_TYPE=Release \
           -DGGML_CUDA=ON \
           -DCMAKE_CUDA_ARCHITECTURES=86 && \
  make -j$(nproc)
  ```
  
- [ ] Install Python dependencies (requirements-docker.txt)
  - torch>=2.0 (CPU-only for builder stage; GPU version in runtime)
  - transformers
  - huggingface-hub
  - huggingface-cli (via transformers)
  - pydantic
  - pytest (for in-container tests)
  
- [ ] Copy compiled binaries to `/opt/llama-cpp-rq/bin`, `/opt/llama-cpp-rq/lib`

- [ ] Runtime stage: `FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04`
  - Install: Python 3.11, pip, curl (health checks)
  - Copy from builder: `/opt/llama-cpp-rq/`, benchmark script
  - Create non-root user: `llm:llm`
  - Set WORKDIR `/app`, expose 8080
  
- [ ] Validate Dockerfile: `docker build --progress=plain -t rotorquant:test .`

---

### Phase 2: Entrypoint Script with Model Download & Server Bootstrap

**Files**:
- `/home/ravi/repos/turbo/entrypoint.sh`

**Tasks**:

- [ ] Script structure: POSIX-compliant shell script (not bash-specific)
  ```bash
  #!/bin/sh
  set -e
  trap 'kill $(jobs -p) 2>/dev/null || true' SIGTERM SIGINT
  ```
  
- [ ] Load model registry: Parse `/app/config/models.toml` into associative arrays
  - Model name: `MODEL_NAME` env var (default: `qwen3.5-27b`)
  - Validate: Check if `MODEL_NAME` exists in registry
  
- [ ] Model download logic (idempotent)
  ```bash
  REPO=$(get_config "$MODEL_NAME" "repo")
  FILENAME=$(get_config "$MODEL_NAME" "filename")
  MODEL_PATH="/models/$FILENAME"
  
  if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading $MODEL_NAME from $REPO..."
    huggingface-cli download "$REPO" "$FILENAME" --cache-dir /models
  else
    echo "Model $FILENAME already cached."
  fi
  ```
  
- [ ] Lookup model-specific flags from registry
  ```bash
  LLAMA_FLAGS=$(get_config "$MODEL_NAME" "llama_cpp_flags")
  ```
  
- [ ] Health check endpoint setup (background HTTP server or wrapper)
  - Simple approach: expose `/health` via llama.cpp's built-in OpenAI-compatible `/v1/models` endpoint check
  - Alternative: Spin up tiny Python/busybox HTTP server that checks port 8080 availability
  
- [ ] Start llama.cpp server with RotorQuant KV cache enabled
  ```bash
  /opt/llama-cpp-rq/bin/server \
    --model "$MODEL_PATH" \
    --port 8080 \
    --nthreads 8 \
    --n-gpu-layers 40 \
    --kv-cache-type "iso3" \
    --kv-cache-config "iso3" \
    $LLAMA_FLAGS \
    &
  wait
  ```
  
- [ ] Signal handling: forward SIGTERM to llama.cpp process, clean exit
  - Test: `docker run ... ; docker stop <container>` should gracefully shutdown within 10s

- [ ] Log model info on startup: repo, filename, size, llama.cpp flags (for auditing)

- [ ] Error handling: Fail fast if model download fails, registry invalid, or build flags missing

---

### Phase 3: Docker Compose Configuration

**Files**:
- `/home/ravi/repos/turbo/docker-compose.yml`

**Tasks**:

- [ ] Service definition: `rotorquant`
  ```yaml
  services:
    rotorquant:
      build:
        context: .
        dockerfile: Dockerfile
      image: rotorquant:latest
      container_name: rotorquant-server
      runtime: nvidia
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
  ```
  
- [ ] GPU passthrough: `runtime: nvidia` + `deploy.resources.reservations.devices`
  - Alternative (older Docker): `--gpus all` in command section
  - Ensure NVIDIA Container Runtime installed: `docker run --rm --runtime=nvidia nvidia/cuda:12.2.2-base nvidia-smi`
  
- [ ] Volume configuration
  ```yaml
  volumes:
    llm-models:
      driver: local
  services:
    rotorquant:
      volumes:
        - llm-models:/models
        - ./config:/app/config:ro
  ```
  
- [ ] Environment variables
  ```yaml
  environment:
    - MODEL_NAME=qwen3.5-27b
    - ROTORQUANT_KV_CACHE=iso3:iso3
    - TOKENIZERS_PARALLELISM=false
  ```
  
- [ ] Port mapping: `8080:8080`

- [ ] Health check
  ```yaml
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 60s
  ```
  
- [ ] Restart policy: `restart_policy: unless-stopped`

- [ ] Resource limits (optional, commented)
  ```yaml
  deploy:
    resources:
      limits:
        memory: 40G
  ```

- [ ] Test: `docker-compose up -d && docker-compose logs -f && docker-compose down`

---

### Phase 4: Model Registry Configuration & Validation

**Files**:
- `/home/ravi/repos/turbo/config/models.toml`
- `/home/ravi/repos/turbo/scripts/validate_models_config.py` (optional)

**Tasks**:

- [ ] Create `/app/config/models.toml` with all three models
  - Validate TOML syntax: `python3 -m toml models.toml` or `pip install toml`
  - Include: repo, filename, size_gb, llama_cpp_flags, optional sampling_preset
  
- [ ] Gemma 4 MoE special handling
  - Document: 30 layers, 3.8B active params (MoE 2/8)
  - Note: No special llama.cpp flags required; router token handling is automatic
  - Sampling preset: temperature=1.0, top_p=0.95, top_k=64 (store in registry for client use)
  
- [ ] Write validation script (optional, for CI/CD)
  - Check: all models exist on HF (401/404 detection)
  - Validate: file sizes match expected
  - Parse: GGUF metadata (via gguf-py or llama.cpp)
  
- [ ] Copy config to image in Dockerfile
  ```dockerfile
  COPY config/models.toml /app/config/models.toml
  ```

- [ ] Test model lookup: `docker run -e MODEL_NAME=gemma4-26b rotorquant:test /bin/sh -c 'source /app/config/models.toml && echo $REPO'`

---

### Phase 5: Benchmark Script Integration

**Files**:
- `/home/ravi/repos/turbo/scripts/benchmark_rotorquant.py` (already exists; copy to image)

**Tasks**:

- [ ] Copy benchmark script into image
  ```dockerfile
  COPY scripts/benchmark_rotorquant.py /opt/llama-cpp-rq/benchmark.py
  ```
  
- [ ] Create wrapper: `/app/run_benchmark.sh`
  - Validates model is loaded (health check)
  - Runs benchmark script against localhost:8080
  - Saves results to `/models/benchmark_${MODEL_NAME}_${DATE}.json`
  
- [ ] Document: Benchmark CLI invocation
  ```bash
  docker run --gpus all -e MODEL_NAME=qwen3.5-27b \
    -v llm-models:/models rotorquant:latest \
    /app/run_benchmark.sh
  ```

---

### Phase 6: Security Hardening & Health Checks

**Files**:
- `Dockerfile` (security sections)
- `entrypoint.sh` (signal handling)
- `docker-compose.yml` (health check, restart policy)

**Tasks**:

- [ ] Non-root user setup
  ```dockerfile
  RUN groupadd -g 1000 llm && useradd -u 1000 -g llm llm
  USER llm
  ```
  
- [ ] Verify no HF tokens baked into image layers
  - Check: `.dockerignore` excludes `.env`, `*.token`, `credentials*`
  - Test: `docker inspect rotorquant:test | grep -i 'hf_token\|huggingface'` returns empty
  
- [ ] Signal handling in entrypoint
  - SIGTERM: forward to llama.cpp, wait for graceful shutdown (10s timeout)
  - SIGINT: same behavior
  - Test: `docker stop <container>` completes in <10s
  
- [ ] Health check endpoint
  - Implement in entrypoint or wrapper: GET `/health` returns `{"status": "healthy", "model": "..."}`
  - Timeout: 10s per request
  - Retries: 3 consecutive failures = container unhealthy
  - Start grace period: 60s (model download + server startup)
  
- [ ] Read-only config: `volumes: [./config:/app/config:ro]` in docker-compose.yml

- [ ] File permissions audit
  - `/models`: 755 (readable by all, writable by llm)
  - `/app/config`: 755 (readable by all)
  - `/opt/llama-cpp-rq/bin/server`: 755 (executable)

---

## Files Summary

| File Path | Purpose | Key Content |
|-----------|---------|-------------|
| `/home/ravi/repos/turbo/Dockerfile` | Multi-stage build (CUDA 12.2 + RotorQuant fork) | Builder: clone, cmake, build; Runtime: Python 3.11, entrypoint, non-root user |
| `/home/ravi/repos/turbo/entrypoint.sh` | Model download + server bootstrap | Model registry parsing, HF cli download (idempotent), llama.cpp server start with KV cache flags, signal handling |
| `/home/ravi/repos/turbo/docker-compose.yml` | Orchestration + GPU passthrough | Service config, GPU device, volumes (models, config), health check, restart policy, env vars |
| `/home/ravi/repos/turbo/config/models.toml` | Model registry (repo, filename, flags) | 3 models: qwen3.5-27b, qwen3.5-27b-reasoning, gemma4-26b (with MoE sampling preset) |
| `/home/ravi/repos/turbo/.dockerignore` | Exclude secrets from image | `.env`, `*.token`, `__pycache__`, `.git` |
| `/home/ravi/repos/turbo/scripts/validate_models_config.py` | (Optional) Config validation | TOML parse, HF API checks, GGUF metadata extraction |

---

## Definition of Done

All checklist items per phase **completed and tested**:

- [ ] **Phase 1**: `docker build -t rotorquant:test .` succeeds, binary exists at `/opt/llama-cpp-rq/bin/server`
- [ ] **Phase 2**: `docker run rotorquant:test /bin/sh -c 'echo $MODEL_NAME'` prints `qwen3.5-27b`
- [ ] **Phase 3**: `docker-compose up -d && docker-compose ps` shows service running
- [ ] **Phase 4**: All 3 models in `models.toml` have valid HF repos and filenames (no 404s)
- [ ] **Phase 5**: Benchmark script copies to image and is executable
- [ ] **Phase 6**: No HF tokens in `docker inspect rotorquant:test`; health check passes within 90s of startup

**Integration Tests**:

- [ ] `docker-compose up -d` with default MODEL_NAME (qwen3.5-27b)
  - Server listens on 8080 (confirmed via `curl http://localhost:8080/v1/models`)
  - Health check passes (3/3 retries)
  
- [ ] Model persistence: `docker-compose down`, `docker-compose up -d`, model reused (no re-download)

- [ ] Model switching
  - `docker-compose exec rotorquant bash -c 'export MODEL_NAME=gemma4-26b && /app/entrypoint.sh'`
  - Server starts with gemma4-26b within 60s
  
- [ ] Graceful shutdown: `docker-compose stop` completes in <10s, no orphaned processes

- [ ] OpenAI API compatibility: POST `/v1/chat/completions` with valid request returns 200 + response

- [ ] Benchmark runs: `/app/run_benchmark.sh` produces JSON output

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| **HF Token Leakage** | Medium | Critical | No tokens in Dockerfile/image; pass via `HF_TOKEN` env var at runtime; audit image layers post-build |
| **GPU OOM on Model Load** | Low | High | Pre-validate model sizes fit RTX 5090 (32GB); set `--n-gpu-layers` conservatively; document memory requirements |
| **Entrypoint Script Bugs** | Medium | Medium | Test POSIX compliance (sh, not bash); validate signal handling with `docker stop`; log all steps to stdout |
| **Model Download Timeout** | Low | Medium | Set reasonable timeout (600s) for huggingface-cli; handle partial downloads gracefully |
| **Docker Build Cache Invalidation** | Low | Low | Pinned CUDA base image version (`12.2.2-runtime-ubuntu22.04`); document rebuild triggers |
| **Gemma 4 MoE Router Token Overhead** | Low | Low | Monitor token throughput vs. baseline Qwen3.5; set `top_k=64` to limit router sampling |

---

## Security

### Threat Model

1. **Unauthorized Model Access**: Container runs as non-root; volumes accessible only to `llm:llm` user
2. **Secrets in Image Layers**: No HF tokens, credentials, or API keys baked in
3. **Arbitrary Code Execution via Model Input**: llama.cpp sandbox (input is prompt tokens only); no file I/O from model
4. **Denial of Service via Resource Exhaustion**: Health check + restart policy mitigates hung processes; GPU memory capped per model

### Controls

| Control | Mechanism |
|---------|-----------|
| **Non-Root User** | `USER llm` (UID 1000) in Dockerfile; prevents container escape privilege escalation |
| **Read-Only Config** | `config:/app/config:ro` mount in docker-compose; entrypoint validates before use |
| **No Token Injection** | `huggingface-cli` reads `HF_TOKEN` from env var (injected at `docker run`); no hardcoding in image |
| **Signal Handling** | SIGTERM caught in entrypoint.sh; llama.cpp server gracefully shutdowns within 10s |
| **Health Check** | Periodic HTTP GET; fails if server unresponsive; triggers container restart |
| **Audit Logging** | entrypoint.sh logs model name, repo, size, flags to stdout (captured by Docker daemon) |
| **Volume Permissions** | `/models` mode 755; `config` read-only bind mount |

### Validation

- [ ] Security scan: `docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image rotorquant:test`
- [ ] No hardcoded secrets: `docker inspect rotorquant:test | grep -i 'env\|hf_token\|token' | grep -v TOKENIZERS_PARALLELISM`
- [ ] Non-root verify: `docker run rotorquant:test whoami` outputs `llm`

---

## Dependencies

### External

| Dependency | Version | Purpose | License |
|------------|---------|---------|---------|
| NVIDIA CUDA | 12.2.2 | GPU compute (RotorQuant kernels) | Proprietary |
| Ubuntu | 22.04 LTS | Base OS image | GPL |
| NVIDIA Container Runtime | ≥1.13 | Docker GPU passthrough | Proprietary |
| HuggingFace transformers | ≥4.30 | Model metadata, tokenizer handling | Apache 2.0 |
| PyTorch | ≥2.0 | Dependency of transformers | BSD |
| huggingface-cli | bundled in transformers | Model download from HF hub | Apache 2.0 |

### Internal

| Dependency | Path | Reason |
|------------|------|--------|
| RotorQuant Fork | `johndpope/llama-cpp-turboquant:feature/planarquant-kv-cache` | KV cache compression + CUDA kernels |
| Benchmark Script | `/home/ravi/repos/turbo/scripts/benchmark_rotorquant.py` | Token throughput measurements |
| Base Image | `nvidia/cuda:12.2.2-runtime-ubuntu22.04` | CUDA libraries + libc |

### Build-Time (Consumed During Docker Build, Not in Runtime Image)

- cmake, gcc-11, git, python3.11-dev, pkg-config, libssl-dev

---

## Open Questions

1. **Gemma 4 MoE Token Count**: Does llama.cpp report total tokens or unique tokens (excluding router)? Affects billing/throughput metrics.
   - **Resolution Path**: Run benchmark with `--verbose`, inspect token count output, compare to Qwen3.5 baseline

2. **RotorQuant KV Cache Flag Syntax**: Does `--kv-cache-type` and `--kv-cache-config` accept `iso3:iso3` or separate args?
   - **Resolution Path**: Run `/opt/llama-cpp-rq/bin/server --help | grep -i kv`, test with `docker run rotorquant:test server --help`

3. **Model Download Resume on Network Interruption**: Does `huggingface-cli download` support partial resume, or must entire file re-download?
   - **Resolution Path**: Test by interrupting download mid-way, restart container, observe huggingface-cli behavior

4. **Health Check Endpoint**: Should `/health` be custom (wrapper HTTP server) or leverage llama.cpp's OpenAI API?
   - **Resolution Path**: Test `curl http://localhost:8080/v1/models` response time under load; if <10s consistently, use as proxy health check

5. **GPU Device Selection**: For multi-GPU hosts, should entrypoint respect `CUDA_VISIBLE_DEVICES` or `docker-compose` `device` list?
   - **Resolution Path**: Document both patterns; default to `--gpus all` for single-GPU dev, `device=N` for production multi-GPU

6. **Model Preload on Startup**: Should first request wait for model load, or preload model in entrypoint before server start?
   - **Resolution Path**: Preload in entrypoint (blocks startup); server immediately ready for requests; measure startup time

7. **Sampling Preset Per-Model**: Store in registry (`gemma4_moe`) or enforce via client? Who owns sampling config?
   - **Resolution Path**: Document in config as metadata; client reads and applies; entrypoint logs recommendation to stdout

---

## Appendices

### A. Build & Run Quick Start

```bash
# Build
docker build -t rotorquant:latest .

# First run (downloads model, ~30 min on RTX 5090)
docker-compose up -d
docker-compose logs -f rotorquant

# Check health
curl http://localhost:8080/health

# Test inference
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.5-27b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'

# Switch model (restart required)
docker-compose down
docker-compose up -d -e MODEL_NAME=gemma4-26b

# Benchmark
docker-compose exec rotorquant /app/run_benchmark.sh

# Graceful shutdown
docker-compose stop  # <10s
```

### B. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_NAME` | `qwen3.5-27b` | Which model to load (key in models.toml) |
| `ROTORQUANT_KV_CACHE` | `iso3:iso3` | KV cache compression type:config |
| `HF_TOKEN` | (empty) | HuggingFace token for private repos (optional) |
| `TOKENIZERS_PARALLELISM` | `false` | Suppress huggingface warning |

### C. File Permissions & Non-Root Behavior

```bash
# Inside container (as llm:llm)
$ whoami
llm

$ ls -la /models
drwxr-xr-x  3 llm  llm  4096 Apr  2 10:00 .

$ ls -la /app/config
drwxr-xr-x  2 root root 4096 Apr  2 10:00 .
-rw-r--r--  1 root root  512 Apr  2 10:00 models.toml

$ ls -la /opt/llama-cpp-rq/bin
-rwxr-xr-x  1 root root 50M Apr  2 10:00 server
```

### D. Troubleshooting

**Container exits immediately**:
- `docker-compose logs rotorquant` → inspect entrypoint errors
- Check: `MODEL_NAME` in models.toml, HF repo accessible, disk space for model

**Model download hangs**:
- `docker exec rotorquant ps aux | grep huggingface-cli` → check if still running
- Increase `start_period` in health check from 60s to 180s
- Check network access: `docker run --rm ubuntu:22.04 curl https://huggingface.co`

**GPU not visible**:
- Host: `nvidia-smi` → confirm NVIDIA driver installed
- Host: `docker run --rm --runtime=nvidia nvidia/cuda:12.2.2-base nvidia-smi` → test Container Runtime
- Compose: Ensure `runtime: nvidia` or `--gpus all` enabled

**Server port in use**:
- Change `ports: ["8081:8080"]` in docker-compose.yml
- Or: `docker-compose down && docker-compose up`

