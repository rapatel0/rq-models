# Sprint 002: Infrastructure and Deployment (RotorQuant Dockerization)

## Overview
This sprint focuses on the containerization of the RotorQuant llama.cpp fork to provide a production-ready, high-performance inference server. We will implement a multi-stage Docker build that produces a slim runtime image, an entrypoint script for automated model management (download/validation), and a Docker Compose orchestration setup for easy deployment. The primary goal is to support the RTX 5090's capabilities with ISO3/ISO3 KV cache quantization enabled by default.

## Use Cases
- **Dynamic Model Switching:** Run different models (Qwen 3.5, Gemma 4) by simply changing the `MODEL_NAME` environment variable.
- **Automated Infrastructure:** Zero-touch model downloading from Hugging Face on the first run.
- **Optimized Memory Usage:** Leverage RotorQuant's ISO3/ISO3 KV cache compression to fit 128K context for large models (26B-27B) within 32GB VRAM.
- **Local Benchmarking:** Easily run the `benchmark_rotorquant.py` script inside the container environment.

## Architecture
The solution follows a standard containerized microservice architecture:
- **Build Stage:** Compiles the RotorQuant fork with CUDA support (fused kernels).
- **Runtime Stage:** A slim Ubuntu-based image containing only necessary binaries and Python scripts.
- **Persistent Storage:** Models are stored in a named Docker volume (`llm-models`) mapped to `/models`.
- **Runtime Configuration:** All parameters (model name, KV cache type, threads, GPU layers) are controlled via environment variables.

## Implementation

### Phase 1: Multi-stage Dockerfile
- **Task 1.1:** Define `build` stage using `nvidia/cuda:12.4.0-devel-ubuntu22.04`.
- **Task 1.2:** Implement optimized build commands for RotorQuant (fused kernels enabled).
- **Task 1.3:** Define `runtime` stage using `nvidia/cuda:12.4.0-runtime-ubuntu22.04`.
- **Task 1.4:** Copy only the required binaries (`llama-server`, `llama-cli`) and the benchmark script.

### Phase 2: Entrypoint & Model Management
- **Task 2.1:** Create `scripts/entrypoint.sh` to parse `MODEL_NAME`.
- **Task 2.2:** Implement Hugging Face download logic using `llama-server`'s native `-hf` flag or a custom downloader if needed.
- **Task 2.3:** Map `MODEL_NAME` to specific Hugging Face repositories and GGUF files via the Model Config Map.

### Phase 3: Orchestration & Configuration
- **Task 3.1:** Create `docker-compose.yml` with NVIDIA runtime support and volume mappings.
- **Task 3.2:** Implement the Model Config Map (`config/models.json`) to store repository IDs and recommended sampling parameters.
- **Task 3.3:** Set default environment variables for RotorQuant (e.g., `LLAMA_ARG_CACHE_TYPE_K=iso3`).

### Phase 4: Validation
- **Task 4.1:** Verify build on target hardware.
- **Task 4.2:** Smoke test for all 3 supported models (Qwen 3.5 27B, Reasoning, Gemma 4 26B).
- **Task 4.3:** Validate health check endpoint (`/health`).

## Files Summary
| File | Description |
|------|-------------|
| `Dockerfile` | Multi-stage build for RotorQuant (CUDA enabled) |
| `scripts/entrypoint.sh` | Runtime logic: model selection, download, and server launch |
| `docker-compose.yml` | Orchestration for the server and persistent volumes |
| `config/models.json` | Model configuration map (HF repos, filenames, defaults) |

## Definition of Done
- [ ] Docker image builds successfully without warnings.
- [ ] Server starts and serves the OpenAI-compatible API on port 8080.
- [ ] Model download is automated and persistent across container restarts.
- [ ] `benchmark_rotorquant.py` can be executed within the container.
- [ ] ISO3/ISO3 KV cache is confirmed active in the server logs.

## Risks
- **Model Size:** Large GGUF files (17GB+) may cause timeouts during the initial download phase.
- **VRAM Limits:** MoE models (Gemma 4) or 27B dense models with high context might hit the 32GB limit if offloading is not perfectly tuned.
- **Binary Compatibility:** Fused kernels depend on specific CUDA versions; ensured via static base image version.

## Security
- **API Keys:** Use `LLAMA_API_KEY` environment variable to protect the endpoint.
- **Non-root user:** Image should run as a non-root user (llama) where possible.
- **Mount permissions:** Volume mounts should be scoped correctly.

## Dependencies
- **Hardware:** NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090/5090).
- **Software:** Docker, NVIDIA Container Toolkit, `nvidia-docker2`.
- **Upstream:** RotorQuant fork (`johndpope/llama-cpp-turboquant`).

## Open Questions
- Should we include the `public_legacy` WebUI or only the new one?
- Do we need a separate sidecar container for the benchmark script reporting?
- Should we provide pre-quantized ISO3 models, or perform KV quantization at runtime (current plan)?
