# RotorQuant llama.cpp server — multi-model Docker image
#
# Self-contained multi-stage build with CUDA 13.1.
# Compiles the RotorQuant llama.cpp fork from source inside Docker.
#
# Build: docker build -t rotorquant -f docker/Dockerfile .
# Run:   docker run --gpus all -e MODEL_NAME=qwen3.5-27b -v llm-models:/models -p 8080:8080 rotorquant

# ============================================================================
# Build Stage: compile llama.cpp with RotorQuant + CUDA 13.1
# ============================================================================
FROM nvidia/cuda:13.1.0-devel-ubuntu24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# Sprint 004 (rebase + DFlash cherry-pick) — pinned to rapatel0 fork.
# Rebases 150-commit RotorQuant fork onto upstream master 78433f606 and
# squash-merges PR #22105 (DFlash) at upstream pinned-head 67cb0d507.
ARG ROTORQUANT_REPO=https://github.com/rapatel0/llama-cpp-turboquant.git
ARG ROTORQUANT_BRANCH=feature/sprint-004-rebase-dflash
ARG ROTORQUANT_COMMIT=bd7a7aabbfd247ae3bdc7a61c04194eae0c9e1be

RUN git clone --depth 100 -b ${ROTORQUANT_BRANCH} \
    ${ROTORQUANT_REPO} /src \
    && cd /src && git checkout ${ROTORQUANT_COMMIT}

WORKDIR /src

# SM 80=A100, 86=A10G, 89=RTX 4090, 90=H100, 100=B100, 120=RTX 5090
# Link against CUDA driver stub for cuGetErrorString etc.
ENV LDFLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build -j$(nproc)

# ============================================================================
# Runtime Stage: slim image with binaries + model download tooling
# ============================================================================
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl python3 python3-pip ca-certificates libgomp1 \
    && pip3 install --break-system-packages --no-cache-dir huggingface-hub[cli] \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Non-root user for runtime
RUN useradd -m -u 1001 -s /bin/bash llm

# Copy binaries and shared libraries from build stage.
# Sprint 004: speculative-simple, checkpoint-bench, cli, bench added for DFlash
# smoke + Phase 5 benchmark harness.
COPY --from=build /src/build/bin/llama-server /app/bin/llama-server
COPY --from=build /src/build/bin/llama-perplexity /app/bin/llama-perplexity
COPY --from=build /src/build/bin/llama-cli /app/bin/llama-cli
COPY --from=build /src/build/bin/llama-bench /app/bin/llama-bench
COPY --from=build /src/build/bin/llama-checkpoint-bench /app/bin/llama-checkpoint-bench
COPY --from=build /src/build/bin/llama-speculative-simple /app/bin/llama-speculative-simple
COPY --from=build /src/build/bin/lib*.so* /app/lib/

# Copy entrypoint and benchmark script
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY scripts/benchmark_rotorquant.py /app/benchmark.py
RUN chmod +x /app/entrypoint.sh

# Create models directory (volume mount point) and HF cache dir
RUN mkdir -p /models && chown llm:llm /models
RUN mkdir -p /home/llm/.cache/huggingface && chown -R llm:llm /home/llm/.cache

ENV LD_LIBRARY_PATH=/app/lib
ENV PATH="/app/bin:${PATH}"
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

USER llm
WORKDIR /app

ENTRYPOINT ["/app/entrypoint.sh"]
