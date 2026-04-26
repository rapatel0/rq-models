#!/bin/bash
set -euo pipefail

# ============================================================================
# rq-vllm entrypoint — Sprint 004 Phase 0
#
# Launches the OpenAI-compatible vLLM API server with rq-models defaults.
# Reads minimal env vars; passes through the rest to vllm.entrypoints.openai.
#
# Env vars:
#   MODEL                (required)  HF model id, e.g. Qwen/Qwen3-27B
#   PORT                 (optional)  Server port (default: 8080)
#   GPU_MEMORY_UTILIZATION (opt)     Fraction of VRAM (default: 0.90)
#   MAX_MODEL_LEN        (optional)  Max context window (default: model max)
#   QUANTIZATION         (optional)  vLLM quantization arg (Phase 0: none)
#   ROTORQUANT_MODE      (optional)  Phase 1+: planar3 | iso3 | iso4 | planar4
#   EXTRA_ARGS           (optional)  Additional flags passed verbatim
# ============================================================================

MODEL="${MODEL:?ERROR: MODEL is required (HF id, e.g. Qwen/Qwen3-27B)}"
PORT="${PORT:-8080}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

ARGS=(
    --model "${MODEL}"
    --host 0.0.0.0
    --port "${PORT}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
)

if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

# Phase 0: --quantization is left unset (f16 baseline).
# Phase 1+: when RotorQuantConfig lands in the fork, set
#   QUANTIZATION=rotorquant
#   ROTORQUANT_MODE=planar3
if [[ -n "${QUANTIZATION:-}" ]]; then
    ARGS+=(--quantization "${QUANTIZATION}")
fi
if [[ -n "${ROTORQUANT_MODE:-}" ]]; then
    # Forwarded as-is; the rq-vllm fork must accept this flag in
    # RotorQuantConfig (Sprint 004 Phase 1).
    ARGS+=(--rotorquant-mode "${ROTORQUANT_MODE}")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206  # word-splitting intended
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

echo "rq-vllm starting:"
echo "  rq-vllm commit: $(cat /etc/rq-vllm-commit 2>/dev/null || echo unknown)"
echo "  model: ${MODEL}"
echo "  port: ${PORT}"
echo "  args: ${ARGS[*]}"

exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
