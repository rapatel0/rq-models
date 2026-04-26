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

# Phase 0: KV cache dtype unset (vLLM defaults to model dtype, fp16/bf16).
# Phase 1+: RotorQuant integrates as a NEW value of --kv-cache-dtype, e.g.
#   --kv-cache-dtype rotorquant_planar3
# (Architecture corrected after vLLM source review: RotorQuant is KV-cache
# compression, NOT weight quantization. The plug-in point is the CacheDType
# Literal in vllm/config/cache.py, not the QuantizationConfig registry.)
if [[ -n "${ROTORQUANT_MODE:-}" ]]; then
    case "${ROTORQUANT_MODE}" in
        planar3|iso3|iso4|planar4)
            ARGS+=(--kv-cache-dtype "rotorquant_${ROTORQUANT_MODE}")
            ;;
        *)
            echo "ERROR: unsupported ROTORQUANT_MODE='${ROTORQUANT_MODE}'." >&2
            echo "Valid: planar3 (Sprint 004), iso3 / iso4 / planar4 (Sprint 005+)." >&2
            exit 2
            ;;
    esac
fi
# QUANTIZATION (weight quantization, e.g., gptq/awq) is orthogonal to
# RotorQuant KV; pass it through if the user sets it for a quantized
# model checkpoint.
if [[ -n "${QUANTIZATION:-}" ]]; then
    ARGS+=(--quantization "${QUANTIZATION}")
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
echo "  args: ${ARGS[*]} $*"

# Pass through any positional args from `docker run rq-vllm <flags>` so users
# can add ad-hoc CLI flags (e.g., --enforce-eager) without rebuilding the
# image or threading another env var through.
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}" "$@"
