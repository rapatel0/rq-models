#!/bin/bash
set -euo pipefail

# ============================================================================
# RotorQuant llama.cpp Server — Entrypoint
#
# Env vars:
#   MODEL_NAME     (required)  One of: qwen3.5-27b, qwen3.5-27b-reasoning, gemma4-26b
#   KV_CACHE_TYPE  (optional)  KV cache quantization type (default: iso3)
#   CTX_SIZE       (optional)  Context window size (default: per-model, 131072)
#   PORT           (optional)  Server port (default: 8080)
#   GPU_LAYERS     (optional)  Layers to offload to GPU (default: 99 = all)
#   HF_TOKEN       (optional)  HuggingFace token for gated models
#   EXTRA_ARGS     (optional)  Additional llama-server flags
# ============================================================================

# ── Model Registry ──────────────────────────────────────────────────────────
# Format: "HF_REPO|FILENAME|DEFAULT_CTX|EXTRA_FLAGS"
declare -A MODELS=(
  # 24-32 GB GPUs (Q4 — best quality)
  [qwen3.5-27b]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-Q4_K_M.gguf|114688|"
  [qwen3.5-27b-reasoning]="mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF|Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-Q4_K_M.gguf|131072|"
  [gemma4-26b]="unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf|131072|--samplers top_p,top_k,temperature --temp 1.0 --top-p 0.95 --top-k 64"

  # 16 GB GPUs (imatrix quants — fit with usable context)
  [qwen3.5-27b-q3]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-UD-Q3_K_XL.gguf|32768|"
  [qwen3.5-27b-q3-xxs]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-UD-IQ3_XXS.gguf|65536|"
  [qwen3.5-27b-iq4]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-IQ4_XS.gguf|16384|"
  [gemma4-26b-q3]="unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q3_K_M.gguf|49152|--samplers top_p,top_k,temperature --temp 1.0 --top-p 0.95 --top-k 64"
)

# ── Parse env vars ──────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:?ERROR: MODEL_NAME is required. Options: ${!MODELS[*]}}"
KV_CACHE="${KV_CACHE_TYPE:-iso4}"
PORT="${PORT:-8080}"
NGL="${GPU_LAYERS:-99}"

# ── Validate model name ─────────────────────────────────────────────────────
if [[ -z "${MODELS[$MODEL_NAME]+x}" ]]; then
  echo "ERROR: Unknown MODEL_NAME='$MODEL_NAME'"
  echo "Available models:"
  for key in "${!MODELS[@]}"; do
    IFS='|' read -r repo file ctx extra <<< "${MODELS[$key]}"
    echo "  $key  →  $file ($repo)"
  done
  exit 1
fi

# ── Parse model config ──────────────────────────────────────────────────────
IFS='|' read -r HF_REPO FILENAME DEFAULT_CTX EXTRA_FLAGS <<< "${MODELS[$MODEL_NAME]}"
CTX="${CTX_SIZE:-$DEFAULT_CTX}"
MODEL_PATH="/models/${FILENAME}"

# ── Download model if missing ───────────────────────────────────────────────
if [ ! -f "$MODEL_PATH" ]; then
  echo ""
  echo "╔══════════════════════════════════════════════════╗"
  echo "║  Downloading model: $FILENAME"
  echo "║  From: $HF_REPO"
  echo "║  Size: ~17 GB — this will take a few minutes"
  echo "╚══════════════════════════════════════════════════╝"
  echo ""

  HF_ARGS=(download "$HF_REPO" "$FILENAME" --local-dir /models)
  if [ -n "${HF_TOKEN:-}" ]; then
    HF_ARGS+=(--token "$HF_TOKEN")
  fi

  hf "${HF_ARGS[@]}"

  if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Download completed but $MODEL_PATH not found"
    echo "Contents of /models:"
    ls -la /models/
    exit 1
  fi
  echo "Download complete: $(du -h "$MODEL_PATH" | cut -f1)"
else
  echo "Model already downloaded: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
fi

# ── Build server command ────────────────────────────────────────────────────
CMD=(
  /app/bin/llama-server
  --model "$MODEL_PATH"
  --n-gpu-layers "$NGL"
  --ctx-size "$CTX"
  --cache-type-k "$KV_CACHE"
  --cache-type-v "$KV_CACHE"
  --flash-attn on
  --host 0.0.0.0
  --port "$PORT"
)

# Add model-specific flags
if [ -n "$EXTRA_FLAGS" ]; then
  read -ra EXTRA_ARRAY <<< "$EXTRA_FLAGS"
  CMD+=("${EXTRA_ARRAY[@]}")
fi

# Add user-provided extra args
if [ -n "${EXTRA_ARGS:-}" ]; then
  read -ra USER_ARRAY <<< "$EXTRA_ARGS"
  CMD+=("${USER_ARRAY[@]}")
fi

# ── Print startup banner ───────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  RotorQuant LLM Server                          ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Model:    $MODEL_NAME"
echo "║  File:     $FILENAME"
echo "║  KV Cache: $KV_CACHE / $KV_CACHE (RotorQuant)"
echo "║  Context:  $CTX tokens"
echo "║  Port:     $PORT"
echo "║  GPU:      $NGL layers offloaded"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Starting: ${CMD[*]}"
echo ""

# ── Launch server (exec replaces shell for proper signal handling) ──────────
exec "${CMD[@]}"
