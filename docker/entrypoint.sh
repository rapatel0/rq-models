#!/bin/bash
set -euo pipefail

# ============================================================================
# RotorQuant llama.cpp Server — Entrypoint
#
# Env vars:
#   MODEL_NAME     (required)  One of: qwen3.6-35b, qwen3.6-27b, qwen3.5-27b, qwen3.5-27b-reasoning, gemma4-26b, ...
#   KV_CACHE_TYPE  (optional)  KV cache quantization type (default: iso4)
#   CTX_SIZE       (optional)  Context window size (default: per-model)
#   PORT           (optional)  Server port (default: 8080)
#   GPU_LAYERS     (optional)  Layers to offload to GPU (default: 99 = all)
#   N_PARALLEL     (optional)  Concurrent request slots (default: 2)
#   CACHE_RAM      (optional)  Prompt cache size in MiB, system RAM (default: 8192)
#   HF_TOKEN       (optional)  HuggingFace token for gated models
#   EXTRA_ARGS     (optional)  Additional llama-server flags
# ============================================================================

# ── Model Registry ──────────────────────────────────────────────────────────
# Format: "HF_REPO|FILENAME|DEFAULT_CTX|EXTRA_FLAGS"
declare -A MODELS=(
  # ── Qwen3.6-35B-A3B MoE (default) ───────────────────────────────────
  # 32 GB+ (RTX 5090, A100): UD-Q4_K_XL — best quality
  [qwen3.6-35b]="unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf|262144|"
  # 24 GB (RTX 4090): UD-Q3_K_XL — best fit
  [qwen3.6-35b-q3]="unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf|131072|"
  # 16 GB (RTX 5060/4060 Ti): IQ3_XXS — max compression
  [qwen3.6-35b-iq3]="unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf|32768|"

  # ── Qwen3.6-27B dense ────────────────────────────────────────────────
  # 32 GB+ (RTX 5090, A100): UD-Q4_K_XL — best quality
  [qwen3.6-27b]="unsloth/Qwen3.6-27B-GGUF|Qwen3.6-27B-UD-Q4_K_XL.gguf|131072|"
  # 24 GB (RTX 4090): UD-Q3_K_XL
  [qwen3.6-27b-q3]="unsloth/Qwen3.6-27B-GGUF|Qwen3.6-27B-UD-Q3_K_XL.gguf|131072|"
  # 16 GB (RTX 5060/4060 Ti): UD-IQ3_XXS
  [qwen3.6-27b-iq3]="unsloth/Qwen3.6-27B-GGUF|Qwen3.6-27B-UD-IQ3_XXS.gguf|32768|"

  # ── Qwen3.5-27B dense (legacy) ───────────────────────────────────────
  # 24-32 GB GPUs (Q4 — best quality)
  [qwen3.5-27b]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-Q4_K_M.gguf|114688|"
  [qwen3.5-27b-reasoning]="mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF|Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-Q4_K_M.gguf|114688|"
  [gemma4-26b]="unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf|114688|--samplers top_p,top_k,temperature --temp 1.0 --top-p 0.95 --top-k 64"

  # 16 GB GPUs (imatrix quants — fit with usable context)
  [qwen3.5-27b-q3]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-UD-Q3_K_XL.gguf|28672|"
  [qwen3.5-27b-q3-xxs]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-UD-IQ3_XXS.gguf|57344|"
  [qwen3.5-27b-iq4]="unsloth/Qwen3.5-27B-GGUF|Qwen3.5-27B-IQ4_XS.gguf|14336|"
  [gemma4-26b-q3]="unsloth/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-UD-Q3_K_M.gguf|40960|--samplers top_p,top_k,temperature --temp 1.0 --top-p 0.95 --top-k 64"

  # ── DeepSeek V4-Flash MoE (158B total, ~30B active) ──────────────────
  # GGUF declares general.architecture=deepseek2; runs on the existing
  # DEEPSEEK2 codepath. Needs ~120 GB VRAM at Q4 — TP across 4× V100 fits.
  # Recommended runtime args:
  #   SPLIT_MODE=row  TENSOR_SPLIT="1,1,1,1"  N_PARALLEL=1  CTX_SIZE=32768
  [dsv4-flash]="tecaprovn/deepseek-v4-flash-gguf|DeepSeekV4-Flash-158B-Q4_K_M.gguf|32768|"
  [dsv4-flash-q3]="tecaprovn/deepseek-v4-flash-gguf|DeepSeekV4-Flash-158B-Q3_K_M.gguf|32768|"
  [dsv4-flash-q5]="tecaprovn/deepseek-v4-flash-gguf|DeepSeekV4-Flash-158B-Q5_K_M.gguf|32768|"
)

# ── Parse env vars ──────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:?ERROR: MODEL_NAME is required. Options: ${!MODELS[*]}}"
KV_CACHE="${KV_CACHE_TYPE:-planar4}"
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
  --parallel "${N_PARALLEL:-2}"
  --cache-type-k "$KV_CACHE"
  --cache-type-v "$KV_CACHE"
  --cache-ram "${CACHE_RAM:-8192}"
  --flash-attn on
  --host 0.0.0.0
  --port "$PORT"
)

# Multi-GPU controls (no-ops when only 1 GPU is visible to the container).
# When the orchestrator (k8s, docker --gpus '"device=…"', etc.) exposes
# multiple GPUs, set these to control how layers / tensors split:
#   SPLIT_MODE     layer | row | none           (llama.cpp --split-mode)
#   TENSOR_SPLIT   "1,1,1,1" — proportional per-GPU weight share
#   MAIN_GPU       0..N-1
if [ -n "${SPLIT_MODE:-}" ]; then
  CMD+=(--split-mode "$SPLIT_MODE")
fi
if [ -n "${TENSOR_SPLIT:-}" ]; then
  CMD+=(--tensor-split "$TENSOR_SPLIT")
fi
if [ -n "${MAIN_GPU:-}" ]; then
  CMD+=(--main-gpu "$MAIN_GPU")
fi

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
echo "║  Parallel: ${N_PARALLEL:-2} slots"
echo "║  Port:     $PORT"
echo "║  GPU:      $NGL layers offloaded"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Starting: ${CMD[*]}"
echo ""

# ── Launch server (exec replaces shell for proper signal handling) ──────────
exec "${CMD[@]}"
