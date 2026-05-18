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
#   UBATCH_SIZE    (optional)  Physical batch size (MTP default: 32)
#   CACHE_RAM      (optional)  Prompt cache size in MiB, system RAM (default: 8192)
#   MTP_SPEC_TYPE  (optional)  auto, draft-mtp, or mtp (default: auto)
#   MTP_DRAFT_N_MAX (optional) MTP draft tokens per step (default: 4)
#   MTP_DRAFT_P_MIN (optional) MTP minimum draft probability (default: 0.75)
#   NO_WARMUP      (optional)  1/true/on to pass --no-warmup (MTP default: 1)
#   MTP_MLOCK      (optional)  1/true/on to pass --mlock (requires memlock privileges)
#   PREVIEW        (optional)  1/true/on to allow explicitly gated preview paths
#   MTP_MULTISLOT  (optional)  1/true/on to allow preview MTP with N_PARALLEL>1
#   HF_TOKEN       (optional)  HuggingFace token for gated models
#   EXTRA_ARGS     (optional)  Additional llama-server flags
# ============================================================================

# ── Model Registry ──────────────────────────────────────────────────────────
# Format: "HF_REPO|FILENAME|DEFAULT_CTX|EXTRA_FLAGS"
declare -A MODELS=(
  # ── Qwen3.6-35B-A3B MoE (default) ───────────────────────────────────
  # 32 GB+ (RTX 5090, A100): UD-Q4_K_XL — best quality
  [qwen3.6-35b]="unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf|262144|"
  # MTP speculative decoding. Requires an MTP-capable llama.cpp build.
  [qwen3.6-35b-mtp]="unsloth/Qwen3.6-35B-A3B-MTP-GGUF|Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf|262144|"
  # 24 GB (RTX 4090): UD-Q3_K_XL — best fit
  [qwen3.6-35b-q3]="unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf|131072|"
  # 16 GB (RTX 5060/4060 Ti): IQ3_XXS — max compression
  [qwen3.6-35b-iq3]="unsloth/Qwen3.6-35B-A3B-GGUF|Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf|32768|"

  # ── Qwen3.6-27B dense ────────────────────────────────────────────────
  # 32 GB+ (RTX 5090, A100): UD-Q4_K_XL — best quality
  [qwen3.6-27b]="unsloth/Qwen3.6-27B-GGUF|Qwen3.6-27B-UD-Q4_K_XL.gguf|131072|"
  # MTP speculative decoding. Requires an MTP-capable llama.cpp build.
  [qwen3.6-27b-mtp]="unsloth/Qwen3.6-27B-MTP-GGUF|Qwen3.6-27B-UD-Q4_K_XL.gguf|131072|"
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
MTP_SPEC_TYPE="${MTP_SPEC_TYPE:-auto}"
MTP_DRAFT_N_MAX="${MTP_DRAFT_N_MAX:-4}"
MTP_DRAFT_P_MIN="${MTP_DRAFT_P_MIN:-0.75}"
LLAMA_HELP_CACHE=""

load_llama_help() {
  if [ -z "$LLAMA_HELP_CACHE" ]; then
    LLAMA_HELP_CACHE="$(/app/bin/llama-server --help 2>&1 || true)"
  fi
}

help_has_word() {
  local word="$1"
  load_llama_help
  grep -qw -- "$word" <<< "$LLAMA_HELP_CACHE"
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_kv_cache_type() {
  local requested="$1"
  local base suffixed

  if help_has_word "$requested"; then
    printf '%s' "$requested"
    return
  fi

  case "$requested" in
    planar3|iso3|planar4|iso4|tbq3|tbq4)
      suffixed="${requested}_0"
      if help_has_word "$suffixed"; then
        printf '%s' "$suffixed"
        return
      fi
      ;;
    planar3_0|iso3_0|planar4_0|iso4_0|tbq3_0|tbq4_0)
      base="${requested%_0}"
      if help_has_word "$base"; then
        printf '%s' "$base"
        return
      fi
      ;;
  esac

  printf '%s' "$requested"
}

resolve_mtp_spec_type() {
  local requested="${MTP_SPEC_TYPE:-auto}"

  if [ "$requested" != "auto" ]; then
    printf '%s' "$requested"
    return
  fi

  if help_has_word "draft-mtp"; then
    printf '%s' "draft-mtp"
    return
  fi
  if help_has_word "mtp"; then
    printf '%s' "mtp"
    return
  fi

  echo "ERROR: MODEL_NAME='$MODEL_NAME' requires MTP, but llama-server does not advertise an MTP --spec-type." >&2
  echo "Build an MTP-capable llama.cpp/RotorQuant image or choose a non-MTP MODEL_NAME." >&2
  exit 1
}

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
KV_CACHE_REQUESTED="$KV_CACHE"
KV_CACHE="$(normalize_kv_cache_type "$KV_CACHE_REQUESTED")"
IS_MTP_MODEL=false
MTP_SPEC_TYPE_RESOLVED=""

if [[ "$MODEL_NAME" == *-mtp ]]; then
  IS_MTP_MODEL=true
  MTP_SPEC_TYPE_RESOLVED="$(resolve_mtp_spec_type)"
fi

if $IS_MTP_MODEL; then
  PARALLEL="${N_PARALLEL:-1}"
  UBATCH="${UBATCH_SIZE:-32}"
  NO_WARMUP="${NO_WARMUP:-1}"
  if [ "$PARALLEL" != "1" ]; then
    if is_truthy "${PREVIEW:-}" && is_truthy "${MTP_MULTISLOT:-}"; then
      echo "WARN: preview MTP multislot enabled with N_PARALLEL='$PARALLEL'." >&2
      echo "WARN: this path is experimental; keep production on N_PARALLEL=1 unless benchmarks pass." >&2
    else
      echo "ERROR: MTP profiles currently require N_PARALLEL=1; got N_PARALLEL='$PARALLEL'." >&2
      echo "Set PREVIEW=1 and MTP_MULTISLOT=1 only for explicit multislot MTP experiments." >&2
      exit 1
    fi
  fi
else
  PARALLEL="${N_PARALLEL:-2}"
  UBATCH="${UBATCH_SIZE:-}"
  NO_WARMUP="${NO_WARMUP:-}"
fi

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
  --parallel "$PARALLEL"
  --cache-type-k "$KV_CACHE"
  --cache-type-v "$KV_CACHE"
  --cache-ram "${CACHE_RAM:-8192}"
  --flash-attn on
  # Always expose Prometheus /metrics. Endpoint is no-cost when not
  # scraped; required for ServiceMonitor / OpenMetrics integration.
  --metrics
  --host 0.0.0.0
  --port "$PORT"
)

if [ -n "$UBATCH" ]; then
  CMD+=(--ubatch-size "$UBATCH")
fi

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

# Enable MTP speculative decoding for MTP model entries. llama.cpp renamed this
# mode from "mtp" to "draft-mtp"; resolve_mtp_spec_type handles both builds.
if $IS_MTP_MODEL; then
  CMD+=(
    --spec-type "$MTP_SPEC_TYPE_RESOLVED"
    --spec-draft-n-max "$MTP_DRAFT_N_MAX"
  )

  if help_has_word "spec-draft-p-min"; then
    CMD+=(--spec-draft-p-min "$MTP_DRAFT_P_MIN")
  elif [ -n "${MTP_DRAFT_P_MIN:-}" ]; then
    echo "WARN: llama-server does not advertise --spec-draft-p-min; ignoring MTP_DRAFT_P_MIN." >&2
  fi

  if is_truthy "$NO_WARMUP"; then
    CMD+=(--no-warmup)
  fi

  if is_truthy "${MTP_MLOCK:-}"; then
    CMD+=(--mlock)
  fi
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
if [ "$KV_CACHE" != "$KV_CACHE_REQUESTED" ]; then
  echo "║  KV Cache: $KV_CACHE / $KV_CACHE (RotorQuant; requested $KV_CACHE_REQUESTED)"
else
  echo "║  KV Cache: $KV_CACHE / $KV_CACHE (RotorQuant)"
fi
if $IS_MTP_MODEL; then
  echo "║  MTP:      $MTP_SPEC_TYPE_RESOLVED, draft-n-max=$MTP_DRAFT_N_MAX, draft-p-min=$MTP_DRAFT_P_MIN"
fi
echo "║  Context:  $CTX tokens"
echo "║  Parallel: $PARALLEL slots"
if [ -n "$UBATCH" ]; then
  echo "║  UBatch:   $UBATCH tokens"
fi
echo "║  Port:     $PORT"
echo "║  GPU:      $NGL layers offloaded"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Starting: ${CMD[*]}"
echo ""

# ── Launch server (exec replaces shell for proper signal handling) ──────────
exec "${CMD[@]}"
