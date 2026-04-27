#!/bin/bash
set -euo pipefail

# ============================================================================
# RotorQuant llama.cpp Server — Entrypoint
#
# Env vars:
#   MODEL_NAME            (required)  Target model key (see MODELS below)
#   KV_CACHE_TYPE         (optional)  Target KV cache quantization (default: planar4)
#   CTX_SIZE              (optional)  Target context window (default: per-model)
#   PORT                  (optional)  Server port (default: 8080)
#   GPU_LAYERS            (optional)  Layers to offload to GPU (default: 99 = all)
#   N_PARALLEL            (optional)  Concurrent request slots (default: 2; speculative forces 1)
#   CACHE_RAM             (optional)  Prompt cache size in MiB, system RAM (default: 8192)
#   HF_TOKEN              (optional)  HuggingFace token for gated models
#   EXTRA_ARGS            (optional)  Additional llama-server flags
#
# Speculative-decoding env vars (Sprint 004):
#   SPECULATIVE_MODE      (optional)  target-only | autoregressive | dflash (default: target-only)
#   DRAFT_MODEL_NAME      (required if SPECULATIVE_MODE != target-only) Draft model key
#   DRAFT_KV_CACHE_TYPE   (optional)  Draft KV cache type (default: KV_CACHE_TYPE)
#   DRAFT_N_MAX           (optional)  Max draft tokens per verify round (default: 16)
#   EXPERIMENTAL          (optional)  Required (=1) to enable qwen3.6-35b-dflash
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

  # ── DFlash draft GGUFs (Sprint 004) ──────────────────────────────────
  # Pinned 2026-04-27. Format mismatch known: community drafts emit non-canonical
  # tensor names — see SPRINT-004.md Phase 3 / BENCHMARK-REPORT.md §10. Pin so
  # the hf cache is deterministic; smoke runs once source-converted drafts land.
  [qwen3.6-27b-dflash]="spiritbuun/Qwen3.6-27B-DFlash-GGUF|dflash-draft-3.6-q4_k_m.gguf|131072|"
  [qwen3.6-35b-dflash]="lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test|Qwen3.6-35B-A3B-DFlash-q8_0.gguf|65536|"
)

# Repo SHAs pinned at Phase-4 entry (2026-04-27). Informational — `hf download`
# resolves to whatever the repo's main currently points at; bumping this dict
# is the canonical record of what we last verified.
declare -A MODELS_HASH=(
  [qwen3.6-27b-dflash]=5e4442a299deb9282b3dfe179de6e8330b19d9de
  [qwen3.6-35b-dflash]=3813f31a9fa837b79dce98e6ec49ddeaa4082772
)

# ── Helpers ─────────────────────────────────────────────────────────────────

# Resolve a registered model. Downloads if missing. Echoes the absolute path
# on stdout; status/progress goes to stderr so callers can use $(...) directly.
download_model_if_missing() {
  local key="$1"
  if [[ -z "${MODELS[$key]+x}" ]]; then
    echo "ERROR: Unknown model key: $key" >&2
    return 1
  fi
  local repo file ctx extra
  IFS='|' read -r repo file ctx extra <<< "${MODELS[$key]}"
  local path="/models/${file}"
  if [ -f "$path" ]; then
    echo "Model present: $path ($(du -h "$path" | cut -f1))" >&2
    echo "$path"
    return 0
  fi
  {
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  Downloading: $file"
    echo "║  From:        $repo"
    [ -n "${MODELS_HASH[$key]:-}" ] && echo "║  Pinned SHA:  ${MODELS_HASH[$key]}"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
  } >&2
  local args=(download "$repo" "$file" --local-dir /models)
  [ -n "${HF_TOKEN:-}" ] && args+=(--token "$HF_TOKEN")
  hf "${args[@]}" >&2
  if [ ! -f "$path" ]; then
    echo "ERROR: Download completed but $path not found" >&2
    ls -la /models/ >&2
    return 1
  fi
  echo "Download complete: $(du -h "$path" | cut -f1)" >&2
  echo "$path"
}

# ── Parse env vars ──────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:?ERROR: MODEL_NAME is required. Options: ${!MODELS[*]}}"
KV_CACHE="${KV_CACHE_TYPE:-planar4}"
PORT="${PORT:-8080}"
NGL="${GPU_LAYERS:-99}"

SPECULATIVE_MODE="${SPECULATIVE_MODE:-target-only}"
DRAFT_MODEL_NAME="${DRAFT_MODEL_NAME:-}"
DRAFT_KV_CACHE_TYPE="${DRAFT_KV_CACHE_TYPE:-$KV_CACHE}"
DRAFT_N_MAX="${DRAFT_N_MAX:-16}"

case "$SPECULATIVE_MODE" in
  target-only|autoregressive|dflash) ;;
  *)
    echo "ERROR: invalid SPECULATIVE_MODE='$SPECULATIVE_MODE'"
    echo "       must be one of: target-only, autoregressive, dflash"
    exit 1
    ;;
esac

# ── Validate target model name ──────────────────────────────────────────────
if [[ -z "${MODELS[$MODEL_NAME]+x}" ]]; then
  echo "ERROR: Unknown MODEL_NAME='$MODEL_NAME'"
  echo "Available models:"
  for key in "${!MODELS[@]}"; do
    IFS='|' read -r repo file ctx extra <<< "${MODELS[$key]}"
    echo "  $key  →  $file ($repo)"
  done
  exit 1
fi

# ── Speculative validation + experimental gate ──────────────────────────────
if [ "$SPECULATIVE_MODE" != "target-only" ]; then
  if [ -z "$DRAFT_MODEL_NAME" ]; then
    echo "ERROR: SPECULATIVE_MODE=$SPECULATIVE_MODE requires DRAFT_MODEL_NAME"
    exit 1
  fi
  if [[ -z "${MODELS[$DRAFT_MODEL_NAME]+x}" ]]; then
    echo "ERROR: Unknown DRAFT_MODEL_NAME='$DRAFT_MODEL_NAME'"
    exit 1
  fi
  # qwen3.6-35b MoE + DFlash is the experimental tier (per Sprint 004 spec).
  if [[ "$MODEL_NAME" == qwen3.6-35b* ]] && [ "$SPECULATIVE_MODE" = "dflash" ]; then
    if [ "${EXPERIMENTAL:-0}" != "1" ]; then
      echo "ERROR: $MODEL_NAME + DFlash requires EXPERIMENTAL=1"
      echo "       MoE speedup is not gated; expect 0.6–1.3× decode."
      exit 1
    fi
    echo "[experimental] $MODEL_NAME + DFlash enabled by EXPERIMENTAL=1"
  fi
fi

# ── Resolve target + (optional) draft ───────────────────────────────────────
IFS='|' read -r HF_REPO FILENAME DEFAULT_CTX EXTRA_FLAGS <<< "${MODELS[$MODEL_NAME]}"
CTX="${CTX_SIZE:-$DEFAULT_CTX}"
MODEL_PATH=$(download_model_if_missing "$MODEL_NAME")

DRAFT_PATH=""
if [ "$SPECULATIVE_MODE" != "target-only" ]; then
  DRAFT_PATH=$(download_model_if_missing "$DRAFT_MODEL_NAME")
fi

# Speculative decoding requires single-slot serving. Phase 4 doc: N_PARALLEL=1.
if [ "$SPECULATIVE_MODE" != "target-only" ]; then
  N_PARALLEL_EFFECTIVE=1
else
  N_PARALLEL_EFFECTIVE="${N_PARALLEL:-2}"
fi

# ── Build server command ────────────────────────────────────────────────────
CMD=(
  /app/bin/llama-server
  --model "$MODEL_PATH"
  --n-gpu-layers "$NGL"
  --ctx-size "$CTX"
  --parallel "$N_PARALLEL_EFFECTIVE"
  --cache-type-k "$KV_CACHE"
  --cache-type-v "$KV_CACHE"
  --cache-ram "${CACHE_RAM:-8192}"
  --flash-attn on
  --host 0.0.0.0
  --port "$PORT"
)

# Speculative arguments — autoregressive and dflash share draft-model plumbing;
# --dflash flips the draft graph + verify path.
if [ "$SPECULATIVE_MODE" != "target-only" ]; then
  CMD+=(
    --model-draft "$DRAFT_PATH"
    --draft-max "$DRAFT_N_MAX"
    --cache-type-k-draft "$DRAFT_KV_CACHE_TYPE"
    --cache-type-v-draft "$DRAFT_KV_CACHE_TYPE"
  )
  if [ "$SPECULATIVE_MODE" = "dflash" ]; then
    CMD+=(--dflash)
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
echo "║  Mode:     $SPECULATIVE_MODE"
echo "║  Target:   $MODEL_NAME"
echo "║  File:     $FILENAME"
echo "║  KV Cache: $KV_CACHE / $KV_CACHE (RotorQuant)"
echo "║  Context:  $CTX tokens"
if [ "$SPECULATIVE_MODE" != "target-only" ]; then
  echo "║  Draft:    $DRAFT_MODEL_NAME"
  echo "║  Draft KV: $DRAFT_KV_CACHE_TYPE"
  echo "║  Draft N:  $DRAFT_N_MAX"
fi
echo "║  Parallel: $N_PARALLEL_EFFECTIVE slots"
echo "║  Port:     $PORT"
echo "║  GPU:      $NGL layers offloaded"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Starting: ${CMD[*]}"
echo ""

# ── Launch server (exec replaces shell for proper signal handling) ──────────
exec "${CMD[@]}"
