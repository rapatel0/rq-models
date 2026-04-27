#!/bin/bash
# Convert z-lab DFlash draft safetensors → canonical GGUFs using the rebased +
# cherry-picked rotorquant fork's `convert_hf_to_gguf.py`. Resolves F-001 from
# SPRINT-004-FOLLOWUPS.md: the community-published GGUFs miss tensors expected
# by PR #22105's schema, so we run the canonical convert path locally.
#
# Output GGUFs land in the `llm-models` named docker volume so the rotorquant
# compose profiles (qwen36-27b-dflash, qwen36-dflash) pick them up unchanged.
#
# Idempotent: skips repos already present, conversions already on disk, and the
# volume copy if the file inside the volume matches.
set -euo pipefail

ROTORQUANT_REPO="${ROTORQUANT_REPO:-/home/ravi/repos/llama-cpp-turboquant}"
WORK_DIR="${WORK_DIR:-/tmp/dflash-convert}"
DOCKER_VOLUME="${DOCKER_VOLUME:-llm-models}"
OUTTYPE="${OUTTYPE:-bf16}"

# Pairs: <draft-repo>|<target-repo>|<output-filename>
# Source dtype is bf16; staying in bf16 keeps the convert lossless and the
# draft file ~2x smaller than f32. Quantize downstream with llama-quantize.
PAIRS=(
  "z-lab/Qwen3.6-27B-DFlash|Qwen/Qwen3.6-27B|Qwen3.6-27B-DFlash-${OUTTYPE}.gguf"
  "z-lab/Qwen3.6-35B-A3B-DFlash|Qwen/Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-DFlash-${OUTTYPE}.gguf"
)

# Tokenizer-only files we pull from the target repo (no .safetensors weights).
TARGET_INCLUDES=(
  config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  special_tokens_map.json
  chat_template.jinja
  generation_config.json
)

if [ ! -d "$ROTORQUANT_REPO" ]; then
  echo "ERROR: ROTORQUANT_REPO=$ROTORQUANT_REPO not found"; exit 1
fi
if [ ! -f "$ROTORQUANT_REPO/convert_hf_to_gguf.py" ]; then
  echo "ERROR: convert_hf_to_gguf.py missing in $ROTORQUANT_REPO"; exit 1
fi

mkdir -p "$WORK_DIR"
echo "Working in $WORK_DIR (override with WORK_DIR=...)"
echo "Target docker volume: $DOCKER_VOLUME"
echo ""

# Verify the volume is local + not in use by a running container that holds /models open.
if ! docker volume inspect "$DOCKER_VOLUME" >/dev/null 2>&1; then
  echo "Creating docker volume $DOCKER_VOLUME"
  docker volume create "$DOCKER_VOLUME" >/dev/null
fi

for pair in "${PAIRS[@]}"; do
  IFS='|' read -r draft_repo target_repo out_name <<< "$pair"
  draft_dir="$WORK_DIR/$(basename "$draft_repo")"
  target_dir="$WORK_DIR/$(basename "$target_repo")-tokenizer"
  out_path="$WORK_DIR/$out_name"

  echo "════════════════════════════════════════════════════════════════"
  echo " $draft_repo  →  $out_name"
  echo "════════════════════════════════════════════════════════════════"

  # 1. Download draft safetensors.
  if [ ! -f "$draft_dir/model.safetensors" ]; then
    echo "[1/4] Downloading draft safetensors from $draft_repo ..."
    hf_args=(download "$draft_repo" --local-dir "$draft_dir")
    [ -n "${HF_TOKEN:-}" ] && hf_args+=(--token "$HF_TOKEN")
    hf "${hf_args[@]}"
  else
    echo "[1/4] Draft safetensors present: $draft_dir/model.safetensors"
  fi

  # 2. Download target tokenizer/config (no weights).
  if [ ! -f "$target_dir/tokenizer.json" ]; then
    echo "[2/4] Downloading target tokenizer files from $target_repo ..."
    hf_args=(download "$target_repo" --local-dir "$target_dir")
    for f in "${TARGET_INCLUDES[@]}"; do hf_args+=(--include "$f"); done
    [ -n "${HF_TOKEN:-}" ] && hf_args+=(--token "$HF_TOKEN")
    hf "${hf_args[@]}"
  else
    echo "[2/4] Target tokenizer present: $target_dir"
  fi

  # 3. Convert. Stream into a per-pair log; tail only the result lines so the
  # caller's terminal doesn't get flooded by per-tensor INFO lines and the
  # python process doesn't get SIGPIPE-killed by a downstream `head`.
  log_path="$WORK_DIR/convert-$(basename "$draft_repo").log"
  if [ ! -f "$out_path" ]; then
    echo "[3/4] Converting → $out_path (outtype=$OUTTYPE), log=$log_path ..."
    PYTHONPATH="$ROTORQUANT_REPO/gguf-py:${PYTHONPATH:-}" \
      python3 "$ROTORQUANT_REPO/convert_hf_to_gguf.py" \
        --target-model-dir "$target_dir" \
        --outfile "$out_path" \
        --outtype "$OUTTYPE" \
        "$draft_dir" > "$log_path" 2>&1 || {
          echo "convert failed; last 20 lines of log:"
          tail -20 "$log_path"
          exit 1
        }
    echo "[3/4] Converted: $(stat -c '%s' "$out_path") bytes"
  else
    echo "[3/4] GGUF already converted: $out_path"
  fi

  # 4. Copy into the named docker volume.
  echo "[4/4] Publishing to volume $DOCKER_VOLUME:/models/$out_name ..."
  docker run --rm \
    -v "$DOCKER_VOLUME:/models" \
    -v "$out_path:/in/$out_name:ro" \
    alpine:3.20 sh -c "
      if cmp -s /in/$out_name /models/$out_name 2>/dev/null; then
        echo 'Volume already has identical file — skipping copy'
      else
        cp /in/$out_name /models/$out_name
        echo 'Copied: '\$(stat -c '%s' /models/$out_name)' bytes'
      fi
    "
  echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo " All DFlash drafts converted + published to $DOCKER_VOLUME"
echo " Bump entrypoint MODELS registry to point at:"
for pair in "${PAIRS[@]}"; do
  IFS='|' read -r _ _ out_name <<< "$pair"
  echo "   /models/$out_name"
done
echo "════════════════════════════════════════════════════════════════"
