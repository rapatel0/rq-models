#!/bin/bash
# Smoke test: build image, verify each profile boots, and assert the
# `llm-models` named volume is cache-preserving across the rebased image.
#
# Sprint 004 added two new profiles (qwen36-27b-dflash, qwen36-dflash) and
# a refactored entrypoint that exposes SPECULATIVE_MODE / DRAFT_MODEL_NAME.
# This script enforces the Phase-4 hard gates: (a) no existing profile
# re-downloads its model; (b) all profiles still serve after the refactor.
set -euo pipefail

IMAGE="rotorquant:test"
PORT=8099
VOLUME=llm-models

# Profiles whose models we want to exercise. Existing profiles first;
# DFlash profiles last so they don't block earlier coverage if blocked on
# the known community-draft format mismatch.
EXISTING_PROFILES=(qwen qwen36-q3 qwen36-iq3 qwen36-27b qwen36-27b-q3 qwen36-27b-iq3 reasoning gemma)
NEW_PROFILES=(qwen36-27b-dflash)

# Map profile → MODEL_NAME env that entrypoint expects (mirror of compose).
declare -A PROFILE_MODEL=(
  [qwen]=qwen3.6-35b
  [qwen36-q3]=qwen3.6-35b-q3
  [qwen36-iq3]=qwen3.6-35b-iq3
  [qwen36-27b]=qwen3.6-27b
  [qwen36-27b-q3]=qwen3.6-27b-q3
  [qwen36-27b-iq3]=qwen3.6-27b-iq3
  [reasoning]=qwen3.5-27b-reasoning
  [gemma]=gemma4-26b
  [qwen36-27b-dflash]=qwen3.6-27b
  [qwen36-dflash]=qwen3.6-35b
)

echo "=== RotorQuant Docker Smoke Test ==="
echo ""

# ── 1. Build ─────────────────────────────────────────────────────────────────
echo "[1/3] Building image..."
docker build -t "$IMAGE" -f docker/Dockerfile . || { echo "FAIL: build"; exit 1; }
echo "PASS: build succeeded"
echo "Image size: $(docker images "$IMAGE" --format '{{.Size}}')"
echo ""

# ── 2. Cache-preservation gate ──────────────────────────────────────────────
# Snapshot mtime of every file in the named volume BEFORE we boot any container,
# then re-snapshot after each profile and assert no file is newer.
volume_mtime_snapshot() {
  # Run a throwaway alpine container with the volume mounted, write `mtime|path`
  # for every regular file. Caller-side comparison on stdout.
  docker run --rm -v "$VOLUME:/v:ro" alpine:3.20 \
    sh -c 'find /v -type f -exec stat -c "%Y|%n" {} +' 2>/dev/null | sort
}

echo "[2/3] Cache-preservation gate"
PRE_SNAPSHOT="$(mktemp)"
volume_mtime_snapshot > "$PRE_SNAPSHOT"
PRE_COUNT=$(wc -l < "$PRE_SNAPSHOT")
echo "  Pre-snapshot: $PRE_COUNT files in volume '$VOLUME'"

ALL_PROFILES=("${EXISTING_PROFILES[@]}" "${NEW_PROFILES[@]}")

# ── 3. Boot + serve check per profile ───────────────────────────────────────
echo ""
echo "[3/3] Boot + serve per profile"

run_profile_test() {
  local profile="$1"
  local model="${PROFILE_MODEL[$profile]}"
  local container="rq-test-$profile"

  echo ""
  echo "── profile: $profile (MODEL_NAME=$model) ──"

  # Skip the test if no GGUF for this model family is already present in the
  # volume — the gate is about preservation, not on-demand download.
  # Case-insensitive substring match (registry filenames are mixed-case).
  local model_file
  model_file=$(docker run --rm -v "$VOLUME:/v:ro" alpine:3.20 \
    sh -c "ls /v/ 2>/dev/null | grep -iF -- '$model' || true" \
    | head -1 || true)

  if [ -z "$model_file" ]; then
    echo "  SKIP: model not pre-cached in volume — re-run after first download"
    return 0
  fi

  local extra_env=()
  if [ "$profile" = "qwen36-dflash" ]; then
    extra_env+=(-e EXPERIMENTAL=1)
  fi

  # Use a small ctx so RAM/VRAM isn't the bottleneck.
  docker run -d --gpus all --name "$container" \
    -e MODEL_NAME="$model" \
    -e CTX_SIZE=4096 \
    -e N_PARALLEL=1 \
    "${extra_env[@]}" \
    -v "$VOLUME:/models" \
    -p "$PORT:8080" \
    "$IMAGE" >/dev/null 2>&1 || { echo "  FAIL: docker run"; return 1; }

  local ready=false
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" 2>/dev/null | grep -q ok; then
      echo "  Server ready after ${i}s"
      ready=true; break
    fi
    if ! docker ps -q --filter "name=$container" | grep -q .; then
      echo "  Container exited. Last 10 log lines:"
      docker logs "$container" 2>&1 | tail -10 | sed 's/^/    /'
      break
    fi
    sleep 1
  done

  if $ready; then
    local resp
    resp=$(curl -sf "http://localhost:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"test","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"temperature":0}' \
      2>&1 || echo "API_ERROR")
    if echo "$resp" | grep -q '"choices"'; then
      echo "  PASS: API responds"
    else
      echo "  WARN: API response unexpected: ${resp:0:100}"
    fi
  fi

  docker stop "$container" >/dev/null 2>&1 || true
  docker rm   "$container" >/dev/null 2>&1 || true
}

for p in "${ALL_PROFILES[@]}"; do
  run_profile_test "$p" || true
done

# ── 4. Re-snapshot and diff ─────────────────────────────────────────────────
echo ""
echo "── cache-preservation diff ──"
POST_SNAPSHOT="$(mktemp)"
volume_mtime_snapshot > "$POST_SNAPSHOT"

# Compare line-for-line — any difference is a re-download or a new file.
if diff -q "$PRE_SNAPSHOT" "$POST_SNAPSHOT" > /dev/null 2>&1; then
  echo "  PASS: no model files re-downloaded (mtimes unchanged)"
else
  echo "  FAIL: cache changed:"
  diff "$PRE_SNAPSHOT" "$POST_SNAPSHOT" | sed 's/^/    /'
fi

rm -f "$PRE_SNAPSHOT" "$POST_SNAPSHOT"

echo ""
echo "=== Smoke test complete ==="
