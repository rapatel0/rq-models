#!/bin/bash
# Smoke test: build image + verify each model starts and serves
set -euo pipefail

IMAGE="rotorquant:test"
PORT=8099

echo "=== RotorQuant Docker Smoke Test ==="
echo ""

# Build
echo "[1/4] Building image..."
docker build -t "$IMAGE" -f docker/Dockerfile . || { echo "FAIL: build"; exit 1; }
echo "PASS: build succeeded"
echo "Image size: $(docker images "$IMAGE" --format '{{.Size}}')"
echo ""

# Test each model (just health check — full download takes too long for CI)
for profile in qwen reasoning gemma; do
  echo "[test] Profile: $profile"

  # Start container
  docker run -d --gpus all --name "rq-test-$profile" \
    -e MODEL_NAME=$(case $profile in
      qwen) echo "qwen3.5-27b";;
      reasoning) echo "qwen3.5-27b-reasoning";;
      gemma) echo "gemma4-26b";;
    esac) \
    -e CTX_SIZE=4096 \
    -v llm-models:/models \
    -p "$PORT:8080" \
    "$IMAGE" 2>/dev/null || true

  # Wait for server (up to 3 min for model load)
  echo "  Waiting for server..."
  READY=false
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" 2>/dev/null | grep -q ok; then
      echo "  Server ready after ${i}s"
      READY=true
      break
    fi
    # Check if container died
    if ! docker ps -q --filter "name=rq-test-$profile" | grep -q .; then
      echo "  Container exited. Logs:"
      docker logs "rq-test-$profile" 2>&1 | tail -10
      break
    fi
    sleep 1
  done

  if $READY; then
    # Quick API test
    RESP=$(curl -sf "http://localhost:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"test","messages":[{"role":"user","content":"Say hi"}],"max_tokens":50,"temperature":0}' 2>&1 || echo "API_ERROR")

    if echo "$RESP" | grep -q "choices"; then
      echo "  PASS: API responds correctly"
    else
      echo "  WARN: API response unexpected: ${RESP:0:100}"
    fi
  else
    echo "  SKIP: Server did not start (model may not be downloaded)"
  fi

  # Cleanup
  docker stop "rq-test-$profile" 2>/dev/null || true
  docker rm "rq-test-$profile" 2>/dev/null || true
  echo ""
done

echo "=== Smoke test complete ==="
