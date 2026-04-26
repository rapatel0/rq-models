#!/bin/bash
# Sprint 004 Phase 1 smoke test runner
# Runs Phase 0 (fp16 baseline) + Phase 1 (rotorquant_planar3 passthrough)
# and verifies bit-identical output. Hard gate for Phase 1.
#
# Usage:
#   bash scripts/sprint004_phase1_smoke.sh [MODEL] [IMAGE]
#
# Defaults:
#   MODEL = Qwen/Qwen3-4B (fits on 24 GB)
#   IMAGE = rq-vllm:phase1
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B}"
IMAGE="${2:-rq-vllm:phase1}"
PORT="${PORT:-8090}"
HOST="127.0.0.1"
PROMPT='Hello, what is the capital of France?'

cleanup() {
    docker rm -f rq-vllm-smoke 2>/dev/null || true
}
trap cleanup EXIT

run_one() {
    local label="$1"
    local extra_env="$2"
    local outfile="/tmp/sprint004-${label}.json"

    cleanup
    echo "=== ${label}: starting server ==="
    docker run -d --rm --name rq-vllm-smoke --gpus all \
        -p "${PORT}:${PORT}" \
        -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
        -e "MODEL=${MODEL}" \
        -e "PORT=${PORT}" \
        -e "MAX_MODEL_LEN=2048" \
        -e "GPU_MEMORY_UTILIZATION=0.85" \
        ${extra_env} \
        "${IMAGE}" \
        --enforce-eager > /dev/null

    # Wait for /health (up to 5 minutes)
    local deadline=$(( $(date +%s) + 300 ))
    until curl -sf "http://${HOST}:${PORT}/health" > /dev/null 2>&1; do
        if [ "$(date +%s)" -gt "$deadline" ]; then
            echo "ERROR: ${label} server didn't become healthy in 5 min"
            docker logs rq-vllm-smoke | tail -30
            return 1
        fi
        sleep 5
    done

    echo "=== ${label}: probe /v1/models ==="
    curl -sf "http://${HOST}:${PORT}/v1/models" | python3 -m json.tool | head

    echo "=== ${label}: deterministic generation ==="
    local payload
    payload=$(python3 -c "
import json
print(json.dumps({
    'model': '${MODEL}',
    'messages': [{'role':'user','content':'''${PROMPT}'''}],
    'max_tokens': 32,
    'temperature': 0.0,
    'seed': 42,
}))")
    curl -sf -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "${payload}" > "${outfile}"

    python3 -c "
import json
d = json.load(open('${outfile}'))
print('label=${label}')
print('content:', repr(d['choices'][0]['message']['content']))
"

    echo "=== ${label}: server log tail ==="
    docker logs --tail 5 rq-vllm-smoke 2>&1 | head -10
    cleanup
}

run_one "phase0-fp16" ""
run_one "phase1-rotorquant_planar3" "-e ROTORQUANT_MODE=planar3"

echo
echo "=== bit-identicality diff ==="
python3 -c "
import json
a = json.load(open('/tmp/sprint004-phase0-fp16.json'))
b = json.load(open('/tmp/sprint004-phase1-rotorquant_planar3.json'))
ca = a['choices'][0]['message']['content']
cb = b['choices'][0]['message']['content']
print('phase0 content:', repr(ca))
print('phase1 content:', repr(cb))
if ca == cb:
    print('VERDICT: PASS — bit-identical')
else:
    print('VERDICT: FAIL — outputs differ')
    raise SystemExit(1)
"
