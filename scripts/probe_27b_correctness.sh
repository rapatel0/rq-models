#!/bin/bash
# Sprint 005 Phase 0.5: 27B DFlash correctness probe.
#
# Decisively answers: is the dense 27B-DFlash output token-equivalent to
# target-only at greedy sampling? If yes, the 37% acceptance observed in
# Sprint 004 smoke is a perf observation; if no, the verify path has a
# bug specific to the dense 27B (or its hybrid layer ratio).
#
# Probe procedure:
#   1. Bring up qwen36-27b (target-only).
#   2. POST /v1/chat/completions, save token sequence.
#   3. make stop.
#   4. Bring up qwen36-27b-dflash (PREVIEW=1).
#   5. POST same prompt + sampling params, save token sequence.
#   6. make stop.
#   7. Diff both JSON token arrays.
#
# Pass: target_tokens == dflash_tokens (every position).
# Fail: any divergence — capture position + expected vs actual.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ravi/repos/turbo}"
PORT="${PORT:-8080}"
PROMPT="${PROMPT:-Write a quicksort algorithm in Python. Write code only.}"
TOKENS="${TOKENS:-256}"
SEED="${SEED:-42}"
TEMP="${TEMP:-0}"
TOP_K="${TOP_K:-1}"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/docs/sprints}"

target_tokens_path="$RESULT_DIR/SPRINT-005-27b-target-only.tokens.json"
dflash_tokens_path="$RESULT_DIR/SPRINT-005-27b-dflash.tokens.json"
report_path="$RESULT_DIR/SPRINT-005-27b-correctness-probe.md"

cd "$REPO_ROOT"

# Single completion against the currently-running server. Saves the response
# as JSON. Caller is responsible for ensuring the right profile is up.
post_one() {
  local outpath="$1"
  curl -sf "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(cat <<JSON
{
  "model": "rotorquant",
  "messages": [{"role": "user", "content": "$(echo "$PROMPT" | sed 's/"/\\"/g')"}],
  "max_tokens": $TOKENS,
  "temperature": $TEMP,
  "top_k": $TOP_K,
  "seed": $SEED,
  "stream": false
}
JSON
)" > "$outpath" || return 1
  if [ ! -s "$outpath" ]; then
    echo "ERROR: empty response written to $outpath" >&2
    return 1
  fi
  return 0
}

wait_for_health() {
  for i in $(seq 1 240); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

bring_up() {
  local profile="$1"
  shift
  echo "── bringing up $profile ──"
  "$@" docker compose --profile "$profile" up -d
  if ! wait_for_health; then
    echo "ERROR: server did not respond on /health within 240s for profile $profile" >&2
    docker compose --profile "$profile" logs --tail 30
    return 1
  fi
  echo "  $profile up + healthy"
}

bring_down() {
  local profile="$1"
  echo "── stopping $profile ──"
  "${@:2}" docker compose --profile "$profile" down >/dev/null 2>&1 || true
}

# ── Stage 1: target-only ────────────────────────────────────────────────────
bring_up qwen36-27b
echo "[1/2] capturing target-only completion → $target_tokens_path"
post_one "$target_tokens_path"
bring_down qwen36-27b

# ── Stage 2: target + DFlash ────────────────────────────────────────────────
bring_up qwen36-27b-dflash PREVIEW=1
echo "[2/2] capturing target+DFlash completion → $dflash_tokens_path"
post_one "$dflash_tokens_path"
bring_down qwen36-27b-dflash

# ── Diff ────────────────────────────────────────────────────────────────────
echo "── diffing token sequences ──"
python3 - "$target_tokens_path" "$dflash_tokens_path" "$report_path" <<'PY'
import json, sys

target_path, dflash_path, report_path = sys.argv[1:4]
with open(target_path) as f: t = json.load(f)
with open(dflash_path) as f: d = json.load(f)

t_text = t.get("choices", [{}])[0].get("message", {}).get("content", "")
d_text = d.get("choices", [{}])[0].get("message", {}).get("content", "")

# llama-server doesn't expose token IDs in the OpenAI response shape; the
# completion text is the artefact we have. Compare byte-for-byte.
match = (t_text == d_text)
diverge = -1
if not match:
    for i, (a, b) in enumerate(zip(t_text, d_text)):
        if a != b:
            diverge = i
            break
    if diverge < 0:
        diverge = min(len(t_text), len(d_text))

t_usage = (t.get("usage") or {}).get("completion_tokens")
d_usage = (d.get("usage") or {}).get("completion_tokens")

with open(report_path, "w") as f:
    f.write("# Sprint 005 Phase 0.5: 27B DFlash correctness probe\n\n")
    f.write(f"**Prompt**: {t.get('_prompt', '(see post body in scripts)')}\n")
    f.write(f"**Tokens requested**: per-call max_tokens\n")
    f.write(f"**Sampling**: temp=0, top-k=1, seed=42 (greedy, deterministic)\n\n")
    f.write(f"**Target-only completion length**: {len(t_text)} chars / {t_usage} tokens\n")
    f.write(f"**Target+DFlash completion length**: {len(d_text)} chars / {d_usage} tokens\n\n")
    if match:
        f.write(f"## Result: PASS — completions byte-equal\n\n")
        f.write(f"The dense 27B-DFlash produces output identical to target-only at\n")
        f.write(f"greedy sampling. The 37% acceptance rate observed in Sprint 004\n")
        f.write(f"smoke is therefore a perf observation, not a correctness bug.\n")
        f.write(f"PREVIEW gate stays for \"drafts iterating\", not for \"broken\".\n")
    else:
        f.write(f"## Result: FAIL — completions diverge at position {diverge}\n\n")
        head_t = t_text[max(0, diverge - 40):diverge + 40]
        head_d = d_text[max(0, diverge - 40):diverge + 40]
        f.write(f"```\n")
        f.write(f"  target-only context (±40 chars around divergence):\n")
        f.write(f"    {head_t!r}\n")
        f.write(f"  target+DFlash    context (±40 chars around divergence):\n")
        f.write(f"    {head_d!r}\n")
        f.write(f"```\n\n")
        f.write(f"This is a correctness bug specific to the dense 27B-DFlash\n")
        f.write(f"verify path. PREVIEW gate must be reinforced; F-011 opens.\n")

print(f"\nresult: {'PASS' if match else f'FAIL at char {diverge}'}")
print(f"report: {report_path}")
sys.exit(0 if match else 1)
PY
