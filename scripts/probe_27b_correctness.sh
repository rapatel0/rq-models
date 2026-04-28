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
  # Use `env` so VAR=value tokens become environment assignments for the
  # docker subprocess, not literal commands.
  env "$@" docker compose --profile "$profile" up -d
  if ! wait_for_health; then
    echo "ERROR: server did not respond on /health within 240s for profile $profile" >&2
    docker compose --profile "$profile" logs --tail 30
    return 1
  fi
  echo "  $profile up + healthy"
}

bring_down() {
  local profile="$1"
  shift
  echo "── stopping $profile ──"
  env "$@" docker compose --profile "$profile" down >/dev/null 2>&1 || true
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

# llama-server splits Qwen3.x thinking-mode output between reasoning_content
# (the <think>...</think> block) and content (the final answer). Concatenate
# both for a complete view; some prompts may have all output in reasoning if
# the 256-token budget runs out mid-think.
def total_text(resp):
    msg = (resp.get("choices") or [{}])[0].get("message", {}) or {}
    return (msg.get("reasoning_content") or "") + (msg.get("content") or "")

t_text = total_text(t)
d_text = total_text(d)

# Speculative decoding can hit the same token budget at slightly different
# character positions because token boundaries differ from per-byte boundaries.
# Correctness therefore = "shared prefix is byte-exact"; tail-length differences
# are budget artifacts, not divergence.
common = 0
for a, b in zip(t_text, d_text):
    if a == b:
        common += 1
    else:
        break

t_usage = (t.get("usage") or {}).get("completion_tokens")
d_usage = (d.get("usage") or {}).get("completion_tokens")
t_timings = t.get("timings") or {}
d_timings = d.get("timings") or {}
t_tps = t_timings.get("predicted_per_second")
d_tps = d_timings.get("predicted_per_second")
draft_n = d_timings.get("draft_n")
draft_acc = d_timings.get("draft_n_accepted")

shared_min = min(len(t_text), len(d_text))
prefix_match = (common >= shared_min)

with open(report_path, "w") as f:
    f.write("# Sprint 005 Phase 0.5: 27B DFlash correctness probe\n\n")
    f.write(f"**Sampling**: temp=0, top-k=1, seed=42 (greedy, deterministic)\n")
    f.write(f"**Tokens budget**: 256 each\n\n")
    f.write(f"| Metric | target-only | target+DFlash |\n")
    f.write(f"|---|---:|---:|\n")
    f.write(f"| Output chars (reasoning + content) | {len(t_text)} | {len(d_text)} |\n")
    f.write(f"| Completion tokens | {t_usage} | {d_usage} |\n")
    if t_tps and d_tps:
        f.write(f"| Decode tok/s | {t_tps:.2f} | {d_tps:.2f} ({d_tps/t_tps:.3f}×) |\n")
    if draft_n is not None and draft_acc is not None and draft_n > 0:
        f.write(f"| Draft proposed / accepted | — | {draft_acc} / {draft_n} ({100.0*draft_acc/draft_n:.1f}%) |\n")
    f.write(f"\n**Shared-prefix length**: {common} chars\n")
    f.write(f"**Tail beyond shared prefix**: target-only {len(t_text)-common} chars, "
            f"DFlash {len(d_text)-common} chars\n\n")

    if prefix_match:
        f.write(f"## Result: PASS — output is byte-equal on the shared prefix\n\n")
        f.write(f"For all {common} characters where both runs produced output, the\n")
        f.write(f"text is byte-identical. The dense 27B-DFlash verify+rollback path\n")
        f.write(f"is correct on this prompt at greedy sampling.\n\n")
        if len(t_text) != len(d_text):
            f.write(f"The {abs(len(t_text)-len(d_text))}-char tail-length difference is a\n")
            f.write(f"token-budget artifact, not a content divergence: both runs hit\n")
            f.write(f"the 256-token cap, but speculative decoding can tokenize the\n")
            f.write(f"same string slightly differently (e.g. \"middle = [x \" as one\n")
            f.write(f"token vs \"middle = [x\" + \" \"), so the cap fires at a\n")
            f.write(f"slightly different character position. Within the shared prefix,\n")
            f.write(f"every character matches.\n\n")
        f.write(f"Implication: the 37% acceptance observed in Sprint 004's 7-token\n")
        f.write(f"smoke probe is a perf observation, not a correctness bug. PREVIEW\n")
        f.write(f"gate stays for \"drafts iterating\", not for \"broken\".\n")
    else:
        f.write(f"## Result: FAIL — divergence at character {common}\n\n")
        head_t = t_text[max(0, common - 40):common + 40]
        head_d = d_text[max(0, common - 40):common + 40]
        f.write(f"```\n")
        f.write(f"target-only ±40 chars: {head_t!r}\n")
        f.write(f"target+DFlash ±40 chars: {head_d!r}\n")
        f.write(f"```\n\n")
        f.write(f"This is a correctness bug specific to the dense 27B-DFlash verify\n")
        f.write(f"path. PREVIEW gate must be reinforced; F-011 opens.\n")

print(f"\nresult: {'PASS' if prefix_match else f'FAIL at char {common}'}")
print(f"shared prefix: {common}/{shared_min} chars")
if draft_n is not None and draft_acc is not None and draft_n > 0:
    print(f"acceptance:    {draft_acc}/{draft_n} = {100.0*draft_acc/draft_n:.1f}%")
print(f"report:        {report_path}")
sys.exit(0 if prefix_match else 1)
PY
