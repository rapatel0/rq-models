#!/bin/bash
# Sprint 006-dflash experiment driver.
# Runs the standard 3-leg orchestration (target-only / autoregressive / dflash)
# with experiment-specific env overrides and writes per-experiment artifacts.
#
# Usage:
#   ./scripts/run_sprint006_experiment.sh EN [extra-env-flags...]
#
# Env overrides honored by the entrypoint / fork:
#   DRAFT_N_MAX                  -- block size knob (E2)
#   LLAMA_SPEC_DISABLE_SKIP=1    -- bypass v8 skip flag (E1)
#   LLAMA_SPEC_ADAPTIVE_SKIP_K=N -- skip after N consecutive partials (E4)
#
# E2 is run by setting DRAFT_N_MAX=N on each leg.
# E1, E4 are run by setting LLAMA_SPEC_* env on the speculative legs.
# E3, E5 are pure measurement — no env overrides; data lives in result_timings.
set -euo pipefail

EXP=${1:?"usage: $0 EN [extra env in NAME=VAL form...]"}
shift
EXTRA_ENV=("$@")

PROFILE=${PROFILE:-qwen}
NO_THINK=${NO_THINK:-1}    # default to PR #22105 regime
DRAFT=${DRAFT:-qwen3.6-27b-dflash}

OUT_DIR=/home/ravi/repos/turbo/docs/sprints/SPRINT-006-dflash-experiments/${EXP}
mkdir -p "$OUT_DIR"
JSON=$OUT_DIR/results-${PROFILE}.json
MD=$OUT_DIR/summary-${PROFILE}.md
LOG=$OUT_DIR/run.log

THINK_FLAG=""
[ "$NO_THINK" = "1" ] && THINK_FLAG="--no-think"

cd /home/ravi/repos/turbo

echo "==[$EXP]== profile=$PROFILE no_think=$NO_THINK extra=${EXTRA_ENV[*]:-(none)}" | tee "$LOG"

run_leg () {
  local leg=$1 ; shift
  local server_env=("$@")
  echo "  -- leg=$leg env=${server_env[*]:-(none)} --" | tee -a "$LOG"
  make stop 2>&1 | tail -2 >> "$LOG"
  if [ "$leg" = "target-only" ]; then
    env "${server_env[@]}" make run-${PROFILE}-target-only-bg 2>&1 | tail -2 >> "$LOG"
  elif [ "$leg" = "autoregressive" ]; then
    env "${server_env[@]}" SPECULATIVE_MODE=autoregressive DRAFT_MODEL_NAME=${DRAFT} make run-${PROFILE}-bg 2>&1 | tail -2 >> "$LOG"
  else
    env "${server_env[@]}" make run-${PROFILE}-bg 2>&1 | tail -2 >> "$LOG"
  fi
  python3 scripts/bench_speculative.py --profile $PROFILE --leg $leg $THINK_FLAG \
    --output "$JSON" --md-output "$MD" 2>&1 | tee -a "$LOG"
}

run_leg target-only     "${EXTRA_ENV[@]}"
run_leg autoregressive  "${EXTRA_ENV[@]}"
run_leg dflash          "${EXTRA_ENV[@]}"

make stop 2>&1 | tail -2 >> "$LOG"

python3 scripts/bench_speculative.py --profile $PROFILE --finalize \
  --output "$JSON" --md-output "$MD" 2>&1 | tee -a "$LOG"

echo "==[$EXP]== done. results=$JSON summary=$MD log=$LOG" | tee -a "$LOG"
cat "$MD"
