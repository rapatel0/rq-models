# Sprint 005-dflash Codex Execution Log

Date: 2026-04-27 (America/Chicago)

## Phase status

### Phase 0 — DONE (pre-existing)

Result:
- Confirmed branch/state prerequisites matched sprint inputs.

Artifacts:
- None newly produced in this phase.

### Phase 0.5 — DONE (pre-existing)

Result:
- Confirmed existing PASS result (27B correctness probe, 937/937 shared-prefix byte match).

Artifacts:
- `docs/sprints/SPRINT-005-27b-correctness-probe.md` (pre-existing)

### Phase 1 — PARTIAL

Result:
- Implemented orchestration and harness changes:
  - `Makefile`: added `bench-dflash-all` with `PROFILE={qwen,qwen36}` sequencing.
  - `scripts/bench_speculative.py`: profile-scoped outputs, acceptance capture, retry/health recovery, request timeout controls.
- Collected complete `qwen` target-only leg (`5 prompts x 3 trials`).
- Could not complete full three-leg x two-profile canonical sweep due speculative runtime instability on this host.

Artifacts:
- `docs/sprints/SPRINT-005-L4-results-qwen.json` (partial: target-only leg present)
- `docs/sprints/SPRINT-005-L4-results-qwen36.json` (blocked placeholder)
- `docs/sprints/SPRINT-005-L4-summary.md` (partial summary)

What is left:
- Full L4 runs for `qwen` and `qwen36` with all three legs.

Exact commands:
```bash
nvidia-smi
make bench-dflash-all PROFILE=qwen
make bench-dflash-all PROFILE=qwen36
```

Why not completed:
- In speculative modes, first completion typically succeeds, then subsequent requests frequently reset/hang.

### Phase 2 — PARTIAL

Result:
- Added `scripts/sweep_dflash.py` (resumable sweep driver for target-quant, draft-KV, draft-max, N_PARALLEL).
- Added retry/health/timeout hardening in sweep runner.
- Completed one level: `target_weight_quant/q4_k_xl` (3 trials).

Artifacts:
- `docs/sprints/SPRINT-005-experiments.json` (partial)

What is left:
- Remaining target quant levels (`q5_k_m`, `q8_0`) and all other sweep knobs.

Exact commands:
```bash
# prerequisite models into llm-models volume:
# /models/Qwen3.6-27B-UD-Q5_K_M.gguf
# /models/Qwen3.6-27B-UD-Q8_0.gguf

nvidia-smi
python3 scripts/sweep_dflash.py --only-sweep target_weight_quant --rerun
python3 scripts/sweep_dflash.py --only-sweep draft_kv_cache_type
python3 scripts/sweep_dflash.py --only-sweep draft_max
python3 scripts/sweep_dflash.py --only-sweep n_parallel
```

Why not completed:
- Missing Q5/Q8 target GGUFs and same speculative runtime instability during repeated requests.

### Phase 3 — PARTIAL

Result:
- Fork-side code landed locally:
  - `LLAMA_SPEC_FORCE_REJECT_AT` parsed in `common/speculative.cpp` and threaded into verification sampling.
  - Added forced-reject checkpoint harness `tests/test-checkpoint-hybrid-state.cpp` and CTest registrations.
  - Updated verification call site in `tools/server/server-context.cpp`.
- Local fork commit created:
  - `afec36229f12253a60497099c5933e708da7e450`
- Turbo-side updates applied:
  - `tests/test_speculative.py` -> `@pytest.mark.xfail(strict=True)`
  - `docker/Dockerfile` pin updated to `afec36229f12253a60497099c5933e708da7e450`

Artifacts:
- fork changes in `/home/ravi/repos/llama-cpp-turboquant`
- turbo file updates in `/home/ravi/repos/turbo`

What is left:
- Push fork commit to `rapatel0/llama-cpp-turboquant` branch `feature/sprint-004-rebase-dflash`.
- Rebuild rotorquant image from pushed SHA.
- Run pytest against live server with `LLAMA_SPEC_FORCE_REJECT_AT` env and confirm xpass-strict behavior.

Exact commands:
```bash
# in /home/ravi/repos/llama-cpp-turboquant
git push origin feature/sprint-004-rebase-dflash

# in /home/ravi/repos/turbo
make build
LLAMA_SPEC_FORCE_REJECT_AT=8 make run-qwen-bg
python -m pytest tests/test_speculative.py -k force_reject_preserves_output -vv
make stop
```

Why not completed:
- No push permission to `rapatel0/llama-cpp-turboquant` from this environment (`SSH publickey denied`; HTTPS token user lacks repo permission).
- With unreachable SHA, Docker build fails at checkout.

### Phase 4 — PARTIAL

Result:
- Added blocked/partial reporting artifacts:
  - `docs/sprints/SPRINT-005-FOLLOWUPS-dflash.md`
  - partial `docs/sprints/SPRINT-005-L4-summary.md`
- Did not fully fill BENCHMARK-REPORT §10 TBD tables (missing canonical L4 + complete experiments).
- Did not fully update README headline ratio text (headline not yet measured to completion).

Artifacts:
- `docs/sprints/SPRINT-005-FOLLOWUPS-dflash.md`
- `docs/sprints/SPRINT-005-L4-summary.md`

What is left:
- Fill BENCHMARK-REPORT §10 tables once Phase 1/2 complete.
- Update README speculative headline with finalized Sprint 005 numbers.

### Phase 5 — PARTIAL

Result:
- Produced this execution log with DONE/PARTIAL/BLOCKED detail and rerun commands.
- Sprint status not flipped to complete due open blockers.

What is left:
- Update `docs/sprints/SPRINT-005-dflash.md` status/checklists after blocked items are resolved.

## Headline numbers

- Phase 1 hard-gate metric (median DFlash× for `qwen` vs target-only): **not yet available** (canonical sweep incomplete).
- Gate result (≥1.3×): **not evaluable**.

Available partial point:
- `qwen` target-only prompt-median median tok/s: `69.41`
- `q4_k_xl` quicksort sweep point in DFlash mode: `73.48 tok/s`, acceptance `100%`, quicksort ratio vs `qwen` target-only quicksort median: `1.06x`

## Deviations from plan

1. Added retry/health/timeout resilience to benchmark/sweep scripts.
- Why: runtime exhibited frequent connection resets/restarts after first speculative completion.

2. Fork test harness currently skips non-hybrid fixture model in generic CTest.
- Why: default tiny fixture with planar/iso cache args asserts during memory fitting; harness is intended for hybrid Qwen3.6 model runs.

3. Docker pin bumped before upstream push completed.
- Why: phase requirement is to pin new fork SHA; however this leaves local `make build` blocked until the commit is reachable on the remote branch.
