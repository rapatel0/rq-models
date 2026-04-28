# Sprint 005 Follow-ups (-dflash)

Execution-discovered follow-ups while running Sprint 005-dflash.

---

## F-011: DFlash draft path asserts on second request after prompt-cache miss — RESOLVED 2026-04-27

**Severity**: was Critical (blocks canonical L4 and broad sweep completion); now resolved

**Resolution**: fork commit `40856a1d2c26e79156a697daaf222f482989d7c7`
("F-011: reset DFlash + EAGLE3 cumulative state in begin()") on
`feature/sprint-004-rebase-dflash`, pushed to
`rapatel0/llama-cpp-turboquant`. `docker/Dockerfile`
`ROTORQUANT_COMMIT` bumped accordingly; rotorquant image rebuilt.

Verified 2026-04-27: `make run-qwen36-bg` followed by 3 sequential
greedy `POST /v1/chat/completions` (the prior abort happened on
request 2). All 3 succeeded, server stayed up, 100% draft acceptance
(54/54 each). docker logs has zero `GGML_ASSERT` lines.

**Root cause** (kept for posterity): `dflash_n_past` and
`accumulated_ctx` are cumulative per-request state. The DFlash
`begin()` override was a no-op (`GGML_UNUSED(prompt)`), so when a new
request landed and the slot rebuilt its KV cache + prompt, the stale
position from the previous request was still there. With a 280-token
previous final state and a 24-token new prompt, `n_new == -256` and
the `n_new >= 1` assert fired immediately on the first draft call.

The fix mirrors what the legacy speculative state already does in
`begin()` (line 255): clear per-request scratch in the lifecycle hook.
EAGLE3 had the identical bug shape (also a no-op `begin()`) — fix
applied there defensively even though Sprint 005 doesn't exercise it,
to avoid landing the same fix twice.

**What**: The fork's DFlash draft path asserts and crashes
`llama-server` when a request lands after the prompt cache has been
invalidated (forcing full re-processing). Concretely:

```
/src/common/speculative.cpp:789: GGML_ASSERT(n_new >= 1 && "must have at least 1 new token") failed
common_speculative_state_dflash::draft(...)
common_speculative_draft(...)
```

Repro confirmed on `qwen36` (Qwen3.6-35B-A3B + DFlash MoE) on
2026-04-27: first request completes cleanly (256 tokens, 100%
acceptance, ~104 tok/s). Second request triggers
`forcing full prompt re-processing due to lack of cache data
(likely due to SWA or hybrid/recurrent memory, see PR #13194)`,
which then drives the DFlash draft path into a state where
`n_new == 0` and the assert fires.

**Crucially**: the bug is **NOT model-specific**. Codex first
observed it on `qwen` (Qwen3.6-27B dense + DFlash) and hypothesized
the dense-hybrid layer ratio. A direct repro on `qwen36`
(Qwen3.6-35B-A3B MoE + DFlash) hit the same assert in the same
code path, ruling out the model-architecture hypothesis. Phase 0.5
ran cleanly because it issued **one** completion — the bug only
fires on subsequent requests.

**Why discovered**: Repeatedly reproduced during
`make bench-dflash-all PROFILE=qwen` and confirmed on
`make bench-dflash-leg PROFILE=qwen36 LEG=dflash` with crash logs
in `docker logs rotorquant-qwen36`.

**Suggested sprint**: Immediate (before finishing Sprint 005
measurement gates). This is the next thing to fix on the dflash
track. Likely owner: fork-side speculative draft + cache invalidation
interaction in the DFlash branch of PR #22105's draft graph.

**Investigation pointers**:
- `common/speculative.cpp:789` is the assert site; check what
  produces `n_new` and whether the slot's cache-miss path resets
  any state DFlash relies on.
- The "forcing full prompt re-processing" log comes from
  `tools/server/server.cpp` slot update logic (cache miss
  handling for SWA / hybrid memory per upstream PR #13194). The
  DFlash draft path probably needs to handle the
  `n_new == 0` case gracefully (return early, no draft) rather
  than asserting.
- Reproducer (~30 seconds): `make run-qwen36-bg && curl -s
  localhost:8080/v1/chat/completions ... && curl -s ...same... &&
  docker logs rotorquant-qwen36 | grep GGML_ASSERT`.

**Files**:
- fork: `common/speculative.cpp` (the assert site, the DFlash
  draft state)
- fork: `tools/server/server.cpp` (the cache-miss path that
  hands `n_new == 0` to draft)
- repo: `scripts/bench_speculative.py` (retry hardening landed
  but is wallpaper over the assert; can't recover from a dead
  server)
- repo: `scripts/sweep_dflash.py` (same)

---

## F-012: Target-quant sweep prerequisites missing for Q5/Q8

**Severity**: Important (blocks target-quant matrix completion)

**What**: `Qwen3.6-27B-UD-Q5_K_M.gguf` and `Qwen3.6-27B-UD-Q8_0.gguf` are not present in `llm-models`, so `target_weight_quant` levels beyond `q4_k_xl` cannot boot.

**Why discovered**: `scripts/sweep_dflash.py --only-sweep target_weight_quant` completed `q4_k_xl`, then stalled in health checks while the container restart-looped for missing model artifacts.

**Suggested sprint**: Immediate (Sprint 005 completion)

**Files**:
- `docs/sprints/SPRINT-005-experiments.json`
- `scripts/sweep_dflash.py`

---

## F-013: Fork push permission mismatch for `rapatel0` remote — RESOLVED 2026-04-27

**Severity**: was Critical (blocks Docker pin validation path); now resolved

**What**: Fork-side Phase 3 commit existed locally but push initially
failed (`SSH publickey denied`, HTTPS `403` from `rpsdm0` token).

**Resolution**: Codex's environment had a stale `SSH_AUTH_SOCK`
pointing at a defunct socket; the orchestrator located a live agent
socket at `/tmp/ssh-kMGLQyjwnC/agent.1778809` carrying the
"SDM Personal" key and pushed via `SSH_AUTH_SOCK=...
git push origin feature/sprint-004-rebase-dflash`. Result:
`1c9b77fdd..afec36229` landed on
`rapatel0/llama-cpp-turboquant feature/sprint-004-rebase-dflash`
on 2026-04-27.

`docker/Dockerfile` `ROTORQUANT_COMMIT` bump committed in the
same session; `make build` now reachable.

**Files**:
- fork: commit `afec36229f12253a60497099c5933e708da7e450` (pushed)
- `docker/Dockerfile` (pin landed)

---

## F-014: Prompt-3 ("Plan a 1 day trip to DC") transport-errors on speculative legs

**Severity**: Important (drops one of five gate-prompt cells; doesn't
block the sprint but means the median DFlash× is computed over 4 of 5
prompts).

**What**: After the F-011 fix, the qwen Phase 1 L4 sweep collected
target-only fine on all 5 prompts, but prompt 3 ("Plan a 1 day trip
to DC.") failed on **both** speculative legs (autoregressive AND
DFlash) with the bench harness's "transport error / connection
reset" pattern, exhausting all 9 retries (3 trials × 3 attempts). The
other 4 prompts each completed 3 trials cleanly.

```
[autoregressive][3/5][trial 0..2] retry after transport error (attempt 1..3/3)
[dflash][3/5][trial 0..2] retry after transport error (attempt 1..3/3)
```

100% draft acceptance on the prompts that did complete (P1,2,4,5),
so this isn't an acceptance/correctness regression — the speculative
path can't sustain a request on this specific prompt content.

**Why interesting**: target-only finishes prompt 3 fine, so it isn't
an OOM or model-load issue. The repro is the *same draft* + *same
target* + *one specific prompt* failing reproducibly across two
distinct speculative modes. Suggests a content-specific sequence in
P3 trips a state in the draft graph that legacy autoregressive and
DFlash both share.

**Why discovered**: Phase 1 L4 sweep on `qwen` (Qwen3.6-27B + DFlash),
2026-04-28. Logged in `/tmp/sprint005-bench-qwen.log`.

**Suggested sprint**: Sprint 006-dflash or whenever DFlash performance
work happens — no immediate need to block on it; the 4 remaining
prompts gave a clear gate verdict (FAIL ≥1.3×) so P3 wouldn't have
flipped the outcome.

**Investigation pointers**:
- Repro: `make run-qwen-bg && curl ... -d '...Plan a 1 day trip to
  DC...' --max-time 60`. Compare to `--max-time 60` against
  qwen-target-only.
- Check `docker logs rotorquant-qwen` immediately after the failure
  for any speculative-path warnings or `LOG_ERR` lines (DFlash
  draft / encoder failures log on lines 805, 829 of
  `common/speculative.cpp`).
- The harness's retry waits + restarts the request; if the server
  hangs (vs crashes), the bench exhausts retries while the server is
  fine. That suggests a deadlock or infinite-loop in the draft path
  rather than a crash.

**Files**:
- fork: `common/speculative.cpp` (DFlash draft path, autoregressive
  draft path)
- repo: `scripts/bench_speculative.py` (retry harness — the existing
  retry hardening did its job here, surfacing the failure cleanly
  rather than masking it)

---

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| F-011 | resolved | — | fork commit `40856a1d2`; rotorquant image rebuilt at the new pin; smoke verified 3 sequential requests stable |
| F-012 | Important | Immediate | `docs/sprints/SPRINT-005-experiments.json`, `scripts/sweep_dflash.py` |
| F-013 | resolved | — | fork commit `afec3622...` pushed; `docker/Dockerfile` pin landed |
| F-014 | Important | Sprint 006-dflash or DFlash perf work | fork `common/speculative.cpp` draft paths; repo `scripts/bench_speculative.py` (harness handled correctly) |
| F-015 | Important | Sprint 006-dflash | redesign `tests/test_speculative.py::TestForceReject` to actually exercise the env hook (the monkeypatch.setenv approach can't reach the dockerized server) |

---

## F-015: Pytest force-reject test design bug — env never reaches the server

**Severity**: Important (Phase 3 had pytest validation as part of Hard Gate #4; the test as-written doesn't validate the env hook).

**What**: `tests/test_speculative.py::TestForceReject::test_force_reject_preserves_output`
uses `monkeypatch.setenv("LLAMA_SPEC_FORCE_REJECT_AT", "8")` between
two completions and asserts the outputs match. Problem:
- `monkeypatch.setenv` sets the env in the pytest process
- `LLAMA_SPEC_FORCE_REJECT_AT` is read by `common_speculative_init` in
  the dockerized `llama-server` (a different process, started before
  pytest runs)
- The env never reaches the server, so both completions run with the
  same server config and produce identical tokens deterministically,
  spuriously passing the assertion.

The original `xfail()` was acknowledging the env hook hadn't landed.
After fork commit `afec36229` landed the hook, codex flipped
`xfail(strict=True)` expecting xpass. But the structural test bug means
the test still always passes (just for the wrong reason); strict=True
turns that spurious pass into a suite failure. Reverted in the
2026-04-28 session.

**Why this isn't a Critical-blocker for Sprint 005**: Hard Gate #4
covers C++ subtests F-H in the fork's
`tests/test-checkpoint-hybrid-state.cpp` AND the pytest. The C++ ctest
runs in-process with env set at process start, so it CAN actually
validate the hook. The pytest is a complementary check, not the
primary one.

**Suggested redesign**: server fixture should accept
`LLAMA_SPEC_FORCE_REJECT_AT` as a parameter and bring up two distinct
docker compose containers (one with env unset, one with env=8) on
different ports. The test compares outputs across the two.
Alternative: a server-side debug endpoint that toggles the value at
runtime (more invasive on fork side).

**Files**:
- `tests/test_speculative.py` (redesign)
- potentially `tests/conftest.py` (param fixture)
- fork: nothing additional needed — the C++ ctest already covers
  this properly
