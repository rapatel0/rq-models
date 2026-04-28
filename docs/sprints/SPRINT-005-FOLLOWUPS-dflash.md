# Sprint 005 Follow-ups (-dflash)

Execution-discovered follow-ups while running Sprint 005-dflash.

---

## F-011: DFlash draft path asserts on second request after prompt-cache miss

**Severity**: Critical (blocks canonical L4 and broad sweep completion)

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

## F-013: Fork push permission mismatch for `rapatel0` remote

**Severity**: Critical (blocks Docker pin validation path)

**What**: Fork-side Phase 3 commit exists locally, but push to `rapatel0/llama-cpp-turboquant` failed (`Permission denied` / `403`).

**Why discovered**: Required push step for `ROTORQUANT_COMMIT` pinning failed with both SSH and HTTPS attempts.

**Suggested sprint**: Immediate (credential/permission fix, then rebuild and run pytest)

**Files**:
- fork: commit `afec36229f12253a60497099c5933e708da7e450` (local)
- `docker/Dockerfile` (pin currently points to unreachable commit until push succeeds)

---

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| F-011 | Critical | Immediate | fork `common/speculative.cpp:789`, fork `tools/server/server.cpp` cache-miss path |
| F-012 | Important | Immediate | `docs/sprints/SPRINT-005-experiments.json`, `scripts/sweep_dflash.py` |
| F-013 | Critical | Immediate | fork commit `afec3622...`, `docker/Dockerfile` |
