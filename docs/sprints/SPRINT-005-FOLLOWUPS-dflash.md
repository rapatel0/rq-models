# Sprint 005 Follow-ups (-dflash)

Execution-discovered follow-ups while running Sprint 005-dflash.

---

## F-011: Speculative server instability under repeated requests

**Severity**: Critical (blocks canonical L4 and broad sweep completion)

**What**: On both `qwen` and `qwen36` speculative profiles, the first completion request usually succeeds, then subsequent requests frequently fail (`Remote end closed connection without response`, `connection reset`) or hang.

**Why discovered**: Repeatedly reproduced during `make bench-dflash-all PROFILE=qwen` and direct OpenAI-compatible request loops.

**Suggested sprint**: Immediate (before finishing Sprint 005 measurement gates)

**Files**:
- `scripts/bench_speculative.py` (retry hardening landed)
- `scripts/sweep_dflash.py` (retry hardening landed)
- fork runtime paths around speculative verify/restart behavior

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
| F-011 | Critical | Immediate | `scripts/bench_speculative.py`, `scripts/sweep_dflash.py`, fork speculative runtime |
| F-012 | Important | Immediate | `docs/sprints/SPRINT-005-experiments.json`, `scripts/sweep_dflash.py` |
| F-013 | Critical | Immediate | fork commit `afec3622...`, `docker/Dockerfile` |
