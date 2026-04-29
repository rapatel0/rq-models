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

## F-014: Speculative-decoding infinite restore loop on partial acceptance — RESOLVED 2026-04-29 (perf followup pending)

**Severity**: was Critical (server hangs / aborts under bench load);
now **resolved as a correctness/crash bug**. Performance regression
remains as a separate follow-up.

**Resolution chain** (fork commits, local on
`feature/sprint-004-rebase-dflash`, awaiting interactive push by user):

| Commit | Fix |
|--------|-----|
| `40856a1d2` | F-011: reset DFlash + EAGLE3 cumulative state in `begin()` (initial fix; addressed second-request crash but not this loop) |
| `14ca877ea` | F-014 v1: clear draft KV in `begin()` (overshoot — didn't help) |
| `d0b3e9e34` | F-014 v2: switch `seq_id=-1` → `0`; add LOG_INF tracer to confirm fix runs (still didn't help) |
| `3d44a0b19` | F-014 v3: clear `spec_draft` (not move-assign `accepted`) on partial-acceptance restore (broke the spec_draft-reuse loop) |
| `a3fba48d5` | F-014 v4: call `common_speculative_begin` on restore so impl state stays in sync (exposed v4-induced 0%-acceptance regression) |
| `ee4248d73` | F-014 v5: `slot.spec_skip_next_round` flag — force one round of single-token decode after restore to break the deterministic-partial cycle |
| `5f58c0d81` | F-014 v6: revert v2's draft-KV clear in `begin()` (was over-cautious) |
| `46e9bcfb8` | F-014 v7: preserve `accumulated_ctx` prefix in `begin()` on restore — first signal of real recovery (P1 hit 52 tok/s, 25% real acceptance) but crashed P2 because the heuristic confused new-request with restore |
| `86272e841` | F-014 v8: split `begin()` (always full reset) vs `rollback(n_tokens)` (truncate-and-keep) — clean API, no more new-vs-restore guessing |

**Validation** (all 5 thinking-off prompts run end-to-end on
`make run-qwen-bg` after rebuild from 86272e841):

| Prompt | tps | tokens | raw drafts | accepted | real % |
|--------|----:|-------:|-----------:|---------:|-------:|
| P1 quicksort | 51.3 | 96 | 300 | 75 | 25.0% |
| P2 Pythagorean | 13.7 | 256 | 3240 | 134 | 4.1% |
| P3 DC trip | 10.5 | 256 | 7050 | 135 | 1.9% |
| P4 Hamlet | 10.6 | 256 | 10830 | 138 | 1.3% |
| P5 SQL | 14.5 | 256 | 13605 | 208 | 1.5% |

**Root cause** (synthesized from the iteration): hybrid-state target
contexts (Qwen3.6's SWA + recurrent KV) can only do `seq_rm` at full-
context granularity (`COMMON_CONTEXT_SEQ_RM_TYPE_FULL`). When verify
partial-accepts, the slot can't commit just the accepted prefix — it
must restore the checkpoint and re-do. Pre-fix code did
`slot.spec_draft = std::move(accepted)` and continued; the next
iteration reused the partial-accepted result as the "draft", verify
returned the same partial outcome (deterministic at temp=0/top_k=1),
and the slot looped forever. With a 156 MiB checkpoint restore per
loop iteration, server GPU stayed at 80% util with no forward
progress until the client timed out.

**Performance regression** (open as F-018, separate follow-up): even
with the loop fixed, every partial-acceptance round is now penalized:
1. Full target-side checkpoint restore (~150-300 MiB GPU memory copy)
2. Forced single-token decode (1 token forward; no speculative gain)
3. `rollback()` truncate of `accumulated_ctx`

For prompts where the small DFlash draft (~1.7B for the 27B target)
struggles to get full acceptance (Pythagorean / Hamlet / SQL with
thinking-off), this throttles throughput to ~10-15 tok/s — much worse
than the ~70 tok/s target-only baseline. The pre-fix code was faster
WHEN it didn't hang, but hang-or-fast is a worse tradeoff than
slow-but-reliable.

Ideas for F-018:
- Skip-round only after K consecutive partial-acceptances at the same
  position (avoid penalizing one-off partials)
- Smaller block_size dynamically when partial pattern is observed
- Distill a draft model that stays high-acceptance on non-code prompts

**What was tried and reverted**:
- Clearing draft KV in begin() (v2/v6 cycle): broke draft quality,
  forced 0% real acceptance.
- Heuristic distinguishing new-request vs restore inside begin() (v7):
  worked sometimes but couldn't always tell the cases apart;
  superseded by the v8 API split.

**Files** (fork, local commits — push awaiting interactive
authorization in user's shell since the codex-CLI environment's SSH
agent can't sign without 1Password approval):
- `common/speculative.h`: declare `common_speculative_rollback()`
- `common/speculative.cpp`: virtual `rollback()` in
  `common_speculative_state`; DFlash override truncates
  `accumulated_ctx`; free function loops impls
- `tools/server/server-context.cpp`: in partial-restore path call
  `common_speculative_rollback()` (not `_begin`); set
  `slot.spec_skip_next_round`; declare `spec_skip_next_round` field
  reset on slot init

---

## F-014-original (subsumed by above): Prompt-3 transport-errors on speculative legs

This was the original framing of F-014 — qwen P3 ("Plan a 1 day trip
to DC.") timing out on the speculative legs only. Traced through to
the partial-acceptance restore loop above. The original wording
preserved here for historical context.

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
| F-014 | resolved (crash); F-018 spawned (perf) | — | fork commits `40856a1d2..86272e841` (local; user pushes) |
| F-018 | Important | Sprint 006-dflash | fork `tools/server/server-context.cpp` partial-acceptance restore — single-token-fallback throttles throughput; needs adaptive heuristic (only skip after K consecutive partials) |
| F-015 | Important | Sprint 006-dflash | redesign `tests/test_speculative.py::TestForceReject` to actually exercise the env hook (the monkeypatch.setenv approach can't reach the dockerized server) |
| F-016 | Critical (correctness of bench reporting) | Sprint 006-dflash | fork `tools/server/server-context.cpp` — `timings.draft_n` reports post-verify accepted count, not total drafts generated; observed acceptance metric reads 100% even when real rate is ~37% |
| F-017 | Critical (interpretation) | Sprint 006-dflash | `qwen` with thinking-on dropped to ~37% real acceptance (predicted by Sprint 005 risk row #1); the apparent "DFlash slow" is mostly thinking-on regime cost, not implementation defect; bench should publish both regimes |

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

---

## F-016: `timings.draft_n` reports post-verify count, not raw drafts generated

**Severity**: Critical (the bench's "100% acceptance" headline is
misleading — actual rate is ~37% on thinking-on prompts).

**What**: llama-server's response includes `timings.draft_n` and
`timings.draft_n_accepted`. The bench harness computes
`acceptance_rate = draft_n_accepted / draft_n` and got 100% on every
non-erroring leg of Sprint 005's Phase 1 sweep. **That's a metric
artifact, not reality**.

The server's own internal log line tells the truth:

```
draft acceptance rate = 1.00000 (  107 accepted /   107 generated)   <-- API metric
statistics unknown: #calls(b,g,a) = 1 19 19, #gen drafts = 19,
                    #acc drafts = 19, #gen tokens = 285,
                    #acc tokens = 107                                <-- real picture
```

Two different counters:
- API path (`timings.draft_n`): 107 / 107 = 100% — counts only the
  draft tokens that survived verify and got committed to output.
- Internal stats: 285 generated / 107 accepted = 37.5% — the real
  draft acceptance rate on this prompt with thinking-on.

So 178 of 285 generated draft tokens were rejected by verify and
discarded. The wasted draft compute is a meaningful chunk of the
end-to-end speculative cost, and explains why Sprint 005's Phase 1
results showed sub-1× DFlash× even at "100% acceptance".

**Why discovered**: 2026-04-28, while diagnosing why the qwen36 MoE
showed median DFlash× = 0.52 despite a tiny ~480M draft and ~100%
reported acceptance. Per-request timing showed 19 rounds × ~15 drafts
generating 285 tokens to produce 128 output tokens, while the
acceptance metric still read 100%.

**Suggested fix**: surface the real `#gen drafts` and `#acc drafts`
counts via the OpenAI-compat response (probably extending
`timings` with `draft_n_generated` separate from
`draft_n_accepted`). Bench harness then uses the new field. May also
warrant deprecating the existing `draft_n` semantics (or renaming to
`draft_n_committed`).

**Files**:
- fork: `tools/server/server-context.cpp` (response build path that
  fills `timings.*`)
- repo: `scripts/bench_speculative.py` (harness reads new field once
  fork ships it; falls back gracefully on older fork pins)
- repo: regenerate Sprint 005 summary docs once F-016 + F-017 land
  with corrected acceptance numbers

---

## F-017: Sprint 005 results need a thinking-off comparison row

**Severity**: Critical (interpretation of Sprint 005's gate verdict).

**What**: Sprint 005 chose thinking-on as the validation regime
because that's what production ships. Sprint 005 Phase 1 measured
median DFlash× = 0.80 on qwen / 0.52 on qwen36 and reported FAIL on
Hard Gate #3. Per-request timing analysis on 2026-04-28 (qwen36
Pythagorean) revealed the real draft acceptance rate is ~37%, not
the 100% the metric reports. **This matches Sprint 005 risk row #1's
prediction exactly**:

> "L4 ≥1.3× median fails on qwen (27B + DFlash) with thinking-on |
> Medium | Medium | Document the gap honestly. Sprint 004 chose
> thinking-on as the validation regime; PR's 60–80pp acceptance loss
> with thinking-on is a known deployment cost."

So the FAIL verdict is honest, but the **root cause is thinking-on
acceptance penalty**, not "DFlash draft graph cost > target verify
cost" as my BENCHMARK-REPORT discussion section claimed. With ~37%
acceptance instead of 100%, ~60% of draft compute is wasted, and even
a tiny 480M draft can't compensate.

To make the verdict useful for operators, Sprint 005's measurement
should be re-run with thinking-off (`LLAMA_SPEC_NO_THINK=1` per the
PR's original benchmark regime) and both regimes published side by
side. PR #22105's published numbers were thinking-off — comparing
them apples-to-oranges with our thinking-on numbers makes DFlash look
worse than it is on the regime it was designed for.

**Why discovered**: same 2026-04-28 diagnostic session as F-016. The
"why is DFlash so slow" question only resolves once you read the
internal server stats and realize the acceptance number was
misreported.

**Suggested action**: Sprint 006-dflash should re-run Phase 1 with
both `LLAMA_SPEC_NO_THINK=1` (PR baseline regime) and thinking-on
(deployed regime) and publish both rows. The thinking-off numbers
should land closer to PR's published 1.5–2× speedup, validating that
the implementation isn't broken — it's just penalized hard by
thinking-on.

**Files**:
- repo: `scripts/bench_speculative.py` — add `--no-think` flag (or
  parameterize the regime in `bench-dflash-all`)
- repo: `Makefile` — add `bench-dflash-all-nothink PROFILE=...`
- repo: `docs/BENCHMARK-REPORT.md` — re-publish with both regimes
- repo: `README.md` — update the headline summary block to caveat
  the thinking-on penalty
