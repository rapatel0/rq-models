# Sprint 004 Follow-up Items

Items discovered during Sprint 004 execution (Phases 0–6) that need to be
addressed in future work. This is the execution-discovered companion to
`SPRINT-004-DEFERRED-dflash.md` (planning-time deferrals) — items here were not
on the original deferred list but emerged as the sprint actually ran.

---

## F-001: Source-converted DFlash drafts

**Status**: resolved (both pairs converted; both behind opt-in gates)

**What**: The two community DFlash drafts originally pinned in
`docker/entrypoint.sh` used a non-canonical GGUF schema relative to PR
`#22105`. Resolved by source-converting from z-lab safetensors via
`scripts/convert_dflash_drafts.sh` + a fork-side tokenizer-hash mapping
(Qwen3.6 BPE → reuses `qwen35` pre-tokenizer, since the regex is
identical and only the vocab IDs differ).

**35B-A3B**: Done. `Qwen3.6-35B-A3B-DFlash-bf16.gguf` (959 MB) loads
clean, end-to-end speculative on RTX 5090 produced **128.9 tok/s decode
with 100% acceptance** on a 16-token "capital of France is" probe
(target Q4_K_XL + draft bf16, ngl=99/99, ctx=2048, greedy). DFlash
auto-setup correctly extracted target layers `[2, 11, 20, 29, 38]`
matching the draft's `dflash_config.target_layer_ids`.

**27B**: Done. Access was granted by z-lab; converted
`Qwen3.6-27B-DFlash-bf16.gguf` (3.47 GB) loads and runs DFlash speculative
end-to-end. Smoke (target Q4_K_XL + draft bf16, ngl=99/99, ctx=2048,
greedy, thinking-on): **75.4 tok/s decode, 37.3% acceptance** on a
7-token probe. The 37% acceptance is the thinking-on regime (Sprint 004
keeps this as the validation default; PR #22105's headline 60–80%
acceptance is no-think and ships as opt-in only).

**Gating change (2026-04-27)**: 27B-DFlash is now behind `PREVIEW=1`
(parallel to `EXPERIMENTAL=1` for the 35B MoE) because the z-lab draft
training is still iterating. `make run-qwen36-27b-dflash` sets PREVIEW=1
implicitly. Operators should re-run `make convert-drafts` after each
upstream draft refresh.

**Impact**: Both pairs convertible reproducibly via `make convert-drafts`.
L2/L3/L4 measurement gates are now runnable on both `qwen36-dflash`
(EXPERIMENTAL=1) and `qwen36-27b-dflash` (PREVIEW=1) profiles.

**27B correctness probe (2026-04-27)**: ran `scripts/probe_27b_correctness.sh`
against the rebuilt `rotorquant:latest` image. Quicksort prompt, 256 tokens,
greedy, thinking-on. **Result: PASS** — 937/937 char shared prefix between
target-only and target+DFlash output (zero divergence within the shared
length); 28-char tail-length difference is a token-budget artifact (both
runs hit the 256-token cap, speculative tokenization can place the cap at
slightly different character positions in the same string). Acceptance on
the longer prompt: **221/221 = 100%**, decode 1.053× target-only. Confirms
the 37% acceptance from the 7-token smoke was a prompt-regime observation,
not a verify-path bug. Report: `docs/sprints/SPRINT-005-27b-correctness-probe.md`.

**Suggested next action**: Run the L4 5-prompt benchmark with thinking-on
on both profiles to establish baseline numbers, then explore the
quality-vs-acceptance tradeoffs (target Q5_K_M, draft KV variants,
chat-template canonical form) on the 27B for headline-prompt optimization.

**Files**: `scripts/convert_dflash_drafts.sh` (host-side converter);
`scripts/probe_27b_correctness.sh` (correctness probe); fork
`convert_hf_to_gguf.py` (one-line tokenizer-hash mapping for Qwen3.6);
`docker/entrypoint.sh:57-72` (registry now points at local GGUFs, with
`local/...` short-circuit in `download_model_if_missing()`).

---

## F-002: `LLAMA_SPEC_FORCE_REJECT_AT=N` debug env in fork

**Status**: open (Phase 2 deferred; tracked back through Phase 5 harness)

**What**: A debug env that forces speculative rejection at a specific
position N in the draft, exercising the checkpoint-restore + replay path
deterministically. Phase 2 deferred this once it became clear PR `#22105`
heavily rewrites `common/speculative.cpp` and the right hook points would
be clearer post-cherry-pick. Phase 3 cherry-pick has now landed; the hook
points are reachable.

**Impact**: Without this env, forced-rejection coverage in
`tests/test_speculative.py::test_force_reject_preserves_output` cannot
fire reliably — the test is currently `@pytest.mark.xfail` because the
env hook is itself deferred. Hard gate 5 (forced-rejection correctness)
also depends on this for `tests/test-checkpoint-hybrid-state.cpp`
subtests F-H.

**Suggested next action**: Add the env in fork's `common/speculative.cpp`
post-cherry-pick block-draft + verify orchestration. Once it lands and
ROTORQUANT_COMMIT bumps, flip the `xfail` decorator on
`test_force_reject_preserves_output` to `@pytest.mark.xfail(strict=True)`
or remove it.

**Files**: fork `common/speculative.cpp`; this repo's
`tests/test_speculative.py`.

---

## F-003: Formal C++ unit test `tests/test-checkpoint-hybrid-state.cpp`

**Status**: partially-mitigated

**What**: The Sprint 004 plan called for a formal C++ unit-test file in
the fork covering subtests A (deferred f16 staging), B (4 quantized K
layouts: planar3/planar4/iso3/iso4), C (recurrent state save→mutate→
restore on `linear_attention` layers), D (cross-layer mixed-batch
checkpoint), E (convert-during-checkpoint TOCTOU guard), and F-H
(forced-rejection bytes-equal asserts).

**Impact**: Subtest C is already exercised inline in
`examples/checkpoint-bench/checkpoint-bench.cpp` and verified passing on
both production targets — the load-bearing recurrent-state correctness
proof is in place. Subtests A, B, D, E formalize what's currently only
covered by the inline check; subtests F-H are blocked on F-002.

**Suggested next action**: Author the file in fork as a CTest target.
Subtest C can wrap the existing inline check. Subtests F-H wait on F-002.

**Files**: fork `tests/test-checkpoint-hybrid-state.cpp` (new),
`tests/CMakeLists.txt`.

---

## F-004: Runtime guard in `llama_kv_cache_unified`

**Status**: partially-mitigated (defensive only)

**What**: Phase 2 belt-and-suspenders flags `prefill_complete` and
`deferred_drained` plus a verify-batch quantized append helper that
treats multi-token verify appends as quantized decode appends, not as
fresh prefill. The runtime guard refuses to arm speculative until
`prefill_complete && deferred_drained` is true.

**Impact**: In normal flow, `convert_deferred_keys()` runs at end of
prefill before any speculative arm; the race window doesn't open. The
guard is defensive correctness for unusual call patterns or future
multi-slot work (where ordering invariants weaken — see SPRINT-004-
DEFERRED.md D-002).

**Suggested next action**: Land in fork alongside F-003 subtest E (TOCTOU
exercise). Low priority; not gating any current correctness story.

**Files**: fork `src/llama-kv-cache.cpp`.

---

## F-005: `docker/test.sh` cache-preservation gate run

**Status**: open (logic shipped; first execution pending)

**What**: Phase 4 rewrote `docker/test.sh` as a 4-stage smoke (build →
cache snapshot → per-profile boot → cache diff). The cache-preservation
hard gate (Definition of Done #9) requires the `llm-models` named volume
to be byte-identical pre/post for every existing profile. The script
implements `volume_mtime_snapshot` + `diff -q` for that check.

**Impact**: Until this runs end-to-end on a host with a warmed
`llm-models` volume *and* the rebuilt rotorquant image (rebuild needed
because Dockerfile bumps `ROTORQUANT_COMMIT` to `bd7a7aabb`), the cache
preservation gate is unverified. The 8 existing profiles' boot+complete
checks are likewise unverified post-entrypoint refactor.

**Suggested next action**: First task on a host with the model cache
populated. Build the new image (`make build`), then `make smoke`. Address
any profile that fails to boot before declaring Phase 4 fully done.

**Files**: `docker/test.sh`, `docker/Dockerfile`.

---

## F-006: z-lab/dflash commit pin

**Status**: open

**What**: `scripts/validate_dflash.py:52` currently sets
`ZLAB_COMMIT = "HEAD"` as a placeholder. Sprint 004 security
considerations explicitly state z-lab clone "never `HEAD`".

**Impact**: First L3 invocation against a moving HEAD risks
non-reproducible reference outputs. Soft impact only until F-001 unblocks
L3 measurement.

**Suggested next action**: On the first L3 run, capture
`git rev-parse HEAD` from the cloned repo and pin that SHA in
`scripts/validate_dflash.py:52`. Commit the pin alongside the L3 results
JSON.

**Files**: `scripts/validate_dflash.py`.

---

## F-007: Single `make bench-dflash-all` orchestration target

**Status**: open (operator-friendly improvement)

**What**: The L4 reproducibility flow currently requires three sequential
manual cycles per the Makefile usage block (target-only, autoregressive,
dflash; each is profile-up + leg-bench + profile-stop). A single
`bench-dflash-all` target could orchestrate the sequence with health
gating between legs.

**Impact**: Cosmetic; affects operator experience for repeat L4 runs
once F-001 unblocks measurement. The current per-leg approach is more
robust to mid-sequence failure but slower for the green-path case.

**Suggested next action**: Add `bench-dflash-all` as an aggregate target
that chains the three legs with `&&` and a health-poll between each.
Keep the per-leg `bench-dflash-leg LEG=...` target for failure-isolation
debugging.

**Files**: `Makefile`.

---

## F-008: Sprint 005 EAGLE3 — profile + harness only

**Status**: informational (changes Sprint 005 scope)

**What**: Phase 3 cherry-pick squash-merged the entire shared tree of
PRs `#22105` (DFlash) and `#18039` (EAGLE3); both arrived in fork in a
single commit. The Sprint 004 plan envisioned cherry-picking only
"minimal EAGLE3 foundation" commits and deferring full EAGLE3 to
Sprint 005, but the full EAGLE3 model graph (`src/models/eagle3.cpp`,
+186 LOC) is now present in fork.

**Impact**: Sprint 005's EAGLE3 scope shrinks: no fork-side cherry-pick
needed. What remains is repo-side: a Docker compose profile (e.g.
`qwen36-27b-eagle3`), entrypoint command-builder branch for
`SPECULATIVE_MODE=eagle3`, and validation harness + benchmarks
analogous to Sprint 004's DFlash gates.

**Suggested next action**: When Sprint 005 plans, note that D-001 in
SPRINT-004-DEFERRED-dflash.md is partially-mitigated — the full graph is
already in fork. Plan the harness work, not the cherry-pick.

**Files**: SPRINT-005 planning docs (when created).

---

## F-010: Multi-slot speculative — batched draft inference

**Status**: open (optimization, not correctness)

**What**: The cherry-picked PR #22105 instantiates one `common_speculative`
context per server slot
(`tools/server/server-context.cpp:928`) but all slots share a single
draft `llama_context` with `params_dft.n_parallel = 1` (server-context.cpp:779).
Multi-slot speculative is functionally correct — each slot has independent
state and the verify path handles per-slot rollback — but the draft side
serializes: only one slot's draft inference runs at a time.

**Impact**: When multiple users hit a DFlash profile concurrently (`N_PARALLEL`
> 1), draft inference becomes a serialization point. For drafts where draft
inference is fast relative to target verify (which is the case for both our
27B and 35B draft pairs), this matters less; for pathological cases the
multi-slot speedup is sub-linear.

The TODO at server-context.cpp:339 (`TAG_SERVER_SPEC_REWORK`) explicitly
calls out the optimization: "perform the speculative drafting for all
sequences at the same time in a single batch". Upstream PR #18961 is
the in-flight rework.

**Suggested next action**: Don't fork-pick #18961 (large, still iterating).
Run a throughput experiment with `N_PARALLEL=2/4/8` on
`qwen36-27b-dflash` (PREVIEW=1) at fixed ctx + greedy + thinking-on to
characterize the serialization cost in our setup. If sub-linear by
>50% at N=4, escalate.

**Files**: `docker/entrypoint.sh:188-198` (default-but-overridable
`N_PARALLEL`); fork `tools/server/server-context.cpp:339, 779`.

---

## F-009: `LLAMA_SPEC_NO_THINK=1` documentation

**Status**: closed (documented in README + BENCHMARK-REPORT.md §10)

**What**: An env that suppresses Qwen3.x thinking-mode tokens for the
draft-aligned chat template. Read in
`examples/speculative-simple/speculative-simple.cpp:134`. Verified to
have come into the fork via PR `#22105` (commit `9993e8ae8`), so it's
upstream-side, not fork-only.

**Impact**: Per upstream, thinking-on drops acceptance rate by 60–80
percentage points on Qwen3.x. Documented in the new "Speculative
Decoding (Experimental)" subsection of `README.md` and in
BENCHMARK-REPORT.md §10's acceptance-rate notes paragraph.

**Suggested next action**: None. Close on Sprint 004 wrap.

**Files**: `README.md`, `docs/BENCHMARK-REPORT.md` §10,
`scripts/bench_speculative.py:265` (sets the env unconditionally).

---

## Summary table

| Item | Title | Status | Suggested next action |
|------|-------|--------|------------------------|
| F-001 | Source-converted DFlash drafts | resolved | Both pairs converted; re-run `make convert-drafts` after upstream draft refresh |
| F-002 | `LLAMA_SPEC_FORCE_REJECT_AT` env in fork | open | Add to fork's `common/speculative.cpp`; flip `xfail` to xpass-strict |
| F-003 | Formal C++ checkpoint test file | partially-mitigated | Author `tests/test-checkpoint-hybrid-state.cpp` (subtest C already inline) |
| F-004 | Runtime guard `prefill_complete`/`deferred_drained` | partially-mitigated | Defensive only; land alongside F-003 |
| F-005 | `docker/test.sh` cache-preservation gate run | open | Run on host with warmed `llm-models` + rebuilt image |
| F-006 | z-lab commit SHA pin | open | Capture + pin SHA on first L3 run |
| F-007 | `make bench-dflash-all` | open | Add aggregate Makefile target |
| F-008 | Sprint 005 EAGLE3 scope | informational | Plan harness only; cherry-pick already done |
| F-009 | `LLAMA_SPEC_NO_THINK=1` doc | closed | n/a (documented) |
| F-010 | Multi-slot speculative batched-draft optimization | open | Throughput experiment with N_PARALLEL>1; escalate if >50% sub-linear at N=4 |
