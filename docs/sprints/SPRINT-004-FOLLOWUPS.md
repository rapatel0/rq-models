# Sprint 004 Follow-up Items

Items discovered during Sprint 004 execution (Phases 0–6) that need to be
addressed in future work. This is the execution-discovered companion to
`SPRINT-004-DEFERRED.md` (planning-time deferrals) — items here were not
on the original deferred list but emerged as the sprint actually ran.

---

## F-001: Source-converted DFlash drafts

**Status**: open (load-bearing for L2/L3/L4 measurement gates)

**What**: The two community DFlash drafts pinned in
`docker/entrypoint.sh:60-62` (`spiritbuun/Qwen3.6-27B-DFlash-GGUF` and
`lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test`) use a non-canonical GGUF schema
relative to PR `#22105`'s `convert_hf_to_gguf.py`: arch-string and metadata
key-prefix mismatch (worked around in fork commit `bd7a7aabb` via an LLM_KV
arch-name override) plus a tensor-name mismatch (missing `fc.weight`, etc.)
that has no fork-side fix yet.

**Impact**: Every Phase 5 end-to-end gate is blocked behind this single
issue: L2 greedy equivalence + forced-rejection, L3 z-lab pytorch parity,
L4 speedup median ≥1.3× on Qwen3.6-27B, plus actual runs of
`tests/test_dflash_e2e.py` and `scripts/validate_dflash.py`. Profiles boot
through llama-server start; draft load is what fails.

**Suggested next action**: Either (a) gain access to z-lab's gated
safetensors and run the cherry-picked PR's `convert_hf_to_gguf.py` to
emit canonical GGUFs locally, or (b) wait for a community draft that
matches PR `#22105`'s schema. Option (a) is the controllable path; option
(b) is a pure scheduling dependency. Once a working draft drops, the L2
and L3 commands in BENCHMARK-REPORT.md §10 become one-shot and L4 is
three sequential profile-up/stop cycles.

**Files**: `docker/entrypoint.sh:57-63` (draft registry — bump pinned
SHA when a working draft is identified); BENCHMARK-REPORT.md §10 TBD
tables.

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
SPRINT-004-DEFERRED.md is partially-mitigated — the full graph is
already in fork. Plan the harness work, not the cherry-pick.

**Files**: SPRINT-005 planning docs (when created).

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
| F-001 | Source-converted DFlash drafts | open | Get z-lab safetensors access or wait for canonical-format community draft |
| F-002 | `LLAMA_SPEC_FORCE_REJECT_AT` env in fork | open | Add to fork's `common/speculative.cpp`; flip `xfail` to xpass-strict |
| F-003 | Formal C++ checkpoint test file | partially-mitigated | Author `tests/test-checkpoint-hybrid-state.cpp` (subtest C already inline) |
| F-004 | Runtime guard `prefill_complete`/`deferred_drained` | partially-mitigated | Defensive only; land alongside F-003 |
| F-005 | `docker/test.sh` cache-preservation gate run | open | Run on host with warmed `llm-models` + rebuilt image |
| F-006 | z-lab commit SHA pin | open | Capture + pin SHA on first L3 run |
| F-007 | `make bench-dflash-all` | open | Add aggregate Makefile target |
| F-008 | Sprint 005 EAGLE3 scope | informational | Plan harness only; cherry-pick already done |
| F-009 | `LLAMA_SPEC_NO_THINK=1` doc | closed | n/a (documented) |
