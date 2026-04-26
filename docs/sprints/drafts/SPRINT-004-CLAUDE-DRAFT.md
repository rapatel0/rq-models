# Sprint 004: Rebase Fork + Hybrid-Aware Speculative Decoding for Qwen3.6

## Overview

Both production targets (Qwen3.6-35B-A3B MoE and Qwen3.6-27B "dense") are
**75% recurrent-state hybrid models** — `linear_attention` Gated Delta Net
layers in a 3:1 pattern with `full_attention` blocks. The recurrent layers
have non-decomposable per-token state, which makes naive speculative decoding
(reliant on `llama_memory::seq_rm()` partial truncation) functionally broken
on every model we ship. The fix already exists upstream: PRs #19493
(speculative checkpointing) and #22227 (hybrid model state fallback) are
**merged to mainline llama.cpp** and replace `seq_rm`-based rollback with a
snapshot/restore mechanism. PR #22105 (DFlash block-diffusion drafts)
explicitly builds on this checkpointing API. None of this is in our fork
today.

This sprint rebases the RotorQuant fork onto current master to bring in
#19493 + #22227 as the **load-bearing prerequisite** for any speculative
work, validates that the upstream snapshot/restore mechanism correctly
handles our deferred-K state and quantized planar/iso K layouts, then
cherry-picks #22105 to add DFlash drafts on top. The deliverables are
two new Docker profiles (`qwen36-dflash`, `qwen36-27b-dflash`), a fork-level
C++ unit test for the checkpoint × deferred-K interaction, a differential
correctness harness against the z-lab pytorch reference, and a measured
decode-speedup result on Qwen3.6-27B.

The realistic decode-speedup target is **1.5–2.0× on Qwen3.6-27B** — the
PR's own data on Qwen3.5-9B (the closest hybrid analogue) shows 1.10–2.77×,
not the 1.77–8.08× of pure-attention models. The MoE target may come in
near 1.0×; we ship it as `EXPERIMENTAL=1`. Anything claiming "2.0–4.0×"
ignores the architectural reality of recurrent layers; we are not going
to make that claim.

## Use Cases

1. **Faster interactive coding/reasoning on Qwen3.6-27B (24 GB tier)**:
   1.5–2.0× decode at greedy temp=0 cuts a 2,000-token reasoning trace
   from ~30 s to ~15–20 s on RTX 5090, without any quality regression
   (greedy bit-identical to target-only).

2. **Free MoE upgrade path**: If snapshot cost permits, Qwen3.6-35B-A3B
   gets a no-quality-cost speedup. Even at 1.0–1.2× (the realistic MoE
   floor), we ship the harness and unblock future improvements.

3. **Validated hybrid speculative codepath in our fork**: A C++ unit test
   that exercises the full checkpoint → mutate quantized K → restore loop
   becomes a permanent regression guard — every future fork rebase or
   kernel change must keep it green.

4. **Fork that tracks mainline**: A clean rebase eliminates ~5 months of
   accumulated drift, bringing not just speculative checkpointing but every
   other mainline improvement (fattn, kv-unified, etc.).

5. **Sprint 005 readiness**: Once hybrid speculative is correct, we can
   reach for non-greedy sampling, multi-slot speculative, or alternate
   draft architectures with confidence. Sprint 004 establishes the
   correctness floor.

## Architecture

### Three-stage delivery

```
            ┌─────────────────────────────────┐
            │  Stage A: REBASE                │
            │  feature/planarquant-kv-cache   │
            │      ──rebased onto──►          │
            │  ggml-org/llama.cpp master      │
            │                                 │
            │  Brings in:                     │
            │   • #19493 checkpoint API       │
            │   • #22227 hybrid fallback      │
            │   • mainline drift since 2026-04│
            │                                 │
            │  L1 GATE: PPL within ±0.05      │
            │  on 4 KV × 2 corpora × 2 models │
            └─────────────────┬───────────────┘
                              │
            ┌─────────────────▼───────────────┐
            │  Stage B: CHECKPOINT × KV       │
            │  Validate snapshot/restore on   │
            │  our quantized K cache          │
            │                                 │
            │  • f16 staging buffer (defer_k) │
            │  • planar3/planar4 quant K      │
            │  • iso3/iso4 quant K            │
            │                                 │
            │  Fork-level C++ test asserts    │
            │  bit-equality on restore        │
            │                                 │
            │  Cost benchmark @ 65K, 262K     │
            │  (snapshot per verify step)     │
            └─────────────────┬───────────────┘
                              │
            ┌─────────────────▼───────────────┐
            │  Stage C: DFLASH                │
            │  Cherry-pick #22105 onto        │
            │  rebased fork                   │
            │                                 │
            │  L2 GATE: greedy target-only    │
            │  ≡ target+DFlash (3 prompts)    │
            │                                 │
            │  L3 GATE: first 64 tokens match │
            │  z-lab pytorch reference;       │
            │  acceptance rate ±5pp           │
            │                                 │
            │  Decode speedup ≥1.5× on        │
            │  Qwen3.6-27B (greedy, coding)   │
            └─────────────────────────────────┘
```

### Hybrid speculative checkpointing × RotorQuant deferred-K

Why this is novel territory: upstream's snapshot/restore was written for
mainline KV layouts (`f16`, `q8_0`, etc.). RotorQuant's K cache has two
states the snapshot must capture correctly:

```
       ┌────────────────────────────────────────────────────────┐
       │  Per attention layer, K cache state during a request   │
       └────────────────────────────────────────────────────────┘

   Prefill in flight (defer_k=true):
       K layout   = f16 staging buffer  [n_tokens, n_kv_heads, head_dim]
       Status     = convert_deferred_keys() has NOT run
       Snapshot   = must capture full f16 buffer + "deferred" flag
       Restore    = put f16 back; do NOT trigger conversion

   Steady state (post convert_deferred_keys):
       K layout   = planar3/planar4/iso3/iso4 block-quantized
                    (50 B per 128 vals at 3.125 bpe)
       Status     = K is compressed; V is compressed in same shape
       Snapshot   = must capture quantized blocks + per-block scales
       Restore    = put quantized blocks back; no re-conversion needed

   Verify-time append (decode-style insertion of N draft tokens):
       Question   = does append at this stage re-trigger defer_k?
                    By construction NO — defer_k is a prefill-only
                    flag. Verify appends go straight to quantized
                    K. We assert this in the C++ unit test.
```

The snapshot interface from #19493 (verified by reading PR diff) operates
on `llama_memory_*::checkpoint_save() / checkpoint_restore()` returning
opaque snapshot handles. Mainline calls are `memcpy`-style on contiguous
backend buffers. Our quantized K cache **is** a contiguous backend buffer
(GGML_TYPE_PLANAR3 etc. are first-class types with known `type_size`),
so the basic mechanism should work. What we don't know without testing:

- Does the snapshot walk the recurrent-state buffers AND the attention KV
  buffer per layer? (Required: yes, since hybrid means both per layer.)
- Does delta/COW exist or is it full snapshot every verify? Cost matrix:

| Context | planar3 K size | Snapshot/verify (full) | DFlash 16-tok block | Verify steps in 256-tok decode |
|---------|---------------:|-----------------------:|:--------------------:|:------------------------------:|
| 4K      | ~50 MB         | ~50 MB GPU memcpy     | per block            | ~16                            |
| 65K     | ~3.3 GB        | ~3.3 GB GPU memcpy    | per block            | ~16                            |
| 262K    | ~13 GB         | **infeasible — exceeds VRAM headroom** | per block | ~16 |

If upstream is full-snapshot only, **262K speculative decoding is dead on
arrival** on a 32 GB GPU with our 27B + KV + draft model footprint. We
test at 65K as the realistic upper bound; 262K speculative becomes a
deferred follow-up.

### Sprint scope diagram (what's in / what's out)

```
  IN                                 │  OUT (deferred or moot)
  ──────────────────────────────────│──────────────────────────────────
  Rebase fork onto master           │  Block-aware seq_rm partial trunc
  Bring in #19493 + #22227          │  (superseded by checkpointing)
  Validate KV PPL post-rebase (L1) │  Custom kv_cache_quantized_seq_rm
  Checkpoint × deferred-K C++ test │  Standalone qwen36-spec autoregressive
  Cherry-pick #22105 (DFlash)      │  profile (it tested seq_rm — moot)
  qwen36-dflash + qwen36-27b-dflash│  Multi-slot speculative
  z-lab differential harness        │  Non-greedy speculative (sampler agreement)
  Snapshot cost benchmark           │  EAGLE3 cherry-pick beyond what #22105 needs
  bench_speculative.py              │  Open WebUI integration
  BENCHMARK-REPORT.md §10           │  CI bench runner (carry-forward D-013)
  Update README + QUANT-GUIDE
```

## Implementation

### Phase 1: Architecture Audit + Rebase Prep (~5% of effort)

**Files (this repo):**
- `docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md` — this doc
- `docs/sprints/SPRINT-004-DEFERRED.md` — to be created at sprint close

**Files (fork: johndpope/llama-cpp-turboquant):**
- `src/llama-context.cpp` — pre-rebase reading, conflict surface map
- `src/llama-kv-cache.cpp` — confirm `defer_k`, `convert_deferred_keys()`
  hooks vs mainline equivalents
- `src/llama-memory*.{cpp,h}` (mainline) — read #19493 diff to find
  `checkpoint_save`/`checkpoint_restore` symbol surface
- `src/llama-memory-recurrent.cpp` (mainline) — read `seq_rm` rejection
  path and #22227 hybrid fallback hook

**Tasks:**
- [ ] Pull `gh pr view 19493 --json title,body,files,commits` for both
      #19493 and #22227; record exact symbol names of checkpoint API
- [ ] Pull `gh pr view 22105 --json title,body,files`; record DFlash
      file list and confirm zero overlap with our `ggml-cuda/` kernels
- [ ] Read mainline `src/llama-context.cpp` post-#19493 to find call
      sites for `checkpoint_save`/`restore`; mark conflict zones against
      our deferred-K hooks
- [ ] Confirm our fork's `linear_attention` layer support (we never
      built against hybrid before — Qwen3.5-27B was pure attention).
      If layer dispatch is broken: this becomes a new phase, not a task
- [ ] Decide: rebase strategy is **interactive squash + replay**, not
      merge — the linear history matters for `git bisect` on regressions
- [ ] Identify the exact rebase base commit hash on master (pin in
      `docker/Dockerfile`)

### Phase 2: Rebase + KV Regression Gate L1 (~25% of effort)

**Files (fork):**
- `src/llama-context.cpp` — primary conflict file (+377/−6 from #22105
  on top of our deferred-K hooks)
- `src/llama-kv-cache.cpp` — verify deferred-K paths still compile and
  link against the rebased memory API
- `ggml-cuda/CMakeLists.txt` — likely conflict on template instance file
  list (mainline drift in CUDA build system)
- `ggml-cuda/cpy-planar-iso.cu`, `set-rows-planar-iso.cuh`,
  `planar-iso-constants.cuh`, `fattn-common.cuh` — verify no API drift
  on the ggml types we extend

**Files (this repo):**
- `docker/Dockerfile` — pin `ROTORQUANT_COMMIT` to rebased SHA
- `scripts/benchmark_kv_regression.py` — new; wraps existing
  `benchmark_rotorquant.py` to run the 4×2×2 PPL sweep

**Tasks:**
- [ ] Create branch `feature/planarquant-kv-cache-rebased`; rebase onto
      master pinned commit
- [ ] Resolve `src/llama-context.cpp` conflicts: deferred-K hooks must
      coexist with #22105's DFlash path additions (DFlash is brought in
      Phase 4; here we just keep deferred-K working against rebased
      memory API)
- [ ] Build with `-DGGML_CUDA=ON -DGGML_CUDA_FA=ON` for SM 80;86;89;90;100;120
- [ ] Smoke: load Qwen3.6-27B UD-Q4_K_XL with `--cache-type-k planar3`,
      do 64-token decode, no NaN
- [ ] **L1 GATE**: run PPL sweep, must match `BENCHMARK-REPORT.md` §1
      within ±0.05 PPL:
  - Qwen3.6-27B × {wikitext-2, C4} × {planar3, planar4, iso3, iso4}
  - Qwen3.6-35B-A3B × {wikitext-2, C4} × {planar3, planar4, iso3, iso4}
  - 16 PPL data points; any outside tolerance → block on diagnosis
- [ ] Commit rebased fork; push branch; pin SHA in `docker/Dockerfile`
- [ ] Rebuild image; verify all 8 existing Docker profiles still launch
      (`qwen`, `qwen36-q3`, `qwen36-iq3`, `qwen36-27b`, `qwen36-27b-q3`,
      `qwen36-27b-iq3`, `reasoning`, `gemma`) — health endpoint + one
      chat completion each

### Phase 3: Checkpoint × Deferred-K Validation (~25% of effort)

This is the load-bearing novel work. Without this, Phases 4–5 build on
unverified foundation.

**Files (fork):**
- `tests/test-checkpoint-deferred-k.cpp` — **NEW** C++ unit test
- `tests/CMakeLists.txt` — register new test
- `src/llama-kv-cache.cpp` — possible: snapshot hook for our K layout
  if upstream's interface needs an opt-in for non-mainline types
- `src/llama-memory-hybrid.cpp` (or wherever #22227 landed) — read-only
  reference

**Files (this repo):**
- `scripts/bench_snapshot_cost.py` — measures snapshot/restore wall time
  per verify step at varying contexts
- `tests/test_speculative_kv.py` — Python integration test (high level)

**Tasks:**
- [ ] Read upstream `checkpoint_save/restore` impl carefully — does it
      operate on opaque buffer view, or does it dispatch by `ggml_type`?
- [ ] Write `test-checkpoint-deferred-k.cpp` with three subtests:
  - **A: f16 staging restore** — set `defer_k=true`, append 128 tokens
    of K (kept in staging f16), `checkpoint_save`, modify staging buffer
    in-place, `checkpoint_restore`, assert byte-equality of staging buffer
    AND `defer_k` flag
  - **B: planar3 quantized K restore** — run prefill to completion so
    `convert_deferred_keys()` fires, K is now planar3, `checkpoint_save`,
    overwrite quantized K bytes with garbage, `checkpoint_restore`,
    assert byte-equality of quantized K buffer
  - **C: iso3 quantized K restore** — same as B with iso3 (different
    rotation, same block layout)
- [ ] Verify recurrent state (Gated Delta Net SSM state) is also captured
      by snapshot — write a 4th subtest that mutates SSM state between
      save/restore
- [ ] Bench: `bench_snapshot_cost.py` measures snapshot wall time at
      ctx ∈ {4K, 16K, 65K, 131K, 262K} for both planar3 and iso3, P=1 slot
- [ ] **Decision gate**: if snapshot cost at 65K exceeds 50% of expected
      verify-step wall time (~5 ms target → 2.5 ms budget), flag as
      "DFlash speedup capped at long context" and document; do not block
      Phase 4 — speedup gate is at 4K–16K context range anyway
- [ ] Document findings in `docs/sprints/SPRINT-004-FOLLOWUPS.md` for
      delta-snapshot or COW work

### Phase 4: DFlash Cherry-pick + Correctness Gate L2 (~20% of effort)

**Files (fork):**
- Cherry-pick #22105 commits onto rebased fork; expected file additions:
  - `src/models/dflash.cpp` (+161, new)
  - `src/models/eagle3.cpp` (+186, new)
  - `src/models/qwen35*.cpp` (+14–25 each, mods)
  - `src/models/qwen3moe.cpp` (+11)
  - `src/llama-graph.{cpp,h}` (+44)
  - `src/llama-arch.{cpp,h}` (+39)
  - `common/speculative.cpp` (+331/−23)
  - `examples/speculative-simple/speculative-simple.cpp` (+77/−20)
  - `convert_hf_to_gguf.py` (+187)
  - `gguf-py/gguf/constants.py` (+53)
- Conflicts expected only in `llama-context.cpp`; resolve preserving
  both checkpointing and deferred-K hooks

**Files (this repo):**
- `docker/Dockerfile` — bump `ROTORQUANT_COMMIT` to post-cherry-pick SHA
- `docker/entrypoint.sh` — extend model registry with DFlash draft
  entries; add `DRAFT_MODEL_NAME` and `DFLASH=1` env vars; build
  `--draft-model` and `--dflash` flags into command array
- `docker-compose.yml` — add two profiles: `qwen36-dflash`,
  `qwen36-27b-dflash`
- `Makefile` — add `run-qwen-dflash`, `run-qwen36-27b-dflash`
- `tests/test_dflash_e2e.py` — Python integration test

**Tasks:**
- [ ] Cherry-pick #22105 onto rebased branch; resolve `llama-context.cpp`
- [ ] Rebuild; smoke-load `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test` and
      `spiritbuun/Qwen3.6-27B-DFlash-GGUF` (pre-built, no conversion
      needed)
- [ ] Extend model registry in `docker/entrypoint.sh`:
  - `qwen3.6-27b-dflash-draft` → `spiritbuun/Qwen3.6-27B-DFlash-GGUF`
  - `qwen3.6-35b-dflash-draft` → `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test`
- [ ] Update `entrypoint.sh` to support draft model launch:
  - If `DRAFT_MODEL_NAME` is set: download draft, append
    `--draft-model /models/<draft>` and `--dflash` (or whatever flag
    PR #22105 settles on)
- [ ] Add Docker Compose profiles `qwen36-dflash`, `qwen36-27b-dflash`
- [ ] **L2 GATE**: greedy correctness on 3 prompts × 256 tokens at
      `--temp 0 --top-k 1 --seed 42`:
  - Run target-only, run target+DFlash, diff token streams
  - Must be byte-identical (greedy means draft acceptance/rejection
    cannot change the trajectory if both backends agree)
  - Prompts: "Write a quicksort algorithm.", "Summarize: <fixed
    paragraph>", "Translate to French: <fixed sentence>"
- [ ] If L2 fails: bisect — is it checkpoint × deferred-K (Phase 3 test
      should have caught it), or DFlash kernel itself? PR #22105 has its
      own tests; run those first before suspecting our quant path.

### Phase 5: z-lab Differential Harness + L3 Gate (~15% of effort)

**Files (this repo):**
- `scripts/validate_dflash.py` — clones z-lab/dflash, runs both backends
  on a fixed prompt set, compares
- `tests/test_dflash_zlab_diff.py` — pytest wrapper that fails on
  >5pp acceptance rate divergence
- `external/dflash-zlab/` — git submodule pointer to <https://github.com/z-lab/dflash>

**Tasks:**
- [ ] Add z-lab repo as submodule; pin to a specific commit SHA
- [ ] `validate_dflash.py`:
  - Same prompt + same seed in both backends
  - Greedy decode, 64 tokens
  - Assert first-64-token sequence matches exactly between z-lab pytorch
    and our llama.cpp fork
  - Log per-step acceptance rate from each backend; assert within ±5pp
- [ ] **L3 GATE**: passes for both Qwen3.6-27B-DFlash and
      Qwen3.6-35B-A3B-DFlash
- [ ] Document any structured differences (e.g., known floating-point
      rounding past token 32) in `docs/sprints/SPRINT-004-FOLLOWUPS.md`

### Phase 6: Benchmarks + Docs (~10% of effort)

**Files (this repo):**
- `scripts/bench_speculative.py` — measures decode tok/s for
  (target-only, target+autoregressive draft, target+DFlash) on a
  fixed 5-prompt set; thinking-on and thinking-off variants
- `docs/BENCHMARK-REPORT.md` — append §10 Speculative Decoding
- `README.md` — append "Speculative Decoding" section under "Available
  Models"
- `docs/QUANTIZATION-GUIDE.md` — note draft model VRAM additive cost
  per quant tier (16 GB tier likely cannot fit DFlash draft + target)

**Tasks:**
- [ ] `bench_speculative.py` runs the 5-prompt set on:
  - Qwen3.6-27B target-only @ planar3, ctx=8K
  - Qwen3.6-27B + DFlash @ planar3, ctx=8K
  - Qwen3.6-35B-A3B target-only @ iso3, ctx=8K
  - Qwen3.6-35B-A3B + DFlash @ iso3, ctx=8K
  - Report per-prompt tok/s, acceptance rate, snapshot cost percentage
- [ ] **Hard speedup gate**: ≥1.5× on Qwen3.6-27B for "Write a quicksort
      algorithm" with thinking-off (per revised intent §6 — Qwen3.5-9B
      hybrid floor was 1.10×, this 27B target with similar architecture
      should hit 1.5–2.0× per PR data extrapolation; if <1.5× we report
      result and document a Sprint 005 follow-up rather than failing
      the sprint)
- [ ] **Soft speedup gate**: ≥1.0× on Qwen3.6-35B-A3B (no regression).
      Anything ≥1.2× is a win; ship behind `EXPERIMENTAL=1` env var if
      between 1.0× and 1.2×
- [ ] Update README §Available Models with new commands:
  - `make run-qwen36-27b-dflash` (24+ GB tier)
  - `docker compose --profile qwen36-dflash up` (32+ GB tier,
    EXPERIMENTAL)
- [ ] Update QUANTIZATION-GUIDE.md to document additive draft-model
      VRAM cost — likely pushes 16 GB tier off the speculative path
      entirely; keep them on autoregressive
- [ ] Append §10 to BENCHMARK-REPORT.md with measured speedups,
      acceptance rates, snapshot cost vs context

## Files Summary

### Fork (johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache-rebased)

| File | Action | Purpose |
|------|--------|---------|
| `src/llama-context.cpp` | Modify (rebase + cherry-pick) | Resolve conflicts: deferred-K + checkpointing + DFlash |
| `src/llama-kv-cache.cpp` | Modify (rebase) | Verify deferred-K hooks work against rebased memory API |
| `ggml-cuda/CMakeLists.txt` | Modify (rebase) | Reconcile template instance list with mainline drift |
| `ggml-cuda/cpy-planar-iso.cu` | Modify (rebase) | API drift checks only |
| `ggml-cuda/set-rows-planar-iso.cuh` | Modify (rebase) | API drift checks only |
| `ggml-cuda/planar-iso-constants.cuh` | Modify (rebase) | API drift checks only |
| `ggml-cuda/fattn-common.cuh` | Modify (rebase) | API drift checks only |
| `tests/test-checkpoint-deferred-k.cpp` | Create | C++ unit test for snapshot × deferred-K × planar/iso |
| `tests/CMakeLists.txt` | Modify | Register new test |
| `src/models/dflash.cpp` | Cherry-pick | From #22105 |
| `src/models/eagle3.cpp` | Cherry-pick | From #22105 |
| `src/models/qwen35*.cpp` | Cherry-pick | From #22105 |
| `src/models/qwen3moe.cpp` | Cherry-pick | From #22105 |
| `src/llama-graph.{cpp,h}` | Cherry-pick | From #22105 |
| `src/llama-arch.{cpp,h}` | Cherry-pick | From #22105 |
| `common/speculative.cpp` | Cherry-pick | From #22105 |
| `examples/speculative-simple/speculative-simple.cpp` | Cherry-pick | From #22105 |
| `convert_hf_to_gguf.py` | Cherry-pick | From #22105 |
| `gguf-py/gguf/constants.py` | Cherry-pick | From #22105 |

### This repo (turbo)

| File | Action | Purpose |
|------|--------|---------|
| `docker/Dockerfile` | Modify | Pin `ROTORQUANT_COMMIT` to rebased + cherry-picked SHA |
| `docker/entrypoint.sh` | Modify | Add DFlash draft registry entries; `DRAFT_MODEL_NAME` and `DFLASH` env vars; build `--draft-model` flag |
| `docker-compose.yml` | Modify | Add `qwen36-dflash`, `qwen36-27b-dflash` profiles |
| `Makefile` | Modify | Add `run-qwen-dflash`, `run-qwen36-27b-dflash` targets |
| `scripts/benchmark_kv_regression.py` | Create | Wraps existing PPL sweep for L1 gate |
| `scripts/bench_snapshot_cost.py` | Create | Snapshot wall time vs context |
| `scripts/bench_speculative.py` | Create | Decode tok/s × backend × prompt × thinking-on/off |
| `scripts/validate_dflash.py` | Create | Differential harness vs z-lab pytorch reference |
| `tests/test_speculative_kv.py` | Create | Python integration: greedy target-only ≡ target+DFlash |
| `tests/test_dflash_e2e.py` | Create | End-to-end Docker profile test |
| `tests/test_dflash_zlab_diff.py` | Create | pytest wrapper around `validate_dflash.py` |
| `external/dflash-zlab/` | Create (submodule) | Pinned z-lab/dflash reference |
| `docs/BENCHMARK-REPORT.md` | Modify | Append §10 Speculative Decoding |
| `README.md` | Modify | Speculative decoding usage section |
| `docs/QUANTIZATION-GUIDE.md` | Modify | Draft-model VRAM additive cost note |
| `docs/sprints/SPRINT-004.md` | Create (at sprint close) | Final merged sprint doc |
| `docs/sprints/SPRINT-004-DEFERRED.md` | Create (at sprint close) | Deferred items |
| `docs/sprints/SPRINT-004-FOLLOWUPS.md` | Create (at sprint close) | Snapshot delta/COW, multi-slot, non-greedy |

## Definition of Done

### Hard gates (sprint fails if missed)

- [ ] **L1 — KV regression**: PPL for 4 KV × 2 corpora × 2 models matches
      `BENCHMARK-REPORT.md` §1 within ±0.05 PPL (16 data points)
- [ ] **Checkpoint × deferred-K bit-equality**:
      `tests/test-checkpoint-deferred-k.cpp` passes all 4 subtests
      (f16 staging, planar3 quant, iso3 quant, recurrent SSM state)
- [ ] **L2 — Speculative correctness**: greedy decode (`--temp 0 --top-k 1
      --seed 42`, 256 tokens) target-only ≡ target+DFlash on 3 prompts
      for both Qwen3.6-27B and Qwen3.6-35B-A3B (byte-identical token
      streams)
- [ ] **Decode speedup**: ≥1.5× on Qwen3.6-27B "Write a quicksort
      algorithm" thinking-off, planar3 KV, ctx=8K (revised gate; was
      2.0× in v0 intent — adjusted for hybrid reality per PR data on
      Qwen3.5-9B)
- [ ] **All 8 existing Docker profiles still launch and serve**:
      `qwen`, `qwen36-q3`, `qwen36-iq3`, `qwen36-27b`, `qwen36-27b-q3`,
      `qwen36-27b-iq3`, `reasoning`, `gemma` — health + 1 chat completion
      each, model cache preserved (no re-download)

### Soft gates (sprint succeeds with caveats)

- [ ] **L3 — z-lab differential**: first 64 greedy tokens match z-lab
      pytorch reference; acceptance rate within ±5pp on at least 3
      of 5 prompts. Failures documented but non-blocking.
- [ ] **MoE speedup**: Qwen3.6-35B-A3B at iso3 ≥1.0× (no regression);
      ≥1.2× ships without `EXPERIMENTAL=1`, 1.0–1.2× ships with it
- [ ] **Snapshot cost**: at 65K context, snapshot/verify wall time
      ≤50% of decode wall time; if exceeded, document and disable
      DFlash for ctx >65K via `entrypoint.sh` flag check
- [ ] **Validation harness reusable**: `validate_dflash.py` accepts
      `(target, draft)` GGUF pair as args without code changes

### Documentation

- [ ] `docs/BENCHMARK-REPORT.md` §10 contains: speedup table per model,
      acceptance rate, snapshot cost vs context, "EXPERIMENTAL"
      annotation where applicable
- [ ] `README.md` "Available Models" table includes DFlash rows with
      command and tier guidance
- [ ] `docs/QUANTIZATION-GUIDE.md` calls out additive draft-model VRAM
      cost; explicitly notes 16 GB tier excluded from speculative path
- [ ] `docs/sprints/SPRINT-004.md` final document; `SPRINT-004-DEFERRED.md`
      and `SPRINT-004-FOLLOWUPS.md` populated
- [ ] Sprint commit `Sync all docs with latest benchmarks and code state`
      analogue at sprint close — every measured number lands in docs

### Code quality

- [ ] All `pytest tests/` green
- [ ] All fork-level C++ tests green (`ctest` after rebase build)
- [ ] No new `--no-verify` commits; all commits use
      `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`
- [ ] `docker/Dockerfile` pinned to a specific fork SHA, not branch HEAD

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Rebase produces gnarly conflicts in `llama-context.cpp` (5 mo of mainline drift + #19493 + #22227) | High | Medium | Phase 1 maps conflict zones before touching code; reserve full Phase 2 budget for rebase alone; commit incrementally |
| Upstream snapshot interface doesn't recognize our quant types as snapshotable | Medium | High | Phase 3 unit test catches this immediately; if true, add opt-in dispatch table in `llama-kv-cache.cpp` exposing `type_size`/`blck_size` to checkpoint API |
| Snapshot is full-state and prohibitively expensive at long context | Medium | Medium | Cost benchmarked in Phase 3; document as DFlash usable @ ≤65K only; disable in `entrypoint.sh` for higher ctx |
| #22105 (DFlash) has API drift since last fetch — unmerged PR keeps moving | Medium | Medium | Pin to a specific PR commit SHA before cherry-pick; if PR breaks API, hold sprint at Phase 4 boundary and replan |
| Hybrid layer support in our fork is silently broken (we never built against hybrid) | Medium | High | Phase 1 audit; smoke test in Phase 2 after rebase; if broken, this becomes the long pole and DFlash work slides |
| Decode speedup falls below 1.5× even on dense Qwen3.6-27B | Medium | Medium | Per revised intent §6: 1.10× was the floor on Qwen3.5-9B coding-prompt → not impossible we land ≤1.5× on 27B too. DoD soft-fail: report measured number; document; do not block sprint close |
| MoE speedup falls below 1.0× | Medium | Low | Ship behind `EXPERIMENTAL=1`; document in QUANT-GUIDE; soft gate, not hard |
| z-lab pytorch reference drifts from PR #22105 implementation | Low | Medium | Pin both submodule SHA and PR commit SHA; fail loud on diff |
| Draft model VRAM pushes 24 GB tier into OOM territory | Medium | Medium | Profile draft VRAM in Phase 4; if blocks 24 GB tier, restrict DFlash profiles to 32 GB+ recommendation in QUANT-GUIDE |
| Pre-built DFlash GGUFs (`spiritbuun`, `lym00`) become unavailable on HuggingFace | Low | High | Mirror to a private bucket once verified; pin filenames in entrypoint registry |
| GPU contention with training jobs blocks Phase 2 PPL sweep (16 runs) | Medium | Low | Sweep is overnight-runnable; chunk by model; fall back to Qwen3.5-27B for early validation if Qwen3.6 GPU unavailable |
| Cherry-picking #22105 reintroduces conflicts with our Phase 2 resolutions in `llama-context.cpp` | High | Low | Expected; Phase 4 budgets resolution time; the Phase 3 test guards regression on the deferred-K path |

## Security Considerations

- Pre-built DFlash GGUFs are loaded from third-party HuggingFace repos
  (`spiritbuun`, `lym00`). Pin exact commit SHAs in the entrypoint
  registry; do not auto-update. Verify SHA-256 of downloaded blobs
  against a manifest committed to this repo.
- z-lab/dflash submodule is a third-party Python repo; pin to a specific
  commit; treat as untrusted code (run only inside the Docker container,
  never on the host).
- Speculative checkpointing introduces a new GPU memory state surface;
  ensure no cache contents are leaked across request slots if multi-slot
  is enabled in a future sprint (single-slot only here).
- No secrets are added to Dockerfile or compose; HF_TOKEN remains
  env-only.

## Dependencies

- **Sprint 002**: Docker Compose infrastructure, `llama-perplexity`
  binary, `entrypoint.sh` model registry, profile pattern.
- **Sprint 003**: Negative result on SpectralQuant; not load-bearing
  here, but confirms our path stays on rotation-based KV (planar/iso),
  which the rebased fork must preserve.
- **Upstream llama.cpp mainline**: PRs #19493 (merged), #22227 (merged).
  These are non-negotiable prerequisites — *their merge to mainline is
  the reason this sprint is feasible at all on Qwen3.6 hybrid models.*
- **Upstream PR #22105 (DFlash)**: BLOCKED on review, MERGEABLE.
  Cherry-pick is the correct choice (per revised intent §4) — waiting
  is unbounded; the underlying #19493+#22227 we need is already in
  master.
- **External Python**: `safetensors`, `torch`, `transformers` (already
  installed). No new package additions in this repo.
- **External submodule**: `z-lab/dflash` for L3 differential reference.
- **Hardware**: RTX 5090 (32 GB) for Phases 2–6 benchmarks; CPU is
  sufficient for unit tests and Phase 1 audit reading.
- **Pre-built draft GGUFs**:
  - `spiritbuun/Qwen3.6-27B-DFlash-GGUF` (Qwen3.6-27B draft)
  - `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test` (Qwen3.6-35B-A3B draft)

## Open Questions

These map 1:1 to the 8 open questions in `SPRINT-004-INTENT.md`. My
opinionated take on each:

1. **Phase boundary for "rebase complete"**:
   **Strict L1 gate before any speculative work.** Phase 2 cannot exit
   until the 16-point PPL sweep matches BENCHMARK-REPORT.md within
   ±0.05. The rebase brings hybrid checkpointing — if it broke our
   quant path, that *also* breaks DFlash. No Phase 3 work begins until
   the L1 gate passes. Non-negotiable.

2. **Speculative checkpointing × RotorQuant deferred-K interaction**:
   **The Phase 3 C++ unit test is the load-bearing artifact of this
   entire sprint.** My take on the sub-questions:
   - (a) Upstream's `checkpoint_save/restore` operates on opaque
     contiguous backend buffers (verified by reading #19493 diff).
     Our quant types are first-class `ggml_type` with known
     `type_size`/`blck_size`, so memcpy snapshot should "just work"
     — but "should" is not "does". The test exists exactly to
     turn that into "does".
   - (b) Cost: I expect upstream to be **full-snapshot per verify**,
     not delta. At 65K planar3 = 3.3 GB, that is 3.3 GB GPU memcpy
     per verify step, or ~50 GB across a 256-token decode. This is
     fine on PCIe 5.0 (≥30 GB/s) for ~1.5 s of overhead per 256
     tokens but **caps speedup at modest contexts**. Fine for v1.
     Delta/COW is a Sprint 005 follow-up.
   - (c) Verify mid-prefill is impossible by construction (speculative
     is post-prefill). Document and assert in `entrypoint.sh`.

3. **Draft model on which KV cache type**:
   **Draft inherits target's KV type.** PR #22105's draft loader
   uses `--cache-type-k`/`-v` from the parent CLI. The DFlash drafts
   are themselves hybrid (few layers initialized from target), so
   they also need checkpointing. Their KV is small (3.3 GB → ~30 MB
   for the 4-layer draft), so cost is negligible regardless. Same
   answer as v0 intent.

4. **Should we wait for upstream merge of #22105?**:
   **Cherry-pick now.** PR is BLOCKED on review with no merge ETA.
   The hybrid-state fallback we *need* (#19493 + #22227) is *already*
   in master and arrives via the rebase. Whether we also take
   #22105 (DFlash) on top is a distinct, lower-risk decision. We
   pin to a specific PR commit SHA so that further rebases by
   `ruixiang63` don't move our floor.

5. **Single-bench parallel slot interaction**:
   **Single slot for v1.** Snapshot/restore semantics across
   parallel slots are uncharted at upstream too. Defer multi-slot
   to a follow-up sprint. The throughput profiles (`qwen36-throughput`)
   stay on the autoregressive path — they explicitly need 8 slots
   and DFlash is single-slot anyway.

6. **MoE-vs-dense speedup expectations**:
   **Use Qwen3.5-9B as the hybrid floor, not Qwen3-8B as the optimistic ceiling.**
   PR data: Qwen3.5-9B w/o thinking ranges 1.10–2.77×. Hard gate
   on Qwen3.6-27B is **1.5×** (≈mid of that range, accounting for
   larger model = more decode time = better hidden snapshot cost).
   MoE soft-gates at 1.0× (no regression); 1.2× is a win. Ship
   MoE behind `EXPERIMENTAL=1` between 1.0× and 1.2×.

7. **What gets deferred from this sprint?**:
   - **OUT**: Block-aware `kv_cache_quantized_seq_rm()` (moot under
     checkpointing). Standalone `qwen36-spec` autoregressive profile
     (it tested seq_rm — no longer the right validation vehicle).
     Multi-slot speculative. Non-greedy speculative. Open WebUI.
     CI bench runner (D-013, carry-forward).
   - **IN**: Snapshot × deferred-K C++ test. Snapshot cost
     benchmark. Architecture audit at Phase 1 (we never built
     against hybrid before).

8. **Test coverage for hybrid speculative checkpointing × deferred-K**:
   **Both tests, C++ is load-bearing.**
   - **Fork-level C++** (`tests/test-checkpoint-deferred-k.cpp`):
     The primary regression guard. Catches snapshot bugs at the
     buffer level, before they masquerade as DFlash bugs higher
     up the stack. 4 subtests: f16 staging, planar3, iso3, SSM
     state.
   - **Repo-level Python integration** (`tests/test_dflash_e2e.py`):
     End-to-end greedy decode through Docker profile, with forced
     rejections (manipulate draft logits to induce rejection on
     ≥30% of tokens), assert target-only equivalence.
   - **Repo-level differential** (`tests/test_dflash_zlab_diff.py`):
     z-lab pytorch reference cross-check; soft gate (±5pp).
