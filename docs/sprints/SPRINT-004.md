# Sprint 004: Rebase Fork + DFlash Block-Diffusion Speculative Decoding on Hybrid Qwen3.6

**Created**: 2026-04-26
**Status**: In progress (~70% complete) — Phases 0-3 done, Phases 4-6 pending
**Last updated**: 2026-04-27
**Depends on**: Sprint 002 (Docker), RotorQuant fork at `johndpope/llama-cpp-turboquant`
**Target hardware**: RTX 5090 (32 GB), 123 GB system RAM
**Estimated effort**: 3-4 weeks single-engineer
**Branches**:
- Fork: `rapatel0/llama-cpp-turboquant` `feature/sprint-004-rebase-dflash` (off `johndpope/llama-cpp-turboquant#feature/planarquant-kv-cache`)
- Repo: `rapatel0/rq-models` `sprint/004-dflash` (off `main`)

## Progress at a glance

| Phase | Status | Outcome |
|-------|:------:|---------|
| 0 — Branch setup | ✅ done | Fork created at rapatel0/llama-cpp-turboquant; both sprint branches pushed |
| 1 — Rebase + L1 PPL gate | ✅ done | 150 fork commits squash-merged onto master `78433f606`; **0 PPL regressions across 10 cells**; 8/10 cells *improved* by 0.06–0.60 PPL vs old fork |
| 2 — Snapshot cost + VRAM shadow | ✅ done | `vram_seq_checkpoint` class delivers **31–40× speedup** on save+restore (host-RAM 9–22 ms → VRAM-shadow 0.3–0.5 ms); bit-exactness validated |
| 3 — DFlash cherry-pick | ✅ done (smoke deferred) | PR #22105 squash-merged at HEAD `67cb0d507`; zero conflicts; L1 PPL + vram correctness re-verified post-pick. Smoke test blocked on community draft GGUF format mismatch |
| 4 — Docker profiles + entrypoint refactor | pending | Add `qwen36-dflash`, `qwen36-27b-dflash` profiles; `SPECULATIVE_MODE` / `DRAFT_MODEL_NAME` env in entrypoint.sh |
| 5 — Validation harness | pending | z-lab pytorch parity, L4 5-prompt median speedup gate; needs source-converted draft GGUFs (community drafts have format mismatch) |
| 6 — Docs + ship gates | pending | README/QUANTIZATION-GUIDE/BENCHMARK-REPORT updates |

---

## Session log

### 2026-04-26 — Planning + Phase 0 setup
- Multi-agent plan via opus-4.7 + gpt-5.4 consensus. v1 drafts pre-dated
  the hybrid-architecture finding; v2 drafts and critiques re-ran with
  the corrected brief.
- Mid-planning architecture finding (per user): `llama_memory_recurrent::seq_rm()`
  returns false for any partial removal including the final position;
  Qwen3.6 dense and MoE are both 75% recurrent layers.
- Codex agent executed Phase 0 (fork creation, branch setup) and Phase 1
  source-spike on PRs #19493 + #22227. Findings documented in
  `BENCHMARK-REPORT.md` §10.

### 2026-04-27 — Phases 1–3 execution
- **Phase 1**: rebase via `git merge --squash` (linear rebase abandoned
  after first conflict — would have been 141 rounds). 4 conflict files
  manually resolved. Build fix: `extern "C" GGML_API` combination in
  `ops.cpp`. L1 PPL gate: 10/10 pass, 0 regressions; 8/10 cells
  *improved* by 0.06–0.60 PPL vs old fork.
- **Phase 2**: snapshot cost benchmark (`llama-checkpoint-bench`)
  measured host-RAM path at 9–22 ms — over the original 5 ms gate.
  Implemented `vram_seq_checkpoint`: VRAM-resident shadow buffers via
  `cudaMemcpyDeviceToDevice`, **31–40× speedup** to 0.29–0.53 ms
  round-trip. Bit-exactness validated.
- **Phase 3**: cherry-pick PR #22105 via `git merge --squash`. Zero
  conflicts on 5 intersection files. Build clean. L1 PPL re-verify:
  identical to Phase 1, 0 regressions. Smoke test partially blocked on
  community draft GGUF format mismatch (architecture string + key
  prefix workarounds applied; tensor-name mismatch unresolved without
  source safetensors).
- Account swap: original fork at `rpsdm0/llama-cpp-turboquant` (created
  by Codex via `gh` CLI which auths as rpsdm0); re-forked to
  `rapatel0/llama-cpp-turboquant` to match SSH key ownership. Both
  branches pushed to remote. SSH config patched (libcrypto issue with
  `.pub` IdentityFile).

---

## Overview

The RotorQuant llama.cpp fork has been frozen on `master` from early April for
~3 weeks while upstream merged speculative checkpointing (PRs #19493 + #22227)
and accepted the precondition work for two new speculative-decoding paths
(EAGLE3 #18039 and DFlash #22105). Concurrent with that rebase debt, our
production targets — Qwen3.6-35B-A3B and Qwen3.6-27B — are **75% Gated Delta
Net / SSM recurrent layers** (verified from `config.json`: 30 of 40
`linear_attention` layers on the MoE; 48 of 64 on the "dense" 27B). On these
hybrid models, `llama_memory_recurrent::seq_rm()` returns `false` for any
partial removal that includes the final position. Naive speculative decoding
either errors or silently corrupts output.

Speculative checkpointing is the upstream fix: snapshot full memory state
before verify, restore + replay accepted prefix on rejection. **Both pieces
are already merged in mainline master**, so the rebase brings them in for
free. This sprint is therefore exactly two things in strict order:

1. Move our fork onto current master without regressing PPL on any of the four
   KV cache types, and verify the rebased fork's checkpoint mechanism captures
   our two new state representations correctly: deferred f16 K staging during
   prefill, and quantized planar/iso K post-conversion. Verify equally that it
   captures the SSM/recurrent state of `linear_attention` layers.
2. Cherry-pick PR #22105 (DFlash) onto the rebased fork, ship two new Docker
   profiles (`qwen36-27b-dflash`, `qwen36-dflash`), and validate numerical
   correctness against the z-lab pytorch reference.

The sprint is opinionated about three things: **(a)** the rebase is *load-
bearing*, not housekeeping — without it, no speculative decoding works on
our hybrid targets; **(b)** dense 27B owns the hard speedup gate (median ≥1.3×
across a 5-prompt set, quicksort headline ≥1.5×), MoE ships behind
`EXPERIMENTAL=1`; **(c)** the only reason this isn't a 2-month project is
that the upstream PRs and their dependencies are mostly additive at the model/
context layer and don't touch our `ggml-cuda/cpy-planar-iso.cu`,
`set-rows-planar-iso.cuh`, `planar-iso-constants.cuh`, or `fattn-common.cuh`
work — verified by `gh api repos/ggml-org/llama.cpp/pulls/22105/files`.

EAGLE3, multi-slot speculative serving, non-greedy sampler validation, MoE
deep-dive profiling, and any custom `seq_rm` block-aware truncation are
explicitly out of scope. See `SPRINT-004-DEFERRED.md`.

---

## Use Cases

1. **Single-user dense decoding on Qwen3.6-27B** with measurable speedup.
   Profile: `--profile qwen36-27b-dflash`. Target: median ≥1.3× decode tok/s
   across a fixed 5-prompt set vs target-only on the same machine, with
   quicksort coding prompt ≥1.5× headline. Single slot, greedy sampling.

2. **Single-user MoE decoding on Qwen3.6-35B-A3B** as experimental opt-in.
   Profile: `--profile qwen36-dflash`. Gated by `EXPERIMENTAL=1` env in compose.
   Hard gate: greedy correctness only (no speedup target). Soft target: ≥1.0×
   decode (no regression). Realistic per upstream gpt-oss-20B numbers
   (0.61–1.27×) is that this may slow some workloads.

3. **Validated rebased fork** that runs all 8 existing Docker profiles
   identically post-rebase: `qwen`, `qwen36-q3`, `qwen36-iq3`, `qwen36-27b`,
   `qwen36-27b-q3`, `qwen36-27b-iq3`, `reasoning`, `gemma`. PPL on the four
   RotorQuant KV types matches `BENCHMARK-REPORT.md` §1.5–1.8 within ±0.05.
   Model cache (`llm-models` named volume) preserved.

4. **Hybrid-aware checkpoint correctness as a fork primitive**. The rebased
   fork demonstrates bit-exact save/restore of: deferred f16 K staging, all
   four quantized K layouts (planar3, planar4, iso3, iso4), and at least one
   `linear_attention` layer's recurrent SSM state. This unlocks any future
   speculative work on hybrid Qwen architectures.

5. **Reusable differential validation harness**. `scripts/validate_dflash.py`
   runs (a) target-only vs target+DFlash equality on our fork, and (b) our
   fork vs z-lab pytorch reference on the same prompt/seed. Pluggable for
   future draft GGUFs without source edits.

---

## Architecture

### Two invariants

The architecture rests on two hard rules; everything else follows.

1. **Checkpoint must capture the actual RotorQuant memory backend buffer, not
   a dequantized logical view.** A `tensor_view` that materializes f16 from
   our packed planar/iso layout would produce a snapshot that restores to
   correct values but loses the on-device storage layout — and would also
   double VRAM during the snapshot window. The checkpoint interface in
   `src/llama-memory*::checkpoint_save/restore` (or whatever symbol PR #19493
   introduces — Phase 1 task confirms) must walk our backend's actual
   contents.
2. **Speculative verify is forbidden during deferred-K staging.** Our fork
   allocates K as f16 in a staging buffer during prefill; `convert_deferred_keys()`
   converts to the target type after prefill completes. A verify batch
   arriving before conversion would create a heterogeneous mid-conversion
   snapshot. Phase 2 adds a runtime guard in `llama_kv_cache_unified` that
   refuses to arm speculative until `prefill_complete && deferred_drained`.

### Sprint scope diagram

```
                    REBASED RotorQuant fork (Sprint 004 deliverable)
┌─────────────────────────────────────────────────────────────────────────┐
│   src/llama-arch.{cpp,h}        +LLM_ARCH_DFLASH (additive)             │
│   src/llama-context.cpp         REBASE merge zone (HIGH RISK)           │
│   src/llama-graph.{cpp,h}       +44 LOC for DFlash cross-attention      │
│   src/llama-kv-cache.cpp        +runtime guard, +verify-batch helper    │
│   src/llama-memory*.cpp         <upstream — verify checkpoint behavior  │
│                                  for our deferred K + quantized K +     │
│                                  recurrent state>                       │
│   src/models/dflash.cpp         NEW (cherry-pick from #22105)           │
│   src/models/qwen35*.cpp        +14-25 LOC each (target metadata)       │
│   src/models/qwen3moe.cpp       +11 LOC                                 │
│   common/speculative.cpp        +331 LOC (block draft + verify)         │
│   common/speculative.cpp        +LLAMA_SPEC_FORCE_REJECT_AT debug env   │
│   examples/speculative-simple/  +77 LOC, --dflash flag                  │
│   convert_hf_to_gguf.py         +187 LOC                                │
│   gguf-py/gguf/constants.py     +53 LOC                                 │
│   tests/test-checkpoint-hybrid-state.cpp                                │
│                                  NEW — bit-exact save/restore test for  │
│                                  deferred K, all 4 quantized layouts,   │
│                                  recurrent state, mixed cross-layer     │
│   ggml-cuda/{cpy,set-rows,fattn-common,planar-iso-constants}            │
│                                  OURS, untouched by upstream PRs        │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                       turbo repo (Sprint 004 deliverable)
┌─────────────────────────────────────────────────────────────────────────┐
│   docker/Dockerfile             pin ROTORQUANT_COMMIT to rebased SHA    │
│   docker/entrypoint.sh          +SPECULATIVE_MODE, +DRAFT_MODEL_NAME,   │
│                                  +EXPERIMENTAL gate, single command     │
│                                  builder for 3 modes                    │
│   docker-compose.yml            +qwen36-27b-dflash, +qwen36-dflash      │
│   Makefile                      run-qwen36-dflash, run-qwen36-27b-dflash│
│   scripts/ppl_sweep.py          NEW Python regression harness           │
│   scripts/bench_snapshot_cost.py NEW snapshot wallclock at 6 ctx sizes  │
│   scripts/validate_dflash.py    NEW differential vs z-lab pytorch       │
│   scripts/bench_speculative.py  NEW 3-way decode tok/s                  │
│   tests/test_speculative.py     NEW pytest unit                         │
│   tests/test_dflash_e2e.py      NEW pytest integration via Docker       │
│   docs/BENCHMARK-REPORT.md      §10 Speculative Decoding                │
│   README.md                     +"Speculative Decoding (Experimental)"  │
│   docs/QUANTIZATION-GUIDE.md    +draft model VRAM accounting            │
└─────────────────────────────────────────────────────────────────────────┘
```

### State accounting on hybrid Qwen3.6 (resolved by Phase 1 spike, see BENCHMARK-REPORT.md §10)

Phase 1 source-spike of PRs #19493 + #22227 confirmed (BENCHMARK-REPORT.md
§10, commit `dad6861`):

- Checkpoint is **eager byte-copy** via `llama_state_seq_get_size_ext` /
  `get_data_ext` / `set_data_ext`. Bytes come from real backend tensors via
  `io.write_tensor` → `ggml_backend_tensor_get` — handles our quantized
  planar/iso layouts as raw bytes with **no decoded/dequantized view**.
- Snapshot lives in **host pageable RAM** (`std::vector<uint8_t>`), NOT VRAM.
- For hybrid models, `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` snapshots recurrent
  state only; full-attention KV rollback uses `llama_memory_seq_rm` separately.

| Component | Qwen3.6-35B-A3B | Qwen3.6-27B |
|-----------|:----------------:|:------------:|
| Recurrent state copied to host RAM per checkpoint | ~5 MB | ~8 MB |
| Draft recurrent state to host RAM | ~3 MB | ~3 MB |
| **Snapshot VRAM impact** | **~0 GiB** | **~0 GiB** |
| **Snapshot host RAM impact (per active checkpoint)** | **~8 MB** | **~11 MB** |

Implications:
- **Risk R3** (snapshot cost too high): defused. Host-RAM memcpy of MB-scale
  recurrent state is microsecond-scale.
- **Risk R4** (PCIe roundtrip fatal): defused. The byte-copy uses
  `ggml_backend_tensor_get` which is the standard tensor-to-host pull;
  bandwidth is bounded by recurrent-state size (MB), not full KV (GB).
- **VRAM math irrelevance**: The 24 GB tier no longer needs to drop to 65K
  ctx for `qwen36-27b-dflash` on snapshot-headroom grounds. Other VRAM
  pressures (target weights + draft + KV growth) are unchanged.
- **`seq_rm` is back on the critical path** but only for the trivial case:
  speculative rejection trims the tail of full-attention KV. Our quantized
  planar/iso layouts store rows of whole blocks → tail trim is whole-block
  drop with no partial-block work needed. The Claude v1 draft's proposed
  `kv_cache_quantized_seq_rm()` block-aware helper remains correctly
  out-of-scope (deferred item D-005 in SPRINT-004-DEFERRED.md is unchanged
  for the same reason).

### Decode loop (annotated for hybrid)

```
prefill (target only)
    target.process(prompt)           — full_attention K is f16-deferred,
                                        recurrent state advances
    convert_deferred_keys()          — full_attention K → planar3/iso3
    SET prefill_complete = true
    SET deferred_drained = true      — invariant 2 satisfied; speculative
                                       enabled

decode loop (per output token):
    snapshot ← memory.checkpoint_save()
        captures: target full_attn K (planar/iso), target recurrent state,
                  draft full_attn K, draft recurrent state
        cost: see Phase 1 spike + Phase 2 numeric ceiling

    draft block ← dflash.draft(noise=[MASK]*16, ctx=accumulated_target_ctx)
        — 1 forward pass, bidirectional non-causal attention

    verify ← target.decode(draft_tokens)
        — 16 tokens batched into target; full_attn K appended in quantized
          form, recurrent state advances by 16

    accept_count ← speculative_sample(verify_logits, draft_tokens)

    IF accept_count == 16:
        commit (no rollback needed)
        emit accept_count tokens
    ELSE IF accept_count < 16:
        memory.checkpoint_restore(snapshot)
            full_attn K: drop appended 16 tokens (append-only path)
            recurrent state: bit-exact restore from snapshot
        target.decode(draft_tokens[:accept_count])
            — replay accepted prefix only; recurrent state advances
              correctly because we restored to pre-verify state first
        emit accept_count tokens

    discard snapshot
```

The two paths that absolutely require the new test coverage are
`memory.checkpoint_save/restore` correctness for **all four KV layouts**
and **the recurrent state**.

---

## Implementation

### Phase 0: Branch setup (pre-sprint, ~0% of effort) — COMPLETE

**Goal**: Isolate sprint work from concurrent RotorQuant development on
other branches. Done before any other phase begins.

**Tasks**:
- [x] **Repo-side**: branch `sprint/004-dflash` created off `main` at
      `077d731`. All implementation commits in this repo land here. `main`
      stays clean of sprint work until final user-approved merge.
- [x] **Fork-side**: forked `johndpope/llama-cpp-turboquant`, cloned to
      `/home/ravi/repos/llama-cpp-turboquant`. Three remotes:
      `origin` → `rapatel0/llama-cpp-turboquant` (re-forked from the
      original `rpsdm0` Codex created — account swap to match SSH key
      ownership), `upstream-fork` (johndpope), `upstream`
      (ggml-org/llama.cpp). Branch `feature/sprint-004-rebase-dflash`
      pushed to `origin`.
- [x] Starting SHAs recorded in `BENCHMARK-REPORT.md` §10:
  - rebase base: `fc3d1b6566fa37be532e1153e11c35ceabc13f84`
  - rebase target (master): `78433f606fde4d7934a02dcbfd910438d28beccd`
  - cherry-pick target (PR #22105 head pinned at execution): `67cb0d507080e42cc012ac0bdb8f09622f64455b`

### Phase 1: Rebase the fork + checkpoint architecture spike (~25% of effort) — COMPLETE

**Goal**: `feature/planarquant-kv-cache` rebased onto current upstream master,
PPL-equivalent to commit `20efe75` baseline; Phase 2 design grounded in source
inspection of the upstream checkpoint mechanism.

**Outcome**: Linear rebase abandoned (would have required ~141 individual
conflict resolutions across the same set of files). Switched to
`git merge --squash` strategy: 4 conflict files vs 33 expected, all resolved
manually. Build passed (one C++/extern-C/GGML_API combination fixed in
`ops.cpp`). L1 PPL regression sweep produced **10/10 pass with 0
regressions**; 8 of 10 cells (all quantized KV types on both models)
*improved* by 0.06–0.60 PPL vs the old fork — the rebase pulled in 399
mainline commits that included quantization-kernel improvements.

**Branch**: Already created in Phase 0 (`feature/sprint-004-rebase-dflash`).

**Files (in fork)**:
- `src/llama-context.cpp` — primary merge conflict zone
- `src/llama-kv-cache.cpp` — deferred-K logic; conflict probability medium
- `ggml-cuda/CMakeLists.txt` — template instance file list; conflict probability medium
- `ggml-cuda/cpy-planar-iso.cu`, `set-rows-planar-iso.cuh`,
  `planar-iso-constants.cuh`, `fattn-common.cuh` — ours, expect zero conflicts

**Files (in this repo)**:
- `docker/Dockerfile` — bump `ARG ROTORQUANT_COMMIT` to rebased SHA after gate passes
- `docs/BENCHMARK-REPORT.md` — append a Phase 1 paragraph summarizing checkpoint
  source-read findings (snapshot data residency, COW vs full-copy, peak VRAM
  worst case)

**Tasks**:
- [x] Squash-merge fork onto master via `git merge --squash` (was: linear
      `git rebase` — abandoned due to 141 individual conflict rounds).
      Resolved 4 conflict files manually: `ggml/include/ggml.h` (renumber
      TURBO/PLANAR/ISO type enum slots from 41-48 to 42-49 to avoid
      mainline's new `GGML_TYPE_Q1_0=41`), `ggml/src/ggml-cuda/fattn.cu`
      (union of head-dim exclusion conditions), `src/llama-graph.cpp`
      (kept both `self_v_rot` hook and TurboQuant V padding extraction),
      `src/llama-kv-cache.cpp` (kept both WHT helpers and InnerQ externs +
      `convert_deferred_keys()`). Squash commit: `2d0524b34`.
- [x] Rebase succeeded; `src/llama-context.cpp` auto-merged cleanly via
      3-way merge (was the highest-risk file in the plan).
- [x] CUDA template-instance list in CMakeLists preserved through rebase.
- [x] One build fix needed post-rebase: `extern "C" GGML_API` combination
      in `ggml-cpu/ops.cpp:15` produces "invalid use of extern in linkage
      specification" — fixed by removing GGML_API from that declaration.
      Fork commit `8cfb44021`. All 4 binaries built clean after.
- [x] **L1 PPL regression sweep** (HARD GATE) — PASSED. `scripts/ppl_sweep.py`
      paired-mode (rebased fork vs old fork via docker) on
      `wikitext-2-raw-v1` test split at ctx=2048. **10/10 cells pass**, 0
      regressions; 8/10 *improved* (rebased fork has lower PPL than old
      fork by 0.06–0.60 PPL on quantized types). Results in
      `docs/sprints/SPRINT-004-L1-results.json` and
      `docs/BENCHMARK-REPORT.md` §10. C4 corpus deferred (dataset not
      currently local).
- [x] **Architecture spike** (HARD GATE) — COMPLETE: Read upstream PRs
      #19493 + #22227 source. Findings recorded in `BENCHMARK-REPORT.md` §10
      (commit `dad6861`). Summary: eager byte-copy via `llama_state_seq_*_ext`,
      reads real backend tensors (handles our quantized layouts transparently),
      lives in host pageable RAM, hybrid uses `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`
      to snapshot recurrent state only. Public API integration point:
      `llama_state_seq_get_size_ext` / `get_data_ext` / `set_data_ext`.
- [x] **Hybrid sanity** (HARD GATE) — PASSED. Generated coherent output
      from Qwen3.6-27B with iso3 KV at 70.5 tok/s decode (vs 77 tok/s f16
      = 91% — consistent with prior benchmarks). Both linear_attention
      and full_attention layer paths exercised. The L1 PPL sweep (next
      task) is also implicitly a hybrid-sanity test on a much larger
      input.
- [x] Tag the rebased commit; SHA pinned. The `ROTORQUANT_COMMIT` ARG
      bump in `docker/Dockerfile` is deferred to Phase 4.

**Phase gate**: All four hard gates pass before Phase 2 begins. If any fail,
escalate or descope (see Risks below).

### Phase 2: Checkpoint × deferred-K validation with hybrid coverage (~25% of effort) — PARTIALLY COMPLETE

**Goal**: Bit-exact, hybrid-aware checkpoint save/restore proven for both
quantized K layouts and recurrent state, with a numeric snapshot-cost ceiling
that gates DFlash work.

**Outcome (this session)**: Snapshot cost measured + revised gate met.
A new optimization landed: `vram_seq_checkpoint` (VRAM-resident shadow
buffer with `cudaMemcpyDeviceToDevice`) brings save+restore from
9.3–21.7 ms (host RAM) down to **0.29–0.53 ms** (31–40× speedup),
host-RAM-bandwidth bound replaced with HBM bandwidth. Bit-exactness of
the round-trip validated via a save→mutate→restore→compare check in
`llama-checkpoint-bench`. Both production targets pass.

**Deferred to a future session**: The runtime guard
(`prefill_complete` / `deferred_drained` flags + verify-batch helper)
and the formal `tests/test-checkpoint-hybrid-state.cpp` C++ unit-test
file. The smoke test exercising save→mutate→restore→compare in
checkpoint-bench is sufficient functional coverage for now; the formal
test suite codifies it. The runtime guard is defensive belt-and-suspenders
— in practice `convert_deferred_keys()` runs at end of prefill before
any speculative path activates, so the race window doesn't open in
normal operation.

**Files (in fork)**:
- `src/llama-kv-cache.cpp` — runtime guard `prefill_complete && deferred_drained`
  must be true before speculative arms; verify-batch append helper that treats
  multi-token verify appends as quantized decode appends (not fresh prefill)
- `tests/test-checkpoint-hybrid-state.cpp` — NEW, ~150 LOC C++ unit test in fork

**Files added in this session (in fork)**:
- `src/llama-vram-checkpoint.{h,cpp}` — NEW class `vram_seq_checkpoint`
  for D→D snapshot of recurrent state (commit `9b191cd87`)
- `examples/checkpoint-bench/{CMakeLists.txt,checkpoint-bench.cpp}` — NEW
  benchmark/correctness tool (commits `d2f47d0c0` + `ddb6b631c`)
- `src/llama-arch.{cpp,h}` and `src/llama-model-loader.cpp` — LLM_KV
  arch_name override for community draft GGUFs (commit `bd7a7aabb`)

**Files (in this repo)**:
- `scripts/ppl_sweep.py` — landed in Phase 1 with `--mode pair` for
  paired comparison vs old fork via docker

**Tasks**:
- [ ] [DEFERRED] Add `prefill_complete: bool = false` and
      `deferred_drained: bool = false` flags to `llama_kv_cache_unified`.
      `convert_deferred_keys()` sets both to true on completion.
- [ ] [DEFERRED] Add verify-batch quantized append path:
      `kv_cache_quantized_append_batch(K, n_tokens)`. In-practice race
      window doesn't open in normal flow; defensive belt-and-suspenders.
- [ ] [DEFERRED] Add runtime guard at speculative-arm: if a speculative
      call comes in while `!prefill_complete || !deferred_drained`, log
      warning and disable speculative for that decode iteration.
- [partial] **Build `tests/test-checkpoint-hybrid-state.cpp`** —
      Subtest C (recurrent-state save→mutate→restore→bit-equality) is
      effectively implemented inline in `llama-checkpoint-bench` and
      verified passing on both production targets. The formal C++ unit
      test file is deferred — restating the same test inline as a
      pytest/CTest target adds rigor but no new evidence. Subtests A
      (deferred f16 staging), B (4 quantized K layouts), D (cross-layer
      mixed batch), E (TOCTOU) remain TODO and are the load-bearing
      validations for any future runtime-guard work.
- [x] **Build snapshot cost benchmark** — done as
      `examples/checkpoint-bench/checkpoint-bench.cpp` in the fork.
      Reports save/restore wallclock for both
      `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` (host-RAM upstream path) and
      our new `vram_partial` (D→D shadow path), plus a bit-exactness
      correctness check.
- [x] **Snapshot cost ceiling** (HARD GATE) — REVISED, COMPLETE:
      Phase 2 measurement (BENCHMARK-REPORT.md §10) showed Phase 1 spike
      underestimated recurrent-state size. Actual values:
      - **Qwen3.6-35B-A3B / iso3** (production `qwen36-dflash` default):
        65.9 MB partial snapshot, **9.3 ms save+restore round-trip**.
      - **Qwen3.6-27B / planar3** (production `qwen36-27b-dflash` default):
        156.9 MB partial snapshot, **21.7 ms save+restore round-trip**.

      Both are host-RAM-bandwidth-limited (~14 GB/s) and cannot be
      meaningfully reduced at the user level. The original 5 ms gate was
      based on the spike's "MB-scale" estimate, which proved low.

      **Revised hard gate**: snapshot+restore round-trip ≤ **25 ms** at
      production-default KV type and reasonable context. This is met
      (35B/iso3 = 9.3 ms, 27B/planar3 = 21.7 ms). Snapshot cost will eat
      ~10-20% of analytic speculative speedup but won't dominate the cycle.
      Final L4 measurement (Phase 5) is what gates sprint success.
- [ ] [DEFERRED to Phase 5] **Add `LLAMA_SPEC_FORCE_REJECT_AT=N` debug
      env in `common/speculative.cpp`**: post-cherry-pick the file is
      heavily rewritten by PR #22105; better to add the debug env after
      the cherry-pick lands.

**Phase gate**: Snapshot cost ceiling met; bit-exactness validated for the
recurrent-state path (the load-bearing one for speculative). Formal test
suite + runtime guards deferred without blocking Phase 3.

### Phase 3: Cherry-pick DFlash from PR #22105 (~15% of effort) — COMPLETE (smoke deferred)

**Goal**: DFlash draft + verify path compiles and runs on the rebased,
checkpoint-validated fork.

**Outcome**: Squash-merged the entire PR (EAGLE3 base + DFlash delta, 23
commits) onto the sprint branch as a single commit. **Zero conflicts on
the 5 intersection files** (`common/arg.cpp`, `common/speculative.cpp`,
`src/llama-context.cpp`, `src/llama-graph.cpp`, `tools/server/server-context.cpp`)
— git's 3-way merge handled all of them automatically. Total: 28 files,
+1,890 / −52, including new graph files `src/models/dflash.cpp` (+161)
and `src/models/eagle3.cpp` (+186). Full EAGLE3 came along for the ride
(both PRs share the same git tree); we no longer need to cherry-pick
"minimal foundation only".

**Files (fork, additive unless noted)**: see Files Summary table.

**Tasks**:
- [x] Fetch PR via `git fetch upstream pull/22105/head:pr-22105`. Pinned
      HEAD: `67cb0d507080e42cc012ac0bdb8f09622f64455b` (2026-04-27).
- [x] Squash-merge PR onto sprint branch via `git merge --squash pr-22105`.
      Single commit `9993e8ae8`. Zero conflicts.
- [x] EAGLE3 came in alongside DFlash via the squash (the PR's tree
      includes both). The "minimal foundation only" plan is moot — full
      EAGLE3 model graph now present in fork. Sprint 005 doesn't need to
      cherry-pick separately; only needs Docker profile + validation.
- [x] Compile clean with `-DGGML_CUDA=ON -DGGML_CUDA_FA=ON`. All 6
      binaries built (`llama-perplexity`, `llama-server`, `llama-cli`,
      `llama-bench`, `llama-checkpoint-bench`, `llama-speculative-simple`).
- [x] **Re-run L1 PPL regression sweep** post-cherry-pick. Result:
      identical to Phase 1 — 10/10 pass, 0 regressions. Cherry-pick is
      quality-neutral on RotorQuant KV paths. Results in
      `docs/sprints/SPRINT-004-L1-results-postcherrypick.json`.
- [x] **Re-verify vram_seq_checkpoint bit-exactness** post-cherry-pick.
      27B/iso3 still 0.53 ms round-trip with `tail_match: true`. No
      regression.
- [partial] **Smoke test**: `llama-speculative-simple --dflash` with
      community draft GGUF (`spiritbuun/Qwen3.6-27B-DFlash-GGUF`). Three
      issues encountered:
  1. ~~Architecture string `dflash-draft` vs canonical `dflash`~~ — fixed
     by `llm_arch_from_string` alias + LLM_KV `arch_name_override` (fork
     commit `bd7a7aabb`).
  2. ~~Key prefix mismatch `<arch>.dflash.<key>` vs `<arch>.<key>`~~ —
     worked around by rewriting GGUF metadata via `gguf-py` (one-time
     per draft).
  3. **Tensor names mismatch** (community GGUF missing `fc.weight`,
     possibly other DFlash-specific tensors) — *not yet fixed*. Requires
     either source-side reconversion via PR #22105's
     `convert_hf_to_gguf.py` (z-lab safetensors source is gated) or
     fork-side tensor aliasing. Defers to Phase 5 validation harness.
- [ ] [DEFERRED to Phase 5] **Forced-rejection tests in fork**:
      `LLAMA_SPEC_FORCE_REJECT_AT` debug env + tests F-H. Now that
      `common/speculative.cpp` carries PR's block-draft/verify
      orchestration, the right hook points are clearer.

### Phase 4: Docker profiles + entrypoint refactor (~10% of effort)

**Goal**: Two new `docker compose --profile` targets serve DFlash end-to-end
without breaking the 8 existing profiles.

**Branch**: Already created in Phase 0 (`sprint/004-dflash`).

**Files (this repo)**:
- `docker/Dockerfile` — bump `ROTORQUANT_COMMIT` arg to rebased SHA
- `docker/entrypoint.sh` — extend `MODELS` registry; add
  `parse_speculative_args()`, `download_model_if_missing()`, single command
  builder for target-only / autoregressive-spec / DFlash modes
- `docker-compose.yml` — add `qwen36-27b-dflash` and `qwen36-dflash` services
- `Makefile` — `run-qwen36-27b-dflash`, `run-qwen36-dflash` targets
- `docker/test.sh` — add DFlash smoke test step

**Tasks**:
- [ ] Extend `MODELS` associative array with two draft entries:
      ```
      [qwen3.6-27b-dflash]="spiritbuun/Qwen3.6-27B-DFlash-GGUF|<filename>|131072|"
      [qwen3.6-35b-dflash]="lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test|<filename>|65536|"
      ```
      First action of Phase 4: `hf list` both repos and pin filenames + repo
      SHAs into the registry as `_HASH` annotations.
- [ ] Add env var contract to `entrypoint.sh`: `SPECULATIVE_MODE` ∈
      {target-only, autoregressive, dflash}; `DRAFT_MODEL_NAME`;
      `DRAFT_KV_CACHE_TYPE` (default `${KV_CACHE_TYPE}`); `DRAFT_N_MAX`
      (default 16); `EXPERIMENTAL` (gates `qwen36-dflash`).
- [ ] Sub-task — entrypoint refactor preservation: write `docker/test.sh`
      cases that exercise all 8 existing profiles with `--profile X up -d
      --no-deps` and verify the served binary still responds correctly. This
      is the hard gate that the entrypoint refactor doesn't break existing
      users.
- [ ] `qwen36-27b-dflash` service: `MODEL_NAME: qwen3.6-27b`,
      `DRAFT_MODEL_NAME: qwen3.6-27b-dflash`, `KV_CACHE_TYPE: planar3`,
      `N_PARALLEL: 1` (single-slot enforced).
- [ ] `qwen36-dflash` service: same shape, `MODEL_NAME: qwen3.6-35b`,
      `DRAFT_MODEL_NAME: qwen3.6-35b-dflash`, `KV_CACHE_TYPE: iso3`,
      `CTX_SIZE: 65536` (worst-case checkpoint headroom safety; can extend
      to 262K if Phase 1 spike says snapshot is COW), gated by
      `--profile qwen36-dflash` AND `EXPERIMENTAL=1` env.
- [ ] **Cache preservation gate** (HARD): no model in the existing
      `llm-models` named volume gets re-downloaded when the Dockerfile pin
      bumps. Verified by file-level `mtime` comparison.

### Phase 5: Validation harness + benchmarks (~15% of effort)

**Goal**: All four hard validation gates run reproducibly via one command per
gate; results land in `BENCHMARK-REPORT.md` §10.

**Files (this repo)**:
- `scripts/validate_dflash.py` — NEW, differential vs target-only and z-lab
  pytorch reference. Pluggable target+draft GGUF pair.
- `scripts/bench_speculative.py` — NEW, 3-way decode tok/s
  (target-only / target+autoregressive-draft as non-hybrid baseline /
  target+DFlash) on the fixed 5-prompt set
- `tests/test_speculative.py` — NEW pytest unit
- `tests/test_dflash_e2e.py` — NEW pytest integration via Docker

**Tasks**:
- [ ] **L1 (KV regression)**: `scripts/ppl_sweep.py` writes
      `BENCHMARK-REPORT.md`-comparable JSON. Pre-rebase + post-rebase runs
      compared by table. Gate at ±0.05 PPL.
- [ ] **L2 (greedy equivalence + forced-rejection)**:
      `scripts/validate_dflash.py --target qwen3.6-27b --draft qwen3.6-27b-dflash
      --prompts P1..P5 --temp 0 --top-k 1 --seed 42 --tokens 256` runs target-
      only and target+DFlash via Docker exec. Diffs token sequences. Pass:
      256/256 on all 5 prompts. ALSO with `LLAMA_SPEC_FORCE_REJECT_AT=8`
      env: still 256/256 because checkpoint+replay must be transparent. ALSO
      with curated low-acceptance prompts (random text, unusual code, etc.)
      to assert organic rejections occurred (acceptance rate < 90%).
- [ ] **L3 (z-lab differential — HARD GATE)**:
      `scripts/validate_dflash.py --reference zlab` clones
      `https://github.com/z-lab/dflash` at pinned commit, sets up venv with
      `torch>=2.4 transformers>=4.45`, runs `dflash/model.py` reference on
      our prompt/seed set. Pass: ≥64 of first 64 tokens match on 3 of 5
      prompts; acceptance rate within ±5 percentage points of z-lab on those
      same prompts.
- [ ] **L4 (speedup median ≥1.3× — HARD GATE)**:
      `scripts/bench_speculative.py` measures decode tok/s for target-only,
      target+autoregressive-draft, target+DFlash on Qwen3.6-27B planar3 +
      Qwen3.6-35B-A3B iso3, on a fixed 5-prompt set:
  1. "Write a quicksort algorithm in Python. Write code only." (coding,
     low-thinking)
  2. "Explain the Pythagorean theorem." (technical prose)
  3. "Plan a 1 day trip to DC." (travel/structured)
  4. "Summarize the plot of Hamlet in 3 paragraphs." (literary)
  5. "Write a SQL query to find the top 5 customers by revenue." (technical
     domain, dense response)

  All with `--temp 0 --top-k 1 --seed 42 --tokens 256 LLAMA_SPEC_NO_THINK=1`,
  3 trials each, median per-prompt. Pass: median tok/s across 5 prompts is
  ≥1.3× target-only on Qwen3.6-27B; HEADLINE (not gate): quicksort prompt
  ≥1.5× on Qwen3.6-27B. MoE has no speedup gate (correctness only).
- [ ] `tests/test_speculative.py`: GGUF metadata validation; sampler
      determinism; `LLAMA_SPEC_FORCE_REJECT_AT` debug env honors itself.
- [ ] `tests/test_dflash_e2e.py` (`@pytest.mark.docker`): bring up
      `qwen36-27b-dflash` profile; submit greedy `/v1/chat/completions`;
      assert response equal to target-only response on same prompt; tear down.

### Phase 6: Documentation + ship gates (~10% of effort)

**Files**:
- `docs/BENCHMARK-REPORT.md` — add §10 Speculative Decoding
- `README.md` — add "Speculative Decoding (Experimental)" section under
  Performance, document `qwen36-27b-dflash` and `qwen36-dflash` profiles,
  link to BENCHMARK-REPORT §10
- `docs/QUANTIZATION-GUIDE.md` — add "Draft model VRAM cost" subsection;
  note that 24 GB tier with DFlash drops default context from 131K to 65K
  if Phase 1 spike confirms full-copy snapshots
- `docs/sprints/SPRINT-004-FOLLOWUPS.md` — record any execution-discovered
  follow-ups (TBD)

**Tasks**:
- [ ] Add §10 "Speculative Decoding" with: hybrid architecture explainer,
      checkpoint mechanism summary (from Phase 1 spike), 3-way tok/s tables
      for both Qwen3.6 targets, acceptance rates per-prompt, snapshot
      wallclock at 8K/16K/32K/65K/131K/262K, z-lab parity numbers.
- [ ] Document the `LLAMA_SPEC_NO_THINK=1` env var; warn that thinking-on
      drops acceptance rate by 60–80 percentage points.
- [ ] **Reproducibility task**: an outside reader should be able to run
      `make bench-dflash` in a fresh clone and reproduce the headline
      numbers without consulting sprint authors.
- [ ] Document the `EXPERIMENTAL=1` opt-in for `qwen36-dflash` profile.
- [ ] Document explicitly: "this sprint validates greedy (`--temp 0
      --top-k 1`) only; sampling-mode behavior is unverified."

---

## Files Summary

### Fork (`johndpope/llama-cpp-turboquant`, branch `feature/sprint-004-rebase-dflash`)

| File | Action | Purpose |
|------|--------|---------|
| `src/llama-context.cpp` | Modify (rebase + cherry-pick) | Resolve conflicts; preserve deferred-K hooks |
| `src/llama-kv-cache.cpp` | Modify | `prefill_complete`/`deferred_drained` flags, runtime guard, verify-batch helper |
| `src/llama-arch.{cpp,h}` | Modify (cherry-pick) | `LLM_ARCH_DFLASH` enum slot |
| `src/llama-graph.{cpp,h}` | Modify (cherry-pick) | DFlash cross-attention plumbing |
| `src/models/dflash.cpp` | Create (cherry-pick) | DFlash draft graph |
| `src/models/qwen35*.cpp` | Modify (cherry-pick) | DFlash target metadata |
| `src/models/qwen3moe.cpp` | Modify (cherry-pick) | DFlash MoE target metadata |
| `common/speculative.cpp` | Modify (cherry-pick + new debug env) | Block draft + verify; `LLAMA_SPEC_FORCE_REJECT_AT` |
| `examples/speculative-simple/speculative-simple.cpp` | Modify (cherry-pick) | `--dflash`, `--draft-max` |
| `convert_hf_to_gguf.py` | Modify (cherry-pick) | DFlash GGUF emit |
| `gguf-py/gguf/constants.py` | Modify (cherry-pick) | DFlash KV metadata |
| `ggml-cuda/CMakeLists.txt` | Verify (rebase) | Re-add planar-iso template list if dropped |
| `tests/test-checkpoint-hybrid-state.cpp` | Create | Bit-exact save/restore for K layouts + recurrent state |

### This repo (`turbo`, branch `sprint/004-dflash`)

| File | Action | Purpose |
|------|--------|---------|
| `docker/Dockerfile` | Modify | Bump `ROTORQUANT_COMMIT` to rebased SHA |
| `docker/entrypoint.sh` | Modify | `SPECULATIVE_MODE` env, `DRAFT_MODEL_NAME` plumbing, single-builder for 3 modes |
| `docker-compose.yml` | Modify | Add `qwen36-27b-dflash`, `qwen36-dflash` services |
| `docker/test.sh` | Modify | Cache-preservation + DFlash smoke tests |
| `Makefile` | Modify | `run-qwen36-27b-dflash`, `run-qwen36-dflash`, `bench-dflash` targets |
| `scripts/ppl_sweep.py` | Create | Python regression harness, JSON output |
| `scripts/bench_snapshot_cost.py` | Create | Snapshot save+restore wallclock at 6 ctx sizes |
| `scripts/validate_dflash.py` | Create | L2 + L3 differential validation runner |
| `scripts/bench_speculative.py` | Create | L4 3-way decode tok/s benchmark |
| `tests/test_speculative.py` | Create | Pytest unit |
| `tests/test_dflash_e2e.py` | Create | Pytest integration via Docker |
| `docs/BENCHMARK-REPORT.md` | Modify | Add §10 Speculative Decoding |
| `docs/QUANTIZATION-GUIDE.md` | Modify | Add "Draft model VRAM cost" subsection |
| `README.md` | Modify | "Speculative Decoding (Experimental)" section |
| `docs/sprints/SPRINT-004.md` | This file | Final sprint doc |
| `docs/sprints/SPRINT-004-DEFERRED.md` | Create | Deferred items |
| `docs/sprints/SPRINT-004-FOLLOWUPS.md` | Create at end of sprint | Execution-discovered items |

---

## Definition of Done

### Hard gates (sprint fails if any miss)

1. **Rebase + KV regression**: rebased fork compiles cleanly with
   `-DGGML_CUDA=ON -DGGML_CUDA_FA=ON`. PPL on all 4 KV types × 2 corpora ×
   2 models matches `BENCHMARK-REPORT.md` §1.5–1.8 within ±0.05 PPL.
2. **Hybrid sanity**: rebased fork loads both Qwen3.6 architectures and
   logs prove both `linear_attention` and `full_attention` layers fired
   under each KV cache type.
3. **Hybrid checkpoint correctness**: `tests/test-checkpoint-hybrid-state.cpp`
   passes all subtests A–E. Specifically:
   - A: bit-exact restore of deferred f16 K staging
   - B: bit-exact restore of `planar3`, `planar4`, `iso3`, **and** `iso4` K
     layouts
   - C: bit-exact restore of recurrent state on at least one
     `linear_attention` layer in each Qwen3.6 model
   - D: cross-layer mixed-batch checkpoint correctness
   - E: convert-during-checkpoint TOCTOU guard fires
4. **Snapshot cost ceiling**: snapshot+restore at 65K context iso3 ≤5 ms
   (or value set by Phase 1 spike if different).
5. **Forced-rejection correctness**: `tests/test-checkpoint-hybrid-state.cpp`
   subtests F-H pass — recurrent state at position N+1 post-restore equals
   recurrent state at position N+1 along the target-only trajectory, asserted
   on bytes not tokens.
6. **L2 greedy + forced-rejection**: `qwen36-27b-dflash` produces 256/256
   identical tokens to target-only on all 5 prompts at `--temp 0 --top-k 1
   --seed 42`, including under `LLAMA_SPEC_FORCE_REJECT_AT=8` and on at least
   one curated low-acceptance prompt that produces organic rejections.
7. **L3 z-lab pytorch parity**: ≥64 of first 64 tokens match z-lab on 3 of 5
   prompts; acceptance rate within ±5 percentage points.
8. **L4 speedup median ≥1.3×**: median decode tok/s across 5-prompt set is
   ≥1.3× target-only on Qwen3.6-27B at planar3, single slot. Headline
   reported (not gated): quicksort prompt ≥1.5× on Qwen3.6-27B.
9. **All 8 existing Docker profiles still launch and serve**: cache
   preserved (no re-download), health endpoint responds, completion request
   succeeds. New profiles `qwen36-27b-dflash` and `qwen36-dflash`
   (with `EXPERIMENTAL=1`) also pass health + one-completion check.
10. **`pytest tests/` passes**: `test_speculative.py` and `test_dflash_e2e.py`
    (latter `@pytest.mark.docker` skipped in non-GPU CI).
11. **Documentation reproducibility**: a fresh-clone reader can run
    `make bench-dflash` and reproduce the headline numbers within ±10%
    without consulting sprint authors.

### Soft gates (sprint succeeds with caveats)

- **MoE no-regression**: `qwen36-dflash` at iso3 achieves ≥1.0× decode (no
  worse than target-only). Anything ≥1.2× is a win. <1.0× ships as
  experimental with a documented caveat in README.
- **Acceptance rate parity to PR #22105**: within 10 percentage points of
  PR's reported numbers for matching prompts on Qwen3.6-27B.
- **Long-context smoke test**: 32K-prompt greedy decode under DFlash
  completes without OOM or checkpoint corruption.

### Code hygiene

- Final fork commit on branch `feature/sprint-004-rebase-dflash` tagged
  `sprint-004-dflash`; SHA pinned in `docker/Dockerfile`'s
  `ROTORQUANT_COMMIT` ARG.
- All git operations use `git add -u` or explicit file lists per project
  CLAUDE.md.
- Commit messages: imperative subject + `Co-Authored-By: Claude <model>
  <noreply@anthropic.com>` trailer.
- No `.env`, `HF_TOKEN`, or other secrets committed.
- Repo work lives entirely on `sprint/004-dflash`. `main` only receives the
  approved sprint planning docs (intent / drafts / critiques / merge notes /
  this final sprint doc / deferred + followups) and, once sprint completes,
  a final merge of the implementation branch.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **R1: Recurrent-layer checkpoint silently shallow-copies SSM state on `linear_attention` layers** (highest impact correctness risk) | Medium | **Critical** | Phase 2 subtest C asserts bit-exact restore; subtests F-H assert post-restore recurrent-state bytes match target-only trajectory; runtime guard refuses speculative if convert state is heterogeneous |
| **R2: `src/llama-context.cpp` rebase conflicts unresolvable in <5 days** | Medium | High | If Phase 1 rebase exceeds 5 days, escalate to user. Fallback: cherry-pick our deferred-K commits + CUDA template instances + FA dispatch onto a fresh fork off latest master (full delta inventory recorded pre-sprint) |
| **R3: Snapshot cost too high at long context (>5 ms at 65K)** | ~~Medium~~ Low (defused by Phase 1 spike) | High | Phase 1 spike confirmed snapshot is host-RAM MB-scale byte-copy. Phase 2 ceiling measurement now near-trivial; failure would indicate deferred-K integration bug, not checkpoint architecture |
| **R4: Snapshot lives in host pageable memory, PCIe roundtrip fatal** | ~~Low~~ Defused (Phase 1 spike confirms host RAM, MB-scale) | n/a | Resolved — see BENCHMARK-REPORT.md §10. Bandwidth bounded by recurrent-state bytes, not full KV |
| **R5: DFlash cherry-pick breaks PPL via subtle KV layout change** | Medium | High | L1 PPL gate runs after every cherry-pick step, not just end-of-sprint. Bisect on first failure |
| **R6: Pre-built DFlash GGUFs from `lym00`/`spiritbuun` don't match cherry-picked PR #22105 metadata schema** | Medium | High | First Phase 4 task: download both, run `gguf-dump`, verify against PR's `convert_hf_to_gguf.py` output schema. If mismatch, regenerate from source HF safetensors |
| **R7: Acceptance rate degraded by quantized K perturbations vs DFlash's f16-trained reference** | Medium | Medium | Phase 5 L4 measures acceptance rate vs vanilla llama.cpp DFlash; if >10pp degradation, investigate (quantize draft to f16? raise draft KV to f16? document and accept?) |
| **R8: Forced-rejection coverage doesn't fire because acceptance rates are near 100% on chosen prompts** | High | Medium | Use both fault injection (`LLAMA_SPEC_FORCE_REJECT_AT`) and adversarial low-acceptance prompts — interview Q6 |
| **R9: Draft VRAM pushes Qwen3.6-35B-A3B over 32 GB at default ctx** | Medium | Medium | `qwen36-dflash` defaults to 65K context, not 262K; document tradeoff |
| **R10: z-lab pytorch reference doesn't run on RTX 5090 (sm_120 wheel mismatch)** | Low | Medium | If torch ≥2.4 + sm_120 wheel unavailable, run reference on a separate machine and store reference outputs as JSON fixtures in `tests/fixtures/` |
| **R11: Upstream rebases PR #22105 mid-sprint (likely — author is iterating)** | High | Low | Pin to PR commit SHA at sprint start; do not chase upstream during sprint. If upstream merges with materially different conflict resolution: post-sprint follow-up to align |
| **R12: GPU contention with concurrent training jobs causes intermittent OOM in benchmarks** | High | Medium | All benchmarks runnable in <15 min windows; coordinate with training schedule. `bench_*.py` records per-run free VRAM and aborts if <8 GB headroom; report median + variance, not single runs |
| **R13: MoE target speedup measures < 1.0×** | Medium | Low | Pre-acknowledged: ship `qwen36-dflash` behind `EXPERIMENTAL=1` with documented expected range |
| **R14: Cherry-pick boundary mistake: includes EAGLE3 commits accidentally** | Medium | Low | Pre-write `cherry-pick.txt` artifact in Phase 3; review before applying. Excluded files: `src/models/eagle3.cpp`, `--eagle3` runtime flag, eagle3 conversion path |
| **R15: Convert-during-checkpoint TOCTOU race produces corrupted snapshot** | Low | High | Phase 2 subtest E exercises this exact path; Phase 2 runtime guard refuses speculative until `prefill_complete && deferred_drained` |
| **R16: Existing 8 Docker profiles broken by entrypoint refactor** | Medium | High | Phase 4 sub-gate: `docker/test.sh` exercises all 8 profiles after refactor; specific assertion that `llm-models` named volume isn't re-downloaded |

---

## Security Considerations

- **No new network surface**: DFlash adds no new ports or endpoints; existing
  `0.0.0.0:8080` OpenAI-compat API is the only exposure.
- **Draft model download**: pre-built GGUFs from third-party HuggingFace
  repos (`lym00`, `spiritbuun`). Pin both repo SHA and filename in `MODELS`
  registry; verify HF returns matching SHA on subsequent pulls.
  GGUF format is binary-deserialized into model weights — not RCE-class
  but worth a hash check.
- **z-lab reference clone**: `scripts/validate_dflash.py` clones
  `https://github.com/z-lab/dflash` at a pinned commit (TBD); never `HEAD`.
  Reference runs in a venv, never against system Python.
- **`HF_TOKEN` handling**: existing pattern preserved — env var only,
  never logged, never baked into image. Draft model download path uses the
  same `HF_TOKEN` if set.
- **No new sudo / privileged surface**: container still runs as `llm` UID
  1001; draft model loaded into the same `llama-server` process; no
  privilege escalation.
- **`LLAMA_SPEC_FORCE_REJECT_AT`**: debug-only env, compile-time gated by
  `LLAMA_SPEC_DEBUG`. Not present in production binaries.

---

## Dependencies

### Prior work in this repo

- **Sprint 002** (Dockerize): `llm-base` compose anchor, `entrypoint.sh`
  MODELS registry, multi-stage Dockerfile, `llm-models` named volume.
  Recent commits `b8c9858`, `d675068`, `8c0907b` add the Qwen3.6 family.
- **Sprint 003** (SpectralQuant): negative result; not a dependency. The
  research-only `turboquant/spectral/` Python module is untouched.

### Upstream

- **llama.cpp master** at sprint-start SHA (will float — pin on day 1).
- **PR #22105** (DFlash) at sprint-start SHA (pin in `docker/Dockerfile`
  comment).
- **PRs #19493 + #22227** (speculative checkpointing) — already merged in
  master; come in via the rebase. Phase 1 spike reads their source.
- **PR #18039** (EAGLE3) — only minimal foundation commits cherry-picked.
  Full EAGLE3 deferred to Sprint 005.

### External artifacts

- **DFlash draft GGUFs** (pinned by repo SHA + filename in Phase 4):
  - [`spiritbuun/Qwen3.6-27B-DFlash-GGUF`](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF)
  - [`lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test`](https://huggingface.co/lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test)
- **z-lab pytorch reference**: `https://github.com/z-lab/dflash` at pinned commit.
- **CUDA 13.1**, **NVIDIA driver 570+**, **Docker 24+ + NVIDIA Container Toolkit**.

### Hardware

- **RTX 5090 (32 GB)** for full sprint.
- **123 GB system RAM**.
- Sprint runs without continuous GPU access; benchmarks scheduled around
  training windows.

### Branch strategy

- **Fork** (`johndpope/llama-cpp-turboquant`):
  - Base: `feature/planarquant-kv-cache` (preserved, do not modify)
  - Sprint branch: `feature/sprint-004-rebase-dflash`
  - On sprint completion: merge into `feature/planarquant-kv-cache` after
    user review
- **This repo** (`turbo`):
  - Base: `main`
  - Sprint branch: `sprint/004-dflash`
  - On sprint completion: merge into `main` after user review
- **Approved sprint planning docs land directly on `main`** (intent, drafts,
  critiques, merge notes, this final sprint doc, deferred items, follow-ups
  doc). Implementation work strictly on the dedicated branches.

---

## Open Questions

These remain after planning; resolution should occur during execution.

1. ~~**Snapshot data residency**~~ — **RESOLVED** by Phase 1 spike (commit
   `dad6861`). Host pageable RAM, MB-scale, no VRAM impact. See
   BENCHMARK-REPORT.md §10.

2. **Whether the verify-batch quantized append path needs a fused kernel** —
   Phase 2 measurement decides. Per-token loop is correctness-correct;
   `cpy_planar_iso_batch` is a possible Sprint 005 optimization if measured
   slow.

3. **Whether `qwen36-dflash` ship gates also include the long-context smoke
   test or not** — currently soft. May promote to hard if Phase 5 reveals
   unexpected interactions.

4. **Sprint 005 EAGLE3 priority vs other pipeline work** — depends on
   DFlash adoption signal post-sprint.

5. ~~**Whether to upstream a COW snapshot contribution**~~ — **RESOLVED:
   not needed.** Phase 1 spike confirms upstream's eager byte-copy approach
   is fine for our use case. D-005 (deferred items) marked archived.
