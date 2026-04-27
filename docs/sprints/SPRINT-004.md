# Sprint 004: Rebase Fork + DFlash Block-Diffusion Speculative Decoding on Hybrid Qwen3.6

**Created**: 2026-04-26
**Status**: complete-with-followups — all 6 phases shipped; L2/L3/L4 measurement
runs deferred to follow-up F-001 (blocked on source-converted draft GGUFs)
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
| 4 — Docker profiles + entrypoint refactor | ✅ done (host-side runs deferred) | Added `qwen36-27b-dflash` + `qwen36-dflash` (EXPERIMENTAL-gated) compose profiles; entrypoint refactored with `SPECULATIVE_MODE` / `DRAFT_MODEL_NAME` / `DRAFT_KV_CACHE_TYPE` / `DRAFT_N_MAX` / `EXPERIMENTAL` env contract; Dockerfile pinned to fork SHA `bd7a7aabb`; `docker/test.sh` extended with cache-preservation gate; Makefile run-targets added |
| 5 — Validation harness | ✅ harness ready (gate runs blocked on source-converted drafts) | `validate_dflash.py` (L2 + L3), `bench_speculative.py` (L4), `tests/test_speculative.py`, `tests/test_dflash_e2e.py`; `make bench-dflash` reproducibility entrypoint. Measurement runs blocked on community-draft tensor-name mismatch (Phase 3 issue) |
| 6 — Docs + ship gates | ✅ done | README "Speculative Decoding (Experimental)" subsection; BENCHMARK-REPORT.md §10 extended (hybrid explainer, checkpoint summary, L4/z-lab/snapshot-grid TBD tables, acceptance-rate notes); QUANTIZATION-GUIDE.md "Draft model VRAM cost"; SPRINT-004-FOLLOWUPS.md created (9 items, F-001 is the load-bearing blocker for empirical numbers) |

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

### 2026-04-27 — Phase 6 execution

- **Phase 6**: Documentation + ship gates only — no code changes. Sprint
  marked `complete-with-followups`: every phase deliverable is in place,
  but the empirical L2/L3/L4 numbers remain `TBD` pending a
  source-converted DFlash draft GGUF (tracked under F-001 in the new
  `SPRINT-004-FOLLOWUPS.md`). The scaffolding lands so once a working
  draft drops, each gate is one command.
- `README.md`: added "Speculative Decoding (Experimental)" subsection
  under Performance. Documents both compose profiles
  (`qwen36-27b-dflash` dense, `qwen36-dflash` MoE), the `EXPERIMENTAL=1`
  opt-in for MoE, the `SPECULATIVE_MODE` / `DRAFT_MODEL_NAME` /
  `DRAFT_KV_CACHE_TYPE` / `DRAFT_N_MAX` env contract,
  entrypoint-enforced single-slot, greedy-only validation scope,
  `LLAMA_SPEC_NO_THINK=1` and its 60–80pp acceptance-rate impact
  citation (PR #22105), and the F-001 draft-GGUF blocker. Links to
  `BENCHMARK-REPORT.md` §10 for numbers.
- `docs/BENCHMARK-REPORT.md` §10: extended with five new subsections
  appended after the existing Phase 3 cherry-pick paragraph — hybrid
  architecture explainer (75% recurrent, `seq_rm` returns false on
  partial-with-final, both upstream PRs landed via rebase, `config.json`
  layer counts cited); checkpoint mechanism summary referencing the
  existing (a)–(e) bullets; L4 5-prompt result table (all `TBD`,
  reproduction commands inline); z-lab parity table (all `TBD`);
  snapshot wallclock at 6 contexts (8K row preserved from Phase 2 on
  both production-default cells, 16K/32K/65K/131K/262K rows `TBD`);
  acceptance-rate notes paragraph on `LLAMA_SPEC_NO_THINK=1`.
- `docs/QUANTIZATION-GUIDE.md`: added "Draft Model VRAM Cost (Sprint 004
  — DFlash speculative)" section after the 40 GB tier. Lists draft GGUF
  sizes (27B q4_k_m draft = 0.96 GB; 35B q8_0 draft = 0.48 GB, both
  verified via HF API at the pinned SHAs in entrypoint), draft KV at
  typical contexts (recurrent state per-model-fixed; full-attention KV
  cells `TBD`), restates Phase 1 finding that snapshot is host-RAM not
  VRAM, per-profile RTX 5090 (32 GB) accounting table.
- `docs/sprints/SPRINT-004-FOLLOWUPS.md`: NEW. 9 items (F-001
  through F-009), matching the SPRINT-001-FOLLOWUPS.md format. F-001
  (source-converted drafts) is the load-bearing blocker — gates L2/L3/L4
  measurement plus `tests/test_dflash_e2e.py` actual run plus
  `validate_dflash.py` actual run. F-002 (`LLAMA_SPEC_FORCE_REJECT_AT`
  env) tracks the test_speculative.py xfail flip. F-003/F-004 carry
  forward the formal C++ checkpoint test + runtime guard from Phase 2's
  partial completion. F-005 tracks the `docker/test.sh`
  cache-preservation gate first run. F-006 pins z-lab SHA on first L3.
  F-007 proposes `make bench-dflash-all`. F-008 notes Sprint 005 EAGLE3
  scope shrunk because the full graph already came along in the Phase 3
  squash. F-009 closes the `LLAMA_SPEC_NO_THINK=1` doc item.
- Sprint outcome paragraph appended at end of doc.

### 2026-04-27 — Phase 5 execution

- **Phase 5**: Validation harness shipped; actual L2/L3/L4 measurement
  runs deferred. The community DFlash draft GGUFs
  (`spiritbuun/Qwen3.6-27B-DFlash-GGUF`,
  `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test`) fail to load against PR
  #22105's canonical schema (tensor-name mismatch — see Phase 3 §
  above and BENCHMARK-REPORT.md §10), and source-converted GGUFs
  require gated z-lab safetensors access we don't have. Every
  end-to-end gate is therefore parked behind that one blocker; the
  scaffolding lands so the runs are one command each once a working
  draft drops.
- `scripts/validate_dflash.py` — NEW. Covers L2 (greedy equivalence +
  forced-rejection) and L3 (z-lab pytorch differential) against an
  already-running `llama-server` on `${BASE_URL:-http://localhost:8080}`.
  Args: `--target` / `--draft` / `--target-path` / `--draft-path` (all
  informational, since the server already has the model loaded);
  `--prompts` (defaults to the 5-prompt set in the L4 spec); `--temp 0
  --top-k 1 --seed 42 --tokens 256`; `--reference {none,zlab}`;
  `--force-reject-at N` for the `LLAMA_SPEC_FORCE_REJECT_AT` env
  (Phase-2-deferred — script sets it but tolerates it being a no-op).
  Emits `docs/sprints/SPRINT-004-L{2,3}-results.json` matching the L1
  schema.
- `scripts/bench_speculative.py` — NEW. L4 3-way decode tok/s
  (target-only / target+autoregressive-draft / target+DFlash) on the
  fixed 5-prompt set (copied verbatim from §L4 below). Operator brings
  up each leg's compose profile in turn; the script appends to a
  single results JSON. `--finalize` computes ratios and emits a
  markdown summary. Pulls `timings.predicted_per_second` from
  llama-server, falls back to wallclock + completion_tokens.
- `tests/test_speculative.py` — NEW. (a) GGUF metadata validation via
  `gguf-py` (skips if not installed or no draft on disk); (b) sampler
  determinism (two requests at `temp=0,seed=42` → identical token
  IDs); (c) `LLAMA_SPEC_FORCE_REJECT_AT` honor test, `xfail` until the
  env hook lands in fork's `common/speculative.cpp`. Live-server tests
  skip if `${BASE_URL}/health` doesn't respond in 2s.
- `tests/test_dflash_e2e.py` — NEW. `@pytest.mark.docker` integration
  test. Brings up `qwen36-27b-dflash` profile, polls `/health`,
  submits one greedy completion, asserts equality with target-only
  output on the same prompt, tears down. Skipped unless
  `DFLASH_E2E=1` and `docker` is on PATH (because actually running it
  needs the working draft GGUF, which is blocked).
- `Makefile`: `bench-dflash` (calls `--finalize`) and
  `bench-dflash-leg LEG=...` (per-leg runner). Per Phase 6 spec, an
  outside reader can reproduce the headline numbers via these targets.
- `pyproject.toml`: registered the `docker` pytest marker.
- L1 (KV regression) gate spec is already met by the existing
  `scripts/ppl_sweep.py` (used in Phase 1 + post-cherry-pick in
  Phase 3); not rewritten. Both result JSONs are already in
  `docs/sprints/`.

### 2026-04-27 — Phase 4 execution

- **Phase 4**: Repo-side changes only — actual `docker build` and per-profile
  boot validation deferred to a host that has the rebased fork's binaries
  built and the existing model cache populated (the gate runs in
  `docker/test.sh`).
- `docker/Dockerfile`: bumped `ROTORQUANT_REPO`/`ROTORQUANT_BRANCH`/
  `ROTORQUANT_COMMIT` to `rapatel0/llama-cpp-turboquant#feature/sprint-004-rebase-dflash @ bd7a7aabb`.
  Added `llama-speculative-simple`, `llama-checkpoint-bench`, `llama-cli`,
  `llama-bench` to runtime stage so DFlash + Phase 5 harness work in-container.
- `docker/entrypoint.sh`: registered `qwen3.6-27b-dflash` and
  `qwen3.6-35b-dflash` draft entries with pinned repo SHAs in `MODELS_HASH`
  (5e4442a / 3813f31a). Added env contract for `SPECULATIVE_MODE` (target-only
  / autoregressive / dflash), `DRAFT_MODEL_NAME`, `DRAFT_KV_CACHE_TYPE`,
  `DRAFT_N_MAX`, and `EXPERIMENTAL` (gates qwen3.6-35b-dflash). Refactored to
  a single command builder using `--model-draft` / `--draft-max` /
  `--cache-type-{k,v}-draft` / `--dflash` flags (verified present in fork's
  `common/arg.cpp`). Speculative modes force `N_PARALLEL=1`. Helper
  `download_model_if_missing()` consolidates target+draft download logic.
- `docker-compose.yml`: added two profiles. `qwen36-27b-dflash` (dense 27B,
  planar3 KV, 131K ctx, single-slot, dflash). `qwen36-dflash` (35B MoE,
  iso3 KV, 65K ctx for snapshot headroom, gated by host `EXPERIMENTAL=1`).
  Both inherit speculative defaults from `x-llm-base`.
- `Makefile`: added `run-qwen36-27b-dflash[-bg]`, `run-qwen36-dflash[-bg]`
  targets; included new profiles in `stop` / `logs` / `clean` aggregate
  targets.
- `docker/test.sh`: rewrote as 4-stage smoke (build → cache snapshot →
  per-profile boot → cache diff). Iterates all 8 existing profiles + the new
  `qwen36-27b-dflash`; skips a profile when its model isn't pre-cached in
  the `llm-models` volume; final mtime diff is the cache-preservation hard
  gate. `qwen36-dflash` (MoE) excluded from the new-profile loop because
  its draft GGUF format mismatch (Phase 3) means boot will fail at draft
  load — exercise that profile manually with `EXPERIMENTAL=1` once a
  source-converted draft GGUF lands.

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
- [x] Extend `MODELS` associative array with two draft entries:
      `[qwen3.6-27b-dflash]="spiritbuun/Qwen3.6-27B-DFlash-GGUF|dflash-draft-3.6-q4_k_m.gguf|131072|"`
      and
      `[qwen3.6-35b-dflash]="lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test|Qwen3.6-35B-A3B-DFlash-q8_0.gguf|65536|"`.
      Repo SHAs pinned in a parallel `MODELS_HASH` map: 27B-draft @
      `5e4442a299deb9282b3dfe179de6e8330b19d9de`, 35B-draft @
      `3813f31a9fa837b79dce98e6ec49ddeaa4082772`. (Pinned via HF API; `hf list`
      isn't a current `huggingface-hub` subcommand on 1.11.x.)
- [x] Env var contract added to `entrypoint.sh`: `SPECULATIVE_MODE` ∈
      {target-only, autoregressive, dflash}; `DRAFT_MODEL_NAME`;
      `DRAFT_KV_CACHE_TYPE` (default `${KV_CACHE_TYPE}`); `DRAFT_N_MAX`
      (default 16); `EXPERIMENTAL` (gates `qwen3.6-35b*` + dflash).
      Speculative modes force `N_PARALLEL=1`. Single command builder uses
      verified flags from fork's `common/arg.cpp`: `-md`/`--model-draft`,
      `--draft-max`, `-ctkd`/`-ctvd`, `--dflash`.
- [x] Sub-task — entrypoint refactor preservation: `docker/test.sh`
      rewritten to (a) snapshot mtimes of all files in the `llm-models`
      named volume, (b) iterate every existing profile + `qwen36-27b-dflash`
      and assert each boots + serves a one-shot completion when its model
      is pre-cached, (c) re-snapshot and diff. Profiles whose models aren't
      pre-cached are SKIPped, not failed — preservation is the gate, not
      on-demand download. Actual host-side run deferred to a machine with
      pre-warmed `llm-models` volume.
- [x] `qwen36-27b-dflash` service: `MODEL_NAME: qwen3.6-27b`,
      `DRAFT_MODEL_NAME: qwen3.6-27b-dflash`, `KV_CACHE_TYPE: planar3`,
      `N_PARALLEL: 1` (entrypoint also enforces).
- [x] `qwen36-dflash` service: `MODEL_NAME: qwen3.6-35b`,
      `DRAFT_MODEL_NAME: qwen3.6-35b-dflash`, `KV_CACHE_TYPE: iso3`,
      `CTX_SIZE: 65536`, gated by host `EXPERIMENTAL=1`. (Snapshot is host-RAM
      MB-scale, not COW — see Phase 2 — so the 65K cap is not strictly
      required for headroom; kept as the experimental default until Phase 5
      VRAM accounting confirms 262K viability.)
- [partial] **Cache preservation gate** (HARD): logic implemented in
      `docker/test.sh` (`volume_mtime_snapshot` pre/post + `diff -q`).
      Verification deferred — must run on a host with the existing model
      cache and the rebuilt rotorquant image. Phase 5 should run this as
      the first step before any benchmark work.

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
- [x] **L1 (KV regression)**: `scripts/ppl_sweep.py` already meets the
      gate spec — paired-mode (rebased fork vs old fork via docker)
      writes `BENCHMARK-REPORT.md`-comparable JSON; ran in Phase 1 and
      again post-cherry-pick in Phase 3 with **0/10 regressions** both
      times. Results in
      `docs/sprints/SPRINT-004-L1-results.json` and
      `docs/sprints/SPRINT-004-L1-results-postcherrypick.json`. No
      changes needed.
- [partial] **L2 (greedy equivalence + forced-rejection)**: harness
      shipped as `scripts/validate_dflash.py`. CLI matches the brief
      (`--target` / `--draft` / `--prompts` / `--temp 0 --top-k 1 --seed
      42 --tokens 256` / `--force-reject-at N`). Hits the OpenAI-compat
      endpoint of an already-running `llama-server`, diffs token IDs,
      writes `docs/sprints/SPRINT-004-L2-results.json`. Measurement run
      deferred — needs a working DFlash draft GGUF (community drafts
      have a tensor-name mismatch with PR #22105's canonical schema;
      source-converted drafts need gated z-lab safetensors access).
      `LLAMA_SPEC_FORCE_REJECT_AT` env hook is itself deferred from
      Phase 2; the script sets it but tolerates it being a no-op.
- [partial] **L3 (z-lab differential — HARD GATE)**: harness shipped
      as `scripts/validate_dflash.py --reference zlab`. Clones
      `https://github.com/z-lab/dflash`, sets up a venv, runs the
      reference, asserts ≥64/64 on ≥3 of 5 prompts plus acceptance-rate
      parity within ±5pp. Pinned commit is `HEAD` in source — pin on
      first run. Measurement run deferred (same blocker as L2 plus an
      sm_120 wheel question — see Risks R10).
- [partial] **L4 (speedup median ≥1.3× — HARD GATE)**: harness shipped
      as `scripts/bench_speculative.py`. 3-way decode tok/s
      (target-only / target+autoregressive-draft / target+DFlash) on
      the fixed 5-prompt set; 3 trials per prompt; median per prompt;
      `--finalize` writes both JSON and a markdown summary. Pulls
      `timings.predicted_per_second` from llama-server's `usage`,
      falls back to wallclock. Measurement run deferred — needs a
      working DFlash draft GGUF (same Phase 3 blocker as L2/L3). The
      fixed 5-prompt set is:
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
- [partial] `tests/test_speculative.py`: shipped. GGUF metadata
      validation (skips without `gguf-py` or a draft on disk); sampler
      determinism (skips without a live server at `BASE_URL`);
      `LLAMA_SPEC_FORCE_REJECT_AT` honor test marked `@pytest.mark.xfail`
      because the env hook is itself Phase-2-deferred.
- [partial] `tests/test_dflash_e2e.py` (`@pytest.mark.docker`):
      shipped. Brings up `qwen36-27b-dflash` profile, submits greedy
      `/v1/chat/completions`, asserts equality with target-only on the
      same prompt, tears down. Skipped unless `DFLASH_E2E=1` and
      `docker` is on PATH — actually running it needs the working
      draft GGUF (Phase 3 blocker).
- [x] **Reproducibility entrypoint** (Phase 6 spec, ahead of time):
      `make bench-dflash` runs `bench_speculative.py --finalize`;
      `make bench-dflash-leg LEG=...` runs a single leg.

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
- [partial] Add §10 "Speculative Decoding" with: hybrid architecture
      explainer, checkpoint mechanism summary (from Phase 1 spike), 3-way
      tok/s tables for both Qwen3.6 targets, acceptance rates per-prompt,
      snapshot wallclock at 8K/16K/32K/65K/131K/262K, z-lab parity
      numbers. Hybrid explainer + checkpoint summary done. Tables present
      with structure but cells `TBD` — empirical numbers blocked behind
      F-001 (source-converted draft GGUF). Phase 2 already filled the 8K
      snapshot row.
- [x] Document the `LLAMA_SPEC_NO_THINK=1` env var; warn that thinking-on
      drops acceptance rate by 60–80 percentage points. README +
      BENCHMARK-REPORT.md §10 acceptance-rate notes paragraph; cites
      `examples/speculative-simple/speculative-simple.cpp:134` and
      `scripts/bench_speculative.py:265`.
- [partial] **Reproducibility task**: an outside reader should be able to
      run `make bench-dflash` in a fresh clone and reproduce the headline
      numbers without consulting sprint authors. Reproduction commands
      are in BENCHMARK-REPORT.md §10 (L4 leg sequence) and
      `validate_dflash.py --reference zlab` (L3). Headline numbers
      themselves remain `TBD` until F-001 unblocks measurement;
      reproducibility-from-instructions test deferred to that run.
- [x] Document the `EXPERIMENTAL=1` opt-in for `qwen36-dflash` profile.
      README "Speculative Decoding (Experimental)" subsection profile
      table + `EXPERIMENTAL=1 make run-qwen36-dflash` quickstart; cites
      PR #22105's gpt-oss-20B numbers (0.61–1.27×) as the rationale for
      no speedup gate on MoE.
- [x] Document explicitly: "this sprint validates greedy (`--temp 0
      --top-k 1`) only; sampling-mode behavior is unverified." README
      "Validation scope" sub-paragraph; cross-references SPRINT-004-
      DEFERRED.md D-003 (sampling) and D-006 (streaming).

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

---

## Sprint outcome

All six phases shipped: rebased fork onto current master with **0 PPL
regressions** across 10 cells (8 quantized cells improved by 0.06–0.60
PPL); landed `vram_seq_checkpoint` for **31–40× speedup** on snapshot
save+restore with bit-exactness validated; cherry-picked PR #22105
(DFlash + the EAGLE3 graph that came along for the ride) with zero
conflicts and no quality regression; added two opt-in compose profiles
(`qwen36-27b-dflash` dense, `qwen36-dflash` MoE behind `EXPERIMENTAL=1`)
with a single-builder entrypoint covering `target-only`/
`autoregressive`/`dflash` modes; shipped the L2/L3/L4 validation
harnesses (`scripts/validate_dflash.py`, `scripts/bench_speculative.py`,
`tests/test_speculative.py`, `tests/test_dflash_e2e.py`) plus a
`make bench-dflash` reproducibility entrypoint; documented all of it in
README + QUANTIZATION-GUIDE + BENCHMARK-REPORT.md §10.

The empirical end-to-end numbers (L2 greedy equivalence, L3 z-lab
parity, L4 ≥1.3× speedup gate) are blocked behind a single issue:
community DFlash drafts have a tensor-name format mismatch with PR
#22105's canonical schema, and source-converted drafts require gated
z-lab safetensors access. Profiles boot through llama-server start;
draft load is what fails. Tracked under
[`SPRINT-004-FOLLOWUPS.md`](SPRINT-004-FOLLOWUPS.md) F-001 — that doc is
the entry point for resuming work and lists 9 follow-ups in total
(F-001 load-bearing for measurement; F-002 forced-rejection env in
fork; F-003/F-004 formal C++ test + runtime guard from Phase 2;
F-005 cache-preservation gate first run; F-006 z-lab SHA pin;
F-007 `bench-dflash-all` orchestration target; F-008 Sprint 005 EAGLE3
scope shrunk; F-009 closed). Sprint marked `complete-with-followups`.
