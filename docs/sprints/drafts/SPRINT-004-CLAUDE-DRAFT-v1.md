# Sprint 004: Rebase Fork + DFlash Block-Diffusion Speculative Decoding on RotorQuant KV

**Status**: Draft (Claude)
**Created**: 2026-04-26
**Depends on**: Sprint 002 (Docker), RotorQuant fork at `johndpope/llama-cpp-turboquant`, branch `feature/planarquant-kv-cache`, current pinned commit `20efe75cf1127268cb2ad73accd5ccb6f33064ff`
**Target hardware**: RTX 5090 (32 GB), GPU may be co-occupied by training jobs

---

## Overview

The RotorQuant llama.cpp fork has been frozen on a stale `master` for ~3 weeks while
upstream merged a steady stream of context, server, and CUDA changes. Concurrently,
two upstream PRs — `ggml-org/llama.cpp#18039` (EAGLE3) and `ggml-org/llama.cpp#22105`
(DFlash) — landed in DRAFT state with collectively ~3 kLOC of speculative-decoding
infrastructure. DFlash promises 2.6–8.1× decode speedup on dense Qwen3 targets and
~1.0–1.3× on MoE. Both PRs touch `src/llama-context.cpp` heavily — the same file
that hosts our deferred-K hooks — so the longer we wait, the worse the merge.

This sprint does three things in strict order: **(1) rebase** the RotorQuant fork
onto mainline `master`, then prove the rebase is correctness-neutral via a full
PPL regression sweep; **(2) cherry-pick DFlash** (PR #22105) onto the rebased
branch, leaving EAGLE3 (PR #18039) for a follow-up sprint to keep diff size sane;
**(3) ship Docker profiles** `qwen36-27b-dflash` and `qwen36-dflash` with a
validation harness that proves greedy-decode equivalence to target-only.

The sprint is opinionated about staging: **DFlash only, dense-target only, single
slot only, greedy only**. EAGLE3, MoE-target speedup, parallel-slot composition,
and non-greedy sampler agreement are deferred to a Sprint 005 follow-up. The
goal is a *defensible* speedup on the model where the data is strongest
(Qwen3.6-27B dense), not a comprehensive sweep that turns a 3-week sprint into
a 6-week one.

---

## Use Cases

1. **Single-user dense decoding on Qwen3.6-27B**: Interactive chat at 24 GB
   tier (`--profile qwen36-27b-dflash`). Target ≥2.0× decode tok/s on coding
   prompts ("Write a quicksort algorithm in Python") with thinking-off and
   greedy sampling. This is the lowest-risk speedup target per PR #22105's
   data (Qwen3-8B sees 8.08× w/o thinking; we expect lower on 27B but ≥2×).

2. **Single-user MoE decoding on Qwen3.6-35B-A3B**: Experimental profile
   (`--profile qwen36-dflash`) shipped behind a `EXPERIMENTAL=1` env gate.
   Target: no regression vs target-only at iso3 (≥1.0×). Anything above 1.0×
   is a win; PR #22105's gpt-oss-20B numbers (0.61–1.27×) imply this is
   genuinely uncertain.

3. **Validated rebased fork**: All 8 existing Docker profiles (`qwen`,
   `qwen36-q3`, `qwen36-iq3`, `qwen36-27b`, `qwen36-27b-q3`, `qwen36-27b-iq3`,
   `reasoning`, `gemma`) launch and serve identically post-rebase, with PPL
   matching `docs/BENCHMARK-REPORT.md` Section 1 within ±0.05 PPL across the
   4 KV types × 2 corpora × 2 model families.

4. **Reusable differential validation harness**: A
   `scripts/validate_dflash.py` runner that takes any (target.gguf, draft.gguf)
   pair and confirms greedy-decode equivalence to target-only — so when EAGLE3
   lands in Sprint 005 (or when upstream actually merges either PR), validation
   is one command.

---

## Architecture

### Cherry-pick boundary

DFlash (PR #22105) is *built on top of* EAGLE3 (PR #18039) as its parent. Per
the PR text: "Please focus on the DFlash-specific commit(s); the EAGLE3 commits
will disappear from the diff once #18039 merges." We treat this literally:

- **In scope**: DFlash-specific commits only (the additive `src/models/dflash.cpp`
  graph and the `--dflash` flag plumbing in `examples/speculative-simple/`).
- **Borrowed from EAGLE3 PR**: only the *foundation* commits that DFlash
  depends on — the `LLM_ARCH_*` registration plumbing in `src/llama-arch.cpp/h`,
  the `g_embeddings` decoder-input handling, and `GGML_TENSOR_FLAG_SYNC`. These
  are required even if we don't ship EAGLE3 as a runtime mode.
- **Out of scope**: EAGLE3 model graph (`src/models/eagle3.cpp`), `--eagle3`
  flag, EAGLE3 conversion path in `convert_hf_to_gguf.py`, EAGLE3 GGUFs.
  Carry-forward to Sprint 005.

### Component diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  Rebased RotorQuant fork (post-Sprint-004)                           │
│                                                                      │
│   src/llama-arch.{cpp,h}        ← +LLM_ARCH_DFLASH (additive)        │
│   src/llama-context.cpp         ← rebase merge zone (HIGH RISK)      │
│   src/llama-graph.{cpp,h}       ← +44 LOC for DFlash cross-attn      │
│   src/llama-kv-cache.cpp        ← deferred-K + verify-time hook      │
│                                   (NEW: kv_seq_rm w/ quantized K)    │
│   src/models/dflash.cpp         ← NEW (+161 LOC, draft graph)        │
│   src/models/qwen35*.cpp        ← +14-25 LOC each (target metadata)  │
│   src/models/qwen3moe.cpp       ← +11 LOC                            │
│   common/speculative.cpp        ← +331 LOC (block draft + verify)    │
│   examples/speculative-simple/  ← +77 LOC, --dflash flag             │
│   convert_hf_to_gguf.py         ← +187 LOC, DFlash GGUF emit         │
│   gguf-py/gguf/constants.py     ← +53 LOC, DFlash KV metadata        │
│   ggml-cuda/{cpy,set-rows,fattn-common,planar-iso-constants}         │
│                                  ← OURS, untouched by upstream PRs   │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  This repo (turbo)                                                   │
│   docker/Dockerfile             ← bump ROTORQUANT_COMMIT pin         │
│   docker/entrypoint.sh          ← --draft-model, --dflash, dual-DL   │
│   docker-compose.yml            ← qwen36-27b-dflash, qwen36-dflash   │
│   scripts/bench_speculative.py  ← NEW (decode tok/s, 3-way)          │
│   scripts/validate_dflash.py    ← NEW (z-lab pytorch differential)   │
│   tests/test_speculative.py     ← NEW (pytest unit)                  │
│   tests/test_dflash_e2e.py      ← NEW (pytest integration)           │
│   docs/BENCHMARK-REPORT.md      ← +§10 Speculative Decoding          │
│   README.md                     ← Speculative section + quickstart   │
└──────────────────────────────────────────────────────────────────────┘
```

### Verify-time deferred-K behavior (resolution of seed Q2)

The current fork's `convert_deferred_keys()` runs once when the prefill batch
completes. With DFlash, after prefill ends the verify path appends a draft
**block** of N tokens (N = `--draft-max`, default 16) in a single `decode()`
call. Three options exist; this sprint **chooses (b) with a guard**:

- **(a) Treat verify batch as fresh prefill**: incorrect — re-defers and
  re-quantizes already-committed K. Wasteful and risks double-conversion bugs.
- **(b) Treat each token as a separate decode append**: correct, ~16× quant
  kernel calls per verify step. Acceptable: the per-token quant kernel is
  microseconds vs the ~1ms target forward; verify cost is dominated by attention,
  not quantization.
- **(c) Batched-decode quant path**: a new fused kernel `cpy_planar_iso_batch`
  for `n_tokens > 1, prefill_done == true`. Faster than (b) but adds kernel
  surface area; deferred to Sprint 005 if (b) measures slow.

The fix is a **single `bool prefill_complete` flag** in `llama_kv_cache_unified`
set inside `convert_deferred_keys()`, and a guard in the K-append path that
quantizes inline when `prefill_complete == true && n_tokens >= 1` (per-token
loop). This is ~30 LOC in `src/llama-kv-cache.cpp`.

### Speculative `seq_rm` rollback with quantized K

When verify rejects a suffix, llama.cpp calls `seq_rm(seq_id, p0, p1)` to
truncate the KV cache. The fork's quantized K layout is block-aligned
(50 B per 128-element block for planar3); arbitrary `p0:p1` truncation must
not leave a partial block behind. Two cases:

- **Aligned truncation** (`p1 % BLOCK_SIZE == 0`): trivial, drop tail blocks.
- **Mid-block truncation**: dequantize the partial block, re-quantize the
  retained prefix. Implemented by a new `kv_cache_quantized_seq_rm()` helper
  that runs only on the trailing partial block. Cost: one block dequant +
  one block quant, both microsecond-scale.

This must be implemented in the fork (`src/llama-kv-cache.cpp` ~80 LOC) and
tested at three boundaries: aligned, off-by-one, off-by-block-minus-one.

### Draft-model VRAM budget (resolution of seed Q3)

Draft GGUFs from `lym00` and `spiritbuun` are not yet sized; assume 1–2 GB
each (typical EAGLE3/DFlash drafts). Both target's KV type and draft's KV
type must match in a single `llama-server` process: llama.cpp's KV pool is
per-context, and mixing types in one server is unsupported. **Decision: draft
inherits target's `--cache-type-k`/`-v` automatically via a single env var
in `entrypoint.sh`.** No mixed-mode support in this sprint.

VRAM budget on RTX 5090 (32 GB) for `qwen36-27b-dflash`:
- Qwen3.6-27B UD-Q4_K_XL weights: 16.4 GB
- Target KV at 131K planar3, 1 slot: ~3.3 GB
- Draft weights (estimated 1.5–2.5 GB)
- Draft KV at 131K planar3, 1 slot: ~0.4 GB (small head_dim)
- Activations + buffers: ~3 GB
- **Total: ~24–26 GB → fits with 6+ GB headroom**

For `qwen36-dflash` (MoE 35B target): tighter — 20.8 GB target + ~7 GB iso3
KV + ~2 GB draft + buffers ≈ 31 GB. May force `CTX_SIZE` reduction from
524288 → 262144. Documented in compose comment.

---

## Implementation

### Phase 1: Rebase the RotorQuant fork (~35% of effort)

**Goal**: `feature/planarquant-kv-cache` rebased onto current `master`,
PPL-equivalent to commit `20efe75` baseline.

**Files (in fork `johndpope/llama-cpp-turboquant`):**
- `src/llama-context.cpp` — primary merge conflict zone
- `src/llama-kv-cache.cpp` — deferred-K logic; conflict probability medium
- `ggml-cuda/CMakeLists.txt` — template instance file list; conflict probability medium
- `ggml-cuda/cpy-planar-iso.cu`, `set-rows-planar-iso.cuh`,
  `planar-iso-constants.cuh`, `fattn-common.cuh` — ours, expect zero conflicts
- All other fork-touched files — assume clean rebase

**Tasks:**
- [ ] Branch `feature/planarquant-kv-cache-rebase` off the current
      `feature/planarquant-kv-cache` HEAD; `git rebase --onto upstream/master
      $(git merge-base feature/planarquant-kv-cache upstream/master)
      feature/planarquant-kv-cache-rebase`
- [ ] Resolve `src/llama-context.cpp` conflicts: preserve our deferred-K
      hooks at `llama_context::process_ubatch()` and `llama_context::decode()`;
      take upstream changes everywhere else
- [ ] Re-validate `ggml-cuda/CMakeLists.txt` template-instance list against
      upstream's CUDA file glob; manually re-add `cpy-planar-iso.cu` and
      friends if dropped
- [ ] Compile clean: `cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA=ON
      -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120" -DCMAKE_BUILD_TYPE=Release
      && cmake --build build -j$(nproc)` — must succeed with no warnings
      from our files
- [ ] Run `tools/perplexity/perplexity` smoke test on Qwen3.5-9B Q4_K_M with
      `--cache-type-k iso3 --cache-type-v iso3` and confirm coherent loss values
- [ ] Tag the rebased commit; record SHA in this doc and in
      `docker/Dockerfile` `ROTORQUANT_COMMIT` arg

**L1 regression gate (HARD):** `scripts/ppl_sweep.sh` — run all 4 KV types
(`planar3`, `planar4`, `iso3`, `iso4`) × 2 corpora (wikitext-2, C4) × 2 models
(Qwen3.6-35B-A3B, Qwen3.6-27B). All 16 numbers must match
`docs/BENCHMARK-REPORT.md` §1.5–1.8 within ±0.05 PPL. Sprint blocks here if
this fails — no DFlash work begins until this gate passes (resolution of seed Q1).

### Phase 2: Verify-time deferred-K + seq_rm support (~10% of effort)

**Goal**: Make the fork's KV cache safe for parallel-verify and rollback,
*before* DFlash exercises those code paths.

**Files (fork):**
- `src/llama-kv-cache.cpp` — add `prefill_complete` flag, post-prefill K
  quantization in `kv_cache_unified::set_input_kq()` (or equivalent), block-aware
  `seq_rm` for quantized K
- `tests/test-kv-quantized-seq-rm.cpp` — NEW unit test in fork (resolution
  of seed Q8: yes, fork gets a C++ unit test for the C++ path)

**Tasks:**
- [ ] Add `bool prefill_complete = false;` member to
      `llama_kv_cache_unified`; set to `true` at end of `convert_deferred_keys()`
- [ ] Add `if (prefill_complete && ubatch.n_tokens >= 1) { quantize_inline_per_token(); }`
      branch in K-append path
- [ ] Implement `kv_cache_quantized_seq_rm(seq_id, p0, p1)`: identify
      partial trailing block via `p1 % BLOCK_SIZE`; if non-zero, dequant +
      re-quant the retained prefix
- [ ] C++ unit test: build a 256-token quantized K cache; `seq_rm` at
      offsets {64 (aligned), 65 (off-by-1), 127 (off-by-block-minus-1)};
      assert dequant of remaining prefix matches reference within 1e-4 cosine
- [ ] Sanity: run `llama-cli` on Qwen3.5-9B with autoregressive
      speculative decoding (`-md` flag, no `--dflash`) using a small Qwen3-0.6B
      draft; confirm output matches target-only at `--temp 0 --seed 42`

This phase doubles as a sanity probe for the speculative path *before*
DFlash adds non-causal block draft complexity.

### Phase 3: Cherry-pick DFlash from PR #22105 (~25% of effort)

**Goal**: DFlash draft + verify path compiles and runs on the rebased fork.

**Files (fork, additive unless noted):**
- `src/llama-arch.{cpp,h}` — register `LLM_ARCH_DFLASH` (carried from PR #18039)
- `src/llama-graph.{cpp,h}` — cross-attention plumbing for DFlash decoder
- `src/models/dflash.cpp` — NEW, DFlash draft graph
- `src/models/qwen35*.cpp`, `src/models/qwen3moe.cpp` — DFlash target metadata
- `common/speculative.cpp` — block-draft + verify orchestration
- `examples/speculative-simple/speculative-simple.cpp` — `--dflash`,
  `--draft-max`, `LLAMA_SPEC_NO_THINK` env
- `convert_hf_to_gguf.py` — DFlash GGUF emit (we won't run this — pre-built
  GGUFs exist — but cherry-pick anyway for completeness)
- `gguf-py/gguf/constants.py` — DFlash KV metadata constants

**Tasks:**
- [ ] Identify DFlash-specific commits via `gh pr view 22105 --json commits`;
      filter out commits that originated in PR #18039 (EAGLE3 parent)
- [ ] `git cherry-pick <DFlash-commit-list>` onto rebased branch; resolve
      conflicts with our deferred-K work in `src/llama-context.cpp` (HIGH RISK)
- [ ] Cherry-pick the *minimal* EAGLE3 foundation commits required as
      DFlash dependencies: `LLM_ARCH_*` enum slots, `g_embeddings` plumbing,
      `GGML_TENSOR_FLAG_SYNC`. Skip `src/models/eagle3.cpp` and `--eagle3` flag
- [ ] Compile clean with `-DGGML_CUDA=ON -DGGML_CUDA_FA=ON`
- [ ] Smoke test: `./build/bin/llama-speculative-simple
      -m Qwen3.6-27B-UD-Q4_K_XL.gguf -md Qwen3.6-27B-DFlash.gguf
      --dflash -p "Hello" -n 32 --draft-max 16 --temp 0 --top-k 1 --seed 42
      -ngl 99 -ngld 99` — produces output without crash

### Phase 4: Docker + entrypoint.sh + compose profiles (~10% of effort)

**Goal**: Two new `docker compose --profile` targets serve DFlash end-to-end.

**Files (this repo):**
- `docker/Dockerfile` — bump `ROTORQUANT_COMMIT` arg to rebased SHA
- `docker/entrypoint.sh` — extend `MODELS` registry with `qwen3.6-27b-dflash`
  and `qwen3.6-35b-dflash` entries; add `DRAFT_MODEL_NAME` env handling;
  pass `--model-draft`, `--dflash`, `--draft-max` to `llama-server`
- `docker-compose.yml` — add `qwen36-27b-dflash` and `qwen36-dflash` services
- `Makefile` — `run-qwen36-27b-dflash`, `run-qwen36-dflash` targets
- `docker/test.sh` — add DFlash smoke test step

**Tasks:**
- [ ] Extend `MODELS` associative array in `docker/entrypoint.sh`:
  ```
  [qwen3.6-27b-dflash]="spiritbuun/Qwen3.6-27B-DFlash-GGUF|<filename>|131072|--dflash --draft-max 16"
  [qwen3.6-35b-dflash]="lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test|<filename>|262144|--dflash --draft-max 16"
  ```
  (Filenames TBD on first download; the sprint must list them once HF browse
  reveals the exact names.)
- [ ] Add `DRAFT_MODEL_NAME` env handling: if set, look up draft GGUF in
      same registry, download if missing, pass `--model-draft <path>` to server.
      Draft inherits `KV_CACHE_TYPE` automatically (no separate var)
- [ ] `qwen36-27b-dflash` service: `MODEL_NAME: qwen3.6-27b`,
      `DRAFT_MODEL_NAME: qwen3.6-27b-dflash`, `KV_CACHE_TYPE: planar3`,
      `N_PARALLEL: 1` (single-slot enforced for v1; resolution of seed Q5)
- [ ] `qwen36-dflash` service: same shape, `MODEL_NAME: qwen3.6-35b`,
      `DRAFT_MODEL_NAME: qwen3.6-35b-dflash`, `KV_CACHE_TYPE: iso3`,
      `CTX_SIZE: 262144`. Add comment: `# EXPERIMENTAL — MoE speedup
      0.6-1.3× per upstream PR #22105`
- [ ] `docker/test.sh`: launch `qwen36-27b-dflash` profile, hit `/health`,
      submit a greedy completion, assert response is non-empty and matches
      target-only response on the same prompt
- [ ] Verify all 8 pre-existing profiles still launch (no Docker volume
      re-download — `llm-models` named volume cache must be preserved)

### Phase 5: Validation harness (~15% of effort)

**Goal**: Three-layer validation runnable via one command per layer.

**Files (this repo):**
- `scripts/validate_dflash.py` — NEW, differential vs z-lab pytorch reference
- `scripts/bench_speculative.py` — NEW, 3-way decode tok/s
- `tests/test_speculative.py` — NEW, pytest unit
- `tests/test_dflash_e2e.py` — NEW, pytest integration via Docker

**Tasks:**
- [ ] **L1 (KV regression)**: extend `scripts/ppl_sweep.sh` to write
      `docs/BENCHMARK-REPORT.md`-comparable JSON; runner asserts ±0.05 PPL
      vs baseline table (already covered by Phase 1 gate; this just makes
      it CI-ready)
- [ ] **L2 (greedy equivalence)**: `scripts/validate_dflash.py
      --target qwen3.6-27b --draft qwen3.6-27b-dflash --prompts P1,P2,P3
      --temp 0 --top-k 1 --seed 42 --tokens 256` runs `llama-cli` twice
      (target-only and target+DFlash via Docker exec into running server);
      diffs token sequences; prints per-prompt match length. Pass: 256/256 on
      ≥3 prompts
- [ ] **L3 (z-lab differential)**: same script with `--reference zlab` flag
      clones `https://github.com/z-lab/dflash`, sets up venv with
      `torch>=2.4 transformers>=4.45`, runs `dflash/model.py` reference on
      same prompt/seed, diffs first 64 tokens. Pass: 64/64 match,
      acceptance-rate within ±5 percentage points
- [ ] `tests/test_speculative.py`:
      - `test_dflash_gguf_metadata`: open the DFlash GGUF, assert it has
        the expected `dflash.*` metadata keys and N transformer layers
      - `test_sampler_determinism`: 3× target-only runs at `--temp 0
        --seed 42` produce identical 256-token output
- [ ] `tests/test_dflash_e2e.py` (marker `@pytest.mark.docker`, skip if
      no GPU): bring up `qwen36-27b-dflash` profile via `docker compose up -d`;
      poll `/health`; submit greedy `/v1/chat/completions`; assert response equal
      to target-only response on same prompt; tear down
- [ ] `scripts/bench_speculative.py`: 3-way decode tok/s — target-only,
      target+autoregressive draft, target+DFlash — on a fixed 5-prompt set
      derived from PR #22105's prompts ("Write a quicksort algorithm in
      Python", "Explain the Pythagorean theorem", "Plan a 1 day trip to DC",
      and 2 of our own); JSON output piped into `docs/BENCHMARK-REPORT.md` §10

### Phase 6: Documentation (~5% of effort)

**Files:**
- `docs/BENCHMARK-REPORT.md` — add §10 Speculative Decoding
- `README.md` — add "Speculative Decoding (Experimental)" section under
  Performance, document the two new profiles, link to BENCHMARK-REPORT §10
- `docs/QUANTIZATION-GUIDE.md` — add "Draft model VRAM cost" subsection;
  note that 24 GB tier may need to drop to UD-Q3_K_XL when DFlash is enabled
- `docs/sprints/SPRINT-004-FOLLOWUPS.md` — record EAGLE3 cherry-pick,
  multi-slot DFlash, non-greedy validation, MoE deep-dive

**Tasks:**
- [ ] Add a 3-row tok/s comparison table per model (target-only, +AR draft,
      +DFlash) with prompt names and acceptance-rate column
- [ ] Document the `LLAMA_SPEC_NO_THINK=1` env var; warn that thinking-on
      drops acceptance rate by 60–80 percentage points
- [ ] Update commit hash in `docker/Dockerfile` and reference it in
      `BENCHMARK-REPORT.md` "Runtime" header
- [ ] Run final `git diff --stat` against pre-sprint `main` to confirm no
      unintended file changes

---

## Files Summary

### Fork (`johndpope/llama-cpp-turboquant`, branch `feature/planarquant-kv-cache`)

| File | Action | Purpose |
|------|--------|---------|
| `src/llama-context.cpp` | Modify (rebase) | Resolve conflicts; preserve deferred-K hooks |
| `src/llama-kv-cache.cpp` | Modify | `prefill_complete` flag, inline post-prefill K quant, block-aware `seq_rm` |
| `src/llama-arch.{cpp,h}` | Modify (cherry-pick) | `LLM_ARCH_DFLASH` enum slot |
| `src/llama-graph.{cpp,h}` | Modify (cherry-pick) | DFlash cross-attention plumbing |
| `src/models/dflash.cpp` | Create (cherry-pick) | DFlash draft graph (~161 LOC from PR #22105) |
| `src/models/qwen35*.cpp` | Modify (cherry-pick) | DFlash target metadata |
| `src/models/qwen3moe.cpp` | Modify (cherry-pick) | DFlash MoE target metadata |
| `common/speculative.cpp` | Modify (cherry-pick) | Block draft + verify orchestration |
| `examples/speculative-simple/speculative-simple.cpp` | Modify (cherry-pick) | `--dflash`, `--draft-max` flags |
| `convert_hf_to_gguf.py` | Modify (cherry-pick) | DFlash GGUF emit |
| `gguf-py/gguf/constants.py` | Modify (cherry-pick) | DFlash KV metadata |
| `ggml-cuda/CMakeLists.txt` | Verify (rebase) | Re-add planar-iso template list if dropped |
| `tests/test-kv-quantized-seq-rm.cpp` | Create | C++ unit test for block-aware `seq_rm` |

### This repo (`turbo`)

| File | Action | Purpose |
|------|--------|---------|
| `docker/Dockerfile` | Modify | Bump `ROTORQUANT_COMMIT` to rebased SHA |
| `docker/entrypoint.sh` | Modify | `DRAFT_MODEL_NAME` env, `--model-draft` plumbing, 2 new MODELS entries |
| `docker-compose.yml` | Modify | Add `qwen36-27b-dflash`, `qwen36-dflash` services |
| `docker/test.sh` | Modify | Add DFlash smoke test |
| `Makefile` | Modify | `run-qwen36-27b-dflash`, `run-qwen36-dflash` targets |
| `scripts/validate_dflash.py` | Create | L2 + L3 differential validation runner |
| `scripts/bench_speculative.py` | Create | 3-way decode tok/s benchmark |
| `scripts/ppl_sweep.sh` | Modify | JSON output for L1 gate automation |
| `tests/test_speculative.py` | Create | Pytest unit (GGUF metadata, determinism) |
| `tests/test_dflash_e2e.py` | Create | Pytest integration (Docker e2e) |
| `docs/BENCHMARK-REPORT.md` | Modify | Add §10 Speculative Decoding |
| `docs/QUANTIZATION-GUIDE.md` | Modify | Add "Draft model VRAM cost" subsection |
| `README.md` | Modify | "Speculative Decoding (Experimental)" section |
| `docs/sprints/SPRINT-004.md` | Create | Final merged sprint doc (this file → there) |
| `docs/sprints/SPRINT-004-FOLLOWUPS.md` | Create | Carry-forward EAGLE3, multi-slot, non-greedy, MoE deep-dive |

---

## Definition of Done

### Hard gates (sprint fails if missed)

- [ ] **L1 KV regression**: PPL for all 4 KV types × 2 corpora × 2 models
      (Qwen3.6-35B-A3B, Qwen3.6-27B) matches `docs/BENCHMARK-REPORT.md` §1.5–1.8
      within ±0.05 PPL on the rebased fork
- [ ] **All 8 existing Docker profiles launch and serve**: `qwen`, `qwen36-q3`,
      `qwen36-iq3`, `qwen36-27b`, `qwen36-27b-q3`, `qwen36-27b-iq3`,
      `reasoning`, `gemma`. Volume `llm-models` cache preserved (no re-download)
- [ ] **L2 greedy equivalence**: `qwen36-27b-dflash` produces identical
      256-token output to target-only on ≥3 prompts at `--temp 0 --top-k 1 --seed 42`
- [ ] **L3 z-lab pytorch parity**: first 64 tokens match z-lab reference on
      ≥1 prompt; acceptance rate within ±5 percentage points of PR #22105's
      reported number for matching prompt
- [ ] **Decode speedup**: ≥2.0× decode tok/s on Qwen3.6-27B dense for "Write
      a quicksort algorithm in Python. Write code only." with `LLAMA_SPEC_NO_THINK=1`,
      `--temp 0 --top-k 1 --seed 42`, `n=256` (single slot, planar3 KV)
- [ ] **C++ `seq_rm` unit test passes** for aligned, off-by-1, and
      off-by-block-minus-1 truncation offsets
- [ ] **`pytest tests/` passes** including `test_speculative.py` and
      `test_dflash_e2e.py` (the e2e test may be marked `@pytest.mark.docker`
      and skipped in non-GPU CI)

### Soft gates (sprint succeeds with caveats; ship as experimental)

- [ ] **MoE speedup**: `qwen36-dflash` at iso3 achieves ≥1.0× decode (no
      regression). ≥1.2× is a win; ship behind `EXPERIMENTAL=1` if
      0.9× ≤ speedup < 1.0× (resolution of seed Q6: yes, MoE ships as experimental)
- [ ] **Acceptance rate parity**: within 10 percentage points of PR #22105's
      reported numbers for matching prompts on the dense target
- [ ] **Validation harness reusable**: `scripts/validate_dflash.py` accepts
      arbitrary `(target, draft)` GGUF pair and works without code changes
- [ ] **Long-context smoke test**: 32K-prompt greedy decode with DFlash
      completes without OOM or seq_rm bug

### Documentation

- [ ] `docs/BENCHMARK-REPORT.md` §10 added with 3-way tok/s table for
      Qwen3.6-27B and Qwen3.6-35B-A3B, including acceptance rates
- [ ] `README.md` "Speculative Decoding (Experimental)" section live
- [ ] `docs/QUANTIZATION-GUIDE.md` "Draft model VRAM cost" subsection live
- [ ] `docs/sprints/SPRINT-004-FOLLOWUPS.md` lists EAGLE3, multi-slot,
      non-greedy, MoE deep-dive

### Code hygiene

- [ ] Final commit on the fork tagged `sprint-004-dflash`; SHA pinned in
      `docker/Dockerfile`'s `ROTORQUANT_COMMIT` arg
- [ ] All git operations use `git add -u` or explicit file lists (per
      project CLAUDE.md)
- [ ] Commit messages: imperative subject + `Co-Authored-By: Claude Opus 4.7
      <noreply@anthropic.com>` trailer
- [ ] No `.env`, `HF_TOKEN`, or other secrets committed; `.dockerignore`
      verified

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `src/llama-context.cpp` rebase conflicts unresolvable in <1 day | Medium | High | If rebase exceeds 2 days, abort to a 3-way merge approach: cherry-pick our 4 deferred-K commits onto a fresh fork off latest master |
| DFlash cherry-pick breaks PPL via subtle KV layout change | Medium | High | L1 gate runs *after* every cherry-pick step, not just end-of-sprint. Bisect on first failure |
| Verify-time deferred-K bug masks DFlash correctness issue | Medium | High | Phase 2 sanity (autoregressive speculative on Qwen3.5-9B) runs *before* DFlash cherry-pick; isolates the two failure modes |
| Pre-built DFlash GGUFs from `lym00`/`spiritbuun` don't match upstream PR's expected layout | Medium | High | First action of Phase 5: download both, run `gguf-dump`, verify against PR #22105's `convert_hf_to_gguf.py` output schema. If mismatch, regenerate locally |
| Draft VRAM pushes Qwen3.6-35B-A3B over 32 GB at default ctx | Medium | Medium | Reduce `CTX_SIZE` from 524288 → 262144 in `qwen36-dflash` profile; document trade-off |
| z-lab pytorch reference doesn't run on RTX 5090 (compute capability 12.0 mismatch) | Low | Medium | If torch ≥2.4 + sm_120 wheel unavailable, run reference on a separate machine and store reference outputs as JSON fixtures in `tests/fixtures/` |
| Upstream rebases PR #22105 mid-sprint (likely — author is actively iterating) | High | Low | Pin to PR commit SHA at sprint start; do not chase upstream during sprint. Note SHA in this doc |
| `seq_rm` block-aware fix has off-by-one error in production | Low | High | Three-way unit test (aligned, off-by-1, off-by-block-minus-1) in `tests/test-kv-quantized-seq-rm.cpp`; integration test runs greedy decode for 256 tokens with high rejection rate to exercise the path |
| MoE target speedup measures < 1.0× (slowdown) | Medium | Low | Pre-acknowledged: ship `qwen36-dflash` behind `EXPERIMENTAL=1`. Document in README. Slowdown does not block sprint |
| Docker image size grows past 10 GB with DFlash binaries | Low | Low | Multi-stage build already strips unused; binaries are <100 MB delta |
| GPU co-occupancy with training jobs causes intermittent OOM in benchmarks | Medium | Medium | All benchmarks runnable in <10 min windows; coordinate with training job schedule. `bench_speculative.py` records per-run free VRAM and aborts if <8 GB headroom |
| `convert_hf_to_gguf.py` cherry-pick conflicts with our none changes there | Low | Low | We don't modify that file; should be additive |
| Cherry-pick boundary mistake: include EAGLE3 commits accidentally | Medium | Low | Pre-write the commit list as a `cherry-pick.txt` in Phase 3; review before applying. Excluded files: `src/models/eagle3.cpp`, `--eagle3` runtime flag |

---

## Security Considerations

- **No new network surface**: DFlash adds no new ports or endpoints to
  `llama-server`; existing `0.0.0.0:8080` OpenAI-compat API is the only
  exposure (unchanged from Sprint 002).
- **Draft model download**: Pre-built GGUFs come from third-party HuggingFace
  repos (`lym00`, `spiritbuun`). Add a SHA256 pin to `entrypoint.sh` once
  files are downloaded once, and verify on subsequent pulls. GGUF format
  is binary-deserialized into model weights — not RCE-class but worth a
  hash check.
- **z-lab reference clone**: `scripts/validate_dflash.py` clones
  `https://github.com/z-lab/dflash` at a pinned commit (TBD); never `HEAD`.
  Reference runs in a venv, never against system Python.
- **`HF_TOKEN` handling**: Existing pattern preserved — env var only,
  never logged, never baked into image. Draft model download path uses the
  same `HF_TOKEN` if set.
- **No new sudo / privileged surface**: Container still runs as `llm`
  UID 1001; draft model loaded into the same process; no privilege escalation.

---

## Dependencies

### Prior work

- **Sprint 002** (Dockerize): the `llm-base` compose anchor, `entrypoint.sh`
  MODELS registry, multi-stage Dockerfile, and `llm-models` named volume are
  the foundation Phase 4 extends. The `qwen36-*` profiles added between
  Sprint 002 and now (commits `b8c9858`, `d675068`, `8c0907b`) are the
  template for the new DFlash services.
- **Sprint 003** (SpectralQuant): negative result; not a dependency. The
  research-only `turboquant/spectral/` Python module is untouched.

### External

- **Upstream llama.cpp** at current `master` (will float — pin a SHA on day 1
  of Phase 1).
- **PR #22105 commits** at sprint-start SHA (pin in `docker/Dockerfile`
  comment; do not float).
- **Pre-built DFlash GGUFs**:
  - [`lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test`](https://huggingface.co/lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test)
  - [`spiritbuun/Qwen3.6-27B-DFlash-GGUF`](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF)
- **z-lab reference**: `https://github.com/z-lab/dflash` (Python; for L3
  validation only; not a runtime dependency)
- **CUDA 13.1**, **NVIDIA driver 570+**, **Docker 24+ + NVIDIA Container
  Toolkit** (unchanged from Sprint 002)

### Hardware

- **RTX 5090 (32 GB)** for full sprint. Sprint runs without continuous GPU
  access; benchmarks are scheduled around training job windows.
- **123 GB system RAM** — sufficient for both target + draft weights in
  prompt cache.

---

## Open Questions

The 8 questions from `SPRINT-004-INTENT.md` resolved by this draft:

### Q1. Phase boundary for "rebase complete"
**Resolution: Strict gate.** Phase 1 must produce a passing L1 PPL sweep
before any DFlash code is touched. Rationale: if the rebase silently
breaks K quantization, every subsequent DFlash test failure is ambiguous
between "DFlash bug" and "rebase bug". Cost of strict gate: ~1 day of
serial blocking; benefit: clean root-cause attribution downstream.

### Q2. Verify-time deferred K behavior
**Resolution: Per-token quant inline (option b), guarded by
`prefill_complete` flag.** Add a 30-LOC fix in `src/llama-kv-cache.cpp`
(Phase 2). The faster batched path (option c) is deferred to Sprint 005
unless option b measures > 1% decode overhead in benchmarks.

### Q3. Draft model on which KV cache type
**Resolution: Draft inherits target's KV type.** Both target and draft use
the same `--cache-type-k`/`-v` value, set via single `KV_CACHE_TYPE` env
var. Mixed types in one `llama-server` are not supported by the existing
KV pool, and there's no benchmark evidence that f16 draft KV improves
acceptance rate enough to justify the extra VRAM.

### Q4. Wait for upstream merge?
**Resolution: Cherry-pick now.** Both PRs are still in DRAFT state with
"BLOCKED on review" — author is iterating actively, but neither is days
from merge. Waiting risks 2–4 more weeks of master drift on top of our
already-3-week-stale fork. Pin to a PR SHA at Phase 3 start; treat
upstream changes after that as out-of-scope.

### Q5. Single-bench parallel slot interaction
**Resolution: Single slot for v1 (`N_PARALLEL: 1` in DFlash profiles).**
Multi-slot DFlash is genuinely uncertain — the verify pass batches across
slots in ways neither PR documents. Defer to Sprint 005. Existing
`qwen36-throughput` profile (8 slots) remains untouched.

### Q6. MoE speedup reality check
**Resolution: Exclude MoE from hard speedup gate; ship as experimental.**
Already structured this way in DoD. Add `EXPERIMENTAL=1` env gate on
`qwen36-dflash` profile and document the 0.6–1.3× expected range in the
compose comment and README.

### Q7. What gets deferred from this sprint
**Resolution: Defer EAGLE3, multi-slot, non-greedy, MoE deep-dive.**
Phase cuts:
- **EAGLE3** entirely → Sprint 005. The DFlash PR depends on EAGLE3
  *structurally* but not *operationally*; we cherry-pick only the foundation
  commits, not the EAGLE3 model graph or `--eagle3` flag.
- **Non-greedy sampler agreement** → follow-ups. Sprint scope is `--temp
  0 --top-k 1` only; sampler diff at temp>0 needs its own correctness
  framework.
- **Multi-slot DFlash** → Sprint 005.
- **MoE deep-dive** (per-expert profiling, MoE-aware draft) → follow-ups.
- **`spec4-server` mode** (running speculative inside `llama-server` rather
  than `llama-speculative-simple`) → Sprint 005. Server integration is
  scaffolded by PR #22105 but not validated.

### Q8. Test coverage for `seq_rm` + quantized K
**Resolution: Both — fork unit + repo integration.**
- **Fork**: `tests/test-kv-quantized-seq-rm.cpp` with three boundary
  cases (aligned, off-by-1, off-by-block-minus-1). 80 LOC. Runs as part
  of fork's existing CTest infrastructure.
- **This repo**: `tests/test_dflash_e2e.py` exercises `seq_rm` indirectly
  via 256-token DFlash decode where rejection events trigger the path.
  Pytest marker `@pytest.mark.docker` for GPU/Docker gating.

### Carry-forward open questions

- **Filenames in pre-built GGUFs**: Both HuggingFace repos must be
  inspected on day 1 of Phase 4 to record exact filenames in `MODELS`
  registry. Cannot resolve in advance without HF browse.
- **Exact rebased SHA**: Recorded after Phase 1 completes; placeholder
  `<sprint-004-rebase-head>` until then.
- **DFlash GGUF metadata schema stability**: PR #22105 author may iterate
  on metadata keys; if upstream rev breaks our pinned SHA's schema, we
  regenerate via the cherry-picked `convert_hf_to_gguf.py` from source HF
  models — not blocking, but adds ~1 day if it happens.
