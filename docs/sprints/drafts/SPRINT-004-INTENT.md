# Sprint 004 Intent: Rebase Fork + DFlash Block-Diffusion Speculative Decoding

## Seed

Rebase RotorQuant llama.cpp fork onto current master and add DFlash + EAGLE3
block-diffusion speculative decoding for Qwen3.6 models. Three phases:

1. Rebase fork onto mainline; fix KV cache deferred-quant conflicts;
   regression-test all 4 KV types via PPL sweep.
2. Add autoregressive speculative decoding profile to validate
   KV + `seq_rm` rollback interaction.
3. Cherry-pick PR #18039 (EAGLE3) + #22105 (DFlash); build validation harness
   against z-lab pytorch reference + `BENCHMARK-REPORT.md` baselines; add
   Docker profiles `qwen36-dflash` and `qwen36-27b-dflash`.

**Goal**: 2-4× decode speedup on Qwen3.6-27B dense and 1-1.3× on
Qwen3.6-35B-A3B MoE, **on top of** existing RotorQuant KV compression.

**Risk areas**:
- `src/llama-context.cpp` conflicts (both upstream PRs and our fork modify it heavily)
- DFlash drafts are bidirectional non-causal — cannot use existing autoregressive
  speculative path
- Numerical equivalence validation against z-lab reference

**Pre-built GGUFs already exist**:
- [`lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test`](https://huggingface.co/lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test)
- [`spiritbuun/Qwen3.6-27B-DFlash-GGUF`](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF)

## Context

- **RotorQuant llama.cpp fork** at `johndpope/llama-cpp-turboquant` on branch
  `feature/planarquant-kv-cache`, last touched early April 2026. Our additions:
  4 KV cache types (planar3/planar4/iso3/iso4), deferred K quantization
  (`defer_k=true` during prefill, `convert_deferred_keys()` post-prefill),
  CUDA template instances in `cpy-planar-iso.cu` / `set-rows-planar-iso.cuh` /
  `planar-iso-constants.cuh`, FA dispatch in `fattn-common.cuh`.
- **Production deployment**: Docker Compose with profiles `qwen` (Qwen3.6-35B-A3B
  MoE, iso3 default) and `qwen36-27b` (Qwen3.6-27B "dense", planar3 default).
  Throughput: 196 tok/s single-slot decode on the MoE model, 67.5 tok/s on
  Qwen3.5-27B at iso3 (97% of f16). Full PPL/throughput tables in
  `docs/BENCHMARK-REPORT.md`.
- **Sprint 003 was tabled**: SpectralQuant +70.9 PPL failure on Qwen3.5-9B — not
  relevant to this sprint, but confirms our KV compression path (rotation-based,
  not low-rank) is the right family.
- **No VISION.md** — planning from scratch.
- **Hardware**: RTX 5090 (32 GB), 123 GB system RAM, GPU often co-occupied by
  training jobs.

## Critical Architecture Finding (added 2026-04-26 mid-planning)

**Both production target models are HYBRID architectures with recurrent state
layers — *not* pure attention.** Verified from `config.json` on HuggingFace:

| Model | Total layers | `linear_attention` (recurrent) | `full_attention` | Pattern |
|-------|:-------------:|:------------------------------:|:----------------:|---------|
| Qwen3.6-27B "dense" | 64 | **48 (75%)** | 16 | 3:1 repeating |
| Qwen3.6-35B-A3B MoE | 40 | **30 (75%)** | 10 | 3:1 repeating |

The `linear_attention` layers are Gated Delta Net / SSM-style recurrent layers
(`linear_conv_kernel_dim`, `linear_key_head_dim`, `mamba_ssm_dtype: float32`).
Their state is **not decomposable by token position**.

### Direct consequence for speculative decoding

`llama_memory_recurrent::seq_rm()` returns `false` for any partial removal that
includes the final position — by design, because recurrent state cannot be
rolled back per-token. Naive speculative decoding on a hybrid target produces
either:
- A hard error from `seq_rm` rejection-handling, or
- Silent corruption if rejection is handled but recurrent state isn't reset.

### The fix already exists upstream

Mainline llama.cpp added **speculative checkpointing** in two PRs:
- [#19493 — Speculative checkpointing](https://github.com/ggml-org/llama.cpp/pull/19493)
- [#22227 — Hybrid model state fallback](https://github.com/ggml-org/llama.cpp/pull/22227)

Both **already merged** to master. Mechanism: snapshot full memory state
before verify; on rejection, restore + replay only the accepted prefix
through the target. This bypasses `seq_rm` for hybrid layers entirely.

PR #22105 (DFlash) explicitly relies on this: *"Thanks to speculative
checkpointing #19493 and #22227, llama.cpp now supports fallback for hybrid
model states."*

### Implications for this sprint

1. **The rebase is not optional.** It is the *prerequisite* for any speculative
   decoding to function on Qwen3.6 hybrid models. Without #19493 + #22227,
   speculative is broken on every target we ship.
2. **`seq_rm` block-aware partial-truncation work is *not* on the critical path.**
   Speculative checkpointing handles rollback uniformly across both
   attention-layer types; we don't need a custom `kv_cache_quantized_seq_rm()`.
   Drafts in v1 spent ~80 LOC of design effort here that should be reframed.
3. **A NEW interaction surface emerges**: speculative checkpointing + RotorQuant
   deferred-K. The snapshot must capture our K cache state correctly across
   both representations (f16 staging during prefill / planar-iso quantized
   post-conversion). Snapshot cost may be material at long context (a 7 GB
   quantized K snapshot would be expensive to take per verify step).
4. **Realistic speedup is lower than the optimistic case.** PR #22105 reports
   Qwen3.5-9B (hybrid, comparable family to our targets) at **1.34–2.77×**
   speedup, not the Qwen3-8B (pure attention) numbers of 1.77–8.08×. The
   hybrid penalty is real and structural.
5. **There is no "easy dense path" in our shipped models.** Both `qwen` and
   `qwen36-27b` profiles need hybrid-aware speculative. Sprint can't fall
   back to "dense-only delivery" if MoE is too hard — the dense model is
   75% recurrent-layer too.

## Recent Sprint Context

| Sprint | Title | Status |
|--------|-------|:------:|
| 001 | TurboQuant KV Cache Quantization (Python prototype) | Done |
| 002 | Dockerize RotorQuant llama.cpp Server | Done — production |
| 003 | SpectralQuant KV Cache Integration | Validated negative result; abandoned |

Recent commits (last 2 weeks): Qwen3.6 family added, benchmark report rewritten
to match measured numbers, all 16 GB / 24 GB / 32 GB profile defaults aligned
with empirical "MoE → iso, dense → planar" rule. The repo is in a clean,
documentation-current state — good baseline for a fork-rebase sprint.

## Vision Context

No vision document — planning from scratch.

## Relevant Codebase Areas

### Our llama.cpp fork (johndpope/llama-cpp-turboquant, `feature/planarquant-kv-cache`)

| File | Our changes | DFlash PR changes | Conflict risk |
|------|-------------|-------------------|---------------|
| `src/llama-kv-cache.cpp` | Deferred K conversion, double-buffer | None | None |
| `src/llama-context.cpp` | Some (deferred-K hooks) | +377/−6 | **High** |
| `src/llama-graph.cpp/h` | Likely none | +44 | Low |
| `src/llama-arch.cpp/h` | None | +39 | Low (additive) |
| `common/speculative.cpp` | None | +331/−23 | Low |
| `examples/speculative-simple/speculative-simple.cpp` | None | +77/−20 | Low |
| `convert_hf_to_gguf.py` | None | +187 | Low |
| `gguf-py/gguf/constants.py` | None | +53 | Low |
| `src/models/qwen3moe.cpp` | None | +11 | Low |
| `src/models/qwen35*.cpp` | None | +14-25 each | Low |
| `src/models/dflash.cpp` | n/a (new) | +161 | None |
| `src/models/eagle3.cpp` | n/a (new) | +186 | None |
| `ggml-cuda/cpy-planar-iso.cu` | All ours | None | None |
| `ggml-cuda/set-rows-planar-iso.cuh` | All ours | None | None |
| `ggml-cuda/planar-iso-constants.cuh` | All ours | None | None |
| `ggml-cuda/fattn-common.cuh` | Ours (FA dispatch + V dequant) | None | None |
| `ggml-cuda/CMakeLists.txt` | Template instance file list | Possibly mainline drift | Medium |

**Verified via `gh api`**: PR #22105 touches **zero** files in our `ggml-cuda/`
quantization kernel modifications.

### This repo (turbo)

| File | Purpose | Likely sprint touch |
|------|---------|--------|
| `docker-compose.yml` | Service profiles | Add `qwen36-dflash`, `qwen36-27b-dflash`, `qwen36-spec` |
| `docker/entrypoint.sh` | Model launch | Dual-model launch, `--draft-model`, `--dflash` flag |
| `docker/Dockerfile` | Build llama.cpp binaries | Pin updated commit hash post-rebase |
| `Makefile` | Convenience targets | `run-qwen-dflash`, `run-qwen36-27b-dflash` |
| `scripts/benchmark_*.py` | PPL + throughput sweeps | Add `bench_speculative.py` |
| `tests/` | Unit/integration | Add speculative correctness tests |
| `docs/BENCHMARK-REPORT.md` | Reporting | Add §10 Speculative Decoding |
| `README.md` | User docs | Speculative decoding section |
| `docs/QUANTIZATION-GUIDE.md` | GPU tier guide | Note draft model VRAM cost |

## Constraints

- **Project conventions** (from CLAUDE.md, README, recent commits):
  - All git operations use `git add -u` or explicit file lists — never `git add .`
  - Commit format: imperative subject + `Co-Authored-By: Claude <model> <noreply@anthropic.com>`
  - Sprint docs follow the structure in `docs/sprints/` with `-DEFERRED.md` and
    optional `-FOLLOWUPS.md`
  - Documentation must reflect code state — last sprint produced commit
    `8c0907b "Sync all docs with latest benchmarks and code state"` to enforce this
- **Architectural patterns**:
  - KV cache types are first-class GGML types (`GGML_TYPE_PLANAR3` etc.)
  - Deferred K conversion is correctness-critical; verify-time batches need
    explicit handling
  - All Docker profiles auto-download from HuggingFace on first run
  - Per-profile KV defaults set via `KV_CACHE_TYPE: ${KV_CACHE_TYPE:-X}` in
    docker-compose
- **Hardware constraints**:
  - 32 GB VRAM single GPU; draft model VRAM is additive cost
  - GPU often busy with training — sprint must not block on continuous GPU access
  - 123 GB system RAM available for model weights + draft offload
- **Upstream ecosystem**:
  - PR #22105 status: OPEN, MERGEABLE, BLOCKED on review (not yet merged)
  - PR #18039 (parent): OPEN, MERGEABLE, BLOCKED on review
  - Both authored by `ruixiang63`, recently rebased; risk of further upstream
    changes during our sprint

## Success Criteria

### Hard gates (sprint fails if missed)

1. **No KV regression after rebase**: PPL for all 4 KV types × 2 corpora ×
   2 models matches `BENCHMARK-REPORT.md` Section 1 within ±0.05 PPL.
2. **Speculative correctness**: Greedy decode (`--temp 0 --top-k 1 --seed 42`,
   256 tokens) with target-only and target+DFlash produces **identical token
   sequences** on at least 3 prompts.
3. **Decode speedup**: ≥2.0× decode speedup vs target-only on Qwen3.6-27B
   dense for the "Write a quicksort algorithm" prompt with thinking-off.
   This is the lowest-risk speedup target per the PR data.
4. **All 8 existing Docker profiles still launch and serve**: `qwen`,
   `qwen36-q3`, `qwen36-iq3`, `qwen36-27b`, `qwen36-27b-q3`, `qwen36-27b-iq3`,
   `reasoning`, `gemma`. No re-download (model cache preserved).

### Soft gates (sprint succeeds with caveats)

1. **MoE speedup**: Qwen3.6-35B-A3B at iso3 KV achieves ≥1.0× decode (no
   regression). Anything ≥1.2× is a win; the PR's gpt-oss-20B data suggests
   1.0-1.3× is realistic.
2. **Acceptance rate**: Within 10 percentage points of the PR's reported
   numbers for matching prompts.
3. **Validation harness reusable**: The differential test runner can be
   pointed at any (target, draft) GGUF pair without code changes.

## Verification Strategy

### Three-layer validation harness

| Layer | What | Tool | Pass criterion |
|-------|------|------|----------------|
| **L1: KV regression** | PPL sweep, 4 KV × 2 corpora × 2 models | `llama-perplexity` + scripted runner | Match BENCHMARK-REPORT.md ±0.05 PPL |
| **L2: Speculative correctness** | Greedy 256-token decode, target-only vs target+DFlash | `llama-cli` × 2, diff outputs | Identical token sequence at temp=0 |
| **L3: Numerical reference** | Same prompt/seed, z-lab pytorch vs our fork | New `scripts/validate_dflash.py` against z-lab repo | First-64-token match; acceptance rate ±5pp |

### Reference implementations

- **z-lab pytorch reference**: <https://github.com/z-lab/dflash/blob/main/dflash/model.py>
  — clone alongside, set up venv, run with same prompts/seeds.
- **Mainline llama.cpp at HEAD post-rebase**: target for `seq_rm` and
  speculative API surface.

### Edge cases to cover

- Verify-time KV append: does `defer_k` re-trigger or behave like decode?
  (See open question Q1.)
- Rejection rollback (`seq_rm`): does cache integrity hold for partially
  quantized K?
- Non-greedy sampling: does sampler agreement with draft work at temp>0?
  (Sprint scope: greedy only; non-greedy deferred unless trivial.)
- MoE expert activation overhead during parallel verify (PR notes this is the
  reason MoE speedup is lower).
- Long-context (32K+): does DFlash work when KV is already large at the
  rebase-supported max contexts (262K iso3)?

### Testing approach

- **Unit**: New tests in `tests/test_speculative.py`:
  - DFlash GGUF metadata validation (loads, has expected tensors)
  - Sampler determinism with `--temp 0 --seed 42`
- **Integration**: New `tests/test_dflash_e2e.py`:
  - Run `qwen36-27b-dflash` profile; verify health endpoint; submit greedy
    request; assert response equals target-only response.
- **Differential**: `scripts/validate_dflash.py`:
  - Pytorch reference vs llama.cpp fork; assert token sequences match;
    log acceptance rate.
- **Benchmark**: `scripts/bench_speculative.py`:
  - Decode tok/s for (target-only, target+autoregressive draft, target+DFlash)
    on a fixed prompt set.

## Uncertainty Assessment

- **Correctness uncertainty: HIGH**
  - DFlash + RotorQuant deferred K interaction is unverified.
  - `seq_rm` rollback path with quantized K cache has not been tested.
  - Z-lab pytorch reference is single-source; need disciplined diff harness.
- **Scope uncertainty: HIGH**
  - Rebase scope unknown until we attempt — could be 1 day clean or 1 week messy.
  - Both upstream PRs are unmerged; cherry-pick conflict risk against still-evolving
    targets is real.
  - "Add Docker profiles" expands if `entrypoint.sh` needs material refactor for
    dual-model launch.
- **Architecture uncertainty: MEDIUM**
  - Speculative decoding is a *sibling* concern to KV quantization; they're
    largely orthogonal at the kernel level (verified — PR touches zero KV files).
  - But `llama-context.cpp` is a shared edit zone with material conflict potential.
  - Draft model VRAM cost on top of target has not been profiled — may force
    quant-tier downgrades on the 24 GB recommendation.

## Open Questions

These should drive the drafts and the interview. Numbering changed in this
revision to reflect the hybrid-architecture finding above; old Q2 (seq_rm
batch behavior) and Q8 (seq_rm test coverage) are largely *moot* under
checkpointing and are redirected to checkpointing-vs-deferred-K integration.

1. **Phase boundary for "rebase complete"**: Strict gate at L1 PPL regression
   before any speculative work begins. The rebase is now *load-bearing*
   (brings #19493 + #22227 hybrid-state fallback) — broken rebase means no
   DFlash possible at all.

2. **Speculative checkpointing × RotorQuant deferred-K interaction**
   (NEW, supersedes old Q2): the snapshot/restore mechanism in #19493 must
   capture and restore our K cache across both states:
   - **Pre-conversion (prefill in flight)**: K is f16 in the deferred staging
     buffer. `convert_deferred_keys()` has not run.
   - **Post-conversion (steady-state)**: K is in planar3/iso3 quantized layout.

   Three sub-questions:
   (a) Does the upstream snapshot interface in `llama_memory_*::checkpoint()`
       (or equivalent symbol from #19493) walk the cache backend's actual
       buffer, or expect a uniform `tensor_view` it can copy? RotorQuant K
       has a non-standard block layout.
   (b) Cost: at 65K context, planar3 K is ~3.3 GB. A naive snapshot per verify
       step (16-token DFlash block, ~5 ms per step → ~30 verify steps in 256
       tokens of decode) means ~100 GB of GPU<->somewhere copies if the
       snapshot is full state. Does upstream do delta snapshots? COW pages?
   (c) When the verify batch starts mid-prefill (which shouldn't happen
       practically but is possible in edge cases), what is the snapshot
       semantics? Probably: "speculative is a post-prefill operation" by
       construction; document and assert.

3. **Draft model on which KV cache type**: Same answer as before — draft
   inherits target's KV type. But add: draft model is also hybrid? Or pure
   attention? The DFlash draft at `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test`
   inherits target's architecture (it's a few transformer layers initialized
   from the target). So **draft is also hybrid** and also needs speculative
   checkpointing — but the draft's checkpoint cost is small because the draft
   has small KV.

4. **Should we wait for upstream merge of #22105?**: Cherry-pick now. PR is
   BLOCKED on review with no merge ETA. The hybrid-state fallback we *need*
   (#19493 + #22227) is *already* in master; the rebase brings it. Whether
   we also take #22105 (DFlash) on top of the rebased fork is the cherry-pick
   question, distinct from the rebase question.

5. **Single-bench parallel slot interaction**: Single slot for v1.
   Snapshot-restore in checkpointing scales with slot count and recurrent
   state size; multi-slot speculative is genuinely uncharted at the upstream
   layer. Defer to a follow-up sprint.

6. **MoE-vs-dense speedup expectations** (REVISED): both targets are 75%
   recurrent. Use Qwen3.5-9B numbers as the floor, not Qwen3-8B's. PR data:
   - Qwen3.5-9B w/o thinking on coding prompt: **2.77×**
   - Qwen3.5-9B w/o thinking on travel prompt: **1.10×**
   - Qwen3.5-4B (smaller hybrid) w/o thinking: 1.36–5.91×

   So **Qwen3.6-27B realistic dense speedup hard gate ≈ 1.5–2.0×**, not 2.0×
   minimum. The MoE may genuinely come in below 1.0× per upstream gpt-oss
   data; ship as experimental with EXPERIMENTAL=1.

7. **What gets deferred from this sprint?**: Now constrained by hybrid reality.
   Things that drop *out* of scope:
   - Block-aware `kv_cache_quantized_seq_rm()` (not needed; checkpointing
     handles rollback uniformly).
   - Investigation of partial-block dequant/requant (same reason).
   - Standalone autoregressive `qwen36-spec` profile *as a `seq_rm`
     correctness vehicle* — it doesn't validate the right thing because
     hybrid layers go through checkpointing, not `seq_rm`.

   Things that newly appear *in* scope:
   - Snapshot-restore unit test against quantized K.
   - Snapshot cost benchmark at long context (65K, 262K).
   - Architecture audit step at Phase 1 start: confirm rebased fork's
     `linear_attention` layer support is intact (we never built against
     hybrid; this is novel territory for our fork).

8. **Test coverage for hybrid speculative checkpointing × deferred-K**
   (NEW, supersedes old Q8): Where does the test for the new interaction
   surface live?
   - **Fork-level C++ test** (`tests/test-checkpoint-deferred-k.cpp`):
     unit test that runs through `checkpoint_save → modify cache →
     checkpoint_restore` with both f16 staging and planar-iso quantized K,
     asserting bit-equality on restore.
   - **Repo-level integration test** (`tests/test_dflash_e2e.py`): end-to-end
     greedy decode under DFlash with forced rejections, asserts target-only
     equivalence.
   - Both are needed; the C++ test is the load-bearing one (catches snapshot
     bugs before they masquerade as DFlash bugs).

## Items From Prior Sprints

### Deferred items now actionable

| ID | Source | Why now actionable in this sprint |
|----|--------|----------------------------------|
| (none directly) | — | DFlash work is independent of D-001..D-009 (Sprint 001) and D-001..D-009 (Sprint 003 SpectralQuant deferrals). |

### Deferred items NOT in scope (carry forward)

| ID | Source | Recommendation |
|----|--------|----------------|
| D-013 | Sprint 002 — Benchmark CI | Carry forward; new bench scripts make this more valuable but still requires GPU runner setup |
| D-011 | Sprint 002 — Open WebUI | Carry forward; orthogonal to this sprint |
| D-005 (Sprint 003) | Qwen3.6-35B SpectralQuant | Stale — Sprint 003 was a negative result; this item is moot |
| D-007 (Sprint 003) | Benchmark CI duplicate | Same as D-013, carry forward |

### Follow-up items now actionable

None of F-001..F-005 (Sprint 001 follow-ups) are relevant — those are about the
Python `turboquant` library which is research-only at this point. Production is
the C/CUDA fork.
