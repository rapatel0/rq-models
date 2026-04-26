# Sprint 004: Rebase RotorQuant Fork + DFlash Speculative Decoding for Qwen3.6

## Overview

Sprint 004 should do two things in a strict order: first, move the RotorQuant fork from the currently pinned commit `20efe75cf1127268cb2ad73accd5ccb6f33064ff` onto current `ggml-org/llama.cpp` master at `78433f606fde4d7934a02dcbfd910438d28beccd` without regressing any of the four KV cache types; second, layer speculative decoding on top of that stable base. The rebase is not housekeeping. Both EAGLE3 and DFlash land in the same conflict zone as RotorQuant deferred-K handling, especially `src/llama-context.cpp`, so treating “rebase complete” as a hard quality gate is the only defensible sequencing.

The sprint should ship three runtime profiles, not one. `qwen36-spec` exists only to validate standard autoregressive speculative decoding plus `seq_rm` rollback against quantized KV. `qwen36-27b-dflash` is the production candidate and carries the hard speedup target. `qwen36-dflash` targets Qwen3.6-35B-A3B and ships as experimental because the upstream PR data already shows MoE speedups can collapse to parity or slight slowdown. That means the dense 27B path gets the hard performance bar; the MoE path gets correctness and “no material regression” only.

The sprint should cherry-pick upstream work now rather than wait for merge. PR `ggml-org/llama.cpp#18039` and PR `#22105` are both still open and blocked, but the conflict surface is already known and the current repo is documentation-clean. Waiting for upstream merge lowers one class of risk while increasing another: the longer the fork stays on April-era code, the more expensive the rebase becomes and the less useful the current benchmark baselines are.

## Use Cases

1. **Dense Qwen3.6 serving with meaningful latency reduction**: Run `qwen36-27b-dflash` on a single RTX 5090 and achieve a real decode-speed win over target-only inference while keeping RotorQuant KV compression enabled.
2. **Safe upstream refresh of the production fork**: Rebase the external `johndpope/llama-cpp-turboquant` branch without silently breaking `planar3`, `planar4`, `iso3`, or `iso4` cache behavior.
3. **Rollback correctness under speculative rejection**: Prove that `llama_memory_seq_rm(...)` plus RotorQuant deferred-K logic preserves exact greedy outputs when draft tokens are rejected.
4. **Experimental DFlash support for Qwen3.6 MoE**: Expose an opt-in profile for `Qwen3.6-35B-A3B` that is correct, documented, and benchmarked even if the speedup is modest.
5. **Repeatable differential validation against the PyTorch reference**: Compare llama.cpp DFlash outputs against the z-lab implementation and capture acceptance-rate and token-match metrics in repo-owned scripts and docs.

## Architecture

Fork paths below are relative to the external repository `johndpope/llama-cpp-turboquant`. Repo-local paths are relative to this `turbo` repository.

### System Layout

```text
docker-compose profile
    -> docker/entrypoint.sh
        -> llama-server (rebased RotorQuant fork)
            -> target model context
                -> RotorQuant KV cache types: planar3 / planar4 / iso3 / iso4
                -> deferred-K prefill path
            -> speculative subsystem
                -> standard draft path (`qwen36-spec`) for rollback validation
                -> EAGLE3/DFlash path (`--dflash`) for production feature work
            -> OpenAI-compatible API

validation scripts
    -> scripts/ppl_sweep_rotorquant.py
    -> scripts/validate_speculative.py
    -> scripts/validate_dflash.py
    -> scripts/bench_speculative.py
        -> docs/BENCHMARK-REPORT.md
        -> README.md
```

### Upstream Integration Plan

The rebase target is upstream `master` at `78433f606fde4d7934a02dcbfd910438d28beccd`. After the fork rebases cleanly and passes L1 regression, speculative features should be imported in two batches:

1. EAGLE3 base stack from PR `#18039` using the linear non-merge commits:
   `8fac4b1cc8689bcddd7816889f63c34ef2121232`,
   `ac5667dcc6ea7d820c468e83a6e52bf646e63f71`,
   `5a79c1900f9ed31be400b827424890b774be5dfb`,
   `c0d99e65d2d27f44df7f16e98dc7f28b6fe832cb`,
   `71ba283a6573b3735fa07c39d6e5f8cdeb9a34ab`,
   `3da288d78dc68005502481c50cb8bb3d482a6127`,
   `13a9f31de3c4112c65693db3ed3e08223a069365`,
   `75883cde73fbbd0792cd578cb572ab4382d7b8c3`,
   `7b78bfa9845f3de31e634809a4fdbaf10000bc29`,
   `b3537924efa7552a5e30c64b48800a8d77abfc09`,
   `9fea2434af1b0647ee424a5cc433892956b175c0`,
   `07e2c9707cc9d4a6693e480b8b2619f2fad1c66a`.
2. DFlash delta from PR `#22105` using only the DFlash-specific commits:
   `0724d66e5c85cebf84eba0be4e053872e13998ce`,
   `85a0089e60ff26bf47158471208847a22f6eb3e0`,
   `e344c4a71736e1cdaa25e590a109f694dfb8119f`.

This avoids cherry-picking PR head merge commits (`91b03e4c93d84f0be81ef0d9a1382664b62eadf1` and `e344c4a71736e1cdaa25e590a109f694dfb8119f`’s ancestors via merge state) and keeps conflict resolution reviewable.

### Key Technical Decisions

1. **Rebase gate is strict**: No DFlash work starts until the rebased fork matches current `docs/BENCHMARK-REPORT.md` PPL baselines within ±0.05.
2. **Verify-time deferred K should be treated as batched decode, not prefill**: the fork should add a dedicated verify-append helper in `src/llama-kv-cache.cpp` and wire it from the speculative verify path in `src/llama-context.cpp`, so a 16-token DFlash verification batch does one quantized append pass instead of re-running deferred prefill behavior.
3. **Single-slot only for speculative profiles**: `n_parallel > 1` should be rejected in `tools/server/server-context.cpp` for both `params_base.speculative.eagle3` and `params_base.speculative.dflash`. This is already upstream behavior for EAGLE3 and should become explicit for DFlash.
4. **Draft KV defaults should match the target**: the repo should allow `DRAFT_KV_CACHE_TYPE` override, but the default should be `${KV_CACHE_TYPE}` so the target and draft follow the same storage semantics unless benchmarking proves otherwise.
5. **Dense is the hard gate, MoE is experimental**: `qwen36-27b-dflash` must hit the speed target; `qwen36-dflash` only needs correctness and no severe regression.

### DFlash Data Flow

The upstream PR already provides the important symbol hooks. The sprint should preserve them and only adapt the RotorQuant interaction points:

```text
target model forward
    -> llama_context::extract_dflash_features(...)
    -> llama_get_dflash_target_features(ctx_tgt)
    -> common_speculative_state_dflash::draft(...)
        -> DFlash encoder context
        -> llama_set_dflash_accumulated_target_ctx(ctx_dft_dec, ...)
        -> DFlash decoder context
        -> sample draft block
    -> target verify pass
        -> llama_memory_seq_rm(...) on rejection
        -> RotorQuant verify-append helper for committed tokens only
```

The critical integration point is not `src/models/dflash.cpp`; it is the handoff between speculative rollback and RotorQuant’s deferred-K state machine.

## Implementation

### Phase 1: Rebase the Fork and Lock the Baseline (~30% of effort)

**Files:**
- `docker/Dockerfile` — replace `ARG ROTORQUANT_COMMIT=20efe75cf1127268cb2ad73accd5ccb6f33064ff` with the rebased fork commit created from upstream `78433f606fde4d7934a02dcbfd910438d28beccd`
- `scripts/ppl_sweep_rotorquant.py` — new regression harness with `run_perplexity()`, `parse_final_estimate()`, and `run_matrix()`
- `docs/BENCHMARK-REPORT.md` — capture post-rebase confirmation table before speculative changes
- `src/llama-context.cpp` — resolve rebase conflicts while preserving deferred-K behavior
- `src/llama-kv-cache.cpp` — preserve `convert_deferred_keys()` semantics and isolate any verify-path adjustments
- `ggml-cuda/CMakeLists.txt` — adjust only for upstream build drift
- `ggml-cuda/cpy-planar-iso.cu` — verify our template instantiations survive the rebase unchanged
- `ggml-cuda/set-rows-planar-iso.cuh` — verify unchanged CUDA write path
- `ggml-cuda/planar-iso-constants.cuh` — verify unchanged packing constants
- `ggml-cuda/fattn-common.cuh` — verify FA dispatch still routes RotorQuant KV types correctly

**Tasks:**
- [ ] Rebase `feature/planarquant-kv-cache` from `20efe75cf1127268cb2ad73accd5ccb6f33064ff` onto `78433f606fde4d7934a02dcbfd910438d28beccd`.
- [ ] Resolve `src/llama-context.cpp` conflicts conservatively: keep RotorQuant deferred-K behavior first, then port upstream API changes around it.
- [ ] Reject any “quick fix” that changes `ggml-cuda/*` kernels unless a regression test proves it is required. The PRs do not touch those files.
- [ ] Create `scripts/ppl_sweep_rotorquant.py` rather than extending the legacy shell scripts; it should run 4 KV types × 2 corpora × 2 models and emit a machine-readable JSON summary plus markdown-ready tables.
- [ ] Run the L1 gate on Qwen3.6-35B-A3B and Qwen3.6-27B against the current numbers in `docs/BENCHMARK-REPORT.md`. Do not proceed until every cell is within ±0.05 PPL.
- [ ] Update `docker/Dockerfile` to pin the newly rebased fork commit only after the L1 gate passes.

### Phase 2: Validate `seq_rm` + Quantized KV with Standard Draft Speculation (~15% of effort)

**Files:**
- `docker-compose.yml` — add `qwen36-spec` profile and single-slot constraints
- `docker/entrypoint.sh` — add `build_speculative_args()`, `parse_draft_model_config()`, and `download_model_if_missing()`
- `Makefile` — add `run-qwen36-spec`, `validate-spec`, and `bench-spec`
- `scripts/validate_speculative.py` — new target-vs-draft correctness harness with `run_case()`, `extract_token_ids()`, and `assert_identical_greedy()`
- `docker/test.sh` — add speculative profile smoke coverage
- `common/speculative.cpp` — validate rollback path against quantized KV
- `tools/server/server-context.cpp` — reject `n_parallel > 1` for speculative profiles
- `src/llama-kv-cache.cpp` — add a verify-batch append helper for committed tokens
- `src/llama-context.cpp` — call the verify-batch helper from the speculative acceptance path

**Tasks:**
- [ ] Add a `qwen36-spec` profile that uses the existing autoregressive draft model path, not DFlash, to isolate rollback correctness from block-diffusion complexity.
- [ ] Implement a dedicated verify-append fast path in `src/llama-kv-cache.cpp` so accepted speculative batches are handled like decode appends, not as fresh prefill.
- [ ] Confirm that `llama_memory_seq_rm(...)` plus the new verify-append path produces identical greedy outputs to target-only decoding across at least three prompts and at least one forced-rejection case.
- [ ] Add the same single-slot restriction to `tools/server/server-context.cpp` for DFlash that upstream already applies to EAGLE3; speculative features should fail fast instead of silently corrupting shared slot state.
- [ ] Keep this phase green before cherry-picking DFlash. If `qwen36-spec` is not exact, DFlash debugging will be noise.

### Phase 3: Cherry-pick EAGLE3 and DFlash into the Rebases Fork (~25% of effort)

**Files:**
- `common/arg.cpp` — import `--eagle3` and `--dflash` flags
- `common/common.h` — preserve `common_params_speculative` additions for `eagle3` and `dflash`
- `common/speculative.cpp` — integrate `common_speculative_init_eagle3(...)`, `common_speculative_state_dflash`, and rollback-safe RotorQuant hooks
- `common/speculative.h` — expose EAGLE3 initialization symbols
- `convert_hf_to_gguf.py` — import `DFlashModel` and EAGLE3 conversion logic
- `gguf-py/gguf/constants.py` — add GGUF metadata keys for DFlash and EAGLE3
- `include/llama.h` — expose `llama_set_dflash(...)`, `llama_get_dflash_target_features(...)`, `llama_set_dflash_accumulated_target_ctx(...)`, `llama_model_dflash_block_size(...)`, and `llama_model_dflash_mask_token_id(...)`
- `src/llama-arch.cpp` — register `LLM_ARCH_EAGLE3` and `LLM_ARCH_DFLASH`
- `src/llama-arch.h` — arch enum additions
- `src/llama-context.cpp` — reconcile upstream DFlash feature extraction with RotorQuant KV semantics
- `src/llama-context.h` — add DFlash/EAGLE3 method declarations
- `src/llama-cparams.h` — persist speculative extraction toggles
- `src/llama-graph.cpp` — keep graph routing compatible with new architectures
- `src/llama-graph.h` — add DFlash/EAGLE3 graph builders
- `src/llama-hparams.h` — store DFlash block size, mask token id, and target layer ids
- `src/llama-model-loader.cpp` — load DFlash GGUF metadata
- `src/llama-model.cpp` — preserve `llama_context::set_dflash(...)`, `get_dflash_target_features()`, `set_dflash_accumulated_target_ctx(...)`, and `extract_dflash_features(...)`
- `src/llama-model.h` — DFlash/EAGLE3 model state
- `src/models/dflash.cpp` — upstream DFlash encoder/decoder graph
- `src/models/eagle3.cpp` — upstream EAGLE3 graph
- `src/models/models.h` — register graph builders
- `src/models/llama.cpp` — keep new architecture dispatch intact
- `src/models/openai-moe-iswa.cpp` — required by upstream EAGLE3/DFlash support
- `src/models/qwen3.cpp` — target-model support
- `src/models/qwen35.cpp` — target-model support
- `src/models/qwen35moe.cpp` — target-model support
- `src/models/qwen3moe.cpp` — target-model support
- `tools/server/server-context.cpp` — server-side DFlash/EAGLE3 setup and slot gating
- `examples/speculative-simple/speculative-simple.cpp` — upstream smoke/debug path

**Tasks:**
- [ ] Cherry-pick the EAGLE3 linear stack first, then the three DFlash-only commits. Do not cherry-pick upstream merge commits.
- [ ] Preserve the upstream DFlash symbols exactly enough that future rebases stay tractable: `common_speculative_state_dflash`, `llama_context::set_dflash(...)`, `llama_context::extract_dflash_features(...)`, `llama_set_dflash_accumulated_target_ctx(...)`, and the DFlash GGUF metadata accessors.
- [ ] Keep `src/models/dflash.cpp` as close to upstream as possible. The planned customization point is the RotorQuant verify/rollback edge, not the DFlash graph itself.
- [ ] Audit `common/speculative.cpp` so `dflash_n_past` and RotorQuant deferred-K state advance together on accept and rejection.
- [ ] Do not resurrect the snapshot-and-replay rollback experiment removed in upstream commit `e344c4a71736e1cdaa25e590a109f694dfb8119f`. Dense 27B must prove that `seq_rm` is enough; if MoE needs something stronger, MoE remains experimental.
- [ ] Keep `examples/speculative-simple/speculative-simple.cpp` buildable because it is the fastest way to isolate fork-level failures before Docker is involved.

### Phase 4: Wire the Repo Profiles, Validation Harnesses, and Benchmarks (~20% of effort)

**Files:**
- `docker-compose.yml` — add `qwen36-spec`, `qwen36-dflash`, and `qwen36-27b-dflash`
- `docker/entrypoint.sh` — add draft-model registry entries, speculative argument assembly, and single-slot enforcement
- `Makefile` — add `run-qwen36-dflash`, `run-qwen36-27b-dflash`, `run-qwen36-spec`, `validate-dflash`, and `bench-dflash`
- `docker/test.sh` — smoke-test new profiles when models are present
- `scripts/validate_dflash.py` — differential runner with `run_llama_case()`, `run_reference_case()`, `compare_token_sequences()`, and `compare_acceptance_rate()`
- `scripts/bench_speculative.py` — benchmark runner with `bench_target_only()`, `bench_autoregressive_spec()`, `bench_dflash()`, and `emit_markdown_table()`
- `README.md` — new speculative decoding section, profile table rows, and single-slot caveat
- `docs/BENCHMARK-REPORT.md` — add speculative decoding section and regression appendix
- `docs/QUANTIZATION-GUIDE.md` — note target-plus-draft VRAM budgeting

**Tasks:**
- [ ] Add registry entries in `docker/entrypoint.sh` for the draft GGUFs published in:
  `spiritbuun/Qwen3.6-27B-DFlash-GGUF` at model repo SHA `5e4442a299deb9282b3dfe179de6e8330b19d9de` with `dflash-draft-3.6-q4_k_m.gguf` as default and `dflash-draft-3.6-q8_0.gguf` as optional,
  and `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test` at model repo SHA `3813f31a9fa837b79dce98e6ec49ddeaa4082772` with `Qwen3.6-35B-A3B-DFlash-q8_0.gguf` as the default draft artifact.
- [ ] Expose `SPECULATIVE_MODE`, `DRAFT_MODEL_NAME`, `DRAFT_CTX_SIZE`, `DRAFT_GPU_LAYERS`, `DRAFT_KV_CACHE_TYPE`, and `DRAFT_N_MAX` in `docker-compose.yml`, with `N_PARALLEL=1` hard-coded for speculative profiles.
- [ ] Make `docker/entrypoint.sh` build one command path for target-only, one for standard draft speculation, and one for DFlash. Do not fork the entire script into three parallel copies.
- [ ] Create `scripts/validate_dflash.py` to compare target-only, target+autoregressive draft, target+DFlash, and z-lab reference outputs on the same prompt and seed.
- [ ] Create `scripts/bench_speculative.py` to report decode tok/s, acceptance rate, and exact-token match status in one run.
- [ ] Add `qwen36-dflash` to docs as experimental and `qwen36-27b-dflash` as the primary production candidate.

### Phase 5: Documentation, Release Criteria, and Cleanup (~10% of effort)

**Files:**
- `README.md` — final user-facing commands and caveats
- `docs/BENCHMARK-REPORT.md` — final performance tables and verification evidence
- `docs/QUANTIZATION-GUIDE.md` — VRAM guidance for target-plus-draft deployments
- `docker/Dockerfile` — final pinned fork commit
- `docker-compose.yml` — final profile defaults and comments

**Tasks:**
- [ ] Publish the final rebased fork commit hash in `docker/Dockerfile`, `README.md`, and `docs/BENCHMARK-REPORT.md`.
- [ ] Document the exact benchmark prompt set, seeds, and command lines used for the hard gate.
- [ ] Keep the default non-speculative `qwen` and `qwen36-27b` paths unchanged for users who do not opt in.
- [ ] Leave multi-slot speculative serving, non-greedy validation, and automated GPU CI out of the release path; those are follow-up work, not reasons to block this sprint.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `docker/Dockerfile` | Modify | Pin the newly rebased fork commit and keep Docker build reproducible |
| `docker/entrypoint.sh` | Modify | Add draft-model registry support and a single speculative command builder |
| `docker-compose.yml` | Modify | Add `qwen36-spec`, `qwen36-dflash`, and `qwen36-27b-dflash` profiles |
| `Makefile` | Modify | Add run/validate/bench targets for speculative profiles |
| `docker/test.sh` | Modify | Smoke-test new profiles and their single-slot assumptions |
| `README.md` | Modify | Document speculative profiles, caveats, and commands |
| `docs/BENCHMARK-REPORT.md` | Modify | Record rebase regression results and speculative performance tables |
| `docs/QUANTIZATION-GUIDE.md` | Modify | Explain draft-model VRAM tradeoffs |
| `scripts/ppl_sweep_rotorquant.py` | Create | Post-rebase PPL regression matrix runner |
| `scripts/validate_speculative.py` | Create | Standard draft rollback correctness validation |
| `scripts/validate_dflash.py` | Create | DFlash differential validation against target-only and reference |
| `scripts/bench_speculative.py` | Create | Target-only vs draft vs DFlash throughput and acceptance benchmark |
| `common/arg.cpp` | Modify | Import `--eagle3` and `--dflash` CLI flags from upstream |
| `common/common.h` | Modify | Persist speculative mode flags in common params |
| `common/speculative.cpp` | Modify | Integrate upstream DFlash/EAGLE3 logic with RotorQuant-safe rollback |
| `common/speculative.h` | Modify | Expose EAGLE3 initialization API |
| `convert_hf_to_gguf.py` | Modify | Support DFlash and EAGLE3 GGUF conversion paths |
| `gguf-py/gguf/constants.py` | Modify | Add DFlash/EAGLE3 GGUF metadata keys |
| `include/llama.h` | Modify | Expose DFlash/EAGLE3 public APIs |
| `src/llama-context.cpp` | Modify | Reconcile speculative verification with deferred-K and rollback |
| `src/llama-context.h` | Modify | Add DFlash/EAGLE3 method declarations |
| `src/llama-kv-cache.cpp` | Modify | Add a verify-batch append path for committed speculative tokens |
| `src/llama-model.cpp` | Modify | Keep DFlash feature extraction and accumulated target context support intact |
| `src/llama-model.h` | Modify | Persist DFlash/EAGLE3 model state |
| `src/models/dflash.cpp` | Add/Modify | Upstream DFlash encoder/decoder graph |
| `src/models/eagle3.cpp` | Add/Modify | Upstream EAGLE3 graph |
| `tools/server/server-context.cpp` | Modify | Enforce single-slot speculative mode and initialize draft features correctly |

## Definition of Done

- [ ] The external RotorQuant fork is rebased from `20efe75cf1127268cb2ad73accd5ccb6f33064ff` onto upstream `78433f606fde4d7934a02dcbfd910438d28beccd`.
- [ ] All four KV cache types (`planar3`, `planar4`, `iso3`, `iso4`) match the current `docs/BENCHMARK-REPORT.md` PPL baselines within ±0.05 on both target models before DFlash is enabled.
- [ ] `qwen36-spec` proves greedy exact-match output equality versus target-only decoding across at least three prompts and at least one forced rejection case.
- [ ] `qwen36-27b-dflash` produces identical greedy token sequences to target-only decoding for the sprint prompt set at `--temp 0 --top-k 1 --seed 42`.
- [ ] `qwen36-27b-dflash` achieves at least 2.0x decode throughput versus target-only on the primary dense benchmark prompt set.
- [ ] `qwen36-dflash` is documented and benchmarked as experimental, with correctness passing and no worse than a 10% decode slowdown versus target-only.
- [ ] `scripts/validate_dflash.py` compares llama.cpp against the z-lab reference and records first-64-token agreement plus acceptance-rate deltas.
- [ ] Existing non-speculative profiles (`qwen`, `qwen36-q3`, `qwen36-iq3`, `qwen36-27b`, `qwen36-27b-q3`, `qwen36-27b-iq3`, `reasoning`, `gemma`) still start successfully after the Docker pin update.
- [ ] Tests pass.
- [ ] No regressions.

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `src/llama-context.cpp` rebase conflicts break deferred-K correctness | High | High | Make L1 regression the phase gate and isolate all verify-path edits behind a dedicated helper in `src/llama-kv-cache.cpp` |
| DFlash verification batches accidentally trigger prefill-style deferred conversion | High | High | Add a verify-batch append path and test it first via `qwen36-spec` before cherry-picking DFlash |
| Upstream DFlash/EAGLE3 PRs keep changing while the sprint is in flight | Med | Med | Cherry-pick the explicit commit list above and pin the fork commit in `docker/Dockerfile`; do not track PR head dynamically |
| MoE speedup is negligible or negative | High | Med | Keep MoE experimental and exclude it from the hard speedup gate |
| Speculative decoding corrupts shared slot state under `n_parallel > 1` | Med | High | Reject multi-slot speculative startup in `tools/server/server-context.cpp` and in `docker-compose.yml` defaults |
| Draft GGUF metadata or tokenizer assumptions differ from upstream converter expectations | Med | Med | Validate the exact HF artifacts early and keep `convert_hf_to_gguf.py` changes as close to upstream as possible |
| Differential validation is noisy because prompts, seeds, or samplers diverge | Med | Med | Hard-code the sprint prompt set, `--temp 0 --top-k 1 --seed 42`, and machine-parse token IDs instead of comparing text |
| Rebase work consumes the whole sprint and leaves no time for benchmarks/docs | Med | High | Treat Phase 1 as a fail-fast gate; if it slips, explicitly cut MoE DFlash and leave only dense delivery in scope |

## Security Considerations

- Draft-model downloads should stay registry-based in `docker/entrypoint.sh`; do not accept arbitrary Hugging Face repo strings from unvalidated environment variables.
- `HF_TOKEN` must remain read-only mounted and never be echoed in logs while adding new draft model download paths.
- The speculative profiles should not expose extra ports or sidecar processes; they remain the same single `llama-server` API surface.
- Validation scripts that compare against external reference models should consume local model files and deterministic prompts only; they should not execute untrusted remote code.
- DFlash and EAGLE3 GGUF conversion changes in `convert_hf_to_gguf.py` should preserve current trust assumptions and avoid implicit tokenizer lookup from untrusted paths.

## Dependencies

- Current runtime pin: `johndpope/llama-cpp-turboquant` commit `20efe75cf1127268cb2ad73accd5ccb6f33064ff`
- Rebase target: `ggml-org/llama.cpp` master commit `78433f606fde4d7934a02dcbfd910438d28beccd`
- Upstream speculative PRs:
  - `ggml-org/llama.cpp#18039` EAGLE3 support
  - `ggml-org/llama.cpp#22105` DFlash support
- External reference implementation: z-lab DFlash PyTorch reference
- Draft model artifacts:
  - `spiritbuun/Qwen3.6-27B-DFlash-GGUF` SHA `5e4442a299deb9282b3dfe179de6e8330b19d9de`
  - `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test` SHA `3813f31a9fa837b79dce98e6ec49ddeaa4082772`
- Existing repo baselines in `docs/BENCHMARK-REPORT.md` and Docker profile behavior in `docker-compose.yml`

## Open Questions

1. **When is the rebase “complete”?** Treat the rebase as incomplete until the L1 PPL regression matrix passes. Code that merely compiles is not a valid phase boundary.
2. **How should verify-time deferred K behave?** It should behave as a batched decode append. The sprint should add a dedicated verify-path append helper in `src/llama-kv-cache.cpp` rather than reusing prefill logic or issuing 16 independent token appends.
3. **What KV cache type should the draft model use?** Default the draft to the target’s KV type via `DRAFT_KV_CACHE_TYPE=${KV_CACHE_TYPE}`. Allow override for benchmarking, but do not make f16 the default.
4. **Should we wait for upstream merge?** No. Cherry-pick now from the explicit commit list and pin the resulting fork commit. Waiting helps merge hygiene, but it hurts this fork more than it helps.
5. **Does speculative decoding compose with multi-slot throughput profiles?** Not in this sprint. All speculative profiles are single-slot only; `qwen36-throughput` and `qwen-throughput` stay target-only.
6. **How should MoE success be judged?** Ship `qwen36-dflash` as experimental. Correctness is required; large speedup is not. Dense 27B owns the hard performance bar.
7. **What should be deferred if time gets tight?** Defer non-greedy validation, multi-slot speculative serving, and GPU CI automation first. Do not defer the rebase gate, rollback validation, or dense 27B DFlash benchmarks.
8. **Where should `seq_rm` + quantized-K regression tests live?** In both places. The fork needs the lowest-level rollback/deferred-K test coverage it can support, and this repo needs end-to-end validation via `qwen36-spec` and `qwen36-27b-dflash`.
