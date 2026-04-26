# Sprint 004: Rebase Hybrid-Safe DFlash Speculative Decoding

## Overview

Qwen3.6-27B and Qwen3.6-35B-A3B are both hybrid models with 75% recurrent-state layers. That changes the sprint completely. `seq_rm` is not the right rollback mechanism for these targets, and any plan centered on partial-block removal or quantized `seq_rm` work is solving the wrong problem. The hard prerequisite is rebasing the RotorQuant fork onto mainline that already contains speculative checkpointing from PRs #19493 and #22227, then proving that checkpoint save/restore is correct for our deferred-K KV cache in both staging and quantized forms.

The sprint should therefore treat speculative checkpointing as load-bearing infrastructure, not an implementation detail. Phase 1 is a strict rebase gate: bring in mainline checkpoint support, restore our deferred-K hooks, and re-run the four-KV PPL sweep against the current `docs/BENCHMARK-REPORT.md` baselines. Phase 2 is the real architecture risk: add a fork-level checkpoint fidelity test that exercises both deferred f16 K staging and quantized planar/iso layouts, and benchmark snapshot overhead before spending time on DFlash polish. Only after those two gates pass do we cherry-pick EAGLE3 (#18039) and DFlash (#22105).

This sprint is opinionated about scope. It explicitly does not include `seq_rm`-aware quantized rollback, partial-block dequant/requant, or a standalone autoregressive speculative profile as a proxy for correctness. Checkpoint snapshot/restore supersedes those ideas for hybrid targets. The primary performance gate is Qwen3.6-27B at `planar3` reaching 1.5-2.0x decode speedup on the quicksort prompt with thinking off while preserving exact greedy token output versus target-only generation. Qwen3.6-35B-A3B ships as experimental: it must be correct, benchmarked, and documented, but it is not the hard speed gate because upstream hybrid/MoE data already shows recurrent fallback and wider expert activation can erase most of the theoretical gain.

## Use Cases

1. **Hybrid-safe speculative decoding on shipped models**: Both public Qwen3.6 profiles gain a speculative path that is correct for recurrent-state layers instead of relying on `seq_rm`.
2. **Higher decode throughput on the 27B dense profile**: Users running `qwen36-27b` on 24-32 GB GPUs can trade extra draft-model VRAM for materially better single-slot decode speed.
3. **Experimental MoE speculative serving**: Advanced users can benchmark DFlash on `qwen` behind an explicit experimental gate without destabilizing the default production profile.
4. **Reproducible regression control for the fork**: The repo gains a repeatable checkpoint-fidelity and greedy-equivalence harness that catches deferred-K corruption before it shows up as “DFlash bugs.”
5. **Documentation that matches reality**: The README, benchmark report, and Docker profiles describe the hybrid constraint, the checkpoint prerequisite, the actual speedup target, and the draft-model VRAM tradeoff.

## Architecture

The architectural rule for this sprint is simple: speculative decoding starts only after deferred K conversion is complete, and rollback uses checkpoint restore plus accepted-prefix replay, never `seq_rm`.

```text
prefill
  -> K stored in deferred f16 staging
  -> convert_deferred_keys()
  -> steady-state K stored as planar3/planar4/iso3/iso4

decode loop (single slot only in Sprint 004)
  -> save checkpoint of target memory state
  -> run DFlash draft block
  -> verify draft block on target
  -> full accept: keep new state, discard checkpoint
  -> partial accept/reject:
       restore checkpoint
       replay accepted prefix through target
       continue decode
```

Two invariants matter more than anything else:

1. The checkpoint must capture the actual RotorQuant memory backend, not a dequantized logical view. Deferred-K staging and post-conversion planar/iso buffers are both real state and both must round-trip bit-exactly.
2. Speculative verify is not allowed while deferred-K staging is live. If the target has not yet run `convert_deferred_keys()`, the speculative path must refuse to arm and fall back to normal decode.

The runtime surface in this repo should stay conservative. `docker-compose.yml` and `docker/entrypoint.sh` should expose two new profiles, `qwen36-27b-dflash` and `qwen36-dflash`, both forced to `N_PARALLEL=1`. The 27B profile is the supported benchmark path. The MoE profile is present but requires `EXPERIMENTAL=1` and is documented as correctness-first, speedup-uncertain. Draft models inherit the target KV cache type by default: `planar3` for Qwen3.6-27B and `iso3` for Qwen3.6-35B-A3B. Mixed KV tuning is out of scope for v1.

As of April 26, 2026, the upstream dependency picture is clear enough to plan against:

- `#19493` is merged and provides speculative checkpointing.
- `#22227` is merged and wires hybrid fallback into the speculative path.
- `#18039` and `#22105` are still draft PRs, so this sprint should cherry-pick pinned SHAs instead of waiting for merge timing.

## Implementation

### Phase 1: Rebase Onto Checkpoint-Capable Mainline (~25% of effort)

**Files:**
- `src/llama-context.cpp` (fork) — Restore deferred-K hooks on top of mainline speculative checkpoint support
- `src/llama-kv-cache.cpp` (fork) — Validate deferred conversion behavior after rebase
- `common/speculative.cpp` (fork) — Audit compatibility with the rebased speculative API surface
- `docker/Dockerfile` — Pin the rebased fork commit used by the image build

**Tasks:**
- [ ] Rebase `feature/planarquant-kv-cache` onto current `ggml-org/llama.cpp` master that already includes `#19493` and `#22227`.
- [ ] Resolve `src/llama-context.cpp` conflicts by preserving RotorQuant deferred-K behavior and adopting upstream checkpoint-based hybrid fallback instead of reviving old rollback assumptions.
- [ ] Run the full L1 KV regression sweep for all four KV types across the two Qwen3.6 production models and compare against `docs/BENCHMARK-REPORT.md` with a hard ±0.05 PPL tolerance.
- [ ] Confirm the rebased fork still loads both Qwen3.6 hybrid architectures and that `linear_attention` layers execute correctly with RotorQuant enabled.
- [ ] Freeze the rebased fork SHA in the Docker build before any DFlash work begins.

### Phase 2: Prove Checkpoint Fidelity for Deferred-K and Quantized K (~30% of effort)

**Files:**
- `src/llama-kv-cache.cpp` (fork) — Expose any missing state needed by checkpoint save/restore
- `src/llama-context.cpp` (fork) — Refuse speculative verify while deferred staging is live
- `tests/test-checkpoint-deferred-k.cpp` (fork) — Bit-exact save/restore coverage for staging and quantized K states
- `scripts/bench_speculative.py` — Snapshot-cost and decode-throughput measurement harness

**Tasks:**
- [ ] Add a fork-level C++ test that exercises `checkpoint save -> mutate state -> restore` for both deferred f16 K staging and post-conversion `planar3`, `planar4`, `iso3`, and `iso4` K layouts.
- [ ] Assert bit-equality after restore for the underlying K-state buffers; do not accept “numerically close” here.
- [ ] Add an explicit runtime guard that disables speculative verify until `convert_deferred_keys()` has completed for the active sequence.
- [ ] Measure checkpoint overhead at 65K and 262K contexts on the rebased fork. Treat the result as a hard planning gate, not a benchmark appendix.
- [ ] If the upstream checkpoint mechanism does not snapshot RotorQuant’s real backend buffers correctly, add the minimum RotorQuant-specific checkpoint hook needed to copy raw state. Do not add `seq_rm` fallback code.

### Phase 3: Cherry-Pick EAGLE3 and DFlash on Top of the Rebased Fork (~25% of effort)

**Files:**
- `src/models/eagle3.cpp` (fork) — EAGLE3 draft model support from `#18039`
- `src/models/dflash.cpp` (fork) — DFlash draft model support from `#22105`
- `convert_hf_to_gguf.py` (fork) — DFlash/EAGLE3 GGUF conversion support
- `gguf-py/gguf/constants.py` (fork) — Draft-model metadata support
- `common/speculative.cpp` (fork) — Integrate DFlash into the checkpoint-safe speculative path
- `examples/speculative-simple/speculative-simple.cpp` (fork) — Reference CLI for correctness and benchmark work
- `scripts/validate_dflash.py` — z-lab reference comparison harness

**Tasks:**
- [ ] Cherry-pick pinned SHAs from `#18039` and `#22105` onto the rebased fork; resolve conflicts after the checkpoint work is stable, not before.
- [ ] Validate GGUF conversion for the target and DFlash draft models using the upstream conversion path plus the z-lab reference expectations.
- [ ] Run greedy equivalence tests on at least three prompts with `--temp 0 --top-k 1 --seed 42`, comparing target-only output against target+DFlash output token-for-token.
- [ ] Benchmark Qwen3.6-27B with thinking off and `--draft-max 16`; the hard success gate is `>= 1.5x` on the quicksort prompt, with `2.0x` as stretch rather than requirement.
- [ ] Benchmark Qwen3.6-35B-A3B as experimental only. Document the result even if it is near-flat or slightly negative.

### Phase 4: Repo Runtime Integration, Profiles, and Docs (~20% of effort)

**Files:**
- `docker-compose.yml` — Add `qwen36-27b-dflash` and `qwen36-dflash` single-slot profiles
- `docker/entrypoint.sh` — Add target-plus-draft launch configuration and experimental gating
- `README.md` — Document speculative checkpointing prerequisite, profiles, and realistic speed targets
- `docs/BENCHMARK-REPORT.md` — Add speculative decoding section with correctness and throughput results
- `tests/test_dflash_e2e.py` — End-to-end greedy equivalence test at the repo level
- `Makefile` — Add convenience targets for the DFlash profiles and benchmark harness

**Tasks:**
- [ ] Refactor `docker/entrypoint.sh` to accept a draft model, DFlash enablement flags, and an `EXPERIMENTAL=1` gate for the MoE path.
- [ ] Add `qwen36-27b-dflash` as the supported benchmark profile and `qwen36-dflash` as the explicitly experimental MoE profile, both with `N_PARALLEL=1`.
- [ ] Preserve all existing non-speculative profiles and model-cache behavior; no re-download churn and no default-profile regressions.
- [ ] Add an end-to-end integration test that forces at least one rejection path and still proves target-only greedy equivalence after checkpoint restore.
- [ ] Update README and benchmark docs to explain that checkpoint snapshot/restore, not `seq_rm`, is the rollback mechanism for Qwen3.6.

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `src/llama-context.cpp` (fork) | Modify | Reconcile deferred-K hooks with mainline checkpoint-based speculative fallback |
| `src/llama-kv-cache.cpp` (fork) | Modify | Ensure checkpoint save/restore sees the real deferred and quantized K state |
| `common/speculative.cpp` (fork) | Modify | Adopt checkpoint-safe speculative control flow for RotorQuant targets |
| `src/models/eagle3.cpp` (fork) | Add via cherry-pick | Enable EAGLE3 draft model support required by DFlash stack |
| `src/models/dflash.cpp` (fork) | Add via cherry-pick | Add DFlash draft model execution |
| `convert_hf_to_gguf.py` (fork) | Modify | Convert DFlash/EAGLE3 models to GGUF with correct metadata |
| `gguf-py/gguf/constants.py` (fork) | Modify | Register new GGUF metadata/constants for the draft models |
| `examples/speculative-simple/speculative-simple.cpp` (fork) | Modify | Reference CLI for checkpoint-safe speculative validation |
| `tests/test-checkpoint-deferred-k.cpp` (fork) | Create | Bit-exact checkpoint test for deferred-K and quantized K layouts |
| `scripts/validate_dflash.py` | Create | Compare llama.cpp output against the z-lab DFlash reference |
| `scripts/bench_speculative.py` | Create | Measure checkpoint overhead, acceptance rate, and decode speedup |
| `tests/test_dflash_e2e.py` | Create | End-to-end rejection-path and greedy-equivalence validation |
| `docker-compose.yml` | Modify | Add DFlash profiles and single-slot constraints |
| `docker/entrypoint.sh` | Modify | Plumb draft-model launch config and experimental gating |
| `README.md` | Modify | Document the hybrid checkpoint requirement and user-facing profiles |
| `docs/BENCHMARK-REPORT.md` | Modify | Publish speculative correctness and throughput results |
| `Makefile` | Modify | Add convenience targets for DFlash run and benchmark commands |

## Definition of Done

- [ ] The fork is rebased onto a mainline commit that already contains `#19493` and `#22227`, and RotorQuant’s existing KV functionality survives the rebase.
- [ ] The L1 PPL sweep for all four KV cache types across the two Qwen3.6 production models matches `docs/BENCHMARK-REPORT.md` within ±0.05 PPL.
- [ ] `tests/test-checkpoint-deferred-k.cpp` proves bit-exact checkpoint restore for deferred f16 staging and for quantized `planar3`, `planar4`, `iso3`, and `iso4` K layouts.
- [ ] The speculative path refuses to run before deferred-K conversion is complete; no new `seq_rm`-based partial-rollback path is introduced.
- [ ] Greedy 256-token generation with target-only and target+DFlash yields identical token sequences on at least three prompts.
- [ ] Qwen3.6-27B at `planar3` achieves at least `1.5x` decode speedup on the quicksort prompt with thinking off, with the measured range documented for the theorem and travel prompts as well.
- [ ] Qwen3.6-35B-A3B launches and benchmarks behind `EXPERIMENTAL=1`, with correctness proven and performance documented even if the speedup is marginal.
- [ ] `qwen36-27b-dflash` and `qwen36-dflash` are added without breaking the existing Docker profiles or model-cache behavior.
- [ ] `README.md` and `docs/BENCHMARK-REPORT.md` explain the hybrid rollback constraint, checkpoint prerequisite, and actual measured results.

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Upstream checkpoint save/restore does not capture RotorQuant’s real K-state buffers | High | High | Make Phase 2 a hard gate with a bit-exact C++ checkpoint test before DFlash integration |
| Checkpoint overhead at long context erases most speculative speedup | High | High | Benchmark 65K and 262K early; fail fast if 27B cannot clear the 1.5x gate instead of hiding the problem behind doc work |
| `src/llama-context.cpp` conflicts are deeper than expected during rebase and cherry-pick | High | High | Rebase first, prove KV parity first, then cherry-pick DFlash on top of a stable checkpoint-aware branch |
| MoE verify activates more experts and delivers weak or negative speedup | High | Medium | Ship the MoE profile as experimental only and keep 27B as the core success gate |
| Open draft PRs `#18039` and `#22105` keep moving upstream | Medium | Medium | Pin SHAs as of sprint start and treat later upstream changes as follow-up work, not in-sprint churn |
| Target plus draft model VRAM pressure makes default quants infeasible on smaller GPUs | Medium | Medium | Keep Sprint 004 public path focused on single-slot 27B and document draft-model VRAM tradeoffs clearly |

## Security Considerations

- The sprint adds new model-download paths for draft GGUFs, so the draft model registry must stay explicit and pinned rather than accepting arbitrary remote paths from user input.
- Experimental DFlash profiles should not become the default serving path until correctness and rollback behavior are proven on hybrid models.
- The checkpoint-fidelity test is also a safety control: silent restore corruption is worse than a hard runtime refusal because it can produce wrong model output without obvious failures.

## Dependencies

- Mainline `ggml-org/llama.cpp` after merged PRs `#19493` and `#22227`
- Open draft PRs `#18039` and `#22105`, cherry-picked by pinned SHA rather than awaited by merge status
- Existing RotorQuant fork branch `feature/planarquant-kv-cache`
- Current PPL and throughput baselines in `docs/BENCHMARK-REPORT.md`
- Existing Docker runtime shape in `docker-compose.yml` and `docker/entrypoint.sh`
- Prebuilt or converted DFlash GGUF artifacts for Qwen3.6-27B and Qwen3.6-35B-A3B
- GPU access on the RTX 5090 for checkpoint-cost and throughput measurement
- No dependency on `seq_rm` partial-block work; that line of work is explicitly out of scope

## Open Questions

1. **What counts as “rebase complete”?** Resolved: the rebase is not complete until the full L1 PPL sweep passes for all four KV types on both Qwen3.6 production models. A compiling fork is not enough because this rebase is load-bearing for speculative correctness.
2. **How should speculative checkpointing interact with deferred-K and quantized planar/iso K?** Resolved: treat checkpoint fidelity as the central engineering problem of the sprint. The checkpoint must snapshot the actual memory backend, not a dequantized view. Add a bit-exact C++ save/restore test for both deferred staging and post-conversion quantized K. Operationally, speculative verify is forbidden until deferred staging is drained. Snapshot cost is an early benchmark gate at 65K and 262K; if whole-state copies keep 27B below `1.5x`, the sprint should surface that as the blocker instead of inventing `seq_rm` workarounds.
3. **Which KV cache type should the draft model use?** Resolved: the draft inherits the target’s KV type by default. `planar3` stays paired with Qwen3.6-27B and `iso3` stays paired with Qwen3.6-35B-A3B. Mixed-KV tuning is noise for v1, and the draft is hybrid too, so it needs the same checkpoint-safe semantics anyway.
4. **Should we wait for `#22105` to merge upstream?** Resolved: no. As of April 26, 2026, `#19493` and `#22227` are already merged and unblock the architecture. `#18039` and `#22105` are still draft PRs, so the right move is to rebase now and cherry-pick pinned SHAs.
5. **How much parallel-slot work belongs in this sprint?** Resolved: none beyond preserving existing non-speculative profiles. Sprint 004 speculative benchmarking is single-slot only, and both DFlash profiles should force `N_PARALLEL=1`.
6. **What is the realistic speedup gate?** Resolved: Qwen3.6-27B must hit `1.5-2.0x` on the quicksort prompt with thinking off. `2.0x` is stretch, not minimum. Qwen3.6-35B-A3B is experimental and may legitimately land near flat because hybrid fallback plus MoE verify overhead is structurally expensive.
7. **What is out of scope now that the hybrid constraint is understood?** Resolved: no `kv_cache_quantized_seq_rm()`, no partial-block dequant/requant work, and no standalone autoregressive `qwen36-spec` profile as a correctness proxy. In scope instead are checkpoint fidelity, snapshot-cost measurement, and an explicit hybrid-architecture audit after rebase.
8. **Where should the new tests live?** Resolved: both levels are mandatory. The fork gets `tests/test-checkpoint-deferred-k.cpp` as the load-bearing bit-exact checkpoint test, and this repo gets `tests/test_dflash_e2e.py` plus the validation scripts for end-to-end greedy equivalence and benchmark reporting.
