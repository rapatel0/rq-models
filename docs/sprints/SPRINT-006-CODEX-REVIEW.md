# Sprint 006 Codex Review

## Scope

This review checks the Sprint 006 findings against:

- the published findings and sprint plan
- the stored experiment artifacts
- the actual speculative/checkpoint implementation in the fork under test

The main conclusion is that Sprint 006 found real signals, but several claims are stronger than the artifacts justify. Two cross-cutting problems affect multiple sections:

1. The `draft_n_generated` / `draft_n_acc_tokens` counters used as "real acceptance" are cumulative across requests, not reset per prompt or trial. `common_speculative_begin()` does not reset `n_gen_tokens` / `n_acc_tokens` in [common/speculative.cpp](/home/ravi/repos/llama-cpp-turboquant/common/speculative.cpp:1452), and those counters are incremented cumulatively in [common/speculative.cpp](/home/ravi/repos/llama-cpp-turboquant/common/speculative.cpp:1496) and [common/speculative.cpp](/home/ravi/repos/llama-cpp-turboquant/common/speculative.cpp:1517). The stored baseline artifact shows the telltale pattern `300 -> 600 -> 900` for quicksort across three trials instead of three independent per-request values.
2. E2 and E4 do not preserve the 3-trial-per-cell artifacts the sprint plan called for. The repository only stores single run logs for those experiments. Several summary tables therefore read as stronger than the retained evidence.

## Finding 1: "Checkpoint save is ~38% of wallclock"

**Claim**

`server_get_checkpoint()` save time is the dominant cost center, about 38% of speculative wallclock, and represents the 156 MiB GPU->host copy.

**Supporting evidence**

- E3/E5 baseline summary reports `save %` at 37-40% across prompts.
- The speculative save timer wraps the checkpoint call directly in [server-context.cpp](/home/ravi/repos/llama-cpp-turboquant/tools/server/server-context.cpp:416).
- The server checkpoint path still uses the host snapshot API in [server-context.cpp](/home/ravi/repos/llama-cpp-turboquant/tools/server/server-context.cpp:39).

**Counterfactuals**

- The timer does not isolate the raw memcpy. `llama_state_seq_get_data_ext()` calls `ctx->synchronize()` before the state copy in [llama-context.cpp](/home/ravi/repos/llama-cpp-turboquant/src/llama-context.cpp:3786). So the measured "save" includes at least:
  - waiting for prior async work on the target context to finish
  - host-side state serialization work
  - the device->host copy itself
- The server-observed save time is materially larger than the standalone checkpoint bench. Sprint 004 measured Qwen3.6-27B `PARTIAL_ONLY` host save+restore at about 21.3 ms total and the VRAM shadow path at 0.53 ms total in [BENCHMARK-REPORT.md](/home/ravi/repos/turbo/docs/BENCHMARK-REPORT.md:509). But Sprint 006's Hamlet run reports `8748 ms / 252 saves = 34.7 ms/save` before restore is even counted. That mismatch is strong evidence that the E3 wrapper is timing more than the pure copy.
- Therefore "38% of wallclock" is a valid critical-path statement, but "38% is the GPU->host copy" is overstated.

**Verdict**

`partially sound`

The wallclock-share claim is consistent with the instrumentation. The narrower interpretation that the raw D->H copy itself is 38% is not supported by the code; the timer includes synchronization and other checkpoint-call overhead.

## Finding 2: "Real acceptance scales inversely with N" / "ship N=4"

**Claim**

Acceptance falls sharply as `DRAFT_N_MAX` grows, proving the default block is too large; N=4 should become the new default.

**Supporting evidence**

- The absolute throughput numbers do show a strong N effect. In E2:
  - quicksort: N=8 is slightly best, 62.7 vs 62.2 at N=4
  - the four non-quicksort prompts all improve materially at N=4 vs N=16
  - the prompt-median tps across the five prompts clearly favors N=4 over N=16
- E5's rejection histogram is directionally consistent with smaller blocks on entropic prompts. Hamlet rejects at positions 0-2 in 79% of rejected rounds, and DC trip is close at 68%.

**Counterfactuals**

- The published "real acceptance" numbers are not per-prompt measurements. They are polluted by cumulative counters. The stored baseline artifact already shows cumulative progression across trials, so the E2 percentages should not be treated as true per-prompt acceptance without delta reconstruction.
- Under the `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` path, partial prefixes are discarded on restore in [server-context.cpp](/home/ravi/repos/llama-cpp-turboquant/tools/server/server-context.cpp:3105). That means the acceptance metric effectively rewards full-block success, not "how many early tokens matched before the first miss." Smaller N therefore has a structural advantage even before you ask whether the draft is intrinsically better.
- "96% acceptance at N=4" does not mean 96% of emitted tokens came from the draft. DFlash with `N=4` only drafts `N-1 = 3` tokens per block in [common/speculative.cpp](/home/ravi/repos/llama-cpp-turboquant/common/speculative.cpp:826). Even perfect acceptance would make at most 75% of emitted tokens drafted; the remaining token is the verify/sample token.
- E2 stores only single run logs, not the 3-trial artifacts the sprint plan required. The recommendation is therefore based on a real signal, but a thinner evidence base than the findings document suggests.

**Verdict**

`partially sound`

The throughput conclusion is real: N=16 is too large on this stack, and N=4 is better than N=16 in the stored runs. The stronger acceptance narrative is overstated because the metric is both cumulative and structurally biased toward smaller blocks under the current rollback semantics.

## Finding 3: "Adaptive K=2/3 does not help at temp=0"

**Claim**

Adaptive skip is refuted for `temp=0`; deterministic partials make K=2/3 no better than K=1.

**Supporting evidence**

- The E4 throughput numbers are essentially unchanged between K=1, K=2, and K=3.
- E4 also reports identical save/restore counts to baseline.

**Counterfactuals**

- The experiment does not actually prove the "deterministic partial retry" story, because the code path no longer re-attempts speculative decoding after restore. After rollback, `prompt.size() == dflash_n_past`, so `n_new < 1` and `draft()` returns empty in [common/speculative.cpp](/home/ravi/repos/llama-cpp-turboquant/common/speculative.cpp:833). The server then falls through to single-token decode regardless of K.
- In other words, after the F-018 v1 early-return change, K>=2 is largely inert on this path. That is exactly why the E4 logs show identical `#save` / `#restore`: the supposed "extra speculative retry" never happens.
- So the empirical result "K=2/3 doesn't improve current throughput" is real, but the causal explanation in the findings doc is not established by this experiment.

**Verdict**

`partially sound`

It is fair to say K=2/3 does not help in the current implementation. It is not fair to say Sprint 006 proved adaptive K is the wrong knob because of deterministic-partial retries; the implementation short-circuits before that knob can meaningfully act.

## Finding 4: "No prompt at any N beats target-only" / "DFlash is structurally cost-bound"

**Claim**

Even at optimal N, DFlash loses to target-only on every prompt, so the stack is structurally unfavorable for speculative decoding here.

**Supporting evidence**

- In the stored E2 logs, no prompt beats target-only on user-visible decode tok/s.
- That statement is true for the current shipped server path.

**Counterfactuals**

- The stronger explanation does not follow cleanly from the data. Sprint 006's own measurements identify the unwired checkpoint path as the dominant critical-path cost, and Sprint 004 already contains a validated VRAM-shadow implementation with 31-40x microbenchmark speedup.
- That means the current result may be "structurally cost-bound by the current server wiring," not "structurally impossible on Qwen3.6-27B + 5090."
- A different metric such as `draft_n_generated / wallclock` can absolutely make DFlash look "better" than target-only on some prompts, because it counts discarded speculative work. That is not evidence of better useful-token GPU utilization; it is mostly evidence that the system is burning GPU on uncommitted drafts. Raw drafted-token throughput is not the right success metric here.

**Verdict**

`partially sound`

The narrow claim is correct: with the current host-checkpoint server path, no tested N beats target-only on delivered tok/s. The broader claim that this stack is intrinsically a dead end does not follow, because the dominant overhead is a known, existing, and apparently unwired optimization target.

## Finding 5: "VRAM-shadow checkpoint exists but is not wired into the speculative server path"

**Claim**

There is already a VRAM-resident checkpoint implementation intended for the speculative `PARTIAL_ONLY` path, but `llama-server` still uses the host byte-copy path.

**Supporting evidence**

- The header explicitly says it replaces the host `llama_state_seq_*_ext` path for the speculative `PARTIAL_ONLY` snapshot in [llama-vram-checkpoint.h](/home/ravi/repos/llama-cpp-turboquant/src/llama-vram-checkpoint.h:1).
- The implementation allocates recurrent-state shadows and uses `cudaMemcpyDeviceToDevice` in [llama-vram-checkpoint.cpp](/home/ravi/repos/llama-cpp-turboquant/src/llama-vram-checkpoint.cpp:15).
- The speculative server call sites still use host snapshot APIs:
  - save path in [server-context.cpp](/home/ravi/repos/llama-cpp-turboquant/tools/server/server-context.cpp:39)
  - speculative save timer in [server-context.cpp](/home/ravi/repos/llama-cpp-turboquant/tools/server/server-context.cpp:416)
  - speculative restore in [server-context.cpp](/home/ravi/repos/llama-cpp-turboquant/tools/server/server-context.cpp:3136)
- There is no server-side reference to `vram_seq_checkpoint`.

**Counterfactuals**

- The claimed speedup magnitude needs calibration. Sprint 004's microbenchmark supports a 21.28 ms -> 0.53 ms host-vs-VRAM improvement for Qwen3.6-27B `PARTIAL_ONLY` state in [BENCHMARK-REPORT.md](/home/ravi/repos/turbo/docs/BENCHMARK-REPORT.md:509). That makes the copy-path speedup realistic.
- But Sprint 006's measured "save %" will not disappear 1:1, because the current timer also includes `ctx->synchronize()` and serialization work. A wire-up should still help a lot, but not necessarily erase the entire 38% wallclock tax the way the findings doc suggests.
- The constraints do match the qwen profile:
  - DFlash requires `n_parallel == 1` in [server-context.cpp](/home/ravi/repos/llama-cpp-turboquant/tools/server/server-context.cpp:893), and the `qwen` compose profile sets `N_PARALLEL=1` in [docker-compose.yml](/home/ravi/repos/turbo/docker-compose.yml:71).
  - The server marks contexts that cannot do partial `seq_rm` as `COMMON_CONTEXT_SEQ_RM_TYPE_FULL` in [common.cpp](/home/ravi/repos/llama-cpp-turboquant/common/common.cpp:1424), and speculative Qwen is exactly running in that checkpoint-required mode in [server-context.cpp](/home/ravi/repos/llama-cpp-turboquant/tools/server/server-context.cpp:976).
  - The benchmark evidence already cites Qwen3.6-27B and Qwen3.6-35B-A3B under CUDA.

**Verdict**

`sound`

Your reading is correct. The implementation exists, the speculative server path is not using it, the profile constraints line up, and Sprint 007 should treat this as a wire-up job first, not as a speculative research branch.

## Finding 6: "The skip flag is load-bearing for correctness"

**Claim**

Disabling `spec_skip_next_round` is not a valid performance fix; it is a correctness guard.

**Supporting evidence**

- E1 shows immediate failure when `LLAMA_SPEC_DISABLE_SKIP=1`.
- The partial-restore path explicitly depends on either `spec_skip_next_round` or the `n_new < 1` early-return to avoid drafting against unchanged prefix state in [server-context.cpp](/home/ravi/repos/llama-cpp-turboquant/tools/server/server-context.cpp:3171) and [common/speculative.cpp](/home/ravi/repos/llama-cpp-turboquant/common/speculative.cpp:833).

**Counterfactuals**

- What E1 really proves is "the old path is unsafe if you remove the guard without any other rollback/progress mechanism." It does not prove the current skip heuristic is the only viable design.
- Once VRAM checkpointing exists, alternative progress rules become cheaper to evaluate. That weakens the sprint's framing that skip policy was the primary Sprint 007 lever.

**Verdict**

`sound`

Removing the skip guard without another progress mechanism is not safe on this path.

## Finding 7: "The 1.7B draft is too small for entropic prompts"

**Claim**

The small draft model is the reason prose prompts fail at N=16.

**Supporting evidence**

- E5 shows early rejection on entropic prompts.
- Larger blocks clearly behave badly on those prompts.

**Counterfactuals**

- The evidence does not isolate draft capacity from block-size and checkpoint semantics. On this stack, three effects are entangled:
  - farther lookahead positions are inherently harder
  - partial prefixes are discarded under `FULL` restore
  - checkpoint overhead is charged every round
- Because the acceptance metrics are contaminated and the server never reuses partial prefixes on Qwen, the sprint does not cleanly identify "draft model too small" as the cause rather than "the current server semantics make long horizons non-viable."

**Verdict**

`partially sound`

It is reasonable to say the current draft+block combination fails on entropic prompts. It is not well supported to attribute that primarily to model size.

## Recommendation Review

Sprint 006's recommendation does not fully follow from its own best evidence.

What the data supports:

- N=16 is a bad default on the current path.
- Save/restore overhead is on the critical path and large enough to matter.
- The existing server path is leaving performance on the table.

What the data does not support:

- treating "save-cadence reduction" as the next hard problem before wiring the already-existing VRAM shadow
- concluding from E4 that adaptive skip as a class is dead, rather than that the current implementation does not exercise it meaningfully
- relying on the published acceptance percentages as if they were clean per-prompt measurements

Recommended Sprint 007 sequence:

1. Wire `vram_seq_checkpoint` into the speculative save/restore path first.
2. Keep the host path as fallback for non-CUDA, non-hybrid, or multi-seq contexts.
3. Split instrumentation into:
   - synchronization wait time
   - snapshot copy time
   - restore copy time
   The current "save" timer conflates these.
4. Re-run E2 after the wire-up before freezing a new default `DRAFT_N_MAX`.
5. Only then choose between fixed N=4, fixed N=8, or adaptive N.

## Concrete Sprint 007 Design

Suggested design:

- Add a per-slot optional `vram_seq_checkpoint` object, initialized only when:
  - `GGML_USE_CUDA`
  - `ctx_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL`
  - `n_parallel == 1`
  - the underlying memory object is hybrid/recurrent
- Keep `server_prompt_checkpoint` metadata (`pos_min`, `pos_max`, `n_tokens`) but avoid filling `data` when VRAM shadow is active.
- In speculative save:
  - record metadata
  - synchronize explicitly if needed
  - call `vram_seq_checkpoint::save()`
- In speculative restore:
  - call `vram_seq_checkpoint::restore()`
  - keep the existing `llama_memory_seq_rm(...)`, prompt rollback, sampler rollback, and `common_speculative_rollback(...)`
- Preserve the current host `llama_state_seq_*_ext` path for prompt-cache checkpoints and as a fallback.
- Gate with an env var for bring-up, then flip to auto-enable once verified.

## Risks Of Shipping N=4 Now

- The acceptance tables used to justify N=4 are not trustworthy as written.
- E2 retained only single-run logs, not the 3-trial artifacts promised in the sprint plan.
- Quicksort already prefers N=8 slightly, so a global N=4 default is not "free" even before checkpoint changes.
- N=4 maximizes the number of checkpoint saves. If Sprint 007 wires VRAM shadow, the optimal N may move upward immediately, making the config change premature.
- The strongest measured cost center in Sprint 006 is checkpointing, not N itself. Shipping N=4 first risks optimizing around an avoidable bottleneck.
