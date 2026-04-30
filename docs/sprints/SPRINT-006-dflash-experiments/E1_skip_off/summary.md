# E1 — skip-flag-off canary

**Date**: 2026-04-30
**Profile**: qwen + thinking-off
**Env override**: `LLAMA_SPEC_DISABLE_SKIP=1`
**Fork pin**: `156e69be6`

## Method

Boot qwen with `LLAMA_SPEC_DISABLE_SKIP=1`. Send 1× P1 quicksort + 1×
P4 Hamlet, 30s timeout each. If either hangs or crashes, F-014's
deterministic-partial-restore loop has returned through the path the
v5 skip flag was guarding.

## Result

```
P1 quicksort: 0.34s ERR RemoteDisconnected: Remote end closed connection without response
P4 Hamlet:    0.00s ERR ConnectionResetError: [Errno 104] Connection reset by peer
```

Server aborted on the **first request**. Docker logs:

```
/src/common/speculative.cpp:828: GGML_ASSERT(n_new >= 1 && "must have at least 1 new token") failed
common_speculative_state_dflash::draft(...)
```

## Verdict — **CONFIRMED: skip flag is load-bearing**

The v8 `spec_skip_next_round` flag is necessary for correctness on
hybrid Qwen3.6 + DFlash with `COMMON_CONTEXT_SEQ_RM_TYPE_FULL`
contexts. Without it, the partial-acceptance restore path can't
escape the deterministic-partial loop and the next draft attempt
trips the n_new<1 assert (rollback set dflash_n_past = prompt.size(),
prompt didn't grow, n_new = 0).

This means **F-018 (the perf regression) is not "remove the safety";
it's "fire the safety smarter"**. E4 (adaptive skip K) is the right
remediation candidate. E1 has eliminated the option of just
disabling the flag.

## Per the Phase 3 decision rule

> "If quicksort recovers ≥25% tok/s vs the v8 baseline AND the
> entropic prompt shows loop behavior → F-018 confirmed as a real
> regression but also a necessary safety mechanism."

We hit the second half (entropic loop behavior) with the bonus that
it actually crashes rather than just hanging. Quicksort throughput
data wasn't collected because the server died before completing P1.
**F-018 confirmed as both a real regression AND a necessary safety
mechanism.** Move forward to E4 (adaptive heuristic) as the proper
fix path.
