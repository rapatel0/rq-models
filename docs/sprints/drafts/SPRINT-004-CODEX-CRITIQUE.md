# Codex Critique: `SPRINT-004-CLAUDE-DRAFT.md`

Relative to the revised intent's `## Critical Architecture Finding`, this draft is materially better than the earlier `seq_rm`-centric framing. It correctly recenters the sprint on hybrid-model rollback, upstream speculative checkpointing, and the new checkpointing × deferred-K interaction. The critique below is about what should survive merge and what still needs tightening before this becomes the final sprint doc.

## 1. Strengths To Preserve In Merge

- Preserve the reframe in `## Overview` and `## Architecture`: the draft correctly makes `#19493` + `#22227` the prerequisite and explicitly demotes custom `seq_rm` work. That matches the revised intent's `## Critical Architecture Finding` and removes a now-wrong validation path.
- Preserve `### Three-stage delivery`: the A/B/C staging is clearer than the intent doc's earlier phase mix. In particular, putting a checkpoint-validation gate between rebase and DFlash is the right architectural shape.
- Preserve `### Hybrid speculative checkpointing × RotorQuant deferred-K`: this is the best section in the draft. It names the real novel surface area instead of pretending DFlash is "just another cherry-pick".
- Preserve the addition of a fork-level regression artifact in `### Phase 3: Checkpoint × Deferred-K Validation`. A low-level test is necessary; repo-level token equivalence alone is too far downstream.
- Preserve the more realistic hybrid-speedup framing in `## Overview`, `## Open Questions` item 6, and `## Definition of Done`. The revised target is less promotional and closer to what the intent's hybrid finding implies.
- Preserve the explicit statement in `### Sprint scope diagram` that standalone autoregressive speculative is no longer the correctness vehicle. That is an important merge-time correction.

## 2. Weaknesses With Specific Section References

- `### Hybrid speculative checkpointing × RotorQuant deferred-K` has the right concern but an incomplete cost model. It sizes only planar3 K, then jumps to feasibility conclusions. For a hybrid target, the doc needs explicit state accounting split by:
  `full_attention` KV growth,
  recurrent-state snapshot bytes,
  V-cache bytes,
  target vs draft state.
  Without that split, the section can be directionally right but numerically wrong.
- The same section treats the 75% recurrent / 25% full-attention architecture as a correctness fact, but not as a measurement dimension. The merge doc should require proving which state actually scales with context and which state is fixed per layer. Right now the draft references the hybrid mix, but does not operationalize it.
- `### Phase 3: Checkpoint × Deferred-K Validation` is too buffer-centric. The proposed subtests mutate buffers and restore them, but they do not actually exercise the critical hybrid path: checkpoint -> verify append -> partial rejection -> restore -> replay accepted prefix. That is the path the intent says is now load-bearing.
- `### Phase 3` only names `planar3` and `iso3` subtests, while `### Phase 2` and `## Definition of Done` preserve support for all 4 KV types. If the sprint claims rebased support for `planar4` and `iso4`, the critique is that checkpoint correctness is under-tested for supported modes.
- `### Sprint scope diagram` says standalone autoregressive speculative is out, but `### Phase 6: Benchmarks + Docs` still benchmarks `target+autoregressive draft`. That is a scope contradiction. Either it remains in scope as a comparison baseline, or it is truly out.
- `### Phase 6`, `## Definition of Done`, and `## Risks & Mitigations` disagree on the speedup gate. `## Definition of Done` makes `>=1.5x` a hard gate, but `### Phase 6` says `<1.5x` should be reported and not fail the sprint, and the risk table repeats that softer stance. That needs one answer.
- `## Definition of Done` validates the 8 existing Docker profiles, but it does not require the 2 new DFlash profiles themselves to pass a health check plus one successful completion. For a sprint whose user-facing deliverable is new profiles, that is a hole.
- `## Use Cases` makes concrete latency claims for a 2,000-token reasoning trace before the draft has proved snapshot cost or acceptance behavior on hybrid models. The realism improvement is good; the latency promise is still too eager for merge text.

## 3. Gaps In Risk Analysis

- `## Risks & Mitigations` does not include the risk that the long-context snapshot model is wrong because the draft has not decomposed bytes by `75% recurrent / 25% full_attention`. This is not a small accounting issue; it changes whether 65K/131K/262K conclusions are believable.
- There is no explicit risk for "happy-path L2 passes without any checkpoint restore ever occurring". On hybrid targets, that is the main false-positive mode. Greedy equivalence on 3 prompts is insufficient if all draft blocks are mostly accepted.
- There is no explicit risk for rejection timing around the deferred-K boundary:
  rejection before `convert_deferred_keys()`,
  rejection immediately after conversion,
  rejection after multiple verify appends into already-quantized K.
- There is no explicit risk for checkpoint cost being dominated by restore + replay rather than save alone. `scripts/bench_snapshot_cost.py` is named as snapshot cost, but the user-visible penalty is save + restore + accepted-prefix replay under realistic rejection rates.
- There is no explicit risk for supported-but-unverified KV modes (`planar4`, `iso4`) behaving differently under checkpointing from the production defaults (`planar3`, `iso3`).

## 4. Missing Edge Cases

- Forced rejection coverage is not strong enough. `## Open Questions` item 8 mentions forced rejection in prose, but `### Phase 4`, `### Phase 5`, and `## Definition of Done` do not make it a gate. The merge doc should require at least:
  reject on the first drafted token,
  accept a non-zero prefix then reject,
  repeated rejection on consecutive verify cycles,
  full acceptance as the control case.
- The checkpoint/deferred-K tests in `### Phase 3` should cover transition points, not just static states:
  snapshot while `defer_k=true`,
  snapshot immediately after `convert_deferred_keys()`,
  snapshot after verify has appended into quantized K,
  restore followed by replay of an accepted prefix.
- Long-context validation is underspecified. The draft mentions `65K, 131K, 262K`, but the edge case is not just latency at those contexts. It is whether target + draft + checkpoint headroom still fits, and whether save/restore/replay remains tolerable under non-trivial rejection rates.
- The hybrid-layer split itself needs an edge-case check. Because only `full_attention` layers have token-position KV growth, the sprint should explicitly verify that snapshot bytes grow with that 25% subset while recurrent layers contribute fixed restore state. Right now the draft cites the architecture but does not test the consequence.
- `## Definition of Done` should require at least one correctness run that proves the hybrid fallback path executed. Otherwise the sprint can "succeed" without demonstrating the mechanism that justified the rebase in the first place.

## 5. Definition Of Done Assessment

Short answer: no, not as written.

If the current DoD is met, it would prove:

- the rebase did not obviously regress KV perplexity,
- checkpoint save/restore can round-trip some isolated state,
- DFlash can produce greedy-equivalent output on a small happy-path sample,
- one hybrid target can show a useful speedup on one prompt.

It would not yet prove sprint success on a hybrid target, because it does not require:

- observable checkpoint-restore execution under forced rejection,
- partial-accept / partial-reject correctness across the actual verify loop,
- checkpoint correctness across all supported KV modes,
- a valid cost model split across recurrent vs full-attention state,
- new DFlash Docker profiles themselves to be healthy and usable,
- a single consistent answer on whether `<1.5x` is sprint failure or sprint success with caveats.

The merge standard should be: meeting the DoD proves both correctness and operational viability of speculative decoding on a hybrid architecture, not just that a rebased fork can demo DFlash once.
