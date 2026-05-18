# Qwen3.6 27B MTP 4090 Baseline - 2026-05-18

## Provenance

This baseline preserves the operator-reported homelab matrix from the Sprint
007-mtp planning thread. The implementation agent reported local commits
`5a8d51e`, `16ec054`, `e930e56`, `dfcbd8f`, and `f70afac`, but those commits
were not present in this checkout or on `origin` when Phase 0 began. Treat this
artifact as transcript-derived until raw JSON is imported from the homelab
agent.

## Production Baseline

Production was restored to B1 after the matrix:

```text
MODEL_NAME=qwen3.6-27b-mtp
--ctx-size 196608 --parallel 1
--cache-type-k q4_0 --cache-type-v q4_0
--ubatch-size 32 --cache-ram 8192 --flash-attn on
--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75
--no-warmup
```

Reported runtime state:

- `/slots`: `n_ctx=196608`, `speculative=true`
- VRAM: 21.4 GiB / 23 GiB, about 1.2 GiB free
- Decode probes: 71.4 / 78.1 / 70.1 tok/s for short / medium / long prompts
- MTP acceptance probe: 54.7%

## Multislot Matrix

The decisive A/B matrix compares MTP-off target batching against MTP-on
speculative batching on the same dense 27B Q4 4090 profile.

| Mode | `n_parallel=1` | `n_parallel=2` | `n_parallel=4` | Scaling `np=1 -> 4` |
|---|---:|---:|---:|---:|
| A - MTP off aggregate | 39.7 t/s | 71.5 t/s | 124.5 t/s | 3.14x |
| B - MTP on aggregate | 68.1 t/s | 73.1 t/s | 77.2 t/s | 1.13x |

## Interpretation

- Dense 27B Q4 on the RTX 4090 can benefit from continuous batching. MTP-off
  A4 delivers 3.14x the A1 aggregate throughput.
- MTP-on does not currently scale across slots. B4 is only 1.13x B1 and only
  62% of A4 aggregate throughput.
- The bottleneck is therefore MTP plus multi-slot speculative scheduling, not a
  general dense-27B memory-bandwidth limit.
- Production should remain B1 for solo Hermes-style loops until Sprint 007-mtp
  proves a correct and faster multislot MTP path.

## Promotion Targets For Sprint 007-mtp

The final B4 candidate must satisfy both throughput gates before preview
promotion:

- `B4 >= 1.4 * B1`: at least 95.3 t/s using the preserved B1 value of 68.1 t/s
- `B4 >= 0.70 * A4`: at least 87.2 t/s using the preserved A4 value of 124.5 t/s

Correctness, acceptance, VRAM, and fail-closed deployment gates are defined in
`docs/sprints/SPRINT-007-mtp.md`.
