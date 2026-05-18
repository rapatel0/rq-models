# Qwen3.6 27B MTP Multislot 4090 Result

Date: 2026-05-18

Branch: `sprint/007-mtp-multislot`

Image: `localhost:32000/rotorquant:v7-4090-qwen36-mtp-idlefix`

## Summary

`N_PARALLEL=2` with `draft-mtp` no longer crashes under sparse/partial slot
occupancy. The crash was a null dereference in `common_speculative_draft()`: the
post-draft accounting loop dereferenced every slot's `dp.result`, including
idle slots that were not drafting and therefore had no result pointer.

The fix skips non-drafting slots before reading `dp.result`. This preserves the
existing B1 path and lets the server complete a two-slot MTP workload.

The preview profile is still not throughput-promotable. In the non-traced
512-token run, two concurrent lanes aggregated slightly less throughput than one
lane on the same two-slot server.

## Live Configs Tested

Production B1 restored after testing:

```text
--ctx-size 196608 --parallel 1
--cache-type-k q4_0 --cache-type-v q4_0
--cache-ram 8192 --flash-attn on
--ubatch-size 32
--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75
--no-warmup
```

Preview `np=2` test config:

```text
PREVIEW=1 MTP_MULTISLOT=1
--ctx-size 131072 --parallel 2
--cache-type-k q4_0 --cache-type-v q4_0
--cache-ram 0 --flash-attn on
--ubatch-size 32
--spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.75
--no-warmup
```

## Results

| Test | Result |
|---|---:|
| B1 acceptance probe, 256 tokens | 72.35 t/s |
| B1 draft acceptance | 172 / 330 = 52.1% |
| Traced `np=2`, 2 lanes x 128 tokens | 63.19 aggregate t/s |
| Traced `np=2` pod restarts | 0 |
| Non-traced `np=2` server, 1 lane x 512 tokens | 80.38 aggregate t/s |
| Non-traced `np=2` server, 2 lanes x 512 tokens | 75.70 aggregate t/s |
| Non-traced `np=2` scaling | 0.94x |

Server logs from the non-traced two-lane run reported MTP acceptance around
62-63% per slot, with per-slot decode settling around 39-43 t/s. That means the
MTP draft path is operational across two slots, but the added draft-context work
still cancels the batching benefit for this workload.

## Trace Notes

The traced run emitted 687 `spec_trace:` lines. Two were malformed because
regular server logs interleaved with stderr JSON; the remaining valid lines show:

- `server_target_decode` with `active_slots=2` on 31 events.
- `common_draft_impl` with `active_slots=2` on 28 events.
- `server_draft_done` after the partial-occupancy draft that previously crashed.
- Valid traced acceptance of 192 / 244 draft tokens = 78.7%.

The first fixed partial-occupancy sequence is the key correctness signal:

```text
server_draft_collect active_slots=1 draft_tokens_requested=126
mtp_draft_end active_slots=1 draft_tokens_generated=4
common_draft_impl active_slots=1 draft_tokens_generated=4
common_draft_end active_slots=1
server_draft_done active_slots=1 draft_tokens_generated=4
server_target_decode active_slots=2
```

Before the fix, the process segfaulted after `mtp_draft_end` and before
`common_draft_impl` / `server_draft_done`.

## Promotion Decision

Keep B1 as the only recommended production MTP profile.

`np=2` is now safe enough to keep as an explicit preview/debug profile, but it
does not meet the throughput promotion gate. The next useful work is a broader
correctness harness and a deeper profile of why the draft context cost dominates
the two-slot aggregate path.

## Validation

- Patch applies cleanly to a fresh upstream
  `7ba22c6a0918b5db16029c2a120bf04a56e78b78` checkout.
- Local Apple Silicon `llama-server` build passes.
- `llama-server --help` still advertises `draft-mtp` and RotorQuant KV types.
- `bash -n docker/entrypoint.sh` passes.
- `python3 -m py_compile scripts/mtp_probe.py scripts/bench_n_parallel.py`
  passes.
- Compose preview config renders.
- Helm preview config renders.
- Homelab image build and rollout pass for
  `v7-4090-qwen36-mtp-idlefix`.
- Production restored to B1 on the fixed image after preview testing.
