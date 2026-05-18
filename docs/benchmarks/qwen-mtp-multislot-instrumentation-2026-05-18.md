# Qwen MTP Multislot Instrumentation

Date: 2026-05-18

Branch: `sprint/007-mtp-multislot`

Upstream base: llama.cpp `b9196` (`7ba22c6a0918b5db16029c2a120bf04a56e78b78`)

## Scope

Phase 1 adds trace-only instrumentation to the upstream RotorQuant patch. It
does not change production defaults and does not attempt the multislot MTP
behavioral fix yet.

The trace is controlled by `LLAMA_SPEC_TRACE=1`. With the variable unset, the
added code only checks a cached boolean and does not call the trace timers.

## Instrumented Surfaces

- `common/speculative.cpp`
  - `common_speculative_process`
  - `common_speculative_draft`
  - `common_speculative_accept`
  - MTP `process`
  - MTP `draft`
  - MTP `accept`
- `tools/server/server-context.cpp`
  - server slot collection before `common_speculative_draft`
  - target `llama_decode` used for speculative verification
  - sampler `common_sampler_sample_and_accept_n`
  - rollback / checkpoint restore decision

## Trace Format

Trace lines are emitted to stderr as JSONL with a `spec_trace:` prefix:

```text
spec_trace:{"component":"server.context","event":"server_verify_accept",...}
```

Machine-readable event fields are recorded in
`docs/sprints/artifacts/SPRINT-007-MTP-TRACE-SCHEMA.json`.

The required sprint fields are covered across the trace events:

| Field | Source |
|---|---|
| `active_slots` | common/server draft and target decode events |
| `draft_batch_n` | MTP process/draft decode and target decode events |
| `draft_tokens_requested` | common/server draft begin events |
| `draft_tokens_generated` | MTP/common/server draft end and verify events |
| `draft_tokens_accepted` | MTP/common accept and server verify events |
| `draft_ms` | MTP/common/server draft events |
| `verify_ms` | server target decode and sampler verify events |
| `accept_ms` | MTP/common accept events |
| `rollback_count` | server verify event |
| `slot_id` | MTP/common accept and server verify events |
| `prompt_tokens` | MTP process and server verify events |
| `predicted_tokens` | server verify event |
| `n_parallel` | all trace families |
| peak VRAM | external `nvidia-smi` capture during homelab run |

## Validation

- Patch applies cleanly to a fresh `b9196` checkout.
- Local Apple Silicon build of `llama-server` passes with CUDA and Metal off.
- `llama-server --help` still advertises `draft-mtp` and RotorQuant KV types
  (`tbq3_0`, `tbq4_0`, `planar3_0`, `iso3_0`, `planar4_0`, `iso4_0`).
- `bash -n docker/entrypoint.sh` passes.
- `python3 -m py_compile scripts/mtp_probe.py scripts/bench_n_parallel.py`
  passes.
- `git diff --check` passes.

## Current Limitation

No runnable Qwen MTP GGUF is present on the laptop. The bottleneck
classification gate still requires a homelab trace run on `gpu-02-4090rtx`
with `LLAMA_SPEC_TRACE=1` and concurrent `np=1/2/4` requests.
