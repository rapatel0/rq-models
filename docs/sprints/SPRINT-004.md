# Sprint 004: vLLM Substrate Bring-up + planar3 Port

**Date**: 2026-04-25
**Hardware**: NVIDIA RTX 5090 (32 GB), with Qwen3.6-27B Q4_K_M as primary target
**Model targets**: Qwen3.6-27B (primary, dense, planar3 default), Qwen3.5-27B (parity check), Qwen3.6-35B-A3B (future MoE in Sprint 005+)
**Current best baseline (llama.cpp substrate)**: planar3 PPL = 8.20 (wikitext-2, ctx=4096)
**Status**: planned (supersedes Sprint 003 SpectralQuant — see [SPRINT-003.md](SPRINT-003.md) close note)

**Locked decisions** (resolved 2026-04-25 from the Open Questions list):
- **vLLM pin**: `v0.19.1` (latest stable, released 2026-04-18). v0.20.0 is
  prerelease as of kickoff and skipped. Rebase onto a newer tag is its
  own follow-up sprint.
- **Long-ctx eval**: deferred to Sprint 005. PPL eval in this sprint
  uses ctx=4096 to maintain parity with existing llama.cpp baselines.
- **Multi-GPU**: deferred to Sprint 005. Sprint 004 stays on a single
  RTX 5090; tensor parallelism is a follow-up sprint.
- **HF transformers `quantization_config` schema**: defined in this
  sprint as part of Phase 1 (`{"quant_method": "rotorquant",
  "rotorquant_mode": "planar3"}` minimum). Documented in
  `rq-vllm/docs/ROTORQUANT.md`.
- **Upstream tracking cadence**: manual rebase per sprint; no automated
  CI rebase.

---

## Validation Status (2026-04-26)

Sprint kickoff and Phase 0 scaffolding are landed. Phase 0 GPU smoke
test is the next user-driven action.

| Item | Status | Verified by |
|---|---|---|
| `rapatel0/rq-vllm` fork created | ✅ | `gh api repos/rapatel0/rq-vllm` shows `fork: true`, parent `vllm-project/vllm` |
| Pinned to upstream `v0.19.1` | ✅ | tag pushed to fork; working branch `feature/rotorquant` based on it |
| `ROTORQUANT.md` in fork | ✅ | commit `98b61e668` on `feature/rotorquant` |
| `docker/Dockerfile.vllm` syntax | ✅ | builds rq-vllm fork; multi-stage CUDA-dev → CUDA-runtime; bakes rq-vllm commit SHA |
| `docker/entrypoint.vllm.sh` syntax | ✅ | `bash -n` clean; ROTORQUANT_MODE → `--kv-cache-dtype rotorquant_${MODE}` |
| Integration plug-in points identified in vLLM source | ✅ | `vllm/config/cache.py:14-23` (CacheDType Literal), `vllm/v1/attention/backends/flash_attn.py:65-170, 817` (FlashAttention dispatch sites). See Phase 1 anchor table. |
| `docker build -t rq-vllm -f docker/Dockerfile.vllm .` succeeds | ⏸ pending | requires GPU box (RTX 5090) — disk pressure on dev laptop. |
| Phase 0 step 3: `/v1/chat/completions` returns sensible tokens from unmodified vLLM serving Qwen3-27B | ⏸ pending | requires GPU box. |
| Phase 0 step 4: f16 KV PPL baseline | ⏸ pending | requires GPU box. |
| Phase 1 / 2 implementation | ⏸ pending Phase 0 | sequential — substrate must work before integrating RotorQuant. |

**Validation detour 2026-04-26 (does not block Phase 0):** attempted
local-venv validation on the dev laptop via `uv pip install
vllm==0.19.1`. vLLM imports cleanly and `--help` runs, but the model
architecture inspection subprocess fails with `MemoryError` deep in
`email.feedparser.readline()` while
`importlib.metadata.packages_distributions()` parses installed-package
PKG-INFO files. Fully environmental: bug is in uv's bundled
`cpython-3.12.13-linux-x86_64-gnu` stdlib email parser, NOT in vLLM
or torch. `packages_distributions()` runs cleanly when invoked
directly in the same venv — only fails from vLLM's subprocess context.
The docker path uses Ubuntu 22.04's apt-installed Python 3.12 inside
the container which avoids the uv stdlib quirk. **Don't attempt local
venv validation again; go straight to docker on the GPU box.**

## Overview

rq-models currently runs on a llama.cpp fork (`johndpope/llama-cpp-turboquant`,
branch `feature/planarquant-kv-cache`, commit `20efe75`) which has the
RotorQuant iso3/planar3/planar4/iso4 KV cache kernels integrated. This
substrate is production-stable but limited compared to vLLM in three
dimensions that matter for rq-models's roadmap:

1. **Multi-tenant throughput**: vLLM's continuous batching is materially
   stronger than llama.cpp's parallel mode for serving multiple concurrent
   requests on a single GPU.
2. **Speculative decoding**: vLLM has a maintained plug-in architecture
   (Eagle-3, DFlash via Speculators, n-gram lookup). llama.cpp's specdec
   support lags. When dflash matures (~6+ months out) we want to plug
   into a maintained substrate, not chase porting work.
3. **Multi-GPU**: vLLM tensor + pipeline parallelism is years ahead of
   llama.cpp's split-mode.

Sprint 004 brings up a vLLM-substrate path for rq-models with **one quant
variant (planar3, current dense default) ported end-to-end**, validated
against the llama.cpp baseline at PPL parity. This is a **bring-up
sprint**, not a feature-completeness sprint. Other variants (iso3, iso4,
planar4) and MoE models follow in Sprints 005+.

The win condition is a working serving path: `docker compose --profile
qwen36-27b-vllm up` serves Qwen3.6-27B with planar3 KV at PPL within
Δppl ≤ 0.05% of the llama.cpp substrate, with end-to-end aggregate
throughput at N=4 matching or exceeding the llama.cpp path on the same
hardware.

The **substrate fork** is the durable artifact. Once it exists and
serves one model with one quant variant, future sprints add variants
and models without re-doing fork-setup work.

## Use Cases

1. **Multi-tenant production serving**: rq-models users serving multiple
   concurrent chat sessions on a single RTX 5090 / data-center GPU
   benefit from vLLM continuous batching's higher aggregate throughput.
2. **Future speculative decoding research**: when dflash / Eagle-3 /
   lookup decoding stabilize, plugging them into the rq-models stack
   requires the engine to have a maintained specdec interface. vLLM has
   it; llama.cpp lags.
3. **Multi-GPU deployment**: 4× V100, 2× A100, etc. via vLLM tensor
   parallelism. The k8s + Helm chart work on `main` (commit `58c1ee8`)
   already targets multi-GPU; this sprint gives it an engine that
   actually uses the GPUs in parallel.
4. **Alternate substrate for benchmarking**: cross-substrate PPL +
   throughput comparison validates that RotorQuant is the differentiator,
   not a property of one engine's idiosyncrasies. This matters for
   defensibility of the research contribution.
5. **Foundation for non-GGUF model support**: vLLM loads HF transformers
   format directly. Future models that don't have GGUF builds (or have
   them late) can still run on rq-models via the vLLM path.

## Architecture

### Substrate layout

```
rapatel0/rq-vllm   (forked + pinned to upstream v0.19.1)
├── vllm/
│   ├── config/cache.py                          ← MODIFY: extend CacheDType Literal
│   │                                              with rotorquant_planar3 (S004),
│   │                                              + iso3 / iso4 / planar4 (S005)
│   ├── attention/backends/                      ← MODIFY: dispatch rotorquant in
│   │                                              KV-write + paged_attention read path
│   ├── attention/ops/                           ← NEW: paged-block pack/unpack helpers
│   │   └── rotorquant_kv.py                       in Python; dispatches CUDA kernels
│   └── csrc/                                    ← upstream vLLM CUDA kernel dir
│       └── attention/rotorquant/                ← NEW: ported KV pack/unpack kernels
│           └── planar3_kv.cu                    ← Sprint 004 scope
├── ROTORQUANT.md                                ← already present (commit 98b61e668)
└── tests/kv_cache/test_rotorquant_kv.py         ← NEW: parity tests vs llama.cpp
```

### Plug-in points (corrected after vLLM source review)

**Important correction from the original sprint draft**: RotorQuant is
a **KV cache** compression scheme, not a *weight* quantization scheme.
Inspecting the `rapatel0/rq-vllm@v0.19.1` source:

- Weight quantization plugs in via `QuantizationConfig` subclasses in
  `vllm/model_executor/layers/quantization/` (e.g., `gptq.py`,
  `awq.py`). **Not the right surface for RotorQuant.**
- KV cache dtype is registered in `vllm/config/cache.py` as a typed
  `Literal`:

  ```python
  CacheDType = Literal[
      "auto", "float16", "bfloat16",
      "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla",
  ]
  ```

  CLI flag `--kv-cache-dtype` selects from this enum. The dispatch on
  the chosen dtype lives in the attention backend (FlashAttention,
  FlashInfer, Triton attention) where the KV-write kernel converts
  fp16/bf16 → quantized dtype on store, and the paged-attention read
  unpacks it.

The right RotorQuant integration is therefore:

1. **Extend `CacheDType`** in `vllm/config/cache.py` with new values:
   `rotorquant_planar3` (Sprint 004), `rotorquant_iso3`,
   `rotorquant_iso4`, `rotorquant_planar4` (Sprint 005).
2. **Add KV-write packer**: a custom kernel that takes fp16/bf16 K and
   V vectors, applies the planar3 rotation + Lloyd-Max codebook, packs
   into the 3-bpe paged-block layout. Wire into the attention backend's
   `kv_cache_factory` / write path.
3. **Add KV-read unpacker**: dequant from 3-bpe block back to fp16/bf16
   for the attention compute. Wire into the paged-attention kernel's
   read path.
4. **Block-size negotiation**: vLLM's default paged block size is 16
   tokens. RotorQuant 3-bpe layout assumes a different per-block byte
   shape than fp16's `block_size * n_kv_heads * head_dim * 2`. Either
   reuse vLLM's block_size with internal sub-tiles for the 3 bpe
   packing, or override block_size for rotorquant dtypes (likely the
   simpler route — 16-token blocks at 3 bpe = 48 bpe per K-head × 16
   tokens × n_heads × head_dim).

`mitkox/vllm-turboquant`'s patches (4 commits, `c6b2ee9`, `cee479e`,
`5fc73a3`, `7a8a095`) demonstrate **TurboQuant's** integration shape —
they extend the FlashAttention KV path, not the QuantizationConfig
registry. **Read them as reference for the FlashAttention integration
points specifically**, not the QuantizationConfig pattern.

### KV layout reuse

The current rq-models RotorQuant planar3 kernels (in the
`johndpope/llama-cpp-turboquant` fork at commit `20efe75`) implement
the 3 bpe planar rotation + Lloyd-Max codebook. The math is identical
between substrates. What changes:

- **Block addressing**: llama.cpp's KV cache is a contiguous tensor
  per-layer-per-head; vLLM's paged KV is `[num_blocks, block_size,
  n_kv_heads, head_dim]` with logical-block → physical-block
  indirection through a per-sequence `block_table`.
- **Launch boilerplate**: llama.cpp uses GGML kernel-launch helpers;
  vLLM uses `at::cuda::getCurrentCUDAStream()` and torch C++ API.
- **Math (`__device__` inline functions for rotation + Lloyd-Max
  codebook)**: ports near 1:1.

### vLLM version pinning

Pinned to upstream v0.19.1 (released 2026-04-18). See "Locked
decisions" at the top of this document.

## Implementation

### Phase 0 — Fork setup + bring-up of unmodified vLLM (target: 2-3 days)

1. Fork `vllm-project/vllm` to `rapatel0/rq-vllm`. Pin to v0.19.1.
   **DONE 2026-04-25** — fork at https://github.com/rapatel0/rq-vllm,
   tag v0.19.1 pushed, working branch `feature/rotorquant` from that
   tag, ROTORQUANT.md committed.
2. Add Docker build target alongside existing `docker/Dockerfile`'s
   llama.cpp path: `docker/Dockerfile.vllm` builds vLLM with our fork
   as source. **DONE 2026-04-25** — `docker/Dockerfile.vllm` and
   `docker/entrypoint.vllm.sh` committed in rq-models commit
   `0f5116d`.
3. Bring up Qwen3.6-27B serving on the unmodified fork via `docker
   build -t rq-vllm -f docker/Dockerfile.vllm . && docker run --gpus
   all -p 8080:8080 -e MODEL=Qwen/Qwen3-27B rq-vllm`. Sanity: a
   `/v1/chat/completions` returns sensible tokens. **PENDING — must be
   run on the GPU box**. See "Local validation findings" below for
   the env-side issue we hit and why it's not in our scope.
4. Run `scripts/eval_perplexity.py` against the unmodified vLLM
   serving (with f16 KV) on Qwen3.6-27B; record baseline. **PENDING —
   GPU-box work**.
5. Once Phase 0 step 3 succeeds: add the `qwen36-27b-vllm-baseline`
   profile to `docker-compose.yml` and the `run-qwen36-27b-vllm-baseline`
   target to `Makefile`. Deferred from initial scaffold commit to
   avoid landing untested infra.

#### Local validation findings (2026-04-26)

Attempted to validate the substrate via a venv (using `uv venv` →
`uv pip install vllm==0.19.1`) on the development laptop (RTX 4090,
not the RTX 5090 target). vLLM imports fine and the CLI's `--help`
works, but `vllm.entrypoints.openai.api_server` invocations fail
during the model architecture inspection subprocess with a
`MemoryError` deep in `email.feedparser.readline()` while
`importlib.metadata.packages_distributions()` parses installed-package
PKG-INFO files. Stack-walk shows the MemoryError originates in the
uv-managed `cpython-3.12.13-linux-x86_64-gnu` build's stdlib email
parser, NOT in vLLM or torch. `importlib.metadata.packages_distributions()`
runs cleanly when called directly in the same venv — only fails when
called from vLLM's subprocess context. Confirms the bug is environmental
in uv's bundled Python build, not a vLLM regression at v0.19.1.

**Implication for Sprint 004**: don't try to validate via a local venv
on this laptop. The docker path uses Ubuntu 22.04's apt-installed
Python 3.12 inside the container, which is the upstream-blessed
distribution and avoids the uv stdlib quirk. Validation must happen
from the docker image on the RTX 5090 box.

### Phase 1 — RotorQuant KV-cache dtype registration (target: 2-3 days)

**Corrected scope**: RotorQuant integrates as a KV-cache dtype, not a
weight quantization method (see Architecture > Plug-in points).

Concrete code anchors verified against `rapatel0/rq-vllm@v0.19.1`
(2026-04-26):

| Location | What changes |
|---|---|
| `vllm/config/cache.py:14-23` | Extend `CacheDType` Literal with `"rotorquant_planar3"` (Sprint 004); add `"rotorquant_iso3"` etc. in Sprint 005. |
| `vllm/v1/attention/backends/flash_attn.py:65-69` | Append rotorquant dtypes to `FlashAttentionBackend.supported_kv_cache_dtypes`. |
| `vllm/v1/attention/backends/flash_attn.py:165-170` | Add a branch in `supports_kv_cache_dtype` so rotorquant returns `True`. |
| `vllm/v1/attention/backends/flash_attn.py:120-130` | `get_kv_cache_shape`: rotorquant dtypes need a packed-bit shape (3 bpe → block stores `block_size * n_kv_heads * head_size * 3 // 8` bytes per K and per V). Diverges from the f16 `(2, num_blocks, block_size, n_kv_heads, head_size)` shape. |
| `vllm/v1/attention/backends/flash_attn.py:817` (around `reshape_and_cache_flash` call) | Add a branch: if `self.kv_cache_dtype.startswith("rotorquant_")`, call `reshape_and_cache_rotorquant` (new op) instead. |
| `vllm/model_executor/layers/attention/attention.py:165, 426` | The two existing fp8 kv_cache_dtype string-matches are also where a rotorquant branch goes — read path. |

Steps:

1. Edit `vllm/config/cache.py`:
   - Extend the `CacheDType = Literal[...]` union with
     `"rotorquant_planar3"`.
   - Add a docstring explaining what the new value means and pointing
     to the rotorquant_kv backend module.
2. Add `vllm/attention/ops/rotorquant_kv.py` with passthrough
   pack/unpack stubs (write fp16 K,V → fp16 paged blocks unchanged;
   read fp16 paged blocks → fp16 K,V unchanged). Goal: ensure that
   `--kv-cache-dtype rotorquant_planar3` is accepted by the CLI
   without crashing, even though it doesn't actually compress
   anything yet.
3. Wire dispatch in `flash_attn.py` per the table above so that when
   `cache_dtype == "rotorquant_planar3"` the path calls our pack
   stubs, and when reading it calls our unpack stubs.
4. Smoke test (GPU): `docker run ... -e ROTORQUANT_MODE=planar3`
   (entrypoint translates this to the `--kv-cache-dtype
   rotorquant_planar3` flag) loads Qwen3.6-27B, produces sensible
   tokens. Output should be **bit-identical to the `--kv-cache-dtype
   float16` baseline** since pack/unpack are passthrough.

### Phase 2 — planar3 kernel port (target: 1 week)

1. Locate the planar3 KV-write + KV-read CUDA kernels in
   `johndpope/llama-cpp-turboquant@20efe75`. Likely paths to grep:
   `ggml/src/ggml-cuda/cpy.cu`, `ggml/src/ggml-cuda/dequantize.cu`,
   or a planarquant-specific `.cu` file. Identify the rotation +
   Lloyd-Max codebook helpers and the per-block packing code.
2. Add `vllm/csrc/attention/rotorquant/planar3_kv.cu` with two CUDA
   kernels:
   - `planar3_kv_write`: takes `(K, V, slot_mapping, block_table,
     ...)` and writes packed 3-bpe blocks.
   - `planar3_kv_read`: dequantizes packed blocks back to fp16 for
     the attention matmul.
3. Hook the new kernels into `setup.py`'s extension build so they
   compile alongside vLLM's existing CUDA kernels.
4. Replace the passthrough stubs from Phase 1 with calls to the new
   kernels. The `_custom_ops` Python wrapper imports them.
5. Block-size negotiation: if vLLM's default 16-token paged block
   doesn't divide cleanly with the 3-bpe layout (likely it does — 16 ×
   3 / 8 = 6 bytes per token-channel), use the default. Otherwise,
   override `--block-size` for rotorquant dtypes and document.
6. Smoke test (GPU): `docker run ... -e ROTORQUANT_MODE=planar3`
   loads Qwen3.6-27B and generates sensible tokens. Output should
   *not* be bit-identical to fp16 (we now have actual compression),
   but should be high quality (Δppl validated in Phase 3).

### Phase 3 — PPL validation gate (target: 2-3 days)

1. Run `scripts/eval_perplexity.py` against the planar3 vLLM serving on:
   - Qwen3.6-27B (primary)
   - Qwen3.5-27B (parity check)
2. Compare against the llama.cpp planar3 baselines:
   - llama.cpp planar3 PPL on Qwen3.6-27B: TODO (capture as part of this
     sprint if not already in `docs/BENCHMARK-REPORT.md`)
   - llama.cpp planar3 PPL on Qwen3.5-27B: 8.20
3. **Hard gate**: Δppl ≤ 0.05% (i.e., absolute Δ ≤ 0.0041 PPL units at
   8.20). If Δppl exceeds this threshold, kernel port has a correctness
   bug — debug before Phase 4.
4. Save results to `docs/SPRINT-004-PPL-COMPARISON.md`.

### Phase 4 — Throughput benchmark (target: 2-3 days)

1. Run `scripts/bench_n_parallel.py` against both substrates at N ∈ {1,
   2, 4, 8} using matched workload.
2. Record:
   - Aggregate tok/s
   - Per-session tok/s
   - TTFT (time to first token)
   - End-to-end latency at p50, p95
3. **Soft gate**: vLLM aggregate throughput at N=4 should match or
   exceed llama.cpp at N=4 on the same hardware. If significantly
   slower, document why (likely PagedAttention block-size mismatch with
   3-bpe layout — design follow-up for Sprint 005).
4. Save results to `docs/SPRINT-004-THROUGHPUT-COMPARISON.md`.

### Phase 5 — Deployment + closeout (target: 1-2 days)

1. Add `docker/Dockerfile.vllm` and corresponding entry in
   `docker-compose.yml`: `qwen36-27b-vllm` profile.
2. Update `docker/test.sh` to smoke-test the new profile.
3. Update `Makefile` with `make run-qwen36-27b-vllm` target.
4. Update `README.md` with the substrate-choice section: when to use
   llama.cpp path vs vLLM path.
5. Write `SPRINT-004-FOLLOWUPS.md` for any deferred items found during
   execution.

## Files Summary

**New repository**:
- `rapatel0/rq-vllm` (fork of upstream vllm-project/vllm at latest stable
  tag)

**New / modified files in `rapatel0/rq-vllm`** (corrected integration
shape — KV-cache-dtype, not weight quant):
- `vllm/config/cache.py` — extend `CacheDType` Literal
- `vllm/attention/ops/rotorquant_kv.py` — Python pack/unpack dispatch
- `vllm/csrc/attention/rotorquant/planar3_kv.cu` — CUDA kernels
- FlashAttention KV-write/read hooks for the new dtype
- `tests/kv_cache/test_rotorquant_kv.py`
- `ROTORQUANT.md` — already exists (commit `98b61e668`)

**Files in `rapatel0/rq-models`**:
- `docs/sprints/SPRINT-004.md` (this doc)
- `docs/sprints/SPRINT-004-DEFERRED.md` (companion)
- `docs/SPRINT-004-PPL-COMPARISON.md` (Phase 3 output)
- `docs/SPRINT-004-THROUGHPUT-COMPARISON.md` (Phase 4 output)
- `docker/Dockerfile.vllm` (new)
- `docker-compose.yml` (add `qwen36-27b-vllm` profile)
- `Makefile` (add `run-qwen36-27b-vllm` target)
- `README.md` (substrate-choice section)
- `docs/sprints/SPRINT-004-FOLLOWUPS.md` (Phase 5 output, if any items
  emerge)

## Definition of Done

- [ ] `rapatel0/rq-vllm` exists, forked from upstream vLLM at a pinned
      stable tag, recorded in this doc and the in-fork README.
- [ ] Unmodified vLLM fork serves Qwen3.6-27B successfully (Phase 0
      smoke).
- [ ] `CacheDType` Literal extended with `rotorquant_planar3`; pack/unpack
      passthrough stubs in place; FlashAttention KV-write/read dispatch
      wired (Phase 1).
- [ ] planar3 KV pack + unpack CUDA kernels ported and replacing the
      Phase 1 passthrough stubs (Phase 2).
- [ ] `--kv-cache-dtype rotorquant_planar3` (i.e., `ROTORQUANT_MODE=planar3`
      env var) produces sensible tokens for Qwen3.6-27B and Qwen3.5-27B
      (Phase 2 smoke).
- [ ] **PPL parity gate**: Δppl ≤ 0.05% vs llama.cpp planar3 baseline on
      Qwen3.6-27B and Qwen3.5-27B (Phase 3, hard gate).
- [ ] **Throughput parity check**: vLLM aggregate tok/s at N=4 ≥
      llama.cpp aggregate tok/s at N=4 OR documented justification why
      not (Phase 4, soft gate).
- [ ] `docker compose --profile qwen36-27b-vllm up` produces a working
      server reachable at port 8080.
- [ ] `make run-qwen36-27b-vllm` works.
- [ ] Tests in `rq-vllm` repo: `pytest tests/quantization/test_rotorquant.py`
      green.
- [ ] README documents the substrate-choice question explicitly: when
      llama.cpp path, when vLLM path.
- [ ] Closeout follow-ups document written if needed.

## Risks & Mitigations

Ranked by likelihood × impact.

| # | Risk | L×I | Mitigation |
|---|---|---|---|
| 1 | **vLLM internal API churn** during the sprint window | High × Med | Pin to a specific stable tag at kickoff; do not rebase mid-sprint. |
| 2 | **PPL parity miss** (kernel port bug producing silent quality regression) | Med × High | Hard Δppl ≤ 0.05% gate at Phase 3. If failed, debug before Phase 4. Compare per-token KV outputs against llama.cpp directly if needed. |
| 3 | **3-bpe block packing incompatible with vLLM PagedAttention block-size assumptions** | Med × High | Identify in Phase 2; if hit, use a smaller paged-block size or a custom KV layout. Document any deviation as a Sprint 005 follow-up. |
| 4 | **Throughput regression vs llama.cpp** at N=1-4 | Med × Med | Acceptable in Sprint 004 — soft gate. Document and design follow-up. vLLM's strengths are at N≥4 with continuous batching. |
| 5 | **vLLM build/CUDA compatibility issues** with current driver / CUDA version | Med × Med | Docker isolation; pin CUDA version in Dockerfile. Use matrix testing if needed. |
| 6 | **mitkox/vllm-turboquant patches outdated** vs vLLM API we target | Low × Low | We're not basing on those patches, just reading them. If outdated, just read upstream vLLM's quantization code directly. |
| 7 | **Sprint 004 scope creep** to include iso3 / iso4 / planar4 mid-sprint | Med × Low | Strict scope: planar3 only. Other variants are Sprint 005. Document any cross-cutting changes that benefit them as deferred items. |
| 8 | **HF transformers metadata for rotorquant not yet defined** | Low × Med | Define a minimal `quantization_config` schema in this sprint; document for use by future model conversions. |

## Security Considerations

- vLLM fork inherits upstream vLLM's request validation and OpenAI-API
  authentication patterns. No new auth surface added.
- RotorQuant kernel is a passive transform on KV state; doesn't open
  new privilege paths.
- New Docker image must avoid baking secrets (HF tokens, etc.) into
  layers — use build-args + runtime env, mirroring existing
  `docker/Dockerfile`.

## Dependencies

- **Upstream**: `vllm-project/vllm` latest stable tag. Pinned at sprint
  kickoff.
- **Reference**: `mitkox/vllm-turboquant` (read-only, not depended on).
- **Source kernels**: `johndpope/llama-cpp-turboquant` at commit
  `20efe75` for the planar3 reference implementation to port.
- **Hardware**: RTX 5090 32 GB for the bring-up + benchmark; eventual
  multi-GPU follow-up sprints.
- **Validation harness**: existing `scripts/eval_perplexity.py` + the
  newly migrated `scripts/bench_n_parallel.py` (commit `7560c88`).

## Open Questions

All open questions from the original draft were resolved at sprint kickoff
2026-04-25 — see "Locked decisions" at the top of this document. No
remaining open questions at sprint start; new questions discovered
during execution land in `SPRINT-004-FOLLOWUPS.md` per the standard
process.
