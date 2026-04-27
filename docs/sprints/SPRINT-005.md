# Sprint 005 — Phase 2.5 packed-storage wire-in + multi-GPU

**Date kickoff**: 2026-04-27
**Hardware**: NVIDIA RTX 4090 (24 GB) primary; multi-GPU box if available for the TP item
**Model targets**: Qwen3.5-4B (fast iteration), Qwen3.5-9B (mid), Qwen3.6-27B (primary, 4-bit weights via bnb)
**Status**: planned (proposal — please review and lock decisions)

This sprint cashes in the deferred Phase 2.5 deliverable from Sprint 004
— actually using the planar3 packed-byte cache instead of the
lossy-passthrough fp16 cache — and starts the multi-GPU work that was
also a Sprint 004 lock-time deferral.

The Phase 2.5 kernels (`pack_and_scatter_planar3`,
`gather_and_unpack_planar3`) are shipped in
`rq-vllm@feature/rotorquant` commit `d5f060e64`. The work in this
sprint is the FlashAttention-side wire-in plus the dtype/cache-shape
glue.

## Open questions to lock at kickoff

| # | Question                                                  | Default if not locked |
|---|-----------------------------------------------------------|----------------------|
| 1 | **Read-path strategy**: dense-materialize via `gather_and_unpack` then call standard `flash_attn_varlen_func`, OR write a fused `PagedAttention-with-unpack` kernel? | dense materialize (simpler, ~64 MB temp buffer per request, ships in 1 sprint; fused kernel is Sprint 006 if dense proves bandwidth-bound) |
| 2 | **vLLM version**: stay on v0.19.1 or bump to a nightly that has the hybrid + cpu-offload fix (unlocks Qwen3.6-35B-A3B MoE serve)? | stay on v0.19.1 — bump is its own sprint |
| 3 | **Multi-GPU scope**: just verify TP=2 boots and produces sensible output, or full TP+PP perf characterization? | smoke + single-prompt comparison only |
| 4 | **PPL re-measurement**: re-run the Sprint 004 698-token corpus on Phase 2.5 packed storage and confirm Δppl unchanged from Phase 2c? | yes — cheap and answers "did the storage change introduce any quality difference?" |
| 5 | **Qwen3 calibration**: do we want to scope per-model rotation refit (option 1 in `docs/design/PLANAR3_ROTATION_CALIBRATION.md`) into this sprint? | no — defer until we need a non-Qwen3.5/3.6 target |

## Phase plan

### Phase A — Phase 2.5 wire-in (primary deliverable)

**A.1** Flip the cache dtype. In `vllm/utils/torch_utils.py`,
`STR_DTYPE_TO_TORCH_DTYPE["rotorquant_planar3"]` becomes `torch.uint8`.
`kv_cache_dtype_str_to_dtype` keeps the rotorquant_* branch for the
"this is what the model emits" path but the cache shape now uses uint8.

**A.2** Update cache-shape computation. Find where vLLM derives
`[num_blocks, block_size, num_kv_heads, head_size]`-shape fp16 caches
from the model config and add a rotorquant_planar3 branch that
multiplies the last dim by 50/256 (= 0.195) instead. Concretely the
cache shape becomes
`[num_blocks, block_size, num_kv_heads, blocks_per_head * 50]` uint8
where `blocks_per_head = head_size / 128`. The
`packed_cache_shape` helper in `rotorquant_kv.py` already computes
this.

**A.3** Replace the write path. In `flash_attn.py:817` the
rotorquant_* branch currently calls `rotorquant_kv_write` which does
the round-trip then defers to `reshape_and_cache_flash`. New version
calls `pack_and_scatter_planar3(K, V, key_cache, value_cache,
slot_mapping)` directly. That's it on write.

**A.4** Add the read path. The interesting change. Two sub-options:

* **A.4.dense (default)**: pre-attention, call
  `gather_and_unpack_planar3(key_cache, value_cache, block_table,
  seq_lens, ...)` to materialize dense fp16 K/V tensors of shape
  `[num_seqs, max_seq_len, num_kv_heads, head_size]`. Pass those to
  `flash_attn_varlen_func` as if there were no paging. Cost: extra
  64 MB-ish temp buffer per request at 2 k ctx; one extra kernel
  launch per attention forward. Simpler integration; ships in 1
  sprint.
* **A.4.fused (Sprint 006)**: write a fused `PagedAttention-with-unpack`
  CUDA kernel that reads packed bytes from cache and computes
  attention without materializing the dense K/V. Strictly better
  bandwidth; nontrivial CUDA work.

**A.5** Re-validate. Re-run the Sprint 004 quality battery + 698-token
PPL eval + needle test against the Phase 2.5 build. The expectation:
**output should be bit-identical to Phase 2c on the same prompts**,
since the only thing changing is *where the bytes live* (uint8 cache
instead of fp16 cache); the math through the kernel is the same. Any
divergence means the wire-in introduced a bug.

**A.6** Re-measure perf. The Phase 2c benchmarks (`scripts/perf_bench.py`,
`concurrent_bench.py`, `prefill_bench.py`) re-run on Phase 2.5 should
show:
* Decode tok/s recovers to within a few % of fp16 (single
  pack-and-scatter call on write, no unpack on the write path).
* Concurrent N=16 aggregate recovers most of the −8 % cudagraph gap.
* Available KV cache jumps to ~5 × the Phase 2c size (the actual
  memory win — verify in vLLM's `Available KV cache memory` log line).

### Phase B — Multi-GPU smoke (TP=2)

Sprint 004 lock-time deferral. Goal: confirm `--tensor-parallel-size
2` boots with rotorquant_planar3 and produces sensible output on at
least Qwen3.5-4B. We don't have a multi-GPU box on the dev hardware
unless one's been provisioned; this phase is a noop if we're still
single-GPU.

**B.1** Boot Qwen3.5-9B with `--tensor-parallel-size 2 --kv-cache-dtype
rotorquant_planar3`. Verify the cache is allocated correctly per-rank
(packed bytes split across ranks proportionally).

**B.2** Smoke generation: 7-prompt quality battery should match the
single-GPU output token-for-token (TP doesn't change the kernel, just
splits the heads).

**B.3** If TP works: also do TP=2 + PP=1 perf comparison vs single
GPU. (PP=2 deferred — the PR signaling for PP+rotorquant is more
involved.)

### Phase C — Long-context PPL (4 k+)

Sprint 004 PPL was 698 tokens. Re-run on a wikitext-2 sample with
ctx=4096 to:

**C.1** Tighten the per-paragraph noise band well below 1 %.

**C.2** Verify the +1.02 % Δppl(rq3 vs fp16) holds at scale.

**C.3** If it holds, this is also when we'd reasonably claim the
Sprint 004 0.05 % gate was met (the cross-substrate bit-parity result
already implies it, but a wikitext-scale measurement is the
conventional artifact).

### Phase D — Documentation update

**D.1** `docs/MODEL_COMPATIBILITY.md` — add a `Phase 2.5 verified`
column once A.5 passes; update the per-family rows.

**D.2** Update `docs/design/PLANAR3_ROTATION_CALIBRATION.md` with the
Phase 2.5 layout details (the cache shape, `pack_and_scatter`
signature) and any kernel optimizations that came out of the wire-in.

**D.3** SPRINT-005-CLOSE.md once the deliverables land.

## Milestones / what "done" looks like

| Milestone | Definition |
|-----------|------------|
| Phase 2.5 functional | Qwen3.5-4B + rq3 + Phase 2.5 packed storage boots, 7-prompt quality battery output is bit-identical to Phase 2c output. |
| Phase 2.5 perf | Concurrent N=16 cudagraph throughput recovers to within 5 % of fp16 (or better). VRAM headroom for KV cache reaches ≥ 4 × Phase 2c (i.e., ≥ ~270 k tokens at 2 k ctx). |
| Phase 2.5 quality | Δppl(Phase 2.5 vs Phase 2c) on the 698-token corpus is exactly 0 (bit-identical math). |
| Multi-GPU smoke | TP=2 boots and generates coherent output, OR explicit "no multi-GPU hardware available, deferred to Sprint 006." |
| Long-ctx PPL | wikitext-2 ctx=4096 Δppl(rq3 vs fp16) measured on at least Qwen3.5-4B. |

## Hard gate

`Phase 2.5 functional` and `Phase 2.5 quality` are the two hard
gates. Everything else is bonus.

## Out of scope (carries to Sprint 006+)

* Fused `PagedAttention-with-unpack` kernel (A.4.fused). The dense
  read-path materialization in this sprint is the v1 to ship; the
  fused kernel is the proper memory-efficient implementation.
* Per-model rotation calibration to fix Qwen3 family. Defer until we
  actually need a non-Qwen3.5/3.6 model.
* Speculative decoding integration. Original rq-models roadmap goal,
  not Phase 2.5 critical path.
* Qwen3.6-35B-A3B MoE end-to-end serve (gated on vLLM bnb+MoE
  upstream fix, separate sprint to bump vLLM).
* dflash spec-decode integration. User flagged early as
  "interesting but not the priority" — defer until Phase 2.5 lands.

## Risk register

| # | Risk | Mitigation |
|---|------|-----------|
| 1 | The dense read-path materialization (A.4.dense) might be bandwidth-bound at high concurrency, eating the Phase 2c perf win. | Measure first. If the gather+unpack kernel becomes the bottleneck at N=16+, we have the fused kernel work as a clean v2. |
| 2 | vLLM's KV cache allocator doesn't expect uint8 caches in the rotorquant_* path; may need to update internal accounting in `kv_cache_manager.py`. | Phase 2.5 PR includes any necessary allocator-side changes. Worst case: shadow-allocate fp16 sizes but use uint8 storage (wasteful but unblocks). |
| 3 | CUDAGraph capture might not cover the new gather-and-unpack call cleanly (it has dynamic block_table indexing). | Verified Phase 2c works under CUDAGraph; the new kernel uses the same indirection pattern. Worst case fall back to eager mode; gap is recoverable later. |
| 4 | Qwen3.5-9B + Phase 2.5 might exceed the 24 GB budget if our Phase 2.5 cache layout has extra padding. | Test on Qwen3.5-4B first; only escalate to 9B once 4B is bit-identical. |
