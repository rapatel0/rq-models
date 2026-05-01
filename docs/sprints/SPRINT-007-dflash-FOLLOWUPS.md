# Sprint 007-dflash Follow-ups

Items discovered during Sprint 007 execution that need future work.

## F-024: VRAM-shadow ckpt needs cells[] metadata snapshot

**What**: `vram_seq_checkpoint::save()/restore()` copies only the
recurrent layer tensor data (`r_l`, `s_l`) via cudaMemcpyDeviceToDevice.
It does not snapshot the recurrent memory's `cells[]` array (per-cell
pos, seq_id, tail, src, plus head/used). The host path's
`llama_state_seq_*_data_ext` serializes and restores both.

**Why**: Discovered during Sprint 007 Phase 4 smoke test. With
`LLAMA_SPEC_VRAM_CKPT=1`, Round 1 succeeds, Round 2's `balloc->init`
fails with "Invalid input batch" because the recurrent memory's
seq_pos_max returns positions one-verify-batch ahead of where save
captured. Codex peer review (gpt-5.3-codex high) identified the
metadata gap. Reproduced consistently with debug fprintf showing
pre-save pos_max=38, balloc-time pos_max=54 (= 38 + DRAFT_N_MAX-1).

**Severity**: **Critical** for the VRAM-shadow path to be usable.
Currently the path is in the codebase but `LLAMA_SPEC_VRAM_CKPT` must
be off in production. The host path remains the default (~38% wallclock
save tax measured Sprint 006).

**Suggested sprint**: Sprint 009 or later — Sprint 008 is the EAGLE3
productionization stub. If the speculative decoding architecture
changes substantially with EAGLE3, this work may move with it.

**Files**:
- `src/llama-vram-checkpoint.{h,cpp}` (add `cells_snapshot`,
  `head_snapshot`, `used_snapshot`; update save/restore)
- `src/llama-memory-recurrent.h` (may need to expose cells/head/used
  to `vram_seq_checkpoint` as friend, or expose getters)
- `tools/server/server-context.cpp` (no changes needed; the env-toggle
  and call sites are already wired)

## F-025: Sprint 007 docker-build is full-rebuild every change

**What**: Every iteration of `docker build -f docker/Dockerfile.local
--build-context fork-src=...` recompiles ggml-cuda from scratch
(~25 min) even when the only change is to `tools/server/server-context.cpp`.
The COPY layer cache invalidates the entire build stage when source
changes, so cmake rebuilds everything.

**Why**: Discovered during Sprint 007 debug iteration. Each build
cycle was ~25 min, and we hit ~5 cycles for the VRAM debugging,
totalling ~2 hours of rebuild time.

**Severity**: Important — debug iteration on llama.cpp fork changes
is ~25× slower than necessary. Cmake's incremental build inside a
mounted volume would be ~30 seconds for a single .cpp change.

**Suggested fix**: Either (a) add a dev-mode Dockerfile that mounts
`/src/build` as a volume so cmake's incremental builds persist
across container runs, or (b) build the binary on the host and only
containerize the runtime. Option (b) is simpler since the runtime
image just needs the binaries + libs.

**Suggested sprint**: Whenever DFlash track returns to active development.
Not blocking right now since the host path is stable.

**Files**:
- `docker/Dockerfile.local` (would need a `dev` stage that uses ccache
  or volume-mounts the build dir)
- `Makefile` (add a `build-dev` target)

## F-026: spec_t_ckpt_serialize_us 3-way timer split was deferred

**What**: Sprint 007 plan called for splitting `spec_t_ckpt_save_us`
into three fields: `spec_t_ckpt_sync_us`, `spec_t_ckpt_copy_us`,
`spec_t_ckpt_serialize_us`. Sprint 007 shipped the existing single
`spec_t_ckpt_save_us` only (rationale: that's the operator-visible
wallclock; the 3-way split adds analysis precision but no operator
value until VRAM-shadow ships and we want to attribute residual cost).

**Why**: Pragmatic deferral when Phase 4 blocked.

**Severity**: Nice-to-have. The existing single field is enough for
Sprint 008 to make the EAGLE3 vs further-optimization call.

**Suggested sprint**: Bundle with F-024 (VRAM-shadow fix) so the
post-VRAM measurement can attribute residual save cost to the right
bucket.

**Files**:
- `tools/server/server-task.h` / `.cpp` (extend `result_timings`)
- `tools/server/server-context.cpp` (instrument inside the save block)
- `scripts/bench_speculative.py` (parse new fields)

---

## Summary

| Item | Severity | Suggested Sprint | Files |
|------|----------|------------------|-------|
| F-024: VRAM-shadow needs cells[] snapshot | Critical (for VRAM path) | Sprint 009+ | `src/llama-vram-checkpoint.{h,cpp}`, `src/llama-memory-recurrent.h`, server-context.cpp |
| F-025: full docker rebuild on every fork edit | Important (dev velocity) | Whenever DFlash work resumes | `docker/Dockerfile.local`, `Makefile` |
| F-026: 3-way ckpt timer split | Nice-to-have | Bundle with F-024 | server-task.h/.cpp, server-context.cpp, bench_speculative.py |
