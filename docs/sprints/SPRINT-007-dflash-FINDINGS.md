# Sprint 007-dflash Findings

**Date**: 2026-05-01
**Status**: PARTIAL — host-path fixes shipped, VRAM-shadow wire-up
deferred (Phase 4 gate failed; bug pinpointed but fix is bigger than
the wire-up sprint scoped for).

## Summary

Sprint 007 aimed to wire the existing `vram_seq_checkpoint` class
(Sprint 003 commit `9b191cd87`) into the speculative path so the
~38%-of-wallclock host-PCIe save tax measured in Sprint 006 E3 would
drop to <1%. It also bundled two methodology fixes from the codex peer
review of Sprint 006 (F-022 cumulative counter, F-023 timer split) and
a Sprint 005 number republish.

**What shipped (host path on `feature/sprint-004-rebase-dflash`,
through commit `43c8c1dfe`)**:

1. **F-022 fix**: per-impl cumulative counters (`n_gen_drafts`,
   `n_acc_drafts`, `n_gen_tokens`, `n_acc_tokens`, `n_call_*`) reset
   inside `common_speculative_begin()`. Verified working on the host
   path: `timings.draft_n_generated` is now per-request, not
   cumulative-across-the-slot.
2. **F-023 timer split**: `result_timings` now exposes
   `spec_t_ckpt_save_us`, `spec_t_ckpt_restore_us`, `spec_n_ckpt_save`,
   `spec_n_ckpt_restore`, `spec_rejection_pos_hist`,
   `spec_accepted_prefix_hist` for downstream analysis. (Three-way
   sync/copy/serialize split that was specced in the Sprint 007 plan
   was not added — the existing `spec_t_ckpt_save_us` field is the
   wallclock the operator actually pays, and that's what matters for
   the policy decision in Sprint 008+.)
3. **Sprint 007 wire-up scaffolding** (env-gated, OFF by default):
   - `server_slot::vram_ckpt` field + `spec_ckpt_in_vram` flag
   - Lazy ctor at first save site (eager ctor broke first decode in
     Sprint 007 attempt 1 — see commit `2f65424ea` and the comment
     near `server-context.cpp:1083`)
   - `LLAMA_SPEC_VRAM_CKPT=1` env-toggle to opt in
   - `LLAMA_SPEC_VRAM_CKPT_NOSAVE=1` env-toggle to bisect (ctor only,
     no save call)
   - `llama_synchronize(ctx)` before save/restore to match the host
     path's `state_seq_*_data_ext` semantics
   - 3 call-site swaps in server-context.cpp (save, restore, partial-
     restore) that pick host or VRAM path based on which save fired

**What did NOT ship**:

- VRAM-shadow path is **not functional**. Phase 4 smoke test fails on
  the second speculative round. Diagnosis below.

## VRAM-shadow bug — Phase 4 root cause

With debug `fprintf` instrumentation around `vram_ckpt->save()`:

```
1st round: pre-save pos_max=22 n_tokens=23 draft.size=15  → SUCCEEDS
2nd round: pre-save pos_max=38 n_tokens=39 draft.size=15  →
           balloc->init: KV X=54 Y=39 (gap=15=DRAFT_N_MAX-1)
           "Invalid input batch" — verify decode never runs
```

Round 1 succeeds (full acceptance, 16/16 drafts accepted). Round 2's
save records pos_max=38 correctly. Then between save() and
`balloc->init`, the recurrent memory's seq_pos_max jumps to 54 — the
position the verify batch's last token would sit at if it had been
decoded. Nothing in the speculative path between save and decode
should touch `cells[]`.

**Codex peer review** (gpt-5.3-codex, high reasoning, ~73K tokens of
trace) identified the structural gap: `vram_seq_checkpoint::save()`
copies tensor bytes only (the `r_l` / `s_l` recurrent layer tensors)
via `cudaMemcpyDeviceToDevice`. It does **not** snapshot the
recurrent memory's `cells[]` array — the per-cell metadata
(`pos`, `seq_id` set, `tail`, `src`, `head`, `used`).

The host path's `llama_state_seq_get_data_ext` serializes both: the
metadata via `state_write_meta` and the tensor data via
`state_write_data`. On restore (`llama_state_seq_set_data_ext`), the
metadata is deserialized via `state_read_meta` which calls
`find_slot()` directly — and `find_slot()` writes
`cells[head + i].pos = ubatch.pos[i]` which **commits the saved
positions to cells\[\]**. So the host path's restore re-establishes
position bookkeeping on the recurrent memory; the VRAM path's
restore just memcpys tensor bytes.

The Round 2 failure mode is consistent with this: even when Round 1's
host-path scenario fully accepts (no restore fired), there's still
some cells\[\]-mutation side-effect happening between rounds that
the host path's snapshot/restore implicitly normalizes but the raw-
memcpy VRAM path doesn't. The exact mechanism (which kernel writes
cells[] beyond pos_max=38 → 54 between save and balloc->init in
Round 2) was not pinned down in the time available, but the fix
direction is clear: VRAM-shadow needs a parallel host-side snapshot
of `cells[]`, restored alongside the D→D tensor copy.

## Outcomes vs Sprint 007 plan

| Plan goal | Outcome |
|-----------|---------|
| Wire vram_seq_checkpoint, drop save tax to <1% | **Failed**. Wire-up is in code but not functional. |
| F-022 cumulative counter fix | ✅ Done. Per-request counters now correct on host path. |
| F-023 timer split (3-way) | ⚠️ Partial. `spec_t_ckpt_save_us` is the operator-visible wallclock; the 3-way split was deferred (low marginal value once VRAM is the actual fix). |
| Re-run E2 sweep at N={4,8,16} on wired-up build | **Skipped**. No point re-running on a non-wired build; same Sprint 006 results would emerge. |
| Pick new DRAFT_N_MAX default | **Deferred to Sprint 008**. Sprint 006's recommendation (N=4) stands until VRAM ships. |
| BENCHMARK-REPORT republish | **No update**. No regime change to publish. |

## Sprint 008 recommendation

**Keep Sprint 008 as the EAGLE3 productionization stub** (per the
existing `SPRINT-008-eagle3.md` plan). The VRAM-shadow wire-up is
filed as a Sprint 008+ followup; it's a self-contained fix
(implement `cells[]` snapshot/restore inside `vram_seq_checkpoint`)
and can sequence after EAGLE3 once we know whether the
speculative-decoding path retains its current shape.

If the user wants VRAM-shadow before EAGLE3, the cells\[\] snapshot
pattern is approximately:

```cpp
struct vram_seq_checkpoint {
    // ... existing tensor shadows ...

    // host-side cells[] snapshot (small: 96 bytes/cell × ~size)
    std::vector<llama_memory_recurrent::kv_cell> cells_snapshot;
    uint32_t head_snapshot;
    uint32_t used_snapshot;

    size_t save() {
        // existing tensor copy ...
        cells_snapshot = mem_recr->cells;
        head_snapshot  = mem_recr->head;
        used_snapshot  = mem_recr->used;
        // sync ...
    }

    size_t restore() {
        // existing tensor copy ...
        mem_recr->cells = cells_snapshot;
        mem_recr->head  = head_snapshot;
        mem_recr->used  = used_snapshot;
        // sync ...
    }
};
```

Estimated effort: 1-2 days, mostly bench validation against the host
path on the qwen profile. The structural change is small; the
validation surface (single-seq, hybrid-only, COMMON_CONTEXT_SEQ_RM_TYPE
_FULL) matches the existing `is_valid()` constraints.

## What this means for the DFlash track

Sprint 005's gate (≥1.3× median) failed on the host path with the
38%-save-tax overhead. Sprint 006 confirmed the overhead is real.
Sprint 007 didn't move the needle. The DFlash speculative path
remains shipped but dormant (env-gated to off in production); the
gate stays unmet pending a working VRAM-shadow path, draft model
distillation, or a smaller-target quantization (Q3_K_M / Q2_K)
strategy.
