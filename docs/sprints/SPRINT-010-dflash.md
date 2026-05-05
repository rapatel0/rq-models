# Sprint 010-dflash: DFlash p_min draft truncation

> **Track suffix**: `-dflash`. Continues the diffusion track. Replaces the
> deleted Sprint 010-eagle3 plan (wrong direction — abandoned diffusion
> for autoregressive).

**Status**: In progress (2026-05-05). Patch landed; bench blocked on GPU
contention.
**Sprint type**: Implementation + measurement
**Created**: 2026-05-05
**Depends on**: Sprint 009 close (Q8 ships opt-in; pipeline solved;
prose acceptance is the remaining lever).
**Estimated effort**: ~1 day single-engineer (modulo GPU availability)

**Branches**:
- Repo: `sprint/010-dflash` (off `sprint/008-dflash`)
- Fork: `feature/sprint-004-rebase-dflash` continues from `5833a843c`

---

## Overview

Sprint 008/009 demonstrated the DFlash speculative pipeline is solved
(VRAM-shadow ckpt at ~1% wallclock save tax, DRAFT_N_MAX=2 default).
Sprint 009 confirmed quantization is at its ceiling (Q8 ~+5% over BF16,
matches spiritbuun's findings). The remaining gap to ≥1.3× median is
**draft acceptance on entropic prose** — Hamlet 0.91× and DC trip 1.03×
at N=2; worse at higher N.

The mechanism: DFlash's diffusion forward pass produces K candidate
tokens in parallel (one decode of the draft model = K logit vectors).
On code prompts, the top-1 probability stays high across all positions —
all K drafts get accepted. On prose prompts, top-1 probability decays
sharply past position 2-3 — the draft is "guessing" and the verify
decode rejects positions 3-15.

**The fix**: in step 4 of `common_speculative_state_dflash::draft()`
(the read-out loop, *not* the diffusion itself), break out when the
diffusion's top-token probability drops below `p_min`. Smaller verify
batch, less wasted verify-decode work on tokens we'd reject anyway.

This is conceptually identical to what upstream PR #22506 did for the
*autoregressive* draft path (and what spiritbuun's `cab1fb5` does in
their fork's DFlash path on top of ~10 prior infrastructure commits).
Our patch is a 5-line cherry-pick-equivalent that doesn't require a
fork rebase.

## Use Cases

1. **Operators get higher median DFlash× on diverse prompts.** With
   p_min ≈ 0.5, prose prompts truncate drafts early; the verify-decode
   pays only for high-confidence drafts; effective acceptance rate
   climbs (because the rejected tail isn't drafted), and per-token
   wallclock improves.

2. **The ≥1.3× hard gate (outstanding since Sprint 005) finally clears
   on the qwen profile** — that's the hypothesis. If p_min sweep shows
   it doesn't, we have hard evidence the remaining gap is irreducible
   without draft retraining.

3. **The patch is upstream-friendly.** It's a small targeted change;
   could be PR'd back to upstream's DFlash code path once the data
   shows it pays off. Increases likelihood of upstream consolidation.

## Architecture

```text
fork pin (5833a843c — patch landed)
  ↓
Phase 1: ✅ Patch in fork's common/speculative.cpp DFlash draft loop
        + DRAFT_P_MIN env in entrypoint.sh + docker-compose.yml
  ↓
Phase 2: Smoke test — boot DRAFT_P_MIN=0.5 N=4, send 3 sequential
        Hamlet prompts. Verify --draft-p-min flag passed; verify
        no crashes; verify acceptance pattern is healthier than
        BF16 N=4 baseline.
  ↓
Phase 3: Sweep p_min ∈ {0.3, 0.5, 0.75, 0.9} at N=4. (N=2 has only
        1 draft position so p_min is moot.) Capture per-prompt
        DFlash× + acceptance.
  ↓
Phase 4: Pick optimal p_min. If a value beats N=2 default's median
        (1.21×) AND keeps worst-case ≥0.9×, flip default.
  ↓
Phase 5: Update README + BENCHMARK-REPORT, write FINDINGS, recommend
        Sprint 011.
```

## Implementation

### Phase 1 — Patch ✅

**Done** in fork commit `5833a843c`:

```cpp
// common/speculative.cpp common_speculative_state_dflash::draft()
for (int i = 1; i < block_size; i++) {
    common_sampler_sample(smpl, ctx_dft_dec, i);
    const auto * cur_p = common_sampler_get_candidates(smpl, true);

    if (i > 1 && params.p_min > 0.0f && cur_p->data[0].p < params.p_min) {
        break;  // F-030: confidence-based early-exit
    }

    const llama_token id = cur_p->data[0].id;
    common_sampler_accept(smpl, id, true);
    result.push_back(id);
}
```

`--draft-p-min` already wired in upstream `arg.cpp` →
`params.speculative.p_min`. Default 0.0 = old behavior.
Skips the check on i==1 so we always commit at least one draft token.

`DRAFT_P_MIN` env added to `docker/entrypoint.sh` and propagated through
`docker-compose.yml`. Translates to `--draft-p-min "$DRAFT_P_MIN"` when
non-zero.

### Phase 2 — Smoke test (BLOCKED on GPU)

**Tasks**:
- [ ] Boot `LLAMA_SPEC_VRAM_CKPT=1 DRAFT_N_MAX=4 DRAFT_P_MIN=0.5
      docker compose --profile qwen up -d`.
- [ ] Verify `--draft-p-min 0.5` appears in the server start command.
- [ ] Send 3 sequential Hamlet prompts. Verify HTTP 200, response
      coherent, draft acceptance ≥40% (vs BF16 N=4's ~16% on Hamlet).

**Phase gate**: 3-of-3 succeed; acceptance noticeably better than
N=4 BF16 baseline.

**Current blocker** (2026-05-05 ~12:10 CDT): VLLM EngineCore PID 3480637
holding 30.7 GB of 32.6 GB. Bench resumes when GPU frees.

### Phase 3 — p_min sweep at N=4

**Tasks**:
- [ ] `DRAFT_P_MIN={0.3, 0.5, 0.75, 0.9}` × 5-prompt × 3-trial bench
      with N=4. Save under
      `docs/sprints/SPRINT-010-dflash-experiments/p_min-{N4-V}/`.
- [ ] Compare to Sprint 008 BF16 N=4 baseline (median 1.02, worst 0.53).

**Phase gate**: full sweep completes; comparison table written.

### Phase 4 — Pick default + optional flip

**Decision matrix**:
- If median ≥1.3× AND worst ≥0.9× at any p_min/N combo: flip default
  (`DRAFT_P_MIN=0.5` etc.). Sprint 005 hard gate finally clears.
- If median 1.0–1.3× AND worst ≥0.9×: ship as opt-in via env, document
  the regime where it wins.
- If no improvement vs N=2 default: ship the env knob (operators may
  still want it for specific workloads), don't flip default. Update
  Sprint 011 recommendation toward distillation.

### Phase 5 — Findings

**Tasks**:
- [ ] `docs/sprints/SPRINT-010-dflash-FINDINGS.md` with:
  - p_min sweep table
  - Hypothesis verdict (does p_min unlock prose without hurting code?)
  - Whether ≥1.3× hard gate clears
  - Sprint 011 recommendation

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `common/speculative.cpp` | ✅ Done (fork) | p_min early-exit in DFlash draft loop |
| `docker/entrypoint.sh` | ✅ Done (repo) | DRAFT_P_MIN env → --draft-p-min flag |
| `docker-compose.yml` | ✅ Done (repo) | DRAFT_P_MIN propagation |
| `docs/sprints/SPRINT-010-dflash-experiments/` | Create | p_min sweep artifacts |
| `docs/sprints/SPRINT-010-dflash-FINDINGS.md` | Create | Outcomes |
| `docs/BENCHMARK-REPORT.md` | Modify (repo) | Republish if gate clears |
| `README.md` | Modify (repo) | Add p_min env to var table; guidance |

---

## Definition of Done

### Hard gates

1. p_min patch lands in fork — ✅ Done.
2. 3-of-3 Hamlet smoke test passes with `DRAFT_P_MIN=0.5 N=4`
   (acceptance > N=4-baseline's 16%).
3. p_min sweep at N=4 completes; comparison table vs BF16 N=4 baseline.
4. SPRINT-010-FINDINGS.md exists.

### Soft gates

- Sprint 005 ≥1.3× median hard gate clears at some `(p_min, N)` combo.
- Worst-case stays ≥0.9× across all 5 prompts.
- Default flipped if a clearly winning combo emerges.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU contention blocks bench | Already happened | Medium | Non-GPU work continues; reschedule bench when free |
| p_min cuts too aggressively, hurts code prompts | Possible | Low | Sweep includes p_min=0.9 (almost no truncation) as ceiling; can also test mixed defaults |
| Top-1 probability is uncalibrated → p_min threshold doesn't transfer | Possible | Medium | Sweep is empirical, not theoretical — picks whatever value works on the bench prompts |
| F-022's per-request counter interaction with p_min | Possible | Low | Bench measures both N gen + N acc per request; F-022 was verified Sprint 008 |

---

## Open questions

1. Should p_min apply when i==1 too? Current patch skips i==1 so we
   always commit ≥1 draft. Spiritbuun's full impl has `i > 1`. Leaving
   as i>1 unless data suggests otherwise.

2. Should we also test combined `(p_min, N=8)` and `(p_min, N=16)` to
   see if p_min recovers the wins we lost going from N=16 to N=2 in
   Sprint 008? **Yes** — if p_min works, the optimal N may move back
   up because aggressive blocks plus p_min truncation give us
   "diffusion's parallelism + only-keep-confident" simultaneously.
   Add `N=8 p_min=0.5` to the sweep if Phase 3 budget allows.

3. Should the spiritbuun `683c5ac` upstream cherry-pick (the
   autoregressive p_min fix from PR #22506) also be applied? It only
   matters if anyone uses our autoregressive speculative path; not
   prioritized here.
