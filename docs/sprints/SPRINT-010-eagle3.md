# Sprint 010-eagle3: EAGLE3 productionization

> **Track suffix**: `-eagle3` — sister architecture to the `-dflash` track.
> Same DFlash track convention applies: does not merge to `main`, branch
> chain stays on its own.

**Status**: Planning (2026-05-05)
**Sprint type**: Implementation + measurement
**Created**: 2026-05-05
**Depends on**: Sprint 009-dflash close (DFlash track at median 1.24× Q8;
draft-acceptance-bound, not pipeline-bound; user picked EAGLE3 over
distillation)
**Estimated effort**: ~1 week single-engineer (highly variable on Phase 0
draft availability)

**Branches**:
- Repo: `sprint/010-eagle3` (off `sprint/008-dflash`)
- Fork: `feature/sprint-004-rebase-dflash` continues from `526097eed`
  (no fork work expected — engine is already cherry-picked)

---

## Overview

Sprint 008/009 demonstrated the DFlash speculative pipeline is solved
(VRAM-shadow ckpt, F-024) and tuned (DRAFT_N_MAX=2 default). The
remaining gap to the ≥1.3× median hard gate is **draft acceptance on
entropic prose** — Hamlet/DC at 0.87-1.03× vs target-only.

The architectural reason DFlash struggles: a separate 1.7B draft model
that runs a full forward pass per round. EAGLE3 (cherry-picked into
fork during Sprint 004) takes a different approach — the draft is a
thin head running on top of the **target's hidden states from the
previous decode step**. No separate forward pass; draft cost is
~linear-head-latency instead of ~1.7B-forward-pass-latency. EAGLE3
typically delivers more linear speedup than autoregressive draft
designs because the draft is essentially free.

Sprint 010 wires up EAGLE3 end-to-end on the same qwen profile, runs
the same 5-prompt × 3-trial bench, and decides:

1. Does EAGLE3 clear the ≥1.3× median hard gate that DFlash missed?
2. If yes: ship as default. If no but better worst-case: ship as
   alternative. If worse than DFlash: stick with DFlash.
3. Either way, document the regime where each architecture wins.

## Use Cases

1. **Operators get a speculative path that delivers near-linear
   speedup**: EAGLE3's near-zero draft-gen overhead means acceptance
   directly translates to throughput. If acceptance is 60%+ on prose
   (which we expect from EAGLE3-class drafts), median should clear ≥1.3×.

2. **Architectural diversity in the speculative track**: today the
   DFlash track has a single architecture. After Sprint 010, operators
   can pick DFlash (block diffusion, peaks high on code) vs EAGLE3
   (autoregressive single-token, more consistent across prompts).
   Documented guidance in README.

3. **Sprint 011 has clean signal on what comes next**: if EAGLE3
   clears the gate, we move to non-speculative work (multi-slot
   batching, EAGLE3 + multi-target hot-swap). If it doesn't, we know
   draft training is the irreducible bottleneck and Sprint 011
   becomes the distillation sprint.

## Architecture

```text
fork pin (526097eed baseline) — engine + dispatch already in tree
  ↓
Phase 0: Source EAGLE3 draft weights for Qwen3.6-27B target
        (z-lab / lmsys / community survey; pin one)
  ↓
Phase 1: Convert weights → GGUF; register model key
  ↓
Phase 2: Entrypoint dispatch (--eagle3 flag) + compose profile
  ↓
Phase 3: Validation harness extension (greedy parity smoke test)
  ↓
Phase 4: Full E2-style bench at N={2,4} EAGLE3 vs DFlash baseline
  ↓
Phase 5: Pick default + findings + Sprint 011 rec
```

Constraints:
- Reuses Sprint 004 cherry-pick of EAGLE3 graph (`src/models/eagle3.cpp`,
  +186 LOC). Verified building + loadable since Sprint 004; just no
  weights to load.
- `--eagle3` runtime flag already wired through `common/arg.cpp`.
- `convert_hf_to_gguf.py` has an `Eagle3Model` converter class; should
  Just Work for any z-lab-style EAGLE3 safetensors.
- VRAM-shadow ckpt (Sprint 008) is DFlash-specific (recurrent state
  snapshot). EAGLE3 has different state requirements — Phase 4 will
  surface whether VRAM ckpt applies or if EAGLE3 needs its own path.

## Implementation

### Phase 0 — Source EAGLE3 draft for Qwen3.6-27B

**Goal**: A pinned safetensors path for at least one Qwen3.6-EAGLE3 draft.

**Tasks**:
- [ ] Survey HuggingFace for Qwen3.6-EAGLE3 candidates:
  - `z-lab` org (they ship our DFlash drafts; likely also have EAGLE3)
  - `lmsys` (canonical EAGLE3 publisher upstream)
  - `spiritbuun` (community, ships our DFlash GGUFs)
  - `Tengyunw` / SafeAILab (EAGLE upstream)
- [ ] If a Qwen3.6 EAGLE3 draft exists: pin repo + commit SHA in this
      sprint doc, note license/access status (gated/open).
- [ ] If no Qwen3.6-targeted EAGLE3 exists:
  - Descope to **Qwen3.5-27B-EAGLE3** (drafts more likely to exist
    for older targets) and document the gap as a Sprint 011 followup.
  - Or: train a Qwen3.6-EAGLE3 draft from scratch (out of scope for
    this sprint; close Sprint 010 with "Phase 0 blocker filed" if
    no drafts exist).
- [ ] If gated: ask for access; document in sprint doc.

**Phase gate**: A reachable safetensors path for at least one Qwen3-EAGLE3
draft, license/access status documented.

### Phase 1 — Convert + register

**Files**:
- `scripts/convert_eagle3_drafts.sh` — NEW. Mirror of
  `convert_dflash_drafts.sh`. Idempotent: download draft safetensors,
  download target tokenizer files, run `convert_hf_to_gguf.py`,
  publish into `llm-models` volume.
- `Makefile` — `convert-eagle3-drafts` target.
- `docker/entrypoint.sh` MODELS table — add `qwen3.6-27b-eagle3` (or
  the descoped variant from Phase 0).

**Tasks**:
- [ ] Clone `convert_dflash_drafts.sh` shape, swap repo + filename.
- [ ] Verify `convert_hf_to_gguf.py` EAGLE3 path is functional in our
      fork. Sprint 004 cherry-picked the converter's Eagle3Model class
      but never exercised it on real weights.
- [ ] Quantize to Q8_0 (Sprint 009 finding: Q8 preserves acceptance,
      Q4 may collapse if EAGLE3 has SWA-equivalent fragile layers).
      Q8_0 is also the recommended quant in upstream EAGLE3 docs.
- [ ] Register model key.
- [ ] `llama-cli --model <draft.gguf> -ngl 0 -n 0` smoke test that
      tensor names load correctly.

**Phase gate**: At least one EAGLE3 draft GGUF in `/models` and loads
without errors via `llama-speculative-simple --eagle3`.

### Phase 2 — Entrypoint dispatch + compose profile

**Files**:
- `docker/entrypoint.sh`:
  - Extend `SPECULATIVE_MODE` validator from
    `target-only|autoregressive|dflash` →
    `target-only|autoregressive|dflash|eagle3`.
  - Command builder adds `--eagle3` when `SPECULATIVE_MODE=eagle3`.
- `docker-compose.yml`:
  - New profile `qwen-eagle3` (alongside existing `qwen` / `qwen-target-only`).
  - Single-slot, planar3 KV (mirror of DFlash profile defaults).
  - Same env propagation (DRAFT_N_MAX, LLAMA_SPEC_VRAM_CKPT, etc.).
- `Makefile` — `run-qwen-eagle3[-bg]`.

**Tasks**:
- [ ] Extend SPECULATIVE_MODE case statement.
- [ ] Add `--eagle3` to the CMD builder.
- [ ] New compose service with the EAGLE3 model key.
- [ ] Smoke `make run-qwen-eagle3`. `/health` + one greedy completion.

**Phase gate**: `make run-qwen-eagle3` boots and serves a coherent greedy
completion on the quicksort prompt.

### Phase 3 — Validation harness extension

**Files**:
- `scripts/bench_speculative.py`:
  - Add `eagle3` leg alongside `target-only`, `autoregressive`,
    `dflash`. Update finalize() rendering to handle 4 legs.
- `scripts/run_sprint008_experiment.sh` (or rename to
  `run_speculative_experiment.sh` + add `--leg eagle3` flag).
- `scripts/validate_dflash.py` — consider rename to
  `validate_speculative.py` with `--mode {dflash,eagle3}`. Or
  duplicate as `validate_eagle3.py` to keep the diff small.

**Tasks**:
- [ ] Add eagle3 leg to bench harness.
- [ ] Run greedy validate on the 5-prompt set:
  - L2 reference: target-only output, exact match (greedy, temp=0)
  - Pass criterion: ≥3 of 5 prompts produce 256/256 token match
- [ ] Run `validate_eagle3.py --reference zlab` if a pytorch
      reference exists.

**Phase gate**: L2 (greedy match 256/256 on 3+ prompts) passes for
EAGLE3.

### Phase 4 — Bench EAGLE3 vs DFlash

**Tasks**:
- [ ] Run 5-prompt × 3-trial bench at N=2 and N=4 with EAGLE3 leg.
      Save under `docs/sprints/SPRINT-010-eagle3-experiments/`.
- [ ] Decide whether VRAM-shadow ckpt path applies. If yes, bench
      with `LLAMA_SPEC_VRAM_CKPT=1`. If EAGLE3 doesn't use the
      recurrent state path that vram_seq_checkpoint snapshots, the
      env-toggle is a no-op (fine).
- [ ] 4-way comparison table (target-only / autoregressive /
      DFlash N=2 Q8 / EAGLE3 N=?) at the right N for each.

**Phase gate**: full bench completes without crashes; comparison
table written; per-prompt EAGLE3× ≥0.95× target-only on ≥4 of 5
prompts (correctness gate, not speedup gate).

### Phase 5 — Pick default + findings + Sprint 011 rec

**Decision matrix**:
- **EAGLE3 median ≥1.3× and ≥0.95× on all 5 prompts**: flip
  docker-compose default to EAGLE3. DFlash track stays as opt-in for
  code-heavy peaks where it might still win.
- **EAGLE3 median 1.0–1.3× and stable**: ship as parallel option,
  document regime where each wins, no default flip.
- **EAGLE3 median <1.0× or unstable**: ship as PREVIEW, file gaps,
  recommend distillation as Sprint 011.

**Tasks**:
- [ ] Update `docker-compose.yml` default if EAGLE3 wins.
- [ ] Update README "Speculative Decoding" section: EAGLE3 row in
      the profile table; one-line guidance on when to pick which.
- [ ] Update `docs/BENCHMARK-REPORT.md` with Sprint 010 § (4-way
      comparison).
- [ ] `docs/sprints/SPRINT-010-eagle3-FINDINGS.md` with hypothesis
      verdicts.
- [ ] Sprint 011 recommendation:
  - If EAGLE3 won: multi-slot speculative batching (the upstream
    `TAG_SERVER_SPEC_REWORK` TODO that lets multiple slots share
    one draft `llama_context`) — broader operator value.
  - If EAGLE3 didn't: distillation sprint. Train a smaller DFlash or
    EAGLE3 draft tuned for prose.

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `scripts/convert_eagle3_drafts.sh` | Create (repo) | Convert + publish EAGLE3 GGUF |
| `docker/entrypoint.sh` | Modify (repo) | SPECULATIVE_MODE=eagle3 path |
| `docker-compose.yml` | Modify (repo) | qwen-eagle3 profile |
| `Makefile` | Modify (repo) | run-qwen-eagle3 + convert-eagle3-drafts |
| `scripts/bench_speculative.py` | Modify (repo) | eagle3 leg |
| `scripts/validate_eagle3.py` (or rename) | Create/rename (repo) | Greedy parity |
| `docs/sprints/SPRINT-010-eagle3-experiments/` | Create | Bench artifacts |
| `docs/sprints/SPRINT-010-eagle3-FINDINGS.md` | Create | Outcomes + Sprint 011 rec |
| `docs/BENCHMARK-REPORT.md` | Modify (repo) | 4-way comparison |
| `README.md` | Modify (repo) | Speculative profile guidance |

---

## Definition of Done

### Hard gates

1. EAGLE3 draft GGUF in `/models` for at least one Qwen3 target.
2. EAGLE3 leg passes L2 greedy parity (256/256 token match on ≥3 of
   5 prompts).
3. 4-way bench comparison table written.
4. SPRINT-010-eagle3-FINDINGS.md with hypothesis verdicts and
   Sprint 011 recommendation.

### Soft gates

- EAGLE3 median ≥1.3× (clears the standing Sprint 005 hard gate
  that DFlash hasn't cleared).
- EAGLE3 worst-case ≥0.95× target-only.
- docker-compose default flipped to EAGLE3 if it wins outright.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| No Qwen3.6-EAGLE3 draft exists in the wild | Medium-High | High (sprint blocker) | Phase 0 explicitly checks; descope to Qwen3.5-27B-EAGLE3; or close sprint with Phase 0 blocker |
| `convert_hf_to_gguf.py` EAGLE3 path has tokenizer chkhsh issue | Low | Low | Sprint 004 already added Qwen3.5/3.6 chkhsh mapping (`1c9b77fdd`) |
| EAGLE3 acceptance degrades with thinking-on like DFlash | Medium | Medium | Bench both regimes; report no-think as opt-in default if needed |
| EAGLE3 doesn't use VRAM-shadow ckpt path → no save tax win available | Possible | Low | EAGLE3 may already have low save cost (no recurrent state); if not, file as Sprint 011 followup |
| Greedy parity fails (acceptance is high but outputs diverge) | Low | High | Stop ship; investigate EAGLE3 numerical setup vs target |

---

## Open questions

1. **Where is the Qwen3.6-EAGLE3 draft published?** Phase 0 surveys
   z-lab, lmsys, spiritbuun, SafeAILab. If none exist, this sprint
   becomes a "training run scoped" sprint, which is bigger than 1
   week.

2. **Does EAGLE3 share the recurrent-state ckpt requirement?**
   DFlash has a hybrid-recurrent target context with state that needs
   snapshot/restore between speculative rounds. EAGLE3 typically
   doesn't (autoregressive, no recurrent state). The VRAM-shadow path
   may be a no-op or may not apply at all. Phase 4 confirms.

3. **How does EAGLE3's optimal N compare to DFlash's N=2?** EAGLE3 is
   single-token autoregressive (block_size = 1 effectively); the N
   parameter is interpreted as max-draft-tree-depth or tree-width
   depending on impl. May want to test N={1,2,4,8}.

4. **Should we also pull spiritbuun's KV cache quant changes?** They
   may improve EAGLE3 too. Decide after Phase 4 — if EAGLE3 wins
   without KV changes, no need.
