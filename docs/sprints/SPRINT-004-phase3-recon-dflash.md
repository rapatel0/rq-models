# Sprint 004 Phase 3 Reconnaissance

**Date**: 2026-04-27
**Purpose**: Pre-execution survey of the DFlash cherry-pick scope.

## State of master at our rebase target

Rebase target: `78433f606fde4d7934a02dcbfd910438d28beccd`.

Master already contains:

- **PR #19493** ("server: speculative checkpointing") — merged at
  `455d8e4be` on 2026-04-19. This brings the
  `llama_state_seq_get_size_ext / get_data_ext / set_data_ext` APIs and
  the `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` flag we already exercise via
  `vram_seq_checkpoint`.
- **Speculative dispatch enum scaffolding** in `common/speculative.cpp`:
  `COMMON_SPECULATIVE_TYPE_EAGLE3` enum value, registry entry, factory
  branch, and a `// TODO PR-18039` comment. The enum slot exists; the
  EAGLE3 model graph and runtime logic do not.
- **`common_speculative_state_eagle3` struct shell** — declared but with
  no DFlash sibling.

**Master does NOT yet contain:**
- Any DFlash code (`grep -rE "DFLASH|dflash" src/` returns nothing in src/)
- The EAGLE3 model graph (`src/models/eagle3.cpp` does not exist)
- The DFlash-specific GGUF metadata constants
- The DFlash target metadata changes in `src/models/qwen3*.cpp`

So our cherry-pick will land both EAGLE3 and DFlash code largely from
scratch — master only has the dispatch-layer hooks.

## PR commit list (#22105)

23 commits total, in chronological order. EAGLE3 base (PR #18039 lineage)
is the first 17 commits; the last 4 are DFlash-specific. Two intermediate
commits are merges with master.

```
8fac4b1cc8  2025-12-14  feat: add EAGLE3 speculative decoding support
ac5667dcc6  2025-12-16  fix eagle3 logits sync bug & remove ggml_set_sync()
3e7f376b53  2025-12-17  Merge branch 'master' into pr/18039
5a79c1900f  2025-12-17  eagle3: improve naming
c0d99e65d2  2026-01-08  add eagle3 support for Qwen3 series models
71ba283a65  2026-01-09  add eagle3 support for Qwen3 MoE models
3da288d78d  2026-01-10  eagle3: load lm_head from target if not in draft
13a9f31de3  2026-01-10  eagle3: make d2t mapping optional
75883cde73  2026-01-10  eagle3: gpt-oss-120B support
7b78bfa984  2026-01-16  eagle3: RedHatAI eagle3 speculator support
7d4c223943  2026-02-05  Merge branch 'master' into HEAD
5e224bc190  2026-02-09  Merge branch 'master' into pr/18039
b3537924ef  2026-02-20  eagle3: fix model convert issue
9fea2434af  2026-02-20  eagle3: fix model convert code format
b8ab2cc559  2026-02-23  Merge branch 'master' into pr/18039
07e2c9707c  2026-02-28  eagle3: support --eagle3 in llama-cli
5bb2d50c0f  2026-03-16  Merge branch 'master' into pr/18039
91b03e4c93  2026-04-24  Merge branch 'master' into pr/18039

(DFlash commits, last 4)
0724d66e5c  2026-04-18  dflash: first working POC
85a0089e60  2026-04-19  dflash: add support for qwen3.5/3.6 moe models
e344c4a717  2026-04-24  dflash: remove redundant logic & correct bias naming
67cb0d5070  2026-04-27  dflash: enable llama-cli & llama-server with np=1   ← landed today
```

The `67cb0d5070` commit landed **today** — the PR is actively iterating.
Pin a specific SHA at cherry-pick time.

## Cumulative diff (PR head vs PR base)

```
common/arg.cpp                              +16  -2   modified
common/common.h                             +5   -0   modified
common/speculative.cpp                      +331 -23  modified  ← biggest non-new
convert_hf_to_gguf.py                       +187 -1   modified
examples/speculative-simple/...             +77  -20  modified
gguf-py/gguf/constants.py                   +53  -0   modified
include/llama.h                             +53  -0   modified
src/llama-arch.{cpp,h}                      +24/+15      modified
src/llama-context.cpp                       +377 -6   modified  ← biggest conflict zone
src/llama-context.h                         +31  -0   modified
src/llama-cparams.h                         +2   -0   modified
src/llama-graph.{cpp,h}                     +2/+42       modified
src/llama-hparams.h                         +15  -0   modified
src/llama-model-loader.cpp                  +1   -0   modified
src/llama-model.{cpp,h}                     +169/+14     modified
src/models/dflash.cpp                       +161 -0   ADDED
src/models/eagle3.cpp                       +186 -0   ADDED
src/models/llama.cpp                        +10  -0   modified
src/models/models.h                         +20  -0   modified
src/models/openai-moe-iswa.cpp              +24  -0   modified
src/models/qwen3.cpp                        +25  -0   modified
src/models/qwen35.cpp                       +14  -0   modified
src/models/qwen35moe.cpp                    +14  -0   modified
src/models/qwen3moe.cpp                     +11  -0   modified
tools/server/server-context.cpp             +21  -0   modified
```

Total: **+1,890 / -52 lines**, 28 files.

## Conflict zone with our fork

Our fork modifies several files in this list:

| File | Our delta | PR delta | Conflict risk |
|------|----------:|---------:|--------------:|
| `src/llama-context.cpp` | medium | +377/-6 | **High** (already a high-conflict file from rebase) |
| `src/llama-graph.cpp` | TurboQuant V padding | +2 | Medium |
| `src/models/qwen35.cpp` | none | +14 | Low |
| `src/models/qwen35moe.cpp` | none | +14 | Low |
| `src/models/qwen3moe.cpp` | none | +11 | Low |

**No fork modifications** in: `common/speculative.cpp`, `convert_hf_to_gguf.py`,
`gguf-py/gguf/constants.py`, `include/llama.h`, `src/llama-arch.{cpp,h}`,
`src/llama-cparams.h`, `src/llama-graph.h`, `src/llama-hparams.h`,
`src/llama-model.{cpp,h}`, `src/llama-model-loader.cpp`, `src/models/llama.cpp`,
`src/models/models.h`, `src/models/openai-moe-iswa.cpp`, `src/models/qwen3.cpp`,
`tools/server/server-context.cpp`. Cherry-pick of those should apply cleanly.

## Recommended cherry-pick strategy

Same approach as the rebase: **squash-merge of all 23 PR commits onto our
sprint branch**. Do NOT replay each commit individually — many were
WIP/intermediate, and 23 cherry-picks of overlapping changes is the same
nightmare we avoided with `git merge --squash` in Phase 1.

```bash
cd /home/ravi/repos/llama-cpp-turboquant
git checkout feature/sprint-004-rebase-dflash
git fetch upstream pull/22105/head:pr-22105    # pull the PR branch
git merge --squash pr-22105                    # squash all 23 onto our branch
# Resolve conflicts (mainly src/llama-context.cpp, src/llama-graph.cpp)
git add -A && git commit -m "Cherry-pick DFlash + EAGLE3 from PR #22105 / #18039"
```

Estimate: 1-2 days to resolve conflicts and verify build.

## Validation gates after cherry-pick

1. **Compile clean** — `cmake --build build -t llama-server llama-cli llama-checkpoint-bench llama-perplexity llama-bench`
2. **L1 PPL regression unchanged** — re-run `scripts/ppl_sweep.py --mode pair`. Cherry-pick must not break the 0-regression status from Phase 1.
3. **vram_seq_checkpoint still bit-exact** — re-run `llama-checkpoint-bench` correctness check on both production targets.
4. **`llama-speculative-simple --dflash` smokes** with one of the DFlash GGUFs (`spiritbuun/Qwen3.6-27B-DFlash-GGUF` or `lym00/Qwen3.6-35B-A3B-DFlash-GGUF-Test`). Generates non-empty output without crash.

Phases 4-6 (Docker profiles, validation harness, benchmarks, docs) follow
the cherry-pick. Each is its own session.
