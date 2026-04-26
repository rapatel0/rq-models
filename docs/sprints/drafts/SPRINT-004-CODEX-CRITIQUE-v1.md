# Sprint 004 Draft Critique: Claude Draft

> **Date**: 2026-04-26
> **Reviewer**: Codex
> **Inputs**: `SPRINT-004-INTENT.md`, `SPRINT-004-CLAUDE-DRAFT.md`, current repo layout

---

## Executive Summary

The draft is strongest where it is most disciplined: sequencing the rebase before speculative work, naming the real conflict zones, and insisting on multi-layer validation. Those are worth preserving.

The main problems are scope drift and internal inconsistency. The shared brief explicitly asks for `PR #18039 (EAGLE3) + #22105 (DFlash)` and an intermediate autoregressive speculative profile to validate `seq_rm` rollback first ([intent lines 10-14, 227-272](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-INTENT.md:10)). The draft instead defers EAGLE3 entirely, removes the explicit autoregressive profile, and then contradicts itself on whether speculative server integration is in scope ([draft lines 20-29, 68-80, 281-357, 601-614](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:20)). As written, the team could satisfy large parts of the DoD and still not have delivered the sprint described in the brief.

---

## 1. Strengths To Preserve In Merge

1. **Strict rebase gate before speculative work**: The draft’s insistence that the fork must rebase cleanly and pass the full L1 PPL gate before any DFlash work begins is correct and matches the brief’s most important sequencing concern ([draft lines 20-25, 182-219](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:20), [intent lines 8-9, 227-230](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-INTENT.md:8)).

2. **Good identification of real integration hotspots**: Calling out `src/llama-context.cpp` as the high-risk merge zone and separating additive files from shared edit zones is useful, concrete planning ([draft lines 82-117, 187-193, 255-265](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:82)).

3. **Three-layer validation framing is strong**: L1 regression, L2 exact greedy equivalence, and L3 reference comparison is the right structure. Even if the exact pass criteria change, the layered design should stay ([draft lines 318-357, 428-458](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:318)).

4. **MoE uncertainty is treated honestly**: The draft does not pretend MoE speedup is likely, and it keeps dense 27B as the hard performance target. That is a sound prioritization choice ([draft lines 44-48, 450-458, 595-599](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:44)).

5. **File-level task breakdown is actionable**: The draft is explicit about which files in the fork versus this repo are expected to move. That will help during merge planning even if some scope items change ([draft lines 382-420](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:382)).

---

## 2. Weaknesses With Specific References

1. **Material scope drift from the brief: EAGLE3 is removed instead of staged.**  
   The brief says Phase 3 is to cherry-pick `PR #18039 (EAGLE3) + #22105 (DFlash)` ([intent lines 12-14](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-INTENT.md:12)). The draft instead says “DFlash only,” pushes EAGLE3 to Sprint 005, and only keeps “foundation” pieces ([draft lines 22-29, 68-80, 601-606](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:22)). That is not just refinement; it is a re-scope.  
   **Recommended change**: Either put EAGLE3 back in scope as the brief states, or mark this explicitly as a proposed scope change requiring approval before merge.

2. **The autoregressive speculative validation phase from the brief is effectively deleted.**  
   The brief explicitly asks for Phase 2: “Add autoregressive speculative decoding profile to validate KV + `seq_rm` rollback interaction” ([intent lines 10-12](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-INTENT.md:10)). The draft keeps one fork-level sanity check with `llama-cli` ([draft lines 244-249](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:244)), but it does not preserve the explicit repo-level profile or isolation phase. That weakens debuggability because DFlash and rollback correctness are no longer cleanly separable.  
   **Recommended change**: Restore a first-class autoregressive speculative profile and make it a hard gate before DFlash import.

3. **The draft contradicts itself on whether server integration is in scope.**  
   Phase 4 and Phase 5 clearly plan `llama-server` integration, new compose profiles, `/v1/chat/completions` e2e tests, and server-side draft plumbing ([draft lines 283-316, 349-352](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:283)). But Q7 later says “`spec4-server` mode ... deferred to Sprint 005” and “Server integration is scaffolded ... but not validated” ([draft lines 612-614](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:612)). Those two positions cannot both be true.  
   **Recommended change**: Choose one path. Either server integration is in-scope and validated, or the sprint stays fork-level via `llama-speculative-simple` only.

4. **The `seq_rm` design is too prescriptive and may be solving the wrong problem.**  
   The draft assumes `seq_rm(seq_id, p0, p1)` can leave a “partial block” behind and proposes `p1 % BLOCK_SIZE` logic plus dequant/re-quant of a trailing partial block ([draft lines 143-155, 238-243](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:143)). That is a strong claim, but the cited planar block size is an intra-row quantization detail, while `seq_rm` is a sequence-position operation. If the storage is token-row based, this helper is unnecessary; if the storage is more complex, this description is still underspecified.  
   **Recommended change**: Rewrite this section as an investigation item with behavioral acceptance tests. Do not lock the sprint to `p1 % BLOCK_SIZE` until the actual KV storage layout is inspected.

5. **The draft understates the work needed for the PPL regression harness.**  
   It repeatedly says “extend `scripts/ppl_sweep.sh`” ([draft lines 215-219, 329-332, 413](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:215)), but the current script is a narrow Qwen3.5 shell sweep over model variants, not a generic 4-KV × 2-corpus × 2-model regression harness ([scripts/ppl_sweep.sh](/home/ravi/repos/turbo/scripts/ppl_sweep.sh:1)). This is a realism issue: converting it into a reusable matrix runner is a bigger task than the draft implies.  
   **Recommended change**: Budget a new dedicated regression runner, or explicitly call Phase 1 work a refactor rather than a small extension.

6. **The compatibility gate ignores part of the live repo surface.**  
   The DoD keeps the “8 existing Docker profiles” from the brief ([draft lines 431-433](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:431)), but the repo currently exposes more live profiles, including `qwen-q3`, `qwen-q3-xxs`, `qwen-iq4`, `gemma-q3`, `qwen-throughput`, `qwen-throughput-q3`, and `qwen36-throughput` ([docker-compose.yml](/home/ravi/repos/turbo/docker-compose.yml:171), [Makefile](/home/ravi/repos/turbo/Makefile:57)). The draft should either justify excluding them or widen the non-regression surface.  
   **Recommended change**: Change the gate from “8 profiles” to “all supported profiles in compose/README,” or explicitly mark which profiles are out of support for this sprint.

---

## 3. Gaps In Risk Analysis

1. **Missing tokenizer/chat-template compatibility risk for third-party DFlash GGUFs.**  
   The draft treats GGUF risk mostly as “schema/layout mismatch” ([draft lines 489-490, 507-514, 627-635](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:489)). It does not call out a more basic failure mode: target and draft may disagree on tokenizer, BOS/EOS IDs, or chat template assumptions. That can destroy acceptance rate or exact-match comparisons before any RotorQuant bug exists.  
   **Recommended change**: Add a preflight gate that verifies tokenizer/vocab/chat-template compatibility before benchmarking or correctness testing.

2. **Missing risk that the exact-match tests may never exercise rollback.**  
   L2 asks for three greedy prompts with exact match ([draft lines 333-338, 434-435](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:333)). That is not enough to prove `seq_rm` correctness if the chosen prompts mostly accept draft tokens. The risk table does not mention this coverage hole.  
   **Recommended change**: Add a forced-rejection case or acceptance-rate instrumentation and require evidence that rollback occurred during at least one correctness test.

3. **Single-slot-only is not actually mitigated at runtime.**  
   The draft says speculative serving is single-slot only ([draft lines 27-29, 305-307, 589-593](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:27)). But in the current repo, `docker-compose.yml` and `entrypoint.sh` simply pass `N_PARALLEL` through to `llama-server` ([docker-compose.yml](/home/ravi/repos/turbo/docker-compose.yml:31), [docker/entrypoint.sh](/home/ravi/repos/turbo/docker/entrypoint.sh:107)). A compose default is not a correctness guarantee.  
   **Recommended change**: Add a code-level startup rejection for speculative modes when `n_parallel > 1`, and add a test that this failure mode is explicit.

4. **The rebase fallback mitigation does not cover the full fork delta.**  
   The risk table says if the rebase goes badly, cherry-pick “our 4 deferred-K commits” onto a fresh fork ([draft lines 486-487](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:486)). But the brief’s context lists more than deferred-K hooks: CUDA template instances, FA dispatch, and other fork edits ([intent lines 31-36, 84-88](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-INTENT.md:31)). That fallback may silently drop behavior.  
   **Recommended change**: Add a pre-sprint commit inventory of all fork deltas and make the fallback “reconstruct the full fork delta,” not just deferred-K.

5. **The mitigation for “DFlash cherry-pick breaks PPL” is pointed at the wrong failure class.**  
   Re-running L1 after every cherry-pick step ([draft line 487](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:487)) helps catch non-speculative regressions, but it will not catch verify-time append/rollback bugs. This mitigation does not actually address the risk it is paired with.  
   **Recommended change**: Pair speculative import with an L2 rollback-focused gate, not just PPL.

6. **MoE risk analysis is too shallow for the profile the draft still wants to ship.**  
   The draft acknowledges slowdown risk ([draft lines 44-48, 494, 595-599](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:44)), but its mitigation is mostly “ship experimental.” That does not address prompt sensitivity, acceptance-rate collapse on thinking/router-heavy prompts, or verify-time expert dispatch overhead across different prompt classes.  
   **Recommended change**: Add a small MoE prompt matrix and a “do not ship the profile if correctness passes but speed regresses materially on all tested prompts” rule.

---

## 4. Missing Edge Cases

1. **First verify batch immediately after deferred-K conversion.**  
   The draft discusses post-prefill verify append in the abstract ([draft lines 119-139](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:119)), but it does not require a test for the exact transition point: prefill completes, `convert_deferred_keys()` runs, and the very next decode call is a multi-token verify batch.

2. **Acceptance-count boundary cases inside one DFlash block.**  
   There is no explicit requirement to test accept-all, reject-all, accept-1, accept-15, and accept-16 cases for a 16-token draft block. Those cases stress different rollback/append paths and should not be left implicit.

3. **Repeated accept/reject cycles, not just one rollback.**  
   The planned tests cover one 256-token decode and a three-point unit test ([draft lines 241-246, 442-445](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:241)). They do not explicitly cover multiple speculative rounds where append and rollback happen repeatedly and state can drift.

4. **Rollback cases beyond tail trim.**  
   The draft only describes a tail-oriented model of rollback. It does not mention `p0 != 0`, zero-length trim, repeated trims on the same sequence, or shared-prefix/multi-sequence behavior. Even if llama.cpp uses the common case most of the time, the sprint should verify the actual subset it depends on.

5. **Cross-boundary append then reject.**  
   A critical path is: accepted tokens push the cache over a quantization/storage boundary, then a later rejection rolls back into the previous boundary. That is exactly where deferred-K and speculative bookkeeping are most likely to disagree, and it is not called out.

6. **Long-context speculative verify, not just short greedy decode.**  
   The brief calls out long context explicitly ([intent lines 186-187](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-INTENT.md:186)), but the draft’s concrete exact-match tests are short 256-token runs. One 32K smoke test appears only as a soft gate ([draft lines 457-458](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:457)). That is not enough coverage for the feature most likely to interact with large KV state.

7. **MoE prompt classes with low acceptance.**  
   The draft benchmarks one dense coding prompt and treats MoE as experimental, but it does not call out router-heavy prompts, thinking-on prompts, or prompts likely to produce low acceptance. For MoE, those are the cases that matter most.

8. **Draft/target compatibility before runtime.**  
   The current plan waits until Phase 4 to discover exact filenames and until Phase 5 to inspect metadata ([draft lines 295-301, 489-490, 627-635](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:295)). It should explicitly test compatibility before the Docker/server work starts.

---

## 5. Definition Of Done Completeness

**Verdict: No.** Meeting the current DoD would not be enough to say the sprint fully succeeded.

1. **It does not prove the brief’s Phase 2 goal.**  
   The shared brief asks for an autoregressive speculative profile to validate KV + `seq_rm` rollback interaction before DFlash ([intent lines 10-12](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-INTENT.md:10)). The DoD has no corresponding hard gate. That means the sprint can “succeed” without the isolation stage the brief explicitly requested.

2. **It does not guarantee delivery of the brief’s EAGLE3 scope.**  
   Because EAGLE3 is deferred in the draft, the DoD can pass while still failing the brief’s “PR #18039 + #22105” requirement ([intent lines 12-14](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-INTENT.md:12), [draft lines 68-80](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:68)).

3. **The only end-to-end GPU path can be skipped.**  
   The DoD says `pytest tests/` passes even though the DFlash e2e test may be skipped in non-GPU CI ([draft lines 444-446](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:444)). That is reasonable for CI, but not as a sprint acceptance criterion on a sprint whose core feature is GPU-only speculative decoding.

4. **Rollback is not required to be observed, only inferred.**  
   Exact token match on three prompts and one C++ unit test do not prove that rollback occurred in the integrated path. If the draft accepts most tokens, the sprint could pass DoD without actually covering the risky path.

5. **The performance bar is too narrow.**  
   One prompt for the hard speedup gate ([draft lines 439-441](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:439)) is vulnerable to prompt-specific optimism. A sprint can pass this and still have unconvincing real-world speedup.

6. **The compatibility gate is incomplete for the current repo.**  
   The DoD’s “8 existing profiles” criterion no longer matches the repo’s full supported profile surface, so it is not a complete non-regression guarantee.

7. **MoE ship criteria are still ambiguous.**  
   The soft gate allows experimental shipping even with slight slowdown ([draft lines 450-452](/home/ravi/repos/turbo/docs/sprints/drafts/SPRINT-004-CLAUDE-DRAFT.md:450)). The draft should say more clearly when to omit the `qwen36-dflash` profile entirely versus ship it as experimental.

**Recommended DoD changes**

1. Add a hard gate for an autoregressive speculative profile that forces at least one rejection and proves exact-match rollback safety before DFlash debugging begins.

2. Resolve the EAGLE3 scope mismatch explicitly in the DoD: either require EAGLE3 delivery per brief, or state that the sprint is re-scoped and needs approval.

3. Make at least one GPU-executed end-to-end DFlash server validation non-skippable for sprint acceptance.

4. Require evidence that rollback occurred during at least one integrated correctness test, not just exact output equality.

5. Widen the dense speedup gate from one prompt to a small fixed prompt set, or at minimum require median/no-regression reporting alongside the headline prompt.

6. Replace the “8 profiles” gate with a gate that matches the repo’s actual supported profiles, or explicitly carve out unsupported ones.

7. Add a preflight DoD item for target/draft tokenizer and GGUF metadata compatibility.

8. Rephrase the `seq_rm` implementation section as behavior-first acceptance criteria until the actual KV storage layout is confirmed.
