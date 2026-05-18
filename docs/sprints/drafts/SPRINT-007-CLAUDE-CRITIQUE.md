# Sprint 007 Draft Critique: Codex vs Gemini

> **Date**: 2026-05-18
> **Reviewer**: Claude (automated critique)
> **Inputs**: SPRINT-007-CODEX-DRAFT.md, SPRINT-007-GEMINI-DRAFT.md, SPRINT-007-INTENT.md
> **Target sprint**: True multislot MTP draft inference for upstream `draft-mtp` on Qwen3.6-27B / RTX 4090

---

## Executive Summary

Both drafts converge on the right scope — patch upstream `common/speculative.cpp` and `tools/server/server-context.cpp`, gate the experimental path behind an explicit flag, keep B1 default, and gate promotion on per-slot greedy correctness plus B4 throughput against both B1 (1.4×) and A4 (0.7×).

The **Codex draft** is the stronger plan. It treats the sprint as profile-first ("do not start optimizing from an uncommitted anecdote"), enforces architectural invariants (slot isolation, constant total KV budget, np=1 byte-identical), explicit phase gates, and a documented "publish a failure report and keep B1" failure mode at multiple checkpoints. Its main weaknesses are the absence of time estimates, no LOC-based descope decision point, and an under-specified set of upstream functions to patch.

The **Gemini draft** is significantly thinner. It is faster to read and includes a useful bottleneck-identification framing (compute serialization vs KV contention vs CPU scheduling) and concrete time boxes per phase. But it omits architectural invariants, phase gates, a baseline-preservation step that produces a committed artifact, multiple intent-required correctness cases (np=2 acceptance, rejection-heavy prompts, long prompts, B1 byte-equivalence), and A4 / B1 regression gates. Its DoD is missing at least four items the intent calls out as gates.

Neither draft adequately addresses several edge cases the multi-slot machinery will actually hit: slot churn mid-generation, KV eviction interacting with batch-row metadata, uneven active-slot counts during a draft step, and `b9196` tag mutability.

---

## 1. Codex Draft — Strengths

1. **Profile-first discipline, explicit baseline preservation**. Phase 0's first task is "preserve the reported `A1/A2/A4` and `B1/B2/B4` matrix in committed artifacts before touching the patch." This directly answers the intent's concern that the seeding matrix may not be reproducible from `main` because the homelab agent commits aren't pushed. Both the markdown report and the JSON artifact are required.

2. **Architectural invariants are written down** (§Architecture > Architectural invariants). Slot isolation, constant total KV budget for first-pass experiments, np=1 byte-for-byte identical, and "no fork-internal speculative mode revival" are all explicit constraints — not just narrative. The constant-KV-budget rule in particular is a non-obvious anti-pattern guard ("they do not silently buy throughput by inflating VRAM or shrinking the verification problem").

3. **Phase gates with exit criteria**. Every phase has a "Phase gate" sentence with a measurable threshold. Phase 0 gate: artifact committed, harness reports per-slot acceptance. Phase 1: validation can fail loudly on token divergence. Phase 2: patch applies + builds + drafts at np=2. Phase 3: compose/Helm launch experimental, default remains B1. Phase 4: documented promotion decision.

4. **Multiple failure-path escape hatches**. The DoD says "or the sprint publishes a failure report and keeps `B1` as the default" on both the acceptance gate and the throughput gate. This is the right shape for a research-flavored sprint where the patch might land but the gates might not. The Risk table reinforces this with "ship artifacts and keep `B1` default if `B4` misses the gate."

5. **Instrumentation framed as disambiguator, not decoration**. Phase 0 task 4 says the timing hooks must "distinguish 'draft path is serialized' from 'draft path is batched but rollback is dominating.'" This pinpoints exactly the hypothesis-confirmation role that lets the sprint conclude meaningfully even if it descopes.

6. **Security section names the right concern**: cross-request state isolation under batch-row reordering — "drafted tokens between concurrent sessions when batch-row order changes." This is the specific failure mode for multislot draft batching and Gemini does not call it out.

7. **Open questions are decision-shaped**, not exploratory: scope (MTP-only vs DFlash-generalizable), throughput target (np=2 vs np=4), KV budget (196608 vs 131072 compose default), flag naming (`MTP_MULTISLOT_EXPERIMENT` vs generic `PREVIEW=1`), and follow-up scoping. Each has a concrete decision lurking behind it.

8. **Both machine-readable (JSON) and human-readable (Markdown) artifacts** at Phase 0 and Phase 4. The JSON shape lets later phases diff cleanly; the Markdown lets reviewers read the story.

---

## 2. Codex Draft — Weaknesses

1. **No time estimates per phase**. Sections are labeled "Phase 0/1/2/3/4" with goals but no day/week budget. The intent doesn't require explicit time-boxing, but a multi-week patch-and-bench sprint will benefit from at least relative effort percentages so reviewers can spot ratio mistakes (e.g., is Phase 0 1 day or 5?). The Claude draft uses percent-of-effort; Gemini uses absolute day counts. Codex uses neither.

2. **No LOC-based or change-size descope decision point**. The intent flags "scope: high" ("the sprint may end at instrumentation if upstream `draft-mtp` needs a deeper refactor") yet the Codex plan has no Phase-1 spike with a measurable bound that triggers an early descope. The risk table mentions the possibility but doesn't operationalize a check. If `state_draft_mtp` turns out to need per-sequence promotion across rollback + KV bookkeeping, the sprint should know that by end of Phase 1, not by end of Phase 2.

3. **Patch areas under-specified**. §Phase 2 lists "any nearby helper or state definitions required to carry batch-row metadata" as a patch area — this is the load-bearing change in the entire sprint and it gets one bullet. Naming `common_speculative_state_draft_mtp`, `common_speculative_draft`, `common_speculative_verify`, and the per-slot loop in `process_single_task` (or its equivalent) would let a reviewer cross-check the patch against the plan. The intent already does this in `## Relevant Codebase Areas`; the draft drops the specificity.

4. **No explicit A4 regression gate in DoD**. The intent has a regression gate: "MTP-off A4 must remain within 5% of 124.5 t/s aggregate on the same homelab 4090." The Codex DoD has a B1 regression gate (item 7) but not an A4 gate. Phase 4 captures A4 in the matrix, but the DoD doesn't promote it to a hard requirement, even though the intent explicitly says the multislot path "must not regress the MTP-off path."

5. **Patch hygiene is asserted but not verified by a hard gate**. §Implementation says the patch should be regenerated and not edited in `/tmp`; the DoD says the patch applies cleanly to a fresh `b9196` checkout. There's no DoD item about a clean reverse-apply (helpful for catching merge artifacts) and no requirement that `docker build` reproduces from a `git clean -fdx` state. Without that, "applies cleanly" can be a false positive when local state pollutes the patch.

6. **`b9196` tag mutability is not handled**. Upstream tags can be force-moved; the `Dockerfile` pins `LLAMA_CPP_REF=b9196` (a tag), not a commit SHA. If `b9196` moves during the sprint, the patch may apply to a different tree than the one it was developed against. Neither the Risks section nor the Open Questions raise this; the Claude draft pins the upstream commit SHA in `Dockerfile` as a mitigation.

7. **Open Question 3 (full 196608 vs compose-default 131072 KV budget) is unresolved at plan time**. This is load-bearing for Phase 0's VRAM math (B1 is at 21.4 / 23 GB with ~1.2 GB free per the intent). The sprint should pick one before Phase 0 starts; otherwise the baseline matrix and the experimental matrix may use different context sizes and the comparison won't isolate the multi-slot variable.

8. **No tests/ artifact in Files Summary**. The drafts adds scripts (`mtp_probe.py`, `bench_n_parallel.py`) and benchmark docs but no pytest wrapper around the correctness harness. Sprint 005's `validate_dflash.py` and any prior validation tests show the pattern; this sprint should land equivalent test infrastructure so CI can keep multi-slot correctness from regressing post-merge.

9. **Phase 1's deterministic validation prompt set is "at least three prompt classes"** (normal prose, reject-heavy, one long prompt). Three is the minimum from the intent. The plan should specify what counts as "long" (token count, prefill cost) and what triggers "reject-heavy" reliably — the intent mentions code-style output, but reject behavior is sampler/seed dependent. Without an LLAMA_SPEC_FORCE_REJECT_AT-style mechanism (Sprint 005 introduced one), reject-heavy verification is a coin flip.

10. **Counter / trace surface is not standardized**. §Phase 0 says "add timing and counter hooks around draft, verify, and accept/rollback phases." §Phase 4 says capture "aggregate tok/s, per-slot tok/s, acceptance, rollback counts, batch sizes reaching `ctx_dft`, and any VRAM headroom changes." It's not clear where these surface (server JSON, log scrape, separate file). The Claude draft proposes `timings.spec.*`; the Codex draft leaves this informal.

---

## 3. Gemini Draft — Strengths

1. **Concrete time boxes per phase** (Phase 0: 1–2 days, Phase 1: 2–3 days, Phase 2: 1 week, Phase 3: 3 days). Total ≈ 2–2.5 weeks, which matches what a sprint of this shape should be. Codex omits this; reviewers have to back-derive effort.

2. **Bottleneck-identification framing is sharp** (§Phase 1 Task 2): "Analyze logs to determine if the 77.2 t/s cap at `np=4` is due to draft-compute serialization, KV-cache contention, or CPU-side scheduling delays in the server." This is the right hypothesis space; Codex stops at "is the draft path serialized vs is rollback dominating," which is one slice of the same question.

3. **Concrete absolute throughput target**: "`np=4` MTP-on aggregate throughput ≥ 95 t/s on RTX 4090." This is easier to test against than `B4 ≥ 1.4 × B1` because the multiplier depends on the captured B1 number (which can drift). Codex uses the multiplier and includes a B1 regression gate to keep B1 stable, but an absolute floor is an additional sanity check Gemini provides.

4. **Lighter, more readable**. The 4-page-ish length means the plan is easier to keep loaded in working memory during execution. Codex is more thorough; Gemini is more glanceable.

5. **Tagged with the substrate path early** (§Architecture > Substrate layout): the upstream-patch-in-Docker-image flow is shown in 5 lines, which is the only thing a new reviewer actually needs to know about the build system on Day 1.

6. **`--experimental-multislot-mtp` is a more conventional name** than `MTP_MULTISLOT_EXPERIMENT` or `MTP_MULTISLOT=1`. Three conventions in play — Codex's underscored env, Gemini's CLI-style flag, and the Claude draft's `MTP_MULTISLOT=1` env — and the sprint should pick one consistent with `--spec-type draft-mtp` already in `entrypoint.sh`. Gemini's CLI-flag instinct is closest to upstream's existing flag style.

---

## 4. Gemini Draft — Weaknesses

1. **Architecture section is shallow**. The "Substrate layout" is a directory tree, not a data-flow diagram. The "Technical Approach" lists three concerns (Batch Draft Optimization, MTP Hidden State Carryover, Server-Side Scheduling) as bullet points without saying what changes. A reviewer cannot decide whether the patch is well-bounded from this draft alone.

2. **No phase gates / exit criteria**. Each phase has "Tasks" but no measurable condition that has to hold before advancing. Codex's phase gates double as fall-back checkpoints if the sprint descopes; Gemini's tasks could all be partially complete and the plan wouldn't know.

3. **No baseline-as-committed-artifact requirement**. §Phase 0 Task 2 says "Document findings in `docs/benchmarks/results/SPRINT-007-BASELINE.md`." The intent and the seeding matrix both make clear the baseline must survive into a *machine-readable* form so later phases can diff against it. Markdown alone makes B1-regression and A4-regression gates harder to evaluate automatically.

4. **B1 byte-equal preservation at np=1 is not a gate**. §Implementation Phase 2 says "Optimize the loop... for parallel execution where possible. Resolve any per-slot state leaks." It does *not* require that np=1 produces output identical to B1 with the same seed. The intent treats single-slot equivalence as the easiest sanity check for the patch ("`np=1` must remain byte-for-byte identical to the current `B1` logic apart from added counters and traces" — Codex draft, paraphrased from intent). Gemini's DoD has "MTP-on multi-slot matches MTP-off target-only output" but not "patched MTP-on np=1 matches pre-patch MTP-on np=1."

5. **No A4 regression gate**, same gap as Codex but compounded because Gemini also doesn't list A4 explicitly in the baseline matrix as a hard reproduction target.

6. **Acceptance only gated at np=4**, not np=2 (DoD item 5). The intent says "per-slot MTP acceptance at `np=2/4` stays within ±5 percentage points of B1." If acceptance degrades only at np=2 (e.g., a 2-slot interaction bug that gets masked by averaging at np=4), Gemini's DoD won't catch it.

7. **Patch hygiene gate is absent from DoD**. Item 2 says "`llama-server` built with updated patch successfully handles `N_PARALLEL=4` with `draft-mtp`" — but doesn't say the patch must apply cleanly to a fresh `b9196` checkout. Phase 2 mentions "Apply fixes to a fresh `llama.cpp` checkout" but it's a task, not a gate.

8. **Compose default profile is not asserted**. The DoD has "B1 remains the default production profile in `k8s/values.yaml`" — Helm only. No mention of `docker-compose.yml`'s default profile, Makefile targets, or README guidance. The Codex draft is more complete on the deployment surface.

9. **Files Summary is incomplete**:
   - References the draft itself as a file (`docs/sprints/drafts/SPRINT-007-GEMINI-DRAFT.md`) but no merged sprint doc (`SPRINT-007-mtp.md`).
   - No `bench_n_parallel.py` modification listed.
   - No `k8s/values.yaml`, `k8s/templates/deployment.yaml`, `k8s/README.md`, `docker-compose.yml`, or `Makefile`. Codex's table lists all six.
   - No baseline JSON / final JSON artifacts.
   - No correctness harness file (`mtp_correctness.py` or equivalent).
   - No pytest wrapper.

10. **Risk table is only 4 rows**, with no entry for:
    - Cross-slot state leakage / hidden-state interference (Codex covers this).
    - Upstream patch drift due to `b9196` tag mutability.
    - Preview controls drifting into default deployments (Codex covers this).
    - Homelab 4090 contention / scheduling.
    - Reproducibility of the seeding matrix on a fresh checkout.

11. **Security section is two sentences** and doesn't mention cross-request state isolation, the specific failure mode that multi-slot draft batching introduces. The Codex draft devotes a paragraph to this — it's the security delta that this sprint creates.

12. **Open Question 3 (VRAM-triggered single-slot fallback)** is a runtime feature, not a planning question. If the sprint wants this safety net, it should be in scope or out of scope — leaving it as an open question means it might or might not ship without anyone deciding.

13. **No descope path**. If §Phase 2's patch is harder than the 1-week budget supports, the plan has no contingency. Codex addresses this implicitly via "publish the failure analysis and keep production docs pointed at `B1`"; Gemini's DoD has no equivalent failure-shipped outcome (the only outcomes are pass or unstated).

14. **"Sub-linear Compute Scaling" risk says "the sprint may only deliver instrumentation"** but no phase or DoD item describes what the instrumentation-only deliverable looks like. The Codex draft says explicitly: "If the fork-side change produces correctness but not enough scaling, the sprint still ships the instrumentation, benchmark artifacts, and safe preview path." Gemini gestures at the same outcome without committing to it.

15. **Three prompts is fewer than the intent's "at least three distinct prompts"** — equal-not-greater. The intent's framing leaves room for "prompt classes" (Codex picks up: normal prose, reject-heavy, long); Gemini just says "3 distinct prompts" with no class structure.

---

## 5. Gaps in Risk Analysis (both drafts)

### Both drafts miss

| Gap | Why It Matters |
|-----|----------------|
| **Slot churn mid-generation** | When a request finishes and a new request joins, the active-slot set changes step-to-step. The batched-draft path must handle the transition without leaking last-step's `pending_h` / `verify_h` into the new slot. Neither draft mentions this; it is a likely source of subtle multislot bugs that pass byte-equal tests on fresh-launch but fail in long-running serving. |
| **KV eviction × batch-row metadata** | Multi-slot batched drafting attaches `seq_id` / `pos` per batch row. If the target context evicts a slot's KV mid-generation (likely with `ctx=196608` and high request churn), the row's metadata may still reference an evicted position. Both drafts say "constant total KV budget" or "VRAM headroom" — they don't say what happens when eviction races with a draft step. |
| **Empty / zero-active-slot draft step** | If all slots are idle at a draft tick (gap between requests), the multi-slot batch is empty. The current per-slot loop just doesn't run; the batched version must not call `llama_decode` with an empty batch. Edge cases like this rarely show up in fresh-test bench harnesses. |
| **Sampler RNG seed handling across slots** | If per-sequence samplers share an RNG state (intentionally or via a global), changing batch-row order changes per-slot output even at the same seed. Greedy mode masks this; the moment a tester uses `--temp > 0` it surfaces. Codex's "no sampler crossing slots" invariant is the right shape; neither draft adds a sampler-isolation test. |
| **`b9196` tag mutability** | `LLAMA_CPP_REF=b9196` is a tag pin, not a SHA pin. Upstream tags can be force-moved. Neither draft pins the SHA in Phase 0; the Claude draft handles it in §Risks. If the tag moves mid-sprint, the patch may apply against a different tree silently. |
| **Reproducibility of the seeding matrix** | The reported `np=1/2/4 = 39.7/71.5/124.5 t/s` (MTP-off) and `68.1/73.1/77.2 t/s` (MTP-on) came from homelab agent commits that may not be pushed. If Phase 0 can't reproduce A4, every downstream comparison is on shaky ground. Codex requires the matrix to be preserved as an artifact but only Phase 4 actually re-runs it; Gemini reproduces in Phase 0 but doesn't gate on reproduction. |
| **Homelab 4090 contention with training** | The homelab GPU is shared with training jobs. Intermittent VRAM pressure / thermal throttling can produce noisy bench numbers that look like multislot regressions. Neither draft requires bench runs to be scheduled in dedicated windows or to abort when `/health` shows alien slots. |
| **Throughput gates without per-user latency floor** | Both drafts gate on aggregate throughput. If multi-slot MTP improves aggregate but per-user decode at np=4 drops below B1 single-slot, operators serving low-concurrency interactive workloads regress. The intent's Open Question 3 raises this; neither draft answers it. |
| **MTP draft hidden-state interference under bursty workloads** | If two slots produce drafts that draw from a shared `pending_h` pool by mistake, the bug may surface only when both slots happen to predict the same token at the same step (rare in tests, common in production). Codex's invariant ("no hidden-state carryover between slots") is correct; neither draft proposes a stress-test for it. |

### Codex-specific gaps

| Gap | Why It Matters |
|-----|----------------|
| **No phase-1 LOC bound trigger for descope** | The intent flags scope risk as "high." Codex names the risk but does not operationalize it. Without an "if change > N LOC, descope here" rule, the sprint can drift past the spike into a half-finished refactor. |
| **A4 regression gate absent from DoD** | Intent requires A4 within 5% of 124.5 t/s after the patch. Codex has the B1 gate but not the A4 gate. |
| **No reverse-apply patch check** | Helpful for catching merge artifacts; not in the plan. |

### Gemini-specific gaps

| Gap | Why It Matters |
|-----|----------------|
| **No cross-slot state isolation risk** | The single highest-impact correctness failure mode for this sprint and Gemini's risk table doesn't enumerate it. |
| **No B1 regression gate** | Intent requires B1 within 5%. Gemini's DoD doesn't include it. |
| **No instrumentation-only descope outcome** | Risk mentions it ("the sprint may only deliver instrumentation"); no DoD or phase commits to what that deliverable contains. |
| **No upstream-drift mitigation** | b9196 tag-vs-SHA risk not raised. |
| **No homelab-contention mitigation** | Bench noise risk not raised. |

---

## 6. Missing Edge Cases

| Edge Case | Codex | Gemini | Notes |
|-----------|-------|--------|-------|
| Mid-generation slot churn (request joins / leaves) | Not mentioned | Not mentioned | Multi-slot batched-draft critical case |
| Empty active-slot set (idle tick) | Not mentioned | Not mentioned | `llama_decode` with empty batch behavior |
| KV eviction during a draft step | Not mentioned | Not mentioned | Stale row metadata possible |
| Uneven prefill / one slot still prefilling while others decode | Not mentioned | Not mentioned | Common in real serving |
| Speculative reject avalanche on one slot starving the batch | Not mentioned | Not mentioned | If one slot rejects repeatedly while others succeed, batch utilization collapses |
| Long prompt in one slot, short in another | Implied (Codex prompt classes) | Not mentioned | Codex partially covers this; Gemini doesn't |
| Request with `temp > 0` (sampler RNG isolation across slots) | Not mentioned | Not mentioned | Greedy testing alone won't catch shared-RNG bugs |
| Per-request `--spec-draft-n-max` override | Not mentioned | Not mentioned | If per-request overrides are valid, the batched draft must respect per-slot N_max |
| Request timeout / cancellation mid-decode | Not mentioned | Not mentioned | Cleanup path under multi-slot is new code |
| Sequence continuation / KV reuse across requests | Not mentioned | Not mentioned | `cache_prompt: true` flow under multi-slot |
| `--no-warmup` interaction at higher `n_parallel` | Not mentioned | Not mentioned | Warmup behavior may differ when n_parallel > 1 |
| Graceful behavior when `MTP_MULTISLOT=1` is set against an unpatched binary | Not mentioned | Not mentioned | Pre/post patch image mismatch in deploys |
| Pre-existing `--spec-replace-on-rollback` or related flags | Not mentioned | Not mentioned | If any rollback-affecting flag exists, the multislot path must respect it |
| One slot at `np=4` with `draft-mtp`, others with no draft | Not mentioned | Not mentioned | Heterogeneous draft modes per slot may or may not be supported |

---

## 7. Definition of Done Completeness

### Codex DoD (9 items) — Assessment

| Item | Verdict |
|------|---------|
| 1. Baseline matrix preserved in committed artifacts | Good — required before optimization |
| 2. Patch applies cleanly + builds `llama-server` | Good |
| 3. `np=1` multislot-capable code behaviorally identical to B1 | Good — byte-equal sanity check |
| 4. Greedy correctness at `np=1/2/4` on 3 prompts × 256 tokens | Good |
| 5. Per-slot acceptance ±5pp at `np=2/4` or publish failure | Good with failure escape hatch |
| 6. B4 ≥ 1.4× B1 AND ≥ 0.7× A4, or publish failure | Good — both thresholds |
| 7. B1 within 5% of baseline | Good |
| 8. Compose + Helm require explicit preview control | Good |
| 9. README and k8s docs distinguish preview vs production | Good |
| **Missing**: A4 within 5% of 124.5 t/s | Intent requires this regression gate explicitly |
| **Missing**: VRAM headroom measured during bench | Intent calls out 4090 tightness |
| **Missing**: Instrumentation artifact ships even on descope | Risk + Phase 0 promise this; DoD doesn't gate it |
| **Missing**: pytest wrapper / test infra | Sprint 005 pattern suggests this is expected |
| **Missing**: Spike doc bounds Phase 2 change size | Risk acknowledges scope; DoD doesn't gate it |

### Gemini DoD (7 items) — Assessment

| Item | Verdict |
|------|---------|
| 1. Performance matrix reproduced + documented | Good — but markdown only, no JSON |
| 2. `llama-server` patch handles `N_PARALLEL=4` with `draft-mtp` | Vague — "handles" doesn't mean correct |
| 3. MTP-on multi-slot matches MTP-off target-only on 3 prompts (greedy) | Partial — missing "patched MTP-on np=1 matches pre-patch MTP-on np=1" |
| 4. `np=4` aggregate ≥ 95 t/s | Good — concrete absolute, but doesn't include A4-relative |
| 5. Per-slot acceptance ±5% at `np=4` | Partial — missing `np=2` |
| 6. `entrypoint.sh` allows multi-slot only with experimental flag | Good |
| 7. B1 remains default in `k8s/values.yaml` | Partial — doesn't include compose default, Makefile, README |
| **Missing**: Patch applies cleanly to fresh `b9196` | Intent expects patch hygiene as a gate |
| **Missing**: B1 single-slot regression ≤ 5% | Intent requires this |
| **Missing**: A4 regression ≤ 5% | Intent requires this |
| **Missing**: B4 ≥ 1.4× B1 (Gemini uses 95 t/s absolute instead — they're not equivalent if B1 drifts) | Intent specifies the multiplier |
| **Missing**: Failure-path "or publish a failure report and keep B1" | Intent explicitly permits this; no DoD analog |
| **Missing**: Per-slot acceptance at `np=2` | Intent requires `np=2/4` both |
| **Missing**: Long-prompt and rejection-heavy prompt cases | Intent lists three prompt classes; Gemini just says 3 prompts |
| **Missing**: VRAM headroom measurement | Intent calls out 4090 tightness |
| **Missing**: Helm `nParallel: 1` default + Helm `mtpMultislot: false` default | Compose / Helm parity |
| **Missing**: Docs distinguish preview vs production | Intent calls out promotion-policy documentation |

---

## 8. Recommendation

**Use the Codex draft as the base**, with these modifications drawn from the Gemini draft and this critique:

1. **Add phase time-boxes**. Adopt Gemini's per-phase day estimates or convert to percent-of-effort. Reviewers need this to spot ratio mistakes.

2. **Insert an explicit Phase 1 spike with an LOC bound** before the implementation work. If the change exceeds the bound (e.g., > 200 LOC ripple into rollback/KV bookkeeping), Phase 2 descopes to "instrumentation + report only" and the sprint closes successfully with that outcome documented. This is the highest-leverage addition because the intent flags scope risk as high.

3. **Add A4 regression gate to DoD**: "MTP-off A4 aggregate within 5% of 124.5 t/s after the patch is applied." Intent requires it; neither draft includes it.

4. **Resolve Open Question 3 (KV budget) before Phase 0 begins**. The first experiments need to pick one of `ctx=196608` (matches deployed B1) or `ctx=131072` (matches compose/Helm defaults). Mixing them makes the baseline matrix incomparable to the experimental matrix.

5. **Name the upstream functions to patch** (`common_speculative_state_draft_mtp`, `common_speculative_init`, `common_speculative_draft`, `common_speculative_verify`, and the per-slot loop in `tools/server/server-context.cpp`). The intent does this; the drafts shouldn't drop the specificity.

6. **Pin `LLAMA_CPP_REF` to a commit SHA in Phase 0** (or capture the SHA alongside the tag in `Dockerfile`). The b9196 tag is potentially mutable; the patch is built against a specific tree, so the pin should match.

7. **Standardize instrumentation surface**: pick `timings.spec.*` (or equivalent) so external monitoring can consume it without log scraping. `mtp_probe.py` and `bench_n_parallel.py` extensions should consume the same fields.

8. **Add a pytest wrapper around the correctness harness** (`tests/test_mtp_multislot.py` or equivalent). Sprint 005 set the pattern; sustaining multi-slot correctness post-merge needs CI coverage.

9. **Add edge-case tests to Phase 1 / Phase 6 correctness harness**: slot churn (request joins/leaves mid-generation), empty active-slot tick, uneven prefill, long prompt + short prompt in same batch. Pure 3-prompt greedy parity doesn't catch these.

10. **Add a cross-slot state-isolation explicit invariant test**: run B4 with four distinct prompts and four distinct seeds, then run B1 sequentially with the same prompts/seeds, confirm token-for-token match per slot.

11. **Gate bench runs on dedicated GPU windows / `/health` clean state**, to avoid training-job contention noise. Required only on the homelab 4090 runs.

12. **Document the descope deliverable concretely**: if the sprint ends at instrumentation, what ships? Codex implies it (regenerated patch + traces + matrix + failure report); make it a named outcome in §Phase 4 with explicit DoD items so the sprint doesn't end ambiguously.

13. **Pick one flag convention and use it everywhere** (`MTP_MULTISLOT` env, `--experimental-multislot-mtp` flag, or `mtpMultislot:` Helm value). Both drafts hint at conventions; the merged sprint should commit.

14. **Adopt Gemini's absolute throughput floor as an additional sanity check**: `B4 aggregate ≥ 95 t/s` in addition to the relative `≥ 1.4× B1` and `≥ 0.7× A4` gates. Catches drift in B1 captures.
