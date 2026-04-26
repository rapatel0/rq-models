# Sprint 004 Critique: Claude reviews Codex draft

Reviewing `SPRINT-004-CODEX-DRAFT.md` against `SPRINT-004-INTENT.md`. Goal: identify
what to keep, what to fix, what is missing. Section/line refs are to the Codex draft
unless otherwise noted.

---

## 1. Strengths to preserve in merge

These are decisions the Codex draft got right. The merged sprint should keep them.

- **Strict L1 gate before any DFlash work** (Overview L5; Phase 1 L116; Risks
  table L270). This directly answers Intent Q1 and is the right call: a broken
  rebase compounds with DFlash bugs into an undebuggable mess.
- **Three-profile structure** (`qwen36-spec`, `qwen36-27b-dflash`, `qwen36-dflash`)
  with the standard-draft profile sequenced *before* DFlash (L7; Phase 2 L122).
  This isolates `seq_rm` rollback validation from block-diffusion complexity —
  exactly the right factoring of concerns.
- **MoE is experimental, dense is the hard gate** (L7; KTD #5 L76; DoD L260).
  The intent's soft-gate framing said the same; making it explicit at the profile
  level is cleaner than burying it in success criteria.
- **`DRAFT_KV_CACHE_TYPE` defaults to `${KV_CACHE_TYPE}`** (KTD #4 L75; OQ#3 L304).
  Sensible default; correctly keeps override hatch open for benchmarking.
- **Single-slot enforcement at `tools/server/server-context.cpp`** (KTD #3 L74;
  Phase 2 L131; Phase 4 L200). Good: fails fast at the source rather than
  silently corrupting shared slot state.
- **Explicit cherry-pick commit list with SHAs** (Upstream Integration Plan L50–66).
  Addresses Intent's "PR head moves while sprint is in flight" risk by pinning
  to specific commits. (See §3.1 below for one issue with this section.)
- **Verify-batch append helper as a dedicated path in `src/llama-kv-cache.cpp`**
  (KTD #2 L73; Phase 2 L137; DFlash data flow L94). This is the right answer to
  Intent Q2: it picks option (b)-with-batching (one quantized append pass per
  verify batch), avoiding both the wasteful re-prefill (a) and the 16× kernel
  call thrash of the naïve (b).
- **"Do not resurrect snapshot-and-replay"** (Phase 3 L180). Shows the author
  read upstream commit history carefully, not just the PR description.
- **Files Summary table** (L221–251). Useful merge artifact; intent didn't have
  this and it should survive.

---

## 2. Weaknesses (specific, with line refs)

### 2.1 Internal contradiction in commit list (L63–68)

L66 includes `e344c4a71736e1cdaa25e590a109f694dfb8119f` in the DFlash cherry-pick
list. L68 then says the plan "avoids cherry-picking PR head merge commits
(`91b03e4c…` and `e344c4a71736…`'s ancestors via merge state)". On a literal
read, that lists `e344c4a` as something to avoid. The parenthetical was probably
meant to mean "ancestors-via-merge-state of `e344c4a`", but the sentence breaks
on first read. **Recommendation**: rewrite L68 to name the avoided commits
unambiguously, and confirm whether `e344c4a` is wanted (Phase 3 L180 implies
yes — it is the removal of snapshot-and-replay) or avoided.

### 2.2 DoD lines 263–264 are vacuous

> - [ ] Tests pass.
> - [ ] No regressions.

These mean nothing without scope. The Intent's hard gates are concrete (±0.05
PPL, ≥2.0× speedup, identical token sequences). Two boilerplate checkboxes at
the end of DoD undermine the rest. **Recommendation**: replace with:
- "All `tests/test_speculative.py` and `tests/test_dflash_e2e.py` cases pass."
- "Throughput on the 8 existing profiles is within 5% of pre-rebase baseline
  measured under matched conditions."

### 2.3 MoE soft gate is looser than Intent

DoD L260 allows MoE "no worse than a 10% decode slowdown". Intent §Soft gates
#1 (L153) requires "≥1.0× decode (no regression)". A 10% slowdown is a regression.
**Recommendation**: tighten to match Intent or, if the Codex author believes
−10% is a realistic floor for MoE (PR data shows 0.61× on gpt-oss-20B), surface
that reasoning explicitly and have the user adjudicate. Don't silently relax.

### 2.4 Speedup gate prompt is vaguer than Intent

DoD L259 says "primary dense benchmark prompt set". Intent hard gate #3 (L142)
specifies *the* prompt: "Write a quicksort algorithm" with thinking-off.
**Recommendation**: name the prompt and the thinking-mode setting in the DoD.
"Primary set" lets a future executor cherry-pick favorable prompts.

### 2.5 Acceptance rate and first-64-token thresholds are missing

DoD L261 says `validate_dflash.py` "records first-64-token agreement plus
acceptance-rate deltas" — but no thresholds. Intent §L3 (L167) demands
"First-64-token match; acceptance rate ±5pp" and Intent soft-gate #2 demands
"Within 10pp of the PR's reported numbers". Without thresholds, recording is
not validating. **Recommendation**: encode the thresholds in DoD.

### 2.6 No time budget or fail-fast criterion

The phase percentages (L100, L122, L142, L183, L206) sum to 100% but reference
no calendar. Intent §Uncertainty (L210–213) flags rebase scope as HIGH with a
range of "1 day clean or 1 week messy". The mitigation in Risks L277 says
"if it slips, cut MoE DFlash" but never defines "slips". **Recommendation**:
add an explicit timebox (e.g., "if Phase 1 has not closed L1 gate by day 5,
cut Phases 3 MoE-target and reduce scope to dense-only delivery").

### 2.7 No "Items From Prior Sprints" section

Intent §Items From Prior Sprints (L274–295) discusses D-013 (Benchmark CI),
D-011 (Open WebUI), D-005/D-007 (stale Sprint 003 items). The Codex draft
has no carry-forward section at all. **Recommendation**: add a section that
either schedules these items or explicitly defers them with rationale.

### 2.8 Phase 5 "Documentation" is thin and lacks reproducibility check

Phase 5 (L206–219, ~10% of effort) has 4 tasks. None of them say "an outside
reader can reproduce the 2.0× number from the docs alone". Documentation
without a reproducibility check is documentation that drifts.
**Recommendation**: add a task: "Verify a fresh-clone reader can run
`make bench-dflash` and reproduce the headline numbers without consulting
sprint authors."

### 2.9 Phase 5 omits the cache-preservation gate

Intent hard gate #4 (L146): "No re-download (model cache preserved)" when
existing 8 profiles still launch. Codex DoD L262 only says "still start
successfully". The cache preservation requirement is not represented anywhere
in Codex tasks or DoD. **Recommendation**: add an explicit task in Phase 5 to
verify existing model cache is reused under the new Dockerfile pin.

### 2.10 No mention of GPU contention with training workloads

Intent §Hardware constraints (L125) specifies: "GPU often busy with training —
sprint must not block on continuous GPU access." The Codex draft never
addresses this. **Recommendation**: add to Phase 4 a note that benchmarks
must be runnable in batched off-peak windows and report variance, not single
runs.

### 2.11 "Reusable validation harness" is in Intent soft-gate but not DoD

Intent soft-gate #3 (L156): "The differential test runner can be pointed at
any (target, draft) GGUF pair without code changes." Codex's
`validate_dflash.py` (Phase 4 L201–202) does not state this requirement and
DoD does not check it. **Recommendation**: add a task: "harness accepts
target/draft paths via CLI, no source edit required for new model pairs."

### 2.12 Open Questions section duplicates Intent without adding analysis

OQ L300–309 in Codex restates Intent's questions and answers them in 1–2
sentences. Some answers are good (Q1, Q5), but Q3 ("draft model KV cache
type") doesn't address Intent's actual unknown: "mixing types in one
llama-server instance may not be supported." Defaulting to `${KV_CACHE_TYPE}`
sidesteps the question rather than answering it. **Recommendation**: confirm
or refute upstream support for mixed KV types in one server, and if it's not
confirmed, add it as a Phase 1 spike.

---

## 3. Risk-analysis gaps

### 3.1 Mitigation that may not address its risk: row 2 of risk table

L271 row: "DFlash verification batches accidentally trigger prefill-style
deferred conversion. Mitigation: Add a verify-batch append path **and test
it first via `qwen36-spec`**."

But `qwen36-spec` is the *autoregressive draft* profile (Phase 2 L122, L136).
Standard autoregressive draft typically appends 1 token per verify pass, not
a 16-token DFlash block. So the verify-batch append path may not be
exercised by `qwen36-spec` at all — the bug Codex is mitigating against is
specifically a DFlash batched-append concern. **This is a critical mitigation
flaw.** **Recommendation**: either (a) add a synthetic test in `qwen36-spec`
that posts a multi-token verify batch through the new helper, or (b) move
verify-batch helper validation to Phase 3 with a DFlash-specific unit test
in the fork.

### 3.2 Missing risk: GPU contention with training

Already noted in §2.10. No row in the risk table for this. Likelihood: High.
Impact: Med (variance in benchmark numbers, possibly invalidating ±5% gates).

### 3.3 Missing risk: draft + target VRAM exceeds 24 GB tier

Intent §Architecture uncertainty (L221) flags this: "Draft model VRAM cost
on top of target … may force quant-tier downgrades on the 24 GB
recommendation." Codex risk table does not include this. **Recommendation**:
add row "Draft model VRAM cost on 24GB tier forces target quant downgrade.
Mitigation: profile draft VRAM in Phase 4; document VRAM headroom in
QUANTIZATION-GUIDE.md or remove the 24 GB DFlash recommendation entirely."

### 3.4 Missing risk: long-context (32K+) interaction at iso3 262K

Intent edge cases L187–188 raises this. Codex draft does not mention long-
context anywhere. The deferred-K conversion path interacts with KV size, and
DFlash at 262K iso3 is unverified. **Recommendation**: add as a risk *and*
as an edge case test in Phase 2 or Phase 3 (even if just a 32K smoke test).

### 3.5 Missing risk: z-lab pytorch reference is single-source

Intent §Uncertainty (L209) flags it. Codex risk table row "Differential
validation is noisy" addresses sampler/seed noise, but not the underlying
"what if z-lab is itself wrong" concern. **Recommendation**: add risk +
mitigation, or explicitly accept z-lab as the canonical reference and
document that assumption.

### 3.6 Missing risk: upstream PR is rebased mid-sprint

Codex risk row "Upstream DFlash/EAGLE3 PRs keep changing" mitigates with
"pin the fork commit … do not track PR head dynamically". That mitigation
addresses *our* fork drift. It does **not** address the case where the PR
gets merged upstream during our sprint with a *different* conflict
resolution than ours, leaving our fork on a path the maintainer rejected.
**Recommendation**: add a note about post-sprint rebasing onto the merged
form; if upstream merges during the sprint, evaluate whether to switch to
the merged version.

### 3.7 Missing risk: rebase fails (not slips)

Codex L277 mitigates rebase *slipping* by cutting MoE. But what if the rebase
itself fails — for example, upstream made a structural change that breaks
the deferred-K hook surface? There is no fallback plan. **Recommendation**:
state the abort condition explicitly: "If `src/llama-context.cpp` deferred-K
hooks cannot be reproduced on the new master, abandon the sprint and file
a Sprint 005 redesigning the deferred-K integration."

---

## 4. Missing edge cases

### 4.1 DFlash + RotorQuant deferred-K interaction

The verify-batch append helper (KTD #2 L73) is a sound design but the draft
defines no test cases for the hard boundary conditions:

- **First DFlash verify batch immediately after prefill**: does
  `convert_deferred_keys()` complete before the first verify batch? Order
  matters; if a verify batch arrives while K is still mid-conversion, the
  append goes to the wrong representation.
- **Verify batch when `defer_k=false` (mid-decode)**: does the same helper
  do the right thing, or does it assume deferred-K is always active?
- **Verify batch when prefill produced 0 deferred K** (e.g., KV reuse from
  a prior session): does the helper handle the empty-deferred-buffer state?

**Recommendation**: enumerate these as Phase 3 unit tests in the fork.

### 4.2 `seq_rm` rollback edge cases

Phase 2 task L138 covers "at least one forced-rejection case" but nothing
about:

- **Rollback partway through deferred-K conversion** (race window between
  verify-append and `convert_deferred_keys`).
- **Rollback when accepted token count = 0** (entire draft rejected).
- **Rollback at exact double-buffer boundary** (intent mentions
  "double-buffer" at line 73 — what if `seq_rm` straddles a buffer flip?).

**Recommendation**: add these as named test cases in
`scripts/validate_speculative.py` or Phase 2 unit tests.

### 4.3 MoE-specific edges

Codex correctly demotes MoE to experimental but doesn't propose any MoE-
specific test:

- **Expert activation overhead during parallel verify** (Intent edge cases
  L185): no test, no instrumentation.
- **Expert routing determinism between target and draft**: if they activate
  different experts on the same token, acceptance rate craters in ways
  that look like bugs but aren't.
- **MoE + iso3 KV vs MoE + planar3 KV under DFlash**: `qwen36-dflash` ships
  iso3 by repo convention, but planar3 is also in scope per the rebase L1
  matrix; does DFlash see different acceptance rates between the two?

**Recommendation**: at minimum, add an instrumentation task: "log expert
activation count per verify pass for MoE; surface in benchmark output."

### 4.4 Cross-architecture compatibility

The draft GGUFs come from third parties (`spiritbuun`, `lym00`). Are their
tokenizers byte-for-byte compatible with the production Qwen3.6 target
tokenizers? If draft and target tokenize the same input string differently,
the draft's tokens are not even comparable to the target's, and acceptance
math is meaningless. **Recommendation**: add a Phase 4 pre-flight check:
tokenize a fixed corpus with both and assert byte-equal token sequences.

### 4.5 Profile coexistence

What happens if a user starts both `qwen36-27b` and `qwen36-27b-dflash` at
the same time? On a single 32 GB GPU, that's almost certainly OOM. No
documentation, no compose-level mutual exclusion. **Recommendation**: at
minimum, README note about mutual exclusion; ideally compose-level
`profiles:` configuration that prevents conflict at startup.

### 4.6 Non-greedy is silently deferred

Phase 5 L219 mentions non-greedy as "follow-up work, not reasons to block",
but the draft never explicitly states "this sprint validates only `--temp 0`
greedy decoding; sampling-mode behavior is unverified." This belongs in
README.md as a user-facing caveat, not just a deferred follow-up.

---

## 5. Definition of Done completeness

**Verdict: meeting this DoD as written would NOT prove the sprint succeeded.**
There are gaps that allow a "DoD met" check to coexist with a broken sprint.

Specific holes (cross-referenced to §2 above):

| DoD item | Issue |
|---|---|
| L255 (rebase done) | No definition of "done" — does it require upstream-PR-able? Or just compiling? |
| L256 (PPL ±0.05) | ✓ Concrete, matches intent |
| L257 (qwen36-spec) | ✓ But test inventory missing (§4.2) |
| L258 (qwen36-27b-dflash exact-match) | ✓ But prompt set undefined (§2.4) |
| L259 (2.0× speedup) | Prompt unspecified (§2.4); thinking-mode unspecified |
| L260 (MoE) | Looser than Intent (§2.3) |
| L261 (validate_dflash.py) | No thresholds (§2.5) |
| L262 (8 profiles still start) | Cache preservation missing (§2.9) |
| L263 (Tests pass) | Vacuous (§2.2) |
| L264 (No regressions) | Vacuous (§2.2) |

**Missing entirely from DoD** (should be added):

- Acceptance rate within ±10pp of upstream PR numbers (Intent soft-gate #2).
- Validation harness reusable without code changes (Intent soft-gate #3).
- BENCHMARK-REPORT.md §10 added with reproducible commands.
- README.md and QUANTIZATION-GUIDE.md updates committed.
- Long-context (32K+) smoke pass on at least one profile.
- Tokenizer-equivalence check between draft and target GGUFs.
- Final fork commit hash published to `docker/Dockerfile`,
  `BENCHMARK-REPORT.md`, and README (Phase 5 L216 has it as task; promote
  to DoD).
- Explicit "this sprint validates greedy only; sampling deferred" caveat
  in user-facing docs.

---

## 6. Summary recommendations for merge

In rough priority order, the merged sprint should:

1. **Fix the verify-batch helper test plan** (§3.1). The current mitigation
   doesn't exercise the path it claims to validate. Either add a synthetic
   batched-append test in Phase 2, or move helper validation to Phase 3.
2. **Replace vacuous DoD lines** (§2.2) with concrete checks.
3. **Add acceptance-rate / first-64-token thresholds** to DoD (§2.5).
4. **Tighten MoE gate or surface the relaxation** (§2.3).
5. **Name the speedup-gate prompt and thinking mode** (§2.4).
6. **Add timebox / fail-fast criterion to Phase 1** (§2.6).
7. **Add risks**: GPU contention (§3.2), 24 GB VRAM headroom (§3.3),
   long-context (§3.4), z-lab single-source (§3.5), rebase failure (§3.7).
8. **Resolve commit-list contradiction at L68** (§2.1).
9. **Add edge-case tests**: deferred-K boundary conditions (§4.1),
   `seq_rm` rollback variants (§4.2), tokenizer-equivalence preflight (§4.4).
10. **Add Items From Prior Sprints section** (§2.7) to handle D-013, D-011,
    etc.
11. **Add cache-preservation check** (§2.9) and **reproducibility task**
    (§2.8) to Phase 5.
12. **Add OQ#3 spike** for mixed-KV-type llama-server support if not
    already empirically known (§2.12).

The Codex draft is structurally sound — phases sequenced correctly, profile
factoring clean, key technical decisions defensible. The weaknesses are
mostly about precision (vague DoD, undefined thresholds) and coverage (edge
cases, missing risks). It is a good base; it should be merged with the
above corrections rather than rewritten.
