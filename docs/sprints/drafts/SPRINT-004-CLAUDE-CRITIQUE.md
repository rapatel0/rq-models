# Critique of SPRINT-004-CODEX-DRAFT.md

Reviewer: Claude (Opus 4.7) | Target: Codex draft (gpt-5.4) | Lens: HYBRID-target reality

The draft has correctly internalized the hybrid finding from INTENT.md §"Critical
Architecture Finding" and rebuilt the sprint around speculative checkpointing. That
is the headline win. But it has a load-bearing blind spot: it tests checkpoint
fidelity for the **K cache layouts** (the 25% full_attention slice) and quietly
treats the **recurrent state of the 75% linear_attention slice** as upstream's
problem. That is precisely the surface that motivated rewriting the sprint in the
first place. Several other gaps follow from the same drift.

## 1. Strengths to preserve in merge

These are non-trivial and the merge should not regress them:

- **§Overview / §Architecture, "speculative decoding starts only after deferred K
  conversion is complete, and rollback uses checkpoint restore plus
  accepted-prefix replay, never `seq_rm`"** — this single sentence is the
  correct architectural rule for hybrid targets and is sharper than INTENT's
  prose. Keep verbatim.
- **§Architecture, "two invariants matter more than anything else"** — the
  invariant pair (real backend buffer, not dequantized view; verify forbidden
  while deferred staging is live) is the right framing and should anchor the
  merged draft's architecture section.
- **§Implementation Phase 2 — fork-level bit-exact C++ test
  (`tests/test-checkpoint-deferred-k.cpp`)** — correctly load-bearing, correctly
  *upstream of* DFlash work, correctly demanding bit-equality rather than
  "numerically close." Preserve this ordering.
- **§Implementation Phase 2 task: "explicit runtime guard that disables
  speculative verify until `convert_deferred_keys()` has completed"** — this is
  a concrete defensive control that doesn't appear in INTENT and should
  survive the merge.
- **§Implementation, phase-percentage allocation (25/30/25/20)** — putting 30%
  of effort into checkpoint fidelity, before any DFlash code, is the right
  shape for a sprint where snapshot/restore is the novel surface.
- **§Open Questions, Q6 resolution: "1.5-2.0x ... 2.0x is stretch, not
  minimum"** — INTENT §Success Criteria still has 2.0× as a hard gate; Codex's
  downward revision is correct given Qwen3.5-9B PR data and should win the merge.
- **§Architecture, "MoE profile is present but requires `EXPERIMENTAL=1`"** —
  good explicit gating model. Better than INTENT's softer "ship as
  experimental with EXPERIMENTAL=1" phrasing.
- **§Implementation Phase 3, "Cherry-pick pinned SHAs ... resolve conflicts
  after the checkpoint work is stable, not before"** — correct dependency
  ordering and reduces rework risk.
- **§Risks row "Checkpoint overhead at long context erases most speculative
  speedup"** with the "fail fast" mitigation — preserves engineering honesty
  over wishful documentation.

## 2. Weaknesses (with section refs)

### 2.1 §Architecture / §Implementation Phase 2 — recurrent state is invisible

The draft's central architectural rule is "checkpoint must capture the actual
RotorQuant memory backend, not a dequantized logical view," but Phase 2's
deliverable (`tests/test-checkpoint-deferred-k.cpp`) only exercises **K-state
buffers** in deferred f16 staging and the four planar/iso layouts. Those are
attributes of the 25% full_attention layers. The 75% linear_attention layers
hold Gated Delta Net / SSM state — `linear_conv_kernel_dim`,
`linear_key_head_dim`, `mamba_ssm_dtype: float32` per INTENT §Critical
Architecture Finding — and that state is *the entire reason* `seq_rm` doesn't
work. The test as scoped will pass even if upstream's checkpoint silently
drops or shallow-copies the recurrent slot. Naming is misleading too:
"checkpoint-deferred-k" implies K-only.

**Fix in merge**: rename to `test-checkpoint-hybrid-state.cpp` (or add a
sibling) and require save→mutate→restore coverage for **(a)** deferred-K
staging, **(b)** all four quantized K layouts, **(c)** SSM/recurrent state on
at least one `linear_attention` layer, **(d)** combined cross-layer state on a
mixed batch that has actually executed both layer types.

### 2.2 §Definition of Done — DoD does not gate on the hybrid surface it claims to defend

DoD line 3 ("`tests/test-checkpoint-deferred-k.cpp` proves bit-exact
checkpoint restore for deferred f16 staging and for quantized `planar3`,
`planar4`, `iso3`, and `iso4` K layouts") is the only line that mentions the
checkpoint test, and it is K-only. There is no DoD line for **recurrent-layer
state fidelity**. A team could meet this DoD on a corrupted hybrid pipeline.
See §5 below for full DoD analysis.

### 2.3 §Open Questions, Q2 resolution — declares snapshot cost a "gate" but never sets a number

Q2 says: "Snapshot cost is an early benchmark gate at 65K and 262K; if
whole-state copies keep 27B below `1.5x`, the sprint should surface that as the
blocker." This is a gate on the *output* (downstream speedup), not on the
*input* (snapshot wallclock). By the time you measure end-to-end speedup, you
have already paid for cherry-picking DFlash. The Phase 2 task "Measure
checkpoint overhead at 65K and 262K" needs a numeric ceiling — e.g., snapshot
+ restore round-trip ≤ N ms or ≤ M% of one verify-step latency — *before* Phase
3 begins. Otherwise "fail fast" fails late.

### 2.4 §Implementation Phase 4 — `tests/test_dflash_e2e.py` "forces at least one rejection path" but does not assert that recurrent state was rolled back

The task wording ("forces at least one rejection path and still proves
target-only greedy equivalence after checkpoint restore") is necessary but not
sufficient. Greedy equivalence at temp=0 can be satisfied by code that
*never restored recurrent state* if the recurrent contribution to the next
token is negligible at that step. Forced rejection on hybrid models needs an
explicit positive assertion that the linear_attention layer's state at
position N+1 (post-restore) equals the state at position N+1 (target-only
trajectory), separate from token-equality. See §4.4.

### 2.5 §Open Questions, Q6 vs §Definition of Done line 6 — speedup gate scope mismatch

Q6 resolves to "Qwen3.6-27B must hit `1.5-2.0x` on the quicksort prompt with
thinking off." DoD line 6 echoes that, but Phase 3's task list says the
benchmark should also cover "theorem and travel prompts." INTENT §Open
Questions Q6 cites Qwen3.5-9B at 2.77× on coding and **1.10×** on travel —
i.e., a 2.5× spread depending on prompt. A sprint that ships 1.5× on
quicksort and 0.8× on travel meets the DoD but disappoints any user whose
workload isn't coding. Either widen DoD ("median of N prompts ≥ 1.5×") or
explicitly accept the narrow gate as the headline number.

### 2.6 §Definition of Done — z-lab pytorch reference comparison silently dropped

INTENT §Verification Strategy lists L3 (numerical reference vs z-lab pytorch)
as a verification layer with "first-64-token match; acceptance rate ±5pp."
Codex's draft creates `scripts/validate_dflash.py` (Phase 3) but neither the
DoD nor any task makes its result a gate. If the script lands but is never
plumbed into a pass/fail, this is theater. Either gate on it or remove it.

### 2.7 §Risks & Mitigations — no row for recurrent-state corruption

The risk table covers checkpoint-not-capturing-K, snapshot overhead, rebase
conflicts, MoE weakness, upstream churn, and VRAM. It does **not** list
"upstream checkpoint mechanism captures K but silently shallow-copies SSM
state on linear_attention layers" — which is the failure mode INTENT
explicitly warned about as "silent corruption if rejection is handled but
recurrent state isn't reset." This is the highest-impact correctness risk of
the sprint and it is missing from §Risks.

### 2.8 §Architecture decode-loop pseudocode — elides hybrid mechanics

The pseudo-code shows save-checkpoint → run draft → verify → accept/reject.
For a hybrid target, the *interesting* steps are: how does the linear_attention
layer evolve during the speculative draft block (does it commit state per
token? per block?), and does "restore checkpoint, replay accepted prefix"
re-feed all M accepted tokens through the full target including its recurrent
layers, or only the attention path? The pseudocode obscures the question that
INTENT made central. Keep the pseudocode but annotate the recurrent path.

### 2.9 §Implementation Phase 4 — `entrypoint.sh` refactor scope is one bullet

"Refactor `docker/entrypoint.sh` to accept a draft model, DFlash enablement
flags, and an `EXPERIMENTAL=1` gate" is plausibly the largest single
behavioral change in this repo and gets one task line. INTENT §Uncertainty
Assessment flags this as scope-expansion risk ("'Add Docker profiles' expands
if `entrypoint.sh` needs material refactor for dual-model launch"). Either
budget more weight to Phase 4 or split the entrypoint refactor into its own
task with explicit "preserve all 8 existing profile launches" as a sub-gate.

### 2.10 §Files Summary table — duplicates Phase listings without resolving status

The Files Summary repeats every file already named in §Implementation. Useful
for skim, but it labels `src/llama-context.cpp` as "Modify" with no indication
that this file is the **High** conflict-risk file from INTENT §Relevant
Codebase Areas. Lossy compression of risk signal.

## 3. Gaps in risk analysis

§Risks & Mitigations is six rows. Gaps:

- **R-G1: Recurrent-layer checkpoint fidelity** (see §2.7) — missing entirely.
  This is the *primary* hybrid risk, not a footnote.
- **R-G2: GPU contention with concurrent training jobs** — INTENT §Constraints
  says "GPU often busy with training — sprint must not block on continuous GPU
  access." Not in §Risks. Phase 2's "Measure checkpoint overhead at 65K and
  262K" needs the GPU exclusively for clean numbers; if a training job is
  resident, snapshot cost gets noisy and the gate is unreliable.
- **R-G3: Pre-built DFlash GGUFs from `lym00` / `spiritbuun` may not match
  cherry-picked PR #22105 SHA semantics** — INTENT §Seed and §Dependencies
  reference these GGUFs, but the metadata in `gguf-py/gguf/constants.py`
  changes inside the cherry-pick. If the public GGUFs were converted from an
  earlier revision of the PR, loading will fail or — worse — succeed with
  subtle metadata mismatches. No mitigation listed.
- **R-G4: Upstream merges #22105 mid-sprint with material differences from the
  pinned SHA** — Codex flags movement risk in row 5 but only as "follow-up
  work, not in-sprint churn." This understates the case if upstream removes a
  symbol the cherry-pick depends on.
- **R-G5: The "refuse to arm speculative while deferred staging is live"
  guard fires too aggressively** — there is no mitigation for the false-positive
  case. If the guard misreads convert state, speculative is silently disabled
  and the sprint's headline metric (1.5× decode) silently fails.
- **R-G6: Snapshot retention at decode time** — at single-slot N_PARALLEL=1 the
  spec calls for one live checkpoint at a time, but if checkpoint is held
  during a 16-token DFlash block and target weights/KV are co-resident, peak
  VRAM may briefly double the K cache footprint. No mitigation; INTENT §Open
  Questions Q2(b) raises the cost question but only in time, not VRAM.
- **R-G7: Acceptance-rate degradation from quantized K** — DFlash drafts
  predict tokens against an f16-trained reference; if RotorQuant K
  perturbations shift logits enough to *reduce* acceptance vs vanilla
  llama.cpp DFlash, the speedup gate is unreachable for reasons unrelated to
  checkpointing. INTENT §Soft Gates mentions acceptance rate ±10pp; not in
  Codex's risks.

## 4. Missing edge cases

### 4.1 Speculative checkpointing × deferred-K interaction

Beyond the well-handled "verify forbidden during staging" rule, missing:

- **Re-entry into deferred state mid-decode**: if a long context is *extended*
  past the prefill window mid-decode (KV cache eviction + re-prefill of a
  chunk), does deferred staging re-engage? Does the guard correctly disable
  speculative until the new chunk converts? Phase 2's runtime guard task does
  not specify this.
- **Convert-during-checkpoint race**: snapshot at instant T captures backend
  pointers, conversion runs at T+ε, restore at T+δ — does the restore see a
  buffer whose layout has changed under it? If `convert_deferred_keys()` is
  re-entrant or interruptible, this is a TOCTOU.
- **Checkpoint during partial conversion**: if convert is per-layer, a
  snapshot taken mid-loop captures a heterogeneous state (some layers
  staging, some quantized). The Phase 2 test should explicitly cover this
  intermediate state, not only the two pure end states.

### 4.2 Snapshot cost at long context

Phase 2 measures 65K and 262K. Missing:

- **8K / 16K / 32K** — the *common* user context. A 1.5× speedup that only
  exists at long context is a marketing number, not a product.
- **Delta vs full snapshot path**: INTENT §Open Questions Q2(b) asks "does
  upstream do delta snapshots? COW pages?" — Codex's draft never returns to
  this question. If the answer is "full state copy," the snapshot cost
  budget needs to be revisited and the in-scope work expands to add COW or
  delta support — a much bigger commitment than the draft scopes.
- **Snapshot bandwidth budget vs PCIe / HBM topology**: cost units. INTENT
  estimates ~3.3 GB at 65K for planar3 K. At HBM3 ~3 TB/s that's ~1 ms/copy,
  trivial; at PCIe 4.0 ~32 GB/s host roundtrip, ~100 ms/copy, fatal. No
  mention of where the snapshot lives (device, host pinned, host pageable).

### 4.3 Hybrid layer KV growth split (75% recurrent vs 25% full_attention)

This is the *load-bearing* asymmetry of these models and the draft barely
acknowledges it operationally:

- **Quantization scope**: RotorQuant K compression only applies to the 25%
  full_attention layers (recurrent layers don't have a compressible K cache;
  they have a fixed-size SSM state). The draft never says this explicitly.
  The rebase audit task ("Confirm the rebased fork still loads both Qwen3.6
  hybrid architectures and that `linear_attention` layers execute correctly
  with RotorQuant enabled") is the right impulse but doesn't separate
  "RotorQuant correctly *no-ops* on linear_attention" from "RotorQuant
  correctly applies to full_attention."
- **Memory accounting in DoD**: the headline cost number ("3.3 GB at 65K") is
  for the 25% slice; the recurrent state has its own (small, fixed) cost
  that nonetheless must be checkpointable. No DoD line bounds either.
- **Per-layer-type benchmark**: snapshot cost is dominated by the 25% slice
  (linear/in-token), while *correctness* risk lives in the 75% slice. The
  benchmark and the test target opposite layer slices; the draft doesn't
  acknowledge the inversion.
- **PPL regression sweep coverage**: §Implementation Phase 1 task to "Run the
  full L1 KV regression sweep for all four KV types" — but on hybrid targets,
  the four KV types only differ in 25% of layers. The PPL signal is
  proportionally diluted. Need either a layer-type-restricted PPL gate or
  acknowledgement that ±0.05 PPL on a hybrid model is a weaker signal than
  on a pure-attention model.

### 4.4 Forced-rejection coverage

This is the most under-specified area:

- **What forces rejection at temp=0?** With deterministic sampling, draft
  acceptance is a function of (target weights, draft weights, prompt). To
  *force* rejection you need either an adversarial prompt (slow, unstable
  test), an injected mismatch, or a draft model deliberately misaligned.
  Phase 4's task "forces at least one rejection path" doesn't specify the
  mechanism. A unit-level fault-injection in `common/speculative.cpp` to
  reject the Nth token would be cleaner.
- **Coverage of rejection positions**: rejection at position 1 (full restore +
  no replay), position N-1 (full restore + N-1 replay), position N (no
  rejection) are different code paths. "At least one rejection path" hits one.
- **Recurrent-state assertion under rejection**: the test should snapshot
  recurrent state pre-draft, force rejection, restore, and assert the
  recurrent state matches the pre-draft snapshot bit-for-bit. Token equality
  alone admits silent recurrent corruption (see §2.4).
- **Rejection × deferred-K guard**: if rejection logic ever drives the system
  back into a "needs convert" state (it shouldn't, but the test should
  prove it), the guard must re-engage.
- **Multi-rejection / cascading rejection in same decode**: the e2e test
  should run long enough to see ≥3 rejections, not just ≥1.

## 5. Definition of Done completeness

Question: would meeting the DoD as written prove the sprint succeeded **on a
HYBRID target**? Verdict: **partially, with two large holes**.

Walking the DoD bullets:

| # | Bullet | Proves hybrid success? |
|---|--------|------------------------|
| 1 | Rebase + RotorQuant survives | Yes for KV; **no** verification that linear_attention layers were exercised under RotorQuant |
| 2 | L1 PPL sweep ±0.05 on both models | Partially — PPL parity is diluted on hybrid (only 25% layers differ); does not prove recurrent-layer fidelity |
| 3 | Bit-exact checkpoint test for **K layouts** | **No** — covers 25% layer slice only; recurrent state untested |
| 4 | Speculative refuses pre-conversion; no `seq_rm` path | Yes (negative gate is good) — but no positive proof the guard fires correctly |
| 5 | Greedy equivalence on 3 prompts | **Weak** — temp=0 token equality can mask silent recurrent corruption if rejection rate is low; doesn't gate on rejection actually firing |
| 6 | Qwen3.6-27B ≥ 1.5× quicksort | Yes for that prompt; not generalizable (see §2.5) |
| 7 | MoE behind EXPERIMENTAL=1 | Process gate, not correctness gate |
| 8 | Existing profiles preserved | Yes |
| 9 | README + benchmark docs | Yes |

**Two missing DoD lines that would close the hybrid gap:**

- **DoD-NEW-A**: `tests/test-checkpoint-hybrid-state.cpp` (or extension of the
  existing test) proves bit-exact save/restore of `linear_attention`
  recurrent state on at least one Qwen3.6 layer, including the SSM scratch.
- **DoD-NEW-B**: e2e rejection test asserts that recurrent-layer state at
  position N+1 post-restore equals recurrent-layer state at position N+1
  along the target-only trajectory, *separately* from token equality.

**Two further DoD gaps independent of hybrid:**

- **DoD-NEW-C**: snapshot+restore round-trip wallclock at 65K planar3 ≤ X ms
  (set X during Phase 2; current draft has no number).
- **DoD-NEW-D**: `validate_dflash.py` against z-lab reference passes on
  reference prompts (or remove `validate_dflash.py` from scope; current draft
  is in the middle).

**Recommendation**: a merged draft that adds DoD-NEW-A and DoD-NEW-B is the
minimum bar to honestly claim hybrid-target success. Without them, this sprint
can ship green and silently corrupt user output on rejection paths — exactly
the failure mode INTENT §Critical Architecture Finding flagged when it wrote
"silent corruption if rejection is handled but recurrent state isn't reset."

