# Sprint 004 Deferred Items

Items discussed during planning, proposed in drafts, or raised in critiques
but explicitly excluded from Sprint 004 scope. Each item names what, why
deferred, target sprint, prerequisites, and affected files.

---

## D-001: EAGLE3 Speculative Decoding (full)

**What**: Cherry-pick the complete EAGLE3 implementation from PR #18039:
`src/models/eagle3.cpp` graph (~186 LOC), `--eagle3` runtime flag, EAGLE3
GGUF conversion path in `convert_hf_to_gguf.py`, EAGLE3 tensor mappings,
`d2t`/`t2d` int-tensor handling for speculators-format drafts. Plus a new
Docker profile `qwen36-27b-eagle3` (or similar).

**Why deferred**: Sprint 004 cherry-picks only the *minimal* EAGLE3
foundation commits that DFlash structurally depends on (LLM_ARCH enum
slots, `g_embeddings` plumbing, cross-attention graph hooks). Full EAGLE3
is its own delivery surface — different draft GGUF format, different
acceptance characteristics, different performance envelope. Bundling it
with DFlash doubles diff size and risk.

EAGLE3 is autoregressive 1-token-per-step (vs DFlash's 16-token block);
they're complementary speculative paths with different best-fit prompt
profiles. EAGLE3's main appeal is broader compatibility (more existing
draft GGUFs in the wild), not better speedup on Qwen3.6.

**Target sprint**: Sprint 005 or 006 if DFlash adoption signal is strong.

**Prerequisites**: Sprint 004 DFlash + speculative checkpointing × deferred-K
must be validated and merged. EAGLE3's runtime path uses the same
checkpointing infrastructure DFlash exercises in Sprint 004.

**Files**:
- Fork: `src/models/eagle3.cpp` (full graph), `convert_hf_to_gguf.py`
  (EAGLE3 conversion + d2t/t2d), `common/speculative.cpp`
  (`common_speculative_init_eagle3` integration), `examples/speculative-simple/`
  (`--eagle3` flag).
- Repo: `docker-compose.yml` (new EAGLE3 profile), `docker/entrypoint.sh`
  (EAGLE3 mode in command builder).

---

## D-002: Multi-Slot Speculative Decoding (`N_PARALLEL > 1`)

**What**: Allow `qwen36-27b-dflash` and similar profiles to run with
multiple parallel slots (e.g., `N_PARALLEL=4`). This means each slot has
its own checkpoint state, draft state, and verify pipeline.

**Why deferred**: Speculative checkpointing in upstream is single-slot
proven. Multi-slot interactions with the snapshot/restore mechanism are
genuinely uncharted at the upstream layer — verify timing, snapshot
ordering, and shared draft state across slots are all unsettled. Sprint 004
explicitly hardcodes `N_PARALLEL: 1` in DFlash profiles to avoid this
complexity.

**Target sprint**: Sprint 006 (after DFlash and EAGLE3 are individually
validated single-slot).

**Prerequisites**: Sprint 004 single-slot DFlash validated; upstream
position on multi-slot speculative clarified (may require waiting for
mainline support).

**Files**:
- Fork: `tools/server/server-context.cpp` (currently rejects
  `n_parallel > 1` for speculative); `common/speculative.cpp` (per-slot
  draft state).
- Repo: `docker-compose.yml` (multi-slot DFlash profile variants),
  `docker/entrypoint.sh` (multi-slot validation).

---

## D-003: Non-Greedy Sampler Validation Under Speculative

**What**: Validate DFlash output equivalence to target-only at `--temp >0`,
non-greedy sampling. This requires a different correctness framework
because token sequences are not deterministic — instead, distribution
equivalence (KL divergence on logits or chi-squared on token frequencies
across many seeds) is the right metric.

**Why deferred**: Sprint 004 explicitly scopes greedy (`--temp 0 --top-k 1`)
only. Non-greedy validation is its own substantial work: build a sampler-
agreement test framework, decide on the metric (logit KL? token frequency
chi-squared? rejection-region statistical tests?), generate enough samples
for statistical power. None of this is straightforward and it's tangential
to the "is DFlash + RotorQuant correct?" question that Sprint 004 answers.

**Target sprint**: Sprint 005 or 006 (paired with productionization once
greedy correctness is established).

**Prerequisites**: Sprint 004 greedy correctness validated; sampler-
agreement statistical framework decided.

**Files**:
- Repo: `scripts/validate_dflash_sampling.py` (new), `tests/test_dflash_sampling.py`
  (new).

---

## D-004: MoE Deep-Dive Profiling

**What**: Detailed analysis of why Qwen3.6-35B-A3B MoE speedup is lower
than dense (per upstream PR data, MoE often goes negative on speculative
decoding). Specifically: per-expert activation count during parallel verify
vs autoregressive decode, expert routing determinism between target and
draft, MoE-aware draft model design.

**Why deferred**: Sprint 004 ships `qwen36-dflash` as `EXPERIMENTAL=1`
specifically because the speedup story is uncertain. Investigating *why*
is its own research project that doesn't block productionization for the
dense profile (which is the headline deliverable).

**Target sprint**: Sprint 006 or later (research, not productionization).

**Prerequisites**: Sprint 004 MoE profile shipped and benchmarked; whatever
slowdown/speedup story emerges is the input to this investigation.

**Files**:
- Repo: `scripts/profile_moe_speculative.py` (new); analysis report
  appended to `BENCHMARK-REPORT.md`.

---

## D-005: COW / Delta Snapshot Upstream Contribution

**What**: If Phase 1 of Sprint 004 reveals that upstream's speculative
checkpointing takes a full-copy snapshot (not append-only or COW), and
Phase 2 measures that the snapshot cost ceiling is missed (>5 ms at 65K),
contribute a copy-on-write or delta snapshot path to upstream llama.cpp.
This would be a substantive PR against `ggml-org/llama.cpp` modifying
the `llama_memory_*::checkpoint_*` interface.

**Why deferred**: Sprint 004 only takes this on as a fallback if measurement
forces it. Most likely outcome (per the architecture's append-only KV
behavior): the snapshot is already cheap and this contribution is
unnecessary. If needed, it's a separate sprint.

**Target sprint**: Sprint 005 contingent on Sprint 004 Phase 1/2 spike
results.

**Prerequisites**: Sprint 004 Phase 1 spike concludes "snapshot is full-
copy" AND Phase 2 measurement misses the ceiling AND user decides
upstream contribution is the right path (vs accepting reduced ctx).

**Files**: Upstream llama.cpp `src/llama-memory*.cpp`, `src/llama-context.cpp`,
plus tests.

---

## D-006: Server-Side Streaming `--dflash` Validation

**What**: Validate that `qwen36-27b-dflash` works correctly under
`/v1/chat/completions` streaming mode (`stream: true` in OpenAI-compat
API). Specifically: SSE event stream produces tokens in the right order,
no gaps where speculative checkpointing pauses/restarts, no
double-emission on rejection rollback.

**Why deferred**: Sprint 004 e2e test exercises non-streaming completion
only. Streaming adds another layer of correctness concerns (event
ordering, partial tokens, mid-stream checkpoint). It's an important
production capability but not blocking for "speculative works correctly".

**Target sprint**: Sprint 005 (alongside any productionization push for
DFlash).

**Prerequisites**: Sprint 004 non-streaming DFlash validated.

**Files**:
- Repo: `tests/test_dflash_streaming.py` (new); `scripts/validate_dflash.py`
  (extend with `--stream` flag).

---

## D-007: Quant-Tier Downgrade Analysis on 24 GB Tier with DFlash

**What**: With DFlash enabled, the 24 GB tier (RTX 4090, RTX 3090) may
need to drop from UD-Q4_K_XL (16.4 GB) to UD-Q3_K_XL (~12 GB) for
Qwen3.6-27B to leave room for draft + checkpoint. Document the trade-off
explicitly: PPL impact of going from Q4 to Q3, decode speedup actually
realized on smaller quant.

**Why deferred**: Sprint 004 targets 32 GB hardware; 24 GB is a
recommendation surface, not a hard target. The full analysis is its own
benchmark project after the 32 GB story is solid.

**Target sprint**: Sprint 005 (alongside QUANTIZATION-GUIDE.md update).

**Prerequisites**: Sprint 004 32 GB DFlash benchmarks complete; PPL
baseline for Q3 dense already exists in BENCHMARK-REPORT.md.

**Files**:
- Repo: `docs/QUANTIZATION-GUIDE.md` (revised 24 GB section);
  `docker-compose.yml` (possibly add `qwen36-27b-q3-dflash` profile if Q3
  + DFlash is recommended).

---

## D-008: Open WebUI Integration (carry-forward from Sprint 002 D-011)

**What**: Add Open WebUI as a compose service for browser-based chat.

**Why deferred**: Carried forward from Sprint 002 → Sprint 003 → now
Sprint 004 → Sprint 005. Sprint 004 is too narrowly focused on speculative
correctness to take on a UI integration. Open WebUI has well-understood
integration; defer until quantization/speculative work has stabilized.

**Target sprint**: Sprint 005 or later.

**Prerequisites**: None technical; scheduling constraint only.

**Files**: `docker-compose.yml` (add open-webui service), `docker/`
(possibly nginx proxy config).

---

## D-009: Benchmark CI (carry-forward from Sprint 002 D-013)

**What**: Automated CI pipeline that runs perplexity and throughput
benchmarks on every PR, preventing regression. With Sprint 004's new
`scripts/ppl_sweep.py`, `scripts/bench_speculative.py`, and
`scripts/bench_snapshot_cost.py`, the benchmark scripts become more
valuable to gate on automatically.

**Why deferred**: Carried forward from Sprint 002 → Sprint 003 → now
Sprint 004. Requires GPU-equipped GitHub Actions self-hosted runner setup,
which is its own infrastructure project.

**Target sprint**: Sprint 005 or 006.

**Prerequisites**: Sprint 004 benchmark scripts stable; GPU runner provisioned.

**Files**: `.github/workflows/benchmark.yml` (new); `scripts/`
(benchmark scripts must remain CI-compatible with fixed seeds).

---

## D-010: SpectralQuant Items (stale, from Sprint 003)

**What**: All Sprint 003 deferred items D-001..D-009 (SPECTRAL_3BIT variant,
hybrid SpectralQuant + Clifford rotor, GGUF bridge script, C/CUDA
implementation, Qwen3.6-35B SpectralQuant support, etc.).

**Why deferred**: Sprint 003 produced a negative result — SpectralQuant
+70.9 PPL failure on Qwen3.5-9B. The line of work is **not abandoned but
not active**. Sprint 003's deferred items are stale until/unless someone
revisits the SpectralQuant approach.

**Target sprint**: Marked as parked indefinitely. Will be revisited only
if architecture changes or new evidence emerges that low-rank
KV compression can work on QK-normalized models.

**Prerequisites**: New evidence that low-rank KV compression is viable on
Qwen architectures; or different model family (non-Qwen).

**Files**: `turboquant/spectral/` (research-only Python module, untouched
by Sprint 004).

---

## Summary table

| Item | Description | Target Sprint | Blocker |
|------|-------------|---------------|---------|
| D-001 | EAGLE3 full integration | Sprint 005-006 | DFlash + checkpointing validated |
| D-002 | Multi-slot speculative | Sprint 006 | Single-slot validated; upstream multi-slot story |
| D-003 | Non-greedy sampler validation | Sprint 005-006 | Greedy validated; sampler-agreement framework |
| D-004 | MoE deep-dive profiling | Sprint 006+ | MoE profile shipped (Sprint 004) |
| D-005 | COW snapshot upstream contribution | Conditional Sprint 005 | Phase 1 spike says full-copy + Phase 2 misses ceiling |
| D-006 | Streaming `/v1/chat/completions` validation | Sprint 005 | Non-streaming validated |
| D-007 | 24 GB tier quant-tier downgrade analysis | Sprint 005 | 32 GB benchmarks complete |
| D-008 | Open WebUI integration (carry from S002) | Sprint 005+ | None technical |
| D-009 | Benchmark CI (carry from S002) | Sprint 005-006 | GPU runner provisioned |
| D-010 | SpectralQuant items (stale) | Parked | New low-rank evidence |
