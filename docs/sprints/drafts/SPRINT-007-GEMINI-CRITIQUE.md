# Sprint 007-mtp Critique: True Multislot MTP

This document provides a comparative critique of the draft sprint plans for Sprint 007-mtp, which aims to implement true multislot speculative decoding for the MTP path.

## Comparison Overview

| Feature | Claude Draft | Codex Draft |
| :--- | :--- | :--- |
| **Focus** | Deep architectural instrumentation and "descope-friendly" phase ordering. | Profile-first validation and hard "promotion" gates for production readiness. |
| **Risk Management** | Strong on "stop and descope" if LOC exceeds estimates; detailed spike. | Strong on correctness and VRAM isolation as a primary risk. |
| **Architecture** | Detailed callgraph analysis and specific upstream file targets. | Higher-level "scatter/gather" conceptualization. |
| **Testing** | Five specific test cases (greedy, rejection, long-prompt, etc.). | Generic correctness checks with a focus on greedy equivalence. |
| **DoD** | Highly specific (e.g., "A4 within 5% of 124.5 t/s"). | Clearer on "Promotion" vs "Success" (e.g., "B4 >= 1.4 * B1"). |

---

## 1. Claude Draft Critique

### Strengths
- **Phase Ordering:** The decision to front-load Phase 2 (Instrumentation) ensures the sprint delivers value (data) even if the implementation (Phase 3) is descoped. This is a mature engineering strategy.
- **Traceability:** Explicitly defines `LLAMA_SPEC_TRACE` and JSON schema for timings. This makes the "why" of any failure visible to the operator.
- **Byte-Exact Gates:** Provides a very clear test plan for correctness (rejection-heavy, long-prompt mixed cases) which is critical for speculative decoding where subtle state leaks often hide.
- **Upstream Alignment:** Proactively references PR #18961 to minimize future rebase friction.

### Weaknesses
- **Complexity of Phase 3:** The "promotion of `state_draft_mtp` from per-context to per-sequence" is a significant structural change. Claude identifies this as a risk but might under-estimate the ripples into the sampler and rollback logic in `common/speculative.cpp`.
- **VRAM Detail:** While it mentions VRAM, it doesn't specify a concrete monitoring strategy during the matrix run beyond "headroom check."

### Gaps / Missing Edge Cases
- **KV Fragmentation:** At `np=4`, if the total context is divided but users have different prompt lengths, KV cache fragmentation might occur. The plan doesn't explicitly mention checking for "KV-eviction storms" beyond a brief mention in Phase 2.
- **Sampler State:** If slots share the same draft model but have different sampling params (temp, top-p), the batched draft pass must ensure it either uses greedy drafting (standard for MTP) or correctly applies per-slot samplers.

---

## 2. Codex Draft Critique

### Strengths
- **Promotion Clarity:** Explicitly defines "Hard Gates" for promotion (`B4 >= 1.4 * B1`). This is excellent for preventing "experiment bloat" in production.
- **Security & Isolation:** Stronger focus on cross-request state isolation as a security boundary.
- **Baseline Integrity:** Phase 0 is very strict about not starting without a committed, reproducible baseline.
- **Simplicity:** The "Architectural Invariants" section provides a very clear "north star" for the implementation.

### Weaknesses
- **Vague Implementation Steps:** Phase 2 ("Patch upstream") is less detailed than Claude's equivalent. It doesn't identify the specific structs (like `state_draft_mtp`) that need modification.
- **Descope Strategy:** Lacks the explicit "fail-safe" phases found in Claude's draft. If the patch fails, Codex's plan delivers less intermediate value.

### Gaps / Missing Edge Cases
- **Rollback Accounting:** Doesn't detail how the `llama_batch` will be reconstructed after a partial acceptance (scatter/gather logic).
- **Environment Parity:** Doesn't mention how the homelab 4090 environment will be cleaned or verified between runs to ensure "noisy neighbor" effects don't invalidate the matrix.

---

## 3. Shared Risk & Gap Analysis

### VRAM & Context Budget
Both drafts assume the `B1` VRAM usage is the ceiling. However, at `np=4`, the overhead of 4 sets of `pending_h` / `verify_h` (hidden state carry) for MTP might be non-trivial.
- **Missing Task:** A VRAM-usage profile comparison between `B1` and `B4` (idle and under load) should be a mandatory part of Phase 1.

### Multi-Step MTP vs Single-Step
MTP often drafts multiple tokens ahead. The "batching" needs to handle slots that might be at different "depths" in their speculative window if one slot rejects earlier than another.
- **Missing Edge Case:** "Asynchronous" slot depths in the draft batch. If Slot A needs 4 tokens and Slot B only needs 2 (due to a previous partial acceptance), the batch builder needs to handle this padding or truncation.

### Instrumentation Overhead
Both plans add timing hooks.
- **Risk:** High-frequency timing (`ggml_time_us` inside loops) can sometimes affect the very throughput numbers being measured, especially at high `np`.
- **Mitigation:** Ensure instrumentation is compiled-out or gated by a fast boolean check (`LLAMA_SPEC_TRACE`).

---

## 4. Definition of Done (DoD) Recommendations

To merge these into a final `SPRINT-007-mtp.md`, the following DoD items must be harmonized:

1. **Correctness (Must have):** 100% token match vs single-slot reference for greedy decode across N=1, 2, 4.
2. **Acceptance (Must have):** Per-slot acceptance within ±2pp of baseline (more stringent than Claude's 5pp).
3. **Throughput (Promotion Gate):** B4 aggregate ≥ 1.4x B1 AND B4 aggregate ≥ 0.7x A4.
4. **Visibility:** JSON trace artifact committed showing `draft_batch_n` histogram.
5. **Safety:** Production default (`B1`) remains bit-exact and performance-neutral (regression < 2%).

## Final Recommendation
Use **Claude's Phase Ordering and Test Cases** (Phases 0-2 as data-gathering) combined with **Codex's Promotion Gating and Security Constraints**. The resulting sprint should prioritize the "Instrumentation-only" delivery as a fallback if the "True Multislot" implementation hits architectural blockers in Phase 3.
