# Q4_K_M DFlash draft — N=2 (no imatrix)

**Build**: rotorquant @ commit `5205492` (fork @ `526097eed`).
**Draft model**: `Qwen3.6-27B-DFlash-Q4_K_M.gguf` (981 MiB, vs 3.47 GiB BF16).
**Quantize**: `llama-quantize Q4_K_M` *without* imatrix calibration.
Bartowski's `Qwen_Qwen3.6-27B-imatrix.gguf` (the only published Qwen3.6
imatrix) is NOT compatible — DFlash decoder has different attention
shape (4096) than the target Qwen3.6 backbone (6144).

| Prompt | target-only | DFlash× (Q4) | DFlash acc (Q4) | Notes |
|---|---:|---:|---:|---|
| Quicksort | 70.01 | **1.53** | 100.00% | Better than BF16 (1.41×) |
| Pythagorean | 69.47 | 1.30 | 90.30% | Slightly better than BF16 (1.23×) |
| DC trip | 69.44 | 1.02 | 73.97% | About same as BF16 (1.03×) |
| **Hamlet** | 69.40 | **0.57** | **22.4% avg (67%/0%/0%)** | **Catastrophic regression vs BF16 0.91×, 0% acceptance on 2/3 trials** |
| SQL | 69.42 | 1.28 | 89.55% | Slightly better than BF16 (1.21×) |
| **Median** | — | **1.28** | — | (vs BF16 1.21) |
| **Worst** | — | **0.57** | — | (vs BF16 0.91) |

## Verdict

Plain Q4_K_M (no imatrix) is **unsafe as default**. Hamlet trial-1/2
collapse to 0% draft acceptance — every draft token is rejected, so
the slot pays draft generation cost without any speedup, ending up
worse than target-only.

Trade-off:
- Q4: median **1.28×**, worst **0.57×**, draft size 981 MiB
- BF16: median 1.21×, worst 0.91×, draft size 3470 MiB

Q4 wins median but **catastrophically loses worst-case**. The user's
instinct that imatrix matters was correct precisely because this
class of regression (small custom draft + entropic content + no
imatrix calibration) is what imatrix protects against.

## Status

- BF16 stays as docker-compose default (`DRAFT_MODEL_NAME=qwen3.6-27b-dflash`).
- Q4 stays registered (`qwen3.6-27b-dflash-q4`) as opt-in for code-heavy
  workloads where Hamlet-class regressions don't apply.
- See SPRINT-008-FOLLOWUPS.md F-028 for the imatrix calibration gap.
