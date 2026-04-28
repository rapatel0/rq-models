# Sprint 005 L4 — 3-way decode tok/s (qwen36)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 215.12 | 175.27 | 178.06 | 0.81 | 0.83 | 100.00% | 100.00% |
| 'Explain the Pythagorean theorem.' | 214.75 | 124.11 | 124.51 | 0.58 | 0.58 | 100.00% | 100.00% |
| 'Plan a 1 day trip to DC.' | 215.29 | 83.40 | 81.58 | 0.39 | 0.38 | 100.00% | 100.00% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 215.39 | 76.47 | 75.79 | 0.36 | 0.35 | 100.00% | 100.00% |
| 'Write a SQL query to find the top 5 customers by' | 214.71 | 112.16 | 111.74 | 0.52 | 0.52 | 100.00% | 100.00% |

**Median DFlash×**: 0.52 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 0.83 (headline ≥1.5 → FAIL)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
