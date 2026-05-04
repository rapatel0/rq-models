# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 69.86 | 124.61 | 125.68 | 1.78 | 1.80 | 61.11% | 61.11% |
| 'Explain the Pythagorean theorem.' | 69.30 | 45.54 | 46.28 | 0.66 | 0.67 | 18.53% | 18.53% |
| 'Plan a 1 day trip to DC.' | 69.35 | 20.20 | 20.25 | 0.29 | 0.29 | 0.89% | 0.89% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.31 | 19.25 | 19.24 | 0.28 | 0.28 | 0.17% | 0.11% |
| 'Write a SQL query to find the top 5 customers by' | 69.27 | 19.19 | 19.21 | 0.28 | 0.28 | 3.67% | 0.00% |

**Median DFlash×**: 0.29 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 1.80 (headline ≥1.5 → PASS)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
