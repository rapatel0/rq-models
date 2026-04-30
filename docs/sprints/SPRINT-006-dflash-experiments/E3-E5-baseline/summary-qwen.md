# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 70.30 | 55.60 | 56.09 | 0.79 | 0.80 | 25.00% | 25.00% |
| 'Explain the Pythagorean theorem.' | 69.69 | 14.21 | 14.20 | 0.20 | 0.20 | 5.06% | 5.06% |
| 'Plan a 1 day trip to DC.' | 69.65 | 10.91 | 10.89 | 0.16 | 0.16 | 2.33% | 2.33% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.56 | 10.88 | 10.96 | 0.16 | 0.16 | 1.43% | 1.43% |
| 'Write a SQL query to find the top 5 customers by' | 69.47 | 14.92 | 10.84 | 0.21 | 0.16 | 1.38% | 1.24% |

**Median DFlash×**: 0.16 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 0.80 (headline ≥1.5 → FAIL)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
