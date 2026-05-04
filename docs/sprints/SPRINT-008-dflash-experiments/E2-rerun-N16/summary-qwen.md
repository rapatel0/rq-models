# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 69.90 | 114.29 | 109.87 | 1.63 | 1.57 | 25.00% | 25.00% |
| 'Explain the Pythagorean theorem.' | 69.42 | 28.73 | 28.18 | 0.41 | 0.41 | 2.01% | 2.01% |
| 'Plan a 1 day trip to DC.' | 69.41 | 21.98 | 21.67 | 0.32 | 0.31 | 0.03% | 0.03% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.39 | 22.05 | 22.26 | 0.32 | 0.32 | 0.08% | 0.08% |
| 'Write a SQL query to find the top 5 customers by' | 69.35 | 21.72 | 22.19 | 0.31 | 0.32 | 0.67% | 0.67% |

**Median DFlash×**: 0.32 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 1.57 (headline ≥1.5 → PASS)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
