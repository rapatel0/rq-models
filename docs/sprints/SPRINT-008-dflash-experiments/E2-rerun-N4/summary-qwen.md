# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 69.92 | 155.80 | 155.09 | 2.23 | 2.22 | 96.00% | 96.00% |
| 'Explain the Pythagorean theorem.' | 69.44 | 80.18 | 79.75 | 1.15 | 1.15 | 55.21% | 55.21% |
| 'Plan a 1 day trip to DC.' | 69.41 | 41.83 | 41.92 | 0.60 | 0.60 | 21.51% | 21.51% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.46 | 36.47 | 36.50 | 0.53 | 0.53 | 15.61% | 15.61% |
| 'Write a SQL query to find the top 5 customers by' | 69.32 | 70.06 | 70.55 | 1.01 | 1.02 | 48.08% | 48.08% |

**Median DFlash×**: 1.02 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 2.22 (headline ≥1.5 → PASS)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
