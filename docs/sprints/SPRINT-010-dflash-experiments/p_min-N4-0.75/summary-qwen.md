# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 70.23 | 155.29 | 152.78 | 2.21 | 2.18 | 96.00% | 96.00% |
| 'Explain the Pythagorean theorem.' | 69.62 | 80.30 | 79.99 | 1.15 | 1.15 | 55.21% | 55.21% |
| 'Plan a 1 day trip to DC.' | 69.60 | 42.29 | 42.28 | 0.61 | 0.61 | 21.51% | 21.51% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.61 | 36.72 | 23.88 | 0.53 | 0.34 | 15.61% | 0.00% |
| 'Write a SQL query to find the top 5 customers by' | 69.54 | 70.58 | 70.59 | 1.01 | 1.01 | 48.08% | 48.08% |

**Median DFlash×**: 1.01 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 2.18 (headline ≥1.5 → PASS)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
