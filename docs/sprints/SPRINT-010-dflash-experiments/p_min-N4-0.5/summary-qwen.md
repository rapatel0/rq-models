# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 70.18 | 153.81 | 155.99 | 2.19 | 2.22 | 96.00% | 96.00% |
| 'Explain the Pythagorean theorem.' | 69.63 | 80.11 | 80.24 | 1.15 | 1.15 | 55.21% | 55.21% |
| 'Plan a 1 day trip to DC.' | 69.63 | 42.29 | 42.27 | 0.61 | 0.61 | 21.51% | 21.51% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.55 | 36.77 | 36.71 | 0.53 | 0.53 | 15.61% | 15.61% |
| 'Write a SQL query to find the top 5 customers by' | 69.57 | 70.63 | 70.66 | 1.02 | 1.02 | 21.65% | 48.08% |

**Median DFlash×**: 1.02 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 2.22 (headline ≥1.5 → PASS)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
