# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 70.19 | 152.44 | 155.38 | 2.17 | 2.21 | 96.00% | 96.00% |
| 'Explain the Pythagorean theorem.' | 69.68 | 79.99 | 80.12 | 1.15 | 1.15 | 55.21% | 55.21% |
| 'Plan a 1 day trip to DC.' | 69.63 | 42.31 | 42.17 | 0.61 | 0.61 | 21.51% | 21.51% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.56 | 23.92 | 23.82 | 0.34 | 0.34 | 0.00% | 3.37% |
| 'Write a SQL query to find the top 5 customers by' | 69.61 | 70.66 | 70.46 | 1.02 | 1.01 | 48.08% | 48.08% |

**Median DFlash×**: 1.01 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 2.21 (headline ≥1.5 → PASS)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
