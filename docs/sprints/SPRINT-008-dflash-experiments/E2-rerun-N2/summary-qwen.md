# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 70.01 | 98.11 | 98.84 | 1.40 | 1.41 | 97.92% | 97.92% |
| 'Explain the Pythagorean theorem.' | 69.47 | 85.59 | 85.38 | 1.23 | 1.23 | 90.30% | 90.30% |
| 'Plan a 1 day trip to DC.' | 69.44 | 71.62 | 71.36 | 1.03 | 1.03 | 77.62% | 77.62% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.40 | 63.49 | 63.38 | 0.91 | 0.91 | 68.21% | 68.21% |
| 'Write a SQL query to find the top 5 customers by' | 69.40 | 84.18 | 84.03 | 1.21 | 1.21 | 89.55% | 89.55% |

**Median DFlash×**: 1.21 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 1.41 (headline ≥1.5 → FAIL)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
