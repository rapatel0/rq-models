# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 69.34 | 75.83 | 76.19 | 1.09 | 1.10 | 100.00% | 100.00% |
| 'Explain the Pythagorean theorem.' | 69.32 | 54.38 | 55.34 | 0.78 | 0.80 | 100.00% | 100.00% |
| 'Plan a 1 day trip to DC.' | 69.36 | nan | nan | nan | nan | n/a% | n/a% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.43 | 43.98 | 43.73 | 0.63 | 0.63 | 100.00% | 100.00% |
| 'Write a SQL query to find the top 5 customers by' | 69.39 | 55.24 | 55.91 | 0.80 | 0.81 | 100.00% | 100.00% |

**Median DFlash×**: 0.80 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 1.10 (headline ≥1.5 → FAIL)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
