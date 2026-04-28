# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 70.38 | 129.84 | 125.26 | 1.84 | 1.78 | 100.00% | 100.00% |
| 'Explain the Pythagorean theorem.' | 69.49 | 64.50 | 61.76 | 0.93 | 0.89 | 100.00% | 100.00% |
| 'Plan a 1 day trip to DC.' | 69.48 | nan | nan | nan | nan | n/a% | n/a% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.50 | 33.43 | 31.98 | 0.48 | 0.46 | 100.00% | 100.00% |
| 'Write a SQL query to find the top 5 customers by' | 69.43 | 20.44 | 19.72 | 0.29 | 0.28 | 100.00% | 100.00% |

**Median DFlash×**: 0.67 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 1.78 (headline ≥1.5 → PASS)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
