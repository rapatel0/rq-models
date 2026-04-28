# Sprint 005 L4 — 3-way decode tok/s (qwen36)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 220.73 | 227.10 | 253.77 | 1.03 | 1.15 | 100.00% | 100.00% |
| 'Explain the Pythagorean theorem.' | 215.17 | 110.65 | 112.05 | 0.51 | 0.52 | 100.00% | 100.00% |
| 'Plan a 1 day trip to DC.' | 215.18 | 69.06 | 68.75 | 0.32 | 0.32 | 100.00% | 100.00% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 214.75 | nan | nan | nan | nan | n/a% | n/a% |
| 'Write a SQL query to find the top 5 customers by' | 221.51 | nan | 140.47 | nan | 0.63 | n/a% | 100.00% |

**Median DFlash×**: 0.58 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 1.15 (headline ≥1.5 → FAIL)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
