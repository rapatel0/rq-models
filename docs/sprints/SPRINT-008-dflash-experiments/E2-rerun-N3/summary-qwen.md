# Sprint 005 L4 — 3-way decode tok/s (qwen)

| Prompt | target-only | +autoreg | +DFlash | autoreg× | DFlash× | autoreg acc | DFlash acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 'Write a quicksort algorithm in Python. Write cod' | 70.00 | 140.84 | 140.99 | 2.01 | 2.01 | 100.00% | 100.00% |
| 'Explain the Pythagorean theorem.' | 69.41 | 108.20 | 108.36 | 1.56 | 1.56 | 84.21% | 84.21% |
| 'Plan a 1 day trip to DC.' | 69.38 | 61.22 | 61.36 | 0.88 | 0.88 | 47.33% | 47.33% |
| 'Summarize the plot of Hamlet in 3 paragraphs.' | 69.39 | 49.24 | 49.32 | 0.71 | 0.71 | 33.88% | 33.88% |
| 'Write a SQL query to find the top 5 customers by' | 69.30 | 25.69 | 25.65 | 0.37 | 0.37 | 12.64% | 12.64% |

**Median DFlash×**: 0.88 (gate ≥1.3 → FAIL)

**Quicksort headline DFlash×**: 2.01 (headline ≥1.5 → PASS)

**Leg acceptance rates**: target-only=n/a%, autoregressive=100.00%, dflash=100.00%
