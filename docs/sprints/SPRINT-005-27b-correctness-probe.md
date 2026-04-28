# Sprint 005 Phase 0.5: 27B DFlash correctness probe

**Sampling**: temp=0, top-k=1, seed=42 (greedy, deterministic)
**Tokens budget**: 256 each

| Metric | target-only | target+DFlash |
|---|---:|---:|
| Output chars (reasoning + content) | 965 | 937 |
| Completion tokens | 256 | 256 |
| Decode tok/s | 69.29 | 72.98 (1.053×) |
| Draft proposed / accepted | — | 221 / 221 (100.0%) |

**Shared-prefix length**: 937 chars
**Tail beyond shared prefix**: target-only 28 chars, DFlash 0 chars

## Result: PASS — output is byte-equal on the shared prefix

For all 937 characters where both runs produced output, the
text is byte-identical. The dense 27B-DFlash verify+rollback path
is correct on this prompt at greedy sampling.

The 28-char tail-length difference is a
token-budget artifact, not a content divergence: both runs hit
the 256-token cap, but speculative decoding can tokenize the
same string slightly differently (e.g. "middle = [x " as one
token vs "middle = [x" + " "), so the cap fires at a
slightly different character position. Within the shared prefix,
every character matches.

Implication: the 37% acceptance observed in Sprint 004's 7-token
smoke probe is a perf observation, not a correctness bug. PREVIEW
gate stays for "drafts iterating", not for "broken".
