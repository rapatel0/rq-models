# Sprint 005 L4 Summary (Partial)

Status: PARTIAL / BLOCKED on this host.

## Completed measurements

- `qwen` target-only leg completed (5 prompts x 3 trials).
- Target-only median tok/s across prompt medians: `69.41`.
- `q4_k_xl` quicksort sweep point (`qwen`, DFlash mode) completed in `SPRINT-005-experiments.json`:
  - median tok/s: `73.48`
  - acceptance: `100%`
  - quicksort ratio vs qwen target-only quicksort median (`69.41`): `1.06x`

## Blockers

- `qwen` and `qwen36` speculative runs are unstable on this host:
  - first completion usually succeeds;
  - subsequent completions often fail with connection reset / remote close;
  - some requests hang waiting for HTTP response.
- Because of this, canonical L4 requirements (three legs x five prompts x three trials for both profiles) were not fully captured.

## Rerun commands

```bash
nvidia-smi
make bench-dflash-all PROFILE=qwen
make bench-dflash-all PROFILE=qwen36
```
