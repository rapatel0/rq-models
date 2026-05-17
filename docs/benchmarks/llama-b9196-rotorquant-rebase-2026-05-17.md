# llama.cpp b9196 RotorQuant Rebase

Date: 2026-05-17

## Source

- Upstream stable: `ggml-org/llama.cpp` tag `b9196`
- Upstream commit: `7ba22c6a0918b5db16029c2a120bf04a56e78b78`
- RotorQuant patch: `docker/patches/llama-b9196-rotorquant.patch.gz`

The Docker image now builds from upstream stable and applies the RotorQuant KV
cache patch locally. This keeps Qwen3.6 MTP on upstream's `draft-mtp`
implementation while retaining `tbq3_0`, `tbq4_0`, `planar3_0`, `iso3_0`,
`planar4_0`, and `iso4_0` KV cache types.

## Local Validation

Built `llama-server` locally from the rebased source on Apple M1 Max with Metal:

```bash
cmake -S /tmp/llama-rq-rebase-b9196 -B /tmp/llama-rq-rebase-b9196/build-metal \
  -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama-rq-rebase-b9196/build-metal --target llama-server -j8
```

Verified `llama-server --help` advertises both upstream MTP and RotorQuant KV:

```text
--spec-type none,draft-simple,draft-eagle3,draft-mtp,ngram-simple,...
--cache-type-k TYPE ... tbq3_0, tbq4_0, planar3_0, iso3_0, planar4_0, iso4_0
--cache-type-v TYPE ... tbq3_0, tbq4_0, planar3_0, iso3_0, planar4_0, iso4_0
```

Ran a local smoke probe against the cached Unsloth Qwen3.6 27B MTP GGUF with
`--spec-type draft-mtp`, `--spec-draft-n-max 4`, `--spec-draft-p-min 0.75`,
`--ubatch-size 32`, and `q4_0` KV at 2K context:

```text
mtp: 13.00 tok/s, predicted=96, spec=none,draft-mtp, drafts=69/97 (71.1%)
PASS: MTP is generating and accepting draft tokens
```

## Follow-up Benchmark

Before routing traffic, rebuild the CUDA image and run the 4090 A/B. Restart
the MTP server once with `MTP_DRAFT_N_MAX=4` and once with `MTP_DRAFT_N_MAX=6`,
then probe each run:

```bash
python scripts/mtp_probe.py --mtp-url http://localhost:8080 \
  --base-url http://localhost:8081 --min-acceptance 0.50 --min-speedup 1.05
```
