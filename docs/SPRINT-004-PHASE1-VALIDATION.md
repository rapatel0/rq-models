# Sprint 004 Phase 1 — Validation steps (run on GPU box)

**Phase 1 code is committed:** `rapatel0/rq-vllm@e04001ae3` on
`feature/rotorquant`. This document is the explicit checklist for
validating that the passthrough wiring works on the RTX 5090 box.

## Prerequisites

- RTX 5090 box with NVIDIA driver supporting CUDA 12.9 (or set
  `--build-arg CUDA_VERSION=12.x` for whatever the host driver
  supports — check `nvidia-smi` for "CUDA Version: ...").
- Docker with NVIDIA Container Toolkit installed and `--gpus all`
  working.
- HF cache or HF token for downloading Qwen3-27B (~54 GB at fp16).
- ~20 GB free disk for the docker image.

## Step 1: Build the docker image

```bash
cd ~/repos/rq-models
docker build -t rq-vllm -f docker/Dockerfile.vllm .
```

Expected: ~20-30 min for first build (PyTorch wheel download + vLLM
source compile). Subsequent rebuilds are faster due to layer caching.

If the build fails on `pip install -e .` for vLLM, common fixes:
- `--build-arg CUDA_VERSION=12.6.3` if host driver is older
- `--build-arg RQ_VLLM_BRANCH=v0.19.1` if you want the unmodified tag
  rather than the feature branch (Phase 1 should still build cleanly
  on `feature/rotorquant`)

## Step 2: Phase 0 baseline (unmodified-vLLM equivalent serving)

This validates the substrate works before any rotorquant flag.

```bash
docker run --rm --gpus all -p 8080:8080 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e MODEL=Qwen/Qwen3-27B \
    -e MAX_MODEL_LEN=4096 \
    rq-vllm
```

In another terminal, smoke-test:
```bash
curl http://localhost:8080/v1/models
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-27B","messages":[{"role":"user","content":"Hello"}],"max_tokens":20,"temperature":0,"seed":42}' \
    | jq -r '.choices[0].message.content' \
    | tee /tmp/sprint004-phase0-baseline.txt
```

Expected: a coherent short response. Save it (`tee /tmp/...`) — Step 3
will compare against it.

Stop the container (Ctrl+C in the docker terminal).

## Step 3: Phase 1 smoke — rotorquant passthrough

```bash
docker run --rm --gpus all -p 8080:8080 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e MODEL=Qwen/Qwen3-27B \
    -e MAX_MODEL_LEN=4096 \
    -e ROTORQUANT_MODE=planar3 \
    rq-vllm
```

Watch the entrypoint log line `rq-vllm starting:` — it should print
`args:` containing `--kv-cache-dtype rotorquant_planar3`. If vLLM
rejects the flag at this point with an error like
`unrecognized kv-cache-dtype 'rotorquant_planar3'`, Phase 1 has a
wiring bug — check that the rq-vllm fork's `feature/rotorquant`
branch is what the docker image actually built (commit SHA at
`/etc/rq-vllm-commit` in the container should be `e04001ae3` or
later).

Once serving, run the same smoke:
```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-27B","messages":[{"role":"user","content":"Hello"}],"max_tokens":20,"temperature":0,"seed":42}' \
    | jq -r '.choices[0].message.content' \
    | tee /tmp/sprint004-phase1-rotorquant.txt
```

## Step 4: Bit-identicality check (Phase 1 hard gate)

```bash
diff /tmp/sprint004-phase0-baseline.txt /tmp/sprint004-phase1-rotorquant.txt
```

**Expected: empty diff (bit-identical output).** This is the Phase 1
hard gate. Phase 1 is correct passthrough wiring; it should produce
EXACTLY the same tokens as the fp16 baseline. If the diff is
non-empty, Phase 1 has a hidden side-effect somewhere in the dispatch
plumbing that's perturbing the KV cache contents — debug before any
Phase 2 kernel work.

Common causes of unexpected diffs at this stage:
- Greedy sampling tie-broken differently (rare — normalization shouldn't
  shift). Use `temperature=0,seed=42` to make this deterministic.
- vLLM internally consults `is_quantized_kv_cache(...)` somewhere
  unexpected and our dtype matches `startswith("rotorquant_")` in a
  branch we missed. Grep `vllm/` for `kv_cache_dtype` string-matches.
- FlashAttention version detection (some FA branches use fp8 paths
  when they detect a "non-standard" dtype). Pin
  `--attention-backend FLASHINFER` or use `--enforce-eager` to
  bypass.

## Step 5: PPL check (optional, Phase 1 sanity)

If Step 4 passes, also confirm with the perplexity harness:

```bash
python3 scripts/eval_perplexity.py \
    --model Qwen/Qwen3-27B \
    --bits 16 \
    --eval-dataset wikitext-2-raw-v1 \
    --max-tokens 512 \
    --server http://localhost:8080
```

Run it twice — once with `ROTORQUANT_MODE=planar3` and once without
— and confirm PPL matches to within Δppl ≤ 0.001 (Phase 1 is
passthrough so they should match exactly modulo fp16 nondeterminism in
batch ordering).

## After Phase 1 passes

Once the bit-identicality gate is green:
- Update SPRINT-004 Validation Status table: mark Phase 1 smoke ✅.
- Capture the f16 baseline PPL on Qwen3-27B and Qwen3.5-27B as
  reference numbers for the Phase 3 comparison.
- Move to Sprint 004 Phase 2: port the actual planar3 CUDA kernels
  from `johndpope/llama-cpp-turboquant@20efe75`.
