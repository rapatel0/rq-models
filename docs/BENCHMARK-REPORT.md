# RotorQuant Benchmark Report — Qwen3.5-27B on RTX 5090

**Date**: 2026-04-03
**Hardware**: NVIDIA RTX 5090 (32 GB VRAM)
**Model**: Qwen3.5-27B Q4_K_M (16.7 GB, unsloth/Qwen3.5-27B-GGUF)
**Runtime**: llama.cpp RotorQuant fork (johndpope/llama-cpp-turboquant, commit 20efe75)
**KV Cache**: iso3/iso3 (RotorQuant IsoQuant 3-bit symmetric)
**Default Context**: 131,072 tokens (128K)

---

## 1. Throughput Comparison — iso3/iso3 vs f16/f16

Measured with `llama-bench`, 1 repetition, batch_size=1.

### Prefill (prompt processing, tok/s)

| Context | f16/f16 | iso3/iso3 | Overhead |
|--------:|--------:|----------:|---------:|
| 512 | 3,608 | 3,119 | 1.2x |
| 2,048 | 3,551 | 2,331 | 1.5x |
| 4,096 | 3,490 | 1,711 | 2.0x |
| 8,192 | 3,306 | 1,109 | 3.0x |
| 16,384 | 3,015 | 649 | 4.6x |
| 32,768 | 2,502 | 357 | 7.0x |
| 65,536 | **OOM** | 189 | — |
| 131,072 | **OOM** | 98 | — |

### Decode (token generation, tok/s)

| Config | tok/s | % of f16 |
|--------|------:|---------:|
| f16/f16 | 69.3 | 100% |
| iso3/iso3 | 67.5 | **97%** |

### Batch Size Impact (iso3/iso3, 32K context)

| Batch Size | Prefill tok/s | Decode tok/s |
|-----------:|--------------:|-------------:|
| 512 | 357 | 69.1 |
| 2,048 | 357 | 67.8 |
| 4,096 | 357 | 67.6 |
| 8,192 | 357 | 67.6 |

Batch size has no measurable impact on throughput.

---

## 2. Memory Usage

### VRAM at Different Context Lengths (iso3/iso3)

| Context | VRAM Used | Headroom (of 32 GB) |
|--------:|----------:|--------------------:|
| 4,096 | 17.2 GB | 14.8 GB |
| 128K | 22.3 GB | 9.7 GB |
| 196K | 25.3 GB | 6.7 GB |

### Analytical KV Cache Size — Qwen3.5-27B (64 layers, 8 KV heads, head_dim=128)

| Config | KV at 4K | KV at 32K | KV at 128K | Compression |
|--------|----------|-----------|------------|-------------|
| f16/f16 | 1.0 GB | 8.0 GB | 32.0 GB | 1.0x |
| iso3/iso3 | 0.2 GB | 1.6 GB | 6.5 GB | **4.9x** |
| iso4/iso4 | 0.3 GB | 2.2 GB | 8.5 GB | 3.8x |

### Total VRAM Budget (model Q4_K_M 16.7 GB + KV at 128K)

| Config | Total | Fits RTX 5090? | Fits A100 40GB? |
|--------|------:|:-:|:-:|
| f16/f16 | 48.7 GB | No | No |
| iso3/iso3 | **23.2 GB** | **Yes** | **Yes** |
| iso4/iso4 | 25.2 GB | Yes | Yes |

---

## 3. Perplexity — wikitext-2 (ctx=2048)

| Config | PPL | Delta vs f16 |
|--------|----:|-------------:|
| f16/f16 | 6.38 | — |
| iso3/iso3 | 6.76 | +0.38 (+5.9%) |

---

## 4. Needle-In-A-Haystack (NIAH)

Tested via llama-server API. Needle: "The secret project codename is ZEPHYR-NINE-SEVEN-ALPHA."

### iso3/iso3 Results

| Context | Depth 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Recall |
|---------|:---------:|:---:|:---:|:---:|:---:|-------:|
| 4K | HIT | HIT | HIT | HIT | HIT | **100%** |
| 8K | HIT | HIT | HIT | MISS | HIT | **80%** |

Average recall: **90%**

---

## 5. Battery Test — Competence (27/28 passed, 96%)

Tested via Docker container (`rotorquant:latest`, iso3/iso3, Qwen3.5-27B Q4_K_M).

### Performance Metrics

| Metric | Result |
|--------|--------|
| Decode throughput | 67.5 tok/s |
| Prefill (2K tokens) | 4.0s |
| Time to first token | 0.20s |
| VRAM during battery | 22.8 GB / 32.6 GB (70%) |

### Competence Results

| Category | Pass/Total | Tests |
|----------|:----------:|-------|
| Math | 4/4 | 17×24=408, √144=12, word problem, percentages |
| Reasoning | 4/4 | Syllogism, spatial, causal, counterfactual |
| Code | 2/2 | Factorial generation, list reversal tracing |
| Knowledge | 3/4 | Tokyo, Shakespeare, 1945. H₂O vs H2O (unicode) |
| Instruction following | 3/3 | Numbered lists, constraints, JSON output |
| NIAH (battery) | 6/6 | 100% at 4K and 8K, depths 0.25/0.5/0.75 |
| Consistency | 1/1 | Deterministic at temp=0 (3 identical runs) |
| VRAM | 1/1 | 70% utilization, 10 GB headroom |

The single "failure" (H₂O vs H2O) is a unicode subscript match — the model gave the chemically correct answer with proper formatting.

---

## 6. Key Findings

1. **Decode speed is virtually unchanged**: 67.5 tok/s (iso3) vs 69.3 tok/s (f16) — 97% of baseline. Users experience no perceptible generation slowdown.

2. **Prefill has measurable overhead**: 1.2x at 512 tokens scaling to 7x at 32K. This is the rotation + quantization cost during prompt ingestion. For interactive chat (short prompts), negligible. For bulk document processing, noticeable.

3. **128K context on consumer GPU**: iso3/iso3 enables 128K context on RTX 5090 (22.3 GB) where f16 OOMs at 65K. This is the primary value proposition.

4. **Quality degradation is minimal**: +5.9% perplexity, 90% NIAH recall, 96% battery test pass rate. No catastrophic failures on any reasoning, math, or code tasks.

5. **RotorQuant vs TurboQuant**: RotorQuant's fused CUDA kernels solve the throughput problem our Python TurboQuant implementation had (44ms/token → 67.5 tok/s). The block-diagonal rotation (372 params vs 16,384) is also 44x more parameter-efficient.

---

## 7. Deployment Configuration

```bash
# Start serving Qwen3.5-27B with RotorQuant 128K context
docker compose --profile qwen up

# Or directly:
docker run --gpus all \
  -e MODEL_NAME=qwen3.5-27b \
  -v llm-models:/models \
  -p 8080:8080 \
  rotorquant

# API endpoint:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":500}'
```

### Available Models

| Profile | Model | Size | Notes |
|---------|-------|------|-------|
| `--profile qwen` | Qwen3.5-27B Q4_K_M | 16.7 GB | Base model |
| `--profile reasoning` | Qwen3.5-27B Claude Opus Distilled | 16.6 GB | Reasoning-tuned |
| `--profile gemma` | Gemma 4 26B MoE UD-Q4_K_XL | 17.1 GB | 3.8B active params |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | (required) | Model to serve |
| `KV_CACHE_TYPE` | `iso3` | KV cache quantization (iso3, iso4, planar3, f16) |
| `CTX_SIZE` | `131072` | Context window size |
| `PORT` | `8080` | Server port |
| `GPU_LAYERS` | `99` | Layers offloaded to GPU |
| `HF_TOKEN` | — | HuggingFace token for gated models |
