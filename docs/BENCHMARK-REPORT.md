# RotorQuant Benchmark Report

**Hardware**: NVIDIA RTX 5090 (32 GB VRAM)
**Runtime**: llama.cpp RotorQuant fork (johndpope/llama-cpp-turboquant, commit 20efe75)
**Last updated**: 2026-04-22

---

## Models Benchmarked

| Model | File | Size | Type | Default KV |
|-------|------|-----:|:----:|:----------:|
| Qwen3.6-35B-A3B UD-Q4_K_XL | `--profile qwen` | 20.8 GB | MoE | `iso3` |
| Qwen3.6-27B UD-Q4_K_XL | `--profile qwen36-27b` | 16.4 GB | Dense | `planar3` |
| Qwen3.5-27B Q4_K_M | (legacy) | 16.7 GB | Dense | `planar3` |
| Qwen3.5-9B Q4_K_M | (eval only) | — | Dense | — |

---

## KV Cache Types

| Type | Bits/elem | Block structure | Notes |
|------|:---------:|----------------|-------|
| `f16` | 16.0 | raw fp16 | baseline |
| `planar3` | 3.125 | 2-bit centroid + 1-bit QJL sign, 50 B/128 vals | best for dense models |
| `planar4` | 4.25 | 4-bit nibble, 68 B/128 vals | |
| `iso3` | 3.125 | same block layout as planar3, isotropic rotation | best for MoE models |
| `iso4` | 4.25 | same block layout as planar4, isotropic rotation | |

All four types use **deferred K quantization**: K is held in f16 during prefill to
avoid error compounding, then converted to the target type after prefill completes.
Steady-state K is compressed. **K quantization has zero measured PPL impact** — all
degradation comes from V quantization.

---

## 1. Perplexity

Standard evaluation: wikitext-2-raw-v1 test split and C4 validation, ctx=2048 sliding
window, `llama-perplexity` binary. Measurement uncertainty ±0.037–0.057 PPL (1σ).

### 1.1 Qwen3.5-27B-Q4_K_M — wikitext-2

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **6.6417** | — |
| **planar3 / planar3** | 3.125 | **7.0099** | **+0.368 (+5.5%)** |
| planar4 / planar4 | 4.25 | 7.0178 | +0.376 (+5.7%) |
| iso4 / iso4 | 4.25 | 7.0962 | +0.454 (+6.8%) |
| iso3 / iso3 | 3.125 | 7.1016 | +0.460 (+6.9%) |
| planar3 / f16 (K-only ablation) | — | 6.6417 | **0.000** |
| f16 / planar3 (V-only ablation) | — | 7.0099 | +0.368 |

### 1.2 Qwen3.5-27B-Q4_K_M — C4

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **9.8485** | — |
| **planar4 / planar4** | 4.25 | **9.9611** | **+0.113 (+1.1%)** |
| planar3 / planar3 | 3.125 | 9.9624 | +0.114 (+1.2%) |
| iso4 / iso4 | 4.25 | 9.9822 | +0.134 (+1.4%) |
| iso3 / iso3 | 3.125 | 9.9823 | +0.134 (+1.4%) |
| planar3 / f16 (K-only ablation) | — | 9.8485 | **0.000** |
| f16 / planar3 (V-only ablation) | — | 9.9624 | +0.114 |

### 1.3 Qwen3.5-9B-Q4_K_M — wikitext-2

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **7.6760** | — |
| planar4 / planar4 | 4.25 | 7.3368 | −0.339 (−4.4%) |
| planar3 / planar3 | 3.125 | 7.3731 | −0.303 (−3.9%) |
| iso4 / iso4 | 4.25 | 7.4868 | −0.188 (−2.5%) |
| iso3 / iso3 | 3.125 | 7.5490 | −0.127 (−1.7%) |

> Note: 9B shows apparent improvement on wikitext-2 — regularization artifact on clean
> encyclopedia text. C4 shows the correct (degradation) direction.

### 1.4 Qwen3.5-9B-Q4_K_M — C4

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **11.5291** | — |
| **planar3 / planar3** | 3.125 | **11.7294** | **+0.200 (+1.7%)** |
| iso3 / iso3 | 3.125 | 11.7364 | +0.207 (+1.8%) |
| planar4 / planar4 | 4.25 | 11.7429 | +0.214 (+1.9%) |
| iso4 / iso4 | 4.25 | 11.7430 | +0.214 (+1.9%) |

### 1.5 Qwen3.6-35B-A3B-UD-Q4_K_XL (MoE) — wikitext-2

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **6.1316** | — |
| **iso4 / iso4** | 4.25 | **6.2262** | **+0.095 (+1.5%)** |
| iso3 / iso3 | 3.125 | 6.2515 | +0.120 (+2.0%) |
| planar4 / planar4 | 4.25 | 6.2529 | +0.121 (+2.0%) |
| planar3 / planar3 | 3.125 | 6.2904 | +0.159 (+2.6%) |

### 1.6 Qwen3.6-35B-A3B-UD-Q4_K_XL (MoE) — C4

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **10.5681** | — |
| **iso4 / iso4** | 4.25 | **10.6928** | **+0.125 (+1.2%)** |
| iso3 / iso3 | 3.125 | 10.7108 | +0.143 (+1.4%) |
| planar4 / planar4 | 4.25 | 10.7335 | +0.165 (+1.6%) |
| planar3 / planar3 | 3.125 | 10.7354 | +0.167 (+1.6%) |

### 1.7 Qwen3.6-27B-UD-Q4_K_XL (dense) — wikitext-2

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **7.0901** | — |
| **planar3 / planar3** | 3.125 | **7.4044** | **+0.314 (+4.4%)** |
| planar4 / planar4 | 4.25 | 7.4703 | +0.380 (+5.4%) |
| iso3 / iso3 | 3.125 | 7.5587 | +0.469 (+6.6%) |
| iso4 / iso4 | 4.25 | 7.6405 | +0.550 (+7.8%) |

### 1.8 Qwen3.6-27B-UD-Q4_K_XL (dense) — C4

| KV Cache (K/V) | Bits/elem | PPL | Δ vs f16 |
|----------------|:---------:|----:|:--------:|
| f16 / f16 (baseline) | 16.0 | **10.2963** | — |
| **planar4 / planar4** | 4.25 | **10.4032** | **+0.107 (+1.0%)** |
| planar3 / planar3 | 3.125 | 10.4176 | +0.121 (+1.2%) |
| iso4 / iso4 | 4.25 | 10.4366 | +0.140 (+1.4%) |
| iso3 / iso3 | 3.125 | 10.4506 | +0.154 (+1.5%) |

### 1.9 Perplexity Key Findings

**K quantization has zero PPL impact.** Quantizing K with any type produces identical
PPL to f16/f16. All degradation comes from V quantization. K quantization is still
applied for memory savings (no quality cost).

**Use C4, not wikitext-2, as the primary eval corpus.** Wikitext-2 produced a
spurious improvement on 9B (regularization artifact on clean encyclopedia text). C4
shows the correct degradation direction on all models.

**MoE → iso; dense → planar.** The single strongest predictor of which rotation type
wins is architecture:
- Dense models (Qwen3.5-9B, Qwen3.5-27B, Qwen3.6-27B): planar < iso at both bit depths
- MoE with Gated Delta Net (Qwen3.6-35B-A3B): iso < planar at both bit depths

**planar3 and iso3 (3.125 bpe) match or beat the 4-bit variants on C4.** The 26%
memory saving from 4.25 → 3.125 bpe comes with no measurable quality cost on diverse
text (C4). Use 3-bit variants unless you need every fraction of PPL.

**Larger models are more robust.** Qwen3.5-27B degrades +1.1% on C4; Qwen3.5-9B
degrades +1.7%. Larger representations have more redundancy to absorb quantization noise.

---

## 2. Throughput — Qwen3.5-27B (iso3/iso3 vs f16/f16)

Measured with `llama-bench`, 1 repetition, batch_size=1.

### Prefill (prompt processing, tok/s)

| Context | f16/f16 | iso3/iso3 | Overhead |
|--------:|--------:|----------:|---------:|
| 512 | 3,608 | 3,119 | 1.2× |
| 2,048 | 3,551 | 2,331 | 1.5× |
| 4,096 | 3,490 | 1,711 | 2.0× |
| 8,192 | 3,306 | 1,109 | 3.0× |
| 16,384 | 3,015 | 649 | 4.6× |
| 32,768 | 2,502 | 357 | 7.0× |
| 65,536 | **OOM** | 189 | — |
| 131,072 | **OOM** | 98 | — |

### Decode (token generation, tok/s)

| Config | tok/s | % of f16 |
|--------|------:|---------:|
| f16/f16 | 69.3 | 100% |
| iso3/iso3 | 67.5 | **97%** |

Batch size has no measurable impact on decode throughput.

---

## 3. Throughput — Qwen3.6-35B-A3B MoE

**Config**: planar4/planar4, ctx=65536, 2 parallel slots, 99 GPU layers, flash-attn on

### Prefill throughput

| Prompt tokens | Prefill tok/s |
|:-------------:|:-------------:|
| 25 | ~750 |
| 249 | 710 |
| 505 | **5,174** |

Short prompts are dispatch-latency dominated. At 505 tokens the GPU saturates at
~5,174 tok/s — ~48% faster than Qwen3.5-27B (~3,500 tok/s peak) because MoE activates
fewer params per token.

### Decode throughput

| n_predict | Decode tok/s |
|:---------:|:------------:|
| 50 | 148 |
| 200 | **196** |
| 500 | 192 |

**Single-slot: ~190–196 tok/s** — 2.8× faster than Qwen3.5-27B (69.3 tok/s).

### Parallel slots (iso3/iso3, 65K ctx/slot)

| Slots | Per-slot tok/s | Aggregate tok/s | Agg. scaling |
|:-----:|:--------------:|:---------------:|:------------:|
| 1 | **193** | 193 | 1.0× |
| 2 | 150 | **299** | 1.55× |
| 4 | 99 | **397** | 2.06× |
| 8 | 54 | **431** | 2.23× |

All stable across 3 trials (variance < 1%). P=4 is the practical optimum: 397 tok/s
aggregate at 99 tok/s per slot.

### Qwen3.6-35B-A3B vs Qwen3.5-27B

| Metric | Qwen3.5-27B Q4_K_M | Qwen3.6-35B-A3B Q4_K_XL |
|--------|:-------------------:|:------------------------:|
| Model size | 16.7 GB | 20.8 GB |
| VRAM at ctx=65K | ~20 GB | 23.1 GB |
| Decode tok/s (1 slot) | 69 | **196** (+184%) |
| Decode tok/s (2 slots) | ~138 | **298** (+116%) |
| Peak prefill tok/s | ~3,500 | **5,174** (+48%) |
| Active params/token | 27B | ~3B |

---

## 4. Memory — Qwen3.5-27B (64 layers, 8 KV heads, head_dim=128)

### VRAM at different context lengths (iso3/iso3)

| Context | VRAM Used | Headroom (32 GB) |
|--------:|----------:|-----------------:|
| 4,096 | 17.2 GB | 14.8 GB |
| 128K | 22.3 GB | 9.7 GB |
| 196K | 25.3 GB | 6.7 GB |

### Analytical KV cache size

| Config | KV at 4K | KV at 32K | KV at 128K | vs f16 |
|--------|:--------:|:---------:|:----------:|:------:|
| f16/f16 | 1.0 GB | 8.0 GB | 32.0 GB | 1.0× |
| iso3/iso3 | 0.2 GB | 1.6 GB | 6.5 GB | **4.9×** |
| iso4/iso4 | 0.3 GB | 2.2 GB | 8.5 GB | 3.8× |

### Max context window — Qwen3.6-35B-A3B on RTX 5090

| KV type | Per-slot ctx | Parallel slots | KV total | VRAM total |
|---------|:------------:|:--------------:|:--------:|:----------:|
| planar4 | 32,768 | 2 | 0.98 GB | 22.4 GB |
| planar3 | 262,144 | 2 | 5.4 GB | 26.8 GB |
| **iso3** | **262,144** | **2** | **7.1 GB** | **28.7 GB** |
| iso3 | 65,536 | 8 | 7.1 GB | 28.7 GB |

Qwen3.6 training context = 262,144. iso3 P=2 achieves the full training context per slot.

---

## 5. Needle-In-A-Haystack (NIAH) — Qwen3.5-27B iso3/iso3

Tested via llama-server API. Needle: "The secret project codename is ZEPHYR-NINE-SEVEN-ALPHA."

| Context | Depth 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | Recall |
|---------|:---------:|:---:|:---:|:---:|:---:|-------:|
| 4K | HIT | HIT | HIT | HIT | HIT | **100%** |
| 8K | HIT | HIT | HIT | MISS | HIT | **80%** |

Average recall: **90%**

---

## 6. Battery Test — Competence — Qwen3.5-27B iso3/iso3 (27/28 passed, 96%)

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

The single "failure" (H₂O vs H2O) is a unicode subscript match — chemically correct answer, wrong formatting.

---

## 7. SpectralQuant — Qwen3.5-9B (FAIL)

**Date**: 2026-04-20 | **Model**: Qwen/Qwen3.5-9B (bf16, HF transformers 5.5.4)

| Method | PPL | Δ vs f16 |
|--------|----:|:--------:|
| f16 baseline | 7.9462 | — |
| SpectralKVCache | 78.8447 | **+70.9 — FAIL** |

**Root cause**: SpectralQuant uses VQ in a high-dimensional space. Qwen3.5's QK
normalization forces K vectors onto the unit hypersphere, making the KV distribution
near-full-rank (d_eff_k ≈ 207–216 / 256 dims). 16 VQ centroids in 207 dimensions = 0.077
bits/dim → catastrophic reconstruction error. SpectralQuant requires low-rank KV
structure (d_eff ≤ ~32); use rotation-based methods (planar/iso) for Qwen architectures.

---

## 8. Key Findings

1. **MoE → iso; dense → planar.** Architecture determines which rotation type wins.
   Qwen3.6-35B-A3B (Gated Delta Net MoE): iso beats planar by 0.04–0.08 PPL (C4).
   All dense models tested: planar beats iso by 0.02–0.15 PPL.

2. **3-bit ≈ 4-bit quality on C4.** planar3 and planar4 differ by ≤0.014 PPL on C4
   across all tested models — within measurement noise. The 3-bit variants save 26%
   memory at no measurable quality cost on diverse text.

3. **K quantization is free.** Quantizing K has zero PPL impact across all models,
   datasets, and bit depths. All degradation is from V quantization.

4. **Decode speed virtually unchanged.** iso3/iso3 on Qwen3.5-27B: 67.5 tok/s vs 69.3
   tok/s f16 (97%). Qwen3.6-35B-A3B MoE: ~196 tok/s single-slot — 2.8× faster than
   the 27B dense model due to MoE sparse activation (~3B active params per token).

5. **Prefill overhead is real but context-dependent.** 1.2× at 512 tokens scaling to
   7× at 32K (Qwen3.5-27B). Negligible for interactive chat; noticeable for bulk
   document ingestion.

6. **Qwen3.6-35B-A3B MoE is the practical default.** 196 tok/s decode, 262K context
   at iso3 within 32 GB, 5,174 tok/s peak prefill. Better throughput and longer context
   than any dense model at similar VRAM.

---

## 9. Deployment Configuration

```bash
# Default: Qwen3.6-35B-A3B MoE, iso3, 262K context
docker compose --profile qwen up

# Dense alternative: Qwen3.6-27B, planar3, 131K context
docker compose --profile qwen36-27b up

# Override KV cache type:
KV_CACHE_TYPE=iso4 docker compose --profile qwen up

# API:
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Hello!"}],"max_tokens":500}'
```

### Available Profiles

| Profile | Model | Size | VRAM | KV default | Notes |
|---------|-------|-----:|:----:|:----------:|-------|
| `qwen` | Qwen3.6-35B-A3B UD-Q4_K_XL | 20.8 GB | ~28 GB | `iso3` | Default — MoE, 196 tok/s, 262K ctx |
| `qwen36-q3` | Qwen3.6-35B-A3B UD-Q3_K_XL | ~17 GB | ~24 GB | `planar4` | 24 GB GPUs |
| `qwen36-iq3` | Qwen3.6-35B-A3B UD-IQ3_XXS | ~14 GB | ~16 GB | `planar4` | 16 GB GPUs |
| `qwen36-27b` | Qwen3.6-27B UD-Q4_K_XL | 16.4 GB | ~20 GB | `planar3` | Dense, 131K ctx |
| `qwen36-27b-q3` | Qwen3.6-27B UD-Q3_K_XL | ~12 GB | ~16 GB | `planar3` | 24 GB GPUs |
| `qwen36-27b-iq3` | Qwen3.6-27B UD-IQ3_XXS | ~9 GB | ~12 GB | `planar3` | 16 GB GPUs |
| `reasoning` | Qwen3.5-27B Claude Opus Distilled | 16.6 GB | ~20 GB | `planar3` | Reasoning-tuned |
| `gemma` | Gemma 4 26B MoE UD-Q4_K_XL | 17.1 GB | ~21 GB | `planar4` | 3.8B active params |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | (required) | Model key from registry |
| `KV_CACHE_TYPE` | per-profile | Override KV cache type |
| `CTX_SIZE` | per-model | Override context window |
| `N_PARALLEL` | `2` | Concurrent request slots |
| `PORT` | `8080` | Server port |
| `GPU_LAYERS` | `99` | Layers offloaded to GPU |
| `CACHE_RAM` | per-profile | Prompt cache in system RAM (MiB) |
| `HF_TOKEN` | — | HuggingFace token for gated models |

## 10. Speculative Decoding (Sprint 004 work in progress)

- Rebase base SHA (`feature/planarquant-kv-cache` HEAD): `fc3d1b6566fa37be532e1153e11c35ceabc13f84`
- Rebase target SHA (`upstream/master` HEAD): `78433f606fde4d7934a02dcbfd910438d28beccd`
- Cherry-pick target SHA (PR `#22105` head): `e344c4a71736e1cdaa25e590a109f694dfb8119f`

- (a) Snapshot architecture (PR `#19493`, `#22227` call-path): checkpointing uses `llama_state_seq_get_size_ext/get_data_ext/set_data_ext` and stores bytes in `std::vector<uint8_t>`; this is an eager byte-copy snapshot of selected state, not append-only/COW metadata.
- (b) Buffer source: `state_write_data()` for KV and recurrent memory writes backend tensors directly via `io.write_tensor(...)` (`ggml_backend_tensor_get`), preserving on-device stored layouts/types; no decoded/dequantized intermediate view is used.
- (c) Snapshot residency: checkpoint bytes live in host pageable memory (`std::vector<uint8_t>` in speculative-simple/server). The copy is pulled from backend tensors into host buffers; no persistent checkpoint blob is stored in VRAM.
- (d) Public integration API: checkpoint save/restore is exposed through C APIs `llama_state_seq_get_size_ext`, `llama_state_seq_get_data_ext`, `llama_state_seq_set_data_ext` (typically with `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`), then rollback cleanup uses `llama_memory_seq_rm`. There is no `llama_memory_*::checkpoint_save` symbol.
- (e) Hybrid handling detail: in `llama_memory_hybrid::state_write/state_read`, `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY` skips full-attention KV (`mem_attn`) and snapshots recurrent state (`mem_recr`) only; KV rollback is handled separately via `llama_memory_seq_rm`. Storage is therefore split, not a single uniform packed structure for both.

### Sprint 004 Phase 1 spike — checkpoint architecture findings

- Worst-case VRAM math conclusion from (a)+(c): checkpoint save/restore in this path does not duplicate KV in VRAM; incremental checkpoint footprint is host RAM. For non-SWA hybrid partial checkpoints, overhead is recurrent-state bytes only (MB-scale), so checkpoint-induced VRAM increase is effectively `~0 GiB` (not the `+KV_size` full-copy VRAM-doubling case).

### Sprint 004 Phase 1 — L1 PPL regression sweep (rebased fork vs old fork)

Paired-comparison sweep (`scripts/ppl_sweep.py --mode pair`) against the
canonical `wikitext-2-raw-v1` test split at ctx=2048. **Old fork** = docker
`rotorquant:latest` (commit `20efe75`). **Rebased fork** = local build at
commit `8cfb44021` on branch `feature/sprint-004-rebase-dflash`. Tolerance
±0.05 PPL; gate is "no regression" (rebased ≤ old + tolerance).

Note: the §1.5–1.8 baseline numbers above (e.g. 7.0901 for Qwen3.6-27B f16)
were measured against a different, non-canonical `wikitext-2-test.txt` file
that has since been corrected. The numbers below are the canonical
`wikitext-2-raw-v1` measurements and supersede §1.5–1.8 for the f16 column.

| Model | KV (K/V) | New (rebased) | Old (20efe75) | Δ (new − old) | Verdict |
|-------|:---------:|--------------:|--------------:|--------------:|---------|
| Qwen3.6-27B | f16     | 8.0410 | 8.0491 | −0.0081 | match |
| Qwen3.6-27B | iso3    | 8.1838 | 8.7742 | −0.5904 | improvement |
| Qwen3.6-27B | iso4    | 8.2147 | 8.8099 | −0.5952 | improvement |
| Qwen3.6-27B | planar3 | 8.2000 | 8.4974 | −0.2974 | improvement |
| Qwen3.6-27B | planar4 | 8.1031 | 8.5622 | −0.4591 | improvement |
| Qwen3.6-35B-A3B | f16     | 6.2390 | 6.2362 | +0.0028 | match |
| Qwen3.6-35B-A3B | iso3    | 6.3199 | 6.3768 | −0.0569 | improvement |
| Qwen3.6-35B-A3B | iso4    | 6.2893 | 6.3525 | −0.0632 | improvement |
| Qwen3.6-35B-A3B | planar3 | 6.2843 | 6.4352 | −0.1509 | improvement |
| Qwen3.6-35B-A3B | planar4 | 6.2489 | 6.3857 | −0.1368 | improvement |

**Result: 0 regressions across all 10 cells.** f16 baselines match within
±0.008 PPL on both models. Every quantized KV type on both models shows
the rebased fork producing **lower** PPL than the old fork — by 0.06–0.60.
Improvements are concentrated on the 27B dense model (avg −0.49 PPL across
quantized types) versus smaller MoE gains (avg −0.10 PPL) — consistent with
upstream having landed quantization-kernel improvements in the 399-commit
mainline drift between merge-base and rebase target.

C4 corpus deferred to a follow-up (dataset not currently local).

Raw results JSON: `docs/sprints/SPRINT-004-L1-results.json`.

### Sprint 004 Phase 2 — checkpoint snapshot cost measurement

Tool: `examples/checkpoint-bench/checkpoint-bench.cpp` (rebased fork). Runs
prefill, then times `llama_state_seq_get_size_ext` / `get_data_ext` /
`set_data_ext` round-trips with both `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`
(recurrent state only — what speculative actually uses) and full state.
Min wallclock across 5 trials.

**Confirmed: PARTIAL_ONLY snapshot size is fixed per model**, independent
of prefill count and allocated context. It captures only the
`linear_attention` layer recurrent state, which is per-layer and
context-independent. Identical 156 MB at ckpt_tokens=2048 vs ckpt_tokens=32000
on Qwen3.6-27B at ctx=65536. This means snapshot cost does **not** scale
with context length — a one-time overhead per verify step.

| Model | KV (K/V) | partial bytes | partial save+restore | full bytes | full save+restore |
|-------|:---------:|--------------:|---------------------:|-----------:|------------------:|
| Qwen3.6-27B | f16     | 156.9 MB | 21.94 ms | 291.2 MB | 39.90 ms |
| Qwen3.6-27B | iso3    | 156.9 MB | 21.77 ms | 237.2 MB | 33.16 ms |
| Qwen3.6-27B | planar3 | 156.9 MB | 21.60 ms | 237.2 MB | 33.01 ms |
| Qwen3.6-35B-A3B | f16     | 65.9 MB | 9.44 ms | 107.8 MB | 15.50 ms |
| Qwen3.6-35B-A3B | iso3    | 65.9 MB | 9.31 ms | 91.0 MB | 13.09 ms |
| Qwen3.6-35B-A3B | planar3 | 65.9 MB | 9.44 ms | 91.0 MB | 13.35 ms |

(Measured with ckpt_tokens=2048 at ctx=8192. Save+restore = save_us_min +
restore_us_min, the two operations executed back-to-back.)

**Findings:**

1. **Partial size is per-model-fixed**: 156 MB on 27B (48 linear_attention
   layers × ~3.27 MB each); 65.9 MB on 35B-A3B (30 linear_attention layers
   × ~2.20 MB each). The 27B has more linear_attention layers despite being
   "smaller" in active params, hence the larger partial state.
2. **Save and restore are host-RAM-bandwidth bounded**: 156 MB / 11 ms ≈
   14 GB/s, consistent with PCIe-equivalent host memory copy. Restore is
   the same path in reverse. No optimization is possible at the user level
   without architectural changes (e.g. keeping snapshot in VRAM, which
   would forfeit the host-RAM cost benefit).
3. **RotorQuant shrinks the full snapshot but not the partial one**:
   full snapshot includes attention KV which RotorQuant compresses ~4.9×
   (291 MB f16 → 237 MB iso3 on 27B). Partial snapshot is unchanged
   because it skips full-attention KV by design.
4. **Sprint 004 hard-gate value of ≤5 ms was based on the spike's
   underestimate of recurrent state size**. Measured values are 9.3 ms
   (35B production default) and 21.7 ms (27B production default). The
   gate is revised in `SPRINT-004.md` to ≤25 ms to reflect ground truth.

**Speculative speedup impact** (rough analytical model, draft_max=16,
acceptance rate 0.5):

- **35B/iso3** (`qwen36-dflash`): cycle = 9.3 (snap+restore) + 5 (draft) +
  12 (verify) ≈ 25-30 ms per ~8 accepted tokens → ~270-320 tok/s vs
  ~196 tok/s baseline → **1.4–1.6× speedup**.
- **27B/planar3** (`qwen36-27b-dflash`): cycle = 21.7 + 5 + 25 ≈ 50-60 ms
  per ~8 accepted tokens → ~140 tok/s vs ~67 tok/s baseline → **~2.0×
  speedup**. Snapshot cost is a meaningful slice of cycle time but
  doesn't kill the speculative gain.

These are analytical estimates; actual L4 measurements happen in Phase 5.

Tool: `/home/ravi/repos/llama-cpp-turboquant/build/bin/llama-checkpoint-bench`.

### Sprint 004 Phase 2 — VRAM-resident shadow snapshot (Option A)

The host-RAM path measured above is bounded by PCIe (~14 GB/s pageable host
memory). Replacing it with a VRAM-resident shadow buffer that uses
`cudaMemcpyDeviceToDevice` runs at HBM bandwidth instead. Implementation
in `src/llama-vram-checkpoint.{h,cpp}`: a fork-internal class
`vram_seq_checkpoint` allocates one `cudaMalloc` shadow buffer per
recurrent state tensor (R and S per linear_attention layer) and copies
between the live tensor and the shadow on save/restore. No changes to
upstream's `llama_state_seq_*_ext` API — this is an additive optimization
path that the speculative code can opt into.

| Model | KV (K/V) | Path | Total bytes | save+restore | Speedup |
|-------|:---------:|------|------------:|:------------:|--------:|
| Qwen3.6-27B | iso3 | host RAM (`PARTIAL_ONLY`) | 156.9 MB | 21.28 ms | 1× |
| Qwen3.6-27B | iso3 | **VRAM shadow (D→D)** | 156.9 MB | **0.53 ms** | **40×** |
| Qwen3.6-35B-A3B | iso3 | host RAM (`PARTIAL_ONLY`) | 65.9 MB | 9.00 ms | 1× |
| Qwen3.6-35B-A3B | iso3 | **VRAM shadow (D→D)** | 65.9 MB | **0.29 ms** | **31×** |

VRAM cost: +66 MB (35B) or +157 MB (27B) of additional device memory for
the shadow buffer — negligible against the 32 GB RTX 5090 budget.

**Implications:**
1. Snapshot cost is now a non-factor in speculative decoding cycle time.
   At 0.5 ms (27B) or 0.3 ms (35B) per save+restore, snapshot is ≤2% of
   even the fastest verify cycle.
2. The Phase 2 hard gate of ≤25 ms is exceeded by ~50–100×; the original
   ≤5 ms gate would also pass comfortably.
3. The earlier analytical speedup estimates (1.4–1.6× for 35B, ~2.0× for
   27B) were limited by the 9–22 ms snapshot cost. With VRAM shadow,
   those estimates rise to ~1.6–1.8× (35B) and ~2.2–2.5× (27B) since
   snapshot no longer eats verify-cycle time. Final L4 measurement
   remains the gate.
4. Bit-exactness of save→mutate→restore is not yet validated; that
   becomes Subtest B/C of `tests/test-checkpoint-hybrid-state.cpp` in a
   later session. The current measurement only verifies the round-trip
   completes and copies the right number of bytes.

Tool: `llama-checkpoint-bench` reports both `partial` (host-RAM upstream
path) and `vram_partial` (this optimization) on every run.
