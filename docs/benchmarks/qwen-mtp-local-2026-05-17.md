# Qwen3.6 27B MTP Local Probe - 2026-05-17

Host: Apple M1 Max, 64 GB unified memory, Metal backend.

Build:

- llama.cpp fork: `Indras-Mirror/llama.cpp-turboq-mtp`
- commit: `0bc0a27`
- build flags: `-DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release`

Model:

- repo: `unsloth/Qwen3.6-27B-MTP-GGUF`
- file: `Qwen3.6-27B-UD-Q4_K_XL.gguf`
- metadata: `general.architecture=qwen35`, `qwen35.nextn_predict_layers=1`
- MTP tensors observed: `blk.64.nextn.eh_proj.weight`, `blk.64.nextn.enorm.weight`, `blk.64.nextn.hnorm.weight`, `blk.64.nextn.shared_head_norm.weight`

Run shape:

```bash
llama-server \
  --model Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --n-gpu-layers 99 \
  --ctx-size 4096 \
  --parallel 1 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --flash-attn auto \
  --ubatch-size 32 \
  --spec-type mtp \
  --spec-draft-n-max 4 \
  --spec-draft-p-min 0.75 \
  --no-warmup
```

Result:

| Mode | Decode | Draft acceptance |
|---|---:|---:|
| MTP draft 2 | 9.90 tok/s | 37 / 52 (71.2%) |
| MTP draft 3 | 13.73 tok/s | 63 / 91 (69.2%) |
| MTP draft 4 | 13.74 tok/s | 46 / 67 (68.7%) |
| MTP draft 6 | 10.93 tok/s | 48 / 86 (55.8%) |
| `--spec-type none` | 11.13 tok/s | n/a |

The local Metal result does not predict RTX 4090 throughput, but it proves the
Unsloth GGUF and pinned fork can load the MTP head and accept draft tokens. On
the current pinned RotorQuant fork, draft 4 is the best local default from this
sweep. The homelab gate should still check acceptance and A/B throughput, not
just the startup banner.
