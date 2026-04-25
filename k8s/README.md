# RotorQuant Helm chart

Deploy the [`rapatel0/rq-models`](..) llama.cpp server in Kubernetes with
configurable replication and multi-GPU layout.

## Prerequisites

- A Kubernetes cluster with the **NVIDIA GPU Operator** (or device plugin)
  installed — pods declare GPUs via `nvidia.com/gpu`.
- A pre-built RotorQuant image pushed to a registry your cluster can pull
  from. Build:
  ```bash
  # default (all modern arches: V100, A100, A10G, RTX 4090, H100, B100, RTX 5090)
  docker build -t <registry>/rotorquant:latest -f docker/Dockerfile .

  # V100-only — much faster compile when targeting a single architecture
  docker build --build-arg CUDA_ARCHES="70" \
    -t <registry>/rotorquant:v0-v100 -f docker/Dockerfile .
  ```
- A `ReadWriteMany` StorageClass (NFS / CephFS / etc.) for the shared model
  cache. Each replica reads the same GGUF files; a single download serves
  the whole deployment.

## Quick start

```bash
# 1. Pick a deployment shape (see "Topologies" below). Example: 4× throughput
#    on a 4-GPU node, Qwen3.6-27B dense, RotorQuant planar3 KV.
helm install rotorquant ./k8s \
  --namespace llm --create-namespace \
  --set image.repository=registry.example/rotorquant \
  --set image.tag=v0-v100 \
  --set replicaCount=4 \
  --set gpusPerReplica=1 \
  --set modelName=qwen3.6-27b \
  --set kvCacheType=planar3 \
  --set ingress.host=qwen.example.com

# 2. Wait for first replica's model download (~5 min for a 16 GB GGUF).
#    Subsequent replicas reuse the cached file.
kubectl -n llm logs -f deployment/rotorquant

# 3. Test
curl https://qwen.example.com/v1/models
```

## Topologies

The two knobs `replicaCount` and `gpusPerReplica` cover the practical layouts.
Pick based on whether your model fits on one GPU.

### Model fits on one GPU → maximize throughput

```
replicaCount: N      # one per GPU on the node
gpusPerReplica: 1
```

Each replica owns one GPU, all replicas share the model file via the RWX PVC,
the Service round-robins requests across them. **N concurrent requests, same
per-request latency as single-replica.** Best for interactive multi-session.

### Model spans multiple GPUs → tensor-parallel a single replica

```
replicaCount: 1
gpusPerReplica: K            # K = how many GPUs the model needs
splitMode: row               # tensor-parallel; "layer" = pipeline-parallel
tensorSplit: "1,1,..."       # equal split across K GPUs (optional)
```

Single replica owns K GPUs, llama.cpp splits the weight tensors across them.
**One concurrent request, lower per-token latency** (NVLink-class
interconnect helps; PCIe-only setups may slow down vs. fitting on fewer GPUs).

### Hybrid: M replicas × K GPUs each

```
replicaCount: M
gpusPerReplica: K            # M × K ≤ total GPUs on the node
splitMode: layer | row
```

Useful when the model needs more than one GPU but you still want concurrency.

## All values

See [`values.yaml`](values.yaml) for the full set with comments. Key ones:

| Value | Meaning | Default |
|---|---|---|
| `image.repository` / `image.tag` | Image to deploy | `rotorquant:latest` |
| `replicaCount` | Number of pods | `1` |
| `gpusPerReplica` | GPUs per pod (`nvidia.com/gpu` limit) | `1` |
| `modelName` | Key from entrypoint MODELS table | `qwen3.6-27b` |
| `kvCacheType` | RotorQuant KV type (`iso3`, `planar3`, `planar4`, `f16`) | `planar3` |
| `contextSize` | Override per-model default context (tokens) | `""` |
| `nParallel` | Concurrent slots per replica | `2` |
| `splitMode` | `layer` / `row` / `none` (multi-GPU only) | `""` |
| `tensorSplit` | e.g. `"1,1,1,1"` | `""` |
| `mainGpu` | Primary GPU ordinal | `""` |
| `models.storageClass` | RWX storage class for shared model cache | `nfs-rwx` |
| `models.size` | Model cache PVC size | `80Gi` |
| `ingress.host` | Hostname for the Ingress | `rotorquant.homelab.local` |
| `hfToken` | HF token (only needed for gated repos) | `""` |

## OpenAI-compatible API

The server speaks the OpenAI Chat Completions API at the Service's port.
Example via the Ingress:

```bash
curl https://${ingress.host}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

`/v1/models`, `/v1/embeddings` (if compiled), `/health`, and `/metrics`
(Prometheus) are also exposed.

## Observability

Add a `ServiceMonitor` (kube-prometheus-stack) to scrape per-replica
llama.cpp metrics from `/metrics` on the Service. NVIDIA DCGM metrics come
from the GPU Operator's `nvidia-dcgm-exporter`, separate from this chart.
