# RotorQuant Helm chart

Deploy the [`rapatel0/rq-models`](..) llama.cpp server in Kubernetes with
configurable replication and multi-GPU layout.

## Prerequisites

- A Kubernetes cluster with the **NVIDIA GPU Operator** (or device plugin)
  installed — pods declare GPUs via `nvidia.com/gpu`.
- A pre-built RotorQuant image pushed to a registry your cluster can pull
  from. Build:
  ```bash
  # default (V100 + all modern arches: A100, A10G, RTX 4090, H100, B100, RTX 5090)
  docker build -t <registry>/rotorquant:latest -f docker/Dockerfile .

  # V100-only — much faster compile when targeting a single architecture
  # (also pin CUDA 12.6 — CUDA 13.x dropped Volta/Turing)
  docker build \
    --build-arg CUDA_VERSION=12.6.3 --build-arg CUDA_ARCHES="70" \
    -t <registry>/rotorquant:v0-v100 -f docker/Dockerfile .
  ```
- Storage for the model cache (one of):
  - **`models.kind=nfs`**: a `ReadWriteMany` StorageClass (NFS / CephFS / etc.).
    All replicas read the same GGUF; a single download serves the whole
    deployment. Network read at startup. Default.
  - **`models.kind=local`**: a pre-existing PVC bound to a node-local PV.
    Requires `models.existingClaim` to point at the PVC and that you have
    pre-hydrated the GGUF onto the volume (sample Job below). Faster startup
    (local read), no NFS dependency, but pinned to the affined node(s).

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

### Model fits on one GPU → 1 GPU per replica wins

```
replicaCount: N      # one per concurrent user you want
gpusPerReplica: 1
```

Each replica owns one GPU, all replicas share the model file (via NFS or
local PV), the Service round-robins requests. **N concurrent users, no
NVLink comm overhead.**

Bench data on Tesla V100-SXM2-32GB with Qwen3.6-27B Q4 (16 GB), planar3 KV,
256 K context:

| Topology | Decode (single user) | Notes |
|---|---|---|
| **2 replicas × 1 GPU** | **30 t/s** | best per-user latency; 2 idle GPUs spare |
| 1 replica × 4 GPU (TP) | 21 t/s | 4-way all-reduce dominates |
| 2 replicas × 2 GPU (TP) | 25 t/s | 17% slower per-user vs 1 GPU |

Tensor parallelism on V100 NVLink **strictly loses** for any model that
fits on one card. Comm overhead exceeds compute savings.

### Model doesn't fit on one GPU → tensor-parallel

```
replicaCount: 1
gpusPerReplica: K            # K = ceil(modelGB / GPU_VRAM_GB)
splitMode: row               # tensor-parallel
tensorSplit: "1,1,..."       # equal split across K GPUs (optional)
```

For a 70B Q4 model (~40 GB) on V100s: K=2, splitMode=row.
Helpful only when forced by memory.

### Hybrid: M replicas × K GPUs each

```
replicaCount: M
gpusPerReplica: K            # M × K ≤ total GPUs on the node
splitMode: row
```

When you need both concurrency *and* multi-GPU per replica.

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
| `nParallel` | Concurrent slots per replica | `1` |
| `splitMode` | `layer` / `row` / `none` (multi-GPU only) | `""` |
| `tensorSplit` | e.g. `"1,1,1,1"` | `""` |
| `mainGpu` | Primary GPU ordinal | `""` |
| `models.kind` | `nfs` (auto-create RWX PVC) or `local` (use existingClaim) | `nfs` |
| `models.storageClass` | StorageClass when `kind=nfs` | `nfs-rwx` |
| `models.size` | Auto-PVC size when `kind=nfs` | `80Gi` |
| `models.existingClaim` | Required when `kind=local`; optional otherwise | `""` |
| `models.readOnly` | Mount `/models` read-only in the server pods | `true` |
| `models.hydrate.enabled` | Render a pre-install Job that `hf download`s into the PVC | `false` |
| `models.hydrate.hfRepo` / `hfFile` | HF repo + file to download (when hydrate.enabled) | `""` |
| `podSecurityContext` | Pod-level securityContext (defaults runAsUser=0 for hostPath PV compat) | `{runAsUser:0,runAsGroup:0,fsGroup:0}` |
| `dnsConfig.enabled` | Override pod DNS with explicit nameservers (for clusters with flaky CoreDNS upstream) | `false` |
| `ingress.host` | Hostname for the Ingress | `rotorquant.homelab.local` |
| `hfToken` | HF token (only needed for gated repos) | `""` |

## Local-PV pattern (recommended for single-GPU-node setups)

When you have one GPU node and the model is reproducible from a URL,
local storage beats NFS at startup latency and removes the network
dependency. The chart supports this end-to-end.

### One-time: create the local StorageClass + PV (cluster-scoped)

These are not in the chart (operator-owned, cluster-scoped). Apply once:

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata: {name: local-models}
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
reclaimPolicy: Retain
---
apiVersion: v1
kind: PersistentVolume
metadata: {name: models-<node>}
spec:
  capacity: {storage: 500Gi}
  accessModes: [ReadWriteOnce]
  persistentVolumeReclaimPolicy: Retain
  storageClassName: local-models
  local: {path: /srv/models}        # local NVMe/ZFS path on the GPU node
  nodeAffinity:
    required:
      nodeSelectorTerms:
        - matchExpressions:
            - {key: kubernetes.io/hostname, operator: In, values: [<node>]}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: llm-models-local, namespace: llm}
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: local-models
  resources: {requests: {storage: 500Gi}}
  volumeName: models-<node>
```

### Per-deploy: install with hydration enabled

The chart's pre-install hook Job will `hf download` the model into the PVC
before the deployment starts. The Job is idempotent — skips if the file is
already present, so re-running `helm upgrade` is safe.

```bash
helm install qwen ./k8s \
  --namespace llm \
  --set image.repository=<registry>/rotorquant \
  --set image.tag=v0-v100 \
  --set replicaCount=2 --set gpusPerReplica=1 --set nParallel=1 \
  --set contextSize=262144 --set kvCacheType=planar3 \
  --set modelName=qwen3.6-27b \
  --set models.kind=local \
  --set models.existingClaim=llm-models-local \
  --set models.hydrate.enabled=true \
  --set models.hydrate.hfRepo=unsloth/Qwen3.6-27B-GGUF \
  --set models.hydrate.hfFile=Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --set 'nodeSelector.kubernetes\.io/hostname=<node>' \
  --set 'models.hydrate.nodeSelector.kubernetes\.io/hostname=<node>' \
  --set ingress.host=qwen.example.com
```

Multiple replicas on the **same node** can share a single RWO local PVC.
The deployment mounts it read-only; only the hydration Job mounts read-write.

### NFS-only flow (default, simpler)

If you don't care about startup latency and want shared cache across nodes:

```bash
helm install rotorquant ./k8s \
  --namespace llm \
  --set image.repository=<registry>/rotorquant \
  --set image.tag=v0-v100 \
  --set replicaCount=2 --set modelName=qwen3.6-27b \
  --set ingress.host=qwen.example.com
# `models.kind=nfs` is the default; chart auto-creates an RWX PVC backed by
# `models.storageClass` (default `nfs-rwx`). First replica downloads, others
# wait on the lock, then all read shared.
```

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
