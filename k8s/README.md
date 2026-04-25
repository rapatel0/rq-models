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
| `models.kind` | `nfs` (auto-create RWX PVC) or `local` (use existingClaim) | `nfs` |
| `models.storageClass` | StorageClass when `kind=nfs` | `nfs-rwx` |
| `models.size` | Auto-PVC size when `kind=nfs` | `80Gi` |
| `models.existingClaim` | Required when `kind=local`; optional otherwise | `""` |
| `ingress.host` | Hostname for the Ingress | `rotorquant.homelab.local` |
| `hfToken` | HF token (only needed for gated repos) | `""` |

## Hydrating a local PVC

When `models.kind=local`, you create the PVC + populate it yourself before
installing the chart. Typical flow on a single GPU node:

1. **Create a `local` StorageClass + PV pinned to the node** (one-time):
   ```yaml
   apiVersion: storage.k8s.io/v1
   kind: StorageClass
   metadata:
     name: local-models
   provisioner: kubernetes.io/no-provisioner
   volumeBindingMode: WaitForFirstConsumer
   reclaimPolicy: Retain
   ---
   apiVersion: v1
   kind: PersistentVolume
   metadata:
     name: models-<node>
   spec:
     capacity: {storage: 500Gi}
     accessModes: [ReadWriteOnce]
     persistentVolumeReclaimPolicy: Retain
     storageClassName: local-models
     local: {path: /srv/models}      # or wherever you have local SSD/ZFS
     nodeAffinity:
       required:
         nodeSelectorTerms:
           - matchExpressions:
               - {key: kubernetes.io/hostname, operator: In, values: [<node>]}
   ```

2. **Bind a PVC** in the target namespace:
   ```yaml
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: llm-models-local
     namespace: llm
   spec:
     accessModes: [ReadWriteOnce]
     storageClassName: local-models
     resources: {requests: {storage: 500Gi}}
     volumeName: models-<node>
   ```

3. **Hydrate** with a one-shot Job (re-runnable for adding model variants):
   ```yaml
   apiVersion: batch/v1
   kind: Job
   metadata: {name: hydrate-qwen, namespace: llm}
   spec:
     template:
       spec:
         restartPolicy: Never
         nodeSelector: {kubernetes.io/hostname: <node>}
         containers:
           - name: hf
             image: <registry>/rotorquant:<tag>     # ships with `hf` CLI
             command: ["bash","-c","hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-UD-Q4_K_XL.gguf --local-dir /models"]
             volumeMounts: [{name: models, mountPath: /models}]
         volumes:
           - name: models
             persistentVolumeClaim: {claimName: llm-models-local}
   ```

4. **Install with `kind=local`**:
   ```bash
   helm install rotorquant ./k8s \
     --namespace llm \
     --set models.kind=local \
     --set models.existingClaim=llm-models-local \
     --set image.repository=... --set image.tag=...
   ```

Multiple replicas on the **same node** can share a single RWO local PVC
without trouble.

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
