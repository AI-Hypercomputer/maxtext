# Qwen3.5 Reproduction & Setup Guide (TPU v5p + Pathways + Raiden FFI)

This document contains step-by-step instructions to reproduce distributed Reinforcement Learning (RL) fine-tuning runs with **Qwen3-0.6B** and **Qwen3.5-35B-A3B** on Google Cloud TPU v5p using:
- **Tunix** (Distributed orchestrator and rollout worker)
- **MaxText** (`MaxTextTrainingEngine` with direct parameter sharding and weight conversion)
- **Raiden FFI** (Zero-copy device-to-host tensor transfer across Pathways and vLLM)
- **GKE Pathways** (Shared cluster execution on `mlperf-v5p`)

---

## 1. Prerequisites & Cluster Access

Ensure you have authenticated to GCP and configured `kubectl` for the `mlperf-v5p` cluster:

```bash
# GCP project and cluster settings
export PROJECT=cloud-tpu-multipod-dev
export REGION=europe-west4
export ZONE=europe-west4-b
export CLUSTER=mlperf-v5p

# Set gcloud project and authenticate GKE
gcloud config set project ${PROJECT}
gcloud container clusters get-credentials ${CLUSTER} --zone ${ZONE} --project ${PROJECT}

# Verify cluster connection and nodes
kubectl get nodes
```

---

## 2. Git Repositories, Branches & Commits

To reproduce the exact environment, check out the following branches or cherry-pick the specified commits.

### A. MaxText
- **Repository**: `https://github.com/AI-Hypercomputer/maxtext.git`
- **Branch with all changes**: `igorts/qwen3.5-35b`
- **Review PR Branch (FFI engine changes only)**: `igorts/raiden-ffi` (commit `346a62144`)

If starting from `origin/main`, cherry-pick the following commits in order:

| Commit Hash | Author / Origin | Description |
|---|---|---|
| `71535dcfc` | Anisha Mazumder | Make three silent failures in the Raiden training path visible |
| `3db9d12b2` | Anisha Mazumder | Fix the Raiden synchronizer import path and empty-metrics checkpointing |
| `106e7efdf` | Yixuan Wang | Qwen3.5-35B-A3B direct weight conversion |
| `0cfca938d` | Yixuan Wang | Fix for MaxText to vLLM HF layout conversion |
| `75a6e8206` | Yixuan Wang | Fallback to standalone converter in `maxtext_vllm_rollout` |
| `65f9cdccb` | Yixuan Wang | Fallback for torchax converter |
| `03bb0e143` | Igor Tsvetkov | Stage all weight sync chunks before listener creation with fallback |
| `346a62144` | Igor Tsvetkov | feat(training_engine): support Raiden-FFI weight sync and adopt single synchronizer |

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git fetch origin igorts/qwen3.5-35b
git checkout igorts/qwen3.5-35b
```

### B. Tunix
- **Repository**: `https://github.com/google/tunix.git`
- **Branch with all changes**: `igorts/qwen3.5-35b`

This branch builds upon Lance Wang's Raiden FFI branch (`origin/lancewang/enable-raiden-ffi-20260831`, commit `93f4bbcb`) with 7 required fixes:

| Commit Hash | Description |
|---|---|
| `3e0a51f9` | `fix(weight_sync)`: fix `policy_version` tracking when `sync_request` is None |
| `aecf784c` | `WIP`: multi-rollout worker discovery and FFI compatibility fixes |
| `e52fcddc` | `clean`: remove obsolete `host_stage` and align with Lance's FFI design |
| `5d9ec73c` | `feat`: configure k8s launcher, MaxText vLLM adapter, and Pathways jobset for `mlperf-v5p` |
| `e0dba1e3` | `fix(k8s_launcher)`: set default 4-host trainer slice for `mlperf-v5p` and disable `hf_xet` |
| `1feb677d` | `fix(k8s_launcher)`: assign `ROLLOUT_MESH_TP=2` and `ROLLOUT_MESH_FSDP=2` unconditionally for `mlperf-v5p` |
| `bb89edf9` | `fix(k8s_launcher)`: assign batch size and `train_micro_batch_size` unconditionally for `mlperf-v5p` |

```bash
git clone https://github.com/google/tunix.git
cd tunix
git fetch origin igorts/qwen3.5-35b
git checkout igorts/qwen3.5-35b
```

### C. XPK (Pathways Workload Jobset Template)
- **Repository**: `https://github.com/igorts-git/xpk.git`
- **Branch**: `igorts/fix-headless-init-restart-policy` (commit `5a372dd`)
- **Fix**: Omit `restartPolicy: Always` for init containers in `src/xpk/templates/pathways_workload_create.yaml.j2` when running in headless mode, avoiding pod lifecycle stalls under Kubernetes 1.28+.

```bash
git clone https://github.com/igorts-git/xpk.git
cd xpk
git checkout igorts/fix-headless-init-restart-policy
pip install -e .
```

### D. TPU-Inference & vLLM
Pre-installed inside the Docker image:
- `vllm-project/tpu-inference`: `main` (clean)
- `vllm-project/vllm`: commit `2131b597b` (clean)
No local cherry-picks required.

---

## 3. Container Image

### Using the Pre-Built Image
A pre-built, tested container image with MaxText, Tunix, TPU-Inference, and Raiden FFI is available on GCP Artifact Registry:
```bash
export TUNIX_IMAGE="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-ffi-20260831"
```

### Building Your Own Image (Optional)
If you wish to build your own image from modified source trees:
1. In `tunix`, ensure `Dockerfile.maxtext` is used:
   ```dockerfile
   FROM gcr.io/cloud-tpu-multipod-dev/anisha-tmvp/anisha-0825:igorts-chunk4-v2
   ENV PATH="/opt/venv/bin:$PATH"

   # Install Raiden wheel with FFI support
   COPY .docker/tpu_sync/*.whl /tmp/tpu_sync/
   RUN pip install --force-reinstall --no-deps /tmp/tpu_sync/*.whl && rm -rf /tmp/tpu_sync

   # Install local MaxText
   COPY .docker/maxtext /maxtext
   RUN pip install --no-deps -e /maxtext

   # Install local Tunix
   WORKDIR /app
   COPY . /app
   RUN pip install --no-deps -e /app

   CMD ["bash"]
   ```
2. Download the Raiden FFI wheel:
   ```bash
   mkdir -p .docker/tpu_sync
   gcloud storage cp gs://cloud-tpu-inference-test-datenglin/tpu_raiden_jax-0.0.1+git9a9fb93-cp312-cp312-manylinux_2_31_x86_64.whl .docker/tpu_sync/
   ```
3. Copy MaxText into `.docker/maxtext`:
   ```bash
   cp -r ../maxtext .docker/maxtext
   ```
4. Build and push to your container registry:
   ```bash
   export MY_IMAGE="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/${USER}-maxtext:custom-$(date +%Y%m%d)"
   docker build -t ${MY_IMAGE} -f Dockerfile.maxtext .
   docker push ${MY_IMAGE}
   export TUNIX_IMAGE=${MY_IMAGE}
   ```

---

## 4. Model Checkpoints & Pre-trained Weights

Pre-converted MaxText checkpoints are hosted on Google Cloud Storage:

### Qwen3.5-35B-A3B
- **MaxText Checkpoint**: `gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/scanned/0/items`
- **Hugging Face Model / Tokenizer**: `Qwen/Qwen3.5-35B-A3B`
- **MaxText Model Config**: `qwen3.5-35b-a3b`

### Qwen3-0.6B
- **MaxText Checkpoint**: `gs://maxtext-model-checkpoints/qwen3-0.6b/scanned/0/items`
- **Hugging Face Model / Tokenizer**: `Qwen/Qwen3-0.6B`
- **MaxText Model Config**: `qwen3-0.6b`

---

## 5. Launch Commands

Navigate to the root of the `tunix` repository before launching.

### A. Run Qwen3.5-35B-A3B

```bash
cd tunix

MODEL_NAME="qwen3.5-35b-a3b" \
MODEL_ID="Qwen/Qwen3.5-35B-A3B" \
TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B" \
TRAINER_BACKEND=maxtext \
MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b" \
MAXTEXT_CKPT="gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/scanned/0/items" \
MAXTEXT_OUTPUT_DIR="gs://mohitkhatwani_multipods/pathways_scratch/$USER/maxtext" \
WEIGHT_SYNC_MODE=raiden \
TUNIX_IMAGE="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-ffi-20260831" \
./tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --target=mlperf-v5p --command=start
```

### B. Run Qwen3-0.6B

```bash
cd tunix

MODEL_NAME="Qwen3-0.6B" \
MODEL_ID="Qwen/Qwen3-0.6B" \
TOKENIZER_PATH="Qwen/Qwen3-0.6B" \
TRAINER_BACKEND=maxtext \
MAXTEXT_MODEL_NAME="qwen3-0.6b" \
MAXTEXT_CKPT="gs://maxtext-model-checkpoints/qwen3-0.6b/scanned/0/items" \
MAXTEXT_OUTPUT_DIR="gs://mohitkhatwani_multipods/pathways_scratch/$USER/maxtext" \
WEIGHT_SYNC_MODE=raiden \
TUNIX_IMAGE="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-ffi-20260831" \
./tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --target=mlperf-v5p --command=start
```

---

## 6. Cluster Topology & Settings

The `--target=mlperf-v5p` profile automatically configures the following topology on `mlperf-v5p`:

| Component | TPU Slice | Chip Count / Cores | Mesh Topology | Pod / Host Count |
|---|---|---|---|---|
| **Trainer** | `tpuv5:2x2x4` | 16 chips / 32 cores | `FSDP=8, TP=2` | 4 hosts |
| **Rollout** | `tpuv5:2x2x1` | 4 chips / 8 cores | `TP=2, FSDP=2` | 1 host |
| **Orchestrator** | CPU (`n2d-standard-128`) | - | - | 1 pod |

- **Weight Staging Transport**: Raiden FFI (`weight_synchronizer_ffi`)
  - Direct device memory access without host CPU staging, preventing client-side OOM and multi-minute proxy transfer timeouts.
- **Batching Parameters**:
  - `MINI_BATCH_SIZE=8`
  - `TRAIN_MICRO_BATCH_SIZE=8`
  - `BATCH_SIZE=2`
  - `NUM_GENERATIONS=4`

---

## 7. Monitoring & Troubleshooting

### Inspect Pods & Workloads
```bash
# List all pods for your user
kubectl get pods -l app=${USER}

# View orchestrator logs
kubectl logs -f -l job-name=${USER}-orchestrator

# View trainer logs
kubectl logs -f -l job-name=${USER}-trainer -c main

# View rollout worker logs
kubectl logs -f -l job-name=${USER}-rollout -c main

# Using xpk to inspect active workloads
xpk workload list --cluster=mlperf-v5p --zone=europe-west4-b --project=cloud-tpu-multipod-dev
```

### Stopping / Tearing Down Jobs
When finished or to clear a run before re-launching:
```bash
cd tunix
./tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --target=mlperf-v5p --command=stop
```
This terminates the orchestrator, trainer, rollout, and Pathways server workloads cleanly.
