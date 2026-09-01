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

## 2. Git Repositories & Working Branches

The end-to-end RL training stack spans two primary repositories: **MaxText** (training engine) and **Tunix** (distributed orchestrator and rollout worker).

### Recommended Setup: Use the Verified Working Branches

The quickest and most reliable way to reproduce these runs is to check out the verified working branches directly. These branches contain the full working integration including features that are currently under review in upstream pull requests:

```bash
# 1. Clone MaxText and checkout the working branch
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
git fetch origin igorts/qwen3.5-35b
git checkout igorts/qwen3.5-35b
cd ..

# 2. Clone Tunix and checkout the working branch
git clone https://github.com/google/tunix.git
cd tunix
git fetch origin igorts/qwen3.5-35b
git checkout igorts/qwen3.5-35b
cd ..
```

---

### Upstream PR Status & Technical Context

If you are maintaining a custom branch or rebasing onto `origin/main`, the table below outlines what each repository branch contains and the status of upstream Pull Requests:

#### A. MaxText (`AI-Hypercomputer/maxtext`)
The `igorts/qwen3.5-35b` branch incorporates the following components:
- **Qwen3.5-35B-A3B Model & Weight Converter**: Support for loading and converting weights between MaxText and vLLM layouts. *(Already merged into `main` via [PR #5045](https://github.com/AI-Hypercomputer/maxtext/pull/5045))*.
- **Raiden FFI Engine Integration**: High-performance device-to-host tensor transfer without host CPU proxy staging under Pathways. Prevents client-host OOM and eliminates multi-minute transfer timeouts. *(In review on PR branch [`igorts/raiden-ffi`](https://github.com/AI-Hypercomputer/maxtext/tree/igorts/raiden-ffi))*.
- **Raiden Error Visibility & Checkpoint Guard**: Surfaces silent failures during weight staging and handles empty metrics PyTrees cleanly during checkpointing. *(In review in [PR #5018](https://github.com/AI-Hypercomputer/maxtext/pull/5018))*.
- **Weight Staging Listener Synchronization**: Ensures all tensor chunks are staged prior to listener registration.

#### B. Tunix (`google/tunix`)
The `igorts/qwen3.5-35b` branch incorporates the following components:
- **Rollout Target State Bootstrapping**: Initializes rollout policy state directly from target model parameters. *(Already merged into `main` via [PR #2054](https://github.com/google/tunix/pull/2054))*.
- **Raiden FFI Pathways Weight Sync**: Implements the rollout delegate for zero-copy weight receipt via Raiden FFI. *(In review in [PR #2059](https://github.com/google/tunix/pull/2059))*.
- **`mlperf-v5p` Cluster Target Profile**: Adds the pre-configured `--target=mlperf-v5p` profile to `k8s_launcher.sh` (4-host TPU v5p trainer, 1-host TPU v5p rollout, mesh topologies `FSDP=8, TP=2` and `TP=2, FSDP=2`).
- **Multi-Worker Discovery & Network Stability**: Fixes `policy_version` tracking when sync requests are pending, improves multi-rollout worker discovery, and sets `HF_HUB_DISABLE_XET=1` to prevent Hugging Face download hangs in GKE containers.

#### C. TPU-Inference & vLLM
Pre-installed inside the Docker image; no separate branch or cherry-picking needed:
- `vllm-project/tpu-inference`: `main`
- `vllm-project/vllm`: commit `2131b597b`

#### D. XPK (Pathways Workload Template - Optional)
If you launch workloads via `xpk workload create --headless` directly instead of `k8s_launcher.sh`, note that Kubernetes 1.28+ clusters require omitting `restartPolicy: Always` on headless init containers to prevent JobSet completion stalls:
- Repository: `https://github.com/igorts-git/xpk.git`
- Branch: `igorts/fix-headless-init-restart-policy` (commit `5a372dd`)
*(Note: Not required if launching via `k8s_launcher.sh`)*.

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
