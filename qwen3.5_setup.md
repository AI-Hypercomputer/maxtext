# Qwen3.5 Reproduction & Setup Guide (TPU v5p + Pathways + Raiden FFI)

This document contains step-by-step instructions to reproduce distributed Reinforcement Learning (RL) fine-tuning runs with **Qwen3-0.6B** and **Qwen3.5-35B-A3B** on Google Cloud TPU v5p using:
- **Tunix** (Distributed orchestrator and rollout worker)
- **MaxText** (`MaxTextTrainingEngine` with direct parameter sharding and weight conversion)
- **Raiden FFI** (Zero-copy device-to-host tensor transfer across Pathways and vLLM)
- **GKE Pathways** (Shared cluster execution on `mlperf-v5p`)

By following these instructions, you will cherry-pick the required changes on top of `HEAD` (`origin/main`), build your own custom container image, and launch the distributed workloads. As in-flight PRs are merged upstream, you can progressively drop the cherry-picks and rely purely on `origin/main`.

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

## 2. Git Repositories & Cherry-Pick Instructions

We will set up local branches on top of `origin/main` (`HEAD`) and cherry-pick the necessary commits that are currently under review in upstream Pull Requests.

### A. MaxText (`AI-Hypercomputer/maxtext`)

1. Clone or navigate to your MaxText repository:
   ```bash
   git clone https://github.com/AI-Hypercomputer/maxtext.git
   cd maxtext
   git fetch origin
   git checkout -b qwen35-run origin/main
   ```

2. **Upstream Status**:
   - **Already Merged**: Support for the Qwen3.5-35B model definition and vLLM weight conversion is already merged into `origin/main` via [PR #5045](https://github.com/AI-Hypercomputer/maxtext/pull/5045) (commit `4521fc568`).

3. **Commits to Cherry-Pick**:
   Fetch the PR branches and cherry-pick the Raiden-FFI engine integration and checkpointing fixes:
   ```bash
   # Fetch remote branches containing the pending PR commits
   git fetch origin igorts/raiden-ffi
   git fetch origin anisha/raiden-import-and-metrics-fix

   # 1. Raiden-FFI weight synchronization in MaxTextTrainingEngine (eliminates proxy staging timeouts & client OOM)
   git cherry-pick 346a62144

   # 2. Checkpoint guard for empty accumulated_metrics PyTree
   git cherry-pick 3db9d12b2
   ```

*(Note: Once [PR #5018](https://github.com/AI-Hypercomputer/maxtext/pull/5018) and the Raiden-FFI PR are merged, you will be able to run directly from `origin/main` without any cherry-picks).*

---

### B. Tunix (`google/tunix`)

1. Clone or navigate to your Tunix repository:
   ```bash
   cd ..
   git clone https://github.com/google/tunix.git
   cd tunix
   git fetch origin
   ```

2. **Upstream Status**:
   - **Already Merged**: Rollout policy bootstrapping from target state is already merged into `origin/main` via [PR #2054](https://github.com/google/tunix/pull/2054).
   - **In Review**: Core Raiden FFI integration is currently under review in [PR #2059](https://github.com/google/tunix/pull/2059) (`origin/lancewang/enable-raiden-ffi-20260831`).

3. **Branch Setup & Cherry-Picks**:
   Until PR #2059 merges into `origin/main`, branch from Lance's PR base or checkout the combined working branch:
   ```bash
   # Option 1 (Recommended): Fetch and checkout the verified working branch directly
   git fetch origin igorts/qwen3.5-35b
   git checkout -b qwen35-run origin/igorts/qwen3.5-35b
   ```

   If you prefer to cherry-pick individual commits onto Lance's PR base (`origin/lancewang/enable-raiden-ffi-20260831`):
   ```bash
   git fetch origin lancewang/enable-raiden-ffi-20260831
   git checkout -b qwen35-run origin/lancewang/enable-raiden-ffi-20260831

   # Cherry-pick the 7 launcher & multi-worker fixes from branch igorts/qwen3.5-35b:
   git fetch origin igorts/qwen3.5-35b
   git cherry-pick 3e0a51f9  # fix(weight_sync): policy_version tracking when sync_request is None
   git cherry-pick aecf784c  # WIP: multi-rollout worker discovery and FFI compat fixes
   git cherry-pick e52fcddc  # clean: remove obsolete host_stage
   git cherry-pick 5d9ec73c  # feat: configure k8s launcher & Pathways jobset for mlperf-v5p
   git cherry-pick e0dba1e3  # fix(k8s_launcher): set default 4-host trainer slice for mlperf-v5p and disable hf_xet
   git cherry-pick 1feb677d  # fix(k8s_launcher): assign ROLLOUT_MESH_TP=2 and ROLLOUT_MESH_FSDP=2 for mlperf-v5p
   git cherry-pick bb89edf9  # fix(k8s_launcher): assign batch size and train_micro_batch_size for mlperf-v5p
   ```

*(Note: As [PR #2059](https://github.com/google/tunix/pull/2059) and the launcher changes are merged into `origin/main`, you will simply branch from `origin/main`).*

---

### C. TPU-Inference & vLLM

No separate branch or cherry-picking is required. The pinned versions are installed inside the base container:
- `vllm-project/tpu-inference`: `main`
- `vllm-project/vllm`: commit `2131b597b`

---

## 3. Building Your Container Image (Primary Workflow)

Because you have customized local checkouts with the required PRs and cherry-picks, you should build and push your own Docker image so the GKE pods execute your exact code.

### Build Steps:

1. **Stage the Raiden FFI Wheel**:
   From the root of your `tunix` repository:
   ```bash
   mkdir -p .docker/tpu_sync
   gcloud storage cp gs://cloud-tpu-inference-test-datenglin/tpu_raiden_jax-0.0.1+git9a9fb93-cp312-cp312-manylinux_2_31_x86_64.whl .docker/tpu_sync/
   ```

2. **Stage your local MaxText checkout**:
   Copy your local `maxtext` checkout (containing the cherry-picked changes) into `.docker/maxtext`:
   ```bash
   # Assuming maxtext is located at ../maxtext relative to tunix
   rm -rf .docker/maxtext
   cp -r ../maxtext .docker/maxtext
   ```

3. **Build and Push the Image**:
   Configure your image name in the shared GCP Artifact Registry and build using `Dockerfile.maxtext`:
   ```bash
   export TUNIX_IMAGE="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/${USER}-maxtext:qwen35-$(date +%Y%m%d)"

   docker build -t ${TUNIX_IMAGE} -f Dockerfile.maxtext .
   docker push ${TUNIX_IMAGE}
   ```

> [!NOTE]
> If you encounter Docker build permissions or network constraints on your machine, a verified pre-built fallback image is available at:
> `TUNIX_IMAGE="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-ffi-20260831"`

---

## 4. Pre-trained Weights & Checkpoint Paths

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

Always execute the launcher from the root of your `tunix` repository. The launch script automatically references your custom image via the `TUNIX_IMAGE` environment variable.

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
TUNIX_IMAGE="${TUNIX_IMAGE}" \
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
TUNIX_IMAGE="${TUNIX_IMAGE}" \
./tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --target=mlperf-v5p --command=start
```

---

## 6. Cluster Topology & Settings

The `--target=mlperf-v5p` profile in `k8s_launcher.sh` configures the following topology on `mlperf-v5p`:

| Component | TPU Slice | Chip Count / Cores | Mesh Topology | Pod / Host Count |
|---|---|---|---|---|
| **Trainer** | `tpuv5:2x2x4` | 16 chips / 32 cores | `FSDP=8, TP=2` | 4 hosts |
| **Rollout** | `tpuv5:2x2x1` | 4 chips / 8 cores | `TP=2, FSDP=2` | 1 host |
| **Orchestrator** | CPU (`n2d-standard-128`) | - | - | 1 pod |

- **Weight Staging Transport**: Raiden FFI (`weight_synchronizer_ffi`)
  - Binds directly to device memory without host CPU staging, preventing client-side OOM and eliminating multi-minute proxy transfer timeouts.
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

# Inspect active workloads with xpk
xpk workload list --cluster=mlperf-v5p --zone=europe-west4-b --project=cloud-tpu-multipod-dev
```

### Stopping / Tearing Down Jobs
When finished or to clear a run before re-launching:
```bash
cd tunix
./tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --target=mlperf-v5p --command=stop
```
This terminates the orchestrator, trainer, rollout, and Pathways server workloads cleanly.
