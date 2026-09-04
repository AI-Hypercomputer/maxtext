# Qwen3.5 Distributed Training Instructions: Qwen3.5-0.6B & Qwen3.5-35B

This guide documents all steps required to get to a runnable stage for distributed GRPO training with **Qwen3.5-0.6B** and **Qwen3.5-35B** on GKE TPU clusters (e.g. `bodaborg-v5p-nap` in `europe-west4-b`).

The system architecture consists of:
- **Orchestrator**: Tunix distributed runtime (`K8sExecutor`) driving GRPO sampling, reward scoring (GSM8K), and policy step coordination.
- **Trainer**: MaxText training engine running on TPU pods under the Pathways runtime.
- **Rollout Worker**: vLLM (`InprocessVllmSamplerAdapter`) serving on TPU pods under Pathways.
- **Weight Transfer**: Raiden FFI weight synchronization performing direct TPU-to-TPU host memory / DMA transfer between Trainer and Rollout.

---

## 1. Fast-Track: Running with Prebuilt Artifacts

If you want to run immediately without building Docker images or wheels, use our prebuilt container image and custom Pathways sidecars.

### 1.1 Prebuilt Images and Wheels

| Component | URI / Location | Notes |
| :--- | :--- | :--- |
| **Runner Container** | `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260903` | Contains JAX 0.11, MaxText, Tunix, TPU-Inference, and Raiden FFI. |
| **Pathways Server** | `us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_server:raiden_20260812` | Unsanitized server image supporting Raiden RDMA / DMA. |
| **Pathways Proxy** | `us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_proxy_server:raiden_20260812` | Unsanitized proxy server image. |
| **Raiden Wheel** | `gs://mohitkhatwani-logs/wheels/tpu_sync/tpu_raiden_jax-0.0.1.dev20260903185444-cp312-cp312-manylinux_2_31_x86_64.whl` | Baked into the container image. |

---

### 1.2 Cluster Access Setup

Ensure your local environment is authenticated to GKE:

```bash
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --zone europe-west4-b \
  --project cloud-tpu-shared-capacity

# Verify cluster connectivity
kubectl get nodes
```

---

### 1.3 Running Qwen3.5-0.6B

**Hardware Topology**:
- Trainer: 1x `v5p-32` (16 TPU chips, 4 host workers)
- Rollout: 1x `v5p-8` (4 TPU chips, 1 host worker)

**Execution Command**:
From the `tunix` repository root:

```bash
MODEL_NAME=qwen3-0.6b \
MODEL_ID=Qwen/Qwen3-0.6B \
MAXTEXT_MODEL_NAME=qwen3-0.6b \
TRAINER_BACKEND=maxtext \
MAXTEXT_CKPT=gs://mohitkhatwani_multipods/qwen3-0.6b/pathways-compat/0/items \
WEIGHT_SYNC_MODE=raiden \
VERIFY_WEIGHTS=true \
DEBUG=1 \
MAX_STEPS=2 \
TUNIX_IMAGE=europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260903 \
PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_server:raiden_20260812 \
PATHWAYS_PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_proxy_server:raiden_20260812 \
MAXTEXT_OUTPUT_DIR=/tmp/maxtext_output \
bash tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --target=bodaborg-v5p-nap --command=start
```

---

### 1.4 Running Qwen3.5-35B (MoE)

**Hardware Topology**:
- Trainer: 1x `v5p-128` (64 TPU chips) or 1x `v5p-64` (32 TPU chips)
- Rollout: 1x `v5p-32` (16 TPU chips) or 1x `v5p-64` (32 TPU chips)

**Key Configuration Differences for 35B**:
1. `MAXTEXT_MODEL_NAME=qwen3.5-35b-a3b` uses the MoE architecture defined in `maxtext/configs/models/qwen3.5-35b-a3b.yml`.
2. `TRAINER_MESH_TP=8` and `ROLLOUT_MESH_TP=8`: Sets tensor parallelism to 8 across the 256 routed experts and GDN linear attention layers.
3. `MAXTEXT_PADDED_MOE_MLP_DIM`: For TP=8 on Qwen MoE architectures, intermediate dimension must align to TPU matrix multiplication blocks (e.g., padded to 1024 or 512).
4. `use_standalone_converter`: Wires the Qwen3.5 standalone converter (merged in MaxText PR #5073) into rollout weight synchronization.

**Execution Command**:

```bash
MODEL_NAME=qwen3.5-35b \
MODEL_ID=Qwen/Qwen3.5-35B-A3B \
MAXTEXT_MODEL_NAME=qwen3.5-35b-a3b \
TRAINER_BACKEND=maxtext \
MAXTEXT_CKPT=gs://<YOUR_BUCKET>/qwen3.5-35b/pathways-compat/0/items \
TRAINER_MESH_TP=8 \
TRAINER_MESH_EXPERT=1 \
ROLLOUT_MESH_TP=8 \
MAXTEXT_PADDED_MOE_MLP_DIM=1024 \
WEIGHT_SYNC_MODE=raiden \
VERIFY_WEIGHTS=true \
DEBUG=1 \
MAX_STEPS=2 \
TUNIX_IMAGE=europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260903 \
PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_server:raiden_20260812 \
PATHWAYS_PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_proxy_server:raiden_20260812 \
MAXTEXT_OUTPUT_DIR=/tmp/maxtext_output \
bash tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --target=bodaborg-v5p-nap --command=start
```

---

### 1.5 Monitoring and Cleanup

```bash
# Check pod status
kubectl get pods | grep $(whoami)

# Stream orchestrator output
kubectl logs $(whoami)-orch-proc-0-0-* -f

# Check trainer logs
kubectl logs $(whoami)-train-proc-0-0-* -c main --tail=100

# Check rollout worker logs
kubectl logs $(whoami)-roll-proc-0-0-* -c main --tail=100

# Stop the run and clean up jobsets
bash tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --target=bodaborg-v5p-nap --command=stop
```

---

## 2. Detailed Instructions: Building from Scratch

When preparing custom images or reproducing the development environment, you must assemble the exact patches across the participating repositories.

### 2.1 Repository Setup & Upstream State

Clone the repositories into a shared workspace:

```bash
mkdir -p ~/git && cd ~/git
git clone https://github.com/google/tunix.git
git clone https://github.com/AI-Hypercomputer/maxtext.git
git clone https://github.com/vllm-project/tpu-inference.git
git clone https://github.com/google/tpu-sync.git
```

If you do not have write access to upstream repositories, add your personal GitHub fork remotes:
```bash
cd ~/git/tunix && git remote add fork git@github.com:<YOUR_USER>/tunix.git
cd ~/git/tpu-inference && git remote add fork git@github.com:<YOUR_USER>/tpu-inference.git
cd ~/git/maxtext && git remote add fork git@github.com:<YOUR_USER>/maxtext.git
```

> [!IMPORTANT]
> **Future-Proofing Rule**: Several required patches are in the process of being reviewed and merged upstream. Before cherry-picking, always check `origin/main` to see if the fix has already landed:
> ```bash
> git fetch origin main
> git log origin/main --grep="<keyword>"
> ```
> If the commit or an equivalent PR has merged, skip that cherry-pick and rebase cleanly on `origin/main`.

---

### 2.2 Patches Required Per Repository

> [!TIP]
> **Branch Naming Convention**: To avoid collisions when multiple engineers work on the same repositories, name your feature branches `${USER}/qwen35-run` (e.g. `igorts/qwen35-run`).

#### A. `vllm-project/tpu-inference`

Start from `origin/main`:
```bash
cd ~/git/tpu-inference
git checkout -b ${USER}/qwen35-run origin/main
```

Check and cherry-pick the following branches / commits:

1. **Raiden RL Weight-Sync Core Fixes** (Branch `origin/anisha/raiden-rl-weight-sync` or commit `fc0eda397`):
   - **Why needed**:
     - *Engine-first preload*: Loads `tpu_sync` native shared libraries before JAX initializes libtpu, preventing fatal protobuf descriptor registration collisions (`xla/pjrt/proto/execute_options.proto`).
     - *Auto H2D*: Sets `auto_h2d=True` on rollout destination so weights are installed as slices arrive rather than reading torn buffers.
     - *Checksums*: Implements `__grand_total__`, `__tensor_count__`, and `__element_count__` across all bound variables.
     - *Sampler kwargs*: Fixes None-valued kwargs overriding default top_k/temperature.
     - *Prompt token IDs*: Populates `prompt_token_ids` on `SamplingResponse`.
   - **How to apply**:
     ```bash
     git fetch origin anisha/raiden-rl-weight-sync
     git cherry-pick fc0eda397dcd1cb1bb0603a6d0603a3ccc87f0f7
     ```

2. **vLLM Interface & NVFP4 Compatibility** (Commit `0ec3d5d3c` on branch `igorts/qwen35-run`):
   - **Why needed**:
     - Gracefully catches `ImportError` if `nvfp4` quantization configs are absent in the local vLLM installation.
     - Adds compatibility shims for newer vLLM KV cache interface: `KVCacheTensor.layers` mapping to `shared_by`, and `AttentionSpec.num_states` mapping to `block_size`.
   - **How to apply**:
     ```bash
     git fetch https://github.com/igorts-git/tpu-inference.git igorts/qwen35-run
     git cherry-pick 0ec3d5d3c
     ```

---

#### B. `google/tunix`

Start from `origin/main`:
```bash
cd ~/git/tunix
git checkout -b ${USER}/qwen35-run origin/main
```

Check and cherry-pick the following commits (all available on branch `igorts/qwen35-run` at `https://github.com/google/tunix` and `git@github.com:igorts-git/tunix.git`):

1. **Commit `06b4c599`** (*Make MaxText-trainer Raiden weight-sync work end to end*):
   - Preloads Raiden before JAX initialization in distributed runtime.
   - Wires step-0 sync in `prepare_rollout_policy` so the model starts with trained weights.
   - Adds manifest and checksum validation logging.

2. **Commit `119d4962`** (*Move Raiden preload into `run_*_node.py`*):
   - Modularizes Raiden preloading through `tunix/experimental/weight_sync/raiden_preload.py`.

3. **Commit `7847df3a`** (*Configure bodaborg-v5p-nap launcher, Pathways rollout, and Dockerfile.maxtext*):
   - Adds `--target=bodaborg-v5p-nap` to `k8s_launcher.sh`.
   - Adds Pathways rollout jobset templates (`jobset.pathways.yaml`).
   - Adds `Dockerfile.maxtext` multi-package container build.

4. **Commit `69be4481`** (*Normalize base parameter prefix in Raiden sync*):
   - **Why needed**: `TunixMaxTextAdapter` creates parameters rooted under `['base']` (e.g. `['base']['decoder']...`) while vLLM and HuggingFace models use `['model']`. Normalizes names in `raiden_synchronizer.py` so the controller can pair trainer source slices with rollout destination slices.

5. **Commit `72f7ad67`** (*Allow synchronous submit within running loops and use absolute output path*):
   - **Why needed**:
     - `GrpcRemoteActorHandle.submit()` dispatches to a dedicated background loop via `run_coroutine_threadsafe()`; removes restrictive assertion that caused `RuntimeError` when called from async worker threads.
     - Normalizes `maxtext_output_directory` to an absolute path in `run_trainer_node.py` to prevent Orbax checkpoint path failures.

6. **Commit `d4349c9c`** (*Update launcher defaults and container bootstrap patches*):
   - **Why needed**: Sets `MAXTEXT_OUTPUT_DIR=/tmp/maxtext_output` by default and injects `AttentionSpec.num_states` compatibility shim into container rollout startup command.

**How to apply all in one rebase**:
```bash
git fetch https://github.com/igorts-git/tunix.git igorts/qwen35-run
git rebase FETCH_HEAD
```

---

#### C. `AI-Hypercomputer/maxtext`

Check `origin/main`:
```bash
cd ~/git/maxtext
git checkout -b ${USER}/qwen35-run origin/main
```

- **PR #5073** (`1066d2a24` / `9fa415f2a`):
  - **Status**: **MERGED into `origin/main`** on September 3, 2026.
  - **What it accomplishes**: Fixes the standalone torchax converter for current `tpu-inference`, aligns tensor orientations and expert layouts (`GMM_EP`), and wires `use_standalone_converter` directly into `MaxTextVllmRollout` and `update_params`.
  - **Verification**: Run `git log origin/main --grep="standalone torchax converter"` to confirm. If your branch is tracking current `origin/main`, no additional cherry-picks are required.

---

### 2.3 The Raiden Wheel (`tpu_raiden_jax`)

Raiden provides low-overhead, high-bandwidth TPU DMA transfer between Pathways and vLLM.

#### Using Existing Wheel
Download the prebuilt wheel from GCS:
```bash
gsutil cp gs://mohitkhatwani-logs/wheels/tpu_sync/tpu_raiden_jax-0.0.1.dev20260903185444-cp312-cp312-manylinux_2_31_x86_64.whl /tmp/
```

#### Building the Wheel from Scratch
In `google/tpu-sync`:
1. Prerequisites: Linux x86_64, Bazel 7.x, Python 3.12, C++17 compiler (`g++` or `clang`).
2. Run the Bazel build:
   ```bash
   cd ~/git/tpu-sync
   bazel build -c opt //ci/wheel:raiden_jax_wheel \
     --repo_env=WHEEL_VERSION_EXTRAS=.dev$(date +%Y%m%d%H%M%S)
   ```
3. The resulting `.whl` is emitted to:
   ```bash
   ls -lh bazel-bin/ci/wheel/tpu_raiden_jax-*.whl
   ```

---

### 2.4 Custom Pathways Images

Because Raiden performs direct memory operations via TPU FFI and RDMA listeners, you must use unsanitized Pathways server and proxy images that do not drop custom ports:

```bash
# Server Image:
us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_server:raiden_20260812

# Proxy Server Image:
us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_proxy_server:raiden_20260812
```

These are passed to `k8s_launcher.sh` via the environment variables:
```bash
export PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_server:raiden_20260812
export PATHWAYS_PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_proxy_server:raiden_20260812
```

---

### 2.5 Building the Runner Docker Image

The build uses `tunix/Dockerfile.maxtext`, which layers our wheels and checked-out repositories on top of the verified base image `gcr.io/cloud-tpu-multipod-dev/anisha-tmvp/anisha-0825:igorts-chunk4-v2`.

#### Step 1: Stage the Build Context in `tunix`

```bash
cd ~/git/tunix

# 1. Stage the Raiden wheel
mkdir -p .docker/tpu_sync
gsutil cp gs://mohitkhatwani-logs/wheels/tpu_sync/tpu_raiden_jax-0.0.1.dev20260903185444-cp312-cp312-manylinux_2_31_x86_64.whl .docker/tpu_sync/

# 2. Stage clean checkouts of tpu-inference and maxtext
mkdir -p .docker/tpu_inference .docker/maxtext
rsync -av --delete --exclude='.git' --exclude='venv' --exclude='.venv' ~/git/tpu-inference/ .docker/tpu_inference/
rsync -av --delete --exclude='.git' --exclude='venv' --exclude='.venv' ~/git/maxtext/ .docker/maxtext/
```

#### Step 2: Build the Image with Podman or Docker

```bash
REGISTRY="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext"
TAG="$(whoami)-maxtext:$(date +%Y%m%d)"
FULL_IMAGE="${REGISTRY}/${TAG}"

podman build -t "${FULL_IMAGE}" -f Dockerfile.maxtext .
```

#### Step 3: Push to Artifact Registry

Authenticate with an ephemeral Google Cloud access token and push:

```bash
gcloud auth print-access-token | podman login \
  --authfile=/tmp/podman_auth.json \
  -u oauth2accesstoken \
  --password-stdin europe-west4-docker.pkg.dev

podman push --authfile=/tmp/podman_auth.json "${FULL_IMAGE}"
rm -f /tmp/podman_auth.json
```

Now you can point `TUNIX_IMAGE=${FULL_IMAGE}` in your `k8s_launcher.sh` runs!

---

## 3. Verification & Troubleshooting Checklist

When running, verify the following milestones in the pod logs:

1. **Worker Registration**:
   In orchestrator logs:
   ```
   [Orchestrator] Discovered rollout service (igorts-roll-0) at ...:20001
   [Orchestrator] Discovered trainer service (igorts-train) at ...:20002
   [Orchestrator] All required workers are ready.
   ```

2. **Raiden FFI Transport Initialization**:
   In rollout worker logs:
   ```
   [RolloutNode] rollout bind prepared 310 arrays (proxy_runtime=True)
   [RolloutNode] rollout FFI destination transport ready: shards=['...:34909'] control=['...:45909']
   [RolloutNode] Raiden weight sync warmed up.
   ```

3. **Weight Transfer Schedule**:
   In orchestrator logs:
   ```
   [Orchestrator] Transfer wsync-v0-r0 (uuid=1): generated schedule for [trainer] -> [rollout] (310 variable(s), 7161096 expected blocks)
   [Orchestrator] Weight synchronization complete (policy_version=0).
   ```

4. **Checksum Parity**:
   With `VERIFY_WEIGHTS=true`, verify the trainer logs and rollout logs report checksums:
   - Trainer: `Source weights checksums: {..., '__tensor_count__': 310, '__element_count__': 596049920}`
   - Destination: `destination checksums: {..., '__tensor_count__': 310, '__element_count__': 596049920}`
