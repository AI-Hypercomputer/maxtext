# Qwen3.5 Distributed Training Instructions: Qwen3.5-0.6B & Qwen3.5-35B

This guide documents all steps required to execute distributed GRPO training with **Qwen3.5-0.6B** and **Qwen3.5-35B** on GKE TPU clusters (e.g. `bodaborg-v5p-nap` in `europe-west4-b`).

The system architecture consists of:
- **Orchestrator**: Tunix distributed runtime (`K8sExecutor` / `ClusterOrchestrator`) driving GRPO sampling, reward scoring (GSM8K), and policy step coordination.
- **Trainer**: MaxText training engine running on TPU pods under the Pathways runtime.
- **Rollout Worker**: vLLM using `tpu-inference`'s `RLVllmSampler` (`VllmSamplerAdapter` via `SAMPLER=vllm`) serving on TPU pods.
- **Weight Transfer**: Raiden FFI weight synchronization performing direct TPU-to-TPU host memory DMA transfer between Trainer and Rollout.

---

## 1. Fast-Track: Running with Verified Prebuilt Artifacts

If you want to run immediately without building container images or wheels from scratch, use our verified container image and unified CLI launcher.

### 1.1 Verified Images and Wheels

| Component | URI / Location | Notes |
| :--- | :--- | :--- |
| **Runner Container** | `gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-raiden-debug-0903-2` | Verified working image for Qwen3-0.6B E2E GRPO. Contains JAX, MaxText, Tunix, TPU-Inference, and Raiden FFI. |
| **Pathways Server** | `us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_server:raiden_20260812` | Unsanitized server image supporting Raiden RDMA / DMA. |
| **Pathways Proxy** | `us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_proxy_server:raiden_20260812` | Unsanitized proxy server image. |
| **Raiden Wheel** | `gs://mohitkhatwani-logs/wheels/tpu_sync/tpu_raiden_jax-0.0.1.dev20260903185444-cp312-cp312-manylinux_2_31_x86_64.whl` | Baked into the container image. |

---

### 1.2 Cluster Authentication

Ensure your local shell is authenticated to GKE:

```bash
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --zone europe-west4-b \
  --project cloud-tpu-shared-capacity

# Verify cluster connectivity and active TPU nodes
kubectl get nodes -l cloud.google.com/gke-nodepool
```

---

### 1.3 Running with the Unified Launcher (`launch_raiden.sh`)

We provide a unified CLI launcher located at `tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh` (or `/tmp/launch_raiden.sh`). It manages JobSet lifecycles, nodepool affinity, log streaming, and status monitoring.

#### A. Running Verified Qwen3-0.6B Baseline (1 Rollout Replica)

**Hardware Allocation**:
- **Trainer**: 1x `v5p-16` (`tpuv5:2x2x2`, 8 chips across 2 host nodes, 4 chips/node)
- **Rollout**: 1x `v5p-8` (`tpuv5:2x2x1`, 4 chips on 1 host node, TP=2)
- **Total**: 12 TPU chips (fits within reservation `cloudtpu-20260902214500-1810493672`)

```bash
# Launch the workload
/tmp/launch_raiden.sh start \
  --model qwen3-0.6b \
  --rollout-replicas=1 \
  --image gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-raiden-debug-0903-2
```

#### B. Running with Multi-Worker Rollout (2 Rollout Replicas)

**Hardware Allocation**:
- **Trainer**: 1x `v5p-16` (8 chips)
- **Rollout**: 2x `v5p-8` (4 chips each, 8 chips total)
- **Total**: 16 TPU chips

```bash
/tmp/launch_raiden.sh start \
  --model qwen3-0.6b \
  --rollout-replicas=2 \
  --image gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-raiden-debug-0903-2
```

#### C. Running Qwen3.5-35B (MoE)

**Hardware Allocation (Lean Topology)**:
- **Trainer**: 1x `v5p-16` (`tpuv5:2x2x2`, 8 chips, FSDP=8, TP=1)
- **Rollout**: 1x `v5p-8` (`tpuv5:2x2x1`, 4 chips, TP=2)
- *Note*: Qwen3.5-35B-A3B has ~35B total / ~3B active params. At ~70 GB bfloat16 weights, it fits into the 95 GB HBM per chip.

```bash
/tmp/launch_raiden.sh start \
  --model qwen3.5-35b \
  --rollout-replicas=1
```

---

### 1.4 Monitoring and Lifecycle Management

```bash
# 1. View running JobSets, pod phases, and node placements
/tmp/launch_raiden.sh status

# 2. Follow orchestrator execution and training metrics
/tmp/launch_raiden.sh logs orch -f

# 3. View rollout worker vLLM and Raiden logs
/tmp/launch_raiden.sh logs rollout -f

# 4. View MaxText trainer logs
/tmp/launch_raiden.sh logs trainer -f

# 5. Stop the workload and clean up all JobSets
/tmp/launch_raiden.sh stop
```

---

## 2. Detailed Architecture & Code Modifications

For engineers building from scratch or contributing upstream, the changes across the repositories are organized into clean user branches: `${USER}/qwen35-run` (e.g. `igorts/qwen35-run`).

### 2.1 Summary of Repositories & Commits

#### A. `google/tunix` (Branch: `igorts/qwen35-run`)
- **Remote**: `https://github.com/google/tunix` and `git@github.com:igorts-git/tunix.git`
- **Key Commits**:
  - `b050af2a`: `fix(raiden,rollout): support multi-replica rollout unit registration and sorted layer binding`
    1. **Multi-Replica Rollout Fix** (`vllm_sampler_adapter.py`): Preserves dictionary structure for `unit` while setting `job_name = self.server_id` (`roll-0`, `roll-1`). Prevents rollout workers from colliding under the default `"destination"` identifier.
    2. **Deterministic Layer Sorting** (`raiden_synchronizer.py`): Sorts names and arrays alphabetically in `RaidenSynchronizer.bind()` so `layer_idx` positional metadata matches between trainer and rollout worker.
    3. **Dynamic Host Staging** (`weight_sync_coordinator.py`): Detects `HOST_STAGE` environment variable to configure `RaidenTransferOptions`.
    4. **Unified Launcher** (`launch_raiden.sh`): Adds unified CLI script for reproducible cluster execution.
  - `8c599eb6`: `fix(launcher): update default sampler, bodaborg slice topology, and output directory`
  - `732d059b`: `fix(worker,trainer): allow synchronous actor submit within running loops and use absolute output path`
  - `7847df3a`: `feat: configure bodaborg-v5p-nap launcher, Pathways rollout jobset, and Dockerfile.maxtext`

#### B. `vllm-project/tpu-inference` (Branch: `igorts/qwen35-run`)
- **Remote**: `git@github.com:vllm-project/tpu-inference.git` and `git@github.com:igorts-git/tpu-inference.git`
- **Key Commits**:
  - `e9c486f6b`: `fix(raiden): filter runtime kv cache parameters and sort array bindings`
    1. **NNX Param Filtering** (`raiden_worker_sync.py`): Filters `nnx.Param` in `extract_weight_state` to prevent non-trainable state from polluting the parameter tree.
    2. **Drop Cache Arrays Before Binding** (`raiden_worker_sync.py`): Drops variables containing `"cache"` (`cached_prefill_key`, `cached_prefill_value`) in `_filter_bindable` *before* constructing C++ `WeightSynchronizer(self.arrays)`.
       > [!CRITICAL]
       > **Why filtering must occur before C++ binding**:
       > In C++ `raw_buffer_transport.cc`, shard buffers are allocated by integer index (`layer_idx`). If cache variables are removed in Python after C++ allocation, the indices shift, writing tensor data into mismatched buffers and triggering `Destination out of bounds in batched push` or character gibberish.
    3. **Deterministic Array Sorting** (`raiden_worker_sync.py`): Sorts names and arrays alphabetically in `RaidenWorkerSync.bind()` to match the MaxText trainer.
  - `0ec3d5d3c`: `fix(runner,quantization): compatibility fallback for nvfp4 and vllm kv cache interface`

#### C. `AI-Hypercomputer/maxtext` (Branch: `igorts/qwen35-run`)
- **Remote**: `git@github.com:AI-Hypercomputer/maxtext.git`
- **Key Commits**:
  - `1066d2a24` / `9fa415f2a` (PR #5073, Merged into upstream `main`): Fixes standalone torchax converter for current `tpu-inference`, aligns tensor orientations and expert layouts (`GMM_EP`), and wires `use_standalone_converter`.
  - Upstream `main` branch includes all required engine changes.

---

## 3. Building the Runner Docker Image from Scratch

The build uses `tunix/Dockerfile.maxtext` to create a self-contained container image.

### Step 1: Stage the Build Context in `tunix`

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

### Step 2: Build and Push with Docker

```bash
REGISTRY="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext"
TAG="$(whoami)-maxtext:$(date +%Y%m%d)"
FULL_IMAGE="${REGISTRY}/${TAG}"

# Configure Docker credential helper for Artifact Registry
gcloud auth configure-docker europe-west4-docker.pkg.dev

# Build and push
docker build -t "${FULL_IMAGE}" -f Dockerfile.maxtext .
docker push "${FULL_IMAGE}"
```

---

## 4. Verification Checklist & Expected Outputs

During execution, verify the following milestones in the pod logs:

### 1. Dual-Node Pathways Scheduler Convergence
In trainer pod logs (`kubectl logs <run-id>-train-proc-0-0-* -c main`):
```
[TrainerNode] Creating MaxText device mesh...
[TrainerNode] Num_devices: 8, shape (1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1)
[TrainerNode] Restoring checkpoint from gs://maxtext-model-checkpoints/...
[TrainerNode] [process=0] [sync] Finished load in 13.76 seconds
[TrainerNode] Serving trainer worker on port 20002.
```

### 2. Multi-Worker Discovery Registration
In orchestrator logs (`kubectl logs <run-id>-orch-proc-0-0-*`):
```
[Orchestrator] Discovered rollout service (igorts-v8-06b-roll) at igorts-v8-06b-roll-proc-0-0...:20001
[Orchestrator] Discovered trainer service (igorts-v8-06b-train) at igorts-v8-06b-train-proc-0-0...:20002
[Orchestrator] All required workers are ready. Current counts: {<Role.ACTOR: 'actor'>: 1, <Role.ROLLOUT: 'rollout'>: 1}
```

### 3. Raiden Weight Synchronization & Checksums
In orchestrator logs:
```
[Orchestrator] Transfer wsync-v0-r0 (uuid=1): generated schedule for [trainer] -> [rollout] (310 variable(s), 287060 expected blocks)
[Orchestrator] Weight sync finished in 14.47 seconds.
[Orchestrator] Weight synchronization complete (policy_version=0).
```
Checksum verification (`VERIFY_WEIGHTS=true`):
- Trainer: `__grand_total__: 102835.75, __tensor_count__: 310, __element_count__: 596049920`
- Rollout: `__grand_total__: 13217973.204956055, __tensor_count__: 310, __element_count__: 596049920`

### 4. Rollout Generation & Training Progress
- Rollout worker emits coherent chain-of-thought traces:
  `[RolloutNode] [collector] traj=traj_prompt_0_g0 completion_tokens=67 text="<reasoning>..."`
- Orchestrator computes loss and advances policy version:
  `[Orchestrator] Train step 0 - loss: -0.0000 - reward_mean: 0.0625 - step_time: 51.35s`
  `[Orchestrator] <<< Step 0 finished | Advanced to Policy Version: 1`
  `[Orchestrator] === GRPO Training Finished Successfully ===`
