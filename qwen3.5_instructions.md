# Qwen3-0.6B & Qwen3.5-35B-A3B Distributed GRPO & Raiden FFI Weight Sync Guide

This guide documents the end-to-end setup, execution, and build procedures for distributed Reinforcement Learning (GRPO) with **MaxText** (Pathways trainer with JAX FFI weight sync), **Tunix** (orchestrator), **vLLM / `tpu-inference`** (rollout worker), and **`tpu-sync` / Raiden** (high-bandwidth DMA weight transfer) on Google Kubernetes Engine (GKE) TPU v5p clusters.

It covers both:
1. **Part 1: Fast-Track Execution**: Run immediately using our prebuilt, verified container image (`igorts-maxtext:qwen35-20260904-v17`).
2. **Part 2: Building Everything from Scratch**: Start from clean upstream repositories, inspect merged vs. pending PRs, cherry-pick required unmerged commits from our forks, build the Raiden wheel (`tpu_raiden_jax`), configure Pathways server images, and build the unified Docker image.

---

## Part 1: Fast-Track Execution (Prebuilt Docker Image)

If you want to launch distributed GRPO training and weight synchronization immediately without building images or wheels, use our prebuilt, verified image:

- **Unified Worker Image**: `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v17`
- **Pathways Server Image**: `us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904`
- **Pathways Proxy Server Image**: `us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904`

### 1.1 Fast-Track: Qwen3-0.6B (1 Rollout Worker)

This configuration runs an 8-chip Pathways trainer (`tpuv5:2x2x2`, `FSDP=8`) and a 4-chip vLLM rollout worker (`tpuv5:2x2x1`, `TP=4`).

```bash
cd /usr/local/google/home/$USER/git/tunix

PATH="/usr/local/google/home/$USER/git/maxtext/venv/bin:$PATH" \
KUBECONFIG="$HOME/.kube/config" \
PROJECT=cloud-tpu-shared-capacity \
REGION=europe-west4 \
CLUSTER=bodaborg-v5p-nap \
USER=${USER}-06b-repro \
TUNIX_IMAGE=europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v17 \
K8S_NAMESPACE=default \
CPU_MACHINE=e2-standard-16 \
GCS_SCRATCH_LOCATION=gs://mohitkhatwani-pathways-euw4/tmp \
DISABLE_CHECKPOINTING=1 \
TPU_SYNC_EARLY_IMPORT=0 \
MODEL_NAME=Qwen3-0.6B MODEL_ID=Qwen/Qwen3-0.6B MAXTEXT_MODEL_NAME=qwen3-0.6b \
TRAINER_BACKEND=maxtext \
PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904 \
PATHWAYS_PROXY_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904 \
RAIDEN_DEVICES_PER_HOST=4 \
TRAINER_TPU_SLICE=tpuv5:2x2x2 TRAINER_MESH_FSDP=8 TRAIN_MICRO_BATCH_SIZE=8 \
ROLLOUT_TPU_SLICE=tpuv5:2x2x1 ROLLOUT_MESH_TP=4 ROLLOUT_REPLICAS=1 \
TRAINER_RAIDEN_USE_FFI=1 ROLLOUT_RAIDEN_USE_FFI=0 \
BATCH_SIZE=2 NUM_GENERATIONS=4 MAX_PROMPT_LENGTH=256 MAX_RESPONSE_LENGTH=256 \
SAMPLER=vllm WEIGHT_SYNC_MODE=raiden VERIFY_WEIGHTS=true MAX_STEPS=2 \
DEBUG=1 REWARD_MODE=exact \
bash tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh \
--command=start --image=europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v17
```

**Expected Verification Checksums (`VERIFY_WEIGHTS=true`)**:
- **Trainer Source**: `__grand_total__: 13217973.405761719`, `__tensor_count__: 310`, `__element_count__: 596049920`
- **Rollout Destination**: `__grand_total__: 13217973.130737305`, `__tensor_count__: 310`, `__element_count__: 596049920`
- **Output Quality**: Coherent step-by-step math reasoning (`<reasoning>...</reasoning> Answer: <answer>...</answer>`), `EXIT_CODE=0`.

---

### 1.2 Fast-Track: Qwen3.5-35B-A3B (1 or 2 Rollout Workers)

For `Qwen3.5-35B-A3B` (`base_moe_mlp_dim=512`), setting `ROLLOUT_MESH_TP=2` (`TP=2`) is critical: each TP shard holds `512 / 2 = 256` hidden units per expert, which matches TPU v5p's MXU tile requirement (256) with **0% padding inflation**, keeping the model at its true **35.3B parameter count (`71.4 GB` total weights)**.

- **Trainer**: `tpuv5:2x2x2` (8 chips), `TRAINER_MESH_FSDP=4`, `TRAINER_MESH_TP=2` (`8.9 GB/chip` weights, **68.3 GB free HBM per chip**).
- **Rollout Worker(s)**: `tpuv5:2x2x1` (4 chips per replica), `ROLLOUT_MESH_TP=2` (`TP=2`). Set `ROLLOUT_REPLICAS=1` for 1 rollout worker or `ROLLOUT_REPLICAS=2` for 2 rollout workers.

```bash
cd /usr/local/google/home/$USER/git/tunix

PATH="/usr/local/google/home/$USER/git/maxtext/venv/bin:$PATH" \
KUBECONFIG="$HOME/.kube/config" \
PROJECT=cloud-tpu-shared-capacity \
REGION=europe-west4 \
CLUSTER=bodaborg-v5p-nap \
USER=${USER}-35b-repro \
TUNIX_IMAGE=europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v17 \
K8S_NAMESPACE=default \
CPU_MACHINE=e2-standard-16 \
GCS_SCRATCH_LOCATION=gs://mohitkhatwani-pathways-euw4/tmp \
DISABLE_CHECKPOINTING=1 \
TPU_SYNC_EARLY_IMPORT=0 \
MODEL_NAME=Qwen3.5-35B-A3B MODEL_ID=Qwen/Qwen3.5-35B-A3B MAXTEXT_MODEL_NAME=qwen3.5-35b-a3b \
TRAINER_BACKEND=maxtext \
MAXTEXT_CKPT=gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items \
PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904 \
PATHWAYS_PROXY_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904 \
RAIDEN_DEVICES_PER_HOST=4 \
TRAINER_TPU_SLICE=tpuv5:2x2x2 TRAINER_MESH_FSDP=4 TRAINER_MESH_TP=2 TRAIN_MICRO_BATCH_SIZE=8 \
ROLLOUT_TPU_SLICE=tpuv5:2x2x1 ROLLOUT_MESH_TP=2 ROLLOUT_REPLICAS=2 PREFUSE_MOE_WEIGHTS=true \
TRAINER_RAIDEN_USE_FFI=1 ROLLOUT_RAIDEN_USE_FFI=0 \
BATCH_SIZE=2 NUM_GENERATIONS=4 MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=512 \
SAMPLER=vllm WEIGHT_SYNC_MODE=raiden VERIFY_WEIGHTS=true MAX_STEPS=2 \
DEBUG=1 REWARD_MODE=exact \
bash tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh \
--command=start --image=europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v17
```

**Expected Verification Metrics**:
- **Trainer Registration**: `registered 1 work unit(s) with 633 variables on mesh ('fsdp', 'tensor')`, `host memory max RSS: 25986.1 MB`
- **Raiden Transfer**: `last_tiled_bytes: 71400282112` (`71.4 GB` transferred per rollout replica at `~782.56 Gbps` tiling bandwidth)
- **Output Quality**: Coherent step-by-step math reasoning across all rollout replicas with zero gibberish (`[collector] traj=traj_prompt_... text='<reasoning>\n1. **Identify the initial number...'`).

---

## Part 2: Building Everything From Scratch (Clean Branches & Future-Proof Cherry-Picking)

Because upstream repositories (`AI-Hypercomputer/maxtext`, `google/tunix`, `vllm-project/tpu-inference`) actively merge PRs, follow this **Future-Proofing Protocol** before cherry-picking commits:

### 2.1 Future-Proofing Protocol & Fork Setup

1. **Check Upstream `origin/main` & Merged PRs**:
   - Before cherry-picking any commit below, check `git log origin/main --grep="<keyword>"` or check the status of key upstream PRs:
     - **MaxText**: PR `#5045` (Raiden FFI weight sync & `raiden_unscan.py` support).
     - **Tunix**: PR `#2054` (Pathways Raiden FFI coordinator & progressive memory reclamation).
     - **tpu-inference**: PR `#3516` (`Fix the Raiden RL weight-sync path`).
   - Also review open PRs across all three repositories (`gh pr list --search "raiden"`) to see if newer versions of these fixes have landed.

2. **Add Fork Remotes**:
   Since contributors may not have direct push access to all upstream repositories, add our fork remotes (`igorts-git`) where all verified commits are pushed and maintained:
   ```bash
   # In maxtext
   git remote add fork git@github.com:igorts-git/maxtext.git && git fetch fork

   # In tunix
   git remote add fork git@github.com:igorts-git/tunix.git && git fetch fork

   # In tpu-inference
   git remote add fork git@github.com:igorts-git/tpu-inference.git && git fetch fork
   ```

---

### 2.2 Step-by-Step Repository Setup & Cherry-Picks

#### A. `maxtext` Setup (Target Branch: `fork/igorts/qwen3.5-35b`)

Start from `origin/mohit/rl-raiden-vllm-fixes` (or `origin/main`) and cherry-pick the following commits from `fork/igorts/qwen3.5-35b` if not yet present on your base branch:

| Commit Hash | Description | Why It Is Required |
| :--- | :--- | :--- |
| `106e7efdf`, `0cfca938d`, `75a6e8206`, `65f9cdccb` | **Direct Weight Conversion & Interleaved MoE Unscanning** (`raiden_unscan.py`, `qwen35_moe.py`) | Unlike Qwen3-0.6B (uniform dense layers), Qwen3.5-35B-A3B interleaves 1 dense layer with 3 MoE layers (`cycle_interval=4`). Unrolls scanned layer blocks into flat dictionary keys (`layers_0`..`layers_39`) matching `vLLM` variable names. |
| `80d1d569e`, `8fc4612d4`, `d39bc73d3`, `8fb18c518`, `2a4bf4785`, `319f4eaa7` | **Raiden FFI `D2H` in `MaxTextTrainingEngine`** (`maxtext_engine.py`) | Under Pathways (`JAX_PLATFORMS=proxy`), host staging causes client OOM. Binds directly to TPU device arrays via `weight_synchronizer_ffi` (`self._raiden_sync.d2h()`) using a persistent synchronizer instance. |
| `71535dcfc`, `3db9d12b2` | **Silent Failure Visibility & Empty-Metrics Checkpoint Guard** (`checkpointing.py`) | Surfaces `raiden_synchronizer` import errors immediately and prevents Orbax checkpoint crashes when saving step 0 weights before metrics exist. |

```bash
cd /usr/local/google/home/$USER/git/maxtext
git checkout -b my-qwen35-branch origin/mohit/rl-raiden-vllm-fixes
git cherry-pick 71535dcfc 3db9d12b2 106e7efdf 0cfca938d 75a6e8206 65f9cdccb 80d1d569e 8fc4612d4 d39bc73d3 8fb18c518 2a4bf4785 319f4eaa7
```

---

#### B. `tunix` Setup (Target Branch: `fork/igorts/qwen3.5-35b`)

Start from `origin/mohit/rl-raiden-vllm-fixes` (or `origin/main`) and cherry-pick from `fork/igorts/qwen3.5-35b`:

| Commit Hash | Description | Why It Is Required |
| :--- | :--- | :--- |
| `9d5d89cc`, `f0ffeae7`, `6368f163`, `e52fcddc` | **Pathways Raiden FFI Weight Sync & Progressive Memory Reclamation** (`raiden_synchronizer.py`) | Executes `init_weight_synchronizer_and_d2h` FFI custom calls on Pathways proxy arrays and reclaims temporary unscanned/staging buffers immediately after transfer. |
| `3e0a51f9`, `aecf784c` | **Policy Version Tracking & Multi-Rollout Discovery** (`weight_sync_coordinator.py`) | Fixes `policy_version` tracking when `sync_request` is `None` at step 0 and enables parallel weight sync across multiple rollout replicas (`ROLLOUT_REPLICAS=2`). |
| `5d9ec73c`, `e0dba1e3`, `1feb677d`, `bb89edf9` | **Kubernetes Launcher & Pathways JobSet Configuration** (`k8s_launcher.sh`) | Configures `jobset.pathways.yaml` and default batch/mesh parameters for multi-host vLLM rollout workers and Pathways trainers. |

```bash
cd /usr/local/google/home/$USER/git/tunix
git checkout -b my-qwen35-branch origin/mohit/rl-raiden-vllm-fixes
git cherry-pick 9d5d89cc f0ffeae7 6368f163 e52fcddc 3e0a51f9 aecf784c 5d9ec73c e0dba1e3 1feb677d bb89edf9
```

---

#### C. `tpu-inference` Setup (Target Branch: `fork/igorts/qwen35-run`)

Start from `origin/mohit/rl-raiden-vllm-fixes` (or `origin/main`) and cherry-pick from `fork/igorts/qwen35-run`:

| Commit Hash | Description | Why It Is Required |
| :--- | :--- | :--- |
| `f89fe0090` | **FFI Raiden `H2D` Path on Rollout Worker** (`raiden_worker_sync.py`) | Adds JAX FFI host-to-device (`H2D`) custom call support for receiving weights on the rollout worker. |
| `1fc80182a` | **Alphabetical Sorting of Array Bindings & KV Cache Filtering** (`tpu_worker_jax.py`, `raiden_worker_sync.py`) | **Critical against Gibberish**: Raiden matches tensors purely by positional list index (`tensor i` $\rightarrow$ `tensor i`). Sorting `zip(names, arrays)` alphabetically on both trainer and rollout worker prevents permuted-layer gibberish when PyTree orders differ, and filters out runtime KV cache buffers. |
| `29c6d12db` | **Multi-Shard NUMA Port Routing** (`raiden_worker_sync.py`) | Routes multi-shard rollout endpoints to distinct ports per NUMA node / local rank, preventing port collisions when multiple TPU ranks initialize Raiden listeners. |
| `8cabf13dc`, `4c23bf207` | **vLLM Quantization & KV Cache Interface Fallbacks** | Ensures compatibility with `compressed_tensors` (`is_equal_or_regex_match`) and vLLM KV cache interfaces. |

```bash
cd /usr/local/google/home/$USER/git/tpu-inference
git checkout -b my-qwen35-branch origin/mohit/rl-raiden-vllm-fixes
git cherry-pick f89fe0090 8cabf13dc 1fc80182a 29c6d12db 4c23bf207
```

---

## Part 3: Building the Raiden Wheel & Container Images

### 3.1 Specifying & Building the Raiden Wheel (`tpu_raiden_jax` / `tpu-sync`)

The Raiden weight synchronizer requires the `tpu_raiden_jax` (`tpu-sync`) Python wheel compiled with JAX FFI custom call targets (`init_weight_synchronizer_and_d2h` and `push_weights_resharded`).

1. **Using the Prebuilt Wheel**:
   - Inside our container image (`igorts-maxtext:qwen35-20260904-v17`), the wheel is located at `/app/raiden_wheels/tpu_raiden_jax-*.whl`.
2. **Building the Wheel from Scratch**:
   If building from the internal `tpu-sync` / `tpu_raiden_jax` source repository:
   ```bash
   cd /path/to/tpu-sync
   bazel build //tpu_sync/frameworks/jax:tpu_raiden_jax_wheel
   mkdir -p /usr/local/google/home/$USER/git/tunix/raiden_wheels
   cp bazel-bin/tpu_sync/frameworks/jax/*.whl /usr/local/google/home/$USER/git/tunix/raiden_wheels/
   ```

---

### 3.2 Plugging in Pathways Server & Proxy Server Images

When running on GKE TPU v5p with Pathways, pass the Raiden-compatible Pathways server and proxy server images via environment variables to `k8s_launcher.sh`:

```bash
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904"
export PATHWAYS_PROXY_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904"
```
These images include the native C++ `WeightSynchronizerListener` transport server required by `weight_synchronizer_ffi`.

---

### 3.3 Building & Pushing the Unified Docker Image

The unified worker image packages `maxtext`, `tunix`, `tpu-inference`, and the `tpu_raiden_jax` wheel into a single container used by the orchestrator, trainer, and rollout workers.

```bash
cd /usr/local/google/home/$USER/git/tunix

# Ensure your checked-out maxtext, tpu-inference, and raiden_wheels are staged in the build context
export IMAGE_TAG="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/${USER}-maxtext:qwen35-$(date +%Y%m%d)-v1"

docker build \
  -f Dockerfile.maxtext \
  -t "${IMAGE_TAG}" \
  .

docker push "${IMAGE_TAG}"
```

Once pushed, pass `--image="${IMAGE_TAG}"` (and `TUNIX_IMAGE="${IMAGE_TAG}"`) to `k8s_launcher.sh`.

---

## Part 4: Critical Architectural Pitfalls & Troubleshooting

### 4.1 Why `ROLLOUT_DATA_PARALLEL=2` Causes Gibberish (`8 === 8` Trap)
Never set `ROLLOUT_DATA_PARALLEL=2` (`DP=2`) when syncing from an FSDP trainer without replica-aware broadcasting:
- `DP=2` on an 8-chip slice (`TP=4, DP=2`) creates **two independent data-parallel replicas** (`chips 0..3` and `chips 4..7`).
- Raiden pairs trainer FSDP shards `0..3` with Replica 0 and shards `4..7` with Replica 1.
- Because both sides have 8 shards (`8 === 8`), Raiden reports 0 overlap warnings and identical global checksums (`__grand_total__`), yet each replica holds only **50% of the model weights**, producing immediate gibberish. Always use `ROLLOUT_DATA_PARALLEL=1` (`DP=1`) and scale throughput via `ROLLOUT_REPLICAS=2`.

### 4.2 Why `ROLLOUT_MESH_TP=8` Causes 4x Parameter Inflation & HBM OOM on 35B
In `Qwen/Qwen3.5-35B-A3B`, `base_moe_mlp_dim` (`moe_intermediate_size`) is **512**:
- Each TP shard in `tpu-inference` (`vLLM`) pads its local slice (`512 / TP`) to a multiple of **256** (TPU v5p MXU tile alignment):
  - With **`ROLLOUT_MESH_TP=8` (`TP=8`)**: `512 / 8 = 64` per shard $\rightarrow$ padded to `256` per shard $\rightarrow$ `8 * 256 = 2048` (`TRAINER_PADDED_MOE_MLP_DIM=2048`). This pads all 256 experts across 40 layers by **4x**, inflating the **35.3B model (`71.4 GB`) into a 131.3B model (`262.6 GB` weights = `32.82 GB/chip`)**, exhausting `tpuv5:2x2x2` HBM (`98.46 GB/chip` static state > `95.00 GB`).
  - With **`ROLLOUT_MESH_TP=2` (`TP=2`)**: `512 / 2 = 256` per shard, which is already a multiple of 256 (`2 * 256 = 512`). This produces **0% padding inflation**, keeping the model at its true **35.3B parameter count (`71.4 GB` weights = `8.9 GB/chip` on 8 chips)** and leaving **68.3 GB free HBM per chip**.
