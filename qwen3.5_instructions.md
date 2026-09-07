# Running Distributed RL with Qwen3.5-0.6B and Qwen3.5-35B on GKE TPU v5p
**MaxText Trainer + Tunix GRPO + vLLM Rollout + Raiden-FFI Direct Weight Sync**

---

## 1. Overview & System Architecture

This document contains authoritative, verified step-by-step instructions for configuring, reproducing, and scaling distributed Reinforcement Learning (GRPO) training runs for:
- **Qwen3-0.6B** (Baseline validation, single-host `tpuv5:2x2x1` train, `tpuv5:2x2x1` rollout)
- **Qwen3.5-35B-A3B** (Large MoE model with scanned checkpoints, interleaved MoE layers, and direct weight synchronization)

### System Architecture
1. **Trainer**: `AI-Hypercomputer/maxtext` executing under the **Pathways** runtime (`JAX_PLATFORMS=proxy,cpu`, `grpc://localhost:29000`).
2. **Orchestrator**: `google/tunix` managing distributed GRPO loops, prompt distribution, reward scoring, and actor-learner synchronization.
3. **Rollout Workers**: `vllm-project/tpu-inference` running `flax_nnx` model runners on TPU v5p slices.
4. **Weight Sync**: `tpu-sync` (**Raiden-FFI**) providing low-latency TPU host-to-host DMA weight synchronization directly device-to-device.

> [!IMPORTANT]
> **Direct Device-to-Device (FFI) is Mandatory on the Trainer**:
> The 0.6B and 35B models are development milestones toward scaling distributed RL to **3 Trillion (3T) parameter models**. Direct JAX FFI device-to-device transfers (`weight_synchronizer_ffi`) are strictly required on the **Trainer** side. Staging trainer weights via CPU memory (`HOST_STAGE`) causes fatal host Out-Of-Memory (OOM) errors and layout re-tiling bugs on larger models and is strictly forbidden on the Trainer.
>
> **Rollout Workers & Pathways / FFI Runtime Separation**:
> Rollout workers (`vLLM / tpu-inference`) execute on bare-metal TPU slices without Pathways due to runtime constraints in the inference stack. Because JAX FFI handlers currently rely on Pathways (`proxy`), rollout workers do not use Pathways, operating via the native C++ Raiden worker synchronizer.
>
> **Developer-Only Guardrails (DO NOT MERGE TO PRODUCTION)**:
> Automatic 2-hour worker timeouts (`activeDeadlineSeconds: 7200`), 10-minute cleanup TTLs (`ttlSecondsAfterFinished: 600`), and checkpoint bypasses (`DISABLE_CHECKPOINTING=true`) are strictly intended for rapid developer iteration to prevent quota leaks and save disk I/O. They are structured as standalone commits prefixed with `[DEV ONLY - DO NOT MERGE TO PROD]` and must NOT be propagated to production code.

---

## 2. Core Technical Findings & Root Cause Analysis

### A. The Shard Mismatch & Independent Broadcast Trap (Why Gibberish Occurred)
In legacy instructions (line 132), the recommended configuration was:
- Trainer: `tpuv5:2x2x2` (8 chips, `FSDP=8` $\to$ 8 source shards `[0..7]`)
- Rollout: `tpuv5:2x2x1` (4 chips, `TP=2` $\to$ 2 destination shards `[0..1]`)

**Why this failed with repetitive multilingual/ASCII gibberish**:
1. In Raiden (`weight_sync_coordinator.py`), rollout replicas are registered with distinct `job_name` identifiers (e.g. `rollout-0`, `rollout-1`). Raiden treats each replica as an **independent broadcast destination**, NOT an aggregated shard group.
2. For each destination replica independently, Raiden computes the intersection between the source shards (`[0..7]`) and the destination shards (`[0..1]`).
3. Under `TP=2`, each replica receives only shards 0 and 1. **75% of model weights are never transferred**.
4. Under `TP=4`, each replica receives only shards 0..3. **50% of model weights are never transferred**.
5. The uninitialized/zero weights in the rollout worker cause the language model to generate pure gibberish (e.g. `\u043c\u043e\u0440 nay Dont\u54ea...`).
6. **Core Invariant**: Every rollout replica must independently receive 100% of the model weights. Therefore, the destination shard count per replica must equal the trainer source shard count exactly:
   $$\forall \text{replica } r: \quad N_{\text{dst}} \equiv N_{\text{src}}$$

### B. The Off-by-2 MoE Dimension Padding Bug in `adapter.py` (Why 4 Chips OOMed)
When testing Qwen3.5-35B on 4 chips (`tpuv5:2x2x1`, `FSDP=4`), prior sessions observed a `pjrt.OOM` crash and concluded that 35B could not fit on 4 chips. **This conclusion was incorrect.**

**Root Cause**:
- In `maxtext/src/maxtext/integration/vllm/maxtext_vllm_adapter/adapter.py:169`:
  ```python
  if hidden_size is not None and (hidden_size // moe_mlp_tp_size) % (2 * num_lanes) != 0:
    padded_hidden_size = next_power_of_two(hidden_size)
    while (padded_hidden_size // moe_mlp_tp_size) < (2 * num_lanes):
      padded_hidden_size = next_power_of_two(padded_hidden_size + 1)
  ```
- In `moe.py:fused_moe_matmul`, gate and up weights are fused together along dimension $N$:
  $$\text{size\_n} = 2 \cdot \left(\frac{\text{hidden\_size}}{\text{moe\_mlp\_tp\_size}}\right)$$
- Megablox `gmm_v2.py:1078-1083` checks `size_n % (2 * num_lanes) == 0`. Substituting `size_n`, the factor of 2 cancels:
  $$\left(2 \cdot \frac{\text{hidden\_size}}{\text{moe\_mlp\_tp\_size}}\right) \pmod{2 \cdot \text{num\_lanes}} == 0 \iff \left(\frac{\text{hidden\_size}}{\text{moe\_mlp\_tp\_size}}\right) \pmod{\text{num\_lanes}} == 0$$
- On TPU v5p, $\text{num\_lanes} = 128$. For Qwen3.5-35B (`hidden_size=512`), under `TP=4`:
  $$\frac{512}{4} = 128 \equiv 0 \pmod{128}$$
  The Megablox kernel requirement is already satisfied.
- However, `adapter.py` erroneously checked $128 \pmod{256} \ne 0$ and artificially padded `moe_intermediate_size` from 512 to 1024!
- This forced passing `TRAINER_PADDED_MOE_MLP_DIM=1024`, doubling all 120 expert weight matrices, and inflating static HBM requirements on 4 chips from **53.55 GB to >107 GB**, exceeding the 95 GB TPU v5p HBM and causing `pjrt.OOM`.
- **Fix**: Changing `2 * num_lanes` to `num_lanes` in `adapter.py:169` eliminates the artificial padding, keeping `hidden_size=512` (71.4 GB total weights).

### C. The Verified Zero-Gibberish Architecture ($4 \equiv 4$ Single-Host Slices)
With native `hidden_size=512`:
- **Trainer**: `tpuv5:2x2x1` (4 chips, 1 host node), `FSDP=4`, `TP=1` $\to$ 4 source shards `[0..3]`.
- **Rollout**: `tpuv5:2x2x1` (4 chips, 1 host node), `TP=4`, `REPLICAS=1` $\to$ 4 destination shards `[0..3]`.
- **Shards**: $4 \equiv 4$ (100% full weight transfer, 0 dropped shards, 0 partial overlap warnings).
- **HBM Headroom on TPU v5p (95 GB total)**:
  - Trainer static memory: 53.55 GB / chip (**41.45 GB free HBM** for activations and FFI buffers).
  - Rollout static memory: 17.85 GB / chip (**77.15 GB free HBM** for KV cache and inference).
- **Text Quality**: Completely coherent, valid chain-of-thought mathematical reasoning without gibberish.

---

## 3. Comparison of Operational Options

### Option A: Running with `origin/mohit/rl-raiden-vllm-fixes`
Mohit's branch contains 27 exploratory and diagnostic commits developed during early prototyping on TPU v5e slices (`tpuv5e:4x4`, 16 chips).
- **Key contributions**: Introduced inhomogeneous layer cycle unscanning for MoE, FFI H2D chunked transfers in `tpu-inference`, and `refresh_model_state_leaves()`.
- **Limitations**: Configured for 16-chip v5e topologies (`TRAINER_MESH_FSDP=16`, `ROLLOUT_MESH_TP=16`). When adapted to TPU v5p, it suffered from the shard count mismatch (8 vs 2 shards) and the off-by-2 padding bug in `adapter.py`.
- **Usage**: Suitable for historical baseline comparison and reference.

### Option B (Recommended): Minimal Clean Stack on `origin/main`
Rather than carrying 27 historical commits, Option B applies only the **3 minimal required cherry-picks** on top of clean `origin/main` across all repositories:
1. `maxtext`: Cherry-pick `e0d3e4124` (inhomogeneous MoE unscan) + the 2-line fix in `adapter.py:169`.
2. `tunix`: Cherry-pick `3e0a51f9` (step-0 `policy_version` tracking in weight sync coordinator).
3. `tpu-inference`: Cherry-pick `29c6d12db` (NUMA port routing for multi-shard rollout endpoints).

Option B is clean, fully understood, and verified to run Qwen3.5-35B without gibberish or OOM.

---

## 4. Fast-Track: Running with Pre-Built Verified Images

If you want to run immediately without building container images or compiling C++ extensions from scratch, use our verified pre-built images.

### Verified Images
- **Qwen3.5-35B-A3B Verified Runner Image (Option B / Minimal Stack)**:
  ```text
  europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-opta-verified
  ```
- **Qwen3.5-35B-A3B Image (Option A Baseline)**:
  ```text
  europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v17
  ```
- **Qwen3-0.6B Verified Runner Image**:
  ```text
  europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v5
  ```

### Step 1: Connect to the GKE Cluster
```bash
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --region europe-west4 \
  --project cloud-tpu-shared-capacity
```

### Step 2: Checkout the Verified Launcher
```bash
cd ~/git/tunix
git checkout igorts/qwen35-run
```

### Step 3: Launch Workloads

#### Verified Single-Host $4 \equiv 4$ Topology (Qwen3.5-35B, Zero Gibberish)
```bash
TRAINER_MESH_FSDP=4 ROLLOUT_MESH_TP=4 \
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh start \
  --model qwen3.5-35b \
  --rollout-replicas=1 \
  --debug \
  --reward-mode=exact \
  --image europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-opta-verified
```

#### Verified Baseline (Qwen3-0.6B)
```bash
TRAINER_MESH_FSDP=4 ROLLOUT_MESH_TP=4 \
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh start \
  --model qwen3-0.6b \
  --rollout-replicas=1 \
  --debug \
  --reward-mode=exact \
  --image europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v5
```

### Step 4: Monitor and Inspect
```bash
# Check JobSet and pod status
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh status --model qwen3.5-35b

# Stream trainer logs
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs trainer -f --model qwen3.5-35b

# Stream rollout worker logs
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs rollout -f --model qwen3.5-35b

# Stream orchestrator logs (shows GSM8K prompts, completions, and rewards)
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs orch -f --model qwen3.5-35b

# Stop run and clean up all JobSets
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh stop --model qwen3.5-35b
```

---

## 5. Building From Scratch (Option B Minimal Stack)

Follow this section to build everything from clean upstream git checkouts on `origin/main`.

### 5.1 Clone the Repositories
```bash
mkdir -p ~/git && cd ~/git
git clone https://github.com/AI-Hypercomputer/maxtext.git
git clone https://github.com/google/tunix.git
git clone https://github.com/vllm-project/tpu-inference.git
git clone https://github.com/AI-Hypercomputer/tpu-sync.git
```

---

### 5.2 Required Cherry-Picks & Code Patches

#### A. Repository: `maxtext` (`AI-Hypercomputer/maxtext`)
```bash
cd ~/git/maxtext
git checkout origin/main

# 1. Cherry-pick inhomogeneous MoE layer unscan support
git cherry-pick e0d3e4124
```

**2. Apply 2-line fix in `maxtext_vllm_adapter/adapter.py:169`**:
Replace:
```python
if hidden_size is not None and (hidden_size // moe_mlp_tp_size) % (2 * num_lanes) != 0:
  padded_hidden_size = next_power_of_two(hidden_size)
  while (padded_hidden_size // moe_mlp_tp_size) < (2 * num_lanes):
    padded_hidden_size = next_power_of_two(padded_hidden_size + 1)
```
With:
```python
if hidden_size is not None and (hidden_size // moe_mlp_tp_size) % num_lanes != 0:
  padded_hidden_size = next_power_of_two(hidden_size)
  while (padded_hidden_size // moe_mlp_tp_size) < num_lanes:
    padded_hidden_size = next_power_of_two(padded_hidden_size + 1)
```

---

#### B. Repository: `tunix` (`google/tunix`)
```bash
cd ~/git/tunix
git checkout origin/main

# Cherry-pick step-0 policy_version tracking in weight sync coordinator
git cherry-pick 3e0a51f9
```

---

#### C. Repository: `tpu-inference` (`vllm-project/tpu-inference`)
```bash
cd ~/git/tpu-inference
git checkout origin/main

# Cherry-pick multi-shard NUMA port routing for rollout workers
git cherry-pick 29c6d12db
```

---

### 5.3 Building the Raiden Wheel (`tpu-sync`)

#### Option A: Use Verified Pre-Built Wheel
A verified wheel is available in `tunix/.docker/tpu_sync/`:
```text
tpu_raiden_jax-0.0.1.dev20260903185444-cp312-cp312-manylinux_2_31_x86_64.whl
```
Or install via authenticated Google Artifact Registry:
```bash
pip install keyrings.google-artifactregistry-auth
pip install tpu-raiden-jax --extra-index-url https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/
```

#### Option B: Compile Wheel From Source
```bash
cd ~/git/tpu-sync
git checkout $(cat lkg.version)
./build.sh jax

mkdir -p ~/git/tunix/.docker/tpu_sync
cp dist/tpu_raiden_jax-*.whl ~/git/tunix/.docker/tpu_sync/
```

---

### 5.4 Pathways Container Images
The verified Pathways server and proxy images used for TPU v5p are:
```bash
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904"
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904"
```

---

### 5.5 Building and Pushing the Docker Image

```bash
cd ~/git/tunix

mkdir -p .docker/tpu_inference .docker/maxtext .docker/tpu_sync

rsync -av --delete --exclude='.git' --exclude='venv' --exclude='.venv' \
  ~/git/tpu-inference/ .docker/tpu_inference/

rsync -av --delete --exclude='.git' --exclude='venv' --exclude='.venv' \
  ~/git/maxtext/ .docker/maxtext/

TAG="qwen35-$(date +%Y%m%d)-custom"
IMAGE="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/${USER}-maxtext:${TAG}"

docker build -t "${IMAGE}" -f Dockerfile.maxtext .
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://europe-west4-docker.pkg.dev
docker push "${IMAGE}"
```

---

## 6. Model Configurations & Checkpoint References

| Configuration | Qwen3-0.6B | Qwen3.5-35B-A3B (Verified Minimal) | Qwen3.5-35B-A3B (Legacy 8-chip) |
| :--- | :--- | :--- | :--- |
| **Model ID** | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3.5-35B-A3B` | `Qwen/Qwen3.5-35B-A3B` |
| **MaxText Model** | `qwen3-0.6b` | `qwen3.5-35b-a3b` | `qwen3.5-35b-a3b` |
| **Checkpoint Path** | `gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items` | `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items` | `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items` |
| **Trainer Slice** | `tpuv5:2x2x1` (4 chips) | `tpuv5:2x2x1` (4 chips) | `tpuv5:2x2x2` (8 chips) |
| **Trainer Mesh** | `FSDP=4` | `FSDP=4` | `FSDP=8` |
| **Rollout Slice** | `tpuv5:2x2x1` (4 chips) | `tpuv5:2x2x1` (4 chips) | `tpuv5:2x2x1` (4 chips) |
| **Rollout Mesh** | `TP=4` | `TP=4` | `TP=2` or `TP=4` |
| **Source Shards** | 4 | 4 | 8 |
| **Destination Shards** | 4 | 4 | 2 (with TP=2) or 4 (with TP=4) |
| **Weight Transfer %** | **100% (Full Transfer)** | **100% (Full Transfer)** | **25% (TP=2) or 50% (TP=4) Missing** |
| **Rollout Output** | Clean Mathematical CoT | Clean Mathematical CoT | Repetitive Multilingual Gibberish |
| **Trainer Static HBM** | ~3.8 GB / chip | 53.55 GB / chip (41.45 GB free) | >107 GB / chip (if padded 1024) |
| **Rollout Static HBM** | ~1.5 GB / chip | 17.85 GB / chip (77.15 GB free) | ~35.7 GB / chip |

---

## 7. Known Pitfalls & Hallucination Deconstruction

### 1. Hallucination: "35B Cannot Fit on 4 TPU Chips Due to Architectural Memory Limits"
- **Reality**: Qwen3.5-35B-A3B has 71.4 GB total parameters. Under `FSDP=4`, each chip holds 17.85 GB of weights. With optimizer states and FFI transfer buffers, total static footprint is **53.55 GB per chip**, leaving **41.45 GB of free HBM** on a 95 GB TPU v5p chip.
- It only OOMed in prior tests because `adapter.py` erroneously padded `moe_intermediate_size` from 512 to 1024, doubling all 120 expert weight matrices and exceeding HBM capacity.

### 2. Hallucination: "Multiple Rollout Replicas Aggregate Shards to Match the Trainer"
- **Reality**: In Raiden, each rollout replica registers as an independent broadcast destination. Shards are **not aggregated across replicas**. If the trainer has 8 shards and each replica has 4 shards (`TP=4`), both replicas receive shards 0..3 and both drop shards 4..7 (50% missing weights). Every replica must independently match the trainer shard count ($N_{\text{dst}} \equiv N_{\text{src}}$).

### 3. Hallucination: "`PREFUSE_MOE_WEIGHTS=true` Was Working and Saved Memory"
- **Reality**: `PREFUSE_MOE_WEIGHTS` was completely unplumbed in `k8s_launcher.sh` and not recognized by `tunix`. It had zero effect on memory or execution.

### 4. "Repetitive Newline / Gibberish Text Generation"
- **Cause**: Shard count mismatch between Trainer and Rollout ($N_{\text{dst}} < N_{\text{src}}$), leaving 50% to 75% of model weights uninitialized, or passing `skip_tiling=False` which causes the receiver to re-tile already-tiled DMA memory.
- **Fix**: Run with balanced sharding ($4 \equiv 4$) and leave `skip_tiling=None` in `RaidenTransferOptions`.

### 5. "Checksum Mismatch on Shard 1"
- **Cause**: Rollout workers in `TP=2` or `TP=4` listening on distinct ports per NUMA node, but advertising a single port to the coordinator in `metadata_dict()`.
- **Fix**: Cherry-pick commit `29c6d12db` in `tpu-inference` so `get_local_endpoints()` routes each shard index to its distinct local NUMA port.

### 6. "`compute_on2()` missing required argument `out_memory_spaces`"
- **Cause**: Upstream JAX added `compute_on2` with mandatory `out_memory_spaces`. The TPU-sync FFI wheel calls `@compute_on.compute_on` without this argument.
- **Fix**: Ensure commit `6d41e392` in `tunix` is present to wrap `compute_on2` with default `out_memory_spaces=jax.memory.Space.Device`.
