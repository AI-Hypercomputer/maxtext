# Running Distributed RL with Qwen3.5-0.6B and Qwen3.5-35B on GKE TPU v5p
**MaxText Trainer + Tunix GRPO + vLLM Rollout + Raiden-FFI Direct Weight Sync**

---

## 1. Overview & Architecture

This document contains step-by-step instructions for reproducing and scaling distributed Reinforcement Learning (GRPO) training runs for:
- **Qwen3-0.6B** (Baseline validation, `tpuv5:2x2x2` train, `tpuv5:2x2x1` rollout)
- **Qwen3.5-35B-A3B** (Large MoE model with scanned checkpoints and interleaved MoE layers, `tpuv5:2x2x2` train, `tpuv5:2x2x1` rollout)

### System Architecture
1. **Trainer**: `AI-Hypercomputer/maxtext` executing under the **Pathways** runtime (`JAX_PLATFORMS=proxy,cpu`, `grpc://localhost:29000`).
2. **Orchestrator**: `google/tunix` managing distributed GRPO loops, prompt distribution, reward scoring, and actor-learner synchronization.
3. **Rollout Workers**: `vllm-project/tpu-inference` running `flax_nnx` model runners on TPU v5p slices.
4. **Weight Sync**: `tpu-sync` (**Raiden-FFI**) providing low-latency TPU host-to-host DMA weight synchronization directly device-to-device.

> [!IMPORTANT]
> **Direct Device-to-Device (FFI) is Mandatory on the Trainer**:
> The 0.6B and 35B models are development milestones toward the ultimate objective of scaling distributed RL to **3 Trillion (3T) parameter models**. Direct JAX FFI device-to-device transfers (`weight_synchronizer_ffi`) are strictly required on the **Trainer** side. Staging trainer weights via CPU memory (`HOST_STAGE`) causes fatal host Out-Of-Memory (OOM) errors and layout re-tiling bugs on larger models and is strictly forbidden on the Trainer.
>
> **Rollout Workers & Pathways / FFI Runtime Separation**:
> Rollout workers (`vLLM / tpu-inference`) execute on bare-metal TPU slices without Pathways due to runtime constraints and technical limitations in the inference stack. Because JAX FFI handlers currently rely on Pathways (`proxy`), rollout workers do not use Pathways (and thus do not run FFI, operating via the native C++ Raiden worker synchronizer).
>
> **Developer-Only Guardrails (DO NOT MERGE TO PRODUCTION)**:
> Automatic 2-hour worker timeouts (`activeDeadlineSeconds: 7200`), 10-minute cleanup TTLs (`ttlSecondsAfterFinished: 600`), and checkpoint bypasses (`DISABLE_CHECKPOINTING=true`) are strictly intended for rapid developer iteration to prevent quota leaks and save disk I/O. They are structured as standalone commits prefixed with `[DEV ONLY - DO NOT MERGE TO PROD]` and must NOT be propagated to production code.

---

## 2. Fast-Track: Running with Pre-Built Images

If you want to run immediately without building container images or compiling C++ extensions from scratch, use our verified pre-built images.

### Verified Images
- **Qwen3.5-35B-A3B Verified Runner Image**:
  ```text
  europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v12
  ```
- **Qwen3-0.6B Verified Runner Image**:
  ```text
  europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v5
  ```

### Step 1: Connect to the GKE Cluster
```bash
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --zone europe-west4-b \
  --project cloud-tpu-shared-capacity
```

### Step 2: Clone or Update the Launcher Repository
```bash
git clone https://github.com/google/tunix.git ~/git/tunix
cd ~/git/tunix
git fetch https://github.com/igorts-git/tunix.git igorts/qwen35-run
git checkout -B igorts/qwen35-run FETCH_HEAD
```

### Step 3: Launch Workloads

#### Option A: Run Qwen3.5-35B-A3B (Lean 1-Rollout Topology)
```bash
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh start \
  --model qwen3.5-35b \
  --rollout-replicas=1
```

#### Option B: Run Qwen3-0.6B Baseline
```bash
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh start \
  --model qwen3-0.6b \
  --rollout-replicas=1
```

### Step 4: Monitor and Inspect

```bash
# Check JobSet and pod status
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh status --model qwen3.5-35b

# Stream trainer logs (shows Pathways initialization, model loading, and step progress)
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs trainer -f --model qwen3.5-35b

# Stream rollout worker logs (shows vLLM generation and weight sync reception)
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs rollout -f --model qwen3.5-35b

# Stream orchestrator logs (shows GSM8K batching, reward computation, and registration)
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs orch -f --model qwen3.5-35b

# Run automated triage diagnosis if an error occurs
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh triage --model qwen3.5-35b

# Cleanly stop and delete all JobSets
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh stop --model qwen3.5-35b
```

---

## 3. Building From Scratch

Follow this section to build everything from clean git checkouts, cherry-pick necessary fixes, compile the Raiden C++ wheel, plug in Pathways images, and build the unified runner image.

### 3.1 Clone the Repositories
Create a clean workspace with the four required repositories:
```bash
mkdir -p ~/git && cd ~/git
git clone https://github.com/AI-Hypercomputer/maxtext.git
git clone https://github.com/google/tunix.git
git clone https://github.com/vllm-project/tpu-inference.git
git clone https://github.com/AI-Hypercomputer/tpu-sync.git
```

---

### 3.2 Required Changes & Cherry-Picks

Check upstream `origin/main` first. If any of these changes have already been merged into `main`, skip cherry-picking them. Otherwise, fetch and cherry-pick from the fork branches (`igorts-git/*`):

#### A. Repository: `tpu-inference` (`vllm-project/tpu-inference`)
Fork remote: `https://github.com/igorts-git/tpu-inference.git` (Branch: `igorts/qwen35-run`)

```bash
cd ~/git/tpu-inference
git fetch https://github.com/igorts-git/tpu-inference.git igorts/qwen35-run
```

| Commit | Description | Why It Is Needed |
| :--- | :--- | :--- |
| `f89fe0090` | `Add the FFI Raiden h2d path to the rollout worker` | Base rollout fix from Mohit: adds FFI rollout receiver with chunked H2D memory management and `refresh_model_state_leaves()` to prevent serving stale weights. |
| `8cabf13dc` | `fix(runner,quantization): compatibility fallback for nvfp4 and vllm kv cache interface` | Prevents import crashes in vLLM runner when optional quantization packages are missing. |
| `1fc80182a` | `fix(raiden): filter runtime kv cache parameters and sort array bindings` | Filters out non-trainable dynamic KV cache arrays during variable extraction so that parameter counts match the trainer exactly; sorts bindings alphabetically. |
| `29c6d12db` | `fix(raiden): route multi-shard rollout endpoints to distinct NUMA ports` | **Critical for TPU v5p multi-shard rollout**: In `TP=2` topologies, `tpu-sync` opens distinct sub-synchronizer ports per NUMA node. Routes each shard to its respective local port endpoint instead of all pointing to 20001. |

To cherry-pick on top of a clean `main`:
```bash
git cherry-pick f89fe0090 8cabf13dc 1fc80182a 29c6d12db
```

---

#### B. Repository: `tunix` (`google/tunix`)
Fork remote: `https://github.com/igorts-git/tunix.git` (Branch: `igorts/qwen35-run`)

```bash
cd ~/git/tunix
git fetch https://github.com/igorts-git/tunix.git igorts/qwen35-run
```

| Commit | Description | Why It Is Needed |
| :--- | :--- | :--- |
| `ab694ded` | `Raiden weight sync: transport selection, multihost shard indexing, per-replica jobs` | Base fix from Mohit: sets up transport selection, global device mesh indexing via `jnp.arange`, and per-replica job specifications. |
| `56d8fd53` | `fix(raiden): sort variable bindings alphabetically and route non-FFI shards to local endpoints` | Deterministic alphabetical variable sorting across both trainer and rollout; routes bare-metal rollout shards to distinct local NUMA endpoints. |
| `3fbeae71` | `[DEV ONLY - DO NOT MERGE TO PROD] feat(trainer): support DISABLE_CHECKPOINTING in run_trainer_node` | **Dev Only**: Allows bypassing Orbax checkpointing during rapid testing runs to save disk I/O and avoid proxy transfer overhead. |
| `a7feb9af` | `feat(launcher): add launch_raiden.sh operational tool and Dockerfile.maxtext` | Production operational launcher script for GKE TPU v5p clusters and container Dockerfile. |
| `7d1275d6` | `[DEV ONLY - DO NOT MERGE TO PROD] feat(distributed): add 2-hour activeDeadlineSeconds timeout and 10-minute ttlSecondsAfterFinished cleanup` | **Dev Only**: Safety timeout that terminates orphaned worker JobSets after 2 hours and deletes resources 10 minutes post-finish to prevent quota leakage during development. |

To cherry-pick on top of a clean `main`:
```bash
git cherry-pick ab694ded 56d8fd53 3fbeae71 a7feb9af 7d1275d6
```

---

#### C. Repository: `maxtext` (`AI-Hypercomputer/maxtext`)
Fork remote: `https://github.com/igorts-git/maxtext.git` (Branch: `igorts/qwen35-run`)

```bash
cd ~/git/maxtext
git fetch https://github.com/igorts-git/maxtext.git igorts/qwen35-run
```

| Commit | Description | Why It Is Needed |
| :--- | :--- | :--- |
| `551ea2ff7` | `Support inhomogeneous layer cycles when unscanning for Raiden weight sync` | Base fix from Mohit: unrolls interleaved repeating scanned blocks (1 dense layer + 3 MoE layers) into flat unscanned layers matching vLLM parameter trees. |
| `67ef9188b` | `feat(tunix): support abstract ShapeDtypeStruct unrolling in raiden_unscan` | Adds abstract `ShapeDtypeStruct` array support in `_slice_along_axis` for schedule dry-runs and tracing without allocating TPU memory. |
| `d771295b6` | `[DEV ONLY - DO NOT MERGE TO PROD] feat(engine): support DISABLE_CHECKPOINTING environment variable` | **Dev Only**: Allows `MaxTextTrainingEngine` to bypass Orbax checkpoint saves on step boundaries to accelerate developer iteration. |

To cherry-pick on top of a clean `main`:
```bash
git cherry-pick 551ea2ff7 67ef9188b d771295b6
```

---

### 3.3 Building the Raiden Wheel (`tpu-sync`)

The Raiden C++ extension (`tpu_raiden_jax`) provides the underlying zero-copy DMA engine and JAX FFI bindings.

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
To build the wheel from source:
```bash
cd ~/git/tpu-sync

# 1. Install Bazel 8.6.0 (or via Bazelisk)
sudo wget -O /usr/local/bin/bazel https://github.com/bazelbuild/bazel/releases/download/8.6.0/bazel-8.6.0-linux-x86_64
sudo chmod +x /usr/local/bin/bazel

# 2. Check out the latest known good commit
git checkout $(cat lkg.version)

# 3. Compile JAX extension wheel
./build.sh jax
```
The compiled wheel will be placed in `dist/`. Copy it to `~/git/tunix/.docker/tpu_sync/`:
```bash
mkdir -p ~/git/tunix/.docker/tpu_sync
cp dist/tpu_raiden_jax-*.whl ~/git/tunix/.docker/tpu_sync/
```

---

### 3.4 Pathways Container Images

Pathways runs a distributed resource manager (`pathways-rm`), proxy server (`pathways-proxy`), and TPU worker (`pathways-worker`) alongside user code.

The default verified images used by `launch_raiden.sh` are:
```bash
# Server / Worker image (runs on TPU hosts):
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_server:raiden_20260812"

# Proxy image (runs in user pod):
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/shauryag/unsanitized_proxy_server:raiden_20260812"
```

To plug in custom or newer Pathways server/proxy images:
- **Via environment variables before launching**:
  ```bash
  export PATHWAYS_SERVER_IMAGE="<YOUR_PATHWAYS_SERVER_IMAGE>"
  export PATHWAYS_PROXY_IMAGE="<YOUR_PATHWAYS_PROXY_IMAGE>"
  ./launch_raiden.sh start --model qwen3.5-35b
  ```
- **Via CLI flag in `yaml_generator.py`**:
  ```bash
  python -m tunix.experimental.distributed.deployment.yaml_generator \
    --pathways_server_image="<YOUR_IMAGE>" \
    --pathways_proxy_server_image="<YOUR_IMAGE>"
  ```

---

### 3.5 Building and Pushing the Docker Image

The unified runner image combines `tpu-inference`, `maxtext`, `tunix`, and the patched `tpu_sync` FFI extension on top of a verified base.

#### Step 1: Stage Checkouts into `tunix/.docker`
```bash
cd ~/git/tunix

mkdir -p .docker/tpu_inference .docker/maxtext .docker/tpu_sync

rsync -av --delete --exclude='.git' --exclude='venv' --exclude='.venv' \
  ~/git/tpu-inference/ .docker/tpu_inference/

rsync -av --delete --exclude='.git' --exclude='venv' --exclude='.venv' \
  ~/git/maxtext/ .docker/maxtext/
```

#### Step 2: Review `Dockerfile.maxtext`
The `Dockerfile.maxtext` applies required compatibility patches for JAX FFI and `PartitionSpec`:
```dockerfile
FROM gcr.io/cloud-tpu-multipod-dev/anisha-tmvp/anisha-0825:igorts-chunk4-v2

ENV PATH="/opt/venv/bin:$PATH"

# Reinstall the Raiden wheel with FFI support
COPY .docker/tpu_sync/*.whl /tmp/tpu_sync/
RUN pip install --force-reinstall --no-deps /tmp/tpu_sync/*.whl && rm -rf /tmp/tpu_sync

# JAX compute_on and FFI partition spec patches
RUN sed -i "s/def compute_on(compute_type: str):/def compute_on(compute_type: str, **kwargs):/" /opt/venv/lib/python3.12/site-packages/jax/_src/compute_on.py && \
    sed -i "s/, out_memory_spaces=jax.memory.Space.Device//" /opt/venv/lib/python3.12/site-packages/tpu_sync/frameworks/jax/weight_synchronizer_ffi.py && \
    python3 -c "import pathlib; p = pathlib.Path('/opt/venv/lib/python3.12/site-packages/tpu_sync/frameworks/jax/weight_synchronizer_ffi.py'); s = p.read_text(); old = '''  anchor_spec = jax.sharding.PartitionSpec(\n      *axis_names, *([None] * (len(device_array.shape) - len(axis_names)))\n  )'''; assert old in s, 'anchor_spec pattern not found'; s = s.replace(old, '  anchor_spec = device_array.sharding.spec'); p.write_text(s)"

# Install local tpu-inference
COPY .docker/tpu_inference /tpu-inference
RUN pip install --no-deps -e /tpu-inference

# Install local MaxText and vLLM adapter
COPY .docker/maxtext /maxtext
RUN pip install --no-deps -e /maxtext
RUN pip install --no-deps -e /maxtext/src/maxtext/integration/vllm

# Install local Tunix
WORKDIR /app
COPY . /app
RUN pip install --no-deps -e /app

CMD ["bash"]
```

#### Step 3: Build and Push
```bash
TAG="qwen35-$(date +%Y%m%d)-custom"
IMAGE="europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/${USER}-maxtext:${TAG}"

# Build
docker build -t "${IMAGE}" -f Dockerfile.maxtext .

# Authenticate Docker to Artifact Registry
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://europe-west4-docker.pkg.dev

# Push
docker push "${IMAGE}"
```

#### Step 4: Launch With Custom Image
```bash
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh start \
  --model qwen3.5-35b \
  --image "${IMAGE}" \
  --rollout-replicas=1
```

---

## 4. Model Configurations & Checkpoint References

| Configuration | Qwen3-0.6B | Qwen3.5-35B-A3B |
| :--- | :--- | :--- |
| **Model ID** | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3.5-35B-A3B` |
| **MaxText Model** | `qwen3-0.6b` | `qwen3.5-35b-a3b` |
| **Checkpoint Path** | `gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items` | `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items` |
| **Trainer Slice** | `tpuv5:2x2x2` (8 chips) | `tpuv5:2x2x2` (8 chips) |
| **Trainer Mesh** | `FSDP=8` | `FSDP=8` |
| **Rollout Slice** | `tpuv5:2x2x1` (4 chips) | `tpuv5:2x2x1` (4 chips) |
| **Rollout Mesh** | `TP=2` (unscanned) | `TP=2` (unscanned) |
| **Variables Count** | 227 variables | 673 variables |
| **Weight Sync Mode** | `raiden` (FFI direct D2D) | `raiden` (FFI direct D2D) |

---

## 5. Known Pitfalls & Troubleshooting

### 1. "Repetitive Newline / Gibberish Text Generation"
- **Cause**: Using `HOST_STAGE` or passing `skip_tiling=False` to Raiden-FFI. Device-to-host DMA on TPU naturally leaves 2D matrices in physical `(8, 128)` tiled layout. If `skip_tiling=False` is passed, the receiver re-tiles already-tiled memory, corrupting byte layouts.
- **Fix**: Leave `skip_tiling=None` in `RaidenTransferOptions` so `raiden_controller.py` automatically derives `skip_tiling=True` for FFI aligned transfers.

### 2. "Checksum Mismatch / Discrepancy on Shard 1"
- **Cause**: Rollout workers in `TP=2` listening on distinct ports per NUMA node, but advertising a single port to the coordinator in `metadata_dict()`.
- **Fix**: Ensure commit `f4508df2f` in `tpu-inference` is present so `get_local_endpoints()` registers distinct port endpoints for each shard.

### 3. "Host OOM or Proxy Deadlock During Weight Sync"
- **Cause**: Bypassing FFI and staging weights via client CPU memory or proxy RPCs.
- **Fix**: Verify `RaidenTransferOptions(parallelism=16)` is active without `HOST_STAGE`. MaxText engine will raise an error under Pathways if FFI is not detected.

### 4. "Duplicate Worker ID in Orchestrator"
- **Cause**: A previous worker crashed or restarted and attempted to register with the same `worker_id` before the orchestrator session cleaned up.
- **Fix**: Run `./launch_raiden.sh stop` to tear down stale JobSets before restarting.
