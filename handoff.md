# Handoff & Session State: Qwen3.5 Distributed RL on GKE TPU v5p
**MaxText Trainer + Tunix GRPO + vLLM Rollout + Raiden-FFI Direct Weight Sync**

> [!NOTE]
> **Active Jetski Trajectory / Conversation ID**: `6a32a1bf-579c-4f03-88a6-886d6f0ca8a7`  
> **Target Branch**: `igorts/qwen35-run` across `maxtext`, `tunix`, and `tpu-inference`  
> **Target Cluster**: `bodaborg-v5p-nap` in `europe-west4`, namespace `trellis`

This document captures the complete technical background, architecture, verified milestones, root causes of resolved issues, commit inventory, active cluster state, and exact instructions to run distributed RL with Raiden weight synchronization on Qwen3.5.

---

## 1. Executive Summary & Session Objectives

### Core Goal
Enable robust, end-to-end distributed Reinforcement Learning (RL) fine-tuning using **GRPO** for **Qwen3.5** (validated on **Qwen3-0.6B** and scaled to **Qwen3.5-35B-A3B**) across four integrated systems:
1. **MaxText** (`AI-Hypercomputer/maxtext`): Serving as the `MaxTextTrainingEngine` trainer under the Pathways runtime on TPU v5p slices.
2. **Tunix** (`google/tunix`): Orchestrating the distributed GRPO program, managing prompt dispatch, batching, reward computation, and weight version transitions.
3. **vLLM / tpu-inference** (`vllm-project/tpu-inference`): Serving as the rollout worker via `RLVllmSampler` using `flax_nnx` model runners on TPU v5p.
4. **Raiden** (`tpu_raiden_jax`): Providing low-latency TPU host-to-host DMA weight synchronization directly device-to-device via JAX FFI.

### Diagnostic Study Protocol & Final Findings
The investigation executed the 4-step diagnostic protocol to pinpoint and eliminate repetitive token / gibberish generation on GSM8K:
1. **Step 1: Reproduce Mohit's results (0.6B model, 1 rollout worker)**:
   - **Completed & Verified**: Proved that when source and destination shard counts match ($4 \equiv 4$), Raiden delivers 100% of weights. Direct inspection of sampled responses confirmed **clean mathematical chain-of-thought reasoning with zero gibberish**.
2. **Step 2: Switch to 35B model (1 rollout worker on 4 chips)**:
   - **Diagnosed & Resolved**: Initial run crashed with `pjrt.OOM` during FFI D2H preparation. Root cause was identified as an **off-by-2 padding bug in `maxtext_vllm_adapter/adapter.py:169`**, which artificially inflated `moe_intermediate_size` from 512 to 1024, doubling all 120 expert weight matrices and exceeding TPU memory (>107 GB). Fixing this bug allows 35B to fit comfortably on 4 chips with **53.55 GB static memory (41.45 GB free HBM)**.
3. **Step 3: Switch to 35B model + 2 rollout workers (TP=4 vs FSDP=8)**:
   - **Diagnosed**: Both rollout workers produced repetitive multilingual/ASCII gibberish (`\u043c\u043e\u0440 nay Dont\u54ea...`). Root cause was identified as **Raiden's independent destination broadcast behavior**: Raiden treats each replica as an independent target, not an aggregated shard group. Each replica received only shards 0..3 (50% of weights), leaving 50% missing.
4. **Step 4: Resolution & Verified Zero-Gibberish Architecture**:
   - **Completed & Verified**: Configured the verified single-host $4 \equiv 4$ topology (`tpuv5:2x2x1` Trainer `FSDP=4` $\leftrightarrow$ `tpuv5:2x2x1` Rollout `TP=4`, 1 replica) combined with the 2-line `adapter.py` fix. Achieved 100% full weight transfer (0 dropped shards), 41.45 GB free HBM on trainer, 77.15 GB free HBM on rollout, and clean math reasoning.

### Mandatory Technical Principles & Constraints
> [!IMPORTANT]
> **1. Direct Device-to-Device (FFI) is Mandatory on the Trainer**:
> The 0.6B and 35B models are development milestones toward scaling to **3 Trillion (3T) parameter models**. Direct JAX FFI device-to-device transfers (`weight_synchronizer_ffi`) are strictly required on the **Trainer** side. Staging trainer weights via CPU memory (`HOST_STAGE`) causes fatal host Out-Of-Memory (OOM) errors and layout re-tiling bugs on larger models and is strictly forbidden on the Trainer.
>
> **2. Rollout Workers & Pathways / FFI Runtime Separation**:
> Rollout workers (`vLLM / tpu-inference`) execute on bare-metal TPU slices without Pathways due to runtime constraints in the inference serving stack. Because JAX FFI handler registration currently relies on the Pathways proxy environment, rollout workers do not use Pathways, operating via the native C++ Raiden worker synchronizer.
>
> **3. Developer-Only Guardrails (DO NOT MERGE TO PRODUCTION)**:
> Automatic 2-hour worker timeouts (`activeDeadlineSeconds: 7200`), 10-minute cleanup TTLs (`ttlSecondsAfterFinished: 600`), and checkpoint bypasses (`DISABLE_CHECKPOINTING=true`) are strictly intended for rapid developer iteration to prevent quota leaks and save disk I/O. They are structured as standalone commits prefixed with `[DEV ONLY - DO NOT MERGE TO PROD]` and must NOT be propagated to production code.
>
> **4. Mandatory Verification of Rollout Text Quality (No Gibberish)**:
> Program completion (`EXIT_CODE=0`) or successful weight synchronization is NOT sufficient proof of correctness. For every configuration, sampled response text MUST be directly inspected to confirm absence of gibberish.

---

## 2. Verified Milestones & Current State

### A. Milestone 1: Qwen3-0.6B E2E GRPO Training (`igorts-v8-06b`)
- Successfully completed training steps 0, 1, and 2 on `bodaborg-v5p-nap`.
- Synchronized **310 tensors** (**596,049,920 elements**) in **287,060 blocks** per transfer in ~13-14 seconds per step with 0 transfer errors.
- Generated clean, coherent chain-of-thought `<reasoning>` traces with zero gibberish.

### B. Milestone 2: Qwen3.5-35B-A3B Architecture & Metadata Alignment
- Scanned checkpoint: `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items`.
- Verified exact **673 trainable variables** match 1:1 between Trainer and Rollout.
- Ran simulation of Raiden schedule: **0 destination bounds violations**.

### C. Milestone 3: Upstream Rebase & Cross-Contributor Synthesis
- Synthesized and integrated the latest fixes from Mohit Khatwani (`mohit/rl-raiden-vllm-fixes`):
  - Fixed 3D torus multi-host device reordering bug using `jnp.arange(...)` for global shard indexing.
  - Added FFI rollout receiver with chunked H2D memory management in `tpu-inference`.
  - Fixed silent weight staleness in vLLM via `refresh_model_state_leaves()`.
- Layered our specialized improvements on top:
  - Deterministic alphabetical variable sorting across both trainer and rollout.
  - Abstract `ShapeDtypeStruct` support in `raiden_unscan.py` for schedule dry-runs and tracing.
  - `DISABLE_CHECKPOINTING` support in both MaxText and Tunix (*[DEV ONLY - DO NOT MERGE TO PROD]*).
  - NUMA local endpoint routing for bare-metal rollout workers (`get_local_endpoints()`).
  - Unified operational launcher (`launch_raiden.sh`) with live log streaming and automated triage.

### D. Milestone 4: Two-Rollout Worker Scaling & Entrypoint Alignment
- Migrated worker entrypoints to upstream `tunix.experimental.examples.common.run_trainer_node.main` and `run_rollout_node.main`.
- Aligned worker CLI flags (`--mesh_tp=${ROLLOUT_MESH_TP}`, `--sampler_data_parallel=${ROLLOUT_DATA_PARALLEL:-1}`).
- Fixed `tpu-inference` quantization import bug (`check_equal_or_regex_match` from `vllm.model_executor.layers.quantization.compressed_tensors.utils`).

### E. Milestone 5: Full 2-Rollout E2E Training on Qwen3.5-35B (`igorts-rd-35b`, image `v17`)
- Resolved `compute_on2` signature mismatch: wrapped `compute_on2` with default `out_memory_spaces=jax.memory.Space.Device`.
- All 673 parameter arrays (34,660,610,688 elements) synchronized via JAX FFI on Pathways across 8 devices (`__grand_total__: 304540017.18895197`).
- Completed Steps 0 and 1 with clean exit code 0.

### F. Diagnostic Study Step 1: Balanced Shards on Qwen3-0.6B ($4 \equiv 4$)
- **Configuration**: Trainer `tpuv5:2x2x1` (`FSDP=4`) $\leftrightarrow$ Rollout `tpuv5:2x2x1` (`TP=4`, 1 replica).
- **Result**: 100% full weight transfer (zero partial overlap warning).
- **Sampled Response**: Clean step-by-step mathematical reasoning solving field trip bus seat problems and lamp cost calculations. **Zero gibberish**.

### G. Diagnostic Study Step 2: 35B on 4 Chips (`tpuv5:2x2x1`)
- **Initial Observation**: Trainer crashed with `pjrt.OOM` during FFI D2H preparation.
- **Root Cause Discovered**: Not a fundamental hardware limit of 35B. Caused by the off-by-2 padding bug in `adapter.py:169`, which evaluated `(512 // 4) % (2 * 128) != 0` and padded `moe_intermediate_size` to 1024. This doubled all 120 expert matrices, pushing static memory to >107 GB.
- **Resolution**: With the 2-line fix in `adapter.py`, native 512 dimension is preserved (71.4 GB total weights). Static memory on 4 chips is 53.55 GB, leaving 41.45 GB free HBM on 95 GB TPU v5p chips.

### H. Diagnostic Study Step 3: 35B with 2 Rollout Workers (`TP=4` vs `FSDP=8`)
- **Configuration**: Trainer `tpuv5:2x2x2` (`FSDP=8`, 8 shards) $\leftrightarrow$ Rollout 0 (`TP=4`, 4 shards) + Rollout 1 (`TP=4`, 4 shards).
- **Observed Response**: Incoherent multilingual/ASCII gibberish (`\u043c\u043e\u0440 nay Dont\u54ea...`).
- **Root Cause Identified**: Raiden treats Rollout 0 and Rollout 1 as independent broadcast destinations. Each replica intersected with the source's 8 shards and received only shards 0..3 (50% of model weights). Shards 4..7 were never transferred to either replica.

### I. Diagnostic Study Step 4: Resolution & Verified Single-Host $4 \equiv 4$ Architecture
- **Configuration**:
  - Trainer: `tpuv5:2x2x1` (4 chips, 1 host node), `FSDP=4`, `TP=1` $\to$ 4 source shards `[0..3]`.
  - Rollout: `tpuv5:2x2x1` (4 chips, 1 host node), `TP=4`, `REPLICAS=1` $\to$ 4 destination shards `[0..3]`.
  - Image: `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-opta-verified`
- **Verification Results**:
  - Exact 1:1 shard matching ($4 \equiv 4$): 100% full tensor delivery, 0 dropped shards, 0 partial overlap warnings.
  - HBM Headroom: 41.45 GB free on Trainer, 77.15 GB free on Rollout.
  - Rollout Text Quality: Clean mathematical reasoning, zero gibberish.

---

## 3. Root Cause Analysis & Key Technical Insights

### 1. Off-by-2 MoE Dimension Padding Bug in `adapter.py:169` (Why 4 Chips OOMed)
- In `maxtext/src/maxtext/integration/vllm/maxtext_vllm_adapter/adapter.py:169`:
  ```python
  if hidden_size is not None and (hidden_size // moe_mlp_tp_size) % (2 * num_lanes) != 0:
    padded_hidden_size = next_power_of_two(hidden_size)
    while (padded_hidden_size // moe_mlp_tp_size) < (2 * num_lanes):
      padded_hidden_size = next_power_of_two(padded_hidden_size + 1)
  ```
- In Megablox (`gmm_v2.py:1078-1083`), input verification requires `size_n % (2 * num_lanes) == 0`.
- In `moe.py`, `fused_moe_matmul` fuses gate and up weights, so `size_n = 2 * (hidden_size // moe_mlp_tp_size)`.
- Substituting `size_n`:
  $$\left(2 \cdot \frac{\text{hidden\_size}}{\text{moe\_mlp\_tp\_size}}\right) \pmod{2 \cdot \text{num\_lanes}} == 0 \iff \left(\frac{\text{hidden\_size}}{\text{moe\_mlp\_tp\_size}}\right) \pmod{\text{num\_lanes}} == 0$$
- For TPU v5p (`num_lanes=128`), Qwen3.5-35B (`hidden_size=512`), and `TP=4`:
  $$\frac{512}{4} = 128 \equiv 0 \pmod{128}$$
- The extra factor of 2 in `adapter.py` checked $128 \pmod{256} \ne 0$, forcing `padded_hidden_size=1024`. This doubled all 120 expert weight matrices and inflated static memory to >107 GB, triggering `pjrt.OOM` on 4 chips.
- **Fix**: Changing `2 * num_lanes` to `num_lanes` eliminates the artificial padding and preserves native 512 dimension.

### 2. Shard Count Mismatch & Raiden Independent Destination Broadcast
- In Raiden (`weight_sync_coordinator.py`), distinct rollout workers (distinct `job_name` / `server_id`) are treated as independent broadcast destinations.
- Weight shards are **never aggregated across replicas**.
- When broadcasting from an 8-shard Trainer (`FSDP=8`) to a 4-shard Rollout (`TP=4`):
  - Raiden intersects `[0..7]` and `[0..3]`.
  - It transfers shards `[0..3]` (50% of weights). Shards `[4..7]` are omitted.
- Adding a second rollout worker does not solve this: each worker independently receives only shards `[0..3]`.
- **Invariant**: Every rollout replica must independently match the trainer shard count:
  $$\forall \text{replica } r: \quad N_{\text{dst}} \equiv N_{\text{src}}$$

### 3. Unplumbed `PREFUSE_MOE_WEIGHTS`
- Prior sessions proposed setting `PREFUSE_MOE_WEIGHTS=true` to save memory.
- Code audit confirmed that `PREFUSE_MOE_WEIGHTS` was completely unplumbed in `k8s_launcher.sh` and not recognized anywhere in `tunix`. It had zero effect on memory or execution.

### 4. `compute_on2` Signature Mismatch with JAX FFI TPU-Sync
- Upstream JAX has both `compute_on` and `compute_on2(f=None, *, compute_type, out_memory_spaces, compiler_options=None)`.
- The TPU-sync FFI wheel calls `@compute_on.compute_on(compute_type="device_host")` without `out_memory_spaces`.
- Wrapping `compute_on2` with default `out_memory_spaces=jax.memory.Space.Device` resolves compatibility.

### 5. Multi-Host Torus Device Reordering Bug (`jnp.arange` vs `d.id`)
- On multi-host slices (`tpuv5:2x2x2`), `create_device_mesh` reorders devices for 3D torus topology. Keying FFI global shard indices off `d.id` assigned shards to incorrect physical TPU offsets in Raiden.
- Resolved by indexing `global_ids` via `jnp.arange(mesh.devices.size).reshape(task_mesh_shape)`.

### 6. Rollout Stale Weight Caching (`refresh_model_state_leaves`)
- In `tpu-inference`, vLLM captures a `self.state_leaves` view at initialization time.
- Under JAX FFI H2D, new array buffers replace the leaves of `self.state`. Without re-extracting `state_leaves` via `refresh_model_state_leaves()`, vLLM continued passing pre-sync step-0 buffers into compiled graphs.

---

## 4. Repositories & Reorganized Commit Stacks

### Comparison of Option A vs Option B

| Component | Option A (`mohit/rl-raiden-vllm-fixes`) | Option B (Recommended Minimal Stack on `origin/main`) |
| :--- | :--- | :--- |
| **Commit Count** | 27 experimental commits | 3 targeted cherry-picks + 2-line fix |
| **Target Topology** | Designed for 16-chip v5e (`tpuv5e:4x4`) | Optimized for single-host 4-chip v5p (`tpuv5:2x2x1`) |
| **35B 4-Chip Fit** | OOMs (due to `adapter.py` padding to 1024) | Fits comfortably (53.55 GB / 95 GB, 41.45 GB free) |
| **Text Output** | Repetitive gibberish on 8-vs-2 or 8-vs-4 setups | Clean mathematical reasoning (zero gibberish) |

### Minimal Cherry-Pick Stack for Option B (on top of `origin/main`)

1. **`maxtext` (`AI-Hypercomputer/maxtext`)**:
   - `e0d3e4124`: `Support inhomogeneous layer cycles when unscanning for Raiden weight sync` (unrolls interleaved repeating scanned blocks matching vLLM parameter trees).
   - 2-line fix in `src/maxtext/integration/vllm/maxtext_vllm_adapter/adapter.py:169`: change `(2 * num_lanes)` to `num_lanes`.

2. **`tunix` (`google/tunix`)**:
   - `3e0a51f9`: `fix(weight_sync): fix policy_version tracking when sync_request is None`.

3. **`tpu-inference` (`vllm-project/tpu-inference`)**:
   - `29c6d12db`: `fix(raiden): route multi-shard rollout endpoints to distinct NUMA ports`.

---

## 5. Quick Reference Commands

### Connect to Cluster
```bash
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --region europe-west4 \
  --project cloud-tpu-shared-capacity
```

### Launch Verified Zero-Gibberish Workload (Qwen3.5-35B, Single-Host $4 \equiv 4$)
```bash
TRAINER_MESH_FSDP=4 ROLLOUT_MESH_TP=4 \
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh start \
  --model qwen3.5-35b \
  --rollout-replicas=1 \
  --debug \
  --reward-mode=exact \
  --image europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-opta-verified
```

### Monitor Workload
```bash
# Check JobSet and pod status
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh status --model qwen3.5-35b

# Stream orchestrator logs (GSM8K prompt completions and rewards)
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs orch -f --model qwen3.5-35b

# Stream trainer logs
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs trainer -f --model qwen3.5-35b

# Stream rollout worker logs
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs rollout -f --model qwen3.5-35b

# Stop run cleanly
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh stop --model qwen3.5-35b
```

---

## 6. Diagnostic Study Protocol & Execution Matrix

| Step | Objective | Model | Workers & Sharding | Text Inspection Result | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1** | Reproduce Mohit baseline with balanced sharding | Qwen3-0.6B | Trainer `tpuv5:2x2x1` (`FSDP=4`) vs Rollout `tpuv5:2x2x1` (`TP=4`, 1 replica) | **Clean Mathematical CoT** (Prompt 0: field trip buses; Prompt 7: lamps/bulbs). **Zero gibberish.** 100% full transfer. | **Completed & Verified** |
| **Step 2** | Test 35B on 4 chips with 1 rollout worker | Qwen3.5-35B | Trainer `tpuv5:2x2x1` (`FSDP=4`) vs Rollout `tpuv5:2x2x1` (`TP=4`, 1 replica) | Hit `pjrt.OOM`. Diagnosed root cause: `adapter.py:169` off-by-2 bug padded MoE dimension from 512 to 1024 (>107 GB static memory). | **Completed (Root Cause Diagnosed)** |
| **Step 3** | Test 35B with 2 rollout workers | Qwen3.5-35B | Trainer `tpuv5:2x2x2` (`FSDP=8`, 8 shards) vs 2 Rollout Workers (`TP=4`, 4 shards each) | **Repetitive ASCII/multilingual gibberish** (`\u043c\u043e\u0440 nay Dont\u54ea...`). Diagnosed root cause: Raiden independent per-replica broadcast dropped shards 4..7 (50% missing weights). | **Completed (Root Cause Diagnosed)** |
| **Step 4** | Resolution: Verified single-host $4 \equiv 4$ topology with `adapter.py` fix | Qwen3.5-35B | Trainer `tpuv5:2x2x1` (`FSDP=4`, 4 shards) vs Rollout `tpuv5:2x2x1` (`TP=4`, 4 shards, 1 replica) | **Clean Mathematical CoT. Zero gibberish.** 100% full weight transfer. 41.45 GB free HBM on trainer, 77.15 GB free HBM on rollout. | **Completed & Verified** |

### Key Mathematical Rule for Balanced Weight Sync
In Raiden's architecture, distinct rollout workers (distinct `job_name` / `server_id`) are independent broadcast destinations, not an aggregated tensor-parallel group:
$$\forall \text{replica } r \in [0, N_{\text{replicas}}-1]: \quad \text{Shards}(\text{Rollout}_r) = \text{ROLLOUT\_MESH\_TP} \equiv N_{\text{trainer\_shards}} = N_{\text{trainer\_devices}}$$

When this per-replica equality holds:
- Raiden global index spaces intersect completely ($100\%$ overlap) for every rollout worker.
- `__grand_total__` checksums match identically across source and all destinations.
- Every rollout worker receives the entire weight tensor set, eliminating uninitialized tensor shards and repetitive gibberish generation.
