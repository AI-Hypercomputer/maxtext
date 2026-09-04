# Handoff & Session State: Qwen3.5 Distributed RL on GKE TPU v5p
**MaxText Trainer + Tunix GRPO + vLLM Rollout + Raiden-FFI Direct Weight Sync**

This document captures the complete technical background, architecture, verified milestones, root causes of resolved issues, commit inventory, active cluster state, and exact instructions to resume and scale distributed RL runs with Raiden weight synchronization.

---

## 1. Executive Summary & Session Objectives

### Core Goal
Enable robust, end-to-end distributed Reinforcement Learning (RL) fine-tuning using **GRPO** for **Qwen3.5** (validated on **Qwen3-0.6B**, with architecture prepared for **Qwen3.5-35B-A3B**) across four integrated systems:
1. **MaxText** (`AI-Hypercomputer/maxtext`): Serving as the `MaxTextTrainingEngine` trainer under the Pathways runtime on TPU v5p slices.
2. **Tunix** (`google/tunix`): Orchestrating the distributed GRPO program, managing prompt dispatch, batching, reward computation, and weight version transitions.
3. **vLLM / tpu-inference** (`vllm-project/tpu-inference`): Serving as the rollout worker via `RLVllmSampler` using `flax_nnx` model runners on TPU v5p.
4. **Raiden** (`tpu_raiden_jax`): Providing low-latency TPU host-to-host DMA weight synchronization directly device-to-device via JAX FFI.

### Mandatory Technical Principles & Constraints
> [!IMPORTANT]
> **1. Development Scale vs. Ultimate Target (Scaling to 3T Models)**:
> The **0.6B** and **35B** models are strictly temporary development milestones to validate functionality, synchronization correctness, and distributed runtime orchestration. The ultimate goal of the team is to scale distributed RL to **3 Trillion (3T) parameter models**.
>
> **2. Direct Device-to-Device (FFI) is Mandatory on the Trainer**:
> Because the target is multi-trillion parameter scaling, direct **JAX FFI device-to-device synchronization** (`weight_synchronizer_ffi`) is a strict requirement on the **Trainer** side. Staging trainer weights through client host memory (`HOST_STAGE`) under Pathways inevitably triggers fatal **Host Out-Of-Memory (OOM)** errors, proxy transfer timeouts, and byte layout permutation bugs. Bypassing FFI or falling back to CPU memory staging on the Trainer is completely unacceptable.
>
> **3. Rollout Workers & Pathways / FFI Runtime Separation**:
> While FFI is mandatory on the Trainer, the **Rollout workers** (`vLLM / tpu-inference`) execute on bare-metal TPU slices without Pathways due to technical reasons and runtime limitations within the inference serving stack. Because JAX FFI handler registration currently relies on the Pathways proxy environment, rollout workers do not use Pathways (and thus do not run FFI, operating instead via the native C++ Raiden worker synchronizer). The distributed setup must cleanly bridge the FFI Trainer with bare-metal rollout workers without forcing rollout into Pathways or compromising Trainer FFI scalability.
> 
> **4. Rigorous Bug Reporting & Isolated Reproductions**:
> If any component (such as FFI, tpu-sync, or Pathways) fails or behaves unexpectedly, we must NOT implement hacky bypasses that compromise long-term scalability. Instead:
> - File clear, easy-to-reproduce bug reports for the corresponding component teams (Pathways, TPU Sync, or Compiler).
> - Provide standalone minimal reproduction scripts that isolate the failing behavior **without** pulling in the full complex integration across `tunix` / `maxtext` / `tpu-inference` / `vllm` / `tpu-sync`.
>
> **5. Developer-Only Ephemeral Guardrails (DO NOT MERGE TO PRODUCTION)**:
> Features such as automatic 2-hour worker timeouts (`activeDeadlineSeconds: 7200`), 10-minute cleanup TTLs (`ttlSecondsAfterFinished: 600`), and checkpoint disabling (`DISABLE_CHECKPOINTING=true`) are strictly temporary development scaffolds designed to protect quota and accelerate debugging. They exist as clearly marked standalone commits (`[DEV ONLY - DO NOT MERGE TO PROD]`) and MUST NOT propagate into production branches or releases.

---

## 2. Verified Milestones & Current State

### A. Milestone 1: Qwen3-0.6B E2E GRPO Training (`igorts-v8-06b`)
- Successfully completed training steps 0, 1, and 2 on `bodaborg-v5p-nap`.
- Dual-node Pathways trainer (`FSDP=8`, 2 host nodes, 8 chips).
- Single-node vLLM rollout (`TP=2`, 4 chips).
- Synchronized **310 tensors** (**596,049,920 elements**) in **287,060 blocks** per transfer in ~13-14 seconds per step with 0 transfer errors.
- Generated clean, coherent chain-of-thought `<reasoning>` traces with zero gibberish.

### B. Milestone 2: Qwen3.5-35B-A3B Architecture & Metadata Alignment
- Scanned checkpoint: `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items`.
- Verified exact **673 trainable variables** match 1:1 between Trainer (`FSDP=8`) and Rollout (`TP=2`, unscanned).
- Ran simulation of Raiden schedule: **0 destination bounds violations**.
- Clean container image built from source git checkouts:
  ```text
  europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v12
  ```

### C. Milestone 3: Upstream Rebase & Cross-Contributor Synthesis
- All three repositories (`maxtext`, `tunix`, `tpu-inference`) were rebased cleanly onto the latest `origin/main`.
- Synthesized and integrated the latest fixes from colleague Mohit Khatwani (`mohit/rl-raiden-vllm-fixes`) as foundational base commits:
  - Fixed 3D torus multi-host device reordering bug using `jnp.arange(...)` for global shard indexing.
  - Added FFI rollout receiver with chunked H2D memory management (`RAIDEN_H2D_CHUNK_BYTES`) in `tpu-inference`.
  - Fixed silent weight staleness in vLLM via `refresh_model_state_leaves()`.
  - Added mesh rank reduction (`_reduce_mesh`) to drop singleton axes for FFI `shard_map`.
- Layered our specialized improvements on top:
  - Deterministic alphabetical variable sorting (`sorted(zip(names, arrays))`) across both trainer and rollout.
  - Abstract `ShapeDtypeStruct` support in `raiden_unscan.py` for schedule dry-runs and tracing.
  - `DISABLE_CHECKPOINTING` support in both MaxText and Tunix to eliminate disk I/O bottlenecks (*[DEV ONLY - DO NOT MERGE TO PROD]*).
  - NUMA local endpoint routing for bare-metal non-FFI rollout workers (`get_local_endpoints()`).
  - Unified operational launcher (`launch_raiden.sh`) with live log streaming and automated triage.
  - Automatic 2-hour worker self-destruction (`activeDeadlineSeconds: 7200`) and 10-minute cleanup (`ttlSecondsAfterFinished: 600`) on all JobSets as a safety guard against abandoned development workloads (*[DEV ONLY - DO NOT MERGE TO PROD]*).

### D. Current Cluster & Workload State
- Cluster `bodaborg-v5p-nap` was cleanly purged of all previous test runs (`igorts-rd-35b`).
- No active or pending jobs for user `igorts` are currently running on the cluster.
- All YAML generators and templates are armed with a 2-hour `activeDeadlineSeconds` self-destruct timeout and 10-minute post-completion TTL cleanup.
- Ready to launch fresh workloads when required.

---

## 3. Root Cause Analysis & Key Technical Insights

### 1. Why `skip_tiling=False` Produces Repetitive Newline Gibberish
- On TPU, device-to-host DMA transfers 2D tensors directly in hardware tiled memory layout (typically `(8, 128)`).
- When Raiden-FFI is used without CPU staging, memory remains tiled.
- If `skip_tiling=False` is passed, the receiver mistakenly assumes linear memory and attempts to tile already-tiled memory, corrupting byte layouts and causing the LLM to emit repetitive newline (`\n\n\n...`) tokens.
- Default `RaidenTransferOptions(parallelism=16)` leaves `skip_tiling=None`, allowing `raiden_controller.py` to automatically deduce `skip_tiling=True` for FFI aligned transfers.

### 2. Multi-Host Torus Device Reordering Bug (`jnp.arange` vs `d.id`)
- On single-host slices (`tpuv5:2x2x1`, 4 chips), device IDs are sequential `[0, 1, 2, 3]`.
- On multi-host slices (`tpuv5:2x2x2`, 8 chips across 2 hosts), `jax.experimental.mesh_utils.create_device_mesh` reorders devices for 3D torus topology (e.g. `[0, 1, 3, 2, 6, 7, 5, 4]`).
- Keying FFI global shard indices off `d.id` assigned shards to incorrect physical TPU offsets in Raiden's native layer.
- Resolved by indexing `global_ids` via `jnp.arange(mesh.devices.size).reshape(task_mesh_shape)`.

### 3. Rollout Stale Weight Caching (`refresh_model_state_leaves`)
- In `tpu-inference`, vLLM captures a `self.state_leaves` view at initialization time.
- Under JAX FFI H2D, new array buffers are created and replace the leaves of `self.state`.
- Without re-extracting `state_leaves` via `refresh_model_state_leaves()`, vLLM's `model_fn` continued passing pre-sync step-0 buffers into compiled execution graphs, resulting in silent weight staleness despite reporting successful syncs.

### 4. Why Shard 1 Encountered Corrupted / Missing Weights on Bare-Metal Rollout
- In multi-slice or multi-shard rollout topologies (`TP=2`), `NumaAwareWeightSynchronizer` creates a sub-synchronizer per NUMA node / port.
- Previously, `metadata_dict()` in `raiden_worker_sync.py` advertised `[f"{ip}:{local_port}"] * num_shards`, pointing all shards to port 20001 (sub-synchronizer 0).
- Sub-synchronizer 0 rejected or misrouted pushes destined for shard 1, producing a ~4x checksum discrepancy.
- Resolved by using `self._sync.get_local_endpoints()` to assign each shard index to its distinct NUMA port endpoint.

### 5. Why Pathways Miscomputed `devices_per_host`
- In non-proxy JAX, `num_processes` is computed by counting unique `device.process_index`.
- Under Pathways proxy runtime, all devices report `process_index=0`.
- As a result, `devices_per_host = 8 // 1 = 8` was computed instead of `4`, breaking slice offset calculations.
- Resolved by resolving hosts via `device.task_id` / `device.host_id` and supporting explicit `RAIDEN_DEVICES_PER_HOST` overrides.

### 6. Interleaved Scanned Layers in Qwen3.5-35B-A3B & Abstract Array Support
- Qwen3.5-35B-A3B groups layers into repeating cycles (1 dense layer + 3 MoE layers, grouped into blocks).
- Standard single-axis unscanning fails because layer blocks do not map 1:1 to continuous layer indices.
- Resolved in `raiden_unscan.py` with `cycle_interval` unrolling. Furthermore, `_slice_along_axis` supports `jax.ShapeDtypeStruct` abstract arrays to allow schedule tracing and shape verification without TPU memory allocation.

---

## 4. Repositories & Reorganized Commit Stacks

All repositories have been cleanly rebased on the latest `origin/main` with external contributor commits (Mohit Khatwani) at the base and our refined commits stacked on top:

### A. `maxtext` (`AI-Hypercomputer/maxtext` branch `igorts/qwen35-run`)
```text
d6dbe5fce (HEAD) docs: update operational manual and session handoff with dev-only guardrails
d771295b6 [DEV ONLY] [Ours]  feat(engine): support DISABLE_CHECKPOINTING environment variable
67ef9188b            [Ours]  feat(tunix): support abstract ShapeDtypeStruct unrolling in raiden_unscan
551ea2ff7            [Mohit] Support inhomogeneous layer cycles when unscanning for Raiden weight sync
----------------------------------------------------------------------------------------------------
4cfec5b20 (origin/main)
```

### B. `tpu-inference` (`vllm-project/tpu-inference` branch `igorts/qwen35-run`)
```text
29c6d12db (HEAD) fix(raiden): route multi-shard rollout endpoints to distinct NUMA ports
1fc80182a [Ours]  fix(raiden): filter runtime kv cache parameters and sort array bindings
8cabf13dc [Ours]  fix(runner,quantization): compatibility fallback for nvfp4 and vllm kv cache interface
f89fe0090 [Mohit] Add the FFI Raiden h2d path to the rollout worker
----------------------------------------------------------------------------------------------------
c82492756 (origin/main, already merged PR #3516)
```

### C. `tunix` (`google/tunix` branch `igorts/qwen35-run`)
```text
7d1275d6 (HEAD) [DEV ONLY] feat(distributed): add 2-hour activeDeadlineSeconds timeout and 10-minute ttlSecondsAfterFinished cleanup
a7feb9af                   feat(launcher): add launch_raiden.sh operational tool and Dockerfile.maxtext
3fbeae71        [DEV ONLY] feat(trainer): support DISABLE_CHECKPOINTING in run_trainer_node
56d8fd53                   fix(raiden): sort variable bindings alphabetically and route non-FFI shards to local endpoints
ab694ded                   Raiden weight sync: transport selection, multihost shard indexing, per-replica jobs
----------------------------------------------------------------------------------------------------
24827016 (origin/main)
```

---

## 5. Quick Reference Commands

### Connect to Cluster
```bash
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --region europe-west4 \
  --project cloud-tpu-shared-capacity
```

### SSH Key Setup (for GitHub git operations)
```bash
export SSH_AUTH_SOCK=~/.tmp/.${USER}.ssh_auth_sock
```

### Monitoring the Current Run (`igorts-rd-35b`)
```bash
# Check JobSet and pod status
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh status --model qwen3.5-35b

# Check Kueue TPU quota reservation
kubectl describe clusterqueue default | grep -A 10 "tpu-v5p-flavor"

# Stream orchestrator logs
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs orch -f --model qwen3.5-35b

# Stream trainer logs (once admitted)
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs trainer -f --model qwen3.5-35b

# Stream rollout logs (once admitted)
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh logs rollout -f --model qwen3.5-35b

# Stop run cleanly
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh stop --model qwen3.5-35b
```

### Starting a Fresh 35B Workload
```bash
./tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh start \
  --model qwen3.5-35b \
  --rollout-replicas=1 \
  --image europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v12
```

### Full Operational Documentation
For the complete guide on building container images from scratch, compiling the Raiden C++ wheel, cherry-pick tables, and Pathways server/proxy images, refer to:
[`qwen3.5_instructions.md`](file:///usr/local/google/home/igorts/git/maxtext/qwen3.5_instructions.md)
