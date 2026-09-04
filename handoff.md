# Handoff & Session State: Qwen3.5 Distributed RL on GKE TPU v5p

This document captures the complete background, architecture, verified end-to-end milestones, debugging history, repository diffs, mandatory technical constraints, and exact instructions to resume and scale distributed RL runs with Raiden weight synchronization.

---

## 1. Executive Summary & Session Objectives

### Core Goal
Enable robust, end-to-end distributed Reinforcement Learning (RL) fine-tuning using **GRPO** for **Qwen3.5** (validated on **Qwen3-0.6B**, with architecture prepared for **Qwen3.5-35B-A3B**) across four integrated systems:
1. **MaxText** (`AI-Hypercomputer/maxtext`): Serving as the `MaxTextTrainingEngine` trainer under the Pathways runtime on TPU v5p slices.
2. **Tunix** (`google/tunix`): Orchestrating the distributed GRPO program, managing prompt dispatch, batching, reward computation, and weight version transitions.
3. **vLLM / tpu-inference** (`vllm-project/tpu-inference`): Serving as the rollout worker via `RLVllmSampler` using `flax_nnx` model runners on TPU v5p.
4. **Raiden** (`tpu_raiden_jax`): Providing low-latency TPU host-to-host DMA weight synchronization between trainer and rollout workers over the control/data plane.

### Mandatory Technical Principles & Constraints
> [!IMPORTANT]
> **1. Usage of FFI is Mandatory for Success**:
> Without the JAX FFI backend (`weight_synchronizer_ffi`), we cannot scale the system to larger models (like Qwen3.5-35B). Non-FFI staging routes entire weight trees through client host memory or proxy runtime, which inevitably triggers fatal **Host Out-Of-Memory (OOM)** errors and proxy transfer timeouts. Hacking around or changing the architectural goals is not acceptable.
> 
> **2. Rigorous Bug Reporting & Isolated Reproductions**:
> If any component (such as FFI, tpu-sync, or Pathways) fails or behaves unexpectedly, we must NOT implement hacky bypasses that compromise long-term scalability. Instead:
> - File clear, easy-to-reproduce bug reports for the corresponding component teams (Pathways, TPU Sync, or Compiler).
> - Provide standalone minimal reproduction scripts that isolate the failing behavior **without** pulling in the full complex integration across `tunix` / `maxtext` / `tpu-inference` / `vllm` / `tpu-sync`.

### Target Hardware & Topology
- **Cluster**: `bodaborg-v5p-nap` (GCP Project: `cloud-tpu-shared-capacity`, Region: `europe-west4`, Zone: `europe-west4-b`).
- **Reservation**: `cloudtpu-20260902214500-1810493672`.
- **Trainer Topology**: 1x `v5p-16` (`tpuv5:2x2x2`, 8 chips across 2 host nodes, 4 chips/node).
- **Rollout Topology**: 1x `v5p-8` (`tpuv5:2x2x1`, 4 chips on 1 host node, TP=2) per replica. Single rollout requires 12 total TPU chips; dual rollout requires 16 chips.
- **Orchestrator Topology**: 1x CPU node (`n2d-standard-64` in `cpu-np-456be230`).

---

## 2. Latest Verified Milestone: `igorts-v8-06b` (E2E GRPO Training Succeeded)

The workload `igorts-v8-06b` ran to completion with 0 errors on the cluster using image `gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-raiden-debug-0903-2`.

### Execution Timeline & Results

1. **Pathways Dual-Node Scheduler Convergence**:
   - Both worker slices (`igorts-v8-06b-train-pw-node-0-0` and `pw-node-0-1`) scheduled onto nodes `gke-tpu-4fa40201-l5ss` and `gke-tpu-4fa40201-c092`.
   - Restored base checkpoint `gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items` in **13.76s** at 937.4 MiB/s.
   - Initialized 8-device mesh: `Mesh('fsdp': 8)`.

2. **Raiden Weight Synchronization (Steps 0, 1, 2)**:
   - Synchronized **310 tensors** (**596,049,920 elements**) in exactly **287,060 blocks** per transfer.
   - **Step 0 Transfer**: Completed in **14.47s**.
   - **Step 1 Transfer**: Completed in **12.98s**.
   - **Step 2 Transfer**: Completed in **13.62s**.
   - **Zero errors**: No destination out-of-bounds errors, no transport disconnects.

3. **Exact Checksum Verification**:
   - **Trainer Source Checksums**:
     ```json
     {
       "['base']['decoder']['decoder_norm']['scale']": 1536.0,
       "['base']['decoder']['layers_0']['mlp']['wi_0']['kernel']": 86.5,
       "['base']['decoder']['layers_0']['mlp']['wi_1']['kernel']": 64.5,
       "__grand_total__": 102835.75,
       "__tensor_count__": 310,
       "__element_count__": 596049920
     }
     ```
   - **Rollout Destination Checksums**:
     ```json
     {
       "['base']['decoder']['decoder_norm']['scale'].value": 3933.169921875,
       "['base']['decoder']['layers_0']['mlp']['wi_0']['kernel'].value": 88959.140625,
       "['base']['decoder']['layers_0']['mlp']['wi_1']['kernel'].value": 64017.1796875,
       "__grand_total__": 13217973.204956055,
       "__tensor_count__": 310,
       "__element_count__": 596049920
     }
     ```

4. **vLLM Rollout Generation**:
   - Rollout worker evaluated GSM8K prompts and generated coherent chain-of-thought `<reasoning>` traces with zero gibberish.
   - Prefix cache and KV cache cleanly re-initialized across policy version boundaries.

5. **GRPO Policy Optimization & Checkpoint Commit**:
   - **Step 0**: `loss: -0.0000`, `reward_mean: 0.0625`, `step_time: 51.35s` -> Policy Version advanced to 1. Checkpoint saved to `/tmp/artifacts/math_gsm8k_dist/maxtext/igorts-v8-06b-train/checkpoints/1`.
   - **Step 1**: `loss: 0.0000`, `reward_mean: 0.0000`, `step_time: 14.98s` -> Policy Version advanced to 2. Checkpoint saved to `/tmp/artifacts/math_gsm8k_dist/maxtext/igorts-v8-06b-train/checkpoints/2`.
   - **Clean cluster shutdown**: All workers gracefully unregistered and stopped.

---

## 3. Critical Debugging Insights & Root Cause Analysis

### A. The Raiden Batched Push "Destination out of bounds" Bug
- **Symptom**: Custom builds (`igorts-v7-06b`) crashed during `StartTransfer` with:
  `raw_buffer_transport.cc:341] ProcessPeerRequest failed: INVALID_ARGUMENT: Destination out of bounds in batched push`.
- **False Hypothesis Explored**: An attempt was made to compute `dst_block_offset = dst_offset % dst_block_bytes` inside `raiden_controller.py`.
- **Definitive Findings**:
  1. Modulo offsetting was **invalid**. It broke unit test `test_multi_variable_resharding_planning` (which tests multi-variable non-legacy resharding planning) and caused schedule generation to explode from **287,060 blocks** to **633,826 fragmented blocks**.
  2. In `raw_buffer_transport.cc`, each shard buffer allocated by `WeightSynchronizerBase` for a given `(layer_idx, shard_idx)` is sized for the entire slice (`shard_info.host_size = alloc_size`).
  3. Wire metadata offsets transmitted between trainer and rollout workers are **absolute byte offsets** within that variable's slice buffer.
  4. The true root cause of `Destination out of bounds` in modified images was **metadata array ordering mismatches** between sender and receiver:
     - If the receiver binds arrays in a different order than the sender's metadata list, or if runtime cache variables (`cached_prefill_key`, `cached_prefill_value`) shift the `layer_idx`, the C++ transport writes a larger tensor into a buffer allocated for a smaller tensor.

### B. Pathways GKE Pod Affinity & Dual-Node Scheduling
- **Issue**: Pathways `train-proc` and `pw-node-0-0` would run, but `pw-node-0-1` would remain stuck in `Pending`.
- **Mechanism**:
  - `jobset.pathways.yaml` enforces:
    ```yaml
    affinity:
      podAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
        - topologyKey: cloud.google.com/gke-nodepool
    ```
  - Both nodes of the `v5p-16` slice must reside in the exact same GKE nodepool.
  - When another job (such as a 10-minute diagnostic sleep pod) holds one of the nodes in that nodepool, `pw-node-0-1` cannot schedule until the sibling node terminates or the job is targeted to an alternate idle dual-node pool (e.g. `nap-ct5p-hightp-4t-fv6ijjye`).

### C. Multi-Rollout Worker Collision (`--rollout-replicas=2`)
- **Issue**: Multi-rollout workers failed to register distinct transfer destinations.
- **Root Cause & Fix**:
  - In `tunix/experimental/rollout/vllm_sampler_adapter.py`, the metadata dictionary's `unit` must be explicitly assigned to `self.server_id` (`roll-0`, `roll-1`):
    ```python
    if self.server_id:
      unit = m.get("unit")
      if isinstance(unit, dict):
        unit["job_name"] = self.server_id
      else:
        m["unit"] = {"job_name": self.server_id, "job_replica_id": ""}
    ```
  - With distinct unit names, the controller generates a combined multicast schedule:
    `Transfer: generated schedule for [trainer] -> [roll-0, roll-1] (310 variable(s), 574,120 expected blocks)`.

---

## 4. Summary of Code Changes Made & Rationale

### A. `tunix/experimental/rollout/vllm_sampler_adapter.py`
- **Change**: Updated `get_weight_sync_metadata` to preserve dictionary structure for `unit` while setting `job_name = self.server_id`.
- **Rationale**: Prevents multi-replica rollout workers from overwriting each other under the default `"destination"` identifier during discovery registration.

### B. `tunix/experimental/weight_sync/raiden_synchronizer.py`
- **Change**: Added deterministic alphabetical sorting to `names` and `arrays` in `RaidenSynchronizer.bind`:
  ```python
  if self.names:
    pairs = sorted(zip(self.names, self.arrays), key=lambda x: x[0])
    self.names = [p[0] for p in pairs]
    self.arrays = [p[1] for p in pairs]
  ```
- **Rationale**: Guarantees identical positional ordering (`layer_idx`) between JAX PyTree flattening on the trainer and NNX flattening on the rollout worker, preventing layer buffer size mismatches.

### C. `tunix/experimental/weight_sync/weight_sync_coordinator.py`
- **Change**: Added dynamic `HOST_STAGE` environment variable detection for `RaidenTransferOptions`:
  ```python
  host_stage = os.getenv("HOST_STAGE", "0").lower() in ("1", "true", "yes")
  if host_stage:
    transfer_options = raiden_handler.make_host_staged_transfer_options()
  else:
    transfer_options = raiden_handler.RaidenTransferOptions(parallelism=16)
  ```
- **Rationale**: Allows toggling host staging for large model runs without hardcoding options.

### D. `tunix/experimental/examples/math_gsm8k_dist/launch_raiden.sh`
- **Change**: Built a comprehensive CLI launcher supporting `start`, `stop`, `restart`, `status`, `logs`, `triage`, and `dry-run`.
- **Rationale**: Eliminates fragile manual `kubectl` invocations, auto-resolves GKE nodepools, and handles clean JobSet lifecycles.

---

## 5. Action Items & Next Steps

### A. Scaling to Qwen3.5-35B-A3B (`--model qwen3.5-35b`)
1. **Filter Runtime KV Cache Variables in `tpu-inference`**:
   - In `tpu_inference/rl/raiden_worker_sync.py`:
     - Filter `nnx.State` by `nnx.Param` in `extract_weight_state`.
     - In `_filter_bindable`, drop any names containing `"cache"` (`cached_prefill_key`, `cached_prefill_value`) *prior* to constructing `WeightSynchronizer(self.arrays)`.
   - **Crucial Rule**: Do NOT filter these inside `vllm_sampler_adapter.py` after C++ binding. Filtering must happen before C++ synchronizer allocation so `layer_idx` indexes strictly into the 673 trainable weights.
2. **Build and Tag Verified Container Image**:
   - Build a fresh image containing the sorted `raiden_synchronizer.py` and filtered `raiden_worker_sync.py`.
   - Test manifest preflight on Qwen3.5-35B to verify `673 source var(s) == 673 destination var(s)`.
3. **Verify JAX FFI Integration**:
   - Verify that `weight_synchronizer_ffi` functions as required for Pathways direct TPU device transfer.
   - If FFI encounters runtime incompatibilities with libtpu or Pathways proxies, produce a **standalone reproduction script** (e.g. 2-node minimal JAX FFI transfer without MaxText/Tunix) and file a bug report for the compiler/TPU sync teams.

### B. Validating Multi-Rollout Worker Execution (`--rollout-replicas=2`)
- Run `launch_raiden.sh start --model qwen3-0.6b --rollout-replicas=2`.
- Verify orchestrator schedules 574,120 blocks (287,060 x 2) and both rollout workers complete generation in parallel.

---

## 6. Quick Reference: Resumption Commands

```bash
# 1. Connect to Cluster
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --zone europe-west4-b \
  --project cloud-tpu-shared-capacity

# 2. Launch Verified Qwen3-0.6B Baseline
/tmp/launch_raiden.sh start --model qwen3-0.6b --rollout-replicas=1 --image gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-raiden-debug-0903-2

# 3. Monitor Workload
/tmp/launch_raiden.sh status
/tmp/launch_raiden.sh logs orch -f
/tmp/launch_raiden.sh logs rollout -f
/tmp/launch_raiden.sh logs trainer -f

# 4. Clean Shutdown
/tmp/launch_raiden.sh stop
```
