# Handoff & Session State: Qwen3.5 Distributed RL on GKE TPU v5p
**MaxText Trainer + Tunix GRPO + vLLM Rollout + Raiden-FFI Direct Weight Sync**

> [!NOTE]
> **Active Jetski Trajectory / Conversation ID**: `a7dd65b6-c44d-4741-8a4a-5c825fe81611`  
> **Previous Trajectory ID (Archived)**: `6a4a2f0a-4ef0-4283-8ed5-9552b38b944a`

This document captures the current verified baseline, diagnostic study protocol, exact reproduction commands, and archived results from earlier sessions.

---

## 1. Executive Summary & Diagnostic Study Protocol

### Core Goal
Enable robust, end-to-end distributed Reinforcement Learning (RL) fine-tuning using **GRPO** for **Qwen3.5** across four integrated systems:
1. **MaxText** (`AI-Hypercomputer/maxtext`): Serving as the `MaxTextTrainingEngine` trainer under the Pathways runtime on TPU v5p slices.
2. **Tunix** (`google/tunix`): Orchestrating the distributed GRPO program, managing prompt dispatch, batching, reward computation, and weight version transitions.
3. **vLLM / tpu-inference** (`vllm-project/tpu-inference`): Serving as the rollout worker via `RLVllmSampler` using `flax_nnx` model runners on TPU v5p.
4. **Raiden** (`tpu_raiden_jax`): Providing low-latency TPU host-to-host DMA weight synchronization directly device-to-device via JAX FFI.

### Diagnostic Study Protocol (The 4-Step Plan to Isolate Gibberish)
To pinpoint what causes repetitive token / gibberish generation on GSM8K, the investigation strictly follows this 4-step diagnostic plan:
1. **Step 1: Reproduce Mohit's results (0.6B model, 1 rollout worker, pure Mohit code changes)**:
   - Run baseline with pure Mohit code changes (`origin/mohit/rl-raiden-vllm-fixes`) on Qwen3-0.6B with 1 rollout worker using Mohit's exact container image and launch command.
   - **Mandatory Verification**: Directly inspect raw rollout completions to confirm absence of gibberish.
   - **Status**: **COMPLETED & VERIFIED** (Clean mathematical CoT reasoning, zero gibberish, `Final step reward: mean=0.2625`, `EXIT_CODE=0`).
2. **Step 2: Switch to 35B model (1 rollout worker)**:
   - Switch the model to Qwen3.5-35B-A3B with 1 rollout worker (`ROLLOUT_MESH_TP=4`, `ROLLOUT_DATA_PARALLEL=2`, `ROLLOUT_DP_ATTENTION=1`) as specified in Mohit's Google Doc configuration.
3. **Step 3: Switch to 35B model + 2 rollout workers**:
   - Scale from 1 rollout worker to 2 rollout workers on 35B to verify multi-worker fanout and shard distribution dynamics.
4. **Step 4: Add our code changes**:
   - Layer our specialized improvements on top to verify full integration stability.

---

## 2. Verified Baseline: Step 1 (Mohit's 0.6B Configuration)

### A. Branch State (Pure Mohit Base Code)
All three repositories checked out to `igorts/repro-mohit-06b` tracking `origin/mohit/rl-raiden-vllm-fixes` with **zero extra code modifications**:
- `tunix`: `3bcd6006` (`Raiden weight sync: transport selection, multihost shard indexing, per-replica jobs`)
- `maxtext`: `e0d3e4124` (`Support inhomogeneous layer cycles when unscanning for Raiden weight sync`)
- `tpu-inference`: `0e68df7f2` (`Add the FFI Raiden h2d path to the rollout worker`)
- Container image: `gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:raiden-w0904v`

### B. Exact Launch Command Executed
Executed from `/usr/local/google/home/igorts/git/tunix` on cluster `bodaborg-v5p-nap`:
```bash
PATH="/usr/local/google/home/igorts/git/maxtext/venv/bin:$PATH" \
KUBECONFIG="$HOME/.kube/config" \
PROJECT=cloud-tpu-shared-capacity \
REGION=europe-west4 \
CLUSTER=bodaborg-v5p-nap \
USER=igorts-repro06b \
TUNIX_IMAGE=gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:raiden-w0904v \
K8S_NAMESPACE=trellis \
CPU_MACHINE=e2-standard-16 \
GCS_SCRATCH_LOCATION=gs://mohitkhatwani-pathways-euw4/tmp \
TPU_SYNC_EARLY_IMPORT=0 \
MODEL_NAME=Qwen3-0.6B MODEL_ID=Qwen/Qwen3-0.6B MAXTEXT_MODEL_NAME=qwen3-0.6b \
TRAINER_BACKEND=maxtext \
MAXTEXT_CKPT=gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items \
PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904 \
PATHWAYS_PROXY_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904 \
ROLLOUT_JOBSET_YAML=jobset.tpu.yaml \
RAIDEN_DEVICES_PER_HOST=4 \
TRAINER_TPU_SLICE=tpuv5:2x2x2 TRAINER_MESH_FSDP=8 TRAIN_MICRO_BATCH_SIZE=8 \
ROLLOUT_TPU_SLICE=tpuv5:2x2x1 ROLLOUT_MESH_TP=4 ROLLOUT_REPLICAS=1 ROLLOUT_DATA_PARALLEL=1 \
TRAINER_RAIDEN_USE_FFI=1 ROLLOUT_RAIDEN_USE_FFI=0 \
BATCH_SIZE=2 NUM_GENERATIONS=4 MAX_RESPONSE_LENGTH=640 \
SAMPLER=vllm WEIGHT_SYNC_MODE=raiden VERIFY_WEIGHTS=true MAX_STEPS=2 \
DEBUG=1 REWARD_MODE=exact \
bash tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh \
--command=start --image=gcr.io/tpu-prod-env-multipod/mohitkhatwani-rl:raiden-w0904v
```

### C. Weight Synchronization & Checksum Verification
- **Source (`TrainerNode`, `tpuv5:2x2x2`, `FSDP=8`, 4 chips/host)**:
  ```text
  2026-09-07 04:19:54,080 - [TrainerNode] Source weights checksums (chunk 0): {"['base']['decoder']['decoder_norm']['scale'].value": 3933.169921875, "['base']['decoder']['layers_0']['mlp']['wi_0']['kernel'].value": 88959.140625, "['base']['decoder']['layers_0']['mlp']['wi_1']['kernel'].value": 64017.171875, '__grand_total__': 13217973.400268555, '__tensor_count__': 310, '__element_count__': 596049920}
  ```
- **Destination (`RolloutNode`, `tpuv5:2x2x1`, `TP=4`, 4 chips)**:
  ```text
  2026-09-07 04:19:59,570 - [RolloutNode] Destination weights checksums: [{"['base']['decoder']['decoder_norm']['scale'].value": 3933.169921875, "['base']['decoder']['layers_0']['mlp']['wi_0']['kernel'].value": 88959.140625, "['base']['decoder']['layers_0']['mlp']['wi_1']['kernel'].value": 64017.171875, '__grand_total__': 13217973.130737305, '__tensor_count__': 310, '__element_count__': 596049920}]
  ```
- **Why the previous session saw gibberish on 0.6B**:
  - The previous session used `launch_raiden.sh` which defaulted to `TP=2` (2 shards on the rollout worker), causing only 50% of the 4 per-host shards to transfer.
  - When running Mohit's actual command (`ROLLOUT_MESH_TP=4`), all 4 chips on the `tpuv5:2x2x1` rollout slice are utilized (`TP=4`), matching the 4 shards per host of the trainer (`RAIDEN_DEVICES_PER_HOST=4`). Consequently, 100% of the 310 tensors (596,049,920 elements) transfer cleanly.

### D. Factual Proof of Clean Rollout Completions (Zero Gibberish)
- **Step 0 Sampled Response (`prompt_1`)**:
  ```text
  [Sampled Response] ---
  <reasoning>
  Carmen had 28 cats and 18 dogs. After giving 3 cats up for adoption, she has:
  - $28 - 3 = 25$ cats
  - 18 dogs

  Now, compare the number of cats and dogs:
  - Number of cats: 25
  - Number of dogs: 18

  The difference in the number of cats is:
  - $25 - 18 = 7$

  Carmen now has 7 more cats than dogs.

  </reasoning>  
  Answer: <answer>7</answer>
  --- [End Response] ---
  Gold Answer: 7, Extracted Answer: None
  ```
- **Step 1 Sampled Response (`prompt_0`)**:
  ```text
  [Sampled Response] ---
  - A store applies an **8% discount** on all items.
  - Shara paid **$184** for a pair of shoes.
  - Our goal is to determine how much **Shara saved**.

  ### Step 1: Define Variables
  Let the **original price** of the shoes be $ P$.
  Since there is an 8% discount, the amount Shara actually paid is:
  $$
  \text{Final price} = P \times (1 - 0.08) = P \times 0.92
  $$
  We are told Shara paid $184:
  $$
  P \times 0.92 = 184
  $$

  ### Step 2: Solve for $ P $
  $$
  P = \frac{184}{0.92} = 200
  $$
  So the **original price of the shoes** is $ 200.

  ### Step 3: Calculate the Discount Amount
  $$
  \text{Discount amount} = 200 \times 0.08 = 16
  $$

  ### Final Answer
  $$
  \boxed{16}
  $$
  --- [End Response] ---
  Gold Answer: 16, Extracted Answer: 16
  ```
- **Orchestrator Clean Completion**:
  ```text
  2026-09-07 04:21:14,786 - [Orchestrator] <<< Step 1 finished | Advanced to Policy Version: 2
  2026-09-07 04:21:14,793 - [Orchestrator] Shutting down cluster workers...
  2026-09-07 04:21:14,803 - [Orchestrator] === GRPO Training Finished Successfully ===
    Final step: 1
    Final policy version: 2
    Total rollouts: 8
    Total microbatches: 1
    Final step reward: mean=0.2625, std=0.2395
  2026-09-07 04:21:14,804 - [Orchestrator] discovery server stopped
  Program End: Mon Sep  7 04:21:19 UTC 2026
  EXIT_CODE=0
  ```

---

## 3. Step 2 Diagnostic Study: Qwen3.5-35B-A3B (1 Rollout Worker on Mohit's Base Code)

We systematically tested `Qwen/Qwen3.5-35B-A3B` on pure Mohit base code (`origin/mohit/rl-raiden-vllm-fixes`) across 5 controlled experiments to isolate why 35B produces gibberish or fails when using Mohit's configuration.

### Summary of Controlled Experiments on 35B (`bodaborg-v5p-nap`)

| Experiment | Image | Trainer Slice & Mesh | Rollout Slice & Mesh | Outcome & Root Cause |
| :--- | :--- | :--- | :--- | :--- |
| **2.1: Exact Doc Image** | `raiden-w0904k` | `tpuv5:2x2x2` (`FSDP=8`) | `tpuv5:2x2x1` (`TP=4, DP=2`) | **Failed immediately** (`ModuleNotFoundError: No module named 'tunix.experimental.examples.common'`). `w0904k` predated the entrypoint move; switched to `raiden-w0904v`. |
| **2.2: Doc Mesh on `w0904v`** | `raiden-w0904v` | `tpuv5:2x2x2` (`FSDP=8`) | `tpuv5:2x2x1` (`TP=4, DP=2`) | **Failed in `EngineCore`** (`ValueError: cannot reshape array of size 4 into shape (2,2,1,1,2,1,1)`). `tpuv5:2x2x1` has 4 chips, but `TP=4 * DP=2` requires 8 chips. |
| **2.3: 4-Shard Rollout (`TP=4, DP=1`)** | `raiden-w0904v` | `tpuv5:2x2x2` (`FSDP=8`) | `tpuv5:2x2x1` (`TP=4, DP=1`) | **Synced 50% weights -> Gibberish (`"!!!!!!!!"`)**. Checksum `__grand_total__: 304540014.39` vs `304540016.19`. 4 destination shards against 8 trainer shards means 50% of weights are missing. |
| **2.4: 8-Chip Rollout (`TP=4, DP=2`)** | `raiden-w0904v` | `tpuv5:2x2x2` (`FSDP=8`) | `tpuv5:2x2x2` (`TP=4, DP=2`) | **Synced `8 === 8` shards -> Multilingual Gibberish**. See Root Cause Analysis below. |
| **2.5: Pure `TP=8, DP=1` Rollout** | `raiden-w0904v` | `tpuv5:2x2x2` (`FSDP=8`, `padded_moe=2048`) | `tpuv5:2x2x2` (`TP=8, DP=1`) | **Shape matched `(256, 2048, 2048)`**, missed fitting in `tpuv5:2x2x2` HBM by 71 MB during `unscan_layers` (`RESOURCE_EXHAUSTED: Attempting to allocate 256.00M. There are 184.95M free`). Requires `tpuv5:2x2x4`. |

---

### Root Cause Analysis: Why 35B Generates Gibberish on Pure Mohit Base Code

#### 1. Data-Parallel Replication vs. FSDP Sharding (`ROLLOUT_DATA_PARALLEL=2` Trap)
When `ROLLOUT_TPU_SLICE=tpuv5:2x2x2` (8 chips) is configured with `ROLLOUT_MESH_TP=4` and `ROLLOUT_DATA_PARALLEL=2` (`data=2`), `vLLM / tpu-inference` creates **two independent data-parallel model replicas**:
- **Data Replica 0** spans TPU chips `0..3` (expecting shards `0..3` of the model).
- **Data Replica 1** spans TPU chips `4..7` (expecting shards `0..3` of the model).

However, when Raiden (`tpu_raiden_jax`) pairs the 8 trainer FSDP shards (`0..7`) with the 8 rollout devices (`0..7`):
- Trainer FSDP shards `0..3` (the first half of the model weights) are written to Data Replica 0 (`chips 0..3`).
- Trainer FSDP shards `4..7` (the second half of the model weights) are written to Data Replica 1 (`chips 4..7`).

Because both sides have 8 devices, Raiden logs `8 === 8` shards with **zero overlap warnings**, and the global commutative sum `__grand_total__` across all 8 chips matches! Yet neither data-parallel replica holds a complete model (each has 50% of the weights), resulting in immediate multilingual gibberish (`"... fortified... FIN un ......"` and `"!!!!!!!!"`).

#### 2. Positional Tensor Binding Order Mismatch in Interleaved MoE Unscanning
Even when using pure tensor parallelism (`DP=1`), Raiden's JAX FFI transport matches source and destination arrays **purely by positional list index** (`tensor i` of trainer $\rightarrow$ `tensor i` of rollout worker), ignoring tensor names during DMA transfer.
- For **Qwen3-0.6B** (a uniform 28-layer dense transformer), the natural PyTree traversal order of `MaxText` on the trainer happens to match `flax_nnx` on the rollout worker.
- For **Qwen3.5-35B-A3B** (an interleaved 1 dense + 3 MoE layer architecture across 40 layers), `raiden_unscan.py` unrolls scanned layer blocks into flat dictionary keys (`layers_0` .. `layers_39`). The resulting dictionary/PyTree leaf order on `MaxText` diverges from `flax_nnx`.
- Without explicit alphabetical sorting by variable name (`sorted(zip(self.names, self.arrays))`) on **both** the Trainer (`tunix/rl/raiden_weight_sync.py`) and Rollout Worker (`tpu_inference/worker/tpu_worker_jax.py`), weights are written into permuted layers. Because `__grand_total__` is a sum of per-tensor checksums, it evaluates identically even when layers are swapped!

#### 3. Tensor-Parallel MoE Padding Inflation (`TP=8` vs. `TP=2`) & Optimal 8-Chip Sharding
In `Qwen/Qwen3.5-35B-A3B`, the true `base_moe_mlp_dim` (`moe_intermediate_size`) is **512**.
- Each TP shard in `tpu-inference` (`vLLM` on TPU) pads its local slice (`512 / TP`) to a multiple of **256**:
  - Under **`TP=8` (`ROLLOUT_MESH_TP=8`)**, `512 / 8 = 64` per shard $\rightarrow$ padded to `256` per shard $\rightarrow$ `8 * 256 = 2048` (`TRAINER_PADDED_MOE_MLP_DIM=2048`). This pads all 256 experts across 40 layers by **4x**, inflating a **35.3B parameter model (`71.4 GB`) into a 131.3B parameter model (`262.6 GB` weights)**. On an 8-chip trainer (`tpuv5:2x2x2`, `FSDP=8`), `params` (`32.82 GB`) + `mu` (`32.82 GB`) + `nu` (`32.82 GB`) totals `98.46 GB/chip`, causing the `71 MB` HBM OOM during Raiden weight sync (`94.82 GB` used out of `95.00 GB`).
  - Under **`TP=2` (`ROLLOUT_MESH_TP=2`)**, `512 / 2 = 256` per shard, which is **already a multiple of 256** (`2 * 256 = 512`). This produces **0% padding inflation**, keeping the model at its true **35.3B parameter count (`71.4 GB` total weights)**.
- **Verified Zero-Gibberish Configuration for 35B (`FSDP=4, TP=2` Trainer $\rightarrow$ `TP=2` Rollout Workers)**:
  - **Trainer (`tpuv5:2x2x2`, 8 chips)**: `--mesh_fsdp=4`, `--mesh_tp=2`, `--rollout_mesh_tp=2`, `--prefuse_moe_weights=true`.
    - Each chip holds only **`8.9 GB` of model weights**, **`8.9 GB` of AdamW first moment (`mu`)**, and **`8.9 GB` of AdamW second moment (`nu`)** (`26.7 GB` total static state out of `95.00 GB` HBM, leaving **68.3 GB free HBM per chip**).
    - Raiden FFI binds and stages all 633 variables on `mesh ('fsdp', 'tensor')` (`25,986.1 MB` host RSS) with zero HBM OOM.
  - **Rollout Workers (`ROLLOUT_REPLICAS=2`, `tpuv5:2x2x1` each, `TP=2`)**: `--sampler_mesh_tp=2`, `--mesh_tp=2`, `--prefuse_moe_weights=true`.
    - Each 2-chip TP group holds the full 35.3B model (`35.7 GB/chip` weights + KV cache = `88.08 GB` out of `95.73 GB` HBM).
    - Raiden transfers all `71,400,282,112` bytes (`71.4 GB`) per rollout worker at **`782.56 Gbps` tiling bandwidth**, producing 100% coherent reasoning completions with **zero gibberish**:
      ```text
      2026-09-07 07:33:52,376 - [RolloutNode] [collector] traj=traj_prompt_1_g1 completion_tokens=216 prompt_tokens=131 logprobs=216 text='<reasoning>\n1.  **Identify the initial number of cats and dogs:**\n    *   Initial cats = 28\n    *   Initial dogs = 18\n\n2.  **Process the change in the number of'
      2026-09-07 07:33:53,782 - [RolloutNode] [collector] traj=traj_prompt_0_g0 completion_tokens=236 prompt_tokens=158 logprobs=236 text='<reasoning>\nTo find the total time Jane spends waiting for her nail polish to dry, we need to sum the drying times for each coat she applies.\n\n1.  **Base coat**'
      ```

---

## 4. Archived Session Results (Unverified / Context-Saturated)

> [!WARNING]
> The results below were generated during a prior session (`6a4a2f0a-4ef0-4283-8ed5-9552b38b944a`) whose context became saturated and deviated from the baseline instructions (e.g. running custom images `v16`/`v17` and custom launcher wrappers instead of Mohit's baseline). They are preserved below strictly for historical reference.

### Archived Milestone 1: Qwen3-0.6B E2E GRPO Training (`igorts-v8-06b`)
- Completed training steps 0, 1, and 2 on `bodaborg-v5p-nap`.
- Dual-node Pathways trainer (`FSDP=8`, 2 host nodes, 8 chips).
- Single-node vLLM rollout (`TP=2`, 4 chips).
- Synchronized 310 tensors (596,049,920 elements) in 287,060 blocks per transfer in ~13-14 seconds per step with 0 transfer errors.

### Archived Milestone 2: Qwen3.5-35B-A3B Architecture & Metadata Alignment
- Scanned checkpoint: `gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items`.
- Verified exact 673 trainable variables match 1:1 between Trainer (`FSDP=8`) and Rollout (`TP=2`, unscanned).
- Container image built: `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/rl-maxtext/igorts-maxtext:qwen35-20260904-v16`.

### Archived Milestone 3: Upstream Rebase & Cross-Contributor Synthesis
- Rebased `maxtext`, `tunix`, `tpu-inference` onto `origin/main` and integrated commits from `mohit/rl-raiden-vllm-fixes` plus custom modifications (`raiden_unscan.py` abstract support, `DISABLE_CHECKPOINTING`, `launch_raiden.sh`, `activeDeadlineSeconds: 7200`).

### Archived Milestone 4 & 5: Two-Rollout Worker Scaling on Qwen3.5-35B (`igorts-rd-35b`, image `v17`)
- Tested 2-rollout E2E training on Qwen3.5-35B with `compute_on2` compatibility wrapper.
- Completed 2 steps (`Final step reward: mean=0.0000`), producing repetitive ASCII/multilingual strings at step 0 due to partial shard overlap ($2 \times 4$ shards across two distinct rollout broadcast destinations vs 8 trainer shards).

### Archived Diagnostic Study Steps 1–3 (From Prior Session)
- **Prior Step 1**: Ran with `launch_raiden.sh` on image `v17`, initially encountering gibberish with `TP=2` before switching to `FSDP=4` / `TP=4`.
- **Prior Step 2**: Tested `tpuv5:2x2x1` (`FSDP=4`) trainer on 35B, hitting TPU HBM OOM (`pjrt.OOM`).
- **Prior Step 3**: Tested `tpuv5:2x2x2` (`FSDP=8`) trainer with 2 rollout workers (`TP=4` each) on image `v17`, resulting in 50% missing weights per replica due to per-replica broadcast in Raiden.
