<!--
 Copyright 2025-2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

(diloco-pretraining)=

# DiLoCo (Distributed Low-Communication) Training

This tutorial guides you through configuring and running **DiLoCo** and **Streaming DiLoCo** training in MaxText across multi-slice TPU clusters, multi-datacenter pods, and low-bandwidth DCN/WAN networks.

```{note}
For theoretical background, arithmetic intensity calculations, and in-depth architectural details, refer to the [DiLoCo Theory & Mathematics Reference](../reference/core_concepts/diloco.md).
```

______________________________________________________________________

## 1. Vanilla DiLoCo vs. Streaming DiLoCo

MaxText supports two modes of distributed low-communication training:

```
Vanilla DiLoCo:
[------- H steps Compute (Local ICI) -------][ Full Model All-Reduce (DCN) ][------- H steps Compute -------]
                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                Periodic Barrier Pause

Streaming DiLoCo:
[ Step 1 Compute ][ Step 2 Compute ][ Step 3 Compute ] ...
  └─ Sync Frag 0 ─┘  └─ Sync Frag 1 ─┘  └─ Sync Frag 2 ─┘   (Continuous Pipelined Synchronization)
```

### Key Differences:

- **Vanilla DiLoCo (`enable_streaming_diloco=false`)**:

  - **How it works**: Each computing island trains independently for $H$ inner steps (e.g., $H=100$). At every $H$-th step, training pauses for a global collective all-reduce where the entire model's pseudo-gradient ($\Delta \theta = \theta_{\text{outer}} - \theta_{\text{inner}}$) is averaged across all islands over DCN and updated using outer Nesterov momentum.
  - **When to use**: Simpler baseline, ideal when $H$ is large (e.g. $H \ge 500$) and the periodic all-reduce pause represents a negligible fraction of total training time.

- **Streaming DiLoCo (`enable_streaming_diloco=true`)**:

  - **How it works**: The model parameters are partitioned into $P$ fragments (typically $P = N_{\text{layers}} + 1$). By setting $H = P$, exactly 1 fragment is synchronized on every single local inner step ($\Delta h = 1$).
  - **When to use**: Optimal for high-throughput scaling across lower-bandwidth DCN/WAN networks, as it eliminates bursty communication spikes and removes the periodic step-$H$ idle barrier.

______________________________________________________________________

## 2. Prerequisites

1. **MaxText Environment**: Follow the [installation guide](../install_maxtext.md) to set up your environment (`maxtext[tpu]` or `maxtext[cuda12]`).
2. **Compute Resources**: A Google Kubernetes Engine (GKE) cluster with TPU slices managed via [XPK](https://github.com/AI-Hypercomputer/xpk).
3. **Storage**: A Google Cloud Storage (GCS) bucket for logging and Orbax checkpoints (`gs://<GCS_BUCKET>`).

______________________________________________________________________

## 3. Production Recipe 1: Vanilla DiLoCo Multi-Slice Pre-training

In this recipe, we train a model (e.g., **Qwen3-8B**) across **2 TPU v5p-128 slices** using Vanilla DiLoCo with periodic synchronization every $H=100$ steps:

```bash
python3 -m maxtext.trainers.pre_train.train \
  maxtext/configs/base.yml \
  run_name="vanilla-dlco-8b-01" \
  base_output_directory="gs://your-bucket/maxtext-logs" \
  dataset_path="gs://your-bucket/maxtext-datasets" \
  dataset_name='c4/en:3.0.1' \
  model_name="qwen3-8b" \
  per_device_batch_size=8 \
  max_target_length=2048 \
  enable_diloco=true \
  enable_streaming_diloco=false \
  dcn_diloco_parallelism=2 \
  diloco_sync_period=100 \
  diloco_outer_lr=0.7 \
  diloco_outer_momentum=0.9 \
  pure_nnx=true \
  steps=1000 \
  enable_checkpointing=true \
  checkpoint_period=100
```

### Configuration Breakdown:

- `enable_diloco=true`: Enables outer optimization and multi-slice Low-Communication training across `dcn_diloco_parallelism=2` slices.
- `enable_streaming_diloco=false`: Disables parameter fragmentation and performs full-model pseudo-gradient all-reduce.
- `diloco_sync_period=100`: Islands execute 100 local AdamW steps independently before pausing to sync.
- `diloco_outer_lr=0.7` and `diloco_outer_momentum=0.9`: Outer Nesterov momentum parameters.

______________________________________________________________________

## 4. Production Recipe 2: Streaming DiLoCo Dense Pre-training (Qwen3-8B)

In this recipe, we train **Qwen3-8B** with Streaming DiLoCo across **2 TPU v5p-128 slices** with $H=P=37$ (synchronizing 1 fragment every step) via the SPMD runner script:

```bash
CLUSTER="mlperf-v5p" \
ZONE="europe-west4-b" \
PROJECT="cloud-tpu-multipod-dev" \
DEVICE_TYPE="v5p-128" \
NUM_SLICES="2" \
RUNNAME="stream-dlco-8b-01" \
XPK_WORKLOAD="stream-dlco-01" \
BASE_OUTPUT_DIRECTORY="gs://your-bucket/maxtext-logs" \
DATASET_PATH="gs://your-bucket/maxtext-datasets" \
MODEL_NAME="qwen3-8b" \
STEPS="1000" \
CHECKPOINT_PERIOD="100" \
DILOCO_SYNC_PERIOD="37" \
DILOCO_NUM_FRAGMENTS="37" \
DILOCO_NUM_COMM_OVERLAP_STEPS="0" \
DILOCO_OUTER_LR="0.7" \
DILOCO_OUTER_MOMENTUM="0.9" \
bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_diloco.sh
```

______________________________________________________________________

## 5. Production Recipe 3: Streaming DiLoCo MoE Pre-training (Qwen3-30B-A3B)

For large Mixture-of-Experts (MoE) architectures, this recipe demonstrates Streaming DiLoCo pre-training with the **OLMo Grain** data pipeline across 2x `v5p-128` TPU slices:

```bash
XPK_CLUSTER="mlperf-v5p" \
XPK_ZONE="europe-west4-b" \
XPK_PROJECT="cloud-tpu-multipod-dev" \
XPK_DEVICE_TYPE="v5p-128" \
XPK_NUM_SLICES="2" \
RUN_NAME="qw3-olmo-dlco-01" \
WORKLOAD_NAME="qw3-olmo-01" \
BASE_OUTPUT_DIRECTORY="gs://your-bucket/maxtext-logs" \
OLMO_GCS_BASE="gs://your-bucket/datasets" \
MODEL_NAME="qwen3-30b-a3b" \
ENABLE_STREAMING_DILOCO="true" \
DILOCO_SYNC_PERIOD="49" \
DILOCO_NUM_FRAGMENTS="49" \
DILOCO_OUTER_LR="0.7" \
DILOCO_OUTER_MOMENTUM="0.9" \
bash src/maxtext/trainers/diloco/scripts/run_olmo_qwen3_30b_streaming_diloco.sh
```

- **49 Fragments**: 48 MoE transformer decoder layers + 1 embedding/head fragment ($H=49, P=49$).
- **Grain Pipeline**: Deterministic streaming from pre-tokenized numpy arrays.

______________________________________________________________________

## 6. Checkpointing & Resumption

### Automatic Resumption

To resume an interrupted DiLoCo pre-training run, submit the workload with the same `RUNNAME` and `BASE_OUTPUT_DIRECTORY`:

```bash
RUNNAME="stream-dlco-8b-01" \
XPK_WORKLOAD="dlco-resm-01" \
STEPS="2000" \
bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_diloco.sh
```

MaxText detects existing Orbax checkpoints, restores both the per-replica inner optimizer moments and the outer Nesterov momentum state, and continues training seamlessly.

### Bootstrapping from Single-Slice Weights

To initialize a multi-slice DiLoCo run from standard pre-trained single-slice weights, specify `LOAD_FULL_STATE_PATH`:

```bash
LOAD_FULL_STATE_PATH="gs://your-bucket/checkpoints/base_model/0/items" \
bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_diloco.sh
```

MaxText broadcasts the single-slice model weights across all islands and initializes a clean outer optimizer state automatically.

______________________________________________________________________

## 7. Tuning Guidelines

| Hyperparameter                               | Recommended Setting          | Description                                                                                                                                                                                                                                                                                      |
| :------------------------------------------- | :--------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `diloco_sync_period` ($H$)                   | **`= num_diloco_fragments`** | Sync period. Setting $H = P$ ensures exactly 1 fragment is synchronized every local step ($\Delta h = 1$).                                                                                                                                                                                       |
| `num_diloco_fragments` ($P$)                 | **`num_layers + 1`**         | Partition count (1 for non-scanned embeddings/head + 1 per transformer decoder layer).                                                                                                                                                                                                           |
| `diloco_outer_lr` ($\eta_{\text{outer}}$)    | **`0.3` – `0.9`**            | Outer learning rate. Start from `0.3` – `0.9` (e.g. `0.7`) and tune based on inner LR.                                                                                                                                                                                                           |
| `diloco_outer_momentum` ($\beta$)            | `0.9`                        | Nesterov momentum coefficient for the outer optimizer.                                                                                                                                                                                                                                           |
| `num_communication_overlapping_steps` ($V$)  | `0` (or `1`)                 | Delay in inner steps before applying outer weights. In the current SPMD design, this does not enhance hardware efficiency but simulates the algorithmic behavior of delayed weight merging; it will provide non-blocking hardware overlap in future MPMD multi-threading. Coupled with $\alpha$. |
| `communication_overlapping_alpha` ($\alpha$) | `0.0`                        | Soft parameter blending factor ($\theta_{\text{inner}} \leftarrow \alpha \theta_{\text{inner}} + (1 - \alpha) \theta_{\text{outer}}$). Simulates soft weight interpolation in current SPMD; will be effective alongside $V$ in future MPMD. `0.0` applies direct replacement.                    |

### Practical Tuning Heuristics

- **Synchronize Every Step ($H = P = N_{\text{layers}} + 1$)**:
  Setting `diloco_sync_period` equal to `num_diloco_fragments` with $P = N_{\text{layers}} + 1$ (e.g., $H=37, P=37$ for 36-layer models like Qwen3-8B, or $H=49, P=49$ for 48-layer models like Qwen3-30B) ensures a steady, constant stream of background communications by syncing 1 fragment on every local step.

- **Outer Learning Rate Tuning & Inverse Scaling Rule**:
  Outer LR should be tuned alongside the inner optimizer learning rate. As a core heuristic:

  $$\text{Higher Inner Learning Rate} \implies \text{Lower Outer Learning Rate}$$

  $$\text{Lower Inner Learning Rate} \implies \text{Higher Outer Learning Rate}$$

  When using standard AdamW inner optimization, starting with `diloco_outer_lr: 0.7` is a strong baseline.

- **Overlapping Steps ($V$) and Alpha ($\alpha$) in SPMD vs. MPMD**:
  `num_communication_overlapping_steps` ($V$) and `communication_overlapping_alpha` ($\alpha$) are coupled in defining the asynchronous weight merging policy:

  - **Current SPMD Design**: Because JAX SPMD compiles each step into a synchronous XLA graph, setting $V > 0$ or $\alpha > 0$ **does not enhance hardware training efficiency or hide network latency**. However, it allows researchers to accurately **simulate the algorithmic convergence behavior** of delayed weight merging and soft parameter blending. Setting $V=0$ is standard for performance.
  - **Future MPMD Multi-Threading Design**: In upcoming MPMD architectures with independent background communication threads, $V$ and $\alpha$ will provide true, non-blocking hardware compute/communication overlap.

______________________________________________________________________

## 8. Next Steps

- Deep dive into mathematical foundations: [DiLoCo Theory & Mathematics Reference](../reference/core_concepts/diloco.md).
- Explore input pipeline options: [Data Input Pipeline Guides](../guides/data_input_pipeline.md).
- Learn about sharding strategies on TPUs: [Sharding on TPUs](../guides/optimization/sharding.md).
