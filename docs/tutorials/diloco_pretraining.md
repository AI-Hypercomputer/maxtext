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

```{seealso}
This page is task-oriented: copy a recipe, adjust it, launch. For why the algorithm works, how to size your DCN link, and what each knob does mathematically, see the [DiLoCo Theory & Mathematics Reference](../reference/core_concepts/diloco.md).
```

______________________________________________________________________

## 1. Vanilla DiLoCo vs. Streaming DiLoCo

MaxText supports two modes of distributed low-communication training:

```text
                    inner steps ───────────────────────────▶

Vanilla DiLoCo          H = 4 here, typically 36–500
  compute   ████ ████ ████ ████            ████ ████ ████ ████
  DCN                                 ██████
                                      ▲
                                      └─ whole model at once; compute waits

Streaming DiLoCo        P = 4 fragments, Δh = 1, so H_eff = 4
  compute   ████ ████ ████ ████ ████ ████ ████ ████
  DCN          ▄    ▄    ▄    ▄    ▄    ▄    ▄
               f1   f2   f3   f0   f1   f2   f3
               └─ one fragment per step: same bytes per cycle,
                  1/P the size per transfer, no barrier
```

### Key Differences:

- **Vanilla DiLoCo (`enable_streaming_diloco=false`)**:

  - **How it works**: Each computing island trains independently for $H$ inner steps (e.g., $H=100$). At every $H$-th step, training pauses for a global collective all-reduce where the entire model's pseudo-gradient ($\Delta \theta = \theta_{\text{outer}} - \theta_{\text{inner}}$) is averaged across all islands over DCN and updated using outer Nesterov momentum.
  - **When to use**: Simpler baseline, ideal when $H$ is large (e.g. $H \ge 500$) and the periodic all-reduce pause represents a negligible fraction of total training time.

- **Streaming DiLoCo (`enable_streaming_diloco=true`)**:

  - **How it works**: The model parameters are partitioned into $P$ fragments (typically $P = N_{\text{layers}} + 1$). By setting $H = P$, exactly 1 fragment is synchronized on every single local inner step ($\Delta h = 1$).
  - **When to use**: Optimal for high-throughput scaling across lower-bandwidth DCN/WAN networks, as it eliminates bursty communication spikes and removes the periodic step-$H$ idle barrier.
  - **Requirements**: `scan_layers=true`, `num_diloco_fragments >= 2`, and `num_decoder_layers` divisible by `num_diloco_fragments - 1`. These are validated at startup.

```{admonition} Before your first run: the global batch is split, not replicated
:class: important

With `dcn_diloco_parallelism=K`, each island trains on $GBS/K$ tokens per inner step — adding islands does not multiply your token throughput per step, and $GBS$ must be divisible by $K$ or startup fails.
```

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
  run_name="vanilla-dlco-8b-01" \
  base_output_directory="gs://your-bucket/maxtext-logs" \
  dataset_path="gs://your-bucket/maxtext-datasets" \
  dataset_name='c4/en:3.0.1' \
  model_name="qwen3-8b" \
  tokenizer_type=huggingface \
  tokenizer_path=maxtext/assets/tokenizers/qwen3-tokenizer \
  per_device_batch_size=8 \
  max_target_length=2048 \
  enable_diloco=true \
  enable_streaming_diloco=false \
  dcn_diloco_parallelism=2 \
  diloco_sync_period=100 \
  diloco_outer_lr=0.7 \
  diloco_outer_momentum=0.9 \
  steps=1000 \
  enable_checkpointing=true \
  checkpoint_period=100
```

### Configuration Breakdown:

- `enable_diloco=true`: Enables outer optimization and multi-slice Low-Communication training across `dcn_diloco_parallelism=2` slices. The number of islands is `num_diloco_replicas = ici_diloco_parallelism * dcn_diloco_parallelism`; set `dcn_diloco_parallelism=-1` to infer it from `num_slices`.
- `enable_streaming_diloco=false`: Disables parameter fragmentation and performs full-model pseudo-gradient all-reduce.
- `diloco_sync_period=100`: Islands execute 100 local AdamW steps independently before pausing to sync. Vanilla DiLoCo has no `num_diloco_fragments` / `scan_layers` requirement.
- `diloco_outer_lr=0.7` and `diloco_outer_momentum=0.9`: Outer Nesterov momentum parameters (`optax.sgd(..., nesterov=True)`). Note the MaxText defaults are `0.3` and `0.9`.

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
DILOCO_USE_SEQUENTIAL_LAYERS="false" \
DILOCO_OUTER_LR="0.7" \
DILOCO_OUTER_MOMENTUM="0.9" \
bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_diloco.sh
```

The script builds a Docker image from your local working tree, pushes it, and submits the workload via XPK. It passes `dcn_diloco_parallelism=${NUM_SLICES}`, uses the Grain/TFRecord C4 pipeline, and pins the Qwen3 tokenizer. Add `RESERVATION="<your-reservation>"` if your cluster requires one.

```{admonition} The script's defaults are not this tutorial's recommendations
:class: warning

Omit these and you get different training behavior than the text describes:

| Variable                        | Script default | Recommended here |
| :------------------------------ | :------------- | :--------------- |
| `DILOCO_OUTER_LR`               | `0.1`          | `0.7`            |
| `DILOCO_NUM_COMM_OVERLAP_STEPS` | `2`            | `0`              |

Both are set explicitly in the command above. The overlap-steps default of `2` is the more consequential one: it delays fragment application without any throughput benefit in the current SPMD design.
```

$H = P = 37$ matches Qwen3-8B's 36 decoder layers plus one fragment for the non-scanned embeddings and head, giving $\Delta h = 1$ — one fragment synchronized per step.

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
- **Grain Pipeline**: `dataset_type=olmo_grain`, reading a pre-built index (`OLMO_INDEX_PATH`, default `/tmp/olmo-data/olmo/indices/olmo_index_seq8192.json`). The script mounts `OLMO_GCS_BASE` with gcsfuse at `OLMO_LOCAL_MOUNT` and remaps dataset paths from GCS to the local mount.
- **Derived batch and schedule**: `PER_DEVICE_BATCH_SIZE` defaults to `TARGET_GLOBAL_BATCH / (devices_per_slice * XPK_NUM_SLICES)`, and `STEPS` defaults to a full pass over `TOTAL_INSTANCES`. Override any of these explicitly for shorter runs.

```{admonition} Same outer-LR caveat applies
:class: warning

This script also defaults `DILOCO_OUTER_LR` to `0.1`. The recipe above overrides it to `0.7` explicitly.
```

______________________________________________________________________

## 6. Checkpointing & Resumption

### Automatic Resumption

To resume an interrupted DiLoCo pre-training run, submit the workload with the same `RUNNAME` and `BASE_OUTPUT_DIRECTORY`:

```bash
RUNNAME="stream-dlco-8b-01" \
XPK_WORKLOAD="dlco-resm-01" \
BASE_OUTPUT_DIRECTORY="gs://your-bucket/maxtext-logs" \
STEPS="2000" \
bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_diloco.sh
```

Use a *new* `XPK_WORKLOAD` (workload names must be unique) but the *same* `RUNNAME` and `BASE_OUTPUT_DIRECTORY`, since the checkpoint directory is derived from those. MaxText detects the existing Orbax checkpoint, recognizes it as a multi-replica DiLoCo checkpoint, and restores the per-replica inner optimizer moments together with the outer Nesterov momentum trace.

### Bootstrapping from Single-Slice Weights

To initialize a multi-slice DiLoCo run from standard pre-trained single-slice weights, specify `LOAD_FULL_STATE_PATH`:

```bash
RUNNAME="stream-dlco-8b-boot" \
BASE_OUTPUT_DIRECTORY="gs://your-bucket/maxtext-logs" \
LOAD_FULL_STATE_PATH="gs://your-bucket/checkpoints/base_model/0/items" \
bash src/maxtext/trainers/diloco/scripts/run_spmd_streaming_diloco.sh
```

MaxText recognizes the restored state as a single-replica (non-DiLoCo) checkpoint, broadcasts the weights and optimizer state across all $K$ islands along the `diloco` axis, and initializes a fresh outer Nesterov momentum state. The MoE recipe's script additionally accepts `LOAD_PARAMETERS_PATH` for loading parameters only.

______________________________________________________________________

## 7. Tuning Guidelines

| Hyperparameter                               | Recommended Setting          | Description                                                                                                                                                                                                                                                                                      |
| :------------------------------------------- | :--------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `diloco_sync_period` ($H$)                   | **`= num_diloco_fragments`** | Sync period. Setting $H = P$ ensures exactly 1 fragment is synchronized every local step ($\Delta h = 1$). In general $\Delta h = \max(1, \text{round}(H/P))$ and the effective period is $P \cdot \Delta h$, so prefer an $H$ that is a multiple of $P$.                                        |
| `num_diloco_fragments` ($P$)                 | **`num_layers + 1`**         | Partition count (1 for non-scanned embeddings/head + 1 per transformer decoder layer). Must be $\ge 2$, and `num_decoder_layers` must be divisible by $P - 1$.                                                                                                                                   |
| `use_sequential_layers`                      | `false`                      | Layer-to-fragment assignment. `false` interleaves layers round-robin across fragments (each sync touches the full depth of the network); `true` gives each fragment a contiguous block of layers.                                                                                               |
| `diloco_outer_lr` ($\eta_{\text{outer}}$)    | **`0.3` – `0.9`**            | Outer learning rate. Start from `0.3` – `0.9` (e.g. `0.7`) and tune based on inner LR.                                                                                                                                                                                                           |
| `diloco_outer_momentum` ($\beta$)            | `0.9`                        | Nesterov momentum coefficient for the outer optimizer.                                                                                                                                                                                                                                           |
| `num_communication_overlapping_steps` ($V$)  | `0` (or `1`)                 | Delay in inner steps before applying outer weights. In the current SPMD design, this does not enhance hardware efficiency but simulates the algorithmic behavior of delayed weight merging; it will provide non-blocking hardware overlap in future MPMD multi-threading. Coupled with $\alpha$. |
| `communication_overlapping_alpha` ($\alpha$) | `0.0`                        | Soft parameter blending factor in $[0, 1]$ ($\theta_{\text{inner}} \leftarrow \alpha \theta_{\text{inner}} + (1 - \alpha) \theta_{\text{outer}}$). `0.0` applies direct replacement; `0.5` averages local and global; `1.0` keeps the local fragment, i.e. no effective exchange.                |

### Choosing $P$ for Your Model

$P = N_{\text{layers}} + 1$ is the recommended starting point, but any $P$ satisfying $P \ge 2$ and $N_{\text{layers}} \bmod (P - 1) = 0$ is valid. Larger $P$ means smaller, more frequent transfers:

| Model            | $N_{\text{layers}}$ | Valid $P$ values            | Recommended $P$ | Layers per fragment |
| :--------------- | ------------------: | :-------------------------- | --------------: | ------------------: |
| Qwen3-8B         |                  36 | 2, 3, 4, 5, 7, 10, 13, 19, 37 |              37 |                   1 |
| Qwen3-30B-A3B    |                  48 | 2, 3, 4, 5, 7, 9, 13, 17, 25, 49 |           49 |                   1 |
| A 32-layer model |                  32 | 2, 3, 5, 9, 17, 33          |              33 |                   1 |

Read the table as $P - 1 \in \text{divisors}(N_{\text{layers}})$. If you need a $P$ that doesn't divide evenly, startup fails with an explicit error rather than silently rebalancing.

### Practical Tuning Heuristics

- **Synchronize Every Step ($H = P = N_{\text{layers}} + 1$)**:
  Setting `diloco_sync_period` equal to `num_diloco_fragments` with $P = N_{\text{layers}} + 1$ (e.g., $H=37, P=37$ for 36-layer models like Qwen3-8B, or $H=49, P=49$ for 48-layer models like Qwen3-30B) ensures a steady, constant stream of background communications by syncing 1 fragment on every local step.

- **Keep $H$ a Multiple of $P$**:
  $\Delta h$ is computed as $\max(1, \text{round}(H/P))$, so a period that doesn't divide cleanly is silently rounded. $H=100, P=37$ yields $\Delta h = 3$ and an effective period of $111$ steps — 11% longer than requested. Either set $H = P$, or pick $H$ as an exact multiple of $P$.

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

## 8. Monitoring Convergence

Aggregate scalars (`learning/loss`, throughput, etc.) are reported from replica 0. DiLoCo additionally emits a **per-island loss** for every replica:

```
learning/loss_island_0
learning/loss_island_1
...
learning/loss_island_{K-1}
```

Use these to diagnose the failure mode specific to low-communication training: islands drifting apart between synchronizations. A healthy run shows the per-island losses tracking each other closely and re-converging at each sync. A widening spread — especially one that does not shrink after a sync — usually means $H$ is too large or the outer learning rate is too high.

______________________________________________________________________

## 9. Emulating a Slow Network

To measure DiLoCo's benefit on a cluster whose DCN is *not* actually the bottleneck, MaxText can throttle per-VM egress using a Linux traffic-control token bucket filter:

| Config                     | Default | Description                                                         |
| :------------------------- | :------ | :------------------------------------------------------------------ |
| `dcn_bandwidth_limit`      | `""`    | Per-VM egress limit, e.g. `10gbit`. Empty means no throttling.      |
| `dcn_bandwidth_burst`      | `10mb`  | Token bucket filter burst size.                                     |
| `dcn_bandwidth_latency`    | `50ms`  | Token bucket filter latency threshold.                              |
| `dcn_bandwidth_interface`  | `eth0`  | Network interface the shaping rules are applied to.                 |

This lets you compare synchronous data parallelism against DiLoCo at a realistic WAN bandwidth before committing to a cross-datacenter run.

______________________________________________________________________

## 10. Troubleshooting

All DiLoCo misconfigurations are caught at startup, before any accelerator work begins. The table maps each error to its cause.

| Error message (abridged)                                                              | Cause and fix                                                                                                                       |
| :------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------- |
| `enable_diloco must be True when enable_streaming_diloco is True.`                    | Streaming is a mode of DiLoCo, not a replacement. Set both flags.                                                                   |
| `num_diloco_fragments must be specified when enable_streaming_diloco is True.`        | `num_diloco_fragments` defaults to `null`. Set it explicitly, typically to `num_layers + 1`.                                        |
| `num_diloco_fragments (N) must be at least 2 ...`                                     | You need at least one fragment for non-scanned params and one for layers.                                                           |
| `enable_streaming_diloco=True requires scan_layers=True.`                             | Fragments are slices of the scanned layer stack; unscanned models have no such stack. Enable `scan_layers`, or use Vanilla DiLoCo.  |
| `The number of decoder layers (N) must be divisible by (num_diloco_fragments - 1) ...`| Pick $P$ such that $P - 1$ divides $N_{\text{layers}}$. See the table in §7.                                                        |
| `Batch dimension B is not divisible by num_diloco_replicas K.`                        | The global batch is split across islands. Adjust `per_device_batch_size` or the slice count so $GBS$ is a multiple of $K$.          |
| `Cannot resolve dcn_diloco_parallelism=-1 ...`                                        | `num_slices` isn't divisible by the product of the other DCN axes. Set `dcn_diloco_parallelism` explicitly.                         |

**Run diverges or loss spikes at sync boundaries.** Lower `diloco_outer_lr` first — it is the most sensitive knob, and the inverse scaling rule above means a high inner LR needs a lower outer LR. If per-island losses spread apart steadily, reduce `diloco_sync_period` so islands reconcile more often.

**No throughput gain versus synchronous training.** Confirm the DCN link was actually the bottleneck. On a fast interconnect, DiLoCo mainly adds outer-state memory without a speedup; use `dcn_bandwidth_limit` (§9) to verify the benefit under realistic bandwidth before committing to a multi-datacenter run.

______________________________________________________________________

## 11. Next Steps

- Deep dive into mathematical foundations: [DiLoCo Theory & Mathematics Reference](../reference/core_concepts/diloco.md).
- Explore input pipeline options: [Data Input Pipeline Guides](../guides/data_input_pipeline.md).
- Learn about sharding strategies on TPUs: [Sharding on TPUs](../guides/optimization/sharding.md).
