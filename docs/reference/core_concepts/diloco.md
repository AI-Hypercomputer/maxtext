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

(diloco-theory)=

# DiLoCo and Streaming DiLoCo Theory

This document provides a comprehensive theoretical reference for **DiLoCo (Distributed Low-Communication)** and **Streaming DiLoCo** in MaxText.

```{note}
For step-by-step launch commands and production recipes, see the [DiLoCo Training Tutorial](../../tutorials/diloco_pretraining.md).
```

______________________________________________________________________

## 1. Overview & Motivation

Standard distributed training paradigms (such as Distributed Data Parallelism / FSDP) rely on high-frequency, synchronous collective communications (`all-reduce` or `reduce-scatter`) executed at **every single optimization step**. When scaling across multiple datacenter pods, geographically distributed clusters, or preemptible multi-region compute pools, inter-cluster network bandwidth becomes the primary bottleneck that throttles accelerator compute efficiency.

DiLoCo addresses this challenge through **bi-level optimization**:

1. **Local Inner Loop**: Multiple computing islands (e.g., TPU slices) train independently on their local data shards for $H$ steps (the *inner loop*) using fast local interconnects (such as TPU Inter-Chip Interconnect, ICI).
2. **Global Outer Loop**: Every $H$ steps, islands communicate pseudo-gradients over the slower inter-island network (Data Center Network, DCN) to execute a centralized outer momentum update.

### Comparison of Distributed Training Paradigms

| Feature                           | Synchronous DDP / FSDP               | Vanilla DiLoCo                                              | Streaming DiLoCo                                         |
| :-------------------------------- | :----------------------------------- | :---------------------------------------------------------- | :------------------------------------------------------- |
| **Communication Frequency**       | Every step ($H = 1$)                 | Periodic burst every $H$ steps (e.g., $H = 36$–$500$)       | Pipelined every $\Delta h = \lfloor H/P \rfloor$ steps   |
| **Network Bandwidth Requirement** | High (DCN/WAN bottleneck)            | Low ($100\times$–$500\times$ reduction in volume over time) | Low + constant bandwidth profile                         |
| **Compute Idle Bubbles**          | Frequent stalls on DCN collectives   | Periodic barrier stall at step $H$                          | **Zero stall** (computation overlaps with communication) |
| **Fault Tolerance**               | Single host failure hangs global job | Isolated for $H$ steps; elastic failure recovery            | Isolated + smooth staggered weight blending              |
| **Primary Use Cases**             | Single high-speed pod/slice (ICI)    | Multi-slice, multi-cluster, WAN                             | High-throughput multi-slice & cross-datacenter training  |

______________________________________________________________________

## 2. Arithmetic Intensity & Hardware Rooflines

The communication-to-computation trade-off is governed by **Arithmetic Intensity ($AI$)**:

$$\text{Arithmetic Intensity } (AI) = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}$$

To prevent hardware from stalling on network transfers, the model's operational arithmetic intensity must exceed the physical network's hardware arithmetic intensity:

$$AI_{\text{model}} \ge AI_{\text{hardware}} = \frac{\text{Peak Compute FLOP/s}}{\text{Network Bandwidth (Bytes/s)}}$$

Consider scaling DeepSeek-V3 ($671\text{B}$ MoE) or Qwen3-30B on Google TPU Ironwood / v5p with a global batch size ($GBS$) of $64\text{M}$ tokens across DCN-connected slices:

$$\text{Max DCN Slices} = \frac{GBS}{146{,}000 \times S_{\text{seq\_len}}} \approx 14 \text{ slices}$$

Under standard synchronous data parallelism, scaling beyond 14 DCN slices causes the model's arithmetic intensity to drop below the hardware threshold, making communication the dominant bottleneck.

**DiLoCo bypasses this limit by a factor of $H$**: because inter-slice communication occurs only once every $H$ steps, the effective operational arithmetic intensity scales linearly with $H$:

$$AI_{\text{DiLoCo}} = H \times AI_{\text{standard}}$$

This allows scaling across dozens of TPU slices or low-bandwidth WAN connections without degrading accelerator utilization.

______________________________________________________________________

## 3. Vanilla DiLoCo Algorithm

DiLoCo splits optimization into two distinct levels:

```text
+-----------------------------------------------------------------------------------+
|                        Initial Outer Model Weights (θ_outer)                      |
+-----------------------------------------------------------------------------------+
       |                                                               |
       | Broadcast                                                     | Broadcast
       v                                                               v
+-------------------------------+              +------------------------------------+
|  Island 1 (TPU Slice 1)       |              |  Island 2 (TPU Slice 2)            |
|  Local Inner Loop (Fast ICI)  |              |  Local Inner Loop (Fast ICI)       |
|  Train H steps with AdamW     |              |  Train H steps with AdamW          |
|  θ_1 ← local updates          |              |  θ_2 ← local updates               |
|  Δθ_1 = θ_outer - θ_1         |              |  Δθ_2 = θ_outer - θ_2              |
+-------------------------------+              +------------------------------------+
       |                                                               |
       | Send Pseudo-Gradient (Δθ_1)                                   | Send Pseudo-Gradient (Δθ_2)
       +-------------------------------+-------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|               Global Outer Optimizer (Slow Inter-Island DCN / WAN)                |
|                                                                                   |
|  1. Global All-Reduce Average:  Δθ = (Δθ_1 + Δθ_2) / 2                             |
|  2. Outer Nesterov Momentum:    v  ← β · v + Δθ                                    |
|                                 θ_outer ← θ_outer - η_outer · (Δθ + β · v)        |
+-----------------------------------------------------------------------------------+
       |                                                               |
       +-------------------------------+-------------------------------+
                                       |
                       Broadcast New θ_outer to all Islands
```

### Mathematical Formulation

1. **Inner Optimization (Local per island $k \in \{1, \dots, K\}$)**:
   For local steps $t = 1, \dots, H$:

   $$\theta_{k, t} = \theta_{k, t-1} - \eta_{\text{inner}} \cdot \text{AdamW}(\nabla \mathcal{L}_k(\theta_{k, t-1}))$$

2. **Pseudo-Gradient Computation**:

   $$\Delta \theta_k = \theta_{\text{outer}} - \theta_{k, H}$$

3. **Global All-Reduce**:

   $$\overline{\Delta \theta} = \frac{1}{K} \sum_{k=1}^K \Delta \theta_k$$

4. **Outer Optimizer Step (Nesterov Momentum)**:

   $$v \leftarrow \beta \cdot v + \overline{\Delta \theta}$$

   $$\theta_{\text{outer}} \leftarrow \theta_{\text{outer}} - \eta_{\text{outer}} \cdot (\overline{\Delta \theta} + \beta \cdot v)$$

5. **Broadcast**:

   $$\theta_{k, 0} \leftarrow \theta_{\text{outer}} \quad \forall k$$

______________________________________________________________________

## 4. Streaming DiLoCo: Pipelined Communication Overlapping

While Vanilla DiLoCo reduces total communication volume, it introduces a **periodic barrier**: all workers must pause local training every $H$ steps to exchange full-model weights over DCN.

**Streaming DiLoCo** ([Cherepanov et al., 2025](https://arxiv.org/abs/2501.18512)) eliminates this idle bubble through **pipelined parameter fragmentation and communication overlapping**:

### 1. Parameter Fragmentation ($P$ Fragments)

The model parameters $\Theta$ are partitioned into $P$ disjoint subsets via [`FragmentedTreeManipulator`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/diloco/utils/fragmenter.py):

- **Fragment 0**: Non-scanned parameters (token embeddings, final RMS norm, output projection head).
- **Fragments $1, \dots, P-1$**: Scanned transformer decoder layers partitioned either sequentially or interleaved.

### 2. Staggered Synchronization Schedule

Instead of synchronizing all parameters every $H$ steps, one fragment is synchronized every $\Delta h = \lfloor H / P \rfloor$ steps:

$$\text{Synchronize Fragment } f = \left(\frac{t \bmod H}{\Delta h}\right) \quad \text{when } t > 0 \text{ and } t \bmod \Delta h == 0$$

When $H = P$, $\Delta h = 1$, synchronizing exactly one parameter fragment on **every single step**.

### 3. Asynchronous Apply Delay ($V$)

To overlap the cross-island DCN collective with inner-step computation, the newly updated outer weights for fragment $f$ are merged back into the local replica after a delay of $V$ steps (`num_communication_overlapping_steps`):

$$\text{Apply Fragment } f \text{ at Step } t \quad \text{where } (t - V) \bmod \Delta h == 0$$

### 4. Soft Weight Blending ($\alpha$) & Delayed Merging

An optional interpolation parameter $\alpha$ (`communication_overlapping_alpha`) smoothly blends the local replica weights with the outer synchronized weights:

$$\theta_{\text{inner}}^{(f)} \leftarrow \alpha \cdot \theta_{\text{inner}}^{(f)} + (1 - \alpha) \cdot \theta_{\text{outer}}^{(f)}$$

Setting $\alpha = 0.0$ applies an exact replacement.

```{important}
**SPMD vs. Future MPMD Multi-Threading Design**:
* `num_communication_overlapping_steps` ($V$) and `communication_overlapping_alpha` ($\alpha$) are coupled in defining the asynchronous weight merging policy.
* **Current SPMD Design**: Because JAX SPMD compiles the entire step (compute, collective reduction, and weight update) into a single synchronous XLA graph per step, setting $V > 0$ and $\alpha > 0$ **does not enhance hardware training efficiency or hide network latency**. However, it allows researchers to faithfully **simulate the algorithmic convergence behavior** of delayed weight merging and soft parameter blending on real workloads.
* **Future MPMD Multi-Threading Design**: In future MPMD architectures featuring dedicated background communication threads running independently of the TPU/GPU compute engine, $V$ and $\alpha$ will provide true, non-blocking hardware compute/communication overlap.
```

```
Vanilla DiLoCo:
[------- H steps Compute -------][ Full All-Reduce ][------- H steps Compute -------]
                                   ^^^^^^^^^^^^^^^
                                      Idle Bubble

Streaming DiLoCo:
[ Compute Step 1..k ][ Compute Step k+1..2k ][ Compute Step 2k+1..3k ] ...
  └─ Sync Frag 0 ────┘  └─ Sync Frag 1 ─────┘  └─ Sync Frag 2 ─────┘   (Continuous Fragment Pipelining)
```

______________________________________________________________________

## 5. Pure JAX SPMD & NNX Architecture

MaxText integrates DiLoCo natively with **JAX SPMD and NNX** without relying on external sidecars or multi-controller processes:

### State Representation (`DiLoCoTrainState`)

In [`src/maxtext/trainers/diloco/diloco.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/diloco/diloco.py):

- `inner_state`: Per-replica NNX `TrainState` containing sharded weights and AdamW states with a leading `diloco` mesh axis of dimension $K$ (`dcn_diloco_parallelism`).
- `params`: Global synchronized outer model parameters (PyTree of `Param` leaves).
- `outer_opt_state`: Optax Nesterov momentum state `(TraceState(trace=...), EmptyState())`.
- `step`: Global step tensor.

### Multi-Placement Execution with `drjax`

Local training steps are mapped across replicas using `@drjax.program(placements={"diloco": K})` and `drjax.map_fn`. Collectives across islands use `drjax.reduce_mean` and `drjax.broadcast`.

______________________________________________________________________

## 6. References

1. **DiLoCo**: Douillard, A., Su, Y., Roberts, A., et al. *DiLoCo: Distributed Low-Communication Training of Language Models*. [arXiv:2311.08105](https://arxiv.org/abs/2311.08105), 2023.
2. **Streaming DiLoCo**: Cherepanov, A., et al. *Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch*. [arXiv:2501.18512](https://arxiv.org/abs/2501.18512), 2025.
3. **MaxText Sharding & Arithmetic Intensity Guide**: [Sharding on TPUs](../../guides/optimization/sharding.md).
