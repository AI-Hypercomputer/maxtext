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

```{seealso}
This page covers the *why* and the math. For runnable launch commands and production recipes, see the [DiLoCo Training Tutorial](../../tutorials/diloco_pretraining.md).
```

______________________________________________________________________

## 1. Overview & Motivation

Standard distributed training paradigms (such as Distributed Data Parallelism / FSDP) rely on high-frequency, synchronous collective communications (`all-reduce` or `reduce-scatter`) executed at **every single optimization step**. When scaling across multiple datacenter pods, geographically distributed clusters, or preemptible multi-region compute pools, inter-cluster network bandwidth becomes the primary bottleneck that throttles accelerator compute efficiency.

DiLoCo addresses this challenge through **bi-level optimization**:

1. **Local Inner Loop**: Multiple computing islands (e.g., TPU slices) train independently on their local data shards for $H$ steps (the *inner loop*) using fast local interconnects (such as TPU Inter-Chip Interconnect, ICI).
2. **Global Outer Loop**: Every $H$ steps, islands communicate pseudo-gradients over the slower inter-island network (Data Center Network, DCN) to execute a centralized outer momentum update.

### Comparison of Distributed Training Paradigms

| Feature                       | Synchronous DDP / FSDP               | Vanilla DiLoCo                                   | Streaming DiLoCo                                        |
| :---------------------------- | :----------------------------------- | :----------------------------------------------- | :------------------------------------------------------ |
| **Communicates**              | Every step ($H = 1$)                 | Full model, every $H$ steps                      | One $1/P$ slice of the model, every $\Delta h$ steps    |
| **Bytes per island per step** | $2 \lvert\Theta\rvert b$             | $2 \lvert\Theta\rvert b / H$ (amortized)         | $2 \lvert\Theta\rvert b / H_{\text{eff}}$ (amortized)   |
| **Peak burst size**           | $2 \lvert\Theta\rvert b$             | $2 \lvert\Theta\rvert b$                         | $2 \lvert\Theta\rvert b / P$                            |
| **Compute idle bubbles**      | Frequent stalls on DCN collectives   | Periodic barrier stall every $H$ steps           | No burst barrier; each stall is $1/P$ the size          |
| **Fault tolerance**           | Single host failure hangs global job | Isolated for $H$ steps; elastic failure recovery | Isolated + smooth staggered weight blending             |
| **Primary use cases**         | Single high-speed pod/slice (ICI)    | Multi-slice, multi-cluster, WAN                  | High-throughput multi-slice & cross-datacenter training |

```{admonition} Streaming DiLoCo reduces burst size, not total volume
---
class: note
---
Read the second and third rows together. Vanilla and Streaming DiLoCo move the **same total number of bytes** — each covers the full model exactly once per effective period.

Streaming's contribution is dividing one large transfer into $P$ transfers of $1/P$ the size. That flattens the bandwidth profile, so you provision the link for the average rate instead of the peak. If you are choosing between the two to save on egress charges, they are equivalent; choose Streaming to avoid stalling on a link that cannot absorb the burst.
```

### Notation

| Symbol                | Config knob                           | Meaning                                                     |
| :-------------------- | :------------------------------------ | :---------------------------------------------------------- |
| $K$                   | `num_diloco_replicas`                 | Number of islands (replicas) training independently         |
| $H$                   | `diloco_sync_period`                  | Requested inner steps between synchronizations              |
| $P$                   | `num_diloco_fragments`                | Number of parameter fragments (Streaming only)              |
| $\Delta h$            | derived                               | Inner steps between consecutive fragment syncs              |
| $H_{\text{eff}}$      | derived                               | Effective period: steps to cover all fragments once         |
| $V$                   | `num_communication_overlapping_steps` | Delay, in steps, between syncing a fragment and applying it |
| $\alpha$              | `communication_overlapping_alpha`     | Blend factor between local and global weights at apply time |
| $\eta_{\text{outer}}$ | `diloco_outer_lr`                     | Outer optimizer learning rate                               |
| $\beta$               | `diloco_outer_momentum`               | Outer Nesterov momentum coefficient                         |
| $\lvert\Theta\rvert$  | —                                     | Parameter count                                             |
| $b$                   | —                                     | Bytes per communicated parameter (4 for fp32, 2 for bf16)   |

______________________________________________________________________

## 2. Arithmetic Intensity & Hardware Rooflines

The communication-to-computation trade-off is governed by **Arithmetic Intensity ($AI$)**:

$$\text{Arithmetic Intensity } (AI) = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}$$

To prevent hardware from stalling on network transfers, the model's operational arithmetic intensity must exceed the physical network's hardware arithmetic intensity:

$$AI_{\text{model}} \ge AI_{\text{hardware}} = \frac{\text{Peak Compute FLOP/s}}{\text{Network Bandwidth (Bytes/s)}}$$

For synchronous data parallelism across $K$ DCN-connected slices, each step exchanges the full gradient (one all-reduce over all $|\Theta|$ parameters) while computing over only $GBS/K$ tokens per slice. The per-slice operational intensity therefore *falls* as $K$ grows:

$$AI_{\text{sync}} \approx \frac{6 \cdot |\Theta| \cdot (GBS / K)}{2 \cdot |\Theta| \cdot b_{\text{param}}} = \frac{3 \cdot GBS}{K \cdot b_{\text{param}}}$$

where $b_{\text{param}}$ is the bytes per communicated parameter. Once $AI_{\text{sync}}$ drops below $AI_{\text{hardware}}$, the DCN link — not the accelerator — sets the step time, so adding slices buys no additional throughput. The crossover $K$ depends on your specific $GBS$, gradient dtype, and per-slice DCN egress bandwidth; compute it for your topology rather than assuming a fixed number.

**DiLoCo bypasses this limit by a factor of $H$**: because inter-slice communication occurs only once every $H$ steps, the effective operational arithmetic intensity scales linearly with $H$:

$$AI_{\text{DiLoCo}} = H \times AI_{\text{standard}}$$

This allows scaling across dozens of TPU slices or low-bandwidth WAN connections without degrading accelerator utilization.

### Sizing the Link

The practical question is how much DCN bandwidth a configuration actually demands. Writing $t_{\text{step}}$ for the inner step time, each island must sustain an **average** egress rate of:

$$B_{\text{avg}} = \frac{2 \lvert\Theta\rvert b}{H_{\text{eff}} \cdot t_{\text{step}}}$$

The factor of 2 covers both legs of the exchange: the pseudo-gradient reduction and the broadcast of updated weights back to the islands. This average is identical for Vanilla and Streaming DiLoCo. What differs is the rate needed to avoid stalling during the transfer itself:

$$B_{\text{peak}}^{\text{vanilla}} = \frac{2 \lvert\Theta\rvert b}{t_{\text{step}}}, \qquad B_{\text{peak}}^{\text{streaming}} = \frac{2 \lvert\Theta\rvert b}{P \cdot \Delta h \cdot t_{\text{step}}} = B_{\text{avg}}$$

Vanilla DiLoCo idles on a burst that is $H_{\text{eff}}$ times larger than the per-step average, so a link provisioned at $B_{\text{avg}}$ will stall at every sync. Streaming DiLoCo spreads the same bytes evenly, so its peak *equals* its average — which is precisely what lets you provision the link for the average rate.

**Worked example.** An 8B-parameter model in bf16 ($b = 2$) with $H_{\text{eff}} = 37$ and $t_{\text{step}} = 0.5\,\text{s}$:

$$B_{\text{avg}} = \frac{2 \times 8 \times 10^9 \times 2}{37 \times 0.5} \approx 1.7\,\text{GB/s} \approx 14\,\text{Gbit/s}$$

Under Vanilla DiLoCo the same run moves all $32\,\text{GB}$ inside a single step, demanding $\approx 64\,\text{GB/s}$ ($512\,\text{Gbit/s}$) to avoid a visible bubble — versus $1.7\,\text{GB/s}$ ($14\,\text{Gbit/s}$) sustained for Streaming DiLoCo. Identical total traffic, $37\times$ difference in the link you must provision.

```{admonition} Measure this instead of trusting the formula
---
class: tip
---
The numbers above tell you what to expect; they do not tell you what your cluster does. MaxText can emulate a constrained DCN link on fast hardware so you can observe the trade-off directly: set `dcn_bandwidth_limit` (e.g. `10gbit`) to shape per-VM egress with a Linux traffic-control token-bucket filter, tuned via `dcn_bandwidth_burst`, `dcn_bandwidth_latency`, and `dcn_bandwidth_interface`.

Run the same config at a few bandwidth limits and compare step times. That gives you the real crossover point for your model and topology in an afternoon.
```

______________________________________________________________________

## 3. Vanilla DiLoCo Algorithm

DiLoCo splits optimization into two distinct levels:

```text
                  ┌───────────────────────────────────┐
                  │   outer weights   θ_outer^(s)     │
                  └────────────────┬──────────────────┘
                     broadcast     │     broadcast
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
  ┌────────────────────────────┐        ┌────────────────────────────┐
  │ Island 1                   │        │ Island 2                   │
  │ fast ICI, no DCN traffic   │        │ fast ICI, no DCN traffic   │
  │                            │        │                            │
  │ H inner AdamW steps over   │        │ H inner AdamW steps over   │
  │ its own GBS/K data shard   │        │ its own GBS/K data shard   │
  │            │               │        │            │               │
  │            ▼               │        │            ▼               │
  │ Δθ_1 = θ_outer − θ_1       │        │ Δθ_2 = θ_outer − θ_2       │
  └─────────────┬──────────────┘        └──────────────┬─────────────┘
                │                                      │
                │ pseudo-gradients cross the slow link │
                └──────────────────┬───────────────────┘
                                   ▼      DCN / WAN
      ┌───────────────────────────────────────────────────────┐
      │ outer optimizer — runs once every H steps             │
      │                                                       │
      │  1. all-reduce mean   Δθ_avg = (Δθ_1 + Δθ_2) / K      │
      │  2. Nesterov trace    v ← Δθ_avg + β·v                │
      │  3. outer SGD step    θ_outer ← θ_outer               │
      │                                  − η_outer·(Δθ_avg    │
      │                                              + β·v)   │
      └───────────────────────────┬───────────────────────────┘
                                  │
              broadcast new θ_outer back to every island
                   (parameters only — AdamW moments stay local)
```

### Mathematical Formulation

Let $\theta^{(s)}_{\text{outer}}$ denote the outer weights after $s$ outer rounds, and $\theta_{k,t}$ the weights of island $k$ after $t$ inner steps of the current round.

1. **Inner Optimization (local to island $k \in \{1, \dots, K\}$)**:
   For local steps $t = 1, \dots, H$, starting from $\theta_{k,0} = \theta^{(s)}_{\text{outer}}$:

   $$\theta_{k, t} = \theta_{k, t-1} - \eta_{\text{inner}} \cdot \text{AdamW}\bigl(\nabla \mathcal{L}_k(\theta_{k, t-1})\bigr)$$

   Each island draws its own disjoint $GBS/K$ shard of the batch, so $\mathcal{L}_k$ differs across islands.

2. **Pseudo-Gradient Computation**: the accumulated local progress, expressed as a descent direction:

   $$\Delta \theta_k = \theta^{(s)}_{\text{outer}} - \theta_{k, H}$$

3. **Global All-Reduce** across the `diloco` axis:

   $$\overline{\Delta \theta} = \frac{1}{K} \sum_{k=1}^K \Delta \theta_k$$

4. **Outer Optimizer Step**: `optax.sgd(η_outer, momentum=β, nesterov=True)`. With $v^{(0)} = 0$, the momentum trace and the applied update are:

   $$v^{(s+1)} = \overline{\Delta \theta} + \beta \cdot v^{(s)}$$

   $$\theta^{(s+1)}_{\text{outer}} = \theta^{(s)}_{\text{outer}} - \eta_{\text{outer}} \cdot \bigl(\overline{\Delta \theta} + \beta \cdot v^{(s+1)}\bigr)$$

   The Nesterov form applies the momentum decay *twice* — once building $v^{(s+1)}$, once in the update — which is why the second term uses the freshly updated trace, not the previous one.

5. **Broadcast** the new outer weights back to every island:

   $$\theta_{k, 0} \leftarrow \theta^{(s+1)}_{\text{outer}} \quad \forall k$$

```{admonition} The inner optimizer state survives the barrier
---
class: note
---
Step 5 broadcasts **parameters only**. Each island keeps its own AdamW first- and second-moment estimates across the synchronization, so inner optimization resumes with a warm optimizer instead of re-accumulating moments every $H$ steps. This matches the original DiLoCo formulation, and it is why the sync barrier costs bandwidth but not optimizer convergence.
```

```{admonition} $V$ and $\alpha$ do nothing in Vanilla DiLoCo
---
class: warning
---
`num_communication_overlapping_steps` and `communication_overlapping_alpha` are read **only** by the streaming train step. The vanilla step replaces inner parameters outright at every sync. Setting them with `enable_streaming_diloco=false` is silently ignored — no error, no effect — so if you are tuning them, confirm streaming is actually enabled.
```

______________________________________________________________________

## 4. Streaming DiLoCo: Pipelined Communication Overlapping

While Vanilla DiLoCo reduces total communication volume, it introduces a **periodic barrier**: all workers must pause local training every $H$ steps to exchange full-model weights over DCN.

**Streaming DiLoCo** ([Douillard et al., 2025](https://arxiv.org/abs/2501.18512)) eliminates this idle bubble through **pipelined parameter fragmentation and communication overlapping**:

### 1. Parameter Fragmentation ($P$ Fragments)

The model parameters $\Theta$ are partitioned into $P$ disjoint subsets via [`FragmentedTreeManipulator`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/diloco/utils/fragmenter.py):

- **Fragment 0**: All non-scanned parameters (token embeddings, final norm, output projection head).
- **Fragments $1, \dots, P-1$**: The $N_{\text{layers}}$ scanned decoder layers, split into $P-1$ groups of $N_{\text{layers}} / (P-1)$ layers each. A fragment is a slice (or `jnp.take`) along the scan axis of the stacked layer arrays, not a separate PyTree subtree.

Layers are assigned to fragments in one of two orders:

- **Interleaved (default, `use_sequential_layers=false`)**: fragment $i$ owns layers $\{i-1,\ i-1+(P-1),\ i-1+2(P-1), \dots\}$. A round-robin stride over the depth of the network.
- **Sequential (`use_sequential_layers=true`)**: fragment $i$ owns the contiguous block $[(i-1) \cdot m,\ i \cdot m)$ where $m = N_{\text{layers}} / (P-1)$.

Because fragments are carved out of the *scanned* layer stack, Streaming DiLoCo requires `scan_layers=true`, requires $P \ge 2$, and requires $N_{\text{layers}}$ to be divisible by $P - 1$. All three are enforced at config-validation time.

### 2. Staggered Synchronization Schedule

Instead of synchronizing all parameters every $H$ steps, one fragment is synchronized every $\Delta h$ steps, where $\Delta h$ is derived from the requested sync period $H$ (`diloco_sync_period`) and the fragment count $P$:

$$\Delta h = \max\bigl(1,\ \text{round}(H / P)\bigr), \qquad H_{\text{eff}} = P \cdot \Delta h$$

Note that the **effective** period $H_{\text{eff}}$ — the interval over which all $P$ fragments are covered exactly once — is $P \cdot \Delta h$, which equals the requested $H$ only when $P$ divides $H$. The fragment synchronized at global step $t$ is:

$$f(t) = \left\lfloor \frac{t \bmod H_{\text{eff}}}{\Delta h} \right\rfloor \quad \text{when } t > 0 \text{ and } t \bmod \Delta h = 0$$

When $H = P$, $\Delta h = 1$, synchronizing exactly one parameter fragment on **every single step**.

Each sync touches only fragment $f$: it broadcasts the outer fragment, forms the per-island pseudo-gradient $\theta_{\text{outer}}^{(f)} - \theta_{\text{inner}}^{(f)}$, averages it across islands, and steps the outer optimizer on the corresponding slice of the Nesterov momentum trace. The inner replicas are *not* updated by this step — that is the separate apply stage below.

```{admonition} A sync period that isn't a multiple of $P$ is silently rounded
---
class: warning
---
$\Delta h$ is an integer, so $H$ is effectively rounded to the nearest multiple of $P$ — with no warning at startup. Requesting $H = 100$ with $P = 37$ gives $\Delta h = \text{round}(2.70) = 3$ and an actual period of $H_{\text{eff}} = 111$ steps, 11% longer than you asked for.

Set $H = P$, or an exact multiple of $P$, so that $H_{\text{eff}} = H$ and the number you configured is the number you get.
```

#### Worked Schedule

For $P = 4$ and $H = 12$ (so $\Delta h = 3$, $H_{\text{eff}} = 12$), with apply delay $V = 1$:

| Step $t$ | $t \bmod \Delta h$ | Sync fragment | Apply fragment ($V=1$) |
| -------: | -----------------: | :------------ | :--------------------- |
|        3 |                  0 | 1             | —                      |
|        4 |                  1 | —             | 1                      |
|        6 |                  0 | 2             | —                      |
|        7 |                  1 | —             | 2                      |
|        9 |                  0 | 3             | —                      |
|       10 |                  1 | —             | 3                      |
|       12 |                  0 | 0             | —                      |
|       13 |                  1 | —             | 0                      |

Two properties are worth noting. Fragment 0 synchronizes at $t = 12$, not $t = 0$, because the sync condition requires $t > 0$ — so the first full cycle completes one $\Delta h$ later than a naive reading suggests. And at steps that are neither a sync nor an apply step ($t = 5, 8, 11, \dots$) no cross-island traffic occurs at all; the islands train purely locally.

### 3. Asynchronous Apply Delay ($V$)

The newly updated outer weights for a fragment are merged back into the local replicas as a distinct stage, delayed by $V$ steps (`num_communication_overlapping_steps`). The apply stage fires when:

$$t - V > 0 \quad \text{and} \quad (t - V) \bmod \Delta h = 0, \qquad \text{applying fragment } f(t-V) = \left\lfloor \frac{(t - V) \bmod H_{\text{eff}}}{\Delta h} \right\rfloor$$

With $V = 0$ the sync and apply of a fragment happen in the same step. With $V > 0$ the inner replicas keep training on stale weights for $V$ steps after the fragment's pseudo-gradient was taken — which is exactly the window a real asynchronous implementation would use to hide the collective.

### 4. Soft Weight Blending ($\alpha$) & Delayed Merging

An optional interpolation parameter $\alpha$ (`communication_overlapping_alpha`, range $[0, 1]$) blends the local replica weights with the outer synchronized weights at apply time:

$$\theta_{\text{inner}}^{(f)} \leftarrow \alpha \cdot \theta_{\text{inner}}^{(f)} + (1 - \alpha) \cdot \theta_{\text{outer}}^{(f)}$$

- $\alpha = 0.0$ (default): exact replacement — the inner fragment is overwritten by the outer fragment, discarding the inner updates accumulated during the $V$-step delay. The blend is skipped entirely as a fast path.
- $\alpha = 0.5$: uniform average of local and global fragment parameters.
- $\alpha = 1.0$: the local fragment is kept unchanged, i.e. the islands effectively stop exchanging that fragment.

```{admonition} $V$ and $\alpha$ are research knobs today, not performance knobs
---
class: important
---
JAX SPMD compiles compute, collective reduction, and weight update into a **single synchronous XLA graph per step**. A delayed apply still executes inside that graph, so the collective never leaves the critical path. Setting $V > 0$ or $\alpha > 0$ will **not** hide network latency or improve step time in the current design — if you are tuning for throughput, leave both at their defaults.

Their value is scientific: they reproduce the convergence behavior of delayed merging and soft blending, letting you measure the accuracy cost of asynchrony before the hardware can deliver its speedup. A future MPMD design with dedicated background communication threads will give these same two parameters true non-blocking overlap, so a configuration characterized now carries forward unchanged.
```

```text
                     inner steps ──────────────────────────▶

Vanilla DiLoCo      H_eff = 12
  compute  ████████████████████████████      ████████████████████
  DCN                                  ██████
                                       ▲
                                       │ one transfer of 2|Θ|b
                                       └ compute is blocked for its duration

Streaming DiLoCo    P = 4, Δh = 3, H_eff = 12  — identical total bytes
  compute  ██████████████████████████████████████████████████████
  DCN         ▄▄       ▄▄       ▄▄       ▄▄       ▄▄       ▄▄
              f1       f2       f3       f0       f1       f2
           t= 3        6        9       12       15       18

           each transfer is 2|Θ|b/P — one quarter the size, four times
           as often, so the link carries a flat load instead of spikes
```

______________________________________________________________________

## 5. Pure JAX SPMD & NNX Architecture

MaxText integrates DiLoCo natively with **JAX SPMD and NNX** without relying on external sidecars or multi-controller processes:

### State Representation (`DiLoCoTrainState`)

In [`src/maxtext/trainers/diloco/diloco.py`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/diloco/diloco.py):

- `inner_state`: Per-replica NNX `TrainState` containing sharded weights and AdamW states with a leading `diloco` mesh axis of dimension $K$.
- `params`: Global synchronized outer model parameters (PyTree of `Param` leaves).
- `outer_opt_state`: Optax Nesterov momentum state `(TraceState(trace=...), EmptyState())`, produced by `optax.sgd(diloco_outer_lr, momentum=diloco_outer_momentum, nesterov=True)`.
- `step`: Global step tensor.

### Replica Count and the `diloco` Mesh Axis

$K$ is `num_diloco_replicas`, derived at config resolution time as the product of the two `diloco` parallelism axes:

```
num_diloco_replicas = ici_diloco_parallelism × dcn_diloco_parallelism
```

`diloco` is the leading entry of `mesh_axes`. In the typical multi-slice setup, `dcn_diloco_parallelism` is the number of slices (islands) and `ici_diloco_parallelism` stays at `1`; setting `dcn_diloco_parallelism=-1` resolves it from `num_slices` divided by the product of the other DCN axes. Using `ici_diloco_parallelism > 1` places multiple islands inside a single slice, which is useful for single-slice functional testing of DiLoCo semantics.

### Batch Splitting Across Islands

DiLoCo does **not** replicate the global batch per island. The incoming batch is reshaped from $(GBS, \dots)$ to $(K, GBS/K, \dots)$ along the new `diloco` axis, so each island consumes a disjoint $GBS/K$ slice of tokens per inner step. `GBS` must be divisible by $K$.

### Multi-Placement Execution with `drjax`

Local training steps are mapped across replicas using `@drjax.program(placements={"diloco": K})` and `drjax.map_fn`. Collectives across islands use `drjax.reduce_mean` and `drjax.broadcast`.

Every step executes in a fixed order inside one compiled graph:

1. `drjax.map_fn(train_step, ...)` — all $K$ islands take one independent inner step.
2. **Sync stage** (`jax.lax.cond` on the sync predicate) — vanilla syncs the full state; streaming selects one fragment with `jax.lax.switch` over $P$ branches.
3. **Apply stage** (streaming only, a second `lax.cond`) — merges the synced fragment back into the inner replicas.

Because both stages are `lax.cond` branches rather than Python conditionals, the sync logic is part of the traced graph and the step function is compiled once, not recompiled per schedule position.

### Memory Cost

The inner state gains a leading axis of size $K$, but that axis is sharded over the `diloco` mesh axis — each island physically holds only its own replica, so per-island inner-state memory (weights + AdamW moments) is unchanged relative to a non-DiLoCo run.

The outer state is the additional cost. Both `params` and the outer momentum `trace` are sharded *without* the `diloco` axis, meaning each island carries a full replicated copy of each:

$$\text{extra resident bytes per island} \approx 2 \lvert\Theta\rvert b_{\text{outer}}$$

Budget for this when sizing a run: DiLoCo is not memory-free, and the overhead is proportional to model size rather than to $K$.

### Observability

Reported scalars come from replica 0. In addition, a per-island loss is emitted as `learning/loss_island_{i}` for each $i \in \{0, \dots, K-1\}$, which is the primary signal for spotting islands that are diverging between synchronizations.

______________________________________________________________________

## 6. References

1. **DiLoCo**: Douillard, A., Feng, Q., Rusu, A. A., et al. *DiLoCo: Distributed Low-Communication Training of Language Models*. [arXiv:2311.08105](https://arxiv.org/abs/2311.08105), 2023.
2. **Streaming DiLoCo**: Douillard, A., Donchev, Y., Rush, K., et al. *Streaming DiLoCo with overlapping communication: Towards a Distributed Free Lunch*. [arXiv:2501.18512](https://arxiv.org/abs/2501.18512), 2025.
3. **MaxText Sharding & Arithmetic Intensity Guide**: [Sharding on TPUs](../../guides/optimization/sharding.md).
