<!--
 Copyright 2026 Google LLC

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

# FSDP weight prefetching

With FSDP, every decoder layer's weights are stored sharded over the `fsdp` mesh axes (and any
co-sharded axes such as `expert` or `context`) and must be all-gathered before the layer runs, with
their weight gradients reduce-scattered after the layer's backward pass. By default XLA schedules
these collectives inside each scanned layer's `while` loop body, and depending on the model,
sharding, and remat policy they can end up exposed on the critical path.

`prefetch_fsdp_weights=True` replaces the scanned layer loop with a `jax.custom_vjp` pipeline
(`src/maxtext/layers/fsdp_prefetch.py` for homogeneous stacks and
`src/maxtext/layers/fsdp_prefetch_inhomogeneous.py` for inhomogeneous blocks) that overlaps the
collectives with compute explicitly across layers:

| Pass     | Layer / sub-layer *k*                                                           |
| -------- | ------------------------------------------------------------------------------- |
| Forward  | all-gather W(*k*+1) ‖ compute layer *k*                                         |
| Backward | all-gather W(*k*−1) ‖ recompute + vjp of layer *k*, then reduce-scatter dW(*k*) |

**Without weight prefetching (`prefetch_fsdp_weights=False`):**

![FSDP timeline without weight prefetching](../../_static/fsdp_no_prefetch_overview.svg)

**With weight prefetching (`prefetch_fsdp_weights=True`):**

![FSDP timeline with weight prefetching](../../_static/fsdp_prefetch_overview.svg)

Comparing the two forward-pass timelines:

- **Without prefetching (`prefetch_fsdp_weights=False`):** Each iteration `i` of `scanned_layers`
  issues `AG_i` on SparseCore 0 to gather the *current* layer's weights (`W_i`). Because `AG_i`
  cannot be hoisted across the scan `while` loop boundary, TensorCore stalls on exposed
  `all-gather-done` barriers (`EXPOSED`) before the layer's attention and grouped matmul (`gmm_i`)
  compute can complete, and SparseCore 0 also shares its queue with data-formatting and gather
  offloads.
- **With prefetching (`prefetch_fsdp_weights=True`):** While TensorCore computes `layer i`
  (`splash_attention`, routing, and `gmm_i`) using weights `W_i` that were already prefetched during
  `layer i−1`, SparseCore 0—dedicated to FSDP collectives via
  `--xla_tpu_explicit_fsdp_dedicated_sparse_core_id=0`—concurrently executes `AG_i+1` for the *next*
  layer (`layer i+1`), while SparseCore 1 handles `layer i`'s own data formatting and MoE token
  gathers (`OFFLOAD_GATHER`). By the time `layer i+1` begins on TensorCore, its weights are already
  in HBM, eliminating the exposed `all-gather-done` stall and allowing layers `i` through `i+3` to
  execute back-to-back.

Both passes peel the boundary layers and run a symmetric `length - 2` `jax.lax.scan(..., unroll=2)`
loop (`reverse=True` in the backward pass). Each 2-layer unrolled iteration alternates between two
gathered-weight buffers and orders the second prefetch behind the first layer via
`jax.lax.optimization_barrier` (when the unrolled loop has at least two trips) so that W(*k*) is
dead before `all-gather W(k+2)` starts and reuses its buffer. In the backward pass, dW(*k*) is
reduce-scattered before the per-step optimization barrier so both unrolled steps lower their weight
gradient reductions to `reduce-scatter`s rather than full `all-reduce`s.

Gathered weights are never saved between the forward and backward pass: the forward pass holds at
most **two** layers (or sub-layers) of gathered weights at once (current + prefetched), and the
backward pass holds at most **three** (W(*k*), W(*k*−1), and dW(*k*) until its reduce-scatter).

## Inhomogeneous blocks and nested scans

Some models scan composite blocks made up of heterogeneous sub-layers with different parameter
structures. For example, in **Qwen3.5** (`Qwen3NextScannableBlock`), one outer scan step is a block
of 3 local Gated DeltaNet (linear attention) sub-layers (`local_layers`, which itself is an
inner-scanned module) followed by 1 global full-attention sub-layer (`global_layer`). Prefetching an
entire block as one unit would gather all 4 sub-layers' weights at once and keep 2–3 full blocks of
gathered weights live in HBM.

When every parameter of a scanned block lives inside its sub-layers (`local_layers` + `global_layer`,
or a block defining `prefetch_sublayer_names()`), `NNXDecoder` routes the stack to
`src/maxtext/layers/fsdp_prefetch_inhomogeneous.py`, which prefetches **one sub-layer at a time**
while preserving the block's nested scan structure:

- **Cross-boundary pipelining:** In the forward pass, the all-gather of sub-layer *j*+1 overlaps the
  compute of sub-layer *j*, including across block boundaries (the last sub-layer of block *b*
  prefetches sub-layer 0 of block *b*+1). In the backward pass, the all-gather of sub-layer *j*−1
  overlaps the recompute + vjp of sub-layer *j*, followed by a reduce-scatter of dW(*j*). Peak live
  gathered weights remain bounded to **two sub-layers** in the forward pass and **three sub-layers**
  in the backward pass.
- **Preserving nested scans with `(n - 1) + 1` loop splitting and `unroll=2` double buffering:**
  Consecutive sub-layers that share the same NNX `graphdef`, parameter PyTree structure, and shapes
  form a group of size *n* stacked as `[length, n, ...]`. Within a group of *n* sub-layers, the *n*−1
  sub-layers that prefetch another sub-layer of the *same* group share one inner `jax.lax.scan`,
  while the remaining sub-layer (which prefetches a sub-layer of the adjacent group with a different
  parameter structure) runs in its own single-trip inner scan. Same-group inner scans with at least
  2 steps unroll by 2 (`_sublayer_scan(..., unroll=2)`, running two sub-layers per loop trip with an
  `optimization_barrier` between them while preserving the inner `while` loop even for a single
  2-step trip) so that consecutive sub-layers alternate between two gathered-weight buffers without
  loop-carry weight copies. For Qwen3.5 (3 GDN + 1 global attention):
  - **Forward block body:** `[GDN 0, GDN 1]` (2-step unrolled scan) → `[GDN 2]` (1-trip scan,
    prefetches `attention`) → `[attention]` (1-trip scan, prefetches next block's `GDN 0`).
  - **Backward block body:** `[attention]` (1-trip scan, prefetches `GDN 2`) → `[GDN 2, GDN 1]`
    (2-step unrolled reverse scan) → `[GDN 0]` (1-trip scan, prefetches previous block's
    `attention`).
- **Loop-boundary scheduling fences:** Every inner sub-layer scan is annotated with
  `skip-simplify-while-loops_trip-count-one=true`, and single-trip loop inputs ride in scan `xs`.
  XLA therefore keeps every sub-layer group (including single-trip ones) as a real `while` loop
  nested inside the outer block `while` loop, preventing prefetched all-gathers from being hoisted
  or completed ahead of the previous sub-layer's compute.
- **Whole-block fallback (`skip_block_remat`):** Scanned blocks that carry parameters outside their
  sub-layers (or do not define `prefetch_sublayer_names()`, such as Gemma4 or DeepSeek-V4 blocks)
  are prefetched as a whole via `fsdp_prefetch.py`. When such blocks rematerialize their own
  sub-layers internally (`skip_block_remat=True`), the prefetch pipeline automatically extends the
  remat policy to also save each sub-layer's `decoder_layer_input`, so the backward pass recomputes
  each sub-layer once from its input rather than recomputing the outer block first.

## Remat policies

`remat_policy` (including `full`, `minimal`, `save_dot_except_mlp`, and `custom` policies with
`device` or `offload` activations) applies to each decoder layer (or each sub-layer in the
inhomogeneous path). At trace time, the pipeline abstractly traces `jax.vjp` of the checkpointed
layer once per stack, classifies each residual leaf, and carries **only** the activations saved by
the policy from the forward pass to the backward pass alongside the layer inputs. The backward pass
rebuilds each layer's VJP from those saved activations and the re-gathered weights instead of
recomputing the entire layer.

With `remat_policy=full`, only each layer's input activation is saved. Gathered weights are never
saved regardless of policy: if a policy saves a residual whose shape matches a gathered weight, a
warning is logged. The number and size (in MiB) of residuals saved per layer are logged at trace
time.

## Expert and context parallelism

Under expert parallelism (`ici_expert_parallelism > 1`) or context parallelism, non-routed weights
(such as MLA/attention projections, dense MLP layers, and shared experts) are often co-sharded over
compound axes like `('fsdp', 'expert')` or `('fsdp', 'context')` on their embedding dimensions.

`sharding.fsdp_axes_for_dim` treats `expert`, `context`, and `context_usp_ulysses` as FSDP
co-sharding axes (`sharding.FSDP_CO_AXES`) whenever they follow an FSDP axis (`fsdp` or
`fsdp_transpose`) on the same tensor dimension:

- Dimensions with an FSDP axis strip the FSDP axis and any trailing co-sharding axes, gathering all
  of them in a **single** `xla_explicit_fsdp`-tagged `all-gather` (for example, a 128-way gather
  over `('fsdp', 'expert')`).
- Dimensions without an FSDP axis (such as the routed-expert dimension `'expert'`) and axes that
  precede the first FSDP axis (such as `'expert'` in `('expert', 'fsdp')` for flat MoE weights) keep
  their sharding, so routed experts remain EP-sharded across devices and only gather `'fsdp'`.
- This works with the default `base.yml` rules (`EP_AS_FSDP`), `custom_mesh_and_rule=ep-as-dp`
  (`EP_AS_DP`, where non-routed weights are replicated over `'expert'`), and
  `use_ring_of_experts=True`.

## Recommended flags for best performance

### MaxText flags

- **`prefetch_fsdp_weights=True` and `scan_layers=True`:** Enables the FSDP weight prefetch
  pipeline on scanned decoder layers.
- **`shard_mode=explicit`:** Strongly recommended. With `shard_mode=explicit`, each parameter's
  `NamedSharding` is carried in the traced array types inside `custom_vjp`, allowing
  `_explicit_fsdp_all_gather` to emit `jax.shard_map` + `jax.lax.all_gather(tiled=True)` calls
  tagged with `xla_explicit_fsdp="yes"` and ensuring predictable SparseCore `reduce-scatter`
  lowering in the backward pass. (With `shard_mode=auto`, the pipeline is numerically identical but
  falls back to untagged resharding when shardings are not visible on traced types.)
- **`moe_flat_fsdp_weights=True` *(MoE models)*:** Eliminates FSDP all-gather and reduce-scatter
  relayout copies on routed-expert weights.
  - *Why it matters:* In the default 3D layout `[E, mlp, embed]` sharded
    `P('expert', 'tensor', 'fsdp')`, FSDP shards the minor-most dimension of `wo`. When the per-shard
    width (for example, `7168 / 128 = 56` on DeepSeek-V3) is narrower than the 128-wide TPU tile,
    SparseCore `all-gather` / `reduce-scatter` operate in layout `{1, 0, 2}` while `gmm` / `tgmm`
    need `{2, 1, 0}`, inserting multi-GiB SparseCore data-format transposes and TensorCore copies on
    every layer.
  - *What it does:* Stores routed-expert weights as 2D `[E * rows, cols]` sharded
    `P(('expert', 'fsdp'), ...)` so FSDP shards major dimension 0, making the reshape to
    `[E_local, rows, cols]` inside `shard_map` a zero-copy bitcast and removing all FSDP relayout
    copies around the MoE weights and weight gradients.
  - *Requirements:* `sparse_matmul=True`, `prefuse_moe_weights=False`, and either
    `shard_embed_moe_on_fsdp=True` or unquantized weights (pair with `use_tokamax_gmm=True` and
    `use_gmm_v2=True` for best MoE kernel performance).
- **`remat_policy=custom` *(when HBM headroom allows)*:** Saving attention activations (`mla_q`,
  `mla_kv`, `out_proj`, `attention_out`) or MoE activations (`moe_mlpwi_0`, `moe_mlpwi_1`,
  `moe_mlpwo`, `moe_x_sorted`) on `device` (or `offload`) stacks with prefetching, cutting backward
  recompute (for example, halving backward grouped matmuls from 6 to 3 per MoE layer) while keeping
  gathered weights unsaved.
- **`use_ring_of_experts=True` or `custom_mesh_and_rule=ep-as-dp` *(with `ici_expert_parallelism > 1`)*:**
  Combines cleanly with `prefetch_fsdp_weights=True` to reduce EP dispatch/combine overhead or avoid
  over-sharding small attention projections across `('fsdp', 'expert')`.

### XLA flags (`LIBTPU_INIT_ARGS`)

The prefetched weight gathers are tagged with the `xla_explicit_fsdp` XLA metadata. On TPUs with
SparseCores, set the following flags in `LIBTPU_INIT_ARGS`:

```bash
export LIBTPU_INIT_ARGS="\
  --xla_tpu_enable_fsdp_latency_hiding_scheduler=true \
  --xla_tpu_explicit_fsdp_dedicated_sparse_core_id=0 \
  --xla_tpu_rerun_latency_hiding_scheduler_post_sc_assignment=false"
```

| XLA flag                                                            | Purpose                                                                                                                                                                                                                                                           |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--xla_tpu_enable_fsdp_latency_hiding_scheduler=true`               | Enables XLA's FSDP-aware latency-hiding scheduler, which schedules `xla_explicit_fsdp`-tagged `all-gather`s across the current layer's compute window.                                                                                                            |
| `--xla_tpu_explicit_fsdp_dedicated_sparse_core_id=0`                | Assigns `xla_explicit_fsdp`-tagged `all-gather`s to a dedicated SparseCore (core 0) so they do not queue behind the layer's own SparseCore work (MoE token gathers/scatters, EP `ragged-all-to-all`, Ring-of-Experts collectives, or backward `reduce-scatter`s). |
| `--xla_tpu_rerun_latency_hiding_scheduler_post_sc_assignment=false` | Recommended with Expert Parallelism (`ici_expert_parallelism > 1`); prevents a post-SparseCore-assignment scheduler pass from pulling `all-gather-done` ahead of the layer's MoE compute.                                                                         |

These flags require a `libtpu` build that includes the FSDP latency-hiding scheduler. Without them
the pipeline still runs correctly, but the prefetch collectives are scheduled by the default latency
hiding scheduler on shared SparseCore queues. Under Expert Parallelism (`ici_expert_parallelism=4`),
omitting `--xla_tpu_explicit_fsdp_dedicated_sparse_core_id=0` causes the prefetched gathers to queue
behind the layer's `ragged-all-to-all` and other SparseCore work, reducing the prefetch speedup from
12.4% (1743.0 ms/step) to 1.7% (1955.7 ms/step,
[xid/301803674](https://xmanager.corp.google.com/experiments/301803674),
[trace](https://xprof.corp.google.com/trace_viewer/darisoy-6258667599370129346)).

### Example command

```bash
export LIBTPU_INIT_ARGS="\
  --xla_tpu_enable_fsdp_latency_hiding_scheduler=true \
  --xla_tpu_explicit_fsdp_dedicated_sparse_core_id=0 \
  --xla_tpu_rerun_latency_hiding_scheduler_post_sc_assignment=false"

python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  ... \
  scan_layers=True \
  shard_mode=explicit \
  prefetch_fsdp_weights=True \
  moe_flat_fsdp_weights=True \
  use_tokamax_gmm=True \
  use_gmm_v2=True
```

## Requirements and limitations

- Only the scanned layer stack is affected: `scan_layers=True` is required.
- FSDP must be in use: when no FSDP mesh axis (`fsdp`, `fsdp_transpose`) spans more than one
  device, there is nothing to prefetch and the regular scan is used.
- `save_fsdp_gathered_weights` has no effect while the prefetch path is active: the prefetch
  pipeline never saves gathered weights.
- The prefetch path is skipped (the regular scan is used) when KV caches are passed (inference),
  with forced expert routing, with `parameter_memory_host_offload=True`, or when quantization
  statistics updates are disabled.
- Non-parameter layer state is read-only in this path: state mutated inside a layer call (for
  example sown intermediates) is not propagated back.

## Results

### DeepSeek-V3 (FSDP + `moe_flat_fsdp_weights`)

DeepSeek-V3 architecture (`deepseek3-671b`) with 15 decoder layers (3 dense + 12 MoE,
`base_num_decoder_layers=15`), 128 routed experts (`num_experts=128`), and `num_experts_per_tok=4`
on 64 TPU7x chips (4x4x4), `ici_fsdp_parallelism=-1`, `shard_mode=explicit`,
`per_device_batch_size=2`, `max_target_length=4096`, splash attention blocks 1024 forward / 2048
backward, Tokamax `gmm_v2` (`use_tokamax_gmm=True use_gmm_v2=True`), and full remat
(`remat_policy=custom` saving only `decoder_layer_input`):

| Configuration                                                                                                                             | Step time     | Speedup    | Peak HBM     | Exposed FSDP AG wait | Exposed FSDP RS wait | Exposed SC data-format wait | TC `copy`    | XProf                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ---------- | ------------ | -------------------- | -------------------- | --------------------------- | ------------ | -------------------------------------------------------------------------------- |
| Baseline, `prefetch_fsdp_weights=False` ([xid/302727885](https://xmanager.corp.google.com/experiments/302727885))                         | 2702.9 ms     | —          | 38.2 GiB     | 380.2 ms             | 393.0 ms             | 12.3 ms                     | 276.9 ms     | [trace](https://xprof.corp.google.com/trace_viewer/darisoy-4220726043009085028)  |
| `prefetch_fsdp_weights=True`, 3D expert weights ([xid/302729431](https://xmanager.corp.google.com/experiments/302729431))                 | 2489.4 ms     | −7.9%      | 58.5 GiB     | 18.8 ms              | 345.5 ms             | 116.5 ms                    | 313.1 ms     | [trace](https://xprof.corp.google.com/trace_viewer/darisoy-14051115723292120025) |
| **`prefetch_fsdp_weights=True` + `moe_flat_fsdp_weights=True`** ([xid/302730209](https://xmanager.corp.google.com/experiments/302730209)) | **2172.4 ms** | **−19.6%** | **53.2 GiB** | **19.9 ms**          | **268.2 ms**         | **57.0 ms**                 | **150.3 ms** | [trace](https://xprof.corp.google.com/trace_viewer/darisoy-7381716295166882283)  |

With `remat_policy=custom` saving additional activations on device (`per_device_batch_size=2`,
`num_experts_per_tok=4`, 3D expert weights), prefetching stacks with the saved recompute:

| `remat_policy=custom`, saved on device                            | Baseline                                                                                                                                                            | `prefetch_fsdp_weights`                                                                                                                                            | Speedup | Peak HBM (baseline / prefetch) |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | ------------------------------ |
| `decoder_layer_input` only                                        | 2955 ms ([xid/299684118](https://xmanager.corp.google.com/experiments/299684118), [trace](https://xprof.corp.google.com/trace_viewer/darisoy-10753850547319992397)) | 2650 ms ([xid/299810082](https://xmanager.corp.google.com/experiments/299810082), [trace](https://xprof.corp.google.com/trace_viewer/darisoy-9862584484002790547)) | −10.3%  | 38.4 / 58.4 GiB                |
| + attention (`mla_q`, `mla_kv`, `out_proj`, `attention_out`)      | 2861 ms ([xid/299809939](https://xmanager.corp.google.com/experiments/299809939), [trace](https://xprof.corp.google.com/trace_viewer/darisoy-562152379021556752))   | 2507 ms ([xid/299809597](https://xmanager.corp.google.com/experiments/299809597), [trace](https://xprof.corp.google.com/trace_viewer/darisoy-6608057643495804677)) | −12.4%  | 46.6 / 58.8 GiB                |
| + MoE (`moe_mlpwi_0`, `moe_mlpwi_1`, `moe_mlpwo`, `moe_x_sorted`) | 2703 ms ([xid/299811090](https://xmanager.corp.google.com/experiments/299811090), [trace](https://xprof.corp.google.com/trace_viewer/darisoy-3172849855337006300))  | 2418 ms ([xid/299810085](https://xmanager.corp.google.com/experiments/299810085), [trace](https://xprof.corp.google.com/trace_viewer/darisoy-905718878884127546))  | −10.5%  | 44.2 / 66.1 GiB                |

### DeepSeek-V3 with Expert Parallelism (`ici_expert_parallelism=4`)

DeepSeek-V3 with 15 decoder layers, 256 routed experts (`num_experts=256`), `num_experts_per_tok=4`,
`ici_expert_parallelism=4`, `ici_fsdp_parallelism=-1`, `per_device_batch_size=1`,
`max_target_length=4096`, `shard_mode=explicit`, and `remat_policy=custom` on 64 TPU7x chips (4x4x4)
with the recommended XLA flags above:

| Configuration                                                              | Mean step time | Speedup    | Exposed FSDP AG wait | Peak HBM | Workload                                                                | XProf                                                                            |
| -------------------------------------------------------------------------- | -------------- | ---------- | -------------------- | -------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Baseline (`prefetch_fsdp_weights=False`)                                   | 1989.0 ms      | —          | 384 ms/step          | 23.4 GiB | [xid/301769330](https://xmanager.corp.google.com/experiments/301769330) | [trace](https://xprof.corp.google.com/trace_viewer/darisoy-4547161128859987525)  |
| `prefetch_fsdp_weights=True` (`EP_AS_FSDP`, `base.yml`)                    | 1743.0 ms      | **−12.4%** | **16 ms/step**       | 39.9 GiB | [xid/302206654](https://xmanager.corp.google.com/experiments/302206654) | [trace](https://xprof.corp.google.com/trace_viewer/darisoy-9665020563215820280)  |
| `prefetch_fsdp_weights=True` (`EP_AS_DP`, `custom_mesh_and_rule=ep-as-dp`) | 1721.7 ms      | **−13.4%** | **23 ms/step**       | 40.0 GiB | [xid/302234879](https://xmanager.corp.google.com/experiments/302234879) | [trace](https://xprof.corp.google.com/trace_viewer/darisoy-17383386889066645177) |
| `prefetch_fsdp_weights=True` + `use_ring_of_experts=True`                  | **1683.9 ms**  | **−15.3%** | **23 ms/step**       | 38.8 GiB | [xid/302288817](https://xmanager.corp.google.com/experiments/302288817) | [trace](https://xprof.corp.google.com/trace_viewer/darisoy-8791179500674840887)  |

### Qwen3.5 inhomogeneous nested scan (`qwen3_5-35b-a3b`)

Qwen3.5-35B-A3B (blocks of 3 Gated DeltaNet sub-layers + 1 global attention sub-layer) on 64 TPU7x
chips (4x4x4), `ici_fsdp_parallelism=-1`, `shard_mode=explicit`, `scan_layers=True`,
`remat_policy=custom`, `per_device_batch_size=1`, `max_target_length=1024`, and Tokamax `gmm_v2`
(`use_tokamax_gmm=True use_gmm_v2=True`):

| Configuration                                                                                      | Step time     | Speedup    | Workload                                                                | XProf                                                                                                                 |
| -------------------------------------------------------------------------------------------------- | ------------- | ---------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Baseline (`prefetch_fsdp_weights=False`)                                                           | 1303.2 ms     | —          | [xid/303041931](https://xmanager.corp.google.com/experiments/303041931) | [trace](https://xprof.corp.google.com/trace_viewer/darisoy-9826187937416977410)                                       |
| **`prefetch_fsdp_weights=True` + `moe_flat_fsdp_weights=True` (2D expert weights, double buffer)** | **1038.7 ms** | **−20.3%** | [xid/303149411](https://xmanager.corp.google.com/experiments/303149411) | [trace](https://xprof.corp.google.com/trace_viewer/darisoy-7678417573985827042?view_start=2331.420&view_end=2351.946) |

Combining per-sub-layer nested prefetching with `moe_flat_fsdp_weights=True` and `unroll=2`
double-buffering removes both the per-sub-layer FSDP relayout copies and the loop-carry copies of
the 512 MiB routed-expert weights.
