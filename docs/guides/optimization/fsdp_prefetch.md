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

With FSDP, every decoder layer's weights are stored sharded over the `fsdp` mesh axes and must be
all-gathered before the layer runs (and their gradients reduce-scattered after its backward pass).
By default XLA schedules these collectives, and depending on the model and the remat policy they can
end up partially exposed on the critical path.

`prefetch_fsdp_weights=True` replaces the scanned layer loop with a hand-written pipeline
(`src/maxtext/layers/fsdp_prefetch.py`) that overlaps the collectives with compute explicitly:

| Pass     | Layer k                                                                   |
| -------- | ------------------------------------------------------------------------- |
| Forward  | all-gather W(k+1) ‖ compute layer k                                       |
| Backward | all-gather W(k-1) ‖ recompute + vjp of layer k, then reduce-scatter dW(k) |

Gathered weights are never saved between the forward and backward pass. Both loops walk over pairs
of layers with two gathered-weight buffers used in turn, so the forward holds at most two layers of
gathered weights (current + prefetched) and the backward at most three (W(k), W(k-1) and dW(k)
until its reduce-scatter).

### Remat policies

`remat_policy` (including `custom` policies) applies to each decoder layer. The forward pass carries
the activations that the policy saves (on device or offloaded to host) to the backward pass, which
rebuilds each layer's vjp from them and the re-gathered weights instead of recomputing the whole
layer. With `remat_policy=full` only each layer's input is saved. Gathered weights are never saved,
whatever the policy: if a policy would save a residual shaped like a gathered weight, a warning is
logged. The number and size of the residuals saved per layer are logged at trace time.

## Usage

```bash
python3 -m maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \
  ... \
  scan_layers=True \
  prefetch_fsdp_weights=True
```

### Recommended XLA flags

The prefetch all-gathers are explicit `all_gather`s tagged with the `xla_explicit_fsdp` XLA
metadata. On TPUs with SparseCores, enable XLA's FSDP scheduler so that the tagged gathers run on a
dedicated SparseCore instead of queueing behind the layer's own SparseCore work (for example MoE
token gathers):

```bash
export LIBTPU_INIT_ARGS="--xla_tpu_enable_fsdp_latency_hiding_scheduler=true \
  --xla_tpu_explicit_fsdp_dedicated_sparse_core_id=0"
```

```{note}
These flags require a libtpu build that includes the FSDP latency hiding scheduler. Without them
the pipeline still runs correctly, but the prefetch collectives are scheduled by the default
latency hiding scheduler.
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

DeepSeek-V3 architecture (`deepseek3-671b`) reduced to 15 decoder layers (3 dense + 12 MoE,
`base_num_decoder_layers=15`) and 128 routed experts (`num_experts=128`) so that it fits on 64 TPU
chips (4x4x4). Full remat (`remat_policy=custom` saving only the decoder layer input),
`shard_mode=explicit`, `per_device_batch_size=4`, `max_target_length=4096`, with the XLA flags
above:

| Run                     | Step time | Fwd / layer | Bwd / layer |
| ----------------------- | --------- | ----------- | ----------- |
| Baseline                | 5222 ms   | 92 ms       | 277 ms      |
| `prefetch_fsdp_weights` | 4832 ms   | 79 ms       | 240 ms      |

With remat policies that save more activations (same model, `per_device_batch_size=2`,
`num_experts_per_tok=4`, splash attention blocks 1024 forward / 2048 backward), prefetching keeps
its gain on top of the saved recompute:

| `remat_policy=custom`, saved on device                            | Baseline | `prefetch_fsdp_weights` |
| ----------------------------------------------------------------- | -------- | ----------------------- |
| `decoder_layer_input` only                                        | 2955 ms  | 2650 ms                 |
| + attention (`mla_q`, `mla_kv`, `out_proj`, `attention_out`)      | 2861 ms  | 2507 ms                 |
| + MoE (`moe_mlpwi_0`, `moe_mlpwi_1`, `moe_mlpwo`, `moe_x_sorted`) | 2703 ms  | 2418 ms                 |
