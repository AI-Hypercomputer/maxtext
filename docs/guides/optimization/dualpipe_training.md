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

# Dual-pipe (1F1B) gradient accumulation

`gradient_accumulation_schedule=dual_pipe` enables experimental layer-level 1F1B
gradient accumulation through the normal `maxtext.trainers.pre_train.train`
entry point. The default, `serial`, keeps the existing training path.

For three microbatches, the decoder schedule is `F0, B0F1, B1F2, B2`.
Each combined layer-scan iteration contains backward for old layer `L-1-k`
and forward for next layer `k`, inside the same jitted training step.
DeepSeek's dense and MoE stacks use separate scan segments where their
parameter/residual structures differ. Embedding/head gradients and the real
masked training loss are included; optimizer updates remain in normal training.

## Run a matched comparison

Use an existing, working MaxText `.yml` training configuration, not a launcher
configuration. It should specify the model, data and packing, attention, precision, mesh,
output directory, and profiling settings. Run from an environment importing
this checkout, using the same distributed launch setup as ordinary training.
This is the per-process training command, not an allocation or rank launcher.

```bash
TRAIN_CONFIG=/path/to/training.yml
SCHEDULE=dual_pipe  # Change to serial for the baseline.
RUN_NAME="ga3-${SCHEDULE}-trial1"  # Unique per run; identical on all ranks.

python -m maxtext.trainers.pre_train.train "${TRAIN_CONFIG}" \
  run_name="${RUN_NAME}" \
  gradient_accumulation_schedule="${SCHEDULE}" \
  gradient_accumulation_steps=3 \
  scan_layers=true \
  remat_policy=full
```

Change only the schedule and run name between the two runs. Keep model size,
precision, batch size, input data, initialization/checkpoint, parallelism,
PGLE, and profiling settings identical. The example uses full remat; any
remat policy works the same way for both schedules.
Allocation, container mounts, NCCL/XLA settings, and launcher presets belong
in the deployment/launcher configuration, not in MaxText wrapper scripts.

## Supported configuration

- A compatible JAX build providing `jax.fwd_and_bwd`.
- NNX Llama (`decoder_block=llama2`), DeepSeek V3 or DeepSeek V4 (`decoder_block=deepseek4`). `te_moe_block` is optional:
  with it, DeepSeek needs compatible Transformer Engine and one GPU per process;
  the non-TE MoE path is not yet validated on GPU for this schedule.
- `scan_layers=true`, `shard_mode=auto`, and `dropout_rate=0`.
- Any `remat_policy`, including `custom` with per-tensor `device`/`offload`/`remat`
  settings. Layers are checkpointed exactly as in the regular decoder, and only
  the tensors the policy keeps are stored between a microbatch's forward and
  backward (offloaded ones stay in pinned host memory).
- Llama is unquantized. DeepSeek quantization recipes are not validated by the schedule; recipes
  that keep mutable scale/amax state (e.g. delayed scaling) are unsupported.
- DeepSeek V4 runs its decoder's own layout: each hash-routed prefix layer is its own
  length-1 group (`layers_0`, `layers_1`, ...), followed by one scanned group of
  two-layer (HCA + CSA) blocks. Like the decoder, the block count is
  `(num_decoder_layers - first_num_hash_layers) // 2`; an odd remainder leaves a
  trailing layer that neither the decoder nor this schedule builds or runs (a warning
  is logged), and a model with fewer than two layers after the prefix has no scanned
  group. The hyper-connection streams (`mhc_expansion_rate`) are expanded at the
  embedding and reduced by the head, and every layer receives the microbatch's input
  token ids for hash routing. V4 supports the CSA indexer loss in sparse training
  (`use_indexer=true`, `indexer_loss_scaling_factor>0`, `indexer_sparse_training=true`);
  the dense warm-up that skips the LM loss is rejected. Each
  scanned block checkpoints its two layers internally; the schedule adds an outer
  checkpoint so the inner scan's copies of the layer weights are not saved per slot.
- `te_moe_block=true` works with DeepSeek V3 and V4. The TE capacity counters
  (`te_moe_capacity_overflow`, `te_moe_total_recv_tokens`,
  `te_moe_recv_capacity_per_rank`) are read from wherever the layer sowed them: a V3
  layer sows them at the top level, but a V4 scanned block holds two layers, so its
  counters sit under `layers_0` and `layers_1`. They are combined per block (any
  overflow, largest receive count, smallest capacity) and then across blocks and
  microbatches the same way, so the logged `learning/te_moe_*` metrics match the
  serial schedule's. A block that sows no counters is an error; a hash-routed prefix
  layer that sows none contributes neutral values. TE sows its own load-balance loss
  and a float `+-rate` bias step per layer (not expert counts), so with
  `te_moe_block` the aux loss is a plain mean over layers (Megatron's seq_aux_loss
  does not apply) and the per-microbatch bias steps are summed, as serial gradient
  accumulation does. This path needs a TE build with MoE support and cannot be run
  on CPU, so it is covered by tests of the collection logic on stand-in layers, not
  by a full-model comparison.
- DeepSeek (V3 and V4) supports `load_balance_loss_weight>0` and
  `routed_bias_update_rate>0`. The load-balance loss (and V4's indexer loss) joins
  the objective with the same weighting as the serial schedule: each microbatch's
  loss, averaged over the layers that have one (summed for Megatron's seq_aux_loss)
  and scaled by its token count, is differentiated inside the fused layer scan, so it
  reaches the router, the layer weights and earlier layers. Router-bias updates are
  collected per layer and summed over microbatches; train_step applies one full-batch
  rate * sign(mean load - load) update after the optimizer step. The bias stays fixed
  during the accumulation window.
- Backward cotangents have the same magnitude as under serial gradient accumulation.
  Differentiating the token-summed loss would make every cotangent about W (the
  step's valid-token count) times its value at GA=1, which saturates a static
  (fixed-range) fp8 calibration of the backward pass. Like the serial path, dual_pipe
  differentiates the loss scaled by 1/S (S = global batch size * `max_target_length`)
  and divides the accumulated gradient by W/S; the gradient is mathematically
  unchanged and the reported loss is not scaled. The scale, the microbatch layout,
  the loss/aux-loss bookkeeping and the routed-bias finalization are shared with the
  serial path in `maxtext.utils.gradient_accumulation`, so the two cannot drift.
- Ordinary causal-LM loss with `num_vocab_tiling=1`. Other unsupported modes
  are rejected by `validate_training_config` in
  `src/maxtext/experimental/dualpipe_nnx.py`.

## Verify the schedule

Inspect the optimized training HLO for `dual_pipe/steady_Bi_Fnext` and
`combined_bf_layers`, with both backward and next-forward operations in the
same layer-loop body. Use a GPU profile to determine whether they overlap:
being in one HLO computation does not guarantee concurrent execution.

Full remat recomputes the old microbatch's forward work during backward;
this is distinct from the next microbatch's forward branch. With full remat all
layers share one sliding buffer of layer inputs: microbatch `i` stores layer `j`
at slot `j` (even `i`) or `L-1-j` (odd `i`), so each fused step's forward write
reuses the slot its backward just read, and layer gradients are added into the
running sums in place. With other remat policies the saved tensors
differ per layer group, so each group gets its own slot pool; groups at the
start of the network (read last by the old microbatch, written first by the new
one) need up to twice their layer count in slots. Peak memory still exceeds the
serial schedule by roughly one extra layer's working set (both branches'
gathered weights are resident in a fused step), plus allocator fragmentation.
TE-EP bootstrap capacity sizing is unchanged by this scheduling feature.

CPU tests cover loss/gradient bookkeeping and scan structure, not full-model
numerical equivalence, TE communication correctness, or measured GPU overlap.
