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
- NNX Llama (`decoder_block=llama2`) or DeepSeek V3 with `te_moe_block=true`;
  DeepSeek also requires compatible Transformer Engine and one GPU per process.
- `scan_layers=true`, `shard_mode=auto`, and `dropout_rate=0`.
- Any `remat_policy`, including `custom` with per-tensor `device`/`offload`/`remat`
  settings. Layers are checkpointed exactly as in the regular decoder, and only
  the tensors the policy keeps are stored between a microbatch's forward and
  backward (offloaded ones stay in pinned host memory).
- Llama is unquantized. DeepSeek supports `quantization=te_no_quant` or
  `te_fp8_currentscaling`, and `te_gmm_quantization=te_no_quant` or `te_mxfp8`.
- DeepSeek requires `load_balance_loss_weight=0` and
  `routed_bias_update_rate=0`; an enabled router bias remains frozen.
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
