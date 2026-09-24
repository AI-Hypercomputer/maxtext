# Layer-fused dual-pipe scheduling in normal training

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
PGLE, and profiling settings identical. The example uses full remat in both
runs; it does not preserve a custom remat policy from the input configuration.
Allocation, container mounts, NCCL/XLA settings, and launcher presets belong
in the deployment/launcher configuration, not in MaxText wrapper scripts.

## Supported configuration

- A compatible JAX build providing `jax.fwd_and_bwd`.
- NNX Llama (`decoder_block=llama2`) or DeepSeek V3 with `te_moe_block=true`;
  DeepSeek also requires compatible Transformer Engine and one GPU per process.
- `scan_layers=true`, `shard_mode=auto`, and `dropout_rate=0`.
- `remat_policy=none` or `full`; custom remat is not supported here.
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
this is distinct from the next microbatch's forward branch. Old and new saved
values can coexist, so peak memory can exceed the serial schedule.
TE-EP bootstrap capacity sizing is unchanged by this scheduling feature.

CPU tests cover loss/gradient bookkeeping and scan structure, not full-model
numerical equivalence, TE communication correctness, or measured GPU overlap.
