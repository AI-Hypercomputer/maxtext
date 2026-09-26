# MaxTextTrainingEngine

`MaxTextTrainingEngine` (`maxtext_engine.py`) is a trainer that its caller drives one micro-batch at
a time. Tunix's RL trainer uses it to train MaxText models. This page shows how to run it, how an
optimizer step flows through the engine, which config keys it supports, how to read its
ahead-of-time memory report, and how it is tested.

## How to run

### Standalone training script

`experimental/maxtext_engine/train.py` trains with the engine the way `pre_train/train.py` trains
without it. It takes the same config file and `key=value` overrides, and runs `steps` optimizer steps
of `gradient_accumulation_steps` micro-batches each. On a multi-host TPU slice, run the same command
on every host.

```bash
python3 -m maxtext.experimental.maxtext_engine.train src/maxtext/configs/base.yml \
  run_name=<run_name> base_output_directory=<output_dir> \
  model_name=<model_name> dataset_type=synthetic \
  per_device_batch_size=1 gradient_accumulation_steps=4 steps=10
```

Each step logs `completed step: <n>, seconds: ..., TFLOP/s/device: ..., Tokens/s/device: ...,
Tokens/s/chip: ...`; the step time covers all of the step's device work.

- `base_output_directory` must be writable from every host: a failed metrics write ends the run.
- Train with float32 gradients by adding `grad_dtype=float32 optimizer_memory_host_offload=true`; the
  offload keeps the optimizer state out of device memory.
- `grad_accumulation_dtype=float32` sums the micro-batch gradients in float32.
- `enable_checkpointing=true` saves and restores the train state (the data iterator is not
  checkpointed).
- `profiler=xplane skip_first_n_steps_for_profiler=<n> profiler_steps=<k>` profiles whole optimizer
  steps.
- For large models on TPU, see [Running fast on TPU](#running-fast-on-tpu).

### From Python

This is what the script and Tunix do:

```python
from maxtext.configs import pyconfig
from maxtext.training_engine import maxtext_engine
from maxtext.utils import maxtext_utils

# The same config file and key=value overrides as the script; the first element is ignored.
config = pyconfig.initialize(["", "src/maxtext/configs/base.yml", "model_name=<model_name>"])
mesh = maxtext_utils.get_mesh_from_config(config)
engine = maxtext_engine.MaxTextTrainingEngine(config, mesh=mesh)

engine.compile(first_micro_batch)  # Optional: without it, the first fwd_bwd compiles.
for step in range(num_steps):
  for micro_batch in micro_batches:  # Batches in the format DataLoader.load_next_batch returns.
    engine.fwd_bwd(micro_batch)
  engine.update()
engine.close()
```

### Ahead-of-time memory report

`maxtext_engine_compile.py` compiles the engine's kernels ahead of time for a TPU topology and prints
how much device memory each needs (see [Reading the memory report](#reading-the-memory-report)). It runs on a CPU
host (x86-64 Linux with `libtpu` installed); no TPU is needed:

```bash
python3 -m maxtext.training_engine.maxtext_engine_compile src/maxtext/configs/base.yml \
  compile_topology=<topology, e.g. tpu7x-256> compile_topology_num_slices=1 \
  <the same model and parallelism flags as the run>
```

To check the configuration of an RL trainer, add `compile_engine_loss=grpo`, which compiles the kernels
with Tunix's GRPO loss instead of MaxText's own:

```bash
python3 -m maxtext.training_engine.maxtext_engine_compile src/maxtext/configs/base.yml \
  compile_topology=<topology> compile_topology_num_slices=1 compile_engine_loss=grpo \
  max_target_length=<prompt + completion tokens> max_prefill_predict_length=<prompt tokens> \
  per_device_batch_size=<sequences per micro-batch / chips> \
  compile_engine_grpo_config="{<the trainer's GRPOConfig options, e.g. beta: 0.0>}" \
  compile_engine_logps_chunk_size=<the trainer's compute_logps_chunk_size> \
  <the same model and parallelism flags as the trainer>
```

For a trainer whose micro-batches Tunix packs, add its `max_seq_token_per_tpu` and
`max_segments_per_packed_row` as `compile_engine_max_seq_token_per_tpu` and
`compile_engine_max_segments_per_packed_row`; for one whose rollouts return their MoE routing (Tunix's
`return_routed_experts`), add `compile_engine_router_replay=true`.

The engine is then set up as Tunix's RL trainer sets it up: the model is wrapped in `TunixMaxTextAdapter`,
and the loss (with `has_aux=True`) and its input mapping come from Tunix's `GRPOAdapter`, built from
`GRPOConfig(**compile_engine_grpo_config)` with `decode_sampling_temperature` as the default temperature.
Pass the trainer's own options: they decide which inputs the loss reads and which operations it runs,
and options left out take `GRPOConfig`'s defaults, whose `beta` is non-zero. A non-zero
`compile_engine_logps_chunk_size` computes the log-probabilities in chunks of that many tokens, as Tunix's
`TrainerWorker` does for `compute_logps_chunk_size`; without it the logits of whole sequences are held at
once, which at long context and large vocabularies dominates the report.

The kernels are compiled for the `RLTrainerPayload` Tunix's orchestrator builds, which the tool builds with
Tunix's own code (`GRPOAdapter.create_trainer_payloads` and the batch assembler `create_batch_assembler`
picks) from rollouts of `max_prefill_predict_length` prompt tokens (Tunix's `max_prompt_length`) and the rest
of `max_target_length` as completion tokens (`max_response_length`):

- By default, `PaddedBatchAssembler`'s: `micro_batch_size_to_train_on` sequences (Tunix's
  `train_micro_batch_size`), each padded into a prompt part and a completion part.
- With `compile_engine_max_seq_token_per_tpu`, `SequencePackedBatchAssembler`'s: rows of that many tokens
  (at least `max_target_length`, or Tunix rejects them), each holding as many whole sequences as fit, at most
  `compile_engine_max_segments_per_packed_row`, told apart by `segment_ids`, `segment_positions` and
  `num_segments`. There is a row per device of the mesh's data, fsdp, fsdp_transpose and expert axes, the
  trainer mesh dimensions Tunix's `BatchConfig` sizes a packed micro-batch by; `per_device_batch_size` does
  not set it. Tunix's loss then aggregates per sequence rather than per row.
- Token ids and masks, per-token advantages, the rollout's log-probabilities and whether it was cut off
  (`overlong`); the old policy's log-probabilities when `use_rollout_logps` is true, and the reference
  model's when `beta` is non-zero. Tunix's sampler-trainer agreement step, which adds `sampler_is_weights`
  when `sampler_is` is `token`, is not modeled.
- With `compile_engine_router_replay`, the experts each token was routed to in every MoE layer:
  `routed_experts`, `[sequences, prompt + completion tokens, num_decoder_layers, num_experts_per_tok]` int16,
  which the loss passes to the model to replay, wherever Tunix's batch assembler carries it. When the
  assembler leaves it out, the compiled trainer routes every token itself, and the tool prints a line saying
  so.
- Pad and end-of-sequence id 0, so no tokenizer is loaded. The ids are constants in the program and change
  no shape.

The micro-batch is built in host memory, as the orchestrator builds one, and only its shapes are compiled
for. It is one of `fwd_bwd`'s arguments, so the report counts it in `arg`; fields the loss does not read
are dropped from the program and not counted. This path needs Tunix's RL modules and a model with a
Hugging Face config, which `TunixMaxTextAdapter` reads.

### Running the tests

```bash
JAX_PLATFORMS=cpu python3 -m pytest tests/post_training/unit/maxtext_engine_*test.py \
  tests/post_training/unit/router_replay_engine_test.py
```

The tests run on CPU and need Tunix importable (installed or on `PYTHONPATH`). Some start a child
process with 4 or 8 CPU devices.

## Running fast on TPU

This section collects settings for large models on TPU, and how to profile one. The settings are not
defaults: each can raise device memory by an amount that depends on the configuration. Before using
one, compile your configuration with the [ahead-of-time memory report](#ahead-of-time-memory-report)
with and without it, and measure the step time both ways.

### SparseCore copy offload

On TPUs whose collectives run on the SparseCore (the `--xla_tpu_enable_sparse_core_collective_offload_*`
libtpu flags), try this libtpu flag:

```
--xla_tpu_enable_offloading_copy_to_sparsecore=false
```

XLA can move layout-conversion copies from the TensorCore to the SparseCore, where they share it with
the offloaded collectives, and the backward pass can then wait longer on its all-gathers and
reduce-scatters. With the flag the copies stay on the TensorCore. It changes where copies run and how
buffers are placed, not the arithmetic. It can raise device memory, in `fwd_bwd` and in
`pre_train/train.py`'s step alike. Set it in `LIBTPU_INIT_ARGS` before JAX starts; it then applies to
every program the process compiles. For the memory report, add it to `compile_xla_flags`.

### Casting all-reduced gradients after the all-reduce

When the weights are wider than the accumulation dtype (for example float32 weights with
`grad_dtype=bfloat16`), enable `cast_grads_after_all_reduce=true` together with the flag above. The
gradients that `fwd_bwd` sums across devices by an all-reduce alone then leave it in the weights'
dtype and are cast as they join the accumulator (see [One optimizer step](#one-optimizer-step)). Cast
inside `fwd_bwd`, those all-reduces can become synchronous reductions in the backward pass, which delay
the collectives the rest of the backward pass overlaps with compute. With the key on, they run in the
weights' dtype and are rounded to the accumulation dtype once, after the sum.

It can raise device memory too: the uncast gradients are wider, and the schedule XLA then picks can
hold more at `fwd_bwd`'s peak. Check it together with the flag, in the memory report.

### Float32 gradients

`grad_dtype=float32` keeps the gradients, the accumulator and the optimizer's inputs in float32. On a
tight memory budget, combine it with `optimizer_memory_host_offload=true`, which moves the optimizer
state out of device memory between updates. If that is not enough and a `remat_policy=custom` keeps
the attention output on device (`context=device`), rematerialize it with `context=remat`. With float32
weights no gradient is cast, so `cast_grads_after_all_reduce` changes nothing.

### Profiling a large step

A trace that outgrows the profiler's size limit is truncated: its device timeline stops early and it
contains a `Trace Buffers Dropped` event. To profile a large model, profile a step with few
micro-batches and trace one chip without its SparseCores:

```bash
python3 -m maxtext.experimental.maxtext_engine.train src/maxtext/configs/base.yml <run flags> \
  gradient_accumulation_steps=2 profiler=xplane skip_first_n_steps_for_profiler=<n> profiler_steps=1 \
  enable_tpu_profiling_options=true tpu_num_chips_to_profile_per_task=1 \
  tpu_num_sparse_cores_to_trace=0 tpu_num_sparse_core_tiles_to_trace=0
```

With `scan_layers=true`, each `fwd_bwd` in the trace contains a forward and a backward layer scan
(`while` ops on the device's `XLA Ops` line). Time the TensorCore spends waiting on a collective shows
there as the collective's `-done` op (`all-gather-done`, `reduce-scatter-done`); the asynchronous
collectives themselves are on the `Async XLA Ops` line.

## Recipe: Qwen3.5-397B-A17B on 128 TPU v7x chips

The configuration below trains Qwen3.5-397B-A17B with the standalone script on 128 TPU v7x chips (256
devices): mesh fsdp 32 x context 4 x expert 2 with `custom_mesh_and_rule=cp-as-ep`, sequences of 65,536
tokens, and 1,024 sequences per optimizer step (16 micro-batches of 64), with float32 weights, bfloat16
compute and AdamW. Run it on every host of the slice.

<details>
<summary>Full flags</summary>

```bash
export LIBTPU_INIT_ARGS="--xla_tpu_use_tc_device_shape_on_sc=true --xla_sc_disable_megacore_partitioning=true \
--xla_tpu_enable_offloading_gather_to_sparsecore=true \
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true \
--xla_tpu_aggressive_opt_barrier_removal=true --xla_tpu_scoped_vmem_limit_kib=65536 \
--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false \
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true \
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true \
--xla_tpu_use_single_sparse_core_for_all_gather_offload=false --xla_tpu_enable_concurrent_sparse_core_offloading=true \
--xla_tpu_dvfs_p_state=7 --xla_tpu_enable_offloading_copy_to_sparsecore=false"

python3 -m maxtext.experimental.maxtext_engine.train src/maxtext/configs/base.yml \
  run_name=<run_name> base_output_directory=<output_dir> steps=<steps> \
  model_name=qwen3.5-397b-a17b base_num_decoder_layers=60 override_model_config=true scan_layers=True \
  use_multimodal=false dataset_type=synthetic max_target_length=65536 tokenizer_path='' packing=false \
  opt_type=adamw adam_weight_decay=0.0 adam_b1=0.9 adam_b2=0.999 adam_eps=1e-8 learning_rate=1e-5 \
  dtype=bfloat16 mu_dtype=bfloat16 grad_dtype=bfloat16 cast_grads_after_all_reduce=true \
  megablox=true sparse_matmul=true use_tokamax_gmm=true use_gmm_v2=true use_gmm_v2_heuristic_tiling=true \
  merge_gating_gmm=false use_ring_of_experts=true use_ragged_sort=true use_custom_sort_vjp=false \
  ragged_buffer_factor=2.0 use_random_routing=True num_moe_token_chunks=2 moe_chunk_barrier=false \
  attention=flash use_tokamax_splash=true use_splash_scheduler=true sa_block_q=1024 sa_block_kv=4096 \
  sa_block_kv_compute=512 sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=512 \
  sa_fuse_reciprocal=false sa_use_base2_exp=true dq_reduction_steps=3 \
  gdn_chunk_size=64 use_gdn_kernel=true gdn_cp_mode=auto \
  custom_mesh_and_rule=cp-as-ep ici_tensor_parallelism=1 ici_fsdp_parallelism=32 ici_context_parallelism=4 \
  ici_expert_parallelism=2 context_parallel_strategy=ring context_parallel_load_balance=False \
  allow_split_physical_axes=False \
  remat_policy=custom decoder_layer_input=device context=device gdn=remat gdn_conv=remat \
  num_vocab_tiling=2 use_iota_embed=false \
  per_device_batch_size=0.25 gradient_accumulation_steps=16 enable_checkpointing=false
```

</details>

Variants:

| goal | change to the flags above |
|---|---|
| bfloat16 gradients, fastest (as above) | none |
| float32 gradients | `grad_dtype=float32 optimizer_memory_host_offload=true context=remat` (`cast_grads_after_all_reduce` has no effect with float32 gradients) |
| without the SparseCore settings | drop `--xla_tpu_enable_offloading_copy_to_sparsecore=false` and `cast_grads_after_all_reduce=true` |
| check memory first | run `python3 -m maxtext.training_engine.maxtext_engine_compile` with the same flags, plus `compile_topology=tpu7x-256 compile_topology_num_slices=1` and the libtpu flags above in `compile_xla_flags` |
| memory of the same trainer under Tunix's GRPO loss | add `compile_engine_loss=grpo max_prefill_predict_length=<prompt tokens>` and the trainer's `compile_engine_grpo_config` and `compile_engine_logps_chunk_size` to the compile command, and, if it packs sequences or replays routing, `compile_engine_max_seq_token_per_tpu`, `compile_engine_max_segments_per_packed_row` and `compile_engine_router_replay` (see [Ahead-of-time memory report](#ahead-of-time-memory-report)) |

## Results on TPU v7x

Measured with the recipe above (the standalone script for the engine), steady state over the first
steps after step 0. The `pre_train/train.py` rows ran the same model, flags and code plus two kernel
changes that are not part of MaxText yet (a single-scan vocabulary-tiled loss and a cost estimate for
the gated-delta-net forward kernel), which make it slightly faster; the engine rows ran without them.

| trainer | settings | s/step | tokens/s/chip | vs `pre_train/train.py` |
|---|---|---|---|---|
| `pre_train/train.py` | bfloat16 gradients | 206.12 | 2,543.6 | — |
| `pre_train/train.py` | + `--xla_tpu_enable_offloading_copy_to_sparsecore=false` | does not fit | — | runs out of device memory at load |
| engine | bfloat16 gradients, without the SparseCore settings | 228.86 | 2,291 | 11.0% slower |
| engine | + `--xla_tpu_enable_offloading_copy_to_sparsecore=false` | 214.70 | 2,442 | 4.2% slower |
| engine | + `cast_grads_after_all_reduce=true` (the recipe) | 201.43 | 2,603 | **2.3% faster** |
| engine | float32 gradients, optimizer offload, `context=remat`, without the SparseCore settings | 229.72 | 2,282 | 11.4% slower |
| engine | float32 gradients, optimizer offload, `context=remat`, + `--xla_tpu_enable_offloading_copy_to_sparsecore=false` (the float32 variant) | 217.19 | 2,414 | 5.4% slower |

`pre_train/train.py`'s step time is the interval between its metric lines; its `seconds` field measures
dispatch only on this configuration. The engine's is the script's own measurement, from dispatch until
the step's device work drains.

## Callers

| caller | class | notes |
|---|---|---|
| Tunix RL trainer | `MaxTextTrainingEngine` | constructs the engine under `with mesh:` only |
| `experimental/maxtext_engine/train.py` | `MaxTextTrainingEngine` | standalone script; constructs the engine the way Tunix does, with no outer `logical_axis_rules` |
| `maxtext_engine_compile.py` | `AbstractMaxTextEngine` | ahead-of-time compilation from shapes; allocates no arrays |

`pre_train/train.py` is a separate trainer (one jitted step with `lax.scan` gradient accumulation)
and does not use the engine.

## One optimizer step

| call | kernel | takes | returns |
|---|---|---|---|
| `fwd_bwd(micro_batch)` | `fwd_bwd` | params, rest, batch | loss, aux, rest, grads, denominator |
| `fwd_bwd(micro_batch)`, after the first of a step | `accumulate`, after `fwd_bwd` | accumulator and its denominator (both donated), grads, denominator | the sums |
| `update()` | `update` | train state (donated), accumulator, denominator | new state, grad norm, skipped flag |
| `model_scope(*inputs)` | none; the caller jits | inputs | yields the model and the placed inputs under the sharding rules |
| `fwd_only(fn, *inputs)` | none; `fn` jits | inputs | `fn(model, ...)` |
| `eval_step(batch)` | eval | params, rest, batch | loss, aux |

Every micro-batch runs the same `fwd_bwd` program. The first micro-batch's gradients start the
accumulator, and `accumulate` adds each later one's to it in place. The accumulator is not passed to
`fwd_bwd`: given it, XLA can fold the add into the backward pass, and on large meshes folding it into
the token embedding's gradient scatter-add gathers an unsharded copy of that gradient, which raises the
program's peak memory and can cost compute/communication overlap. Keeping the add apart has a memory
cost of its own: each later micro-batch's gradients are held beside the accumulator until `accumulate`
consumes them, one more gradient tree in the accumulation dtype at `fwd_bwd`'s peak. So the separate
add lowers the peak where the fold was expensive and raises it by that tree where the fold was not; the
[memory report](#reading-the-memory-report) counts it.

`fwd_bwd` returns its gradients cast to the accumulation dtype. With `cast_grads_after_all_reduce=true`,
the gradients it sums across devices by an all-reduce alone, those of parameters sharded over none of
the mesh axes a batch is split over (norm scales, for example), leave it in the parameters' dtype
instead and are cast as they join the accumulator. Cast inside `fwd_bwd`, XLA can move the cast ahead of
the all-reduce, which then runs in the accumulation dtype; cast later, it cannot. Every other gradient
is still cast inside `fwd_bwd`, which keeps most of its output in the accumulation dtype.

`model_scope` implements Tunix's `AbstractTrainer.model_scope`, which Tunix uses to score per-token
log-probabilities with the trainer's weights.

## Sharding rules

Every entry point that traces the model binds the mesh and MaxText's logical axis rules itself
(`_sharding_ctx()`), because callers such as Tunix bind neither. Model construction is the exception:
it binds the rules but clears the mesh. The rules must be live because some layers check their
sharding at construction (for example Tokamax ring attention), while under `jax.set_mesh` flax's
`nnx.eval_shape` re-derives shardings from logical names and rejects some valid MaxText rule sets.

## Config

| key | engine behavior |
|---|---|
| `grad_dtype` | dtype the optimizer receives gradients in |
| `grad_accumulation_dtype` | dtype micro-batch gradients are summed in; `""` (default) uses `grad_dtype`. The sum is divided by the total loss denominator and cast to `grad_dtype` once, in `update`. `"float32"` with `grad_dtype=bfloat16` is more precise but keeps a float32 accumulator on device between micro-batches, and each later micro-batch's float32 gradients beside it while `fwd_bwd` runs |
| `cast_grads_after_all_reduce` (default false) | gradients that `fwd_bwd` sums across devices by an all-reduce alone leave it in the parameters' dtype and are cast to the accumulation dtype as they join the accumulator (see [One optimizer step](#one-optimizer-step)). Meant for TPU together with the libtpu flag `--xla_tpu_enable_offloading_copy_to_sparsecore=false`; see [Running fast on TPU](#running-fast-on-tpu). No effect when the parameters are already in the accumulation dtype, on the eager path (no `compile()`), or under the deferred data-parallel all-reduce, where `fwd_bwd` reduces no gradient |
| `optimizer_memory_host_offload` | the optimizer state lives in pinned host memory; `update` moves it to the device and back, so it is not resident during the forward and backward passes |
| `parameter_memory_host_offload` | not supported; raises |
| `shard_optimizer_over_data` (Zero-1) | shards the optimizer state over the `data` axis inside `update`; requires `shard_mode=explicit` |
| `skip_step_on_nan` (default true), `skip_step_on_spikes`, `max_grad_norm_spike` | a non-finite or spiking step leaves the state unchanged and records `step_skipped` |
| `enable_checkpointing` | `false` disables saving and restoring the train state; `load_parameters_path` still loads weights |
| `gradient_accumulation_steps` | read only by the standalone script; Tunix chooses the number of micro-batches itself |
| `compile_engine_loss` (default `maxtext`), `compile_engine_grpo_config`, `compile_engine_logps_chunk_size` | read only by `maxtext_engine_compile.py`: the loss it compiles the kernels with, MaxText's own (`maxtext`) or Tunix's GRPO loss (`grpo`), and the latter's options; see [Ahead-of-time memory report](#ahead-of-time-memory-report) |
| `compile_engine_max_seq_token_per_tpu` (default 0), `compile_engine_max_segments_per_packed_row` (default 0), `compile_engine_router_replay` (default false) | read only by `maxtext_engine_compile.py`, with `compile_engine_loss=grpo`: Tunix's `max_seq_token_per_tpu` and `max_segments_per_packed_row`, which make the micro-batch sequence-packed, and whether the rollouts return their MoE routing for the trainer to replay; see [Ahead-of-time memory report](#ahead-of-time-memory-report) |

## Reading the memory report

The tool compiles the three training kernels for the topology on CPU and prints each kernel's raw
`memory_analysis()`, followed by a per-device table in GiB:

- `arg`, `out`, `alias` and `temp`, and their `host` counterparts, come from `memory_analysis()`.
- `resident = arg + out - alias + temp`, since a donated buffer is counted in both `arg` and `out`.
- `+state` is what stays in device memory while the kernel runs but is not one of its arguments. For
  `fwd_bwd` that is the optimizer state (`host state` if it is offloaded) and the gradient
  accumulator of the earlier micro-batches, sized by `accumulate`'s outputs; for `accumulate` it is
  the model and optimizer state. The raw analyses of these two kernels therefore do not change when
  the optimizer is offloaded; only `update`'s does.
- `total = resident + +state`.

The last line, `TRAIN-KERNEL DEVICE PEAK`, is the largest `total` over the three kernels. It assumes
gradient accumulation, so `fwd_bwd` is charged with the accumulator even when a step has one
micro-batch. It covers only those kernels: `fwd_only` / `model_scope` scoring, the eval kernel,
generated code and anything the caller holds are excluded, and so is weight-sync staging, which is
estimated on its own `WEIGHT-SYNC STAGING` line when `use_weight_converter=false`.

On TPU, `jax.Device.memory_stats()["peak_bytes_in_use"]` excludes XLA program temporaries, so it is
not a substitute for this report.

The report compiles on a CPU host, where some kernels take their CPU fallback paths (for example the
gated delta net kernels and the SparseCore ragged gathers). The compiled programs can therefore differ
from the ones that run on TPU, and so can their memory. Treat the peak as an estimate, and compare
configurations with each other.

## Tests

The engine's tests are in `tests/post_training/unit/` and run on CPU; they require Tunix.

| test | covers |
|---|---|
| `maxtext_engine_test.py` | optimizer offload placement, numerics and restore; accumulation dtypes; sharding rules during model build and eval; `model_scope`; Zero-1 |
| `maxtext_engine_train_test.py` | the standalone script end to end; the Tunix path through Tunix's own builders and `TrainerWorker.per_token_logps`; parity with `pre_train/train.py` |
| `maxtext_engine_compile_test.py` | the memory report, against real XLA compiles, including optimizer offload; `compile_engine_loss`, including Tunix's GRPO loss on the payload Tunix builds, padded or sequence-packed, with or without the rollouts' routing |
| `maxtext_engine_compile_parity_test.py` | `AbstractMaxTextEngine` compiles Qwen3.5 with the Tunix adapter to the same programs as the live engine |
| `maxtext_engine_model_build_test.py` | model construction with and without a mesh set by the caller |
| `maxtext_engine_checkpoint_test.py` | resuming from a mid-step checkpoint through Orbax reproduces the uninterrupted step |
| `maxtext_engine_grad_cast_test.py` | `cast_grads_after_all_reduce`: which gradients leave `fwd_bwd` uncast, and that moving their cast changes no number on CPU |
| `maxtext_engine_{constructor,data_parallel,xaot,packing,profiling}_test.py`, `router_replay_engine_test.py` | construction, data parallelism, ahead-of-time compilation, sequence packing, profiling, router replay |

## Differences from pre_train/train.py

- Under gradient accumulation, `pre_train/train.py` adds auxiliary losses (MoE load balancing,
  `mtp_loss`, `indexer_loss`) to the unnormalized cross-entropy sum before dividing by the token
  count, which scales them down relative to its path without accumulation. The engine weights them
  as that path does. Configs with these losses off, such as the default `load_balance_loss_weight`
  of 0, are unaffected.
- `pre_train/train.py` sums micro-batch gradients in the dtype of the parameters it differentiates;
  the engine sums them in `grad_accumulation_dtype`, which defaults to `grad_dtype`.
