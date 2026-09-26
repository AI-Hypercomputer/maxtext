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

### Running the tests

```bash
JAX_PLATFORMS=cpu python3 -m pytest tests/post_training/unit/maxtext_engine_*test.py \
  tests/post_training/unit/router_replay_engine_test.py
```

The tests run on CPU and need Tunix importable (installed or on `PYTHONPATH`). Some start a child
process with 4 or 8 CPU devices.

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
| `cast_grads_after_all_reduce` (default false) | gradients that `fwd_bwd` sums across devices by an all-reduce alone leave it in the parameters' dtype and are cast to the accumulation dtype as they join the accumulator (see [One optimizer step](#one-optimizer-step)). Meant for TPU together with the libtpu flag `--xla_tpu_enable_offloading_copy_to_sparsecore=false`. No effect when the parameters are already in the accumulation dtype, on the eager path (no `compile()`), or under the deferred data-parallel all-reduce, where `fwd_bwd` reduces no gradient |
| `optimizer_memory_host_offload` | the optimizer state lives in pinned host memory; `update` moves it to the device and back, so it is not resident during the forward and backward passes |
| `parameter_memory_host_offload` | not supported; raises |
| `shard_optimizer_over_data` (Zero-1) | shards the optimizer state over the `data` axis inside `update`; requires `shard_mode=explicit` |
| `skip_step_on_nan` (default true), `skip_step_on_spikes`, `max_grad_norm_spike` | a non-finite or spiking step leaves the state unchanged and records `step_skipped` |
| `enable_checkpointing` | `false` disables saving and restoring the train state; `load_parameters_path` still loads weights |
| `gradient_accumulation_steps` | read only by the standalone script; Tunix chooses the number of micro-batches itself |

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

## Tests

The engine's tests are in `tests/post_training/unit/` and run on CPU; they require Tunix.

| test | covers |
|---|---|
| `maxtext_engine_test.py` | optimizer offload placement, numerics and restore; accumulation dtypes; sharding rules during model build and eval; `model_scope`; Zero-1 |
| `maxtext_engine_train_test.py` | the standalone script end to end; the Tunix path through Tunix's own builders and `TrainerWorker.per_token_logps`; parity with `pre_train/train.py` |
| `maxtext_engine_compile_test.py` | the memory report, against real XLA compiles, including optimizer offload |
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
