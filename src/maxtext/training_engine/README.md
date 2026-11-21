# MaxTextTrainingEngine

`MaxTextTrainingEngine` (`maxtext_engine.py`) is a trainer that its caller drives one micro-batch at
a time. Tunix's RL trainer uses it to train MaxText models. This page shows how to run it, how an
optimizer step flows through the engine, which config keys it supports, how to read its
ahead-of-time memory report, and how it is tested.

## How to run

### Standalone training script

`experimental/maxtext_engine/train.py` runs standalone training on `MaxTextTrainingEngine`. It takes
the same config file and `key=value` overrides, and runs `steps` optimizer steps of
`gradient_accumulation_steps` micro-batches each. On a multi-host TPU slice, run the same command
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

engine.compile(first_micro_batch)  # Or `engine.compile()` to defer to the first fwd_bwd; omit to stay eager.
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
  tests/post_training/integration/maxtext_engine_parity_test.py \
  tests/post_training/unit/prepare_weight_sync_test.py \
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
buffers are placed, not the arithmetic. It can raise device memory in `fwd_bwd`. Set it in
`LIBTPU_INIT_ARGS` before JAX starts; it then applies to
every program the process compiles. For the memory report, add it to `compile_xla_flags`.

### Casting all-reduced gradients after the all-reduce

When the weights are wider than `grad_dtype` (for example float32 weights with `grad_dtype=bfloat16`),
enable `cast_grads_after_all_reduce=true` together with the flag above. The gradients that `fwd_bwd`
sums across devices by an all-reduce alone then leave it in the weights' dtype and are cast as they
join the accumulator (see [One optimizer step](#one-optimizer-step)). Cast inside `fwd_bwd`, those
all-reduces can become synchronous reductions in the backward pass, which delay the collectives the rest
of the backward pass overlaps with compute. With the key on, they run in the weights' dtype and are
rounded to `grad_dtype` once, after the sum.

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

## Callers

| caller | class | notes |
|---|---|---|
| Tunix RL trainer | `MaxTextTrainingEngine` | `create_maxtext_engine` binds `logical_axis_rules`; `TrainerWorker` runs steps under `with mesh:` only |
| `experimental/maxtext_engine/train.py` | `MaxTextTrainingEngine` | standalone script; constructs the engine with no outer `logical_axis_rules` |
| `maxtext_engine_compile.py` | `AbstractMaxTextEngine` | ahead-of-time compilation from shapes; allocates no arrays |

## One optimizer step

| call | kernel | takes | returns |
|---|---|---|---|
| `fwd_bwd(micro_batch)` | `fwd_bwd` | params, rest, batch | loss, aux, rest, grads, denominator |
| `fwd_bwd(micro_batch)`, after the first of a step | `accumulate`, after `fwd_bwd` | accumulator and its denominator (both donated), grads, denominator | the sums |
| `update()` | `update` | train state (donated), accumulator, denominator | new state, grad norm, skipped flag |
| `fwd_only(payload)` | `fwd_only` (via `with_fwd_only_fn`) | params, rest, payload | `fn(model, payload)` |
| `eval_step(batch)` | eval | params, rest, batch | loss, aux |

Every micro-batch runs the same `fwd_bwd` program. The first micro-batch's gradients start the
accumulator, and `accumulate` adds each later one's to it in place. The accumulator is not passed to
`fwd_bwd`: given it, XLA can fold the add into the backward pass, and on large meshes folding it into
the token embedding's gradient scatter-add gathers an unsharded copy of that gradient, which raises the
program's peak memory and can cost compute/communication overlap. Keeping the add apart has a memory
cost of its own: each later micro-batch's gradients are held beside the accumulator until `accumulate`
consumes them, one more gradient tree in `grad_dtype` at `fwd_bwd`'s peak. So the separate add lowers
the peak where the fold was expensive and raises it by that tree where the fold was not; the
[memory report](#reading-the-memory-report) counts it.

`fwd_bwd` returns its gradients cast to `grad_dtype`. With `cast_grads_after_all_reduce=true`, the
gradients it sums across devices by an all-reduce alone, those of parameters sharded over none of the
mesh axes a batch is split over (norm scales, for example), leave it in the parameters' dtype instead
and are cast as they join the accumulator. Cast inside `fwd_bwd`, XLA can move the cast ahead of the
all-reduce, which then runs in `grad_dtype`; cast later, it cannot. Every other gradient is still cast
inside `fwd_bwd`, which keeps most of its output in `grad_dtype`.

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
| `cast_grads_after_all_reduce` (default false) | gradients that `fwd_bwd` sums across devices by an all-reduce alone leave it in the parameters' dtype and are cast to `grad_dtype` as they join the accumulator (see [One optimizer step](#one-optimizer-step)). Meant for TPU together with the libtpu flag `--xla_tpu_enable_offloading_copy_to_sparsecore=false`; see [Running fast on TPU](#running-fast-on-tpu). No effect when the parameters are already in `grad_dtype`, on the eager path (no `compile()`), or under the deferred data-parallel all-reduce, where `fwd_bwd` reduces no gradient |
| `optimizer_memory_host_offload` | the optimizer state lives in pinned host memory; `update` moves it to the device and back, so it is not resident during the forward and backward passes |
| `parameter_memory_host_offload` | not supported; raises |
| `shard_optimizer_over_data` (Zero-1) | shards the optimizer state over the `data` axis inside `update`; requires `shard_mode=explicit` |
| `skip_step_on_nan` (default false), `skip_step_on_spikes` | a non-finite or spiking step leaves the state unchanged and records `step_skipped` |
| `trainable_parameters_mask` | parameters it does not match are frozen. `fwd_bwd` stops their gradient inside the function it differentiates, so it returns zeros for them without computing a weight gradient, or reducing one inside `fwd_bwd`. The zeros keep their slots in the accumulator (and in `update`'s all-reduce, when that is deferred), so kernel signatures and checkpoints are unchanged. `update` zeroes them again before the gradient norm, clipping and the optimizer, so the norm and clipping cover the trainable parameters only |
| `enable_checkpointing`, `save_checkpoint_on_start`, `save_checkpoint_on_completion`, `colocated_python_checkpointing`, `checkpoint_storage_device_host_concurrent_gb` | `enable_checkpointing=false` disables saving and restoring the train state (`load_parameters_path` still loads weights); `save_checkpoint_on_start` (default true) saves on the first `save_checkpoint()` call when no checkpoint exists, and `save_checkpoint_on_completion` (default true) saves at `self.train_step` inside `close()` even when not divisible by `checkpoint_period` (when idle between steps). `colocated_python_checkpointing=true` (requires `enable_single_controller=true`) dispatches Pathways checkpoint writes and restores through colocated Python instead of Pathways persistence, with `checkpoint_storage_device_host_concurrent_gb` bounding concurrent device-to-host byte transfers. Checkpoints are taken between optimizer steps only: `save_checkpoint` raises mid-step |
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
micro-batch. It covers only those kernels: `fwd_only` scoring, the eval kernel, generated code and
anything the caller holds are excluded.

On TPU, `jax.Device.memory_stats()["peak_bytes_in_use"]` excludes XLA program temporaries, so it is
not a substitute for this report.

The report compiles on a CPU host targeting the requested TPU topology under `jax.set_mesh`, so
mesh-aware guards such as `pltpu.is_tpu_device()` resolve to the target TPU topology and compile
their TPU kernels, whereas guards that inspect the active host backend (such as
`jax.default_backend() == "cpu"`) still take their CPU fallback paths. The compiled programs can
therefore differ slightly from the ones that run on TPU, and so can their memory. Treat the peak as
an estimate, and compare configurations with each other.

## Tests

The engine's unit tests are in `tests/post_training/unit/` and run on CPU (integration tests live in
`tests/post_training/integration/`); they require Tunix.

| test | covers |
|---|---|
| `maxtext_engine_test.py` | optimizer offload placement, numerics and restore; gradient dtypes; sharding rules during model build and eval; step guards; Zero-1 |
| `tests/post_training/integration/maxtext_engine_parity_test.py` | numeric parity between `MaxTextTrainingEngine` and `pre_train/train.py`, including MoE load-balancing loss under gradient accumulation |
| `maxtext_engine_frozen_grad_test.py` | `trainable_parameters_mask`: `fwd_bwd` has no weight-gradient matmul for a frozen kernel and returns zero for it, while the other gradients, the gradient norms and the trained weights stay bitwise those of computing it, under fsdp, under the deferred all-reduce and on the eager path; a stop on a trainable kernel is caught |
| `prepare_weight_sync_test.py` | Raiden weight sync buffer purging and step metadata |
| `maxtext_engine_{constructor,data_parallel,xaot,packing,profiling}_test.py`, `router_replay_engine_test.py` | construction, data parallelism and `cast_grads_after_all_reduce` (deferred cast leaving all-reduced gradients uncast until accumulation), ahead-of-time compilation (including Qwen3.5 compile parity with the live engine), memory report against real XLA compiles (including optimizer offload), sequence packing, profiling, router replay |
