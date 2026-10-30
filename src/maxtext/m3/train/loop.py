# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal m3 training loop, intended for collecting profiles.

Deliberately bare: synthetic data, AdamW at a constant learning rate, plain
next-token cross entropy, no checkpointing, no eval, no metrics writer. It
exists so an m3 model can be stepped end to end and traced; it is not a
replacement for `maxtext.trainers.pre_train.train`.

Accepts typed `(ModelConfig, TrainingConfig)` directly via `ArgumentParser`
(zero-YAML `--model_name` CLI), or a transitional `pyconfig.HyperParameters`
object bridged via `from_pyconfig` for backward compatibility with existing
`base.yml key=value` invocations. `gradient_accumulation_steps > 1`,
`grad_dtype` other than float32, and batch sizes that do not split over the
mesh are rejected rather than emulated.

`setup_model` is where infrastructure is composed, in the order every m3
driver follows:

    construct -> scan -> quantize -> named scopes -> optimizer

Each transform is a function from `maxtext.m3.infra`. Named scopes apply when
`train_cfg.transforms.named_scopes` is True; the others read one config field
and are a no-op when that field is off. Models never see any of it.

Examples (CPU or TPU)::

    # Native zero-YAML CLI (via `ArgumentParser`):
    python3 -m maxtext.m3.train.loop \\
      --model_name=qwen3-0.6b --scan_layers=false \\
      --per_device_batch_size=1 --max_seq_length=256 --steps=8 \\
      --profiler=xplane --skip_first_n_steps_for_profiler=3 --profiler_steps=3 \\
      --output_dir=/tmp/m3 --run_name=qwen3_profile

    # Transitional YAML CLI (via `pyconfig.initialize` -> `from_pyconfig`):
    python3 -m maxtext.m3.train.loop src/maxtext/configs/base.yml \\
      model_name=qwen3-0.6b use_m3_model=true scan_layers=false \\
      per_device_batch_size=1 max_target_length=256 steps=8 \\
      profiler=xplane skip_first_n_steps_for_profiler=3 profiler_steps=3 \\
      base_output_directory=/tmp/m3 run_name=qwen3_profile \\
      enable_checkpointing=false
"""

from collections.abc import Sequence
import functools
import math
import os
import time
import types
from typing import Any

from absl import app
from absl import flags
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import optax

from maxtext.common import profiler
from maxtext.configs import pyconfig
from maxtext.m3 import models as m3_models
from maxtext.m3.configs.cli_parser import ArgumentParser
from maxtext.m3.configs.pyconfig_bridge import from_pyconfig
from maxtext.m3.configs.training_config import TrainingConfig
from maxtext.m3.core.sharding import create_mesh
from maxtext.m3.infra import quantization
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.infra.named_scopes import add_named_scopes
from maxtext.m3.models.model_config import ModelConfig
from maxtext.optimizers import optimizers
from maxtext.utils import max_logging

# Mesh axes the batch dimension is split over. Matches the defaults of
# `create_mesh` (fsdp within a slice, data across slices).
_BATCH_AXES = ("data", "fsdp")


def _resolve_configs(config: Any, train_cfg: TrainingConfig | None = None) -> tuple[ModelConfig, TrainingConfig]:
  """Normalizes either `(ModelConfig, TrainingConfig)` or a `pyconfig` object into typed configs."""
  if isinstance(config, ModelConfig):
    if train_cfg is None:
      raise TypeError("When passing a `ModelConfig`, `train_cfg: TrainingConfig` must also be provided.")
    return config, train_cfg
  if train_cfg is not None:
    raise TypeError(f"Expected `ModelConfig` when `train_cfg` is provided; got {type(config).__name__}.")
  return from_pyconfig(config)


def _global_batch_size(train_cfg: TrainingConfig) -> int:
  return int(train_cfg.per_device_batch_size * len(jax.devices()))


def _tensorboard_dir(train_cfg: TrainingConfig) -> str:
  base_dir = train_cfg.output_dir or os.path.abspath("maxtext_output")
  return os.path.join(base_dir, train_cfg.run_name, "tensorboard", "")


def _make_profiler(train_cfg: TrainingConfig) -> profiler.Profiler:
  prof_view = types.SimpleNamespace(
      **train_cfg.profiling.model_dump(),
      steps=train_cfg.steps,
      tensorboard_dir=_tensorboard_dir(train_cfg),
      managed_mldiagnostics=False,
  )
  return profiler.Profiler(prof_view, offset_step=0)


def make_optimizer(train_cfg: TrainingConfig) -> optax.GradientTransformation:
  """`optimizers.get_optimizer` plus global-norm clipping, which it leaves to the caller."""
  opt = train_cfg.optimizer
  opt_view = types.SimpleNamespace(
      opt_type=opt.optimizer_type,
      adam_b1=opt.adam_b1,
      adam_b2=opt.adam_b2,
      adam_eps=opt.adam_eps,
      adam_eps_root=opt.adam_eps_root,
      adam_weight_decay=opt.adam_weight_decay,
      mu_dtype=opt.mu_dtype.value,
      adamw_mask=opt.adamw_mask,
      trainable_parameters_mask=opt.trainable_parameters_mask,
      skip_step_on_spikes=train_cfg.skip_step_on_spikes,
      skip_step_interval=train_cfg.skip_step_interval,
      skip_step_scaling_factor=train_cfg.skip_step_scaling_factor,
  )
  tx = optimizers.get_optimizer(opt_view, opt.learning_rate)
  if opt.gradient_clipping_threshold > 0:
    return optax.chain(optax.clip_by_global_norm(opt.gradient_clipping_threshold), tx)
  return tx


def setup_model(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    mesh: Mesh,
) -> tuple[nnx.Module, nnx.Optimizer]:
  """Builds the model and optimizer with every transform applied, in the fixed order.

  Must run inside `jax.set_mesh(mesh)`. Scan comes before quantization, which
  records module paths and so must see the final structure; quantization
  traces one forward pass of the training shape. Named scopes come after both:
  they only label ops, and scan needs identical layer classes. The optimizer
  comes last, as it must see the final model.
  """
  model = m3_models.create_model(model_cfg, mesh, rngs=nnx.Rngs(train_cfg.seed))
  model = m3_scan.maybe_apply_scan(model, train_cfg.transforms)
  batch_shape = (_global_batch_size(train_cfg), train_cfg.max_seq_length)
  example_args = (jax.ShapeDtypeStruct(batch_shape, jnp.int32), jax.ShapeDtypeStruct(batch_shape, jnp.int32))
  model = quantization.maybe_quantize(model, train_cfg.transforms, *example_args)
  if train_cfg.transforms.named_scopes:
    model = add_named_scopes(model)
  optimizer = nnx.Optimizer(model, make_optimizer(train_cfg), wrt=nnx.Param)
  return model, optimizer


@functools.lru_cache(maxsize=32)
def _synthetic_batch_fn(batch_size: int, seq_len: int, vocab_size: int, mesh: Mesh):
  """Jitted batch generator for one shape, with output sharded along the batch dimension.

  One asynchronously dispatched program instead of several eager ops plus a
  `device_put`. This avoids idle time at the start of each step. `out_shardings`
  (rather than `with_sharding_constraint`) keeps the exact `P(_BATCH_AXES)` spec
  instead of letting XLA drop size-1 axes.
  """

  def body(key: jax.Array) -> dict[str, jax.Array]:
    tokens = jax.random.randint(key, (batch_size, seq_len + 1), 0, vocab_size, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
    return {"inputs": tokens[:, :-1], "targets": tokens[:, 1:], "positions": positions}

  return jax.jit(body, out_shardings=NamedSharding(mesh, P(_BATCH_AXES)))


def make_synthetic_batch(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    mesh: Mesh,
    key: jax.Array,
) -> dict[str, jax.Array]:
  """Random token batch, sharded along the batch dimension.

  Targets are the inputs shifted left by one, so the loss is ordinary
  next-token prediction rather than a constant.
  """
  fn = _synthetic_batch_fn(
      _global_batch_size(train_cfg),
      train_cfg.max_seq_length,
      model_cfg.vocab_size,
      mesh,
  )
  return fn(key)


def loss_fn(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
  """Mean next-token cross entropy, computed in float32."""
  logits = model(batch["inputs"], batch["positions"])
  with jax.named_scope("cross_entropy"):
    losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch["targets"])
    return jnp.mean(losses)


@nnx.jit(donate_argnums=(0, 1))
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, jax.Array]) -> jax.Array:
  """One forward/backward/update. Mutates `model` and `optimizer` in place."""
  loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
  with jax.named_scope("optimizer"):
    optimizer.update(model, grads)
  return loss


def _log_step(step: int, loss: jax.Array, last_done: float, tokens_per_step: int) -> tuple[float, float]:
  """Blocks until `step` finishes, logs it, and returns its loss and completion time."""
  loss = float(loss)
  done = time.perf_counter()
  step_time = done - last_done
  max_logging.log(f"step {step}: loss {loss:.4f}, {step_time:.3f}s, {tokens_per_step / step_time:,.0f} tokens/s")
  return loss, done


def train_loop(config: Any, train_cfg: TrainingConfig | None = None) -> float:
  """Builds the model and optimizer, runs `steps` steps, and returns the last loss."""
  model_cfg, train_cfg = _resolve_configs(config, train_cfg)

  # `create_mesh()` defaults (fsdp within a slice, data across slices) are used as
  # is; the `ici`/`dcn` mesh dicts are not wired through yet.
  mesh = create_mesh()
  max_logging.log(f"m3 loop: mesh {dict(mesh.shape)} over {len(jax.devices())} device(s)")

  if train_cfg.gradient_accumulation_steps > 1:
    raise ValueError(
        "m3 loop does not accumulate gradients yet; "
        f"got gradient_accumulation_steps={train_cfg.gradient_accumulation_steps}."
    )
  grad_dtype = jnp.dtype(train_cfg.grad_dtype.value)
  if grad_dtype != jnp.float32:
    raise ValueError(f"m3 loop keeps gradients in float32; got grad_dtype={grad_dtype.name}.")
  batch_size = _global_batch_size(train_cfg)
  batch_shards = math.prod(mesh.shape[axis] for axis in _BATCH_AXES)
  if batch_size == 0 or batch_size % batch_shards:
    raise ValueError(
        f"per_device_batch_size={train_cfg.per_device_batch_size} gives a global batch of {batch_size}, "
        f"which does not split over the {batch_shards} shards of mesh axes {_BATCH_AXES}."
    )

  matmul_precision = train_cfg.matmul_precision.value
  quant_name = train_cfg.transforms.quantization.value or "none"
  with jax.set_mesh(mesh), jax.default_matmul_precision(matmul_precision):
    model, optimizer = setup_model(model_cfg, train_cfg, mesh)

    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    tokens_per_step = batch_size * train_cfg.max_seq_length
    max_logging.log(
        f"m3 loop: {model_cfg.model_name}, {num_params / 1e9:.3f}B params, "
        f"batch {batch_size} x {train_cfg.max_seq_length} tokens, "
        f"matmul_precision={matmul_precision}, "
        f"quantization={quant_name}"
    )

    prof = _make_profiler(train_cfg)
    data_key = jax.random.key(train_cfg.seed)
    loss = float("nan")
    # Walk the model and optimizer graphs once before the loop so `nnx.jit`
    # does not split and merge them on every step.
    step_fn = nnx.cached_partial(train_step, model, optimizer)
    batch_fn = _synthetic_batch_fn(batch_size, train_cfg.max_seq_length, model_cfg.vocab_size, mesh)
    # Steps are logged one iteration late. Reading a loss blocks until its step
    # finishes, so doing it right after dispatch would leave the device idle
    # while the host prepares the next step. Reading step N-1's loss after
    # dispatching step N keeps one step queued. Step time is the interval
    # between consecutive completions, so step 0's includes compilation.
    pending = None  # (step, loss) dispatched but not yet logged.
    last_done = time.perf_counter()
    for step in range(train_cfg.steps):
      batch = batch_fn(jax.random.fold_in(data_key, step))
      prof.maybe_activate_profiler(step, pending)
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        step_loss = step_fn(batch)
      if pending is not None:
        loss, last_done = _log_step(*pending, last_done, tokens_per_step)
      pending = (step, step_loss)
      prof.maybe_deactivate_profiler(step, pending)
    if pending is not None:
      loss, last_done = _log_step(*pending, last_done, tokens_per_step)

    if train_cfg.profiling.profiler:
      max_logging.log(f"m3 loop: profile written under {_tensorboard_dir(train_cfg)}")
    return loss


def parse_argv(argv: Sequence[str]) -> tuple[ModelConfig, TrainingConfig]:
  """Dispatches `*.yml`/`*.yaml` config invocations (`argv[1]`) to `from_pyconfig`, else `ArgumentParser`."""
  if len(argv) >= 2 and not argv[1].startswith("-") and argv[1].lower().endswith((".yml", ".yaml")):
    return from_pyconfig(pyconfig.initialize(list(argv)))
  return ArgumentParser().parse_args(argv)


def main(argv: Sequence[str]) -> None:
  model_cfg, train_cfg = parse_argv(argv)
  train_loop(model_cfg, train_cfg)


if __name__ == "__main__":
  app.run(main, flags_parser=lambda argv: flags.FLAGS(argv, known_only=True))
