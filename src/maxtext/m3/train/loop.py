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

Configuration accepts typed `(ModelConfig, TrainingConfig)` directly via
`ArgumentParser` (zero-YAML `--model_name` CLI), or a legacy `pyconfig`
(`MaxTextConfig` / `HyperParameters`) object for backwards compatibility with
existing `base.yml key=value` invocations (bridged via `from_pyconfig` at the
CLI entrypoint `parse_argv`). `gradient_accumulation_steps > 1`, `grad_dtype`
other than float32, and batch or sequence sizes that do not split over the mesh
are rejected rather than emulated.

`setup_model` is where infrastructure is composed, in the order every m3
driver follows:

    construct -> scan -> quantize -> named scopes -> optimizer

Each transform is a function from `maxtext.m3.infra`. Named scopes apply when
`train_cfg.transforms.named_scopes` is True (always for legacy configs); the
others read one config field and are a no-op when that field is off. Models
never see any of it.

Precision follows legacy MaxText: `param_dtype` / `weight_dtype` (float32
master weights) and `compute_dtype` / `dtype` (bfloat16 compute) are applied by
the model at construction; the optimizer keeps its moments in `mu_dtype` /
`weight_dtype`; the loss is computed in float32; and `matmul_precision` is
applied here, by running setup and every step under
`jax.default_matmul_precision`. It only affects float32 matmul operands -- on
TPU, `default` is one bfloat16 pass and `highest` is full float32 -- so with
bfloat16 compute it is a no-op, as in legacy.

Parameters are sharded by the model at construction from its own `Sharding`
strategy (`DefaultSharding` for Qwen3: FSDP over the model dimension, tensor
parallelism over heads and MLP width, optional sequence parallelism over norms
and residuals, and batch over the data axes). `train_loop` resolves
`sharding_from_config(config)` to lay out the input batch and logs the fraction
of parameters each device holds, so replication is visible.

Examples (CPU or TPU)::

    # Native zero-YAML CLI (via `ArgumentParser`):
    python3 -m maxtext.m3.train.loop \\
      --model_name=qwen3-0.6b --transforms.scan_layers=false \\
      --per_device_batch_size=1 --max_target_length=256 --steps=8 \\
      --profiling.profiler=xplane --profiling.skip_first_n_steps_for_profiler=3 \\
      --profiling.profiler_steps=3 --base_output_directory=/tmp/m3 \\
      --run_name=qwen3_profile

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
from maxtext.m3.core import kernels as m3_kernels
from maxtext.m3.core.sharding import (
    MESH_AXIS_NAMES,
    Sharding,
    TensorType,
    ambient_mesh_is_explicit,
    mesh_from_config,
    project_spec_to_mesh,
    sharding_from_config,
)
from maxtext.m3.infra import quantization
from maxtext.m3.infra import scan as m3_scan
from maxtext.m3.infra.named_scopes import add_named_scopes
from maxtext.m3.models.model_config import ModelConfig
from maxtext.optimizers import optimizers
from maxtext.utils import max_logging
from maxtext.utils import max_utils


def _validate_config_pair(config: Any, train_cfg: TrainingConfig | None) -> bool:
  """Returns True when called with a typed `(ModelConfig, TrainingConfig)` pair; False for legacy `config`."""
  if isinstance(config, ModelConfig):
    if train_cfg is None:
      raise TypeError("When passing a `ModelConfig`, `train_cfg: TrainingConfig` must also be provided.")
    if not isinstance(train_cfg, TrainingConfig):
      raise TypeError(f"Expected `TrainingConfig` for `train_cfg`; got {type(train_cfg).__name__}.")
    return True
  if train_cfg is not None:
    raise TypeError(f"Expected `ModelConfig` when `train_cfg` is provided; got {type(config).__name__}.")
  return False


def _global_batch_size(train_cfg: TrainingConfig) -> int:
  """Returns the global per-step batch size derived from `per_device_batch_size` and device count."""
  return int(train_cfg.per_device_batch_size * len(jax.devices()))


def _tensorboard_dir(train_cfg: TrainingConfig) -> str:
  """Derives the profiler/TensorBoard directory matching `MaxTextConfig.tensorboard_dir`."""
  base_dir = train_cfg.base_output_directory or os.path.abspath("maxtext_output")
  if train_cfg.run_name:
    return os.path.join(base_dir, train_cfg.run_name, "tensorboard", "")
  return os.path.join(base_dir, "tensorboard", "")


def _mesh_config_view(model_cfg: ModelConfig, train_cfg: TrainingConfig) -> types.SimpleNamespace:
  """Builds a `mesh_from_config`-compatible namespace from `(model_cfg, train_cfg.mesh)`."""
  ici = train_cfg.mesh.ici or {"fsdp": -1}
  dcn = train_cfg.mesh.dcn or {"data": -1}
  fields: dict[str, Any] = {
      "model_name": model_cfg.model_name,
      "shard_mode": train_cfg.mesh.shard_mode,
  }
  for axis in MESH_AXIS_NAMES:
    if axis == "stage":
      continue
    fields[f"ici_{axis}_parallelism"] = ici.get(axis, 1)
    fields[f"dcn_{axis}_parallelism"] = dcn.get(axis, 1)
  return types.SimpleNamespace(**fields)


def _optimizer_view(train_cfg: TrainingConfig) -> types.SimpleNamespace:
  """Builds an `optimizers.get_optimizer`-compatible namespace from `TrainingConfig`."""
  opt = train_cfg.optimizer
  return types.SimpleNamespace(
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


def _profiler_view(train_cfg: TrainingConfig) -> types.SimpleNamespace:
  """Builds a `profiler.Profiler`-compatible namespace from `TrainingConfig`."""
  prof_fields = {k: getattr(v, "value", v) for k, v in train_cfg.profiling.model_dump().items()}
  return types.SimpleNamespace(
      **prof_fields,
      steps=train_cfg.steps,
      tensorboard_dir=_tensorboard_dir(train_cfg),
      managed_mldiagnostics=False,
  )


def make_optimizer(config: Any) -> optax.GradientTransformation:
  """`optimizers.get_optimizer` plus global-norm clipping, which it leaves to the caller."""
  if isinstance(config, TrainingConfig):
    tx = optimizers.get_optimizer(_optimizer_view(config), config.optimizer.learning_rate)
    clip = config.optimizer.gradient_clipping_threshold
  else:
    tx = optimizers.get_optimizer(config, config.learning_rate)
    clip = config.gradient_clipping_threshold
  if clip > 0:
    return optax.chain(optax.clip_by_global_norm(clip), tx)
  return tx


def setup_model(
    config: Any,
    mesh: Mesh,
    sharding: Sharding | None = None,
    *,
    train_cfg: TrainingConfig | None = None,
) -> tuple[nnx.Module, nnx.Optimizer]:
  """Builds the model and optimizer with every transform applied, in the fixed order.

  Must run inside `jax.set_mesh(mesh)`. Accepts either a typed `ModelConfig`
  (with keyword `train_cfg: TrainingConfig`) or a legacy `pyconfig` config. The
  model places its parameters from `sharding` (or its own default strategy when
  `sharding` is `None`) as it is constructed. Scan comes before quantization,
  which records module paths and so must see the final structure; quantization
  traces one forward pass of the training shape. Named scopes come after both:
  they only label ops, and scan needs identical layer classes. The optimizer
  comes last, as it must see the final model.
  """
  is_typed = _validate_config_pair(config, train_cfg)
  create_kw = {"sharding": sharding} if sharding is not None else {}
  if is_typed:
    assert train_cfg is not None
    seed = train_cfg.seed
    tx_cfg: Any = train_cfg.transforms
    opt_cfg: Any = train_cfg
    batch_shape = (_global_batch_size(train_cfg), train_cfg.max_target_length)
    example_args = (jax.ShapeDtypeStruct(batch_shape, jnp.int32), jax.ShapeDtypeStruct(batch_shape, jnp.int32))
    apply_scopes = train_cfg.transforms.named_scopes
  else:
    seed = config.init_weights_seed
    tx_cfg = config
    opt_cfg = config
    example_args = quantization.example_inputs(config)
    apply_scopes = True

  model = m3_models.create_model(config, mesh, rngs=nnx.Rngs(seed), **create_kw)
  model = m3_scan.maybe_apply_scan(model, tx_cfg)
  quant_val = getattr(tx_cfg, "quantization", "")
  if getattr(quant_val, "value", quant_val) and ambient_mesh_is_explicit():
    raise ValueError(
        "quantization is not supported with shard_mode='explicit' yet: qwix drops out_sharding on quantized matmuls. "
        "Use shard_mode='auto'."
    )
  model = quantization.maybe_quantize(model, tx_cfg, *example_args)
  if apply_scopes:
    model = add_named_scopes(model)
  optimizer = nnx.Optimizer(model, make_optimizer(opt_cfg), wrt=nnx.Param)
  return model, optimizer


def batch_spec(sharding: Sharding) -> P:
  """How a `(batch, length)` token array is laid out: the strategy's activation rule for `batch`."""
  return sharding("input_tokens", ("batch", "length"), TensorType.ACTIVATION)


def _axis_shards(mesh: Mesh, spec: P, dim: int) -> int:
  """Number of ways `spec` splits dimension `dim` on `mesh`."""
  projected = project_spec_to_mesh(spec, mesh)
  axes = projected[dim] if len(projected) > dim else None
  axes = () if axes is None else ((axes,) if isinstance(axes, str) else tuple(axes))
  return math.prod(mesh.shape[axis] for axis in axes)


def _batch_shards(mesh: Mesh, spec: P) -> int:
  """Number of ways `spec` splits the batch dimension on `mesh`."""
  return _axis_shards(mesh, spec, 0)


def params_held_per_device(model: nnx.Module) -> float:
  """Fraction of the parameter elements each device holds: 1.0 means fully replicated."""
  params = nnx.state(model, nnx.Param)
  if not jax.tree.leaves(params):
    return 1.0
  return float(max_utils.calculate_total_params_per_chip(params) / max_utils.calculate_num_params_from_pytree(params))


@functools.lru_cache(maxsize=32)
def _synthetic_batch_fn(batch_size: int, seq_len: int, vocab_size: int, mesh: Mesh, spec: P):
  """Jitted batch generator for one shape, with output laid out as `spec`.

  One asynchronously dispatched program instead of several eager ops plus a
  `device_put`. This avoids idle time at the start of each step. `out_shardings`
  (rather than `with_sharding_constraint`) keeps the exact spec instead of
  letting XLA drop size-1 axes.
  """

  def body(key: jax.Array) -> dict[str, jax.Array]:
    tokens = jax.random.randint(key, (batch_size, seq_len + 1), 0, vocab_size, dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32), (batch_size, seq_len))
    return {"inputs": tokens[:, :-1], "targets": tokens[:, 1:], "positions": positions}

  return jax.jit(body, out_shardings=NamedSharding(mesh, project_spec_to_mesh(spec, mesh)))


def make_synthetic_batch(
    config: Any,
    mesh: Mesh,
    spec: P | Sharding,
    key: jax.Array,
    *,
    train_cfg: TrainingConfig | None = None,
) -> dict[str, jax.Array]:
  """Random token batch, laid out as `spec` (`batch_spec` of the strategy).

  Targets are the inputs shifted left by one, so the loss is ordinary
  next-token prediction rather than a constant.
  """
  is_typed = _validate_config_pair(config, train_cfg)
  if is_typed:
    assert train_cfg is not None
    batch_size = _global_batch_size(train_cfg)
    seq_len = train_cfg.max_target_length
  else:
    batch_size = config.global_batch_size_to_train_on
    seq_len = config.max_target_length
  spec_p = batch_spec(spec) if isinstance(spec, Sharding) else spec
  fn = _synthetic_batch_fn(batch_size, seq_len, config.vocab_size, mesh, spec_p)
  return fn(key)


def loss_fn(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
  """Mean next-token cross entropy, computed in float32."""
  logits = model(batch["inputs"], batch["positions"])
  with jax.named_scope("cross_entropy"):
    logits_fp32 = logits.astype(jnp.float32)
    if not jax.sharding.get_abstract_mesh().empty and ambient_mesh_is_explicit():
      target_sharding = jax.typeof(batch["targets"]).sharding
      target_spec = target_sharding.spec if hasattr(target_sharding, "spec") else P()
      losses = jax.sharding.auto_axes(
          optax.softmax_cross_entropy_with_integer_labels,
          out_sharding=target_spec,
      )(logits_fp32, batch["targets"])
    else:
      losses = optax.softmax_cross_entropy_with_integer_labels(logits_fp32, batch["targets"])
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


def _extract_runtime_view(config: Any, train_cfg: TrainingConfig | None) -> types.SimpleNamespace:
  """Extracts flat scalar runtime settings for either `(ModelConfig, TrainingConfig)` or a legacy config."""
  if _validate_config_pair(config, train_cfg):
    assert train_cfg is not None
    return types.SimpleNamespace(
        mesh_cfg=_mesh_config_view(config, train_cfg),
        attn_cfg=config,
        model_name=config.model_name,
        steps=train_cfg.steps,
        seed=train_cfg.seed,
        per_device_batch_size=train_cfg.per_device_batch_size,
        batch_size=_global_batch_size(train_cfg),
        max_target_length=train_cfg.max_target_length,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        grad_dtype=jnp.dtype(train_cfg.grad_dtype.value),
        matmul_precision=train_cfg.matmul_precision.value,
        quantization=train_cfg.transforms.quantization.value or "none",
        prof_cfg=_profiler_view(train_cfg),
        profiler_enabled=bool(train_cfg.profiling.profiler.value),
        tensorboard_dir=_tensorboard_dir(train_cfg),
    )
  grad_dt = getattr(config.grad_dtype, "value", config.grad_dtype)
  prof_mode = getattr(config.profiler, "value", config.profiler)
  return types.SimpleNamespace(
      mesh_cfg=config,
      attn_cfg=config,
      model_name=config.model_name,
      steps=config.steps,
      seed=config.init_weights_seed,
      per_device_batch_size=config.per_device_batch_size,
      batch_size=config.global_batch_size_to_train_on,
      max_target_length=config.max_target_length,
      gradient_accumulation_steps=config.gradient_accumulation_steps,
      grad_dtype=jnp.dtype(grad_dt),
      matmul_precision=getattr(config.matmul_precision, "value", config.matmul_precision),
      quantization=getattr(config.quantization, "value", config.quantization) or "none",
      prof_cfg=config,
      profiler_enabled=bool(prof_mode),
      tensorboard_dir=config.tensorboard_dir,
  )


def train_loop(config: Any, train_cfg: TrainingConfig | None = None) -> float:
  """Builds the model and optimizer, runs `steps` steps, and returns the last loss.

  Accepts either `(model_cfg: ModelConfig, train_cfg: TrainingConfig)` or a
  single legacy `pyconfig` (`MaxTextConfig` / `HyperParameters`) object.
  """
  rt = _extract_runtime_view(config, train_cfg)
  mesh = mesh_from_config(rt.mesh_cfg)
  m3_kernels.validate_attention_config(rt.attn_cfg, mesh=mesh)
  # Lays out the input batch; `setup_model` lets the model resolve its own
  # parameter strategy from `config`.
  sharding = sharding_from_config(config)
  explicit = all(t == jax.sharding.AxisType.Explicit for t in mesh.axis_types)
  max_logging.log(
      f"m3 loop: mesh {dict(mesh.shape)} over {len(jax.devices())} device(s), "
      f"{'explicit' if explicit else 'auto'} sharding"
  )

  # `global_batch_size_to_train_on` already includes gradient accumulation, and
  # the loop does none, so GA > 1 would silently run one GA-times-larger step.
  if rt.gradient_accumulation_steps > 1:
    raise ValueError(
        f"m3 loop does not accumulate gradients yet; got gradient_accumulation_steps={rt.gradient_accumulation_steps}."
    )
  # Legacy casts gradients to `grad_dtype` before the optimizer; this loop keeps
  # them in float32, so any other setting would be ignored rather than applied.
  if rt.grad_dtype != jnp.float32:
    raise ValueError(f"m3 loop keeps gradients in float32; got grad_dtype={rt.grad_dtype.name}.")
  spec = batch_spec(sharding)
  batch_shards = _batch_shards(mesh, spec)
  if rt.batch_size == 0 or rt.batch_size % batch_shards:
    raise ValueError(
        f"per_device_batch_size={rt.per_device_batch_size} gives a global batch of {rt.batch_size}, which does not "
        f"split over the {batch_shards} shards of {spec}."
    )
  norm_seq_spec = sharding("decoder_layer_input", ("batch", "norm_length", "embed"), TensorType.ACTIVATION)
  seq_shards = _axis_shards(mesh, norm_seq_spec, 1)
  if rt.max_target_length % seq_shards:
    raise ValueError(
        f"max_target_length={rt.max_target_length} is not divisible by the {seq_shards} sequence shards of "
        f"{norm_seq_spec}."
    )

  with jax.set_mesh(mesh), jax.default_matmul_precision(rt.matmul_precision):
    model, optimizer = setup_model(config, mesh, train_cfg=train_cfg)

    num_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    tokens_per_step = rt.batch_size * rt.max_target_length
    max_logging.log(
        f"m3 loop: {rt.model_name}, {num_params / 1e9:.3f}B params, "
        f"{params_held_per_device(model):.1%} of them per device, "
        f"batch {rt.batch_size} x {rt.max_target_length} tokens, "
        f"matmul_precision={rt.matmul_precision}, "
        f"quantization={rt.quantization}"
    )

    prof = profiler.Profiler(rt.prof_cfg, offset_step=0)
    data_key = jax.random.key(rt.seed)
    loss = float("nan")
    # Walk the model and optimizer graphs once before the loop so `nnx.jit`
    # does not split and merge them on every step.
    step_fn = nnx.cached_partial(train_step, model, optimizer)
    # Steps are logged one iteration late. Reading a loss blocks until its step
    # finishes, so doing it right after dispatch would leave the device idle
    # while the host prepares the next step. Reading step N-1's loss after
    # dispatching step N keeps one step queued. Step time is the interval
    # between consecutive completions, so step 0's includes compilation.
    pending = None  # (step, loss) dispatched but not yet logged.
    last_done = time.perf_counter()
    for step in range(rt.steps):
      batch = make_synthetic_batch(config, mesh, spec, jax.random.fold_in(data_key, step), train_cfg=train_cfg)
      prof.maybe_activate_profiler(step, pending)
      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        step_loss = step_fn(batch)
      if pending is not None:
        loss, last_done = _log_step(*pending, last_done, tokens_per_step)
      pending = (step, step_loss)
      prof.maybe_deactivate_profiler(step, pending)
    if pending is not None:
      loss, last_done = _log_step(*pending, last_done, tokens_per_step)

    if rt.profiler_enabled:
      max_logging.log(f"m3 loop: profile written under {rt.tensorboard_dir}")
    return loss


def parse_argv(argv: Sequence[str]) -> tuple[ModelConfig, TrainingConfig]:
  """Dispatches `*.yml`/`*.yaml` config paths (`argv[1]`) to `from_pyconfig`, else `ArgumentParser`."""
  if len(argv) >= 2 and not argv[1].startswith("-") and argv[1].lower().endswith((".yml", ".yaml")):
    return from_pyconfig(pyconfig.initialize(list(argv)))
  cli_tokens = (
      argv[1:] if argv and not argv[0].startswith("-") and not argv[0].lower().endswith((".yml", ".yaml")) else argv
  )
  return ArgumentParser().parse_args(cli_tokens)


def main(argv: Sequence[str]) -> None:
  model_cfg, train_cfg = parse_argv(argv)
  train_loop(model_cfg, train_cfg)


if __name__ == "__main__":
  app.run(main, flags_parser=lambda argv: flags.FLAGS(argv, known_only=True))
