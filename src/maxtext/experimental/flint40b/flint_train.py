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

"""Sharded `FlintTrainState`, Optax AdamW, LR schedule, and `train_step`."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from typing import Any, Generic, cast

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
import optax

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_model
from maxtext.experimental.flint40b import flint_sharding
from maxtext.experimental.flint40b import flint_types

P = PartitionSpec


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class FlintTrainState(Generic[flint_types.T]):
  """Sharded training state (`step`, master `params`, FP32 `opt_state`)."""

  step: flint_types.T
  params: flint_types.FlintParams[flint_types.T]
  opt_state: Any


def build_lr_schedule(
    train_cfg: flint_config.FlintTrainConfig,
) -> optax.Schedule:
  """Builds the Optax learning-rate schedule (`'constant'` or `'warmup_cosine'`).

  Args:
    train_cfg: Validated `FlintTrainConfig`.

  Returns:
    Callable Optax schedule mapping `int32` step count -> scalar learning rate.

  Raises:
    ValueError: If `train_cfg.lr_schedule` is unsupported.
  """
  lr = float(train_cfg.learning_rate)
  warmup = int(train_cfg.warmup_steps)
  total = int(train_cfg.total_steps)

  if train_cfg.lr_schedule == "constant":
    if warmup <= 0:
      return optax.constant_schedule(lr)
    warm_sched = optax.linear_schedule(
        init_value=0.0,
        end_value=lr,
        transition_steps=warmup,
    )
    const_sched = optax.constant_schedule(lr)
    return optax.join_schedules(
        schedules=[warm_sched, const_sched],
        boundaries=[warmup],
    )

  if train_cfg.lr_schedule == "warmup_cosine":
    if warmup <= 0:
      return optax.cosine_decay_schedule(
          init_value=lr,
          decay_steps=max(total, 1),
          alpha=float(train_cfg.min_lr_ratio),
      )
    if warmup >= total:
      return optax.linear_schedule(
          init_value=0.0,
          end_value=lr,
          transition_steps=max(warmup, 1),
      )
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=total,
        end_value=lr * float(train_cfg.min_lr_ratio),
    )

  raise ValueError(f"Unsupported lr_schedule: {train_cfg.lr_schedule!r}.")


def _should_weight_decay_path(path: tuple[Any, ...]) -> bool:
  """Excludes `embed`, `sinks`, and all RMSNorm parameters from weight decay."""
  for entry in path:
    if isinstance(entry, jax.tree_util.GetAttrKey):
      if entry.name in ("embed", "sinks") or "norm" in entry.name:
        return False
  return True


def build_weight_decay_mask(params: Any) -> Any:
  """Builds a boolean PyTree mask (`False` on `embed`, `*norm*`, and `sinks`)."""
  return jax.tree_util.tree_map_with_path(lambda path, _: _should_weight_decay_path(path), params)


def create_optimizer(
    cfg: flint_config.FlintConfig,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
  """Creates the Optax AdamW transformation (FP32 moments) and LR schedule.

  Args:
    cfg: Validated `FlintConfig`.

  Returns:
    Tuple `(tx, lr_sched)` where `tx` expects FP32-cast `params` and `grads` so
    both `mu` and `nu` remain in `jnp.float32` regardless of `param_dtype`.
  """
  t = cfg.train
  lr_sched = build_lr_schedule(t)
  wd_mask = build_weight_decay_mask if t.wd_exclude_norms_and_embed else None
  tx = optax.adamw(
      learning_rate=lr_sched,
      b1=t.adam_b1,
      b2=t.adam_b2,
      eps=t.adam_eps,
      weight_decay=t.weight_decay,
      mu_dtype=jnp.float32,
      mask=wd_mask,
  )
  return tx, lr_sched


def abstract_train_state(
    cfg: flint_config.FlintConfig,
    mesh: Mesh | None = None,
) -> FlintTrainState[jax.ShapeDtypeStruct]:
  """Returns abstract `ShapeDtypeStruct` PyTree for `FlintTrainState`.

  Both `mu` and `nu` inside `opt_state` have `jnp.float32` leaves matching
  `flint_sharding.estimate_persistent_state_bytes_per_device`.

  Args:
    cfg: Validated `FlintConfig`.
    mesh: Optional `Mesh` (accepted for symmetry with `state_shardings`).

  Returns:
    `FlintTrainState[jax.ShapeDtypeStruct]`.
  """
  del mesh
  param_sds = flint_sharding.abstract_param_shapes(cfg)
  param_f32_sds = jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, jnp.float32), param_sds)
  tx, _ = create_optimizer(cfg)
  opt_sds = jax.eval_shape(tx.init, cast(Any, param_f32_sds))
  step_sds = jax.ShapeDtypeStruct((), jnp.int32)
  return FlintTrainState(
      step=step_sds,
      params=param_sds,
      opt_state=opt_sds,
  )


def state_shardings(
    mesh: Mesh,
    cfg: flint_config.FlintConfig,
) -> FlintTrainState[NamedSharding]:
  """Returns exact `NamedSharding` PyTree for `FlintTrainState` on `mesh`."""
  rep_sharding = NamedSharding(mesh, P())
  param_shardings = flint_sharding.named_param_shardings(mesh, cfg)
  abs_state = abstract_train_state(cfg, mesh)
  opt_shardings = jax.tree.map(
      lambda node: (param_shardings if isinstance(node, flint_types.FlintParams) else rep_sharding),
      abs_state.opt_state,
      is_leaf=lambda node: isinstance(node, flint_types.FlintParams),
  )
  return FlintTrainState(
      step=rep_sharding,
      params=param_shardings,
      opt_state=opt_shardings,
  )


def init_train_state(
    key: jax.Array,
    mesh: Mesh,
    cfg: flint_config.FlintConfig,
    *,
    params: flint_types.FlintParams[jax.Array] | None = None,
) -> FlintTrainState[jax.Array]:
  """Initializes sharded `FlintTrainState` directly on `mesh` (multi-host safe).

  Args:
    key: JAX PRNG key used when `params` is `None`.
    mesh: 3D `('fsdp', 'cp', 'ep')` JAX `Mesh`.
    cfg: Validated `FlintConfig`.
    params: Optional pre-loaded sharded `FlintParams` (e.g. from safetensors).

  Returns:
    Sharded `FlintTrainState` with `step=0`, master `params`, and FP32 AdamW
    `mu`/`nu` moments matching `state_shardings(mesh, cfg)`.
  """
  if params is None:
    params = flint_sharding.init_random_params(key, mesh, cfg)

  tx, _ = create_optimizer(cfg)
  shardings = state_shardings(mesh, cfg)

  @jax.jit(
      in_shardings=(shardings.params,),
      out_shardings=(shardings.step, shardings.opt_state),
  )
  def _init_opt(
      p: flint_types.FlintParams[jax.Array],
  ) -> tuple[jax.Array, Any]:
    p_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), p)
    return jnp.zeros((), dtype=jnp.int32), tx.init(cast(Any, p_f32))

  step_zero, opt_state = _init_opt(params)
  return FlintTrainState(
      step=step_zero,
      params=params,
      opt_state=opt_state,
  )


def compute_global_grad_norm_f32(
    grads: flint_types.FlintParams[jax.Array],
) -> jax.Array:
  """Computes global L2 norm across all `FlintParams` leaves in `float32`."""
  sq_terms = [jnp.sum(jnp.square(g.astype(jnp.float32))) for g in jax.tree.leaves(grads)]
  return jnp.sqrt(jnp.sum(jnp.stack(sq_terms)))


def make_train_step(
    cfg: flint_config.FlintConfig,
    mesh: Mesh,
    *,
    donate_state: bool = True,
) -> Callable[
    [FlintTrainState[jax.Array], flint_types.FlintBatch],
    tuple[FlintTrainState[jax.Array], flint_types.FlintTrainMetrics],
]:
  """Compiles a single-JIT sharded AdamW `train_step(state, batch)` function.

  Evaluates `grad_norm`, global-norm clipping, and AdamW moment/weight-decay
  updates in `float32`, casts updated master weights back to `p.dtype` (keeping
  `sinks` in `float32`), and records `learning_rate` at `state.step` before
  incrementing `step + 1`.

  Args:
    cfg: Validated `FlintConfig`.
    mesh: 3D `('fsdp', 'cp', 'ep')` JAX `Mesh`.
    donate_state: If `True`, donates `state` buffers (`donate_argnums=(0,)`).

  Returns:
    `@jax.jit`-compiled `(state, batch) -> (new_state, FlintTrainMetrics)`.
  """
  tx, lr_sched = create_optimizer(cfg)
  s_shardings = state_shardings(mesh, cfg)
  b_shardings = flint_model.batch_named_shardings(mesh)
  rep_sharding = NamedSharding(mesh, P())
  metrics_cls: Any = flint_types.FlintTrainMetrics
  metrics_shardings = metrics_cls(
      loss=rep_sharding,
      ce_loss=rep_sharding,
      aux_loss=rep_sharding,
      grad_norm=rep_sharding,
      learning_rate=rep_sharding,
  )

  def _train_step(
      state: FlintTrainState[jax.Array],
      batch: flint_types.FlintBatch,
  ) -> tuple[FlintTrainState[jax.Array], flint_types.FlintTrainMetrics]:
    (loss, ce_loss, aux_loss), grads_f32 = flint_model.shard_mapped_value_and_grad(
        state.params, batch, cfg, mesh, fp32_grads=True
    )
    grad_norm = compute_global_grad_norm_f32(grads_f32)

    if cfg.train.grad_clip_norm > 0.0:
      max_norm = jnp.asarray(cfg.train.grad_clip_norm, dtype=jnp.float32)
      clip_scale = jnp.minimum(1.0, max_norm / jnp.maximum(grad_norm, 1e-12))
      clipped_grads_f32 = jax.tree.map(lambda g: g * clip_scale, grads_f32)
    else:
      clipped_grads_f32 = grads_f32

    params_f32 = jax.tree.map(lambda p: p.astype(jnp.float32), state.params)
    updates_f32, new_opt_state = tx.update(cast(Any, clipped_grads_f32), state.opt_state, cast(Any, params_f32))
    new_params = jax.tree.map(
        lambda p, u: (p.astype(jnp.float32) + u.astype(jnp.float32)).astype(p.dtype),
        state.params,
        updates_f32,
    )
    lr_now = jnp.asarray(lr_sched(state.step), dtype=jnp.float32)
    new_step = state.step + jnp.asarray(1, dtype=jnp.int32)

    new_state = FlintTrainState(
        step=new_step,
        params=new_params,
        opt_state=new_opt_state,
    )
    metrics = flint_types.FlintTrainMetrics(
        loss=loss,
        ce_loss=ce_loss,
        aux_loss=aux_loss,
        grad_norm=grad_norm,
        learning_rate=lr_now,
    )
    return new_state, metrics

  return jax.jit(
      _train_step,
      in_shardings=(s_shardings, b_shardings),
      out_shardings=(s_shardings, metrics_shardings),
      donate_argnums=(0,) if donate_state else (),
  )
