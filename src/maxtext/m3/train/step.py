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

"""Pure-NNX train/eval step functions, global-token microbatch accumulation, and `RouterBias` updates."""

from collections.abc import Callable, Mapping
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

from maxtext.m3.core.data import M3Batch
from maxtext.m3.core.interfaces import RouterBias
from maxtext.m3.train.loss import LossBreakdown, compute_batch_loss_sums


def reshape_batch_to_microbatches(batch: Mapping[str, Any], gradient_accumulation_steps: int) -> M3Batch:
  """Reshapes a global `[B_global, ...]` `M3Batch` into `[G, B_micro, ...]` microbatches."""
  g_steps = int(gradient_accumulation_steps)
  if g_steps < 1:
    raise ValueError(f"`gradient_accumulation_steps` must be >= 1; got {g_steps}.")
  inputs = batch["inputs"]
  if inputs.ndim == 3:
    if inputs.shape[0] != g_steps:
      raise ValueError(
          f"Pre-shaped microbatch tensor has leading axis {inputs.shape[0]}, "
          f"expected gradient_accumulation_steps={g_steps}."
      )
    return dict(batch)  # type: ignore[return-value]

  global_batch = int(inputs.shape[0])
  if global_batch < g_steps or global_batch % g_steps != 0:
    raise ValueError(f"Global batch size {global_batch} must be divisible by gradient_accumulation_steps={g_steps}.")
  micro_batch = global_batch // g_steps

  def _split_leading(arr: Any) -> Any:
    if hasattr(arr, "shape") and len(arr.shape) >= 1 and arr.shape[0] == global_batch:
      return arr.reshape((g_steps, micro_batch, *arr.shape[1:]))
    return arr

  reshaped: dict[str, Any] = {}
  for key, val in batch.items():
    if key == "extras" and isinstance(val, Mapping):
      reshaped["extras"] = {ek: _split_leading(ev) for ek, ev in val.items()}
    else:
      reshaped[key] = _split_leading(val)
  return reshaped  # type: ignore[return-value]


def _is_param_var(x: Any) -> bool:
  return isinstance(x, nnx.Param)


def _is_router_bias_var(x: Any) -> bool:
  return isinstance(x, RouterBias)


def accumulated_loss_and_grads(
    model: nnx.Module,
    microbatches: Mapping[str, Any],
    *,
    num_vocab_tiles: int = 1,
    z_loss_multiplier: float = 0.0,
    aux_loss_weight: float = 1.0,
    grad_dtype: jnp.dtype = jnp.float32,
) -> tuple[LossBreakdown, nnx.State]:
  """Computes global-valid-token normalized loss and accumulated parameter gradients over microbatches.

  Accepts either unbatched `[B, S, ...]` (`G = 1`) or microbatch-stacked `[G, B_micro, S, ...]`
  batches. Pre-reducing `total_valid_tokens` across all `G` microbatches before `jax.lax.scan`
  guarantees that variable-density packed microbatches produce bit-equivalent (float32) loss and
  gradients to a single `G = 1` full-batch step.
  """
  inputs = microbatches["inputs"]
  is_microbatched = inputs.ndim == 3
  num_micro = int(inputs.shape[0]) if is_microbatched else 1

  weights = microbatches.get("weights")
  if weights is None:
    seg = microbatches.get("segment_ids")
    weights_f32 = jnp.ones_like(inputs, dtype=jnp.float32) if seg is None else (seg != 0).astype(jnp.float32)
  else:
    weights_f32 = weights.astype(jnp.float32)
  total_valid_tokens = jnp.maximum(jnp.sum(weights_f32), 1.0)

  graphdef, params, nondiff = nnx.split(model, nnx.Param, ...)

  def _single_mb_objective(p: nnx.State, mb: Mapping[str, Any]) -> tuple[jax.Array, LossBreakdown]:
    mod = nnx.merge(graphdef, p, nondiff)
    breakdown = compute_batch_loss_sums(
        mod,
        mb,
        num_vocab_tiles=num_vocab_tiles,
        z_loss_multiplier=z_loss_multiplier,
        aux_loss_weight=aux_loss_weight,
    )
    scaled_loss = (breakdown.xent_sum + breakdown.z_loss_sum) / total_valid_tokens + breakdown.aux_loss / float(num_micro)
    return scaled_loss, breakdown

  grad_fn = jax.value_and_grad(_single_mb_objective, has_aux=True)

  if not is_microbatched:
    (_, breakdown), grads = grad_fn(params, microbatches)
    return breakdown, grads

  if num_micro == 1:
    mb0 = jax.tree.map(lambda x: x[0], dict(microbatches))
    (_, breakdown), grads = grad_fn(params, mb0)
    return breakdown, grads

  zero_acc = jax.tree.map(
      lambda v: jnp.zeros_like(v[...], dtype=grad_dtype),
      params,
      is_leaf=_is_param_var,
  )

  def _scan_step(
      grad_acc: Any,
      mb: Mapping[str, Any],
  ) -> tuple[Any, tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]]:
    (_, mb_bd), mb_grads = grad_fn(params, mb)
    next_acc = jax.tree.map(
        lambda acc, g: acc + g[...].astype(grad_dtype),
        grad_acc,
        mb_grads,
        is_leaf=lambda x: _is_param_var(x) or isinstance(x, jax.Array),
    )
    return next_acc, (
        mb_bd.xent_sum,
        mb_bd.z_loss_sum,
        mb_bd.aux_loss / float(num_micro),
        mb_bd.router_token_counts,
    )

  grad_acc_final, (xent_ys, z_ys, aux_ys, counts_ys) = jax.lax.scan(
      _scan_step,
      zero_acc,
      dict(microbatches),
      length=num_micro,
  )

  grads_final = jax.tree.map(
      lambda p_var, acc_arr: p_var.replace(acc_arr.astype(p_var[...].dtype)),
      params,
      grad_acc_final,
      is_leaf=_is_param_var,
  )
  xent_total = jnp.sum(xent_ys)
  z_total = jnp.sum(z_ys)
  aux_total = jnp.sum(aux_ys)
  lm_loss = xent_total / total_valid_tokens
  total_loss = (xent_total + z_total) / total_valid_tokens + aux_total
  counts_total = None if counts_ys is None else jnp.sum(counts_ys, axis=0)

  return (
      LossBreakdown(
          total_loss=total_loss,
          lm_loss=lm_loss,
          xent_sum=xent_total,
          z_loss_sum=z_total,
          aux_loss=aux_total,
          valid_tokens=total_valid_tokens,
          router_token_counts=counts_total,
      ),
      grads_final,
  )


def apply_router_bias_update(
    model: nnx.Module,
    router_token_counts: jax.Array | None,
    update_rate: float,
) -> None:
  """Applies auxiliary-loss-free MoE `RouterBias` sign updates outside autodiff (`b += rate * sign(mean - counts)`)."""
  if router_token_counts is None or update_rate <= 0.0:
    return
  bias_state = nnx.state(model, RouterBias)
  flat_biases = nnx.to_flat_state(bias_state)
  if not flat_biases:
    return

  counts_f32 = jax.lax.stop_gradient(router_token_counts.astype(jnp.float32))
  rate = jnp.asarray(update_rate, dtype=jnp.float32)

  for idx, (_, bias_var) in enumerate(flat_biases):
    cur = bias_var[...]
    if counts_f32.shape == cur.shape:
      layer_counts = counts_f32
    elif counts_f32.ndim == cur.ndim + 1 and idx < counts_f32.shape[0]:
      layer_counts = counts_f32[idx]
    else:
      layer_counts = jnp.broadcast_to(counts_f32, cur.shape)
    mean_count = jnp.mean(layer_counts, axis=-1, keepdims=True)
    delta = rate * jnp.sign(mean_count - layer_counts)
    bias_var[...] = (cur.astype(jnp.float32) + delta).astype(cur.dtype)


def make_train_step(
    *,
    num_vocab_tiles: int = 1,
    z_loss_multiplier: float = 0.0,
    aux_loss_weight: float = 1.0,
    router_bias_update_rate: float = 0.0,
    grad_dtype: jnp.dtype = jnp.float32,
) -> Callable[[nnx.Module, nnx.Optimizer, M3Batch], LossBreakdown]:
  """Builds a jitted, buffer-donating `train_step` compatible with `nnx.cached_partial`."""

  @nnx.jit(donate_argnums=(0, 1))
  def _train_step(model: nnx.Module, optimizer: nnx.Optimizer, microbatches: M3Batch) -> LossBreakdown:
    breakdown, grads = accumulated_loss_and_grads(
        model,
        microbatches,
        num_vocab_tiles=num_vocab_tiles,
        z_loss_multiplier=z_loss_multiplier,
        aux_loss_weight=aux_loss_weight,
        grad_dtype=grad_dtype,
    )
    optimizer.update(model, grads)
    if router_bias_update_rate > 0.0:
      apply_router_bias_update(model, breakdown.router_token_counts, router_bias_update_rate)
    return breakdown

  return _train_step


def make_eval_step(
    *,
    num_vocab_tiles: int = 1,
    z_loss_multiplier: float = 0.0,
    aux_loss_weight: float = 1.0,
) -> Callable[[nnx.Module, M3Batch], LossBreakdown]:
  """Builds a jitted pure evaluation step compatible with `nnx.cached_partial`."""

  @nnx.jit
  def _eval_step(model: nnx.Module, batch: M3Batch) -> LossBreakdown:
    return compute_batch_loss_sums(
        model,
        batch,
        num_vocab_tiles=num_vocab_tiles,
        z_loss_multiplier=z_loss_multiplier,
        aux_loss_weight=aux_loss_weight,
    )

  return _eval_step
