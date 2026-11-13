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

"""FP32 sigmoid top-k MoE router and Switch load-balancing loss for Flint."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp

from maxtext.experimental.flint40b import flint_config
from maxtext.experimental.flint40b import flint_moe_kernels
from maxtext.experimental.flint40b import flint_types
from maxtext.experimental.flint40b import ops

_ROUTER_NORM_EPS: float = 1e-20
_ALL_AXES: tuple[str, str, str] = ("fsdp", "cp", "ep")


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _dense_vjp_top_k(
    scores: jax.Array,
    k: int,
    num_e: int,
    compact: bool,
) -> tuple[jax.Array, jax.Array]:
  """Runs `lax.top_k(scores, k)` with a dense scatter-free VPU VJP."""
  del num_e, compact
  vals, ids = jax.lax.top_k(scores, k)
  return vals, ids.astype(jnp.int32)


def _dense_vjp_top_k_fwd(
    scores: jax.Array,
    k: int,
    num_e: int,
    compact: bool,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
  del num_e
  vals, ids = jax.lax.top_k(scores, k)
  ids_i32 = flint_moe_kernels.checkpoint_router_map(ids.astype(jnp.int32), compact=compact)
  return (vals, ids_i32), ids_i32


def _dense_vjp_top_k_bwd(
    k: int,
    num_e: int,
    compact: bool,
    ids_i32: jax.Array,
    cotangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array]:
  del compact
  g_vals, g_ids = cotangents
  del g_ids
  ids_2d = ids_i32.reshape(-1, k)
  g_2d = g_vals.astype(jnp.float32).reshape(-1, k)
  n_rows = ids_2d.shape[0]
  ar_e = jax.lax.broadcasted_iota(jnp.int32, (n_rows, num_e), 1)
  dscores_2d = jnp.zeros((n_rows, num_e), dtype=jnp.float32)
  for slot in range(k):
    dscores_2d = dscores_2d + jnp.where(
        ar_e == ids_2d[:, slot : slot + 1],
        g_2d[:, slot : slot + 1],
        jnp.float32(0.0),
    )
  return (dscores_2d.reshape(*ids_i32.shape[:-1], num_e).astype(g_vals.dtype),)


_dense_vjp_top_k.defvjp(_dense_vjp_top_k_fwd, _dense_vjp_top_k_bwd)


def flint_router_forward(
    x_local: jax.Array,
    router_shard: flint_types.FlintRouterWeights[jax.Array],
    loss_mask_local: jax.Array | None,
    cfg: flint_config.FlintConfig,
    *,
    compute_aux_stats: bool = True,
) -> flint_types.FlintMoeAux:
  """Computes FP32 sigmoid top-k routing weights and local load-balancing stats.

  Executes inside `jax.shard_map(..., check_vma=False)` over
  `('fsdp', 'cp', 'ep')`. Both `x_local` and `router_shard.gate` are cast to
  `float32` prior to `x_f32 @ gate_f32` (executed at XLA default matmul
  precision on TPU when `compute_dtype == 'bfloat16'`, matching MaxText
  `float32_gate_logits=True`, or `Precision.HIGHEST` when
  `compute_dtype == 'float32'`), followed by exact `float32` sigmoid,
  `top_k(6)`, and sum-normalization.

  Args:
    x_local: Local token activations of shape `[..., hidden_size]`.
    router_shard: Local `FlintRouterWeights` shard (`gate` sharded on axis 0).
    loss_mask_local: Optional local validity mask of shape `x_local.shape[:-1]`
      where entries `> 0` mark valid tokens; if `None`, all local tokens
      contribute to `expert_counts` and `prob_sum`.
    cfg: Validated `FlintConfig`.
    compute_aux_stats: When `False`, returns zero `expert_counts` and `prob_sum`
      without materializing `[N_loc, E]` auxiliary probability/count reductions.

  Returns:
    `FlintMoeAux` with `expert_counts` (`f32[E]`, stop-gradient), `prob_sum`
    (`f32[E]`, differentiable), `topk_ids` (`i32[..., K]`), and `topk_weights`
    (`f32[..., K]`).

  Raises:
    ValueError: If `loss_mask_local` is provided and its shape does not match
      `x_local.shape[:-1]`.
  """
  num_experts = cfg.model.num_experts
  top_k = cfg.model.num_experts_per_tok
  prec = jax.lax.Precision.HIGHEST if cfg.train.compute_jnp_dtype == jnp.float32 else None

  if cfg.kernels.replicate_router_gate:
    gate_f32 = router_shard.gate.astype(jnp.float32)
  else:
    gate_f32 = ops.gather_dense_weight(
        router_shard.gate,
        cfg,
        sharded_axis=0,
        dtype=jnp.float32,
    )
  logits_f32 = jnp.dot(x_local.astype(jnp.float32), gate_f32, precision=prec)
  logits_f32 = jax.ad_checkpoint.checkpoint_name(logits_f32, "moe_router_pack")
  scores = jax.nn.sigmoid(logits_f32)
  compact_maps = cfg.kernels.compact_router_maps
  zero_scatter = cfg.kernels.moe_zero_scatter_router
  if cfg.kernels.use_pallas_router_topk:
    pallas_res = flint_moe_kernels.router_topk_fwd_tpu(scores, top_k)
    if pallas_res is None and flint_moe_kernels.is_tpu_device():
      raise ValueError(
          "use_pallas_router_topk=True failed TPU preconditions for scores"
          f" shape={scores.shape}, dtype={scores.dtype}, top_k={top_k}."
      )
  else:
    pallas_res = None
  if pallas_res is not None:
    vals, ids = pallas_res
    topk_ids_i32 = flint_moe_kernels.checkpoint_router_map(ids.astype(jnp.int32), compact=compact_maps)
    topk_w_raw = flint_moe_kernels.attach_topk_vjp(scores, vals, topk_ids_i32, num_experts)
    topk_w_raw = flint_moe_kernels.checkpoint_router_map(topk_w_raw, compact=compact_maps)
  elif zero_scatter:
    topk_w_raw, topk_ids_i32 = _dense_vjp_top_k(scores, top_k, num_experts, compact_maps)
    topk_w_raw = flint_moe_kernels.checkpoint_router_map(topk_w_raw, compact=compact_maps)
  else:
    topk_w_raw, topk_ids = jax.lax.top_k(scores, top_k)
    topk_ids_i32 = flint_moe_kernels.checkpoint_router_map(topk_ids.astype(jnp.int32), compact=compact_maps)
  topk_weights = (topk_w_raw / (jnp.sum(topk_w_raw, axis=-1, keepdims=True) + _ROUTER_NORM_EPS)) * float(
      cfg.model.routed_scaling_factor
  )
  topk_weights = flint_moe_kernels.checkpoint_router_map(topk_weights, compact=compact_maps)

  lead_shape = x_local.shape[:-1]
  num_local_tokens = math.prod(lead_shape)
  if loss_mask_local is not None and loss_mask_local.shape != lead_shape:
    raise ValueError(f"loss_mask_local shape {loss_mask_local.shape} must match" f" x_local.shape[:-1]={lead_shape}.")
  if not compute_aux_stats:
    zeros_e = jnp.zeros((num_experts,), dtype=jnp.float32)
    return flint_types.FlintMoeAux(
        expert_counts=zeros_e,
        prob_sum=zeros_e,
        topk_ids=topk_ids_i32,
        topk_weights=topk_weights,
    )
  if loss_mask_local is None:
    valid_f32 = jnp.ones((num_local_tokens, 1), dtype=jnp.float32)
  else:
    valid_f32 = (loss_mask_local.reshape(num_local_tokens, 1) > 0).astype(jnp.float32)

  scores_2d = scores.reshape(num_local_tokens, num_experts)
  norm_probs = scores_2d / (jnp.sum(scores_2d, axis=-1, keepdims=True) + _ROUTER_NORM_EPS)
  prob_sum = jnp.sum(norm_probs * valid_f32, axis=0)

  if zero_scatter:
    ids_2d = topk_ids_i32.reshape(num_local_tokens, top_k)
    ar_e = jax.lax.broadcasted_iota(jnp.int32, (num_local_tokens, num_experts), 1)
    counts_2d = jnp.zeros((num_local_tokens, num_experts), dtype=jnp.float32)
    for slot in range(top_k):
      counts_2d = counts_2d + (ar_e == ids_2d[:, slot : slot + 1]).astype(jnp.float32)
    expert_counts = jax.lax.stop_gradient(jnp.sum(counts_2d * valid_f32, axis=0))
  else:
    valid_per_assign = jnp.broadcast_to(valid_f32, (num_local_tokens, top_k)).reshape(-1)
    expert_counts = jax.lax.stop_gradient(
        jnp.bincount(
            topk_ids_i32.reshape(-1),
            weights=valid_per_assign,
            length=num_experts,
        ).astype(jnp.float32)
    )
  return flint_types.FlintMoeAux(
      expert_counts=expert_counts,
      prob_sum=prob_sum,
      topk_ids=topk_ids_i32,
      topk_weights=topk_weights,
  )


def compute_local_moe_aux_loss(
    expert_counts: jax.Array,
    prob_sum: jax.Array,
    global_valid_tokens: jax.Array,
    *,
    num_experts: int,
    top_k: int,
) -> jax.Array:
  """Computes the `check_vma=False`-safe local Switch load-balancing loss term.

  Differentiating this local scalar inside `shard_map(..., check_vma=False)`
  yields exact global gradients without `psum` `xN` scaling; summing the scalar
  across `('fsdp', 'cp', 'ep')` after `value_and_grad` yields the global
  load-balancing loss `E * sum_e (f_e * P_e)`.

  Args:
    expert_counts: Local selected-assignment counts (`f32[..., E]`).
    prob_sum: Local normalized router probability sum (`f32[..., E]`).
    global_valid_tokens: Global count of valid tokens across all mesh ranks.
    num_experts: Total number of routed experts `E`.
    top_k: Number of selected experts per token `K`.

  Returns:
    Scalar `float32` local auxiliary loss contribution.
  """
  stopped_valid = jax.lax.stop_gradient(global_valid_tokens.astype(jnp.float32))
  global_counts = jax.lax.stop_gradient(jax.lax.psum(expert_counts.astype(jnp.float32), _ALL_AXES))
  f_e = global_counts / jnp.maximum(stopped_valid * float(top_k), 1.0)
  local_p_e = prob_sum.astype(jnp.float32) / jnp.maximum(stopped_valid, 1.0)
  return float(num_experts) * jnp.sum(f_e * local_p_e)
