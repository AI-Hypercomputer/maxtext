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

"""Shared routed-expert execution, router replay, load-balancing stats, and overlap hooks.

Model modules own router score computation (softmax, sigmoid, group-limited
top-k, routing bias, shared experts) and pass selected `(weights, indices)` to
a frozen `MoEExecutor` resolved from `(config, sharding)` at construction:

    self.executor = moe.moe_from_config(config, self.sharding)
    ...
    out = self.executor(x, weights, indices, wi_0, wi_1, wo)
    # or for fused (E, D, 2*M) / (E, D, 2, M) gate_up projections:
    out = self.executor.fused(x, weights, indices, gate_up_proj, down_proj)

For RL router replay and teacher-forced routing verification, optional
`forced_routed_experts` (integer shape `[..., num_experts_per_tok]`, where
`UNSET_ROUTED_EXPERT = -1` marks a padded/inactive slot) can be passed to
`route_top_k`, `apply_forced_routed_experts`, `MoEExecutor`, `routed_experts`,
or `routed_experts_fused`.

Resolving `PartitionSpec` axes from `Sharding` at construction avoids reading
`jax.typeof(x).sharding` at trace time, which reports replicated `P(None, ...)`
on `Auto` mesh axes inside `jax.jit` and `ScannedLayers`. Because `MoEExecutor`
is a frozen dataclass of hashable primitives, Flax NNX treats it as a static
attribute across `nnx.split`, `nnx.merge`, `jax.jit`, and layer scan transforms.
"""

from collections.abc import Callable
import contextlib
import dataclasses
import functools
import math
from typing import Any, Literal

# pylint: disable=too-many-positional-arguments

import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from maxtext.m3.core import kernels
from maxtext.m3.core.sharding import DefaultSharding, MeshAxes, Sharding, TensorType

ExecutorKind = Literal["dense", "ragged_dot", "gmm_v2", "tokamax"]
SHARDED_EXECUTOR_KINDS: tuple[str, ...] = ("gmm_v2", "tokamax")
RouterScoring = Literal["softmax", "topk_softmax", "sigmoid"]

UNSET_ROUTED_EXPERT = -1

_COMBINE_DTYPE = jnp.float32
_TOKAMAX_M_ALIGN = 256
_TOKAMAX_KN_ALIGN = 128


def _align_up(value: int, multiple: int) -> int:
  """Rounds `value` up to a positive multiple of `multiple`."""
  return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


# ==============================================================================
# Router replay & forced_routed_experts helpers
# ==============================================================================


def valid_expert_mask(indices: jax.Array, num_experts: int) -> jax.Array:
  """Returns boolean mask where expert IDs are active (`0 <= id < num_experts`)."""
  return (indices >= 0) & (indices < int(num_experts))


def _validate_forced_routed_experts(
    forced_routed_experts: jax.Array,
    token_prefix: tuple[int, ...],
    top_k: int,
    num_experts: int,
) -> None:
  """Validates `forced_routed_experts` shape, integer dtype, and eager concrete ID range.

  Concrete (untraced) arrays with entries outside `[-1, num_experts)` raise
  `ValueError`; traced `jax.jit` arrays skip host value inspection and mask
  out-of-range entries to `-1` (`0.0` weight) via `valid_expert_mask`.
  """
  if num_experts < 1:
    raise ValueError(f"num_experts must be >= 1; got {num_experts}.")
  if top_k < 1:
    raise ValueError(f"top_k (num_experts_per_tok) must be >= 1; got {top_k}.")
  expected_shape = (*token_prefix, int(top_k))
  if forced_routed_experts.ndim != len(expected_shape) or tuple(forced_routed_experts.shape) != expected_shape:
    raise ValueError(
        f"forced_routed_experts shape {forced_routed_experts.shape} must match "
        f"(*token_prefix, num_experts_per_tok)={expected_shape}."
    )
  if jnp.issubdtype(forced_routed_experts.dtype, jnp.bool_) or not jnp.issubdtype(
      forced_routed_experts.dtype, jnp.integer
  ):
    raise ValueError(f"forced_routed_experts must have an integer dtype (not bool); got {forced_routed_experts.dtype}.")
  if hasattr(forced_routed_experts, "min"):
    try:
      if forced_routed_experts.size > 0:
        min_id = int(forced_routed_experts.min())
        max_id = int(forced_routed_experts.max())
        if min_id < UNSET_ROUTED_EXPERT or max_id >= int(num_experts):
          raise ValueError(
              f"forced_routed_experts entries must be in [-1, {num_experts}); got min={min_id}, max={max_id}."
          )
    except (jax.errors.ConcretizationTypeError, TypeError):
      pass


def apply_forced_routed_experts(
    scores: jax.Array,
    forced_routed_experts: jax.Array,
    *,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scores_are_logits: bool = False,
    eps: float = 0.0,
    dtype: Any | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Gathers and normalizes routing weights at `forced_routed_experts` indices.

  Args:
    scores: Full router probabilities or pre-softmax router logits of shape
      `[*token_prefix, num_experts]`.
    forced_routed_experts: Integer tensor of shape
      `[*token_prefix, num_experts_per_tok]` where `-1` (`UNSET_ROUTED_EXPERT`)
      denotes a padded/inactive expert slot.
    norm_topk_prob: Whether to normalize active slot weights by their sum.
    routed_scaling_factor: Multiplicative scaling factor applied after normalization.
    scores_are_logits: If True, applies masked `softmax` over gathered top-k
      logits (excluding `-1` padded slots from the softmax denominator).
    eps: Optional additive epsilon in the normalization denominator.
    dtype: Output dtype for `routing_weights` (defaults to `scores.dtype`).

  Returns:
    Tuple `(routing_weights, effective_indices)` of shape
    `[*token_prefix, num_experts_per_tok]`, where inactive slots have weight
    `0.0` and index `-1` (concrete out-of-range IDs outside `[-1, num_experts)`
    raise `ValueError`, whereas traced `jax.jit` arrays mask them to `-1` with
    `0.0` weight).
  """
  if scores.ndim < 2:
    raise ValueError(f"Expected scores with rank >= 2 ([..., num_experts]); got shape {scores.shape}.")
  num_experts = int(scores.shape[-1])
  token_prefix = tuple(scores.shape[:-1])
  if forced_routed_experts.ndim != scores.ndim:
    raise ValueError(
        f"forced_routed_experts rank ({forced_routed_experts.ndim}, shape={forced_routed_experts.shape}) "
        f"must match scores rank ({scores.ndim}, shape={scores.shape})."
    )
  top_k = int(forced_routed_experts.shape[-1])
  _validate_forced_routed_experts(forced_routed_experts, token_prefix, top_k, num_experts)

  forced_idx = forced_routed_experts.astype(jnp.int32)
  valid_mask = valid_expert_mask(forced_idx, num_experts)
  gather_idx = jnp.where(valid_mask, jnp.maximum(forced_idx, 0), 0)
  scores_f32 = scores.astype(jnp.float32)

  if scores_are_logits:
    gathered_logits = jnp.take_along_axis(scores_f32, gather_idx, axis=-1)
    neg_inf = jnp.asarray(jnp.finfo(jnp.float32).min / 2.0, dtype=jnp.float32)
    masked_logits = jnp.where(valid_mask, gathered_logits, neg_inf)
    weights = jax.nn.softmax(masked_logits, axis=-1)
    weights = jnp.where(valid_mask, weights, jnp.zeros_like(weights))
  else:
    gathered_probs = jnp.take_along_axis(scores_f32, gather_idx, axis=-1)
    weights = jnp.where(valid_mask, gathered_probs, jnp.zeros_like(gathered_probs))

  if norm_topk_prob:
    denom = jnp.sum(weights, axis=-1, keepdims=True)
    denom_with_eps = denom + jnp.asarray(eps, dtype=denom.dtype) if eps > 0.0 else denom
    safe_denom = jnp.where(denom > 0, denom_with_eps, jnp.ones_like(denom))
    weights = weights / safe_denom

  if routed_scaling_factor != 1.0:
    weights = weights * jnp.asarray(routed_scaling_factor, dtype=weights.dtype)

  out_dtype = scores.dtype if dtype is None else dtype
  eff_indices = jnp.where(valid_mask, forced_idx, UNSET_ROUTED_EXPERT)
  return weights.astype(out_dtype), eff_indices


def route_top_k(
    router_logits: jax.Array,
    top_k: int,
    *,
    scoring: RouterScoring = "softmax",
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: jax.Array | None = None,
    num_group: int = 1,
    topk_group: int = 1,
    eps: float = 0.0,
    forced_routed_experts: jax.Array | None = None,
    dtype: Any | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Computes `(routing_weights, router_indices, router_probs)` with optional router replay.

  Supports `"softmax"` (post-softmax top-k, e.g. Qwen3.8 / Gemma4),
  `"topk_softmax"` (pre-softmax top-k, e.g. GraniteMoE / Mixtral), and
  `"sigmoid"` (sigmoid with optional `e_score_correction_bias` and group-limited
  top-k, e.g. GLM-5 / DeepSeek-V4).
  """
  if router_logits.ndim < 2:
    raise ValueError(f"Expected router_logits with rank >= 2 ([..., num_experts]); got {router_logits.shape}.")
  num_experts = int(router_logits.shape[-1])
  if top_k < 1 or top_k > num_experts:
    raise ValueError(f"top_k ({top_k}) must be in [1, num_experts ({num_experts})].")
  if forced_routed_experts is not None:
    _validate_forced_routed_experts(
        forced_routed_experts,
        tuple(router_logits.shape[:-1]),
        top_k,
        num_experts,
    )

  out_dtype = router_logits.dtype if dtype is None else dtype
  logits_f32 = router_logits.astype(jnp.float32)

  if scoring == "softmax":
    router_probs = jax.nn.softmax(logits_f32, axis=-1)
    if forced_routed_experts is not None:
      weights, indices = apply_forced_routed_experts(
          router_probs,
          forced_routed_experts,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=routed_scaling_factor,
          scores_are_logits=False,
          eps=eps,
          dtype=out_dtype,
      )
      return weights, indices, router_probs
    top_vals, top_idx = jax.lax.top_k(router_probs, top_k)
  elif scoring == "topk_softmax":
    router_probs = jax.nn.softmax(logits_f32, axis=-1)
    if forced_routed_experts is not None:
      weights, indices = apply_forced_routed_experts(
          logits_f32,
          forced_routed_experts,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=routed_scaling_factor,
          scores_are_logits=True,
          eps=eps,
          dtype=out_dtype,
      )
      return weights, indices, router_probs
    top_logits, top_idx = jax.lax.top_k(logits_f32, top_k)
    top_vals = jax.nn.softmax(top_logits, axis=-1)
  elif scoring == "sigmoid":
    router_probs = jax.nn.sigmoid(logits_f32)
    if forced_routed_experts is not None:
      weights, indices = apply_forced_routed_experts(
          router_probs,
          forced_routed_experts,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=routed_scaling_factor,
          scores_are_logits=False,
          eps=eps,
          dtype=out_dtype,
      )
      return weights, indices, router_probs
    if e_score_correction_bias is not None:
      scores_for_choice = jax.lax.stop_gradient(router_probs + e_score_correction_bias.astype(jnp.float32))
    else:
      scores_for_choice = router_probs
    if num_group > 1:
      if num_experts % num_group != 0:
        raise ValueError(f"num_experts ({num_experts}) must be divisible by num_group ({num_group}).")
      if topk_group < 1 or topk_group > num_group:
        raise ValueError(f"topk_group ({topk_group}) must be in [1, num_group ({num_group})].")
      exp_per_group = num_experts // num_group
      prefix_shape = scores_for_choice.shape[:-1]
      grouped = jnp.reshape(scores_for_choice, (*prefix_shape, num_group, exp_per_group))
      group_k = min(2, exp_per_group) if e_score_correction_bias is not None else 1
      group_scores = jnp.sum(jax.lax.top_k(grouped, group_k)[0], axis=-1)
      _, group_idx = jax.lax.top_k(group_scores, topk_group)
      g_arange = jnp.arange(num_group, dtype=jnp.int32)
      bcast_shape = (1,) * len(prefix_shape) + (num_group, 1)
      group_mask = jnp.any(jnp.reshape(g_arange, bcast_shape) == group_idx[..., None, :], axis=-1)
      score_mask = jnp.repeat(group_mask, exp_per_group, axis=-1)
      masked_scores = jnp.where(score_mask, scores_for_choice, jnp.finfo(jnp.float32).min)
    else:
      masked_scores = scores_for_choice
    _, top_idx = jax.lax.top_k(masked_scores, top_k)
    top_vals = jnp.take_along_axis(router_probs, top_idx, axis=-1)
  else:
    raise ValueError(f"Unknown router scoring mode {scoring!r}; expected 'softmax', 'topk_softmax', or 'sigmoid'.")

  if norm_topk_prob:
    denom = jnp.sum(top_vals, axis=-1, keepdims=True)
    denom_with_eps = denom + jnp.asarray(eps, dtype=denom.dtype) if eps > 0.0 else denom
    safe_denom = jnp.where(denom > 0, denom_with_eps, jnp.ones_like(denom))
    top_vals = top_vals / safe_denom
  if routed_scaling_factor != 1.0:
    top_vals = top_vals * jnp.asarray(routed_scaling_factor, dtype=top_vals.dtype)
  return top_vals.astype(out_dtype), top_idx.astype(jnp.int32), router_probs


# ==============================================================================
# Custom-VJP scheduling barriers & backward wgrad overlap hook
# ==============================================================================


@jax.custom_vjp
def forward_optimization_barrier(operands: Any) -> Any:
  """Applies `jax.lax.optimization_barrier` in forward; identity in backward."""
  return jax.lax.optimization_barrier(operands)


def _forward_barrier_fwd(operands: Any) -> tuple[Any, None]:
  return jax.lax.optimization_barrier(operands), None


def _forward_barrier_bwd(_: None, cotangents: Any) -> tuple[Any]:
  return (cotangents,)


forward_optimization_barrier.defvjp(_forward_barrier_fwd, _forward_barrier_bwd)


@jax.custom_vjp
def backward_optimization_barrier(operands: Any) -> Any:
  """Identity in forward; applies `jax.lax.optimization_barrier` to cotangents."""
  return operands


def _backward_barrier_fwd(operands: Any) -> tuple[Any, None]:
  return operands, None


def _backward_barrier_bwd(_: None, cotangents: Any) -> tuple[Any]:
  return (jax.lax.optimization_barrier(cotangents),)


backward_optimization_barrier.defvjp(_backward_barrier_fwd, _backward_barrier_bwd)


@jax.custom_vjp
def backward_wgrad_tie(
    x: jax.Array,
    weights: tuple[jax.Array, ...],
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
  """Ties token input cotangents to expert weight cotangents in backward.

  Fencing `dx` with local expert weight gradients (`dw`) inside `shard_map`
  forces expert TGMM kernels to run immediately after `dx` grouped matmuls, so
  FSDP weight-gradient reduce-scatters overlap with the preceding sub-layer's
  attention or linear-attention backward pass instead of queuing at block end.
  """
  return x, weights


def _backward_wgrad_tie_fwd(
    x: jax.Array,
    weights: tuple[jax.Array, ...],
) -> tuple[tuple[jax.Array, tuple[jax.Array, ...]], None]:
  return (x, weights), None


def _backward_wgrad_tie_bwd(
    _: None,
    cotangents: tuple[jax.Array, tuple[jax.Array, ...]],
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
  dx, dweights = jax.lax.optimization_barrier(cotangents)
  return dx, dweights


backward_wgrad_tie.defvjp(_backward_wgrad_tie_fwd, _backward_wgrad_tie_bwd)


# ==============================================================================
# Permutation & grouped-matmul primitives
# ==============================================================================


@jax.custom_vjp
def _permute_rows(inputs: jax.Array, perm: jax.Array) -> jax.Array:
  """Permutes axis 0 by bijection `perm` with gather-only inverse VJP."""
  return inputs[perm]


def _permute_rows_fwd(inputs: jax.Array, perm: jax.Array) -> tuple[jax.Array, jax.Array]:
  return inputs[perm], perm


def _permute_rows_bwd(perm: jax.Array, grad: jax.Array) -> tuple[jax.Array, None]:
  return grad[jnp.argsort(perm)], None


_permute_rows.defvjp(_permute_rows_fwd, _permute_rows_bwd)


def _matmul_precision(dtype: Any) -> jax.lax.Precision:
  """Returns single-pass MXU `DEFAULT` for `bfloat16` and `HIGHEST` otherwise."""
  if jnp.dtype(dtype) == jnp.dtype(jnp.bfloat16):
    return jax.lax.Precision.DEFAULT
  return jax.lax.Precision.HIGHEST


def _pallas_matmul_precision_context(dtype: Any) -> contextlib.AbstractContextManager[Any]:
  """Enables `float32` MXU passes inside Pallas `gmm_v2`/`tgmm_v2` on `float32` inputs."""
  if jnp.dtype(dtype) == jnp.dtype(jnp.float32):
    return jax.default_matmul_precision("float32")
  return contextlib.nullcontext()


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _pallas_grouped_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    backend_kind: str,
    is_tpu: bool,
    out_dtype: Any,
) -> jax.Array:
  """Runs grouped matmul `lhs @ rhs[g]` on the TPU `gmm_v2` backend."""
  group_sizes_i32 = group_sizes.astype(jnp.int32)
  valid_count = jnp.sum(group_sizes_i32)
  valid_rows = (jnp.arange(lhs.shape[0], dtype=jnp.int32) < valid_count)[:, None]
  lhs_safe = jnp.where(valid_rows, lhs, jnp.zeros((), dtype=lhs.dtype))

  if backend_kind in ("gmm_v2", "tokamax"):
    if not is_tpu:
      out = jax.lax.ragged_dot(
          lhs_safe,
          rhs,
          group_sizes_i32,
          precision=_matmul_precision(lhs.dtype),
          preferred_element_type=jnp.float32,
      ).astype(out_dtype)
      return jnp.where(valid_rows, out, jnp.zeros((), dtype=out_dtype))

    from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2  # pylint: disable=import-outside-toplevel

    m, k = lhs.shape
    _, _, n = rhs.shape
    m_pad = _align_up(m, _TOKAMAX_M_ALIGN)
    k_pad = _align_up(k, _TOKAMAX_KN_ALIGN)
    n_pad = _align_up(n, _TOKAMAX_KN_ALIGN)
    lhs_pad = jnp.pad(lhs_safe, ((0, m_pad - m), (0, k_pad - k)))
    rhs_pad = jnp.pad(rhs, ((0, 0), (0, k_pad - k), (0, n_pad - n)))
    with _pallas_matmul_precision_context(lhs.dtype):
      out_pad = gmm_v2.gmm_v2(
          lhs=lhs_pad,
          rhs=rhs_pad,
          group_sizes=group_sizes_i32,
          rhs_scale=None,
          tile_info=gmm_v2.calculate_tiling,
          preferred_element_type=out_dtype,
          maybe_quantize_lhs=False,
      )
    out = out_pad[:m, :n].astype(out_dtype)
    return jnp.where(valid_rows, out, jnp.zeros((), dtype=out_dtype))

  raise ValueError(f"Unsupported Pallas grouped matmul backend {backend_kind!r}.")


def _pallas_grouped_matmul_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    backend_kind: str,
    is_tpu: bool,
    out_dtype: Any,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
  out = _pallas_grouped_matmul(lhs, rhs, group_sizes, backend_kind, is_tpu, out_dtype)
  return out, (lhs, rhs, group_sizes.astype(jnp.int32))


def _pallas_grouped_matmul_bwd(
    backend_kind: str,
    is_tpu: bool,
    out_dtype: Any,
    residuals: tuple[jax.Array, jax.Array, jax.Array],
    grad: jax.Array,
) -> tuple[jax.Array, jax.Array, None]:
  """Backward pass for the TPU `gmm_v2` / `tgmm_v2` grouped matmuls."""
  lhs, rhs, group_sizes = residuals
  m, k = lhs.shape
  num_groups, _, n = rhs.shape
  valid_count = jnp.sum(group_sizes)
  valid_rows = (jnp.arange(m, dtype=jnp.int32) < valid_count)[:, None]
  lhs_safe = jnp.where(valid_rows, lhs, jnp.zeros((), dtype=lhs.dtype))

  if backend_kind in ("gmm_v2", "tokamax"):
    if not is_tpu:
      prec = _matmul_precision(lhs.dtype)
      _, vjp_fn = jax.vjp(
          lambda l, r: jax.lax.ragged_dot(
              l,
              r,
              group_sizes,
              precision=prec,
              preferred_element_type=jnp.float32,
          ).astype(out_dtype),
          lhs_safe,
          rhs,
      )
      grad_safe = jnp.where(valid_rows, grad.astype(out_dtype), jnp.zeros((), dtype=out_dtype))
      dlhs, drhs = vjp_fn(grad_safe)
      dlhs = jnp.where(valid_rows, dlhs, jnp.zeros((), dtype=lhs.dtype))
      return dlhs, drhs, None

    from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_gmm_kernel as gmm_v2  # pylint: disable=import-outside-toplevel
    from maxtext.kernels.megablox import pallas_mosaic_tpu_v2_tgmm_kernel as tgmm_v2  # pylint: disable=import-outside-toplevel

    m_pad = _align_up(m, _TOKAMAX_M_ALIGN)
    k_pad = _align_up(k, _TOKAMAX_KN_ALIGN)
    n_pad = _align_up(n, _TOKAMAX_KN_ALIGN)
    lhs_pad = jnp.pad(lhs_safe, ((0, m_pad - m), (0, k_pad - k)))
    rhs_pad = jnp.pad(rhs, ((0, 0), (0, k_pad - k), (0, n_pad - n)))
    grad_safe = jnp.where(valid_rows, grad.astype(lhs.dtype), jnp.zeros((), dtype=lhs.dtype))
    grad_pad = jnp.pad(grad_safe, ((0, m_pad - m), (0, n_pad - n)))
    kernel_transpose_rhs = jax.dtypes.itemsize_bits(rhs_pad.dtype) >= 8
    dlhs_rhs = rhs_pad if kernel_transpose_rhs else rhs_pad.swapaxes(1, 2)
    with _pallas_matmul_precision_context(lhs.dtype):
      dlhs_pad = gmm_v2.gmm_v2(
          lhs=grad_pad,
          rhs=dlhs_rhs,
          group_sizes=group_sizes,
          rhs_scale=None,
          tile_info=gmm_v2.calculate_tiling,
          preferred_element_type=lhs_pad.dtype,
          maybe_quantize_lhs=False,
          transpose_rhs=kernel_transpose_rhs,
      )
      drhs_pad = tgmm_v2.tgmm_v2(
          lhs=lhs_pad,
          rhs=grad_pad,
          group_sizes=group_sizes,
          num_actual_groups=num_groups,
          rhs_scale=None,
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=rhs_pad.dtype,
          tile_info=tgmm_v2.calculate_tgmm_tiling,
      )
    dlhs = jnp.where(valid_rows, dlhs_pad[:m, :k].astype(lhs.dtype), jnp.zeros((), dtype=lhs.dtype))
    return dlhs, drhs_pad[:, :k, :n].astype(rhs.dtype), None

  raise ValueError(f"Unsupported Pallas grouped matmul backend {backend_kind!r}.")


_pallas_grouped_matmul.defvjp(_pallas_grouped_matmul_fwd, _pallas_grouped_matmul_bwd)


def grouped_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    backend: str = "ragged_dot",
    is_tpu: bool = False,
    preferred_element_type: Any = None,
) -> jax.Array:
  """Grouped matmul dispatcher across `ragged_dot`, `gmm_v2`, and `tokamax`."""
  if backend not in ("ragged_dot", "gmm_v2", "tokamax"):
    raise ValueError(f"Unknown grouped matmul backend {backend!r}; expected 'ragged_dot', 'gmm_v2', or 'tokamax'.")

  out_dtype = jnp.dtype(lhs.dtype if preferred_element_type is None else preferred_element_type)
  group_sizes_i32 = group_sizes.astype(jnp.int32)
  valid_rows = (jnp.arange(lhs.shape[0], dtype=jnp.int32) < jnp.sum(group_sizes_i32))[:, None]
  lhs_safe = jnp.where(valid_rows, lhs, jnp.zeros((), dtype=lhs.dtype))

  if backend == "ragged_dot":
    out = jax.lax.ragged_dot(
        lhs_safe,
        rhs,
        group_sizes_i32,
        precision=_matmul_precision(lhs.dtype),
        preferred_element_type=jnp.float32,
    ).astype(out_dtype)
    return jnp.where(valid_rows, out, jnp.zeros((), dtype=out_dtype))

  if backend == "tokamax" and is_tpu and jax.default_backend() == "tpu":
    import tokamax  # pylint: disable=import-outside-toplevel

    m, k = lhs.shape
    _, _, n = rhs.shape
    m_pad = _align_up(m, _TOKAMAX_M_ALIGN)
    k_pad = _align_up(k, _TOKAMAX_KN_ALIGN)
    n_pad = _align_up(n, _TOKAMAX_KN_ALIGN)
    lhs_pad = jnp.pad(lhs_safe, ((0, m_pad - m), (0, k_pad - k)))
    rhs_pad = jnp.pad(rhs, ((0, 0), (0, k_pad - k), (0, n_pad - n)))
    out_pad = tokamax.ragged_dot(
        lhs=lhs_pad,
        rhs=rhs_pad,
        group_sizes=tokamax.RaggedDotGroupSizes(group_sizes_i32, m_pad),
        preferred_element_type=out_dtype,
        precision=_matmul_precision(lhs.dtype),
        implementation="mosaic",
    )
    out = out_pad[:m, :n].astype(out_dtype)
    return jnp.where(valid_rows, out, jnp.zeros((), dtype=out_dtype))

  return _pallas_grouped_matmul(lhs, rhs, group_sizes_i32, backend, is_tpu, out_dtype)


# ==============================================================================
# Executor descriptor & config resolution
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class MoEExecutor:
  """Static routed-expert execution descriptor resolved from `(config, sharding)`."""

  kind: ExecutorKind = "dense"
  batch: MeshAxes = None
  experts: MeshAxes = None
  mlp: MeshAxes = None
  capacity_factor: float = -1.0
  ragged_buffer_factor: float = -1.0
  dropless_buffer_factor: float = -1.0
  use_ragged_sort: bool = False
  bwd_wgrad_overlap: bool = False
  length: MeshAxes = None

  def __post_init__(self) -> None:
    if (
        isinstance(self.dropless_buffer_factor, bool)
        or not math.isfinite(self.dropless_buffer_factor)
        or (self.dropless_buffer_factor < 0.0 and self.dropless_buffer_factor != -1.0)
    ):
      raise ValueError(
          "dropless_buffer_factor must be -1.0 (auto: 2.0 when ep_size >= 8, unbounded when ep_size < 8), "
          f"0.0 (unbounded), or > 0.0; got {self.dropless_buffer_factor!r}."
      )
    if self.ragged_buffer_factor > 0.0 and self.dropless_buffer_factor > 0.0:
      raise ValueError("ragged_buffer_factor > 0 and dropless_buffer_factor > 0 cannot both be set.")
    if self.capacity_factor > 0.0 and self.dropless_buffer_factor > 0.0:
      raise ValueError("capacity_factor > 0 and dropless_buffer_factor > 0 cannot both be set.")

  def __call__(
      self,
      x: jax.Array,
      weights: jax.Array,
      indices: jax.Array | None,
      wi_0: jax.Array,
      wi_1: jax.Array,
      wo: jax.Array,
      forced_routed_experts: jax.Array | None = None,
      *,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
      swiglu_limit: float | None = None,
      router_probs: jax.Array | None = None,
      norm_topk_prob: bool = False,
      routed_scaling_factor: float = 1.0,
      scores_are_logits: bool = False,
      eps: float = 0.0,
  ) -> jax.Array:
    """Executes routed experts with separate `(wi_0, wi_1, wo)` projections."""
    return routed_experts(
        x,
        weights,
        indices,
        wi_0,
        wi_1,
        wo,
        forced_routed_experts=forced_routed_experts,
        executor=self,
        activation=activation,
        swiglu_limit=swiglu_limit,
        router_probs=router_probs,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        scores_are_logits=scores_are_logits,
        eps=eps,
    )

  def fused(
      self,
      x: jax.Array,
      weights: jax.Array,
      indices: jax.Array | None,
      gate_up_proj: jax.Array,
      down_proj: jax.Array,
      forced_routed_experts: jax.Array | None = None,
      *,
      activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
      swiglu_limit: float | None = None,
      router_probs: jax.Array | None = None,
      norm_topk_prob: bool = False,
      routed_scaling_factor: float = 1.0,
      scores_are_logits: bool = False,
      eps: float = 0.0,
  ) -> jax.Array:
    """Executes routed experts with 3D `(E, D, 2*M)` or 4D `(E, D, 2, M)` `gate_up_proj`."""
    return routed_experts_fused(
        x,
        weights,
        indices,
        gate_up_proj,
        down_proj,
        forced_routed_experts=forced_routed_experts,
        executor=self,
        activation=activation,
        swiglu_limit=swiglu_limit,
        router_probs=router_probs,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        scores_are_logits=scores_are_logits,
        eps=eps,
    )


Executor = MoEExecutor

DENSE = MoEExecutor("dense")
RAGGED_DOT = MoEExecutor("ragged_dot")


def _parallelism_product(config: Any, *attrs: str) -> int:
  """Returns the product of positive (> 1) parallelism attributes on `config`."""
  size = 1
  for attr in attrs:
    val = int(getattr(config, attr, 1) or 1)
    if val > 1:
      size *= val
  return size


def _use_sharded_gmm_v2(config: Any) -> bool:
  """Returns whether `config` selects the sharded `gmm_v2` (`shard_map`) executor."""
  if not bool(getattr(config, "sparse_matmul", True)):
    return False
  megablox = getattr(config, "megablox", None)
  if megablox is not None and bool(megablox):
    return True
  # MaxTextConfig coerces `megablox=False` when `hardware="cpu"`, while
  # `MoEExecutor(kind="gmm_v2")` executes on CPU via `jax.lax.ragged_dot` inside `shard_map`.
  if getattr(config, "hardware", "tpu") == "cpu":
    return True
  if megablox is not None:
    return False
  return getattr(config, "use_tokamax_gmm", None) is None


def _require_sharded_gmm_for_ep_or_tp(config: Any) -> None:
  """Rejects unsharded `ragged_dot` when EP or TP parallelism exceeds 1."""
  ep_cfg = _parallelism_product(config, "ici_expert_parallelism", "dcn_expert_parallelism")
  tp_cfg = _parallelism_product(
      config,
      "ici_tensor_parallelism",
      "dcn_tensor_parallelism",
      "ici_tensor_sequence_parallelism",
      "dcn_tensor_sequence_parallelism",
  )
  if ep_cfg > 1 or tp_cfg > 1:
    raise ValueError(
        "Expert parallelism (expert > 1) or tensor parallelism (tensor > 1) with sparse_matmul=True requires a sharded "
        "GMM backend (gmm_v2 or tokamax) so tokens dispatch via shard_map without all-gathering expert weights."
    )


def resolve_dropless_buffer_factor(
    dropless_buffer_factor: float,
    ep_size: int,
    *,
    capacity_factor: float = -1.0,
    ragged_buffer_factor: float = -1.0,
) -> float:
  """Resolves effective dropless receive-buffer factor (`0.0` when unbounded/disabled)."""
  if (
      isinstance(dropless_buffer_factor, bool)
      or not math.isfinite(dropless_buffer_factor)
      or (dropless_buffer_factor < 0.0 and dropless_buffer_factor != -1.0)
  ):
    raise ValueError(
        "dropless_buffer_factor must be -1.0 (auto: 2.0 when ep_size >= 8, unbounded when ep_size < 8), "
        f"0.0 (unbounded), or > 0.0; got {dropless_buffer_factor!r}."
    )
  if dropless_buffer_factor == 0.0 or ep_size <= 1:
    return 0.0
  if dropless_buffer_factor == -1.0:
    if ep_size >= 8 and capacity_factor <= 0.0 and ragged_buffer_factor <= 0.0:
      return 2.0
    return 0.0
  if dropless_buffer_factor >= float(ep_size):
    return 0.0
  return float(dropless_buffer_factor)


def _extract_dropless_buffer_factor(config: Any) -> float:
  """Extracts and validates `moe_dropless_buffer_factor` from `config`."""
  raw_dbf = getattr(config, "moe_dropless_buffer_factor", -1.0)
  if isinstance(raw_dbf, bool):
    raise ValueError(f"moe_dropless_buffer_factor must be a float, not bool; got {raw_dbf!r}.")
  dbf = -1.0 if raw_dbf is None else float(raw_dbf)
  if not math.isfinite(dbf) or (dbf < 0.0 and dbf != -1.0):
    raise ValueError(
        "moe_dropless_buffer_factor must be -1.0 (auto: 2.0 when ep_size >= 8, unbounded when ep_size < 8), "
        f"0.0 (single-round unbounded), or > 0.0; got {raw_dbf!r}."
    )
  return dbf


def validate_moe_config(config: Any) -> None:
  """Validates m3 MoE configuration fields, failing closed on unsupported modes."""
  num_experts = int(getattr(config, "num_experts", 1) or 1)
  sparse_matmul = bool(getattr(config, "sparse_matmul", True))
  use_tokamax_gmm = bool(getattr(config, "use_tokamax_gmm", False))
  use_megablox = _use_sharded_gmm_v2(config)
  capacity_factor = float(getattr(config, "capacity_factor", -1.0))
  ragged_buffer_factor = float(getattr(config, "ragged_buffer_factor", -1.0))
  dropless_buffer_factor = _extract_dropless_buffer_factor(config)
  use_ragged_sort = bool(getattr(config, "use_ragged_sort", False))
  use_ring_of_experts = bool(getattr(config, "use_ring_of_experts", False))
  bwd_wgrad_overlap = bool(getattr(config, "moe_bwd_wgrad_overlap", False))

  if capacity_factor == 0.0:
    raise ValueError("capacity_factor cannot be 0.0; set < 0 for dropless routing or > 0 for capacity dropping.")
  if ragged_buffer_factor == 0.0:
    raise ValueError("ragged_buffer_factor cannot be 0.0; set < 0 for worst-case buffer or > 0 to bound buffer size.")
  if not math.isfinite(dropless_buffer_factor) or (dropless_buffer_factor < 0.0 and dropless_buffer_factor != -1.0):
    raise ValueError(
        "moe_dropless_buffer_factor must be -1.0 (auto: 2.0 when ep_size >= 8, unbounded when ep_size < 8), "
        f"0.0 (single-round unbounded), or > 0.0; got {dropless_buffer_factor}."
    )
  if ragged_buffer_factor > 0.0 and dropless_buffer_factor > 0.0:
    raise ValueError("ragged_buffer_factor > 0 and moe_dropless_buffer_factor > 0 cannot both be set.")
  if capacity_factor > 0.0 and dropless_buffer_factor > 0.0:
    raise ValueError("capacity_factor > 0 and moe_dropless_buffer_factor > 0 cannot both be set.")

  if num_experts <= 1 and not bwd_wgrad_overlap and not use_ragged_sort and not use_ring_of_experts:
    return

  if use_ring_of_experts:
    raise NotImplementedError(
        "m3 MoE uses ragged_all_to_all expert parallelism; use_ring_of_experts=True is not supported."
    )
  if use_ragged_sort and (not sparse_matmul or (not use_tokamax_gmm and not use_megablox)):
    raise ValueError("use_ragged_sort=True requires sparse_matmul=True with a sharded GMM backend (gmm_v2 or tokamax).")
  if bwd_wgrad_overlap and not sparse_matmul:
    raise ValueError("moe_bwd_wgrad_overlap=True requires sparse_matmul=True.")
  if sparse_matmul and not use_tokamax_gmm and not use_megablox:
    _require_sharded_gmm_for_ep_or_tp(config)


def moe_from_config(config: Any, sharding: Sharding | None = None) -> MoEExecutor:
  """Resolves a static `MoEExecutor` from `config` and `sharding`."""
  validate_moe_config(config)
  sharding = sharding or DefaultSharding()

  sparse_matmul = bool(getattr(config, "sparse_matmul", True))
  use_tokamax_gmm = bool(getattr(config, "use_tokamax_gmm", False))
  use_megablox = _use_sharded_gmm_v2(config)
  capacity_factor = float(getattr(config, "capacity_factor", -1.0))
  ragged_buffer_factor = float(getattr(config, "ragged_buffer_factor", -1.0))
  dbf = _extract_dropless_buffer_factor(config)
  use_ragged_sort = bool(getattr(config, "use_ragged_sort", False))
  bwd_wgrad_overlap = bool(getattr(config, "moe_bwd_wgrad_overlap", False))

  cap = capacity_factor if capacity_factor > 0.0 else -1.0
  rbf = ragged_buffer_factor if ragged_buffer_factor > 0.0 else -1.0

  if not sparse_matmul:
    return MoEExecutor(
        kind="dense",
        capacity_factor=cap,
        ragged_buffer_factor=rbf,
        dropless_buffer_factor=dbf,
    )

  if use_tokamax_gmm:
    kind: ExecutorKind = "tokamax"
  elif use_megablox:
    kind = "gmm_v2"
  else:
    _require_sharded_gmm_for_ep_or_tp(config)
    return MoEExecutor(
        kind="ragged_dot",
        capacity_factor=cap,
        ragged_buffer_factor=rbf,
        dropless_buffer_factor=dbf,
        bwd_wgrad_overlap=bwd_wgrad_overlap,
    )

  tokens_spec = sharding("moe_tokens", ("batch", "length", "embed"), TensorType.ACTIVATION)
  wi_spec = sharding("wi_0", ("experts", "expert_embed", "expert_mlp"), TensorType.WEIGHT)
  return MoEExecutor(
      kind=kind,
      batch=tokens_spec[0],
      experts=wi_spec[0],
      mlp=wi_spec[2],
      capacity_factor=cap,
      ragged_buffer_factor=rbf,
      dropless_buffer_factor=dbf,
      use_ragged_sort=use_ragged_sort,
      bwd_wgrad_overlap=bwd_wgrad_overlap,
      length=tokens_spec[1],
  )


executor_from_config = moe_from_config


def compute_ep_overflow_stats(
    indices: jax.Array,
    num_experts: int,
    ep_size: int,
    dropless_buffer_factor: float = -1.0,
    *,
    num_ep_groups: int = 1,
    dtype: jax.typing.DTypeLike = jnp.float32,
) -> dict[str, Any]:
  """Computes per-EP-rank receive counts and overflow-round trigger statistics.

  Args:
    indices: Routing expert indices as 3D `(B, S, K)` partitioned across
      `num_ep_groups * ep_size` token shards (a single EP group slice when
      `num_ep_groups == 1`), 4D `(ep_size, B_loc, S, K)` for one EP group, or
      5D `(num_ep_groups, ep_size, B_loc, S, K)`.
    num_experts: Total number of routed experts (`E`).
    ep_size: Expert-parallel group size (`EP`).
    dropless_buffer_factor: Configured `moe_dropless_buffer_factor` (`-1.0` for auto).
    num_ep_groups: Number of independent EP groups across `data * fsdp` (`>= 1`).
  """
  if isinstance(num_ep_groups, bool) or num_ep_groups < 1:
    raise ValueError(f"num_ep_groups must be >= 1; got {num_ep_groups!r}.")
  if ep_size < 1 or num_experts % ep_size != 0:
    raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}).")
  eff_dbf = resolve_dropless_buffer_factor(dropless_buffer_factor, ep_size)
  e_loc = num_experts // ep_size
  idx = jnp.asarray(indices, dtype=jnp.int32)
  if idx.ndim == 3:
    total_shards = num_ep_groups * ep_size
    if idx.shape[0] % total_shards != 0:
      raise ValueError(
          f"Batch ({idx.shape[0]}) must be divisible by num_ep_groups * ep_size "
          f"({num_ep_groups} * {ep_size} = {total_shards})."
      )
    idx = jnp.reshape(idx, (num_ep_groups, ep_size, idx.shape[0] // total_shards, idx.shape[1], idx.shape[2]))
  elif idx.ndim == 4 and idx.shape[0] == ep_size:
    if num_ep_groups != 1:
      raise ValueError(f"4D (ep_size, B_loc, S, K) indices represent 1 EP group; got num_ep_groups={num_ep_groups}.")
    idx = idx[None, ...]
  elif idx.ndim == 5 and idx.shape[1] == ep_size:
    if num_ep_groups not in (1, idx.shape[0]):
      raise ValueError(f"5D indices leading axis ({idx.shape[0]}) must match num_ep_groups ({num_ep_groups}).")
  else:
    raise ValueError(
        f"Expected 3D (B, S, K), 4D (ep_size, B_loc, S, K), or 5D (G, ep_size, B_loc, S, K) indices; got {idx.shape}."
    )
  num_slots = int(idx.shape[2] * idx.shape[3] * idx.shape[4])
  row_align = max(1, 32 // jax.dtypes.itemsize_bits(jnp.dtype(dtype)))
  if eff_dbf > 0.0:
    dbf_cap = max(1, int(math.ceil(num_slots * eff_dbf)))
    dbf_cap = ((dbf_cap + row_align - 1) // row_align) * row_align
    recv_buf_len = min(ep_size * num_slots, dbf_cap)
  else:
    recv_buf_len = ep_size * num_slots

  def _group_dst_counts(group_idx: jax.Array) -> jax.Array:
    flat = jnp.reshape(group_idx, (-1,))
    valid = (flat >= 0) & (flat < num_experts)
    safe = jnp.where(valid, flat, 0)
    exp_counts = jnp.bincount(safe, weights=valid.astype(jnp.int32), length=num_experts)
    return jnp.sum(jnp.reshape(exp_counts, (ep_size, e_loc)), axis=1)

  group_dst = jax.vmap(_group_dst_counts)(idx)
  dst_counts = group_dst[0] if group_dst.shape[0] == 1 else group_dst
  max_dst = int(jnp.max(group_dst))
  overflow_rounds = max(0, int(math.ceil(max_dst / recv_buf_len)) - 1) if max_dst > 0 else 0
  return {
      "effective_buffer_factor": eff_dbf,
      "num_slots_per_rank": num_slots,
      "recv_buf_len": recv_buf_len,
      "dst_counts": dst_counts,
      "max_dst_tokens": max_dst,
      "max_load_factor": float(max_dst) / float(max(num_slots, 1)),
      "overflow_triggered": bool(max_dst > recv_buf_len),
      "overflow_rounds": overflow_rounds,
      "tokens_dropped": 0,
  }


# ==============================================================================
# Load-balancing loss & dense combine helpers
# ==============================================================================


def combine_weights(weights: jax.Array, indices: jax.Array, num_experts: int) -> jax.Array:
  """Projects per-token top-k weights into a dense `(batch, length, num_experts)` table."""
  valid = (indices >= 0) & (indices < num_experts)
  safe_indices = jnp.where(valid, indices, 0)
  out_dtype = jnp.promote_types(weights.dtype, _COMBINE_DTYPE)
  safe_weights = jnp.where(valid, weights.astype(out_dtype), jnp.zeros((), dtype=out_dtype))
  return jnp.sum(jax.nn.one_hot(safe_indices, num_experts, dtype=out_dtype) * safe_weights[..., None], axis=-2)


def load_balance_loss(
    indices: jax.Array,
    probs: jax.Array,
    *,
    weight: float = 1.0,
    mask: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
) -> jax.Array:
  """Switch-Transformer load-balance loss masking padding and binarizing `segment_ids`."""
  if probs.ndim != 3 or indices.ndim != 3 or probs.shape[:2] != indices.shape[:2]:
    raise ValueError(
        f"Expected 3D indices (B, S, K) and probs (B, S, E) with matching (B, S); "
        f"got indices={indices.shape}, probs={probs.shape}."
    )
  batch, length, num_experts = probs.shape
  top_k = indices.shape[-1]
  probs_f32 = probs.astype(jnp.float32)

  valid_slots = (indices >= 0) & (indices < num_experts)
  safe_indices = jnp.where(valid_slots, indices, 0)
  expert_mask = jax.nn.one_hot(safe_indices, num_experts, dtype=jnp.float32) * valid_slots[..., None].astype(jnp.float32)

  token_valid: jax.Array | None = None
  if mask is not None:
    if mask.shape != (batch, length):
      raise ValueError(f"mask shape {mask.shape} must match (batch, length)=({batch}, {length}).")
    token_valid = mask != 0
  if segment_ids is not None:
    if segment_ids.shape != (batch, length):
      raise ValueError(f"segment_ids shape {segment_ids.shape} must match (batch, length)=({batch}, {length}).")
    seg_valid = segment_ids != 0
    token_valid = seg_valid if token_valid is None else (token_valid & seg_valid)

  if token_valid is None:
    density = jnp.mean(jnp.sum(expert_mask, axis=2), axis=1) / top_k
    density_prob = jnp.mean(probs_f32, axis=1)
    return (jnp.mean(density * density_prob) * (num_experts**2) * float(weight)).astype(jnp.float32)

  m = token_valid.astype(jnp.float32)
  seq_counts = jnp.sum(m, axis=1, keepdims=True)
  safe_counts = jnp.maximum(seq_counts, 1.0)
  seq_active = (seq_counts[:, 0] > 0).astype(jnp.float32)

  masked_expert_mask = expert_mask * m[:, :, None, None]
  density = jnp.sum(masked_expert_mask, axis=(1, 2)) / (safe_counts * top_k)
  density_prob = jnp.sum(probs_f32 * m[:, :, None], axis=1) / safe_counts
  per_seq = jnp.mean(density * density_prob, axis=-1)
  active_denom = jnp.maximum(jnp.sum(seq_active), 1.0)
  return ((jnp.sum(per_seq * seq_active) / active_denom) * (num_experts**2) * float(weight)).astype(jnp.float32)


# ==============================================================================
# Expert MLP & EP dispatch/combine execution
# ==============================================================================


def _apply_gated_activation(
    gate: jax.Array,
    up: jax.Array,
    *,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    out_dtype: Any,
) -> jax.Array:
  """Computes `activation(gate) * up` with optional symmetric/upper SwiGLU clipping."""
  if swiglu_limit is not None and swiglu_limit > 0:
    limit = jnp.asarray(swiglu_limit, dtype=gate.dtype)
    gate = jnp.minimum(gate, limit)
    up = jnp.clip(up, -limit, limit)
  return (activation(gate) * up).astype(out_dtype)


def _truncate_slots_by_capacity(
    safe_idx: jax.Array,
    slot_valid: jax.Array,
    *,
    num_experts: int,
    expert_cap: int,
    prior_expert_counts: jax.Array | None = None,
) -> jax.Array:
  """Drops slots whose per-expert token-major rank exceeds `expert_cap` in O(M) memory."""
  num_slots = safe_idx.shape[0]
  sort_key = jnp.where(slot_valid, safe_idx, num_experts)
  order = jnp.argsort(sort_key, stable=True)
  counts = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
  starts = jnp.cumsum(counts) - counts
  sorted_exp = safe_idx[order]
  rank_in_exp = jnp.arange(num_slots, dtype=jnp.int32) - starts[sorted_exp]
  if prior_expert_counts is not None:
    rank_in_exp = rank_in_exp + prior_expert_counts[sorted_exp]
  keep_sorted = slot_valid[order] & (rank_in_exp < int(expert_cap))
  return keep_sorted[jnp.argsort(order)]


def _truncate_slots_by_buffer(
    safe_idx: jax.Array,
    slot_valid: jax.Array,
    *,
    num_experts: int,
    buf_cap: int,
) -> jax.Array:
  """Keeps at most `buf_cap` valid slots in expert-sorted order."""
  num_slots = safe_idx.shape[0]
  if buf_cap >= num_slots:
    return slot_valid
  sort_key = jnp.where(slot_valid, safe_idx, num_experts)
  order = jnp.argsort(sort_key, stable=True)
  keep_sorted = slot_valid[order] & (jnp.arange(num_slots, dtype=jnp.int32) < int(buf_cap))
  return keep_sorted[jnp.argsort(order)]


def _ep_ragged_exchange(
    operand: jax.Array,
    out_len: int,
    counts_matrix: jax.Array,
    ep_id: jax.Array,
    ep_axis: str,
    *,
    is_tpu: bool,
) -> jax.Array:
  """Exchanges variable-length token slices across `ep_axis`.

  `counts_matrix[src, dst]` gives the number of rows sent from `src` to `dst`.
  On TPU, dispatches directly to `jax.lax.ragged_all_to_all` using sender-side
  RDMA destination offsets. On CPU (where XLA's CPU thunk emitter lacks
  `ragged-all-to-all`), computes the bit-identical `[out_len, d_dim]` buffer via
  token-only `jax.lax.all_to_all`.
  """
  ep_size = counts_matrix.shape[0]
  d_dim = operand.shape[1]
  send_sz = counts_matrix[ep_id]
  in_off = jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(send_sz)[:-1]], axis=0)
  out_off_all = jnp.cumsum(
      jnp.concatenate([jnp.zeros((1, ep_size), dtype=jnp.int32), counts_matrix], axis=0),
      axis=0,
  )[:-1]
  out_off = out_off_all[ep_id]
  recv_sz = counts_matrix[:, ep_id]

  if is_tpu:
    out_buf = jnp.zeros((out_len, d_dim), dtype=operand.dtype)
    return jax.lax.ragged_all_to_all(
        operand,
        out_buf,
        in_off,
        send_sz,
        out_off,
        recv_sz,
        axis_name=ep_axis,
    )

  in_len = operand.shape[0]
  cap = max(1, min(int(in_len), int(out_len)))
  col = jnp.arange(cap, dtype=jnp.int32)[None, :]
  gather_idx = in_off[:, None] + col
  valid_send = col < send_sz[:, None]
  safe_gather = jnp.clip(gather_idx, 0, max(int(in_len) - 1, 0))
  staged = jnp.where(valid_send[:, :, None], operand[safe_gather], jnp.zeros((), dtype=operand.dtype))
  recvd = jax.lax.all_to_all(staged, ep_axis, split_axis=0, concat_axis=0)

  recv_ends = jnp.cumsum(recv_sz)
  recv_starts = recv_ends - recv_sz
  row_idx = jnp.arange(out_len, dtype=jnp.int32)
  src_idx = jnp.searchsorted(recv_ends, row_idx, side="right")
  valid_row = row_idx < recv_ends[-1]
  safe_src = jnp.minimum(src_idx, ep_size - 1)
  pos_in_src = jnp.clip(row_idx - recv_starts[safe_src], 0, cap - 1)
  return jnp.where(valid_row[:, None], recvd[safe_src, pos_in_src], jnp.zeros((), dtype=operand.dtype))


def _dense_separate(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    *,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    capacity_factor: float,
) -> jax.Array:
  """Runs every expert on every token and combines selected expert outputs."""
  if capacity_factor > 0.0:
    batch, length, _ = x.shape
    top_k = indices.shape[-1]
    num_tokens = batch * length
    flat_idx = jnp.reshape(indices, (-1,)).astype(jnp.int32)
    slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
    safe_idx = jnp.where(slot_valid, flat_idx, 0)
    cap = max(1, int(math.ceil((num_tokens * top_k / num_experts) * capacity_factor)))
    slot_valid = _truncate_slots_by_capacity(safe_idx, slot_valid, num_experts=num_experts, expert_cap=cap)
    eff_indices = jnp.reshape(jnp.where(slot_valid, safe_idx, -1), indices.shape)
    eff_weights = jnp.reshape(jnp.where(slot_valid, jnp.reshape(weights, (-1,)), 0), weights.shape)
  else:
    eff_indices = indices
    eff_weights = weights
  table = combine_weights(eff_weights, eff_indices, num_experts)
  prec = _matmul_precision(x.dtype)
  gate = jnp.einsum("bse,neh->bsnh", x, wi_0, precision=prec)
  up = jnp.einsum("bse,neh->bsnh", x, wi_1, precision=prec)
  hidden = _apply_gated_activation(gate, up, activation=activation, swiglu_limit=swiglu_limit, out_dtype=x.dtype)
  weighted_hidden = (hidden.astype(_COMBINE_DTYPE) * table[..., None].astype(_COMBINE_DTYPE)).astype(wo.dtype)
  return jnp.einsum("bsnh,nhe->bse", weighted_hidden, wo, precision=prec, preferred_element_type=_COMBINE_DTYPE)


def _dense_fused(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    gate_up_proj: jax.Array,
    down_proj: jax.Array,
    *,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    capacity_factor: float,
) -> jax.Array:
  """Dense reference execution for 3D `(E, D, 2*M)` or 4D `(E, D, 2, M)` `gate_up_proj`."""
  if gate_up_proj.ndim == 4:
    if capacity_factor > 0.0:
      batch, length, _ = x.shape
      top_k = indices.shape[-1]
      num_tokens = batch * length
      flat_idx = jnp.reshape(indices, (-1,)).astype(jnp.int32)
      slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
      safe_idx = jnp.where(slot_valid, flat_idx, 0)
      cap = max(1, int(math.ceil((num_tokens * top_k / num_experts) * capacity_factor)))
      slot_valid = _truncate_slots_by_capacity(safe_idx, slot_valid, num_experts=num_experts, expert_cap=cap)
      eff_indices = jnp.reshape(jnp.where(slot_valid, safe_idx, -1), indices.shape)
      eff_weights = jnp.reshape(jnp.where(slot_valid, jnp.reshape(weights, (-1,)), 0), weights.shape)
    else:
      eff_indices = indices
      eff_weights = weights
    table = combine_weights(eff_weights, eff_indices, num_experts)
    prec = _matmul_precision(x.dtype)
    gu = jnp.einsum("bse,negh->bsngh", x, gate_up_proj, precision=prec)
    hidden = _apply_gated_activation(
        gu[..., 0, :], gu[..., 1, :], activation=activation, swiglu_limit=swiglu_limit, out_dtype=x.dtype
    )
    weighted_hidden = (hidden.astype(_COMBINE_DTYPE) * table[..., None].astype(_COMBINE_DTYPE)).astype(down_proj.dtype)
    return jnp.einsum("bsnh,nhe->bse", weighted_hidden, down_proj, precision=prec, preferred_element_type=_COMBINE_DTYPE)
  wi_0, wi_1 = jnp.split(gate_up_proj, 2, axis=-1)
  return _dense_separate(
      x,
      weights,
      indices,
      wi_0,
      wi_1,
      down_proj,
      num_experts=num_experts,
      activation=activation,
      swiglu_limit=swiglu_limit,
      capacity_factor=capacity_factor,
  )


def _run_local_expert_mlp(
    sorted_x: jax.Array,
    projections: tuple[jax.Array, ...],
    group_sizes: jax.Array,
    *,
    fused: bool,
    matmul_fn: Callable[..., jax.Array],
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    out_dtype: Any = None,
) -> jax.Array:
  """Runs gate/up + activation + down grouped matmuls on sorted rows."""
  if fused:
    w_gu, w_dn = projections
    gate_up = matmul_fn(sorted_x, w_gu, group_sizes)
    gate, up = jnp.split(gate_up, 2, axis=-1)
    hidden = _apply_gated_activation(
        gate,
        up,
        activation=activation,
        swiglu_limit=swiglu_limit,
        out_dtype=sorted_x.dtype,
    )
    return matmul_fn(hidden, w_dn, group_sizes, preferred_element_type=out_dtype)

  wi_0, wi_1, wo = projections
  gate = matmul_fn(sorted_x, wi_0, group_sizes)
  up = matmul_fn(sorted_x, wi_1, group_sizes)
  hidden = _apply_gated_activation(
      gate,
      up,
      activation=activation,
      swiglu_limit=swiglu_limit,
      out_dtype=sorted_x.dtype,
  )
  return matmul_fn(hidden, wo, group_sizes, preferred_element_type=out_dtype)


def _grouped_unsharded(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    projections: tuple[jax.Array, ...],
    *,
    fused: bool,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    executor: MoEExecutor,
    backend: str = "ragged_dot",
    is_tpu: bool = False,
) -> jax.Array:
  """Portable `ragged_dot` grouped execution without `shard_map`."""
  if executor.bwd_wgrad_overlap:
    x, projections = backward_wgrad_tie(x, projections)
  if fused and projections[0].ndim == 4:
    w_gu_4d, w_dn = projections
    projections = (
        jnp.reshape(w_gu_4d, (w_gu_4d.shape[0], w_gu_4d.shape[1], 2 * w_gu_4d.shape[3])),
        w_dn,
    )

  batch, length, embed = x.shape
  top_k = indices.shape[-1]
  num_tokens = batch * length
  num_slots = num_tokens * top_k

  flat_idx = jnp.reshape(indices, (num_slots,)).astype(jnp.int32)
  flat_w = jnp.reshape(weights, (num_slots,))
  slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
  safe_idx = jnp.where(slot_valid, flat_idx, 0)

  if executor.capacity_factor > 0.0:
    cap = max(1, int(math.ceil((num_slots / num_experts) * executor.capacity_factor)))
    slot_valid = _truncate_slots_by_capacity(safe_idx, slot_valid, num_experts=num_experts, expert_cap=cap)
  if executor.ragged_buffer_factor > 0.0:
    buf_cap = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
    slot_valid = _truncate_slots_by_buffer(safe_idx, slot_valid, num_experts=num_experts, buf_cap=buf_cap)

  sort_key = jnp.where(slot_valid, safe_idx, num_experts)
  perm = jnp.argsort(sort_key, stable=True)
  inv_perm = jnp.argsort(perm)
  group_sizes = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
  valid_count = jnp.sum(group_sizes)

  slot_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
  slot_rows = jnp.reshape(x, (num_tokens, embed))[slot_token_ids]
  sorted_x = _permute_rows(slot_rows, perm)
  sorted_x = jnp.where((jnp.arange(num_slots, dtype=jnp.int32) < valid_count)[:, None], sorted_x, 0)

  matmul_fn = functools.partial(grouped_matmul, backend=backend, is_tpu=is_tpu)
  expert_out = _run_local_expert_mlp(
      sorted_x,
      projections,
      group_sizes,
      fused=fused,
      matmul_fn=matmul_fn,
      activation=activation,
      swiglu_limit=swiglu_limit,
      out_dtype=_COMBINE_DTYPE,
  )
  unsorted_out = _permute_rows(expert_out, inv_perm)
  eff_w = jnp.where(slot_valid, flat_w.astype(_COMBINE_DTYPE), 0.0)
  safe_out = jnp.where(slot_valid[:, None], unsorted_out.astype(_COMBINE_DTYPE), 0.0)
  weighted = safe_out * eff_w[:, None]
  combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, embed)), axis=1)
  return jnp.reshape(combined, (batch, length, embed))


def _can_use_sparsecore_ragged_sort(num_rows: int, hidden_size: int, dtype: Any) -> bool:
  """Returns True when SparseCore `ragged_sort` kernels support `(num_rows, hidden_size, dtype)` on the active TPU."""
  dt = jnp.dtype(dtype)
  if dt not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
    return False
  if num_rows <= 0 or hidden_size <= 0:
    return False
  if (num_rows * jax.dtypes.itemsize_bits(dt)) % 32 != 0:
    return False
  from jax.experimental.pallas import tpu as pltpu  # pylint: disable=import-outside-toplevel

  if not pltpu.is_tpu_device():
    return False
  try:
    tpu_info = pltpu.get_tpu_info()
  except ValueError:
    return False
  sc_info = tpu_info.sparse_core
  if sc_info is None:
    return False
  num_lanes = int(tpu_info.num_lanes)
  if hidden_size % num_lanes != 0:
    return False
  num_cores = int(sc_info.num_cores) * int(sc_info.num_subcores)
  num_simd_lanes = int(sc_info.num_lanes)
  preferred_num_stages = 4
  num_col_parts = 1
  while num_cores // num_col_parts > num_simd_lanes:
    next_col_parts = num_col_parts * 2
    if num_cores % next_col_parts != 0:
      break
    if hidden_size % (num_lanes * next_col_parts) != 0:
      break
    if hidden_size // (num_lanes * next_col_parts) < preferred_num_stages:
      break
    num_col_parts = next_col_parts
  return (num_cores // num_col_parts) <= num_simd_lanes


def _validate_forced_routed_inputs(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array | None,
    forced_routed_experts: jax.Array,
    num_experts: int,
    *,
    router_probs: jax.Array | None = None,
) -> None:
  """Validates 3D inputs when `forced_routed_experts` overrides routing indices."""
  if x.ndim != 3:
    raise ValueError(f"Expected 3D x (B, S, D); got x={x.shape}.")
  if forced_routed_experts.ndim != 3:
    raise ValueError(
        f"Expected 3D forced_routed_experts (B, S, K); got forced_routed_experts={forced_routed_experts.shape}."
    )
  top_k = int(forced_routed_experts.shape[2])
  _validate_forced_routed_experts(forced_routed_experts, (x.shape[0], x.shape[1]), top_k, num_experts)
  if indices is not None and (indices.ndim != 3 or indices.shape != forced_routed_experts.shape):
    raise ValueError(
        f"indices shape {indices.shape} must match forced_routed_experts shape {forced_routed_experts.shape}."
    )
  if router_probs is not None:
    if router_probs.ndim != 3 or router_probs.shape[:2] != x.shape[:2] or router_probs.shape[2] != num_experts:
      raise ValueError(
          f"router_probs shape {router_probs.shape} must match (B, S, num_experts)="
          f"({x.shape[0]}, {x.shape[1]}, {num_experts})."
      )
  else:
    if indices is not None:
      raise ValueError(
          "forced_routed_experts with indices provided requires full-expert router_probs "
          "(or pass indices=None with full-expert scores in weights; if weights and indices "
          "are already forced-gathered via route_top_k/apply_forced_routed_experts, pass "
          "forced_routed_experts=None)."
      )
    if weights.ndim != 3 or weights.shape[:2] != x.shape[:2]:
      raise ValueError(f"Expected 3D weights matching x batch/length {x.shape[:2]}; got weights={weights.shape}.")
    if weights.shape[2] != num_experts:
      raise ValueError(
          f"When indices is None without router_probs, weights last dim ({weights.shape[2]}) must match "
          f"num_experts ({num_experts}); got weights={weights.shape}."
      )


def _resolve_forced_weights_source(
    weights: jax.Array,
    *,
    router_probs: jax.Array | None = None,
) -> jax.Array:
  """Returns full-expert routing scores (`router_probs` when provided, else `weights`)."""
  return router_probs if router_probs is not None else weights


def _apply_forced_routed_inputs(
    weights_or_scores: jax.Array,
    forced_routed_experts: jax.Array,
    *,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scores_are_logits: bool = False,
    eps: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
  """Resolves per-slot `(weights, indices)` from full-expert scores under `forced_routed_experts`."""
  return apply_forced_routed_experts(
      weights_or_scores,
      forced_routed_experts,
      norm_topk_prob=norm_topk_prob,
      routed_scaling_factor=routed_scaling_factor,
      scores_are_logits=scores_are_logits,
      eps=eps,
      dtype=_COMBINE_DTYPE,
  )


def _sharded_grouped(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    projections: tuple[jax.Array, ...],
    *,
    fused: bool,
    num_experts: int,
    activation: Callable[[jax.Array], jax.Array],
    swiglu_limit: float | None,
    executor: MoEExecutor,
    forced_routed_experts: jax.Array | None = None,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scores_are_logits: bool = False,
    eps: float = 0.0,
) -> jax.Array:
  """Distributed `shard_map` execution supporting `EP >= 1`, `TP >= 1`, and `FSDP >= 1`."""
  mesh = kernels.mesh_in_scope()
  is_tpu = kernels.is_tpu_mesh(mesh)

  batch_ax = kernels.resolve_mesh_axes(mesh, executor.batch)
  experts_ax = kernels.resolve_mesh_axes(mesh, executor.experts)
  mlp_ax = kernels.resolve_mesh_axes(mesh, executor.mlp)

  batch_shards = kernels.mesh_axis_size(mesh, batch_ax)
  ep_size = kernels.mesh_axis_size(mesh, experts_ax)
  tp_size = kernels.mesh_axis_size(mesh, mlp_ax)
  ep_axes = kernels.mesh_axis_names(experts_ax)
  tp_axes = kernels.mesh_axis_names(mlp_ax)

  if len(ep_axes) > 1:
    raise NotImplementedError(f"Expert sharding supports at most one mesh axis; got {ep_axes}.")
  if x.shape[0] % batch_shards != 0:
    raise ValueError(f"Batch {x.shape[0]} does not divide over the {batch_shards} token shards of {batch_ax!r}.")
  if num_experts % ep_size != 0:
    raise ValueError(
        f"num_experts ({num_experts}) must be divisible by expert parallelism size ({ep_size}) on {experts_ax!r}."
    )

  if fused:
    w_gu_global, w_dn_global = projections
    is_4d_fused = w_gu_global.ndim == 4
    if is_4d_fused:
      mlp_dim = w_gu_global.shape[3]
      proj_specs: tuple[P, ...] = (P(experts_ax, None, None, mlp_ax), P(experts_ax, mlp_ax, None))
    else:
      if tp_size > 1:
        raise ValueError(
            "3D fused gate_up_proj cannot be tensor-parallel sharded across tp_size > 1 without replicating expert weights; "
            "store gate_up_proj as 4D (num_experts, emb_dim, 2, moe_mlp_dim)."
        )
      mlp_dim = w_gu_global.shape[2] // 2
      proj_specs = (P(experts_ax, None, None), P(experts_ax, mlp_ax, None))
    if w_dn_global.shape[1] != mlp_dim:
      raise ValueError(f"down_proj MLP dim ({w_dn_global.shape[1]}) must match gate_up MLP dim ({mlp_dim}).")
  else:
    wi_0_global, wi_1_global, wo_global = projections
    is_4d_fused = False
    mlp_dim = wi_0_global.shape[2]
    if wi_1_global.shape != wi_0_global.shape or wo_global.shape[1] != mlp_dim:
      raise ValueError(
          f"Mismatched expert weight shapes: wi_0={wi_0_global.shape}, wi_1={wi_1_global.shape}, wo={wo_global.shape}."
      )
    proj_specs = (P(experts_ax, None, mlp_ax), P(experts_ax, None, mlp_ax), P(experts_ax, mlp_ax, None))

  if mlp_dim % tp_size != 0:
    raise ValueError(f"Expert MLP dim ({mlp_dim}) must be divisible by tensor parallelism size ({tp_size}).")

  raw_length_ax = kernels.resolve_mesh_axes(mesh, executor.length)
  claimed_axes = set(kernels.mesh_axis_names(batch_ax)) | set(ep_axes) | set(tp_axes)
  cand_length_axes = [
      ax for ax in kernels.mesh_axis_names(raw_length_ax) if ax not in claimed_axes and int(mesh.shape[ax]) > 1
  ]
  cand_length_ax: MeshAxes = (
      cand_length_axes[0] if len(cand_length_axes) == 1 else (tuple(cand_length_axes) if cand_length_axes else None)
  )
  length_shards = kernels.mesh_axis_size(mesh, cand_length_ax)
  length_ax = cand_length_ax if (length_shards > 1 and x.shape[1] > 1 and x.shape[1] % length_shards == 0) else None
  tokens_spec = P(batch_ax, length_ax, None)
  matmul_fn = functools.partial(grouped_matmul, backend=executor.kind, is_tpu=is_tpu)
  e_loc = num_experts // ep_size
  m_tp = mlp_dim // tp_size

  def _slice_local_projections(proj_local: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    if not fused:
      return proj_local
    w_gu, w_dn = proj_local
    if is_4d_fused:
      return (jnp.reshape(w_gu, (e_loc, w_gu.shape[1], 2 * m_tp)), w_dn)
    return (w_gu, w_dn)

  def per_device(x_sh: jax.Array, weights_sh: jax.Array, indices_sh: jax.Array, *proj_sh: jax.Array) -> jax.Array:
    proj_tuple: tuple[jax.Array, ...] = tuple(proj_sh)
    if executor.bwd_wgrad_overlap:
      x_sh, proj_tuple = backward_wgrad_tie(x_sh, proj_tuple)
    local_projs = _slice_local_projections(proj_tuple)

    b_loc, seq_len, d_dim = x_sh.shape
    top_k = indices_sh.shape[-1]
    num_tokens = b_loc * seq_len
    num_slots = num_tokens * top_k

    flat_x = jnp.reshape(x_sh, (num_tokens, d_dim))
    flat_idx = jnp.reshape(indices_sh, (num_slots,)).astype(jnp.int32)
    flat_w = jnp.reshape(weights_sh, (num_slots,))
    slot_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), top_k)
    slot_rows = flat_x[slot_token_ids]

    slot_valid = (flat_idx >= 0) & (flat_idx < num_experts)
    safe_idx = jnp.where(slot_valid, flat_idx, 0)

    if executor.capacity_factor > 0.0:
      expert_cap = max(1, int(math.ceil((num_slots / e_loc) * executor.capacity_factor)))
      prior_exp = None
      if ep_size > 1:
        local_counts_pre = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(
            jnp.int32
        )
        all_counts_pre = jax.lax.all_gather(local_counts_pre, ep_axes[0], axis=0, tiled=False)
        ep_id_pre = jax.lax.axis_index(ep_axes[0])
        prior_exp = (jnp.cumsum(all_counts_pre, axis=0) - all_counts_pre)[ep_id_pre]
      slot_valid = _truncate_slots_by_capacity(
          safe_idx,
          slot_valid,
          num_experts=num_experts,
          expert_cap=expert_cap,
          prior_expert_counts=prior_exp,
      )

    if ep_size == 1:
      if executor.ragged_buffer_factor > 0.0:
        buf_cap = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
        slot_valid = _truncate_slots_by_buffer(safe_idx, slot_valid, num_experts=num_experts, buf_cap=buf_cap)

      sort_key = jnp.where(slot_valid, safe_idx, num_experts)
      perm = jnp.argsort(sort_key, stable=True)
      inv_perm = jnp.argsort(perm)
      group_sizes = jnp.bincount(safe_idx, weights=slot_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
      valid_count = jnp.sum(group_sizes)

      if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(num_slots, d_dim, slot_rows.dtype):
        from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

        sorted_x = ragged_sort.a2a_ragged_sort(slot_rows, perm, valid_count)
      else:
        sorted_x = _permute_rows(slot_rows, perm)
        sorted_x = jnp.where((jnp.arange(num_slots, dtype=jnp.int32) < valid_count)[:, None], sorted_x, 0)

      expert_out = _run_local_expert_mlp(
          sorted_x,
          local_projs,
          group_sizes,
          fused=fused,
          matmul_fn=matmul_fn,
          activation=activation,
          swiglu_limit=swiglu_limit,
          out_dtype=_COMBINE_DTYPE,
      )

      if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(num_slots, d_dim, expert_out.dtype):
        from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

        unsorted_out = ragged_sort.a2a_ragged_unsort(expert_out, inv_perm, num_slots)
      else:
        unsorted_out = _permute_rows(expert_out, inv_perm)

      eff_w = jnp.where(slot_valid, flat_w.astype(_COMBINE_DTYPE), 0.0)
      safe_out = jnp.where(slot_valid[:, None], unsorted_out.astype(_COMBINE_DTYPE), 0.0)
      weighted = safe_out * eff_w[:, None]
      combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, d_dim)), axis=1)
      if tp_axes:
        combined = jax.lax.psum(combined, tp_axes)
      return jnp.reshape(combined, (b_loc, seq_len, d_dim))

    # Multi-device Expert Parallelism (EP > 1) via ragged_all_to_all.
    ep_axis = ep_axes[0]
    ep_id = jax.lax.axis_index(ep_axis)

    def _ep_dispatch_round(
        s_rows: jax.Array,
        s_idx: jax.Array,
        r_valid: jax.Array,
        buf_len: int,
        projs_in: tuple[jax.Array, ...],
        *,
        clamp_to_buf: bool,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
      n_slots_in = s_rows.shape[0]
      local_exp_counts = jnp.bincount(s_idx, weights=r_valid.astype(jnp.int32), length=num_experts).astype(jnp.int32)
      all_exp_counts = jax.lax.all_gather(local_exp_counts, ep_axis, axis=0, tiled=False)

      if clamp_to_buf:
        by_dst = jnp.transpose(jnp.reshape(all_exp_counts, (ep_size, ep_size, e_loc)), (1, 0, 2))
        flat_by_dst = jnp.reshape(by_dst, (ep_size, ep_size * e_loc))
        cum_end = jnp.cumsum(flat_by_dst, axis=1)
        cum_start = cum_end - flat_by_dst
        clamped_flat = jnp.clip(buf_len - cum_start, 0, flat_by_dst)
        round_exp_counts = jnp.reshape(
            jnp.transpose(jnp.reshape(clamped_flat, (ep_size, ep_size, e_loc)), (1, 0, 2)),
            (ep_size, num_experts),
        )
        allowed_local = round_exp_counts[ep_id]
        pre_key = jnp.where(r_valid, s_idx, num_experts)
        pre_order = jnp.argsort(pre_key, stable=True)
        pre_starts = jnp.cumsum(local_exp_counts) - local_exp_counts
        pre_exp = s_idx[pre_order]
        pre_rank = jnp.arange(n_slots_in, dtype=jnp.int32) - pre_starts[pre_exp]
        keep_pre = r_valid[pre_order] & (pre_rank < allowed_local[pre_exp])
        round_valid = keep_pre[jnp.argsort(pre_order)]
        leftover_valid = r_valid & ~round_valid
        any_overflow = jnp.any(cum_end[:, -1] > buf_len)
      else:
        round_exp_counts = all_exp_counts
        round_valid = r_valid
        leftover_valid = jnp.zeros_like(r_valid)
        any_overflow = jnp.bool_(False)

      send_key = jnp.where(round_valid, s_idx, num_experts)
      send_perm = jnp.argsort(send_key, stable=True)
      inv_send_perm = jnp.argsort(send_perm)
      send_x = _permute_rows(s_rows, send_perm)

      counts_matrix = jnp.sum(jnp.reshape(round_exp_counts, (ep_size, ep_size, e_loc)), axis=2)
      recv_x = _ep_ragged_exchange(send_x, buf_len, counts_matrix, ep_id, ep_axis, is_tpu=is_tpu)

      recv_exp_by_src = jax.lax.dynamic_slice_in_dim(round_exp_counts, ep_id * e_loc, e_loc, axis=1)
      local_group_sizes = jnp.sum(recv_exp_by_src, axis=0)
      chunk_sizes = jnp.reshape(recv_exp_by_src, (ep_size * e_loc,))
      chunk_ends = jnp.cumsum(chunk_sizes)
      recv_valid_end = chunk_ends[-1]
      row_idx = jnp.arange(buf_len, dtype=jnp.int32)
      chunk_idx = jnp.searchsorted(chunk_ends, row_idx, side="right")
      recv_local_exp = jnp.where(row_idx < recv_valid_end, chunk_idx % e_loc, e_loc)

      local_perm = jnp.argsort(recv_local_exp, stable=True)
      inv_local_perm = jnp.argsort(local_perm)

      if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(buf_len, d_dim, recv_x.dtype):
        from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

        sorted_recv_x = ragged_sort.a2a_ragged_sort(recv_x, local_perm, recv_valid_end)
      else:
        sorted_recv_x = _permute_rows(recv_x, local_perm)
        sorted_recv_x = jnp.where((row_idx < recv_valid_end)[:, None], sorted_recv_x, 0)

      expert_out = _run_local_expert_mlp(
          sorted_recv_x,
          projs_in,
          local_group_sizes,
          fused=fused,
          matmul_fn=matmul_fn,
          activation=activation,
          swiglu_limit=swiglu_limit,
      )

      if executor.use_ragged_sort and is_tpu and _can_use_sparsecore_ragged_sort(buf_len, d_dim, expert_out.dtype):
        from maxtext.kernels.ragged import ragged_sort  # pylint: disable=import-outside-toplevel

        recv_unsorted = ragged_sort.a2a_ragged_unsort(expert_out, inv_local_perm, recv_valid_end)
      else:
        recv_unsorted = _permute_rows(expert_out, inv_local_perm)
        recv_unsorted = jnp.where((row_idx < recv_valid_end)[:, None], recv_unsorted, 0)

      returned_sorted_slots = _ep_ragged_exchange(
          recv_unsorted, n_slots_in, counts_matrix.T, ep_id, ep_axis, is_tpu=is_tpu
      )
      unsorted_slots = _permute_rows(returned_sorted_slots, inv_send_perm)
      safe_slots = jnp.where(round_valid[:, None], unsorted_slots.astype(_COMBINE_DTYPE), 0.0)
      return safe_slots, leftover_valid, any_overflow

    eff_dbf = resolve_dropless_buffer_factor(
        executor.dropless_buffer_factor,
        ep_size,
        capacity_factor=executor.capacity_factor,
        ragged_buffer_factor=executor.ragged_buffer_factor,
    )
    # Align fractional buffer bounds to 32-bit packing so bfloat16 SparseCore ragged_sort stays valid.
    row_align = max(1, 32 // jax.dtypes.itemsize_bits(x_sh.dtype))
    recv_buf_len = ep_size * num_slots
    if executor.capacity_factor > 0.0:
      expert_cap = max(1, int(math.ceil((num_slots / e_loc) * executor.capacity_factor)))
      recv_buf_len = min(recv_buf_len, e_loc * expert_cap)
    if executor.ragged_buffer_factor > 0.0:
      rbf_cap = max(1, int(math.ceil(num_slots * executor.ragged_buffer_factor)))
      rbf_cap = ((rbf_cap + row_align - 1) // row_align) * row_align
      recv_buf_len = min(recv_buf_len, rbf_cap)
    if eff_dbf > 0.0:
      dbf_cap = max(1, int(math.ceil(num_slots * eff_dbf)))
      dbf_cap = ((dbf_cap + row_align - 1) // row_align) * row_align
      recv_buf_len = min(recv_buf_len, dbf_cap)

    safe_slots, rem_valid, any_overflow = _ep_dispatch_round(
        slot_rows,
        safe_idx,
        slot_valid,
        recv_buf_len,
        local_projs,
        clamp_to_buf=(recv_buf_len < ep_size * num_slots),
    )

    # Bounded receive buffer + conditional overflow round(s) with zero dropped tokens.
    if eff_dbf > 0.0 and recv_buf_len < ep_size * num_slots:
      max_overflow_rounds = max(1, int(math.ceil((ep_size * num_slots) / recv_buf_len)) - 1)

      @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
      def _maybe_run_overflow_rounds(
          acc_slots: jax.Array,
          s_rows: jax.Array,
          s_idx: jax.Array,
          r_valid_init: jax.Array,
          ov_init: jax.Array,
          *projs_in: jax.Array,
      ) -> jax.Array:
        projs_tup = tuple(projs_in)

        def _do_round(
            op: tuple[jax.Array, jax.Array],
            round_projs: tuple[jax.Array, ...],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
          cur_acc, cur_valid = op
          extra_slots, next_valid, next_ov = _ep_dispatch_round(
              s_rows, s_idx, cur_valid, recv_buf_len, round_projs, clamp_to_buf=True
          )
          return cur_acc + extra_slots, next_valid, next_ov

        def _skip_round(op: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
          cur_acc, cur_valid = op
          return cur_acc, cur_valid, jnp.bool_(False)

        if max_overflow_rounds == 1:
          out_acc, _, _ = jax.lax.cond(
              ov_init,
              lambda op: _do_round(op, projs_tup),
              _skip_round,
              (acc_slots, r_valid_init),
          )
          return out_acc

        @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
        def _ov_step(
            c_acc: jax.Array,
            c_valid: jax.Array,
            c_ov: jax.Array,
            *step_projs: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
          return jax.lax.cond(
              c_ov,
              lambda op: _do_round(op, tuple(step_projs)),
              _skip_round,
              (c_acc, c_valid),
          )

        def _run_multi_overflow(op: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
          init_acc, init_valid = op

          def _ov_scan_body(state, _):
            c_acc, c_valid, c_ov = state
            next_state = _ov_step(c_acc, c_valid, c_ov, *projs_tup)
            return next_state, None

          final_state, _ = jax.lax.scan(
              _ov_scan_body,
              (init_acc, init_valid, jnp.bool_(True)),
              None,
              length=max_overflow_rounds,
          )
          return final_state

        out_acc, _, _ = jax.lax.cond(ov_init, _run_multi_overflow, _skip_round, (acc_slots, r_valid_init))
        return out_acc

      safe_slots = _maybe_run_overflow_rounds(safe_slots, slot_rows, safe_idx, rem_valid, any_overflow, *local_projs)

    eff_w = jnp.where(slot_valid, flat_w.astype(_COMBINE_DTYPE), 0.0)
    weighted = safe_slots * eff_w[:, None]
    combined = jnp.sum(jnp.reshape(weighted, (num_tokens, top_k, d_dim)), axis=1)
    if tp_axes:
      combined = jax.lax.psum(combined, tp_axes)
    return jnp.reshape(combined, (b_loc, seq_len, d_dim))

  x_in = kernels.ensure_explicit_spec(x, tokens_spec, mesh)
  weights_in = kernels.ensure_explicit_spec(weights, tokens_spec, mesh)
  proj_in = tuple(kernels.ensure_explicit_spec(p, spec, mesh) for p, spec in zip(projections, proj_specs))

  if forced_routed_experts is None:
    sharded_fn = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=(tokens_spec, tokens_spec, tokens_spec, *proj_specs),
        out_specs=tokens_spec,
        check_vma=False,
    )
    indices_in = kernels.ensure_explicit_spec(indices, tokens_spec, mesh)
    return sharded_fn(x_in, weights_in, indices_in, *proj_in)

  def per_device_forced(
      x_sh: jax.Array,
      weights_sh: jax.Array,
      forced_sh: jax.Array,
      *proj_sh: jax.Array,
  ) -> jax.Array:
    eff_w_sh, eff_idx_sh = _apply_forced_routed_inputs(
        weights_sh,
        forced_sh,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        scores_are_logits=scores_are_logits,
        eps=eps,
    )
    return per_device(x_sh, eff_w_sh, eff_idx_sh, *proj_sh)

  sharded_forced_fn = jax.shard_map(
      per_device_forced,
      mesh=mesh,
      in_specs=(tokens_spec, tokens_spec, tokens_spec, *proj_specs),
      out_specs=tokens_spec,
      check_vma=False,
  )
  forced_in = kernels.ensure_explicit_spec(forced_routed_experts, tokens_spec, mesh)
  return sharded_forced_fn(x_in, weights_in, forced_in, *proj_in)


# ==============================================================================
# Public entrypoints
# ==============================================================================


def _validate_routed_inputs(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array,
    num_experts: int,
) -> None:
  """Validates 3D token, routing weight, and routing index tensor shapes."""
  if x.ndim != 3 or weights.ndim != 3 or indices.ndim != 3:
    raise ValueError(
        f"Expected 3D x (B, S, D), weights (B, S, K), and indices (B, S, K); "
        f"got x={x.shape}, weights={weights.shape}, indices={indices.shape}."
    )
  if x.shape[:2] != weights.shape[:2] or weights.shape != indices.shape:
    raise ValueError(
        f"Mismatched batch/length/top_k shapes: x={x.shape}, weights={weights.shape}, indices={indices.shape}."
    )
  if num_experts < 1:
    raise ValueError(f"num_experts must be >= 1; got {num_experts}.")


def routed_experts(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array | None,
    wi_0: jax.Array,
    wi_1: jax.Array,
    wo: jax.Array,
    forced_routed_experts: jax.Array | None = None,
    *,
    executor: MoEExecutor = DENSE,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    swiglu_limit: float | None = None,
    router_probs: jax.Array | None = None,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scores_are_logits: bool = False,
    eps: float = 0.0,
) -> jax.Array:
  """Executes selected experts with separate `(wi_0, wi_1, wo)` projections."""
  if wi_0.ndim != 3 or wi_1.ndim != 3 or wo.ndim != 3:
    raise ValueError(f"Expected 3D wi_0, wi_1, wo; got {wi_0.shape}, {wi_1.shape}, {wo.shape}.")
  num_experts = wi_0.shape[0]
  if forced_routed_experts is None:
    if indices is None:
      raise ValueError("indices must be provided when forced_routed_experts is None.")
    _validate_routed_inputs(x, weights, indices, num_experts)
    if wi_0.shape[1] != x.shape[2] or wo.shape[0] != num_experts or wo.shape[2] != x.shape[2]:
      raise ValueError(f"Incompatible expert weight shapes for x={x.shape}: wi_0={wi_0.shape}, wo={wo.shape}.")
    weights = weights.astype(_COMBINE_DTYPE)
    projections = (wi_0.astype(x.dtype), wi_1.astype(x.dtype), wo.astype(x.dtype))
    match executor.kind:
      case "dense":
        out = _dense_separate(
            x,
            weights,
            indices,
            *projections,
            num_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
            capacity_factor=executor.capacity_factor,
        )
      case "ragged_dot":
        out = _grouped_unsharded(
            x,
            weights,
            indices,
            projections,
            fused=False,
            num_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
            executor=executor,
        )
      case "gmm_v2" | "tokamax":
        out = _sharded_grouped(
            x,
            weights,
            indices,
            projections,
            fused=False,
            num_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
            executor=executor,
        )
      case other:
        raise ValueError(f"Unknown MoE executor kind {other!r}; expected 'dense', 'ragged_dot', 'gmm_v2', or 'tokamax'.")
    return checkpoint_name(out.astype(x.dtype), "moe_routed_output")

  _validate_forced_routed_inputs(
      x,
      weights,
      indices,
      forced_routed_experts,
      num_experts,
      router_probs=router_probs,
  )
  if wi_0.shape[1] != x.shape[2] or wo.shape[0] != num_experts or wo.shape[2] != x.shape[2]:
    raise ValueError(f"Incompatible expert weight shapes for x={x.shape}: wi_0={wi_0.shape}, wo={wo.shape}.")
  projections = (wi_0.astype(x.dtype), wi_1.astype(x.dtype), wo.astype(x.dtype))
  weights_source = _resolve_forced_weights_source(
      weights,
      router_probs=router_probs,
  )

  match executor.kind:
    case "dense":
      eff_weights, eff_indices = _apply_forced_routed_inputs(
          weights_source,
          forced_routed_experts,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=routed_scaling_factor,
          scores_are_logits=scores_are_logits,
          eps=eps,
      )
      out = _dense_separate(
          x,
          eff_weights,
          eff_indices,
          *projections,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          capacity_factor=executor.capacity_factor,
      )
    case "ragged_dot":
      eff_weights, eff_indices = _apply_forced_routed_inputs(
          weights_source,
          forced_routed_experts,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=routed_scaling_factor,
          scores_are_logits=scores_are_logits,
          eps=eps,
      )
      out = _grouped_unsharded(
          x,
          eff_weights,
          eff_indices,
          projections,
          fused=False,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case "gmm_v2" | "tokamax":
      out = _sharded_grouped(
          x,
          weights_source.astype(_COMBINE_DTYPE),
          forced_routed_experts,
          projections,
          fused=False,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
          forced_routed_experts=forced_routed_experts,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=routed_scaling_factor,
          scores_are_logits=scores_are_logits,
          eps=eps,
      )
    case other:
      raise ValueError(f"Unknown MoE executor kind {other!r}; expected 'dense', 'ragged_dot', 'gmm_v2', or 'tokamax'.")
  return checkpoint_name(out.astype(x.dtype), "moe_routed_output")


def _device_is_tpu(dev: Any) -> bool:
  """Returns True when `dev` is a TPU device."""
  return getattr(dev, "platform", None) == "tpu"


def grouped_unsharded_fused(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array | None,
    gate_up_proj: jax.Array,
    down_proj: jax.Array,
    forced_routed_experts: jax.Array | None = None,
    *,
    executor: MoEExecutor = RAGGED_DOT,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    swiglu_limit: float | None = None,
    is_tpu: bool = False,
    router_probs: jax.Array | None = None,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scores_are_logits: bool = False,
    eps: float = 0.0,
) -> jax.Array:
  """Executes fused routed experts on a single device when no `Mesh` is in scope."""
  if gate_up_proj.ndim not in (3, 4) or down_proj.ndim != 3:
    raise ValueError(
        f"Expected 3D or 4D gate_up_proj and 3D down_proj; got gate_up_proj={gate_up_proj.shape}, "
        f"down_proj={down_proj.shape}."
    )
  if gate_up_proj.ndim == 3 and gate_up_proj.shape[2] % 2 != 0:
    raise ValueError(f"3D gate_up_proj last dim must be even (2 * mlp_dim); got {gate_up_proj.shape}.")
  if gate_up_proj.ndim == 4 and gate_up_proj.shape[2] != 2:
    raise ValueError(f"4D gate_up_proj axis 2 must have size 2 (gate, up); got {gate_up_proj.shape}.")
  mlp_dim = gate_up_proj.shape[2] // 2 if gate_up_proj.ndim == 3 else gate_up_proj.shape[3]
  num_experts = gate_up_proj.shape[0]
  if forced_routed_experts is None:
    if indices is None:
      raise ValueError("indices must be provided when forced_routed_experts is None.")
    _validate_routed_inputs(x, weights, indices, num_experts)
    eff_weights = weights.astype(_COMBINE_DTYPE)
    eff_indices = indices
  else:
    _validate_forced_routed_inputs(
        x,
        weights,
        indices,
        forced_routed_experts,
        num_experts,
        router_probs=router_probs,
    )
    weights_source = _resolve_forced_weights_source(
        weights,
        router_probs=router_probs,
    )
    eff_weights, eff_indices = _apply_forced_routed_inputs(
        weights_source,
        forced_routed_experts,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        scores_are_logits=scores_are_logits,
        eps=eps,
    )
  if gate_up_proj.shape[1] != x.shape[2] or down_proj.shape != (num_experts, mlp_dim, x.shape[2]):
    raise ValueError(
        f"Incompatible fused expert weight shapes for x={x.shape}: "
        f"gate_up_proj={gate_up_proj.shape}, down_proj={down_proj.shape}."
    )

  projections = (gate_up_proj.astype(x.dtype), down_proj.astype(x.dtype))
  if executor.kind == "dense":
    out = _dense_fused(
        x,
        eff_weights,
        eff_indices,
        *projections,
        num_experts=num_experts,
        activation=activation,
        swiglu_limit=swiglu_limit,
        capacity_factor=executor.capacity_factor,
    )
    return checkpoint_name(out.astype(x.dtype), "moe_routed_output")

  devs_1d: list[Any] = []
  for arr in (x, gate_up_proj):
    arr_shd = jax.typeof(arr).sharding
    if isinstance(arr_shd, jax.sharding.SingleDeviceSharding):
      devs_1d = list(arr_shd.device_set)[:1]
      break
    if isinstance(arr_shd, jax.sharding.NamedSharding) and not arr_shd.mesh.empty:
      devs_1d = list(arr_shd.mesh.devices.flat)[:1]
      break
  if not devs_1d:
    devs_1d = list(jax.local_devices()[:1] or jax.devices()[:1])
  tpu_1d = bool(is_tpu and devs_1d and _device_is_tpu(devs_1d[0]))
  if tpu_1d and executor.kind in SHARDED_EXECUTOR_KINDS:
    mesh_1d = jax.sharding.Mesh(devs_1d, ("_moe_1d",))
    out = jax.shard_map(
        lambda x_l, w_l, idx_l, gu_l, dn_l: _grouped_unsharded(
            x_l,
            w_l,
            idx_l,
            (gu_l, dn_l),
            fused=True,
            num_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
            executor=executor,
            backend=executor.kind,
            is_tpu=True,
        ).astype(x_l.dtype),
        mesh=mesh_1d,
        in_specs=(P(), P(), P(), P(), P()),
        out_specs=P(),
        check_vma=False,
    )(x, eff_weights, eff_indices, projections[0], projections[1])
    return checkpoint_name(out.astype(x.dtype), "moe_routed_output")

  backend = executor.kind if (tpu_1d and executor.kind in ("ragged_dot", *SHARDED_EXECUTOR_KINDS)) else "ragged_dot"
  out = _grouped_unsharded(
      x,
      eff_weights,
      eff_indices,
      projections,
      fused=True,
      num_experts=num_experts,
      activation=activation,
      swiglu_limit=swiglu_limit,
      executor=executor,
      backend=backend,
      is_tpu=tpu_1d,
  )
  return checkpoint_name(out.astype(x.dtype), "moe_routed_output")


def routed_experts_fused(
    x: jax.Array,
    weights: jax.Array,
    indices: jax.Array | None,
    gate_up_proj: jax.Array,
    down_proj: jax.Array,
    forced_routed_experts: jax.Array | None = None,
    *,
    executor: MoEExecutor = DENSE,
    activation: Callable[[jax.Array], jax.Array] = jax.nn.silu,
    swiglu_limit: float | None = None,
    router_probs: jax.Array | None = None,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    scores_are_logits: bool = False,
    eps: float = 0.0,
) -> jax.Array:
  """Executes selected experts with fused 3D `(E, D, 2*M)` or 4D `(E, D, 2, M)` `gate_up_proj`."""
  if gate_up_proj.ndim not in (3, 4) or down_proj.ndim != 3:
    raise ValueError(
        f"Expected 3D or 4D gate_up_proj and 3D down_proj; got gate_up_proj={gate_up_proj.shape}, "
        f"down_proj={down_proj.shape}."
    )
  if gate_up_proj.ndim == 3 and gate_up_proj.shape[2] % 2 != 0:
    raise ValueError(f"3D gate_up_proj last dim must be even (2 * mlp_dim); got {gate_up_proj.shape}.")
  if gate_up_proj.ndim == 4 and gate_up_proj.shape[2] != 2:
    raise ValueError(f"4D gate_up_proj axis 2 must have size 2 (gate, up); got {gate_up_proj.shape}.")

  num_experts = gate_up_proj.shape[0]
  if forced_routed_experts is None:
    if indices is None:
      raise ValueError("indices must be provided when forced_routed_experts is None.")
    _validate_routed_inputs(x, weights, indices, num_experts)
    if gate_up_proj.shape[1] != x.shape[2] or down_proj.shape[0] != num_experts or down_proj.shape[2] != x.shape[2]:
      raise ValueError(
          f"Incompatible fused expert weight shapes for x={x.shape}: "
          f"gate_up_proj={gate_up_proj.shape}, down_proj={down_proj.shape}."
      )
    weights = weights.astype(_COMBINE_DTYPE)
    projections = (gate_up_proj.astype(x.dtype), down_proj.astype(x.dtype))
    match executor.kind:
      case "dense":
        out = _dense_fused(
            x,
            weights,
            indices,
            *projections,
            num_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
            capacity_factor=executor.capacity_factor,
        )
      case "ragged_dot":
        out = _grouped_unsharded(
            x,
            weights,
            indices,
            projections,
            fused=True,
            num_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
            executor=executor,
        )
      case "gmm_v2" | "tokamax":
        out = _sharded_grouped(
            x,
            weights,
            indices,
            projections,
            fused=True,
            num_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
            executor=executor,
        )
      case other:
        raise ValueError(f"Unknown MoE executor kind {other!r}; expected 'dense', 'ragged_dot', 'gmm_v2', or 'tokamax'.")
    return checkpoint_name(out.astype(x.dtype), "moe_routed_output")

  _validate_forced_routed_inputs(
      x,
      weights,
      indices,
      forced_routed_experts,
      num_experts,
      router_probs=router_probs,
  )
  if gate_up_proj.shape[1] != x.shape[2] or down_proj.shape[0] != num_experts or down_proj.shape[2] != x.shape[2]:
    raise ValueError(
        f"Incompatible fused expert weight shapes for x={x.shape}: "
        f"gate_up_proj={gate_up_proj.shape}, down_proj={down_proj.shape}."
    )
  projections = (gate_up_proj.astype(x.dtype), down_proj.astype(x.dtype))
  weights_source = _resolve_forced_weights_source(
      weights,
      router_probs=router_probs,
  )

  match executor.kind:
    case "dense":
      eff_weights, eff_indices = _apply_forced_routed_inputs(
          weights_source,
          forced_routed_experts,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=routed_scaling_factor,
          scores_are_logits=scores_are_logits,
          eps=eps,
      )
      out = _dense_fused(
          x,
          eff_weights,
          eff_indices,
          *projections,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          capacity_factor=executor.capacity_factor,
      )
    case "ragged_dot":
      eff_weights, eff_indices = _apply_forced_routed_inputs(
          weights_source,
          forced_routed_experts,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=routed_scaling_factor,
          scores_are_logits=scores_are_logits,
          eps=eps,
      )
      out = _grouped_unsharded(
          x,
          eff_weights,
          eff_indices,
          projections,
          fused=True,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
      )
    case "gmm_v2" | "tokamax":
      out = _sharded_grouped(
          x,
          weights_source.astype(_COMBINE_DTYPE),
          forced_routed_experts,
          projections,
          fused=True,
          num_experts=num_experts,
          activation=activation,
          swiglu_limit=swiglu_limit,
          executor=executor,
          forced_routed_experts=forced_routed_experts,
          norm_topk_prob=norm_topk_prob,
          routed_scaling_factor=routed_scaling_factor,
          scores_are_logits=scores_are_logits,
          eps=eps,
      )
    case other:
      raise ValueError(f"Unknown MoE executor kind {other!r}; expected 'dense', 'ragged_dot', 'gmm_v2', or 'tokamax'.")
  return checkpoint_name(out.astype(x.dtype), "moe_routed_output")
