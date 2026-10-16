# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused MoE Top-K Gating and Normalized Softmax Kernel with Analytical VJP.

Optimized Pallas TPU kernel executing on Vector Units (VMEM), eliminating
intermediate HBM round-trips for top-k values, exploiting descending sorted order
to eliminate horizontal tree reductions in softmax, and employing an in-register
closed-form Vector-Jacobian Product (VJP) that eliminates autograd dynamic scatter
overhead.
"""

import functools
from typing import Tuple

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp


def _pallas_topk_kernel(
    logits_ref,  # Shape: (bM, E), Memory: VMEM
    weights_ref,  # Shape: (bM, k), Memory: VMEM
    indices_ref,  # Shape: (bM, k), Memory: VMEM
) -> None:
  """Fused Top-K selection and normalized Softmax kernel on TPU Vector Units."""
  logits_block = logits_ref[...]
  k = weights_ref.shape[-1]

  # Hardware vector unit bitonic selection across expert dimension.
  # TPU v6e/v7x Mosaic backend requires is_stable=False for Pallas top_k lowering.
  topk_vals, topk_indices = jax.lax.top_k(logits_block, k=k, is_stable=False)

  topk_vals_fp32 = topk_vals.astype(jnp.float32)

  # Optimization: Because top_k returns values sorted in descending order,
  # the maximum element along the top-k dimension is identically index 0.
  # Slicing the 0-th column directly in vector registers eliminates the horizontal
  # multi_reduction <maximumf> tree reduction instruction entirely.
  max_val = topk_vals_fp32[:, :1]

  # Compute numerically stabilized exponentials in vector registers:
  exps = jnp.exp(topk_vals_fp32 - max_val)

  # Compute sum of exponentials across top-k dimension:
  sum_exps = jnp.sum(exps, axis=-1, keepdims=True)

  # Optimization: Softmax reciprocal multiplication vectorization.
  inv_sum = jax.lax.reciprocal(sum_exps)
  weights = (exps * inv_sum).astype(logits_ref.dtype)

  weights_ref[...] = weights
  indices_ref[...] = topk_indices


def _pallas_fused_topk_gating(
    logits: jax.Array,
    k: int = 10,
) -> Tuple[jax.Array, jax.Array]:
  """Executes fused top-k gating via Pallas TPU kernel."""
  M, E = logits.shape
  bM = min(M, 2048)

  # Ensure grid evenly divides M
  if M % bM != 0:
    bM = 128
    while bM < M and M % (bM * 2) == 0 and (bM * 2) <= 2048:
      bM *= 2

  grid = (M // bM,)
  pipeline_mode = pl.Buffered(buffer_count=2) if bM < M else None

  in_specs = [
      pl.BlockSpec(
          block_shape=(bM, E),
          index_map=lambda i: (i, 0),
          pipeline_mode=pipeline_mode,
      )
  ]

  out_specs = (
      pl.BlockSpec(
          block_shape=(bM, k),
          index_map=lambda i: (i, 0),
      ),
      pl.BlockSpec(
          block_shape=(bM, k),
          index_map=lambda i: (i, 0),
      ),
  )

  out_shape = (
      jax.ShapeDtypeStruct((M, k), logits.dtype),
      jax.ShapeDtypeStruct((M, k), jnp.int32),
  )

  compiler_params = pltpu.CompilerParams(
      dimension_semantics=["parallel"],
      skip_device_barrier=True,
      disable_bounds_checks=True,
  )

  return pl.pallas_call(
      _pallas_topk_kernel,
      out_shape=out_shape,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs,
      compiler_params=compiler_params,
      debug=False,
  )(logits)


def _analytical_vjp(
    topk_weights: jax.Array,
    topk_indices: jax.Array,
    g_weights: jax.Array,
    num_experts: int,
) -> jax.Array:
  """Exact analytical Vector-Jacobian Product for Top-K normalized softmax."""
  M = topk_weights.shape[0]
  weights_fp32 = topk_weights.astype(jnp.float32)
  g_fp32 = g_weights.astype(jnp.float32)

  # Closed-form derivative of normalized softmax: y * (g - <y, g>)
  dot_yg = jnp.sum(weights_fp32 * g_fp32, axis=-1, keepdims=True)
  vbar = (weights_fp32 * (g_fp32 - dot_yg)).astype(topk_weights.dtype)

  # Scatter gradient back into full router logits dimension
  grad_logits = (
      jnp.zeros((M, num_experts), dtype=topk_weights.dtype)
      .at[jnp.arange(M)[:, None], topk_indices]
      .set(vbar)
  )
  return grad_logits


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def fused_topk_gating(
    logits: jax.Array,
    k: int = 10,
) -> Tuple[jax.Array, jax.Array]:
  """Fused MoE Top-K gating with normalized Softmax and analytical backward pass.

  Args:
    logits: Tensor of gate logits with shape (..., num_experts).
    k: Number of experts to select per token (default: 10).

  Returns:
    topk_weights: Normalized routing probabilities of shape (..., k).
    topk_indices: Top-k expert indices of shape (..., k).
  """
  orig_shape = logits.shape
  num_experts = orig_shape[-1]
  logits_2d = logits.reshape(-1, num_experts)

  if jax.default_backend() == "tpu":
    weights_2d, indices_2d = _pallas_fused_topk_gating(logits_2d, k=k)
  else:
    topk_vals, indices_2d = jax.lax.top_k(logits_2d, k=k)
    weights_2d = jax.nn.softmax(topk_vals.astype(jnp.float32), axis=-1).astype(logits.dtype)

  out_shape = (*orig_shape[:-1], k)
  return weights_2d.reshape(out_shape), indices_2d.reshape(out_shape)


def _fused_topk_gating_fwd(
    logits: jax.Array,
    k: int,
) -> Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, Tuple[int, ...]]]:
  orig_shape = logits.shape
  weights, indices = fused_topk_gating(logits, k)
  return (weights, indices), (weights, indices, orig_shape)


def _fused_topk_gating_bwd(
    k: int,
    res: Tuple[jax.Array, jax.Array, Tuple[int, ...]],
    g: Tuple[jax.Array, jax.Array],
) -> Tuple[jax.Array]:
  weights, indices, orig_shape = res
  g_weights, _ = g  # indices has no gradient

  if g_weights is None:
    return (jnp.zeros(orig_shape, dtype=weights.dtype),)

  num_experts = orig_shape[-1]
  w_2d = weights.reshape(-1, k)
  idx_2d = indices.reshape(-1, k)
  gw_2d = g_weights.reshape(-1, k)

  grad_logits_2d = _analytical_vjp(w_2d, idx_2d, gw_2d, num_experts)
  grad_logits = grad_logits_2d.reshape(orig_shape)
  return (grad_logits,)


fused_topk_gating.defvjp(_fused_topk_gating_fwd, _fused_topk_gating_bwd)
