# Copyright 2023–2026 Google LLC
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

"""Token sorting for MoE layers."""

import functools

import jax
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from maxtext.kernels import gather_reduce_pallas
from maxtext.kernels import gather_reduce_sc


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def route(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
) -> jax.Array:
  """Route tokens to selected experts."""
  return _route_fwd(tokens, selected_experts, use_gather_mosaic_kernel)[0]


def _route_fwd(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
) -> tuple[jax.Array, jax.Array]:
  return (
      _route_impl(tokens, selected_experts, use_gather_mosaic_kernel),
      selected_experts,
  )


def _route_bwd(
    use_gather_mosaic_kernel: bool,
    residuals: jax.Array,
    grads: jax.Array,
) -> tuple[jax.Array, None]:
  selected_experts = residuals
  return _unroute_impl(grads, selected_experts, use_gather_mosaic_kernel), None


route.defvjp(_route_fwd, _route_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def unroute(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
) -> jax.Array:
  return _unroute_fwd(tokens, selected_experts, use_gather_mosaic_kernel)[0]


def _unroute_fwd(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
) -> tuple[jax.Array, jax.Array]:
  return (
      _unroute_impl(tokens, selected_experts, use_gather_mosaic_kernel),
      selected_experts,
  )


def _unroute_bwd(use_gather_mosaic_kernel: bool, residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
  selected_experts = residuals
  return _route_impl(grads, selected_experts, use_gather_mosaic_kernel), None


unroute.defvjp(_unroute_fwd, _unroute_bwd)


def _route_impl(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
) -> jax.Array:
  """Gather `tokens` according to `selected_experts`."""
  assert (
      tokens.shape[0] == selected_experts.shape[0] and selected_experts.ndim == 2
  ), f"{tokens.shape=}, {selected_experts.shape=}"
  inds = jnp.argsort(jnp.ravel(selected_experts)) // selected_experts.shape[1]
  return _sort_impl(tokens, inds, use_gather_mosaic_kernel)


def _unroute_impl(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
) -> jax.Array:
  """Reverse the routing operation, restoring tokens to their original order."""
  assert tokens.shape[0] == selected_experts.shape[0] * selected_experts.shape[1] and selected_experts.ndim == 2
  inds = jnp.argsort(jnp.argsort(jnp.ravel(selected_experts)))
  if use_gather_mosaic_kernel:
    # The kernel currently only supports 8 experts per token.
    assert selected_experts.shape[1] == 8
    kernel = functools.partial(
        gather_reduce_sc.sc_gather_reduce,
        reduce_group_size=selected_experts.shape[1],
        single_sc=True,
    )
    return kernel(tokens, inds)
  num_tokens, k = selected_experts.shape
  inds = jnp.reshape(inds, (num_tokens, k)).T.reshape(-1)
  return jnp.sum(
      jnp.reshape(
          _sort_impl(tokens, inds, use_gather_mosaic_kernel),
          (k, num_tokens) + tokens.shape[1:],
      ),
      axis=0,
  )


def _sort_impl(tokens: jax.Array, inds: jax.Array, use_gather_mosaic_kernel: bool) -> jax.Array:
  return tokens[inds, ...]


def _topk_sum(rows: jax.Array, inds: jax.Array, k: int) -> jax.Array:
  """`rows[inds]` summed over each run of `k` indices, accumulated in f32.

  Runs on the SparseCore gather-reduce kernel (split over every SparseCore) when the shapes fit it: bf16 rows, `2 * k`
  dividing the SparseCore lane count (the kernel writes bf16 output rows in pairs), the indices dividing evenly over
  its subcores, and a 32-aligned column chunk of at most 3584 dividing the width. Otherwise, and on chips without
  SparseCore, the sum runs on the TensorCore.
  """
  try:
    sc = pltpu.get_tpu_info().sparse_core
  except (ValueError, RuntimeError):  # Not a TPU.
    sc = None
  width = rows.shape[1]
  col_chunk = next((c for c in range(min(width, 3584) // 32 * 32, 0, -32) if width % c == 0), None)
  row_chunk = 512  # The kernel's default rows per subcore step.
  if (
      sc is not None
      and col_chunk is not None
      and rows.dtype == jnp.bfloat16
      and sc.num_lanes % (2 * k) == 0
      and inds.shape[0] % (row_chunk * sc.num_cores * sc.num_subcores) == 0
  ):
    return gather_reduce_pallas.sc_gather_reduce(
        rows, inds, reduce_group_size=k, col_chunk_size=col_chunk, row_chunk_size=row_chunk
    )
  return rows[inds].astype(jnp.float32).reshape(-1, k, width).sum(axis=1).astype(rows.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def gather_topk_rows(tokens: jax.Array, sort_indices: jax.Array, k: int) -> jax.Array:
  """Top-k dispatch without the k-fold repeat: row j is token `sort_indices[j] // k`.

  `sort_indices` sorts the flat (token, k) slots by expert. The VJP is `unsort_and_sum`.
  """
  return tokens[sort_indices // k]


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def unsort_and_sum(rows: jax.Array, sort_indices: jax.Array, k: int) -> jax.Array:
  """Top-k combine, the transpose of `gather_topk_rows`: unsort `rows` and sum each token's `k` rows.

  The sum runs in `_topk_sum` (SparseCore when the shapes fit). The VJP is `gather_topk_rows`.
  """
  return _topk_sum(rows, jnp.argsort(sort_indices), k)


gather_topk_rows.defvjp(
    lambda tokens, sort_indices, k: (gather_topk_rows(tokens, sort_indices, k), sort_indices),
    lambda k, sort_indices, grads: (unsort_and_sum(grads, sort_indices, k), None),
)
unsort_and_sum.defvjp(
    lambda rows, sort_indices, k: (unsort_and_sum(rows, sort_indices, k), sort_indices),
    lambda k, sort_indices, grads: (gather_topk_rows(grads, sort_indices, k), None),
)
