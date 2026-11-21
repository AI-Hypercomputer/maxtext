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
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
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


# Row chunk the SparseCore gather-reduce kernel streams per subcore (its default).
_SC_ROW_CHUNK = 512
# Largest column chunk the kernel handles per pass (its default, DeepSeek's 3584-wide hidden state).
_SC_MAX_COL_CHUNK = 3584


def _sparse_core_info():
  """The SparseCore info of the current TPU, or None (CPU, GPU, or a TPU without SparseCore)."""
  try:
    return pltpu.get_tpu_info().sparse_core
  except (ValueError, RuntimeError):
    return None


def _sc_col_chunk(width: int) -> int | None:
  """Largest multiple of 32 that divides `width` and is at most `_SC_MAX_COL_CHUNK`, or None."""
  for chunk in range(min(width, _SC_MAX_COL_CHUNK) // 32 * 32, 0, -32):
    if width % chunk == 0:
      return chunk
  return None


def _sc_topk_sum(tokens: jax.Array, inds: jax.Array, group_size: int) -> jax.Array:
  """`sum(tokens[inds].reshape(-1, group_size, D), axis=1)` in f32, returned as bf16.

  Runs on the SparseCore gather-reduce kernel when the shapes fit it (top-k dividing the SparseCore lane count,
  rows dividing evenly over every SparseCore subcore, a 32-aligned column chunk); otherwise on the TensorCore with
  the same f32 accumulation, so any model and top-k can enable it.
  """
  sc = _sparse_core_info()
  col_chunk = _sc_col_chunk(tokens.shape[1])
  fits = (
      sc is not None
      and col_chunk is not None
      and sc.num_lanes % group_size == 0
      and inds.shape[0] % (_SC_ROW_CHUNK * sc.num_cores * sc.num_subcores) == 0
  )
  if not fits:
    gathered = tokens[inds].astype(jnp.float32).reshape(-1, group_size, tokens.shape[1])
    return jnp.sum(gathered, axis=1).astype(jnp.bfloat16)
  return gather_reduce_pallas.sc_gather_reduce(
      tokens.astype(jnp.bfloat16),
      inds,
      reduce_group_size=group_size,
      single_sc=False,  # Split rows over every SparseCore of the device.
      col_chunk_size=col_chunk,
      row_chunk_size=_SC_ROW_CHUNK,
  )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def gather_topk_rows(tokens: jax.Array, sort_indices: jax.Array, group_size: int) -> jax.Array:
  """Top-k dispatch `tokens[sort_indices // group_size]` whose backward top-k sum runs on SparseCore.

  `sort_indices` is the permutation that sorts the flat (token, k) slots by
  expert, so row j of the result is
  token `sort_indices[j] // group_size`, without materializing the top-k repeat
  of `tokens`.
  """
  return tokens[sort_indices // group_size]


def _gather_topk_rows_fwd(tokens, sort_indices, group_size):
  # The empty array only carries the input dtype to the backward.
  return gather_topk_rows(tokens, sort_indices, group_size), (
      sort_indices,
      jnp.zeros((0,), tokens.dtype),
  )


def _gather_topk_rows_bwd(group_size, residuals, grads):
  sort_indices, dtype_carrier = residuals
  return (
      _sc_topk_sum(grads, jnp.argsort(sort_indices), group_size).astype(dtype_carrier.dtype),
      None,
  )


gather_topk_rows.defvjp(_gather_topk_rows_fwd, _gather_topk_rows_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def unsort_and_sum(tokens: jax.Array, sort_indices: jax.Array, group_size: int) -> jax.Array:
  """Top-k combine on SparseCore, the transpose of `gather_topk_rows`: unsort `tokens` and sum each token's rows.

  The backward is the plain gather `grads[sort_indices // group_size]` (no
  inverse sort).
  """
  return _sc_topk_sum(tokens, jnp.argsort(sort_indices), group_size).astype(tokens.dtype)


def _unsort_and_sum_fwd(tokens, sort_indices, group_size):
  return unsort_and_sum(tokens, sort_indices, group_size), (
      sort_indices,
      jnp.zeros((0,), tokens.dtype),
  )


def _unsort_and_sum_bwd(group_size, residuals, grads):
  sort_indices, dtype_carrier = residuals
  return grads[sort_indices // group_size].astype(dtype_carrier.dtype), None


unsort_and_sum.defvjp(_unsort_and_sum_fwd, _unsort_and_sum_bwd)
