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
from maxtext.kernels import gather_reduce_sc
from maxtext.kernels.ragged.ragged_gather import ragged_gather
from maxtext.kernels.ragged.ragged_gather_reduce import ragged_gather_reduce


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def route(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
    use_ragged_sort: bool = False,
) -> jax.Array:
  """Route tokens to selected experts."""
  return _route_fwd(tokens, selected_experts, use_gather_mosaic_kernel, use_ragged_sort)[0]


def _route_fwd(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
    use_ragged_sort: bool,
) -> tuple[jax.Array, jax.Array]:
  return (
      _route_impl(tokens, selected_experts, use_gather_mosaic_kernel, use_ragged_sort),
      selected_experts,
  )


def _route_bwd(
    use_gather_mosaic_kernel: bool,
    use_ragged_sort: bool,
    residuals: jax.Array,
    grads: jax.Array,
) -> tuple[jax.Array, None]:
  selected_experts = residuals
  return (
      _unroute_impl(grads, selected_experts, use_gather_mosaic_kernel, use_ragged_sort),
      None,
  )


route.defvjp(_route_fwd, _route_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def unroute(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
    use_ragged_sort: bool = False,
) -> jax.Array:
  return _unroute_fwd(tokens, selected_experts, use_gather_mosaic_kernel, use_ragged_sort)[0]


def _unroute_fwd(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
    use_ragged_sort: bool,
) -> tuple[jax.Array, jax.Array]:
  return (
      _unroute_impl(tokens, selected_experts, use_gather_mosaic_kernel, use_ragged_sort),
      selected_experts,
  )


def _unroute_bwd(
    use_gather_mosaic_kernel: bool,
    use_ragged_sort: bool,
    residuals: jax.Array,
    grads: jax.Array,
) -> tuple[jax.Array, None]:
  selected_experts = residuals
  return (
      _route_impl(grads, selected_experts, use_gather_mosaic_kernel, use_ragged_sort),
      None,
  )


unroute.defvjp(_unroute_fwd, _unroute_bwd)


def _route_impl(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
    use_ragged_sort: bool = False,
) -> jax.Array:
  """Gather `tokens` according to `selected_experts`."""
  assert (
      tokens.shape[0] == selected_experts.shape[0] and selected_experts.ndim == 2
  ), f"{tokens.shape=}, {selected_experts.shape=}"
  inds = jnp.argsort(jnp.ravel(selected_experts)) // selected_experts.shape[1]
  if use_ragged_sort:
    # Sort all rows: start=0, end=num_rows. Falls back to dense gather when SC
    # isn't available (handled inside `ragged_gather`).
    return ragged_gather(
        tokens,
        inds,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(inds.shape[0], jnp.int32),
    )
  return _sort_impl(tokens, inds, use_gather_mosaic_kernel)


def _unroute_impl(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_gather_mosaic_kernel: bool,
    use_ragged_sort: bool = False,
) -> jax.Array:
  """Reverse the routing operation, restoring tokens to their original order."""
  assert tokens.shape[0] == selected_experts.shape[0] * selected_experts.shape[1] and selected_experts.ndim == 2
  inds = jnp.argsort(jnp.argsort(jnp.ravel(selected_experts)))
  topk = selected_experts.shape[1]
  if use_ragged_sort:
    # Fused gather + per-token sum-reduce of `topk` rows in one SC kernel.
    n = inds.shape[0]
    return ragged_gather_reduce(
        tokens,
        inds,
        topk_weights=jnp.ones((n,), dtype=jnp.float32),
        valid_rows_mask=jnp.ones((n,), dtype=jnp.bool_),
        reduce_group_size=topk,
    )
  if use_gather_mosaic_kernel:
    # The kernel currently only supports 8 experts per token.
    assert topk == 8
    kernel = functools.partial(
        gather_reduce_sc.sc_gather_reduce,
        reduce_group_size=topk,
        single_sc=True,
    )
    return kernel(tokens, inds)
  return jnp.sum(
      jnp.reshape(
          _sort_impl(tokens, inds, use_gather_mosaic_kernel),
          (-1, topk) + tokens.shape[1:],
      ),
      axis=1,
  )


def _sort_impl(tokens: jax.Array, inds: jax.Array, use_gather_mosaic_kernel: bool) -> jax.Array:
  return tokens[inds, ...]
