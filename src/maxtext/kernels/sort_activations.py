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


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def route(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_custom_mosaic_kernel: bool,
) -> jax.Array:
  """Route tokens to selected experts."""
  return _route_fwd(tokens, selected_experts, use_custom_mosaic_kernel)[0]


def _route_fwd(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_custom_mosaic_kernel: bool,
) -> tuple[jax.Array, jax.Array]:
  return (
      _route_impl(tokens, selected_experts, use_custom_mosaic_kernel),
      selected_experts,
  )


def _route_bwd(
    use_custom_mosaic_kernel: bool,
    residuals: jax.Array,
    grads: jax.Array,
) -> tuple[jax.Array, None]:
  selected_experts = residuals
  return _unroute_impl(grads, selected_experts, use_custom_mosaic_kernel), None


route.defvjp(_route_fwd, _route_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def unroute(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_custom_mosaic_kernel: bool,
) -> jax.Array:
  return _unroute_fwd(tokens, selected_experts, use_custom_mosaic_kernel)[0]


def _unroute_fwd(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_custom_mosaic_kernel: bool,
) -> tuple[jax.Array, jax.Array]:
  return (
      _unroute_impl(tokens, selected_experts, use_custom_mosaic_kernel),
      selected_experts,
  )


def _unroute_bwd(use_custom_mosaic_kernel: bool, residuals: jax.Array, grads: jax.Array) -> tuple[jax.Array, None]:
  selected_experts = residuals
  return _route_impl(grads, selected_experts, use_custom_mosaic_kernel), None


unroute.defvjp(_unroute_fwd, _unroute_bwd)


def _route_impl(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_custom_mosaic_kernel: bool,
) -> jax.Array:
  """Gather `tokens` according to `selected_experts`."""
  assert (
      tokens.shape[0] == selected_experts.shape[0] and selected_experts.ndim == 2
  ), f"{tokens.shape=}, {selected_experts.shape=}"
  if use_custom_mosaic_kernel:
    raise NotImplementedError("Custom Mosaic kernel not implemented.")
  inds = jnp.argsort(jnp.ravel(selected_experts)) // selected_experts.shape[1]
  return _sort_impl(tokens, inds, use_custom_mosaic_kernel)


def _unroute_impl(
    tokens: jax.Array,
    selected_experts: jax.Array,
    use_custom_mosaic_kernel: bool,
) -> jax.Array:
  """Reverse the routing operation, restoring tokens to their original order."""
  assert tokens.shape[0] == selected_experts.shape[0] * selected_experts.shape[1] and selected_experts.ndim == 2
  inds = jnp.argsort(jnp.argsort(jnp.ravel(selected_experts)))
  return jnp.sum(
      jnp.reshape(
          _sort_impl(tokens, inds, use_custom_mosaic_kernel),
          (-1, selected_experts.shape[1]) + tokens.shape[1:],
      ),
      axis=1,
  )


def _sort_impl(tokens: jax.Array, inds: jax.Array, use_custom_mosaic_kernel: bool) -> jax.Array:
  if use_custom_mosaic_kernel:
    raise NotImplementedError("Custom Mosaic kernel not implemented.")
  else:
    return tokens[inds, ...]
