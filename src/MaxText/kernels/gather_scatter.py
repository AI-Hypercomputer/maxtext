# Copyright 2023–2025 Google LLC
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

"""Ops for gather and scatter-add operations related mixture-of-experts."""

import jax
import jax.numpy as jnp


@jax.custom_vjp
def gather(x: jax.Array, expert_assignments: jax.Array) -> jax.Array:
  """Gathers rows of `x` according `expert_assignments`.

  Args:
    x: `(m, d)` array.
    expert_assignments: `(m, n)` array of non-negative integers that correspond
      to the `n` experts that each of `m` tokens is assigned to. Usually, the
      values of this array are bounded between `[0, num_experts)`.

  Returns:
    `(m * n, d)` array of gathered rows of `x` where the rows of `x` are
    duplicated by a factor of `n` and grouped by ascending expert id order.
  """
  return _gather_fwd(x, expert_assignments)[0]


@jax.custom_vjp
def scatter_add(x: jax.Array, expert_assignments: jax.Array) -> jax.Array:
  """Scatter-adds rows of `x` according to the expert assignments.

  Args:
    x: `(m * n, d)` array.
    expert_assignments: `(m, n)` array of non-negative integers that correspond
      to the `n` experts that each of `m` tokens is assigned to. Usually, the
      values of this array are bounded between `[0, num_experts)`.

  Returns:
    `(m, d)` array of gathered rows of `x` where the rows of `x` are
    duplicated by a factor of `n` and grouped by ascending expert id order.
  """
  return _scatter_add_fwd(x, expert_assignments)[0]


def _gather_impl(x: jax.Array, gather_inds: jax.Array) -> jax.Array:
  return x[gather_inds, :]


def _scatter_add_impl(x: jax.Array, scatter_inds: jax.Array) -> jax.Array:
  return jnp.sum(
      jnp.reshape(
          x[jnp.ravel(scatter_inds), :],
          scatter_inds.shape + (x.shape[-1],),
      ),
      axis=1,
  )


def _gather_fwd(
    x: jax.Array, expert_assignments: jax.Array
) -> tuple[jax.Array, jax.Array]:
  gather_inds, scatter_inds = gather_scatter_inds(expert_assignments)
  return _gather_impl(x, gather_inds), scatter_inds


def _scatter_add_fwd(
    x: jax.Array, expert_assignments: jax.Array
) -> tuple[jax.Array, jax.Array]:
  gather_inds, scatter_inds = gather_scatter_inds(expert_assignments)
  return _scatter_add_impl(x, scatter_inds), gather_inds


def _gather_bwd(res: jax.Array, grad: jax.Array) -> tuple[jax.Array, None]:
  scatter_inds = res
  return _scatter_add_impl(grad, scatter_inds), None


def _scatter_add_bwd(res: jax.Array, grad: jax.Array) -> tuple[jax.Array, None]:
  gather_inds = res
  return _gather_impl(grad, gather_inds), None


gather.defvjp(_gather_fwd, _gather_bwd)
scatter_add.defvjp(_scatter_add_fwd, _scatter_add_bwd)


def gather_scatter_inds(
    expert_assignments: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Indexing arrays for gather and scatter-add operations.

  Example:
    # For a system with 4 experts, 3 tokens, and 2 experts per token, we might
    # have expert assignments as follows:
    expert_assignments = [
        [0, 1],  # token 0 is assigned to experts 0 & 1.
        [0, 2],  # token 1 is assigned to experts 0 & 2.
        [1, 3],  # token 2 is assigned to experts 1 & 3.
    ]

    # A valid `gather_inds` array would need group tokens together based on
    # expert assignment. For example:
    #  - expert 0: token 0, token 1.
    #  - expert 1: token 0, token 2.
    #  - expert 2: token 1.
    #  - expert 3: token 2.
    #
    # This can be accomplished with the following `gather_inds`:
    #
    gather_inds = [0, 1, 0, 2, 1, 2]

    # A valid `scatter_inds` must scatter-add back to the original token order
    # by taking the values from the following positions and mapping them back to
    # the following original tokens:
    # - token 0: values from indices 0 and 2.
    # - token 1: values from indices 1 and 4.
    # - token 2: values from indices 3 and 5.
    #
    # This can be accomplished with the following `scatter_inds`:
    #
    scatter_inds = [
        [0, 2],
        [1, 4],
        [3, 5],
    ]

  Args:
    expert_assignments: `(m, n)` array of values within `[0, num_experts)`.

  Returns:
    gather_inds: `(m * n,)` array of integers with values within `[0, m)` that
      duplicates and groups token by ascending expert id order via
      `x[gather_inds, :]`.
    scatter_inds: `(m, n)` array of integers with values within `[0, m * n)`
      that enables the scatter-add operation which returns the processed
      tokens for each expert via
      `jnp.sum(jnp.reshape(x[jnp.ravel(scatter_inds), :], (m, n, -1)), axis=1)`
  """
  m, n = expert_assignments.shape
  gather_inds = jnp.argsort(jnp.ravel(expert_assignments)) // n
  scatter_inds = jnp.sort(jnp.reshape(jnp.argsort(gather_inds), (m, n)), axis=1)
  return gather_inds, scatter_inds
