# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Muon.

Implementation of the
[Muon optimizer](https://github.com/KellerJordan/modded-nanogpt)
by Keller Jordan
"""


import math
from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
import optax.tree

ReshapeFn = Callable[[jax.Array], jax.Array]


class MuonWeightSpec(NamedTuple):
  reduction_axes: tuple[int, ...] | int
  output_axes: tuple[int, ...] | int


is_weight_spec = lambda x: isinstance(x, MuonWeightSpec)


def _normalize_axes(x: jax.Array, spec: MuonWeightSpec) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Normalize the axes in a muon spec to two tuples of non-negative ints."""
  reduction_axes = spec.reduction_axes
  if not isinstance(spec.reduction_axes, tuple):
    reduction_axes = spec._replace(reduction_axes=(spec.reduction_axes,))
  reduction_axes = tuple(ax % x.ndim for ax in reduction_axes)

  output_axes = spec.output_axes
  if not isinstance(spec.output_axes, tuple):
    output_axes = spec._replace(output_axes=(spec.output_axes,))
  output_axes = tuple(ax % x.ndim for ax in output_axes)
  return reduction_axes, output_axes


def _compute_muon_reshape(x: jax.Array, spec: MuonWeightSpec) -> tuple[ReshapeFn, ReshapeFn]:
  """Compute the reshape and inverse functions for an array from a spec."""
  if spec is None:
    return x
  reduction_axes, output_axes = _normalize_axes(x, spec)
  if set(reduction_axes) & set(output_axes):
    raise ValueError("Reduction axes and output axes must be disjoint, got " f"{reduction_axes} and {output_axes}")
  batch_axes = tuple(sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes)))
  transpose = batch_axes + output_axes + reduction_axes
  inv_transpose = tuple(sorted(range(x.ndim), key=lambda i: transpose[i]))
  flat_shape = tuple(map(math.prod, [batch_axes, reduction_axes, output_axes]))

  reshape_fn = lambda x: x.transpose(transpose).reshape(flat_shape)
  inverse_fn = lambda x: x.reshape(batch_axes + output_axes + reduction_axes).transpose(inv_transpose)
  return reshape_fn, inverse_fn


def _shape_factor(x: jax.Array, spec: MuonWeightSpec) -> float:
  reduction_axes, output_axes = _normalize_axes(x, spec)
  return math.prod(x.shape[ax] for ax in output_axes) / math.prod(x.shape[ax] for ax in reduction_axes)


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
    spec: MuonWeightSpec | None = None,
) -> jax.Array:
  r"""Orthogonalize via Newton-Schulz iteration.

  We opt to use a quintic iteration whose coefficients are selected to maximize
  the slope at zero. For the purpose of minimizing steps, it turns out to be
  empirically effective to keep increasing the slope at zero even beyond the
  point where the iteration no longer converges all the way to one everywhere
  on the interval. This iteration therefore does not produce UV^T but rather
  something like US'V^T where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5),
  which turns out not to hurt model performance at all relative to UV^T, where
  USV^T = G is the SVD.

  Args:
    x: A matrix to orthogonalize.
    ns_coeffs: Coefficients for the Newton-schulz iterators.
      Must have shape (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a 2D array.
    eps: Term added to denominators to improve numerical stability.
    spec: Optional spec for reshaping the matrix before and after the
      orthogonalization. Allows supporting non-2D parameters.

  Returns:
    The orthogonalized matrix.
  """
  if x.ndim != 2 and spec is None:
    raise ValueError(f"Input must have shape (m, n), got {x.shape} or the spec" " must be provided.")
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError(
        'Newton-Schulz coefficients must have shape (3,) or (n, 3), '
        f'got {ns_coeffs.shape}'
    )

  def _orthogonalize(x):
    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
      a = x @ x.T
      b = coeffs[1] * a + coeffs[2] * a @ a
      return coeffs[0] * x + b @ x

    transposed = False
    if x.shape[0] > x.shape[1]:
      x = x.T
      transposed = True

    x /= jnp.linalg.norm(x) + eps  # Ensure spectral norm is at most 1
    ns_coeffs_ = ns_coeffs.astype(x.dtype)
    if ns_coeffs_.ndim == 1:
      x = jax.lax.fori_loop(0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs_), x)
    else:
      x, _ = jax.lax.scan(lambda x, abc: (newton_schulz_iterator(x, abc), None), x, ns_coeffs_)
    if transposed:
      x = x.T

    return x

  if spec is None:
    return _orthogonalize(x)
  else:
    reshape_fn, inverse_fn = _compute_muon_reshape(x, spec)
    return inverse_fn(jax.vmap(_orthogonalize)(reshape_fn(x)))


class MuonState(NamedTuple):
  """State for the Muon algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  ns_coeffs: chex.Array  # shape=(), dtype=jnp.int32.


def scale_by_muon(
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    weight_specs: base.Params | None = None,  # a tree of MuonWeightSpec
) -> base.GradientTransformation:
  r"""Rescale updates according to the Muon algorithm.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Args:
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to denominators to improve numerical stability.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    weight_specs: Optional tree of `MuonWeightSpec`s, specifying how to reshape
      the parameters before and after the orthogonalization.

  Returns:
    A `GradientTransformation` object.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
    ns_coeffs_ = jnp.asarray(ns_coeffs)
    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(
          f'ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}'
      )
    return MuonState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        ns_coeffs=ns_coeffs_,
    )

  def update_fn(updates, state, params=None):
    del params
    mu = optax.tree.update_moment(updates, state.mu, beta, 1)
    count_inc = numerics.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: beta * m + (1 - beta) * g,
          optax.tree.bias_correction(
              mu, beta, numerics.safe_increment(count_inc)
          ),
          optax.tree.bias_correction(updates, beta, count_inc),
      )
    else:
      mu_hat = optax.tree.bias_correction(mu, beta, count_inc)
    # Apply Newton-schulz orthogonalization.
    if weight_specs is not None:
      updates = jax.tree.map(
          lambda x, spec: orthogonalize_via_newton_schulz(x, state.ns_coeffs, ns_steps, eps, spec), mu_hat, weight_specs
      )
    else:
      updates = jax.tree.map(lambda x: orthogonalize_via_newton_schulz(x, state.ns_coeffs, ns_steps, eps), mu_hat)
    if adaptive:
      # Scale the orthogonalized updates by the dual norm of the original
      # updates. See https://arxiv.org/abs/2409.20325 for the derivation.
      updates = jax.tree.map(
          # lambda x, y: jnp.einsum('ij,ij,ab->ab', x, y, y), mu_hat, updates
          lambda x, y: jnp.sum(x * y) * y,
          mu_hat,
          updates,
      )
    if weight_specs is not None:
      factors = jax.tree.map(_shape_factor, updates, weight_specs, is_leaf=is_weight_spec)
    else:
      factors = jax.tree.map(lambda x: x.shape[-1] / x.shape[-2], updates)
    updates = jax.tree.map(lambda x, factor: jnp.sqrt(jnp.maximum(1, factor)) * x, updates, factors)
    mu = optax.tree.cast(mu, mu_dtype)
    return updates, MuonState(
        count=count_inc,
        mu=mu,
        ns_coeffs=state.ns_coeffs,
    )
  return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate: base.ScalarOrSchedule,
    ns_coeffs: Union[
        tuple[float, float, float],
        tuple[tuple[float, float, float], ...],
    ] = (3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
    muon_weight_mask: base.Params | None = None,
    muon_weight_specs: base.Params | None = None,
) -> base.GradientTransformation:
  r"""Muon: Momentum Orthogonalized by Newton-schulz.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Note that Muon is currently only defined for 2D parameters, i.e. matrices.
  This is because the Newton-Schulz iterator expects a matrix as input.
  The non-2D parameters are instead passed through an Adam optimizer.

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    ns_coeffs: Coefficients for the Newton-schulz method.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to the denominator to improve numerical stability.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent
      with other frameworks such as PyTorch, but different from
      (Loshchilov et al, 2019) where the weight decay is only multiplied with
      the "schedule multiplier", but not the base learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    muon_weight_mask: A True/False mask indicating which parameters to
      scale by Muon (vs Adam) or a callable returning such a mask given the
      params. If 'None', all params with ndim == 2 are scaled by Muon.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  if muon_weight_mask is None:
    param_labels = lambda params: jax.tree.map(lambda x: "muon" if x.ndim == 2 else "adam", params)
  else:
    # mask comes first since it can be a prefix tree
    # no-op map over parameters to ensure same structure as params
    param_labels = lambda params: jax.tree.map(lambda m, x: "muon" if m else "adam", muon_weight_mask, params)
  return combine.partition(
      transforms={
          "muon": combine.chain(
              scale_by_muon(
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  beta=beta,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  adaptive=adaptive,
                  weight_specs=muon_weight_specs,
              ),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          "adam": alias.adamw(
              learning_rate=learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=eps,
              eps_root=adam_eps_root,
              weight_decay=adam_weight_decay,
              mu_dtype=mu_dtype,
              nesterov=nesterov,
          ),
      },
      param_labels=param_labels,
  )
