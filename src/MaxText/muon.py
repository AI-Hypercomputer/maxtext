# copy from https://github.com/google-deepmind/optax/blob/5bd909532d9814667c188ac09b675183118e76eb/optax/contrib/_muon.py

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
from typing import Any, Callable, NamedTuple, Optional, Union, Sequence

import chex
import jax
import jax.numpy as jnp

from optax._src import alias
from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform
from optax._src import utils
from optax.transforms import _masking
import optax.tree

ReshapeFn = Callable[[jax.Array], jax.Array]


def debug_sharding(array, prefix=""):
  global_shape = array.shape
  jax.debug.inspect_array_sharding(
      array,
      callback=lambda sharding_obj: print(
          prefix + f"\tGlobal Shape: {global_shape}\n"
          f"\tLocal Shape: {sharding_obj.shard_shape(global_shape)}\n"
          f"\tSharding Object: {sharding_obj}\n"
      ),
  )


class MuonDimensionNumbers(NamedTuple):
  """Specification for which weight axes participate in matrix projection.

  Muon defines an orthogonalization for 2D matrix weights for matrix-vector
  products:

  .. math::
    x W = y

  where the first matrix dimension is the reduction axis and the second matrix
  dimension is the output axis. Thus, the default spec consists of 0 and 1
  reduction and output axes respectively.

  .. warning::
    The batch axes are implicit, all axes not specified as reduction or output
    axes are considered batch axes and will be considered independently in the
    orthogonalization (via jax.vmap).
  """
  reduction_axis: Sequence[int] | int = 0
  output_axis: Sequence[int] | int = 1

WeightDimNumOrFn = MuonDimensionNumbers | base.Params | Callable[
    [base.Params], base.Params | None]


_is_weight_dim_nums = lambda x: isinstance(x, MuonDimensionNumbers)


def _normalize_axes(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[
    tuple[int, ...], tuple[int, ...]]:
  """Normalize axes in dimension numbers to two tuples of non-negative ints."""
  if isinstance(dim_nums.reduction_axis, int):
    dim_nums = dim_nums._replace(reduction_axis=(dim_nums.reduction_axis,))
  reduction_axes = tuple(ax % x.ndim for ax in dim_nums.reduction_axis)

  if isinstance(dim_nums.output_axis, int):
    dim_nums = dim_nums._replace(output_axis=(dim_nums.output_axis,))
  output_axes = tuple(ax % x.ndim for ax in dim_nums.output_axis)
  return reduction_axes, output_axes


def _compute_muon_reshape(x: jax.Array, dim_nums: MuonDimensionNumbers
                          ) -> tuple[ReshapeFn, ReshapeFn]:
  """Compute the reshape and inverse functions for an array from a spec."""
  if x.ndim < 2:
    raise ValueError('Muon optimized parameters must have rank >= 2, got'
                     f' {x.ndim=}')
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  if set(reduction_axes) & set(output_axes):
    raise ValueError('Normalized reduction axes and output axes must be'
                     f' disjoint, got {reduction_axes} and {output_axes}.'
                     f' Originally {dim_nums=} and {x.shape=}')
  batch_axes = tuple(sorted(set(range(x.ndim)) - set(reduction_axes)
                            - set(output_axes)))
  transpose = batch_axes + reduction_axes + output_axes
  inv_transpose = tuple(sorted(range(x.ndim), key=lambda i: transpose[i]))
  axes2shape = lambda axes: tuple(x.shape[ax] for ax in axes)
  # Reshape to (batch, reduction, output) to match the (reduction, output)
  # structure of the original muon for 2D weights.
  flat_shape = (
      math.prod(axes2shape(batch_axes)),
      math.prod(axes2shape(reduction_axes)),
      math.prod(axes2shape(output_axes)),
  )
  unflat_shape = (
      axes2shape(batch_axes)
      + axes2shape(reduction_axes)
      + axes2shape(output_axes)
  )
  reshape_fn = lambda x: x.transpose(transpose).reshape(flat_shape)
  inverse_fn = lambda x: x.reshape(unflat_shape).transpose(inv_transpose)
  return reshape_fn, inverse_fn


def _shape_factor(x: jax.Array, dim_nums: MuonDimensionNumbers) -> float:
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  return math.prod(x.shape[ax] for ax in output_axes) / math.prod(
      x.shape[ax] for ax in reduction_axes)


def _newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
  # Implements Newton-Schulz step f(X) = c_0 X + c_1 (XX^T)X + c_2 (XX^T)^2X,
  # with quintic form f(X) = c_0 X + (c_1 A + c_2 AA)X, where A = XX^T.
  # The NS step has the property f(X) = f(X^T)^T. That is, we can get equivalent
  # result by tranposing input and output. In particular, we may tranpose X
  # when rows > cols for effciency.
  a = x @ x.T
  b = coeffs[1] * a + coeffs[2] * a @ a
  out = coeffs[0] * x + b @ x
  rms_adj = 0.2 * math.sqrt(max(x.shape))
  return out * rms_adj


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
    dimension_numbers: MuonDimensionNumbers | None = None,
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
    dimension_numbers: Optional spec for reshaping a tensor before and after the
      orthogonalization, to support non-2D parameters.

  Returns:
    The orthogonalized matrix.
  """
  if x.ndim != 2 and not isinstance(dimension_numbers, MuonDimensionNumbers):
    raise ValueError(
        f'Input must have shape (m, n) or weight dimension numbers must be'
        f' provided. Got shape={x.shape} and {dimension_numbers=}.')
  if x.ndim == 2:
    dimension_numbers = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError('Newton-Schulz coefficients must have shape (3,) or'
                     f' (n, 3), got {ns_coeffs.shape}')

  def _orthogonalize(x):
    transposed = False
    if x.shape[0] > x.shape[1]:
      x = x.T
      transposed = True

    x /= jnp.linalg.norm(x) + eps  # Ensure spectral norm is at most 1
    ns_coeffs_ = ns_coeffs.astype(x.dtype)
    if ns_coeffs_.ndim == 1:
      x = jax.lax.fori_loop(
          0, ns_steps, lambda _, x: _newton_schulz_iterator(x, ns_coeffs_), x,
          unroll=True)  # Unroll to ensure efficient composition with jax.vmap.
    else:
      x, _ = jax.lax.scan(
          lambda x, abc: (_newton_schulz_iterator(x, abc), None), x, ns_coeffs_
      )
    if transposed:
      x = x.T
    return x

  reshape_fn, inverse_fn = _compute_muon_reshape(x, dimension_numbers)
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
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
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
    weight_dimension_numbers: An optional tree with the same structure as the
      params of `MuonDimensionNumbers`s, specifying how to reshape the
      parameters before and after the orthogonalization OR a callble returning
      such a tree. None implies that all parameters are 2D matrices.

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
    debug_sharding(updates["params"]["decoder"]["dense_layers"]["mlp"]["wi_0"]["kernel"])
    del params
    # TODO(rdyro): extend to _masking._mask_callable
    if callable(weight_dimension_numbers):
      # Populate weight_dim_nums if it's a callable. Use updates instead of
      # actual params since only shapes matter and params may not be provided.
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

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
    updates = jax.tree.map(
        lambda x, dim_num: orthogonalize_via_newton_schulz(
            x, state.ns_coeffs, ns_steps, eps, dim_num),
        mu_hat, resolved_weight_dim_nums, is_leaf=_is_weight_dim_nums)
    if adaptive:
      # Scale the orthogonalized updates by the dual norm of the original
      # updates. See https://arxiv.org/abs/2409.20325 for the derivation.
      updates = jax.tree.map(
          lambda x, y: jnp.sum(x * y) * y, mu_hat, updates
      )
    factors = jax.tree.map(_shape_factor, updates, resolved_weight_dim_nums,
                           is_leaf=_is_weight_dim_nums)
    updates = jax.tree.map(
        lambda x, factor: jnp.sqrt(jnp.maximum(1, factor)) * x,
        updates, factors
    )
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
    weight_decay_mask: Optional[
        Union[Any, Callable[[base.Params], Any]]
    ] = None,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_eps_root: float = 0.0,
    adam_weight_decay: float = 0.0,
    muon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
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
    muon_weight_dimension_numbers: An optional tree of `MuonDimensionNumbers`s,
      specifying how to reshape the parameters for orthogonalization otherwise
      muon parameters are assumed to be 2D matrices. A `None` value indicates
      that the parameter is not a muon parameter and will be optimized with
      Adam. A callable takes as input the params and returns a possibly masked
      pytree of specs, similar to `weight_decay_mask`. If not provided, muon is
      applied to all 2D parameters.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  # None at root indicates the default 2D rule.
  if muon_weight_dimension_numbers is None:
    param_labels = lambda params: jax.tree.map(
        lambda x: 'muon' if x.ndim == 2 else 'adam', params
    )
    muon_weight_dimension_numbers = MuonDimensionNumbers()
  else:
    def param_labels(params):
      dim_nums = (muon_weight_dimension_numbers(params)
                  if callable(muon_weight_dimension_numbers)
                  else muon_weight_dimension_numbers)
      populate_subtree_ = lambda dim_num, x: jax.tree.map(
          lambda y: 'muon' if dim_num is not None else 'adam', x)
      # Dimension numbers come first since they can be a prefix mask.
      return jax.tree.map(populate_subtree_, dim_nums, params,
                          is_leaf=lambda x: x is None or _is_weight_dim_nums(x))

  # We need to normalize the dimension numbers because they have to match the
  # tree structure of the masked muon state tree (see `combine.partition`).
  def muon_weight_dim_nums_fn(params):
    # if muon_weight_dimension_numbers is None:
    #   return None
    # Normalize the dimension numbers for `combine.partition`.
    # Insert MaskedNode() where muon state will be masked out.
    dim_nums = (muon_weight_dimension_numbers(params)
                if callable(muon_weight_dimension_numbers)
                else muon_weight_dimension_numbers)
    mask = jax.tree.map(lambda label: label == 'muon', param_labels(params))
    is_leaf = lambda x: (x is None or _is_weight_dim_nums(x)
                         or isinstance(x, _masking.MaskedNode))
    populate_subtree_ = lambda dim_nums, submask: jax.tree.map(
        lambda m: dim_nums if m else _masking.MaskedNode(), submask)
    return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)

  return combine.partition(
      transforms={
          'muon': combine.chain(
              scale_by_muon(
                  ns_coeffs=ns_coeffs,
                  ns_steps=ns_steps,
                  beta=beta,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  adaptive=adaptive,
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
              ),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          'adam': alias.adamw(
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
