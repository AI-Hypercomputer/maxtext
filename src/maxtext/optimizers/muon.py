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

"""Sharding-aware Muon implementation, branched from optax.contrib._muon."""

import functools
import math
from typing import Callable, Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from maxtext.optimizers import reshape_utils
import optax

_PRECONDITIONINGS = ["frobenius", "spectral", "aol", "schatten"]
Preconditioning = Literal["frobenius", "spectral", "aol", "schatten"]

CoeffsType = jax.typing.ArrayLike | Sequence[jax.typing.ArrayLike]
_DEFAULT_NS_COEFFS = (3.4445, -4.7750, 2.0315)
_DION_NS_COEFFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]
_NS_COEFFS_PRESET_DICT = {
    "standard": _DEFAULT_NS_COEFFS,
    "dion": _DION_NS_COEFFS,
}


def _get_xxt_out_sharding(
    flat_sharding: jax.sharding.NamedSharding | None,
) -> jax.sharding.NamedSharding | None:
  """Constructs XX^T output sharding: [*batch_axes, row, col] -> [*batch_axes, None, row]."""
  if flat_sharding is None:
    return None
  *batch_spec, row_spec, _ = flat_sharding.spec
  out_spec = jax.sharding.PartitionSpec(*batch_spec, None, row_spec)
  return jax.sharding.NamedSharding(flat_sharding.mesh, out_spec)


def _get_preferred_element_type(dtype: jnp.dtype) -> jnp.dtype:
  if jnp.issubdtype(dtype, jnp.complexfloating):
    return jnp.promote_types(dtype, jnp.complex64)
  return jnp.promote_types(dtype, jnp.float32)


def xxt(
    x: jax.Array,
    flat_sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Computes X @ X.T.conj() while preserving sharding along the last axis."""
  out_sharding = _get_xxt_out_sharding(flat_sharding)
  pref_type = _get_preferred_element_type(x.dtype)
  if reshape_utils.is_explicit_axes(flat_sharding):
    return jnp.einsum(
        "...mk,...nk->...mn",
        x,
        x.conj(),
        out_sharding=out_sharding,
        preferred_element_type=pref_type,
    ).astype(x.dtype)
  res = jnp.einsum(
      "...mk,...nk->...mn",
      x,
      x.conj(),
      preferred_element_type=pref_type,
  ).astype(x.dtype)
  if out_sharding is not None:
    res = jax.lax.with_sharding_constraint(res, out_sharding)
  return res


def b_times_x(
    b: jax.Array,
    x: jax.Array,
    flat_sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Computes b @ x while outputting to flat_sharding."""
  pref_type = _get_preferred_element_type(x.dtype)
  if reshape_utils.is_explicit_axes(flat_sharding):
    return jnp.einsum(
        "...mk,...kn->...mn",
        b,
        x,
        out_sharding=flat_sharding,
        preferred_element_type=pref_type,
    ).astype(x.dtype)
  res = jnp.einsum(
      "...mk,...kn->...mn",
      b,
      x,
      preferred_element_type=pref_type,
  ).astype(x.dtype)
  if flat_sharding is not None:
    res = jax.lax.with_sharding_constraint(res, flat_sharding)
  return res


def _aol_first_newton_schulz_iteration(
    x: jax.Array,
    coeffs: jax.Array,
    flat_sharding: jax.sharding.NamedSharding | None = None,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  """'Almost Orthogonal Layer' Preconditioning with Newton-Schulz iteration."""
  # Implements the first Newton-Schulz step with AOL preconditioning
  # which allows for better orthogonalization performance.
  a = xxt(x, flat_sharding=flat_sharding)
  rescaling = jnp.clip(jnp.abs(a).sum(axis=-1, keepdims=True), min=eps)
  s = jax.lax.rsqrt(rescaling)
  x, a = x * s, a * s * jnp.swapaxes(s, -1, -2)
  b = coeffs[1] * a + coeffs[2] * xxt(a, flat_sharding=flat_sharding)
  return coeffs[0] * x + b_times_x(b, x, flat_sharding=flat_sharding)


def _schatten_first_newton_schulz_iteration(
    x: jax.Array,
    coeffs: jax.Array,
    flat_sharding: jax.sharding.NamedSharding | None = None,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  """Schatten-4 Preconditioning with Newton-Schulz iteration."""
  # Implements the first Newton-Schulz step with Schatten-4 norm
  # preconditioning which allows for better orthogonalization performance.
  a = xxt(x, flat_sharding=flat_sharding)
  rescaling = jnp.clip(jnp.linalg.norm(a, axis=(-2, -1), keepdims=True), min=eps)
  s = jax.lax.rsqrt(rescaling)
  x, a = x * s, a * (s**2)
  b = coeffs[1] * a + coeffs[2] * xxt(a, flat_sharding=flat_sharding)
  return coeffs[0] * x + b_times_x(b, x, flat_sharding=flat_sharding)


def _base_newton_schulz_iteration(
    x: jax.Array,
    coeffs: jax.Array,
    flat_sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Base quintic Newton-Schulz iteration."""
  # Implements Newton-Schulz step f(X) = c_0 X + c_1 (XX^T)X + c_2 (XX^T)^2X,
  # with quintic form f(X) = c_0 X + (c_1 A + c_2 AA)X, where A = XX^T.
  # The NS step has the property f(X) = f(X^T)^T. That is, we can get equivalent
  # result by transposing input and output. In particular, we may transpose X
  # when rows > cols for efficiency.
  a = xxt(x, flat_sharding=flat_sharding)
  b = coeffs[1] * a + coeffs[2] * xxt(a, flat_sharding=flat_sharding)
  res = coeffs[0] * x + b_times_x(b, x, flat_sharding=flat_sharding)
  if flat_sharding is not None:
    res = reshape_utils.reshard_or_constrain(res, flat_sharding)
  return res


def _aol_ns_iterator(
    i: int,
    x: jax.Array,
    coeffs: jax.Array,
    flat_sharding: jax.sharding.NamedSharding | None = None,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  # Modified first step using AOL rescaling
  return jax.lax.cond(
      i == 0,
      lambda x: _aol_first_newton_schulz_iteration(x, coeffs, flat_sharding=flat_sharding, eps=eps),
      lambda x: _base_newton_schulz_iteration(x, coeffs, flat_sharding=flat_sharding),
      x,
  )


def _schatten_ns_iterator(
    i: int,
    x: jax.Array,
    coeffs: jax.Array,
    flat_sharding: jax.sharding.NamedSharding | None = None,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  # Modified first step using Schatten-4 norm rescaling
  return jax.lax.cond(
      i == 0,
      lambda x: _schatten_first_newton_schulz_iteration(x, coeffs, flat_sharding=flat_sharding, eps=eps),
      lambda x: _base_newton_schulz_iteration(x, coeffs, flat_sharding=flat_sharding),
      x,
  )


def _base_ns_iterator(
    i: int,
    x: jax.Array,
    coeffs: jax.Array,
    flat_sharding: jax.sharding.NamedSharding | None = None,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  del i, eps
  return _base_newton_schulz_iteration(x, coeffs, flat_sharding=flat_sharding)


NS_ITERATORS = {
    "frobenius": _base_ns_iterator,
    "spectral": _base_ns_iterator,
    "aol": _aol_ns_iterator,
    "schatten": _schatten_ns_iterator,
}


class ShardedMuonDimensionNumbers(NamedTuple):
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

  Attributes:
    reduction_axis: Axes to contract over.
    output_axis: Output axes of the matrix projection.
    sharding: Optional NamedSharding specification for this leaf tensor.
  """

  reduction_axis: Sequence[int] | int = 0
  output_axis: Sequence[int] | int = 1
  sharding: jax.sharding.NamedSharding | None = None


def _normalize_axes(x: jax.Array, dim_nums: ShardedMuonDimensionNumbers) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Normalize axes in dimension numbers to non-negative int tuples."""
  reduction_axes = (
      (dim_nums.reduction_axis,) if isinstance(dim_nums.reduction_axis, int) else tuple(dim_nums.reduction_axis)
  )
  reduction_axes = tuple(ax % x.ndim for ax in reduction_axes)

  output_axes = (dim_nums.output_axis,) if isinstance(dim_nums.output_axis, int) else tuple(dim_nums.output_axis)
  output_axes = tuple(ax % x.ndim for ax in output_axes)

  if len(set(reduction_axes) & set(output_axes)) > 0:
    raise ValueError(
        f"Reduction and output axes must be disjoint, got {reduction_axes} " f"and {output_axes} for {x.ndim=}"
    )

  return reduction_axes, output_axes


WeightDimNumOrFn = ShardedMuonDimensionNumbers | optax.Params | Callable[[optax.Params], optax.Params | None]


class MuonState(NamedTuple):
  """State of the `GradientTransformation` returned by `muon`."""

  count: jax.typing.ArrayLike
  mu: jax.typing.ArrayLike
  ns_coeffs: jax.Array


def get_precond_fn(
    preconditioning: Preconditioning, eps: jax.typing.ArrayLike = 1e-8
) -> Callable[[jax.Array], jax.Array]:
  """Returns a preconditioning function."""
  if preconditioning == "frobenius":
    return lambda x: x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
  elif preconditioning == "spectral":
    return lambda x: x / (jnp.linalg.norm(x, ord=2, axis=(-2, -1), keepdims=True) + eps)
  else:
    return lambda x: x


def scale_by_muon(
    ns_coeffs: CoeffsType = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    mu_dtype: jax.typing.DTypeLike | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    preconditioning: Preconditioning = "frobenius",
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    use_all_to_all: bool = True,
) -> optax.GradientTransformation:
  """Rescale updates using Newton-Schulz operations."""
  mu_dtype = jax.dtypes.canonicalize_dtype(mu_dtype)
  precond_fn = get_precond_fn(preconditioning, eps)
  ns_step_fn = functools.partial(NS_ITERATORS[preconditioning], eps=eps)

  def init_fn(params: optax.Params) -> MuonState:
    ns_coeffs_ = jnp.array(ns_coeffs)
    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(f"ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}")
    if ns_coeffs_.ndim == 2:
      # pyrefly: ignore[unsupported-operation]
      if ns_coeffs_.shape[0] < ns_steps:
        raise ValueError(f"Not enough coeffs to perform {ns_steps} steps, got" f" {ns_coeffs_.shape[0]}")
      # pyrefly: ignore[unsupported-operation]
      ns_coeffs_ = ns_coeffs_[-ns_steps:]

    return MuonState(
        count=jnp.int32(0),
        mu=optax.tree.zeros_like(params),
        ns_coeffs=ns_coeffs_,
    )

  def update_fn(
      updates: optax.Updates,
      state: optax.OptState,
      params: optax.Params | None = None,
  ) -> tuple[optax.Updates, optax.OptState]:
    del params
    assert isinstance(state, MuonState)

    # Update the momentum buffer
    mu = optax.tree.update_moment(updates, state.mu, beta, 1)
    count_inc = optax.safe_increment(state.count)
    if nesterov:
      mu_hat = jax.tree.map(
          lambda m, g: beta * m + (1 - beta) * g,
          optax.tree.bias_correction(mu, beta, optax.safe_increment(count_inc)),
          optax.tree.bias_correction(updates, beta, count_inc),
      )
    else:
      mu_hat = optax.tree.bias_correction(mu, beta, count_inc)

    def orthogonalize_leaf(
        leaf: jax.Array | optax.MaskedNode,
        dim_nums: ShardedMuonDimensionNumbers | optax.MaskedNode | None,
    ) -> jax.Array | optax.MaskedNode:
      """Orthogonalize a single leaf tensor."""
      if isinstance(leaf, optax.MaskedNode) or isinstance(dim_nums, optax.MaskedNode):
        return optax.MaskedNode()
      if dim_nums is None:
        dim_nums = ShardedMuonDimensionNumbers()
      return orthogonalize(
          x=leaf,
          ns_coeffs=state.ns_coeffs,
          ns_steps=ns_steps,
          precond_fn=precond_fn,
          ns_step_fn=ns_step_fn,
          dim_nums=dim_nums,
          use_all_to_all=use_all_to_all,
      )

    if callable(weight_dimension_numbers):
      resolved_dim_nums = weight_dimension_numbers(updates)
    elif weight_dimension_numbers is None:
      resolved_dim_nums = jax.tree.map(lambda _: ShardedMuonDimensionNumbers(), updates)
    else:
      resolved_dim_nums = weight_dimension_numbers

    def is_leaf(x):
      return x is None or isinstance(x, (ShardedMuonDimensionNumbers, optax.MaskedNode))

    updates = jax.tree.map(
        orthogonalize_leaf,
        mu_hat,
        resolved_dim_nums,
        is_leaf=is_leaf,
    )

    if adaptive:
      # Scale the orthogonalized updates by the dual norm of the original
      # updates. See https://arxiv.org/abs/2409.20325 for the derivation.
      def scale_by_dual_norm(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.sum(x.conj() * y) * y

      updates = jax.tree.map(scale_by_dual_norm, mu_hat, updates)

    # Downcast mu to the desired dtype
    mu = jax.tree.map(lambda x: x.astype(mu_dtype), mu)
    return updates, MuonState(
        count=count_inc,
        mu=mu,
        ns_coeffs=state.ns_coeffs,
    )

  return optax.GradientTransformation(init_fn, update_fn)


def get_reshape_fns(
    x: jax.Array,
    dim_nums: ShardedMuonDimensionNumbers,
    use_all_to_all: bool = True,
) -> tuple[
    reshape_utils.ReshapeFn,
    reshape_utils.ReshapeFn,
    jax.sharding.NamedSharding | None,
]:
  """Compute reshape functions for a given tensor and dimension numbers."""
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  return reshape_utils.get_reshape_fns(
      x,
      reduction_axes,
      output_axes,
      sharding=dim_nums.sharding,
      use_all_to_all=use_all_to_all,
  )


def orthogonalize(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike,
    precond_fn: Callable[[jax.Array], jax.Array],
    ns_step_fn: Callable[..., jax.Array],
    dim_nums: ShardedMuonDimensionNumbers,
    use_all_to_all: bool = True,
) -> jax.Array:
  """Apply Newton-Schulz iterations to a single leaf tensor."""
  # We apply the following strategy:
  # 1. We reshape and reshard the input tensor.
  # 2. We apply preconditioning.
  # 3. We apply Newton-Schulz iterations.
  # 4. We un-reshape and re-reshard the output tensor.
  reshape_fn, unreshape_fn, flat_sharding = get_reshape_fns(x, dim_nums, use_all_to_all=use_all_to_all)
  ns_coeffs_ = ns_coeffs.astype(x.dtype)

  def apply_newton_schulz(y: jax.Array) -> jax.Array:
    y = precond_fn(y)
    if flat_sharding is not None:
      y = reshape_utils.reshard_or_constrain(y, flat_sharding)
    if ns_coeffs_.ndim == 1:

      def ns_iterator(i, z):
        res = ns_step_fn(i, z, ns_coeffs_, flat_sharding=flat_sharding)
        if flat_sharding is not None:
          res = reshape_utils.reshard_or_constrain(res, flat_sharding)
        return res

      y = jax.lax.fori_loop(
          lower=0,
          upper=ns_steps,
          body_fun=ns_iterator,
          init_val=y,
          unroll=False,
      )
    else:

      def _scan_body(carry, coeffs_step):
        i, z = carry
        z_new = ns_step_fn(i, z, coeffs_step, flat_sharding=flat_sharding)
        if flat_sharding is not None:
          z_new = reshape_utils.reshard_or_constrain(z_new, flat_sharding)
        return (i + 1, z_new), None

      init_carry = (jnp.asarray(0, dtype=jnp.int32), y)
      (_, y), _ = jax.lax.scan(_scan_body, init_carry, ns_coeffs_)
    return y

  x_flat = reshape_fn(x)
  x_flat_orthogonalized = apply_newton_schulz(x_flat)
  return unreshape_fn(x_flat_orthogonalized)


def _get_shape_products(x: jax.Array, dim_nums: ShardedMuonDimensionNumbers) -> tuple[float, float]:
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  fan_in = math.prod(x.shape[ax] for ax in reduction_axes)
  fan_out = math.prod(x.shape[ax] for ax in output_axes)
  return fan_in, fan_out


def _scale_update_for_width_transfer(update: jax.Array, dim_nums: ShardedMuonDimensionNumbers):
  """Apply width scaling from <https://github.com/KellerJordan/Muon>."""
  fan_in, fan_out = _get_shape_products(update, dim_nums)
  scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
  return scale * update


def _scale_update_for_consistent_rms(
    update: jax.Array,
    dim_nums: ShardedMuonDimensionNumbers,
    consistent_rms: jax.typing.ArrayLike,
):
  """Apply consistent RMS scaling from <https://arxiv.org/abs/2502.16982>."""
  fan_in, fan_out = _get_shape_products(update, dim_nums)
  scale = jnp.sqrt(jnp.maximum(fan_in, fan_out)) * consistent_rms
  return scale * update


def scale_by_shape(
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> optax.GradientTransformation:
  """Scale updates by factors derived from parameter shape.

  Args:
    weight_dimension_numbers: An optional tree with the same structure as the
      params of `ShardedMuonDimensionNumbers`s, specifying how to reshape the
      parameters before and after the orthogonalization OR a callable returning
      such a tree. None implies that all parameters are 2D matrices.
    consistent_rms: An optional float to activate consistent RMS scaling. If
      float, scales updates by `sqrt(max(fan_in, fan_out)) * consistent_rms`. If
      None, uses width scaling `sqrt(max(1, fan_out / fan_in))`.

  Returns:
    A `GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params
    if callable(weight_dimension_numbers):
      # Populate weight_dim_nums if it's a callable. Use updates instead of
      # actual params since only shapes matter and params may not be provided.
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

    if consistent_rms is not None:
      base_scaling_fn = functools.partial(_scale_update_for_consistent_rms, consistent_rms=consistent_rms)
    else:
      base_scaling_fn = _scale_update_for_width_transfer

    def scaling_fn(update, dim_nums):
      if isinstance(update, optax.MaskedNode) or isinstance(dim_nums, optax.MaskedNode):
        return optax.MaskedNode()
      if dim_nums is None:
        dim_nums = ShardedMuonDimensionNumbers()
      return base_scaling_fn(update, dim_nums)

    def is_leaf(x):
      return x is None or isinstance(x, (ShardedMuonDimensionNumbers, optax.MaskedNode))

    scaled_updates = jax.tree.map(
        scaling_fn,
        updates,
        resolved_weight_dim_nums,
        is_leaf=is_leaf,
    )
    return scaled_updates, state

  # Use the standard empty_state initializer, as this transform is stateless
  return optax.GradientTransformation(optax.init_empty_state, update_fn)


WeightDecayMask = optax.Params | Callable[[optax.Params], optax.Params]


def muon(
    learning_rate: optax.ScalarOrSchedule,
    ns_coeffs: CoeffsType = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-8,
    weight_decay: jax.typing.ArrayLike = 0.0,
    weight_decay_mask: WeightDecayMask | None = None,
    mu_dtype: jax.typing.DTypeLike | None = None,
    *,
    nesterov: bool = True,
    adaptive: bool = False,
    preconditioning: Preconditioning = "frobenius",
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_weight_decay: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: optax.ScalarOrSchedule | None = None,
    muon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    use_all_to_all: bool = True,
) -> optax.GradientTransformation:
  r"""Muon: Momentum Orthogonalized by Newton-schulz.

  Muon is a variant of Shampoo that uses the Newton-schulz method to
  orthogonalize the momentum accumulated by the optimizer. Mathematically, it
  does steepest descent under the Schatten-p norm, for some large p. With
  p=infty, it is equivalent to Shampoo without accumulation, or steepest
  descent under the Spectral norm.

  Note that Muon is currently only defined for 2D parameters, i.e. matrices.
  This is because the Newton-Schulz iterator expects a matrix as input.
  The non-2D parameters are instead passed through an AdamW optimizer
  (using a weight decay of 0 as default).

  Args:
    learning_rate: A global scaling factor, either fixed or evolving along
      iterations with a scheduler, see :func:`optax.scale_by_learning_rate`.
    ns_coeffs: Coefficients for the Newton-schulz method (can be a string
      indicator for a preset). Existing presets: `muon`, `dion`.
    ns_steps: Number of Newton-schulz iterations. Ignored if `ns_coeffs` is a
      tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to the denominator to improve numerical stability.
    weight_decay: Strength of the weight decay regularization. Note that this
      weight decay is multiplied with the learning rate. This is consistent with
      other frameworks such as PyTorch, but different from (Loshchilov et al,
      2019) where the weight decay is only multiplied with the "schedule
      multiplier", but not the base learning rate.
    weight_decay_mask: A tree with same structure as (or a prefix of) the params
      PyTree, or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the weight decay to, and `False` for those you want to skip.
    mu_dtype: Data type of the momentum accumulator.
    nesterov: Whether to use Nesterov momentum.
    adaptive: Whether to scale the updates by the dual norm of the original
      updates. See <https://arxiv.org/abs/2409.20325>
    preconditioning: What type of preconditioning to use before NS iterations.
      Available options are: - 'frobenius' (default): Use Frobenius rescaling
      before NS: safe, standard, but degrades orthogonalization quality when
      using less than 5 NS steps. - 'spectral' : Use Spectral norm rescaling
      before NS: much more computationally intensive, but better
      orthogonalization quality. - 'aol': Use AOL rescalings to improve
      orthogonality with little to no overhead, usually allows the user to
      remove one iterative NS step. See <https://arxiv.org/abs/2512.04632>. -
      'schatten': Use the Schatten-4 norm for rescaling, allows for better
      performance with little to no extra cost. See
      <https://arxiv.org/abs/2506.10935>.
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    adam_learning_rate: Auxiliary learning rate for the Adam optimizer. If
      `None`, the learning rate for Adam defaults to the same as Muon.
    muon_weight_dimension_numbers: An optional tree of
      `ShardedMuonDimensionNumbers`s, specifying how to reshape the parameters
      for orthogonalization otherwise muon parameters are assumed to be 2D
      matrices. A `None` value indicates that the parameter is not a muon
      parameter and will be optimized with Adam. A callable takes as input the
      params and returns a possibly masked pytree of specs, similar to
      `weight_decay_mask`. If not provided, muon is applied to all 2D
      parameters.
    consistent_rms: An optional float to activate consistent RMS scaling. Scales
      updates by `sqrt(max(fan_in, fan_out)) * consistent_rms` to make root mean
      square (RMS) shape-independent, like AdamW. `0.2` is recommended to match
      AdamW's empirical RMS. See <https://arxiv.org/abs/2502.16982>. If `None`,
      uses width scaling `sqrt(max(1, fan_out / fan_in))`.
    use_all_to_all: Whether to use all-to-all communication to transfer matrix
      sharding to unsharded batch axes. Defaults to True.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024

    Liu et al., `Muon is Scalable for LLM Training`,
    <https://arxiv.org/abs/2502.16982>`_, 2025

    Boissin et al., `Turbo-Muon: Accelerating Orthogonality-Based
    Optimization with Pre-Conditioning`,
    <https://arxiv.org/abs/2512.04632>`_, 2025

    Ahn et al., `Dion: Distributed Orthonormalized Updates`,
    <https://arxiv.org/abs/2504.05295>`_, 2025

    Grishina et al., `Accelerating Newton-Schulz Iteration for Orthogonalization
    via Chebyshev-type Polynomials`,
    <https://arxiv.org/abs/2506.10935>`_, 2025

    Amsel et al., `The Polar Express: Optimal Matrix Sign Methods and Their
    Application to the Muon Algorithm`,
    <https://arxiv.org/pdf/2505.16932>`, 2025
  """

  if adam_learning_rate is None:
    adam_learning_rate = learning_rate

  if isinstance(ns_coeffs, str):
    if ns_coeffs not in _NS_COEFFS_PRESET_DICT:
      raise ValueError(f"Unknown ns_coeff preset string: {ns_coeffs}")
    ns_coeffs_ = _NS_COEFFS_PRESET_DICT[ns_coeffs]
  else:
    ns_coeffs_ = ns_coeffs

  # None at root indicates the default 2D rule.
  if muon_weight_dimension_numbers is None:

    def param_labels(params):
      return jax.tree.map(lambda x: "muon" if x.ndim == 2 else "adam", params)

    muon_weight_dimension_numbers = ShardedMuonDimensionNumbers()
  else:

    def param_labels(params):
      dim_nums = (
          muon_weight_dimension_numbers(params)
          if callable(muon_weight_dimension_numbers)
          else muon_weight_dimension_numbers
      )

      def populate_subtree_(dim_num, x):
        return jax.tree.map(lambda y: "muon" if dim_num is not None else "adam", x)

      # Dimension numbers come first since they can be a prefix mask.
      return jax.tree.map(
          populate_subtree_,
          dim_nums,
          params,
          is_leaf=lambda x: x is None or isinstance(x, ShardedMuonDimensionNumbers),
      )

  # We need to normalize the dimension numbers because they have to match the
  # tree structure of the masked muon state tree (see `combine.partition`).
  def muon_weight_dim_nums_fn(params):
    # Normalize the dimension numbers for `combine.partition`.
    # Insert MaskedNode() where muon state will be masked out.
    dim_nums = (
        muon_weight_dimension_numbers(params)
        if callable(muon_weight_dimension_numbers)
        else muon_weight_dimension_numbers
    )
    mask = jax.tree.map(lambda label: label == "muon", param_labels(params))

    def is_leaf(x):
      return x is None or isinstance(x, (ShardedMuonDimensionNumbers, optax.MaskedNode))

    def populate_subtree_(dim_nums, submask):
      return jax.tree.map(lambda m: dim_nums if m else optax.MaskedNode(), submask)

    return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)

  return optax.partition(
      transforms={
          "muon": optax.chain(
              scale_by_muon(
                  ns_coeffs=ns_coeffs_,  # pyrefly: ignore[bad-argument-type]
                  ns_steps=ns_steps,
                  beta=beta,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  nesterov=nesterov,
                  adaptive=adaptive,
                  preconditioning=preconditioning,
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
                  use_all_to_all=use_all_to_all,
              ),
              scale_by_shape(
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
                  consistent_rms=consistent_rms,
              ),
              # pyrefly: ignore[bad-argument-type]
              optax.add_decayed_weights(weight_decay, weight_decay_mask),
              optax.scale_by_learning_rate(learning_rate),
          ),
          "adam": optax.adamw(
              learning_rate=adam_learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=eps,
              eps_root=adam_eps_root,
              # pyrefly: ignore[bad-argument-type]
              weight_decay=adam_weight_decay,
              mu_dtype=mu_dtype,
              nesterov=nesterov,
          ),
      },
      param_labels=param_labels,
  )
