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

# pylint: disable=unnecessary-lambda-assignment


import functools
import itertools
import logging
import math
from typing import Any, Callable, NamedTuple, Optional, Union, Sequence, Literal

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

from jax.sharding import NamedSharding, PartitionSpec

ReshapeFn = Callable[[jax.Array], jax.Array]

_PRECONDITIONINGS = ["frobenius", "spectral", "aol", "schatten"]
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

  When ``component_splits`` is set, the output axis is split into components
  (e.g., MLA nope/rope) and the batch axes are merged into each component's
  output dimension before NS, matching Megatron's per-component treatment.
  Each component gets its own NS orthogonalization and scale factor.

  When ``scan_axis`` is set (requires ``component_splits``), the specified axis
  is excluded from batch-merge and vmapped over instead, so each scan slice
  gets independent NS — producing identical results to unscan mode.
  """

  reduction_axis: Sequence[int] | int = 0
  output_axis: Sequence[int] | int = 1
  component_splits: tuple[int, ...] | None = None
  scan_axis: int | None = None


WeightDimNumOrFn = MuonDimensionNumbers | base.Params | Callable[[base.Params], base.Params | None]


_is_weight_dim_nums = lambda x: isinstance(x, MuonDimensionNumbers)


def _normalize_axes(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Normalize axes in dimension numbers to two tuples of non-negative ints."""
  if isinstance(dim_nums.reduction_axis, int):
    dim_nums = dim_nums._replace(reduction_axis=(dim_nums.reduction_axis,))
  reduction_axes = tuple(ax % x.ndim for ax in dim_nums.reduction_axis)

  if isinstance(dim_nums.output_axis, int):
    dim_nums = dim_nums._replace(output_axis=(dim_nums.output_axis,))
  output_axes = tuple(ax % x.ndim for ax in dim_nums.output_axis)
  return reduction_axes, output_axes


def _compute_muon_reshape(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[ReshapeFn, ReshapeFn]:
  """Compute the reshape and inverse functions for an array from a spec."""
  if x.ndim < 2:
    raise ValueError("Muon optimized parameters must have rank >= 2, got" f" {x.ndim=}")
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  if set(reduction_axes) & set(output_axes):
    raise ValueError(
        "Normalized reduction axes and output axes must be"
        f" disjoint, got {reduction_axes} and {output_axes}."
        f" Originally {dim_nums=} and {x.shape=}"
    )
  batch_axes = tuple(sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes)))
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
  unflat_shape = axes2shape(batch_axes) + axes2shape(reduction_axes) + axes2shape(output_axes)
  reshape_fn = lambda x: x.transpose(transpose).reshape(flat_shape)
  inverse_fn = lambda x: x.reshape(unflat_shape).transpose(inv_transpose)
  return reshape_fn, inverse_fn


def _get_shape_products(x: jax.Array, dim_nums: MuonDimensionNumbers) -> tuple[float, float]:
  reduction_axes, output_axes = _normalize_axes(x, dim_nums)
  fan_in = math.prod(x.shape[ax] for ax in reduction_axes)
  fan_out = math.prod(x.shape[ax] for ax in output_axes)
  return fan_in, fan_out


def _component_splits_for_output_axis(
    x: jax.Array,
    dim_nums: MuonDimensionNumbers,
    reduction_axes: tuple[int, ...],
    output_axes: tuple[int, ...],
) -> tuple[int, ...]:
  """Return per-output-axis component splits, accepting either per-axis or total-batched splits."""
  if len(output_axes) != 1:
    raise ValueError(f"component_splits requires exactly 1 output axis, got {output_axes}")

  output_axis = output_axes[0]
  splits = dim_nums.component_splits
  if sum(splits) == x.shape[output_axis]:
    return splits

  batch_axes = tuple(sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes)))
  batch_size = math.prod(x.shape[ax] for ax in batch_axes)
  if batch_size > 1 and sum(splits) == batch_size * x.shape[output_axis] and all(s % batch_size == 0 for s in splits):
    return tuple(s // batch_size for s in splits)

  raise ValueError(
      f"component_splits {splits} must sum to output axis size {x.shape[output_axis]} "
      f"or total batched output size {batch_size * x.shape[output_axis]} (and be divisible by batch_size {batch_size})"
  )


def _build_component_scale_tensor(
    update: jax.Array,
    dim_nums: MuonDimensionNumbers,
    scale_fn: Callable[[float, float], float],
) -> jax.Array:
  """Build a broadcast-compatible scale tensor for component_splits.

  Each component region along the output axis gets its own scale factor,
  computed from the per-component fan values (where batch dims are merged
  into the output, matching Megatron's per-component treatment).

  When ``scan_axis`` is set, it is excluded from the batch product so that
  scale factors match the unscan (per-layer) computation.
  """
  reduction_axes, output_axes = _normalize_axes(update, dim_nums)
  batch_axes = tuple(sorted(set(range(update.ndim)) - set(reduction_axes) - set(output_axes)))
  output_axis = output_axes[0]

  # Exclude scan_axis from batch axes for split normalization and fan computation.
  scan_axis = dim_nums.scan_axis
  if scan_axis is not None:
    scan_axis_norm = scan_axis % update.ndim
    batch_axes_for_splits = tuple(ax for ax in batch_axes if ax != scan_axis_norm)
  else:
    scan_axis_norm = None
    batch_axes_for_splits = batch_axes

  if scan_axis is not None:
    # Create a reduced-rank view (without scan axis) so
    # _component_splits_for_output_axis sees the correct batch_size.
    remaining_axes = [i for i in range(update.ndim) if i != scan_axis_norm]
    slice_shape = tuple(update.shape[ax] for ax in remaining_axes)

    def _remap(ax):
      return ax - 1 if ax > scan_axis_norm else ax

    inner_reduction = tuple(_remap(ax) for ax in reduction_axes)
    inner_output = tuple(_remap(ax) for ax in output_axes)
    inner_dim_nums = MuonDimensionNumbers(
        reduction_axis=inner_reduction,
        output_axis=inner_output,
        component_splits=dim_nums.component_splits,
    )
    dummy = jnp.zeros(slice_shape)
    splits = _component_splits_for_output_axis(dummy, inner_dim_nums, inner_reduction, inner_output)
  else:
    splits = _component_splits_for_output_axis(update, dim_nums, reduction_axes, output_axes)

  fan_in = math.prod(update.shape[ax] for ax in reduction_axes)
  batch_prod = math.prod(update.shape[ax] for ax in batch_axes_for_splits)

  # Build a 1D array of scales along the output axis.
  scales = []
  for s in splits:
    fan_out = batch_prod * s
    scale = scale_fn(fan_in, fan_out)
    scales.extend([scale] * s)

  # Reshape to broadcast: all dims are 1 except the output axis.
  shape = [1] * update.ndim
  shape[output_axis] = sum(splits)
  return jnp.asarray(scales, dtype=update.dtype).reshape(shape)


def _scale_update_for_width_transfer(update: jax.Array, dim_nums: MuonDimensionNumbers):
  """Apply width scaling from <https://github.com/KellerJordan/Muon>."""
  if getattr(dim_nums, "component_splits", None) is not None:
    return update * _build_component_scale_tensor(
        update, dim_nums, lambda fan_in, fan_out: math.sqrt(max(1, fan_out / fan_in))
    )
  fan_in, fan_out = _get_shape_products(update, dim_nums)
  scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
  # Cast scale to the update's dtype so shape-scale stays in whatever dtype
  # NS produced (fp32 by default, bf16 if the caller requested Megatron-style
  # parity). This keeps the downstream chain (add_decayed_weights,
  # scale_by_learning_rate) in that same dtype until the fp32 param add.
  return update * jnp.asarray(scale, dtype=update.dtype)


def _scale_update_for_consistent_rms(
    update: jax.Array, dim_nums: MuonDimensionNumbers, consistent_rms: jax.typing.ArrayLike
):
  """Apply consistent RMS scaling from <https://arxiv.org/abs/2502.16982>."""
  if getattr(dim_nums, "component_splits", None) is not None:
    return update * _build_component_scale_tensor(
        update, dim_nums, lambda fan_in, fan_out: (math.sqrt(max(fan_in, fan_out)) * float(consistent_rms))
    )
  fan_in, fan_out = _get_shape_products(update, dim_nums)
  scale = jnp.sqrt(jnp.maximum(fan_in, fan_out)) * consistent_rms
  # Keep update in its original dtype (fp32 by default) so the downstream
  # LR/WD multiplies stay in the NS output's precision.
  return update * jnp.asarray(scale, dtype=update.dtype)


def scale_by_shape(
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
) -> base.GradientTransformation:
  """Scale updates by factors derived from parameter shape.

  Args:
    weight_dimension_numbers: An optional tree with the same structure as the
      params of `MuonDimensionNumbers`s, specifying how to reshape the
      parameters before and after the orthogonalization OR a callable returning
      such a tree. None implies that all parameters are 2D matrices.
    consistent_rms: An optional float to activate consistent RMS scaling.
      If float, scales updates by `sqrt(max(fan_in, fan_out)) * consistent_rms`.
      If None, uses width scaling `sqrt(max(1, fan_out / fan_in))`.

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
      scaling_fn = functools.partial(_scale_update_for_consistent_rms, consistent_rms=consistent_rms)
    else:
      scaling_fn = _scale_update_for_width_transfer

    scaled_updates = jax.tree.map(
        scaling_fn,
        updates,
        resolved_weight_dim_nums,
        is_leaf=_is_weight_dim_nums,
    )
    return scaled_updates, state

  # Use the standard empty_state initializer, as this transform is stateless
  return base.GradientTransformation(base.init_empty_state, update_fn)


def _aol_first_newton_schulz_iteration(
    x: jax.Array,
    coeffs: jax.Array,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  """'Almost Orthogonal Layer' Preconditioning with Newton-Schulz iteration."""
  # Implements the first Newton-Schulz step with AOL preconditioning
  # which allows for better orthogonalization performance.
  a = x @ x.T.conj()
  rescaling = jnp.clip(jnp.abs(a).sum(axis=-1), min=eps)
  s = jnp.expand_dims(jax.lax.rsqrt(rescaling), -1)
  x, a = x * s, a * s * s.transpose(-1, -2)
  b = coeffs[1] * a + coeffs[2] * a @ a
  return coeffs[0] * x + b @ x


def _schatten_first_newton_schulz_iteration(
    x: jax.Array,
    coeffs: jax.Array,
    eps: jax.typing.ArrayLike = 1e-8,
) -> jax.Array:
  """Schatten-4 Preconditioning with Newton-Schulz iteration."""
  # Implements the first Newton-Schulz step with Schatten-4 norm
  # preconditioning which allows for better orthogonalization performance.
  a = x @ x.T
  rescaling = jnp.clip(jnp.linalg.norm(a, ord="fro", axis=(-2, -1)), min=eps)
  s = jnp.expand_dims(jax.lax.rsqrt(rescaling), (0, -1))
  x, a = x * s, a * s**2
  b = coeffs[1] * a + coeffs[2] * a @ a
  return coeffs[0] * x + b @ x


def _base_newton_schulz_iteration(x: jax.Array, coeffs: jax.Array) -> jax.Array:
  # Implements Newton-Schulz step f(X) = c_0 X + c_1 (XX^T)X + c_2 (XX^T)^2X,
  # with quintic form f(X) = c_0 X + (c_1 A + c_2 AA)X, where A = XX^T.
  # The NS step has the property f(X) = f(X^T)^T. That is, we can get equivalent
  # result by transposing input and output. In particular, we may transpose X
  # when rows > cols for efficiency.
  a = x @ x.T.conj()
  b = coeffs[1] * a + coeffs[2] * a @ a
  return coeffs[0] * x + b @ x


_newton_schulz_iterator = _base_newton_schulz_iteration  # backwards compat


def _aol_ns_iterator(i, x, coeffs):
  # Modified first step using AOL rescaling
  return jax.lax.cond(
      i == 0,
      lambda x: _aol_first_newton_schulz_iteration(x, coeffs),
      lambda x: _base_newton_schulz_iteration(x, coeffs),
      x,
  )


def _schatten_ns_iterator(i, x, coeffs):
  # Modified first step using Schatten-4 norm rescaling
  return jax.lax.cond(
      i == 0,
      lambda x: _schatten_first_newton_schulz_iteration(x, coeffs),
      lambda x: _base_newton_schulz_iteration(x, coeffs),
      x,
  )


def _base_ns_iterator(i, x, coeffs):
  del i
  return _base_newton_schulz_iteration(x, coeffs)


def _orthogonalize_components(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike,
    preconditioning: str,
    eps: jax.typing.ArrayLike,
    dimension_numbers: MuonDimensionNumbers,
    ns_unroll: bool = False,
    replicate_ns_matrix_axes: bool = False,
    replicate_ns_batch_axis: str | None = None,
) -> jax.Array:
  """Orthogonalize with per-component splitting on the output axis.

  Matches Megatron's per-component treatment of MLA projections: splits the
  output axis into semantic components (e.g., nope/rope), merges batch axes
  (heads) into each component, and applies NS independently to each resulting
  2D matrix.  Each component sees ``(reduction, batch * comp_size)`` where
  ``comp_size`` is the component's slice of the output axis.

  When ``scan_axis`` is set, that axis is vmapped over (not merged into batch),
  so each scan slice gets independent NS — matching unscan mode exactly.

  Args:
    x: The weight/gradient tensor with shape including reduction, batch (head),
      and output axes.
    ns_coeffs: Newton-Schulz coefficients.
    ns_steps: Number of NS iterations.
    preconditioning: Preconditioning method.
    eps: Numerical stability epsilon.
    dimension_numbers: MuonDimensionNumbers with component_splits set.

  Returns:
    Orthogonalized tensor with the same shape as the input.
  """
  saved_sharding = _get_array_sharding(x) if replicate_ns_matrix_axes else None
  reduction_axes, output_axes = _normalize_axes(x, dimension_numbers)

  # Validate scan_axis if present.
  scan_axis = dimension_numbers.scan_axis
  if scan_axis is not None:
    scan_axis = scan_axis % x.ndim
    if scan_axis in reduction_axes:
      raise ValueError(
          f"scan_axis={dimension_numbers.scan_axis} (normalized: {scan_axis}) "
          f"must not overlap with reduction_axis={reduction_axes}"
      )
    if scan_axis in output_axes:
      raise ValueError(
          f"scan_axis={dimension_numbers.scan_axis} (normalized: {scan_axis}) "
          f"must not overlap with output_axis={output_axes}"
      )

  if scan_axis is not None:
    # Move scan_axis to position 0, vmap the inner function over it,
    # then move it back. The inner function operates on tensors without
    # the scan dimension (e.g. 3D [R, H, O] instead of 4D [R, L, H, O]).
    # Remap dimension_numbers axes to account for the removed scan_axis.
    def _remap_axis(ax):
      return ax - 1 if ax > scan_axis else ax

    inner_reduction = tuple(_remap_axis(ax) for ax in reduction_axes)
    inner_output = tuple(_remap_axis(ax) for ax in output_axes)
    inner_dim_nums = MuonDimensionNumbers(
        reduction_axis=inner_reduction,
        output_axis=inner_output,
        component_splits=dimension_numbers.component_splits,
        scan_axis=None,
    )

    # Move scan_axis to position 0.
    perm = [scan_axis] + [i for i in range(x.ndim) if i != scan_axis]
    x_moved = jnp.transpose(x, perm)

    def _inner(x_slice):
      return _orthogonalize_components(
          x_slice,
          ns_coeffs,
          ns_steps,
          preconditioning,
          eps,
          inner_dim_nums,
          ns_unroll=ns_unroll,
          replicate_ns_matrix_axes=replicate_ns_matrix_axes,
          replicate_ns_batch_axis=replicate_ns_batch_axis,
      )

    result_moved = jax.vmap(_inner)(x_moved)

    # Inverse transpose to restore original axis order.
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
      inv_perm[p] = i
    return _maybe_restore_sharding(jnp.transpose(result_moved, inv_perm), saved_sharding)

  # --- Non-scan path (existing logic) ---
  batch_axes = tuple(sorted(set(range(x.ndim)) - set(reduction_axes) - set(output_axes)))
  output_axis = output_axes[0]
  splits = _component_splits_for_output_axis(x, dimension_numbers, reduction_axes, output_axes)

  # Transpose to (reduction..., batch..., output) canonical order.
  perm = tuple(reduction_axes) + tuple(batch_axes) + (output_axis,)
  x_t = jnp.transpose(x, perm)

  n_reduction = len(reduction_axes)
  n_batch = len(batch_axes)
  reduction_size = math.prod(x_t.shape[:n_reduction])
  batch_size = math.prod(x_t.shape[n_reduction : n_reduction + n_batch])

  # Split along the last axis (output) into components.
  split_indices = list(itertools.accumulate(splits[:-1]))
  components = jnp.split(x_t, split_indices, axis=-1)

  # Process each component: merge batch into output -> 2D -> NS -> reshape back.
  results = []
  for comp in components:
    comp_size = comp.shape[-1]
    # Reshape to 2D: (reduction_prod, batch_prod * comp_size).
    comp_2d = comp.reshape(reduction_size, batch_size * comp_size)
    if replicate_ns_matrix_axes:
      comp_2d = _replicate_matrix_axes_for_ns(comp_2d, _get_current_mesh_for_ns())
    # Apply NS as a standard 2D matrix (no component_splits in the recursive
    # call since MuonDimensionNumbers(0, 1) defaults to None).
    comp_ortho = orthogonalize_via_newton_schulz(comp_2d, ns_coeffs, ns_steps, preconditioning, eps, ns_unroll=ns_unroll)
    # Reshape back to (reduction..., batch..., comp_size).
    comp_back = comp_ortho.reshape(x_t.shape[: n_reduction + n_batch] + (comp_size,))
    results.append(comp_back)

  # Concatenate along output axis and inverse transpose.
  result_t = jnp.concatenate(results, axis=-1)
  inv_perm = tuple(sorted(range(len(perm)), key=lambda i: perm[i]))
  return _maybe_restore_sharding(jnp.transpose(result_t, inv_perm), saved_sharding)


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike = 5,
    preconditioning: Literal["frobenius", "spectral", "aol", "schatten"] = "frobenius",
    eps: jax.typing.ArrayLike = 1e-8,
    dimension_numbers: MuonDimensionNumbers | None = None,
    ns_unroll: bool = False,
    replicate_ns_matrix_axes: bool = False,
    replicate_ns_batch_axis: str | None = None,
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
    preconditioning: Which preconditioning method to use.
    eps: Term added to denominators to improve numerical stability.
    dimension_numbers: Optional spec for reshaping a tensor before and after the
      orthogonalization, to support non-2D parameters.

  Returns:
    The orthogonalized matrix.
  """
  # Dispatch to component-split path before any other logic.
  if isinstance(dimension_numbers, MuonDimensionNumbers):
    if dimension_numbers.scan_axis is not None and dimension_numbers.component_splits is None:
      raise ValueError("scan_axis requires component_splits to be set")
    if dimension_numbers.component_splits is not None:
      return _orthogonalize_components(
          x,
          ns_coeffs,
          ns_steps,
          preconditioning,
          eps,
          dimension_numbers,
          ns_unroll,
          replicate_ns_matrix_axes,
          replicate_ns_batch_axis,
      )

  if x.ndim != 2 and not isinstance(dimension_numbers, MuonDimensionNumbers):
    raise ValueError(
        f"Input must have shape (m, n) or weight dimension numbers must be"
        f" provided. Got shape={x.shape} and {dimension_numbers=}."
    )
  if x.ndim == 2:
    dimension_numbers = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError("Newton-Schulz coefficients must have shape (3,) or" f" (n, 3), got {ns_coeffs.shape}")

  # Cache mesh once so the inner closure can apply rank-2 P(None, None)
  # constraints directly on the dot operands inside vmap.
  ns_mesh = _get_current_mesh_for_ns() if replicate_ns_matrix_axes else None

  def _replicate_rank2_for_ns(x):
    """Apply P(None, None) constraint to a rank-2 matrix inside vmap."""
    if ns_mesh is None:
      return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(ns_mesh, PartitionSpec(None, None)))

  def _orthogonalize(x):
    # Constrain rank-2 slice directly so the dot x @ x.T inside the NS
    # iterator does not trigger an all-reduce on the contraction dim.
    x = _replicate_rank2_for_ns(x)

    transposed = False
    if x.shape[0] > x.shape[1]:
      x = x.T
      transposed = True
      # Re-constrain after transpose; XLA may pick a sharded layout for
      # the transposed tensor otherwise.
      x = _replicate_rank2_for_ns(x)

    ns_iterators = {
        "frobenius": _base_ns_iterator,
        "spectral": _base_ns_iterator,
        "aol": _aol_ns_iterator,
        "schatten": _schatten_ns_iterator,
    }
    if preconditioning not in _PRECONDITIONINGS:
      raise ValueError(f"Unknown preconditioning {preconditioning}")
    _ns_iterator = ns_iterators[preconditioning]

    if preconditioning == "frobenius":
      x /= jnp.linalg.norm(x, ord="fro") + eps
    elif preconditioning == "spectral":
      x /= jnp.linalg.norm(x, ord=2) + eps
    else:
      pass

    ns_coeffs_ = ns_coeffs.astype(x.dtype)

    if ns_coeffs_.ndim == 1:
      x = jax.lax.fori_loop(0, ns_steps, lambda i, x: _ns_iterator(i, x, ns_coeffs_), x, unroll=ns_unroll)
    else:

      def _scan_body(carry, coeffs_step):
        i, x = carry
        x_new = _ns_iterator(i, x, coeffs_step)
        return (i + 1, x_new), None

      init_carry = (jnp.asarray(0, dtype=jnp.int32), x)
      (_, x), _ = jax.lax.scan(_scan_body, init_carry, ns_coeffs_)

    if transposed:
      x = x.T
    return x

  saved_sharding = _get_array_sharding(x) if replicate_ns_matrix_axes else None
  reshape_fn, inverse_fn = _compute_muon_reshape(x, dimension_numbers)
  reshaped = reshape_fn(x)
  if replicate_ns_matrix_axes:
    reshaped = _replicate_matrix_axes_for_ns(reshaped, _get_current_mesh_for_ns(), replicate_ns_batch_axis)
  result = inverse_fn(jax.vmap(_orthogonalize)(reshaped))
  return _maybe_restore_sharding(result, saved_sharding)


class MuonState(NamedTuple):
  """State for the Muon algorithm."""

  count: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  ns_coeffs: jax.typing.ArrayLike  # shape=(), dtype=jnp.int32.


def _get_array_sharding(x):
  """Get sharding from a JAX array, or None if unavailable."""
  return getattr(x, "sharding", None)


def _get_current_mesh_for_ns():
  """Return the active mesh context used to build NS sharding constraints."""
  get_abstract_mesh = getattr(jax.sharding, "get_abstract_mesh", None)
  if get_abstract_mesh is not None:
    mesh = get_abstract_mesh()
    if mesh is not None and not mesh.empty:
      return mesh
  raise ValueError(
      "replicate_ns_matrix_axes=True requires an active mesh context. "
      "Run the optimizer update under jax.set_mesh(...)."
  )


def _replicate_matrix_axes_for_ns(x: jax.Array, mesh, batch_axis=None) -> jax.Array:
  """Replicate matrix (m, n) axes while keeping batch axes sharded.

  Args:
    x: NS input tensor. Supports rank-2 (m, n), rank-3 (batch, m, n),
      and rank-4 chunked input (num_chunks, batch_update_size, m, n).
    mesh: JAX Mesh object for constructing NamedSharding.
    batch_axis: Mesh axis name for the leading batch-like dimension. Must be
      set for rank-3/rank-4 inputs, e.g. "fsdp".
  """
  if x.ndim < 2:
    return x

  if x.ndim == 2:
    spec = PartitionSpec(None, None)
  elif x.ndim == 3:
    if batch_axis is None:
      raise ValueError("replicate_ns_batch_axis must be set when replicating rank-3 Muon NS inputs")
    chosen_axis = _maybe_choose_ns_batch_axis(x.shape[0], mesh, batch_axis)
    if chosen_axis is not None:
      spec = PartitionSpec(chosen_axis, None, None)
    elif _fits_full_replicate_budget(x):
      spec = PartitionSpec(None, None, None)
    else:
      return x
  elif x.ndim == 4:
    if batch_axis is None:
      raise ValueError("replicate_ns_batch_axis must be set when replicating rank-4 Muon NS inputs")
    spec = PartitionSpec(batch_axis, None, None, None)
  else:
    return x

  return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, spec))


def _mesh_axis_size(mesh, axis_name: str | tuple[str, ...] | None) -> int:
  """Return the total size of one mesh axis or a tuple of mesh axes."""
  if axis_name is None:
    return 1
  if isinstance(axis_name, tuple):
    return math.prod(int(mesh.shape[axis]) for axis in axis_name)
  return int(mesh.shape[axis_name])


def _maybe_choose_ns_batch_axis(ns_batch_size, mesh, preferred_axis):
  """Return preferred_axis if ns_batch_size is divisible by its mesh size, else None."""
  if preferred_axis is None:
    return None
  axis_size = _mesh_axis_size(mesh, preferred_axis)
  if axis_size > 1 and ns_batch_size % axis_size == 0:
    return preferred_axis
  return None


_NS_FULL_REPLICATE_MAX_BYTES = 256 << 20  # 256 MB


def _fits_full_replicate_budget(x):
  """Check if full-replicating x for NS won't exceed HBM budget."""
  b, m, n = x.shape
  elem_bytes = x.dtype.itemsize if hasattr(x, "dtype") else 2
  min_dim = min(m, n)
  working_bytes = (b * m * n + 2 * b * min_dim * min_dim) * elem_bytes
  return working_bytes <= _NS_FULL_REPLICATE_MAX_BYTES


def _pad_batch_to_multiple(x: jax.Array, multiple: int) -> tuple[jax.Array, int]:
  """Pad rank-3 Muon NS input on batch dim to a multiple of mesh axis size."""
  total_batch = x.shape[0]
  if multiple <= 1:
    return x, total_batch
  padded_size = math.ceil(total_batch / multiple) * multiple
  if padded_size == total_batch:
    return x, total_batch
  return jnp.pad(x, ((0, padded_size - total_batch), (0, 0), (0, 0))), total_batch


def _maybe_restore_sharding(x: jax.Array, sharding) -> jax.Array:
  """Restore original sharding after NS, if a saved sharding is available."""
  if sharding is None:
    return x
  return jax.lax.with_sharding_constraint(x, sharding)


def _batch_orthogonalize_tree(
    updates: base.Updates,
    weight_dim_nums: base.Params,
    ns_coeffs: jax.Array,
    ns_steps: jax.typing.ArrayLike,
    preconditioning: str,
    eps: jax.typing.ArrayLike,
    batch_update_size: int | None,
    ns_unroll: bool,
    replicate_ns_matrix_axes: bool = False,
    replicate_ns_batch_axis: str | None = None,
) -> base.Updates:
  """Orthogonalize updates by grouping same-shape matrices and batching NS.

  Instead of running Newton-Schulz on each parameter independently, groups
  parameters by their reshaped 2D matrix shape (m, n), stacks them along the
  batch dimension, and runs NS on the combined batch. This amortizes overhead
  for many small matrices. Results are numerically identical to unbatched mode.

  Args:
    updates: PyTree of gradient updates (already momentum-processed).
    weight_dim_nums: PyTree of MuonDimensionNumbers (same structure as updates).
    ns_coeffs: Newton-Schulz coefficients array.
    ns_steps: Number of NS iterations.
    preconditioning: Preconditioning method.
    eps: Numerical stability epsilon.
    batch_update_size: If set, chunk each shape group into batches of this size.

  Returns:
    PyTree of orthogonalized updates (same structure as input).
  """
  flat_updates, treedef = jax.tree.flatten(updates, is_leaf=_is_weight_dim_nums)

  ns_mesh = None
  if replicate_ns_matrix_axes:
    if replicate_ns_batch_axis is None:
      raise ValueError("replicate_ns_batch_axis must be set when replicate_ns_matrix_axes=True")
    ns_mesh = _get_current_mesh_for_ns()

  # Handle weight_dim_nums=None: create matching default dim nums for 2D params.
  if weight_dim_nums is None:
    flat_dim_nums = [None] * len(flat_updates)
  else:
    flat_dim_nums, _ = jax.tree.flatten(weight_dim_nums, is_leaf=_is_weight_dim_nums)

  # For each leaf, reshape to (batch, m, n) and record metadata for regrouping.
  # Group key is (m, n) after ensuring m <= n.
  groups: dict[tuple[int, int], list] = {}
  component_results: dict[int, dict[str, Any]] = {}
  results = [None] * len(flat_updates)

  for i, (leaf, dim_num) in enumerate(zip(flat_updates, flat_dim_nums)):
    if leaf.ndim < 2:
      # Scalar/1D params shouldn't reach here (they go to Adam), but be safe.
      results[i] = leaf
      continue

    if dim_num is not None and getattr(dim_num, "component_splits", None) is not None:
      reduction_axes, output_axes = _normalize_axes(leaf, dim_num)
      batch_axes = tuple(sorted(set(range(leaf.ndim)) - set(reduction_axes) - set(output_axes)))
      splits = dim_num.component_splits

      if len(output_axes) != 1:
        raise ValueError(f"component_splits requires exactly 1 output axis, got {output_axes}")
      output_axis = output_axes[0]

      if sum(splits) != leaf.shape[output_axis]:
        raise ValueError(f"component_splits {splits} must sum to output axis size {leaf.shape[output_axis]}")

      perm = tuple(reduction_axes) + tuple(batch_axes) + (output_axis,)
      inv_perm = tuple(sorted(range(len(perm)), key=lambda j, _perm=perm: _perm[j]))
      leaf_t = jnp.transpose(leaf, perm)
      n_reduction = len(reduction_axes)
      n_batch = len(batch_axes)
      reduction_size = math.prod(leaf_t.shape[:n_reduction])
      batch_size = math.prod(leaf_t.shape[n_reduction : n_reduction + n_batch])
      split_indices = list(itertools.accumulate(splits[:-1]))
      component_results[i] = {
          "shape": leaf_t.shape,
          "splits": splits,
          "inv_perm": inv_perm,
          "parts": [],
          "original_sharding": _get_array_sharding(leaf) if replicate_ns_matrix_axes else None,
      }

      for component_index, comp in enumerate(jnp.split(leaf_t, split_indices, axis=-1)):
        comp_size = comp.shape[-1]
        reshaped = comp.reshape(reduction_size, batch_size * comp_size)
        m, n = reshaped.shape
        transposed = m > n
        if transposed:
          reshaped = reshaped.T
          m, n = n, m
        groups.setdefault((m, n), []).append(
            {
                "idx": i,
                "component_index": component_index,
                "reshaped": reshaped[jnp.newaxis, ...],
                "transposed": transposed,
                "batch_size": 1,
                "component_shape": leaf_t.shape[: n_reduction + n_batch] + (comp_size,),
            }
        )
      continue

    if leaf.ndim == 2:
      dn = MuonDimensionNumbers(reduction_axis=0, output_axis=1)
    else:
      dn = dim_num

    reshape_fn, inverse_fn = _compute_muon_reshape(leaf, dn)
    reshaped = reshape_fn(leaf)  # (batch, m, n)
    _, m, n = reshaped.shape
    transposed = m > n
    if transposed:
      reshaped = reshaped.transpose(0, 2, 1)
      m, n = n, m

    key = (m, n)
    if key not in groups:
      groups[key] = []
    groups[key].append(
        {
            "idx": i,
            "reshaped": reshaped,
            "inverse_fn": inverse_fn,
            "transposed": transposed,
            "batch_size": reshaped.shape[0],
            "original_sharding": _get_array_sharding(leaf) if replicate_ns_matrix_axes else None,
        }
    )

  # Component-split params are pre-flattened to 3D before reaching this path,
  # so chunking can use a fixed (batch, m, n) dim spec for every group.
  dim_num_chunk = MuonDimensionNumbers(reduction_axis=1, output_axis=2)

  def _ns_chunk(_, chunk):
    return _, orthogonalize_via_newton_schulz(
        chunk,
        ns_coeffs,
        ns_steps,
        preconditioning,
        eps,
        dim_num_chunk,
        ns_unroll=ns_unroll,
        replicate_ns_matrix_axes=replicate_ns_matrix_axes,
        replicate_ns_batch_axis=replicate_ns_batch_axis,
    )

  # Process each shape group.
  for (m, n), items in groups.items():
    # Concatenate all items along batch dim.
    all_reshaped = jnp.concatenate([item["reshaped"] for item in items], axis=0)

    # Chunk if batch_update_size is set. Use lax.scan so all chunks share a
    # single traced NS body in HLO instead of N independent fori_loops, which
    # cuts HLO instruction count substantially when groups have many leaves.
    # Safe here because component-split params are pre-flattened to 3D before
    # reaching this path, so the chunking no longer composes with a nested
    # vmap structure that previously triggered an XLA layout assertion.
    total_batch = all_reshaped.shape[0]

    # Auto-cap batch size when replicate_ns_matrix_axes is on. After
    # replication, matrix dims (m, n) are fully materialized on each device
    # instead of being FSDP-sharded. NS peak memory per matrix in the batch:
    #   input(m*n) + gram a=x@xT(m*m) + intermediates(m*m) ≈ m*n + 2*m*m.
    # Cap total working set to NS_REPLICATE_SAFETY_BYTES so that
    # batch_update_size=None doesn't OOM on large meshes.
    effective_batch_size = batch_update_size
    if replicate_ns_matrix_axes and batch_update_size is None:
      NS_REPLICATE_SAFETY_BYTES = 1 << 30  # 1 GB
      elem_bytes = all_reshaped.dtype.itemsize
      per_matrix_bytes = (m * n + 2 * m * m) * elem_bytes
      raw_cap = max(1, NS_REPLICATE_SAFETY_BYTES // per_matrix_bytes)
      # Round down to fsdp multiple so each chunk can use P(fsdp, None, None)
      # instead of falling back to P(None, None, None) full-replicate.
      # Without this, e.g. raw_cap=85 with fsdp=64 produces chunks of size 85
      # which aren't divisible → _maybe_choose_ns_batch_axis returns None →
      # full-replicate → 6x padding expansion → OOM.
      axis_size = _mesh_axis_size(ns_mesh, replicate_ns_batch_axis)
      if 1 < axis_size <= raw_cap:
        auto_cap = (raw_cap // axis_size) * axis_size
      else:
        auto_cap = raw_cap
      if effective_batch_size is None:
        logging.info(
            "Muon NS auto-cap: matrix=(%d,%d), per_matrix_bytes=%d, raw_cap=%d, "
            "aligned_cap=%d, fsdp_size=%d, batch_update_size=%s",
            m,
            n,
            per_matrix_bytes,
            raw_cap,
            auto_cap,
            axis_size,
            batch_update_size,
        )
        effective_batch_size = auto_cap

    if effective_batch_size is not None and total_batch > effective_batch_size:
      num_chunks = math.ceil(total_batch / effective_batch_size)
      padded_size = num_chunks * effective_batch_size
      if padded_size > total_batch:
        padded = jnp.pad(all_reshaped, ((0, padded_size - total_batch), (0, 0), (0, 0)))
      else:
        padded = all_reshaped
      chunked = padded.reshape(num_chunks, effective_batch_size, m, n)
      # Do NOT apply rank-4 constraint here. The scan body applies the
      # rank-3 P(batch_axis, None, None) constraint inside each iteration,
      # which is what actually eliminates per-NS-step all-reduce on the
      # contraction dim. Adding a rank-4 P(fsdp, None, None, None) on top
      # conflicts with the inner constraint at scan boundaries and causes
      # XLA to allocate huge resharding staging buffers (OOM on 256 chips).
      _, all_ortho_chunked = jax.lax.scan(_ns_chunk, None, chunked)
      all_ortho = all_ortho_chunked.reshape(padded_size, m, n)[:total_batch]
    else:
      if replicate_ns_matrix_axes:
        all_reshaped, unpadded_batch = _pad_batch_to_multiple(
            all_reshaped, _mesh_axis_size(ns_mesh, replicate_ns_batch_axis)
        )
        all_reshaped = _replicate_matrix_axes_for_ns(all_reshaped, ns_mesh, replicate_ns_batch_axis)
      all_ortho = orthogonalize_via_newton_schulz(
          all_reshaped,
          ns_coeffs,
          ns_steps,
          preconditioning,
          eps,
          dim_num_chunk,
          ns_unroll=ns_unroll,
          replicate_ns_matrix_axes=replicate_ns_matrix_axes,
          replicate_ns_batch_axis=replicate_ns_batch_axis,
      )
      if replicate_ns_matrix_axes:
        all_ortho = all_ortho[:unpadded_batch]

    # Split back to individual items.
    batch_sizes = [item["batch_size"] for item in items]
    split_indices = list(itertools.accumulate(batch_sizes[:-1]))
    split_results = jnp.split(all_ortho, split_indices, axis=0)

    for item, ortho in zip(items, split_results):
      if item["transposed"]:
        ortho = ortho.transpose(0, 2, 1)
      if "inverse_fn" in item:
        restored = item["inverse_fn"](ortho)
        restored = _maybe_restore_sharding(restored, item.get("original_sharding"))
        results[item["idx"]] = restored
      else:
        component_results[item["idx"]]["parts"].append(
            (item["component_index"], ortho[0].reshape(item["component_shape"]))
        )

  for idx, metadata in component_results.items():
    parts = [part for _, part in sorted(metadata["parts"], key=lambda x: x[0])]
    result_t = jnp.concatenate(parts, axis=-1).reshape(metadata["shape"])
    result = jnp.transpose(result_t, metadata["inv_perm"])
    result = _maybe_restore_sharding(result, metadata.get("original_sharding"))
    results[idx] = result

  return treedef.unflatten(results)


def scale_by_muon(
    ns_coeffs: Union[
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[
            tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
            ...,
        ],
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-7,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    ns_dtype: Optional[jax.typing.DTypeLike] = jnp.float32,
    *,
    nesterov: bool = True,
    nesterov_style: Literal["ema", "sgd"] = "sgd",
    adaptive: bool = False,
    preconditioning: Literal["frobenius", "spectral", "aol", "schatten"] = "frobenius",
    weight_dimension_numbers: WeightDimNumOrFn | None = None,
    batch_update: bool = True,
    batch_update_size: int | None = 16,
    ns_unroll: bool = False,
    replicate_ns_matrix_axes: bool = False,
    replicate_ns_batch_axis: str | None = None,
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
      Default 1e-7 matches Megatron-LM's hardcoded NS normalization epsilon.
    mu_dtype: Data type of the momentum accumulator.
    ns_dtype: Data type for Newton-Schulz orthogonalization. Input is cast to
      this dtype before NS iterations. The output stays in ns_dtype and flows
      through subsequent transforms; JAX promotes to fp32 at param update.
      Default ``jnp.float32`` to keep the five-step NS polynomial in fp32 and
      remove TPU/GPU bf16 matmul-order drift across accelerators. Set to
      ``jnp.bfloat16`` to mirror Megatron-LM's default (bf16 NS output scaled
      by LR in bf16, added to fp32 params), or ``None`` to skip casting (use
      the momentum dtype as-is).
    nesterov: Whether to use Nesterov momentum.
    nesterov_style: Style of momentum accumulation. 'ema' uses the optax
      default (EMA accumulation with bias correction). 'sgd' uses classic
      SGD-style momentum (buf = beta*buf + grad) with Nesterov look-ahead
      (g = grad + beta*buf) and no bias correction, matching Megatron-LM.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    preconditioning: What type of preconditioning to use before NS iterations.
      Available options are:
      - 'frobenius' (default): Use Frobenius rescaling before NS.
      - 'spectral' : Use Spectral norm rescaling before NS.
      - 'aol': Use AOL rescaling to improve orthogonality.
      - 'schatten': Use the Schatten-4 norm for rescaling.
    weight_dimension_numbers: An optional tree with the same structure as the
      params of `MuonDimensionNumbers`s, specifying how to reshape the
      parameters before and after the orthogonalization OR a callable returning
      such a tree. None implies that all parameters are 2D matrices.
    batch_update: If True, group parameters by their reshaped 2D matrix shape
      and stack them along the batch dimension before running Newton-Schulz.
      This amortizes overhead for many small matrices. Results are numerically
      identical to unbatched mode.
    batch_update_size: If set (and batch_update is True), chunk each shape
      group into batches of at most this size before running Newton-Schulz.
    replicate_ns_batch_axis: Mesh axis name for batch dim when
      replicate_ns_matrix_axes=True. Must be set when enabling the flag.

  Returns:
    A `GradientTransformation` object.

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
  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = optax.tree.zeros_like(params, dtype=mu_dtype)  # First moment
    ns_coeffs_ = jnp.asarray(ns_coeffs)

    if ns_coeffs_.ndim > 2 or ns_coeffs_.shape[-1] != 3:
      raise ValueError(f"ns_coeffs must have shape (3,) or (n, 3), got {ns_coeffs_.shape}")
    if ns_coeffs_.ndim == 2:
      if not ns_coeffs_.shape[0] <= ns_steps:
        raise ValueError(f"Not enough coeffs to perform {ns_steps} steps")
      ns_coeffs_ = ns_coeffs_[-ns_steps:]

    return MuonState(
        count=jnp.zeros([], jnp.int32),
        mu=mu,
        ns_coeffs=ns_coeffs_,
    )

  def update_fn(updates, state, params=None):
    del params
    # TODO(rdyro): extend to _masking._mask_callable
    if callable(weight_dimension_numbers):
      # Populate weight_dim_nums if it's a callable. Use updates instead of
      # actual params since only shapes matter and params may not be provided.
      resolved_weight_dim_nums = weight_dimension_numbers(updates)
    else:
      resolved_weight_dim_nums = weight_dimension_numbers

    if nesterov_style == "sgd":
      # Classic SGD-style momentum (Megatron-LM):
      #   buf = beta * buf + grad
      #   g   = grad + beta * buf    (Nesterov look-ahead, no bias correction)
      mu = jax.tree.map(lambda g, m: beta * m + g, updates, state.mu)
      if nesterov:
        mu_hat = jax.tree.map(lambda g, m: g + beta * m, updates, mu)
      else:
        mu_hat = mu
    else:
      # Default optax EMA-style momentum with bias correction
      mu = optax.tree.update_moment(updates, state.mu, beta, 1)
      count_inc = numerics.safe_increment(state.count)
      if nesterov:
        mu_hat = jax.tree.map(
            lambda m, g: beta * m + (1 - beta) * g,
            optax.tree.bias_correction(mu, beta, numerics.safe_increment(count_inc)),
            optax.tree.bias_correction(updates, beta, count_inc),
        )
      else:
        mu_hat = optax.tree.bias_correction(mu, beta, count_inc)

    count_inc = numerics.safe_increment(state.count)
    # Cast to ns_dtype before Newton-Schulz orthogonalization, then let the
    # update stay in ns_dtype through the rest of the optax chain
    # (scale_by_shape, add_decayed_weights, scale_by_learning_rate). JAX
    # promotes to fp32 only at param addition. Default ns_dtype is fp32 so
    # the five-step NS polynomial stays in fp32 — Megatron-LM casts to
    # bfloat16 here (g.bfloat16()) and keeps the update in bf16 through LR
    # scaling, which is available by passing ns_dtype=jnp.bfloat16.
    if ns_dtype is not None:
      ns_input = optax.tree.cast(mu_hat, ns_dtype)
    else:
      ns_input = mu_hat
    # Apply Newton-schulz orthogonalization.
    if batch_update:
      updates = _batch_orthogonalize_tree(
          ns_input,
          resolved_weight_dim_nums,
          state.ns_coeffs,
          ns_steps,
          preconditioning,
          eps,
          batch_update_size,
          ns_unroll,
          replicate_ns_matrix_axes,
          replicate_ns_batch_axis,
      )
    else:
      # Save original leaf shardings for restore after NS.
      if replicate_ns_matrix_axes:
        saved_shardings = jax.tree.map(_get_array_sharding, ns_input, is_leaf=_is_weight_dim_nums)
      updates = jax.tree.map(
          lambda x, dim_num: orthogonalize_via_newton_schulz(
              x,
              state.ns_coeffs,
              ns_steps,
              preconditioning,
              eps,
              dim_num,
              ns_unroll=ns_unroll,
              replicate_ns_matrix_axes=replicate_ns_matrix_axes,
              replicate_ns_batch_axis=replicate_ns_batch_axis,
          ),
          ns_input,
          resolved_weight_dim_nums,
          is_leaf=_is_weight_dim_nums,
      )
      if replicate_ns_matrix_axes:
        updates = jax.tree.map(_maybe_restore_sharding, updates, saved_shardings, is_leaf=_is_weight_dim_nums)
    if adaptive:
      # Scale the orthogonalized updates by the dual norm of the original
      # updates. See https://arxiv.org/abs/2409.20325 for the derivation.
      updates = jax.tree.map(lambda x, y: jnp.sum(x.conj() * y) * y, mu_hat, updates)

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
        tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
        tuple[
            tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.typing.ArrayLike],
            ...,
        ],
        str,
    ] = _DEFAULT_NS_COEFFS,
    ns_steps: jax.typing.ArrayLike = 5,
    beta: jax.typing.ArrayLike = 0.95,
    eps: jax.typing.ArrayLike = 1e-7,
    weight_decay: jax.typing.ArrayLike = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    mu_dtype: Optional[jax.typing.DTypeLike] = None,
    ns_dtype: Optional[jax.typing.DTypeLike] = jnp.float32,
    *,
    nesterov: bool = True,
    nesterov_style: Literal["ema", "sgd"] = "sgd",
    adaptive: bool = False,
    preconditioning: Literal["frobenius", "spectral", "aol", "schatten"] = "frobenius",
    adam_b1: jax.typing.ArrayLike = 0.9,
    adam_b2: jax.typing.ArrayLike = 0.999,
    adam_eps: jax.typing.ArrayLike | None = None,
    adam_eps_root: jax.typing.ArrayLike = 0.0,
    adam_nesterov: bool = False,
    adam_weight_decay: jax.typing.ArrayLike = 0.0,
    adam_learning_rate: base.ScalarOrSchedule | None = None,
    adam_weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    muon_weight_dimension_numbers: WeightDimNumOrFn | None = None,
    consistent_rms: jax.typing.ArrayLike | None = None,
    batch_update: bool = True,
    batch_update_size: int | None = 16,
    ns_unroll: bool = False,
    replicate_ns_matrix_axes: bool = False,
    replicate_ns_batch_axis: str | None = None,
) -> base.GradientTransformation:
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
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a tuple of tuples.
    beta: Decay rate for the exponentially weighted average of grads.
    eps: Term added to the denominator to improve numerical stability.
      Default 1e-7 matches Megatron-LM's hardcoded NS normalization epsilon.
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
    ns_dtype: Data type for Newton-Schulz orthogonalization. Input is cast to
      this dtype before NS iterations. The output stays in ns_dtype through
      LR scaling; JAX promotes to fp32 at param update. Default
      ``jnp.float32`` keeps the NS polynomial in fp32 to avoid TPU/GPU bf16
      matmul-order drift. Set to ``jnp.bfloat16`` to match Megatron-LM's
      bf16 NS, or ``None`` to skip casting.
    nesterov: Whether to use Nesterov momentum.
    nesterov_style: Style of momentum accumulation. 'ema' uses the optax
      default (EMA accumulation with bias correction). 'sgd' uses classic
      SGD-style momentum (buf = beta*buf + grad) with Nesterov look-ahead
      (g = grad + beta*buf) and no bias correction, matching Megatron-LM.
    adaptive: Whether to scale the updates by the dual norm of the
      original updates. See <https://arxiv.org/abs/2409.20325>
    preconditioning: What type of preconditioning to use before NS iterations.
      Available options are:
      - 'frobenius' (default): Use Frobenius rescaling before NS:
        safe, standard, but degrades orthogonalization quality when using
        less than 5 NS steps.
      - 'spectral' : Use Spectral norm rescaling before NS:
        much more computationally intensive, but better orthogonalization
        quality.
      - 'aol': Use AOL rescalings to improve orthogonality with little to
        no overhead, usually allows the user to remove one iterative NS step.
        See <https://arxiv.org/abs/2512.04632>.
      - 'schatten': Use the Schatten-4 norm for rescaling,
        allows for better performance with little to no extra cost.
        See <https://arxiv.org/abs/2506.10935>.
    adam_b1: Exponential decay rate for Adam's first moment estimates.
    adam_b2: Exponential decay rate for Adam's second moment estimates.
    adam_nesterov: Whether to use Nesterov momentum in the Adam partition.
      Default ``False`` to match Megatron-LM's standard Adam (no Nesterov).
    adam_eps_root: Epsilon to stabilize division in Adam, square root version.
    adam_weight_decay: Weight decay factor for Adam.
    adam_learning_rate: Auxiliary learning rate for the Adam optimizer.
      If `None`, the learning rate for Adam defaults to the same as Muon.
    adam_weight_decay_mask: A tree with same structure as (or a prefix of) the
      params PyTree, or a Callable that returns such a pytree given the params.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply Adam weight decay to, and `False` for those you want to skip.
    muon_weight_dimension_numbers: An optional tree of `MuonDimensionNumbers`s,
      specifying how to reshape the parameters for orthogonalization otherwise
      muon parameters are assumed to be 2D matrices. A `None` value indicates
      that the parameter is not a muon parameter and will be optimized with
      Adam. A callable takes as input the params and returns a possibly masked
      pytree of specs, similar to `weight_decay_mask`. If not provided, muon is
      applied to all 2D parameters.
    consistent_rms: An optional float to activate consistent RMS scaling.
      Scales updates by `sqrt(max(fan_in, fan_out)) * consistent_rms` to make
      root mean square (RMS) shape-independent, like AdamW. `0.2` is recommended
      to match AdamW's empirical RMS. See <https://arxiv.org/abs/2502.16982>.
      If `None`, uses width scaling `sqrt(max(1, fan_out / fan_in))`.
    batch_update: If True, group same-shape Muon parameters and run
      Newton-Schulz on stacked batches for efficiency. Numerically identical
      to unbatched mode.
    batch_update_size: Maximum batch size per NS call when batch_update is True.
      If None, all same-shape parameters are stacked into one batch.

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
    param_labels = lambda params: jax.tree.map(lambda x: "muon" if x.ndim == 2 else "adam", params)
    muon_weight_dimension_numbers = MuonDimensionNumbers()
  else:

    def param_labels(params):
      dim_nums = (
          muon_weight_dimension_numbers(params)
          if callable(muon_weight_dimension_numbers)
          else muon_weight_dimension_numbers
      )
      populate_subtree_ = lambda dim_num, x: jax.tree.map(lambda y: "muon" if dim_num is not None else "adam", x)
      # Dimension numbers come first since they can be a prefix mask.
      return jax.tree.map(populate_subtree_, dim_nums, params, is_leaf=lambda x: x is None or _is_weight_dim_nums(x))

  # We need to normalize the dimension numbers because they have to match the
  # tree structure of the masked muon state tree (see `combine.partition`).
  def muon_weight_dim_nums_fn(params):
    # if muon_weight_dimension_numbers is None:
    #   return None
    # Normalize the dimension numbers for `combine.partition`.
    # Insert MaskedNode() where muon state will be masked out.
    dim_nums = (
        muon_weight_dimension_numbers(params)
        if callable(muon_weight_dimension_numbers)
        else muon_weight_dimension_numbers
    )
    mask = jax.tree.map(lambda label: label == "muon", param_labels(params))
    is_leaf = lambda x: (x is None or _is_weight_dim_nums(x) or isinstance(x, _masking.MaskedNode))
    populate_subtree_ = lambda dim_nums, submask: jax.tree.map(
        lambda m: dim_nums if m else _masking.MaskedNode(), submask
    )
    return jax.tree.map(populate_subtree_, dim_nums, mask, is_leaf=is_leaf)

  return combine.partition(
      transforms={
          "muon": combine.chain(
              scale_by_muon(
                  ns_coeffs=ns_coeffs_,
                  ns_steps=ns_steps,
                  beta=beta,
                  eps=eps,
                  mu_dtype=mu_dtype,
                  ns_dtype=ns_dtype,
                  nesterov=nesterov,
                  nesterov_style=nesterov_style,
                  adaptive=adaptive,
                  preconditioning=preconditioning,
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
                  batch_update=batch_update,
                  batch_update_size=batch_update_size,
                  ns_unroll=ns_unroll,
                  replicate_ns_matrix_axes=replicate_ns_matrix_axes,
                  replicate_ns_batch_axis=replicate_ns_batch_axis,
              ),
              scale_by_shape(
                  weight_dimension_numbers=muon_weight_dim_nums_fn,
                  consistent_rms=consistent_rms,
              ),
              transform.add_decayed_weights(weight_decay, weight_decay_mask),
              transform.scale_by_learning_rate(learning_rate),
          ),
          "adam": alias.adamw(
              learning_rate=adam_learning_rate,
              b1=adam_b1,
              b2=adam_b2,
              eps=adam_eps if adam_eps is not None else eps,
              eps_root=adam_eps_root,
              weight_decay=adam_weight_decay,
              mask=adam_weight_decay_mask,
              mu_dtype=mu_dtype,
              nesterov=adam_nesterov,
          ),
      },
      param_labels=param_labels,
  )
