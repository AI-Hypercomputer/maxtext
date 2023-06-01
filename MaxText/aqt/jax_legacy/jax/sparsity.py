# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic functionalities for pruning neural networks implemented in jax."""

import dataclasses
import enum
import functools
import math
import typing
from typing import Tuple, Union

from absl import logging
from aqt.jax_legacy.jax.flax import struct as flax_struct
from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp


dataclass = (
    flax_struct.dataclass if not typing.TYPE_CHECKING else dataclasses.dataclass
)


# TODO(ayazdan): Create abstract class for sparsity configurations.
class SparseType(str, enum.Enum):
  """Pruning types dataclass."""
  STRUCTURED_NM = 'STRUCTURED_NM'
  UNSTRUCTURED = 'UNSTRUCTURED'


@dataclass
class SparseHParams:
  """Hyper parameters for sparsity.

  Attributes:
    type: Input array for which pruning mask is computed.
    prune_rate: Pruning rate.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
      `C` and `R` indicate column-wise and row-wise ordering, respectively.
    absolute: If True, the absolute value of the values are used for sorting.
    smallest: If True, the smallest values in inputs are masked.
    structure_decay: If True, a decaying mechanism is applied on the structure.
    mask_decay_weight: If 0.0, no mask decay is applied. The mask value
      start with 1.0 and each time `num_update_sparsity` * `mask_decay_weight`
      is subtracted from 1.0. Due to overhead of jit, we limited the number of
      updates to `num_update_sparsity` to 16. After 16 iterations, we forcefully
      set `mask_decay_value` to zero. Mask decaying works for both structured
      and unstructured sparsity.
    sparse_ste: If True, a sparse-refined straight-through estimator (SR-STE)
      is applied, following the algorithm described in:
        https://arxiv.org/abs/2102.04010
    sparse_ste_weight: Denotes the relative weight for the sparse-refined term.
      As mentioned in the paper (https://arxiv.org/abs/2102.04010), the best
      default value is 0.0002 (lambda_w in the paper).
  """

  type: SparseType
  # Prune_rate data type varies with SparseHParams::type.
  # float: If SparseHParams::type is UNSTRUCTURED.
  # Tuple[int]: If SparseHParams::type is STRUCTURED_NM,  Tuple of type N, M.
  prune_rate: Union[None, float, Tuple[int, int]]
  order: str = 'R'
  absolute: bool = True
  smallest: bool = True
  structure_decay: bool = False
  mask_decay_weight: float = 0.0
  sparse_ste: bool = False
  sparse_ste_weight: float = 0.0002

  def __post_init__(self):
    if self.prune_rate is not None:
      if self.type == SparseType.STRUCTURED_NM:
        assert isinstance(self.prune_rate,
                          Tuple), ('prune rate should be either '
                                   'None for no pruning or a '
                                   'Tuple (N, M) for '
                                   'STRUCTURED_NM sparsity')
      elif self.type == SparseType.UNSTRUCTURED:
        assert isinstance(self.prune_rate,
                          float), ('prune rate should be either '
                                   'None for no pruning or float for '
                                   'UNSTRUCTURED sparsity')
      else:
        assert False, 'prune rate unknown!'

      assert self.mask_decay_weight >= 0.0, (
          'Invalid value for '
          f'{self.mask_decay_weight}. '
          '`mask_decay_weight` must be positive.')

      if self.sparse_ste:
        if self.mask_decay_weight != 0.0:
          raise ValueError('SR-STE only works with non-decaying mask.')
        if self.structure_decay:
          raise ValueError(
              'SR-STE only works with non-decaying sparse structure.')
        if self.type != SparseType.STRUCTURED_NM:
          raise ValueError('SR-STE only works with structured sparsity.')


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def sr_ste(inputs: jnp.ndarray,
           mask: jnp.ndarray,
           update_mask: bool,
           apply_mask: bool,
           sparsity_hparams: SparseHParams,
           n_sparsity: int = 0,
           m_sparsity: int = 0):
  """Wrapper function for custom derivative rule for structured sparsity.

  Algorithm description: https://arxiv.org/abs/2102.04010

  The last three arguments are forced to be static to simplify
    the implementation.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    mask: The mask matrix which defines which elements to be pruned.
    update_mask: If True, the mask pattern gets updated.
    apply_mask: If True, the mask is applied to input.
    sparsity_hparams: The hyperparmeters related to sparsity.
    n_sparsity: Integer value for N in N:M sparsity.
    m_sparsity: Integer value for M in N:M sparsity.

  Returns:
    The updated input values after applying sparsity.
  """

  return sr_ste_fwd(
      inputs=inputs,
      mask=mask,
      update_mask=update_mask,
      apply_mask=apply_mask,
      sparsity_hparams=sparsity_hparams,
      n_sparsity=n_sparsity,
      m_sparsity=m_sparsity)[0]


@functools.partial(jax.jit, static_argnums=(4, 5, 6))
def sr_ste_fwd(inputs: jnp.ndarray,
               mask: jnp.ndarray,
               update_mask: bool,
               apply_mask: bool,
               sparsity_hparams: SparseHParams,
               n_sparsity: int = 0,
               m_sparsity: int = 0) -> jnp.ndarray:
  """Custom forward pass for structured sparsity."""
  # pylint:disable=g-long-lambda
  updated_mask = jax.lax.cond(
      update_mask, lambda: get_sparsity_mask(
          inputs, sparsity_hparams, n_sparsity, m_sparsity), lambda: mask)
  updated_inputs = jax.lax.cond(apply_mask,
                                lambda: jnp.multiply(updated_mask, inputs),
                                lambda: inputs)
  # pylint:enable=g-long-lambda
  return (updated_inputs, updated_mask,  # pytype: disable=bad-return-type  # jax-ndarray
          jnp.array(SparseHParams.sparse_ste_weight)), (
              inputs, updated_mask, jnp.array(SparseHParams.sparse_ste_weight))


def sr_ste_bwd(sparsity_hparams, n_sparsity, m_sparsity, res, g):
  """Implements custom gradient for backward pass.

  Args:
    sparsity_hparams: Non-diff arguments as defined in `sr_ste`.
    n_sparsity: Non-diff arguments as defined in `sr_ste`.
    m_sparsity: Non-diff arguments as defined in `sr_ste`.
    res: Residuals computed in sr_ste_fwd.
    g: Default calculated gradients.

  Returns:
    Gradients for differentiable inputs:
      - inputs
      - mask
      - update_mask
      - apply_mask
  """
  del sparsity_hparams, n_sparsity, m_sparsity
  inputs, updated_mask, ste_weight = res
  # g contains a list of gradients, one per output.
  # g1: updated_inputs
  g1, _, _ = g
  g1 = g1 + ste_weight * jnp.multiply(~updated_mask, inputs)
  return (g1, None, None, None)


sr_ste.defvjp(sr_ste_fwd, sr_ste_bwd)


class Sparsity(nn.Module):
  """Abstract class sparsity for applying sparsity."""

  sparsity_hparams: SparseHParams

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               *,
               update_mask: bool,
               apply_mask: bool,
               num_update_sparsity: int = 0) -> jnp.ndarray:

    # TODO(shivaniagrawal): make a decision on creating/not creating mask for
    # when sparsity hparams is None itself.
    if (self.sparsity_hparams is None or
        self.sparsity_hparams.prune_rate is None):
      return inputs

    if self.sparsity_hparams.type == 'STRUCTURED_NM':
      n_sparsity = self.sparsity_hparams.prune_rate[0]
      m_sparsity = self.sparsity_hparams.prune_rate[1]
      if self.sparsity_hparams.structure_decay:
        if num_update_sparsity == 1:
          n_sparsity = n_sparsity - 1
        else:
          n_sparsity = int(
              math.ceil(n_sparsity / math.pow(2, num_update_sparsity)))
    else:
      logging.info('Unstructured sparsity does not support structure decaying.')
      n_sparsity = 0
      m_sparsity = 0

    # Due to overhead of jit, we limited the number of updates to
    # `num_update_sparsity` to 16. Once we reach to 16, we forcefully set
    # `mask_decay_value` to zero.
    # TODO(ayazdan): Support more than 16 decay.
    mask_decay_value = 1.0
    if self.sparsity_hparams.mask_decay_weight != 0.0:
      if num_update_sparsity < 16:
        mask_decay_value = max(
            mask_decay_value -
            (num_update_sparsity * self.sparsity_hparams.mask_decay_weight),
            0.0)
      else:
        mask_decay_value = 0.0
    mask = self.variable('sparsity', 'mask', jnp.ones, inputs.shape, jnp.bool_)

    if self.sparsity_hparams.sparse_ste:
      updated_inputs, updated_mask, _ = sr_ste(
          inputs=inputs,
          mask=mask.value,
          update_mask=update_mask,
          apply_mask=apply_mask,
          sparsity_hparams=self.sparsity_hparams,
          n_sparsity=n_sparsity,
          m_sparsity=m_sparsity)
      if update_mask and self.has_variable('sparsity', 'mask'):
        mask.value = updated_mask
      return updated_inputs
    else:
      if update_mask and self.has_variable('sparsity', 'mask'):
        mask.value = get_sparsity_mask(inputs, self.sparsity_hparams,
                                       n_sparsity, m_sparsity)

      if apply_mask and self.has_variable('sparsity', 'mask'):
        if self.sparsity_hparams.mask_decay_weight != 0.0:
          return jnp.multiply(~mask.value * mask_decay_value + mask.value,
                              inputs)
        else:
          return jnp.where(mask.value, inputs,
                           jnp.zeros(inputs.shape, inputs.dtype))
      return inputs


def apply_sparsity(
    inputs: jnp.ndarray,
    sparsity_hparams: SparseHParams,
    n_sparsity: int,
    m_sparsity: int,
) -> jnp.ndarray:
  """Returns sparsified inputs based on sparsity hparams."""
  mask = get_sparsity_mask(inputs, sparsity_hparams, n_sparsity, m_sparsity)
  return jnp.where(mask.value, inputs, jnp.zeros(inputs.shape, inputs.dtype))  # pytype: disable=attribute-error  # jax-ndarray


def get_sparsity_mask(
    inputs: jnp.ndarray,
    sparsity_hparams: SparseHParams,
    n_sparsity: int = 0,
    m_sparsity: int = 0,
) -> jnp.ndarray:
  """Returns sparsified inputs based on sparsity hparams."""
  if sparsity_hparams is None or sparsity_hparams.prune_rate is None:
    return jnp.ones(inputs.shape, dtype=bool)
  prune_rate = sparsity_hparams.prune_rate
  if sparsity_hparams.type == 'STRUCTURED_NM':
    assert isinstance(
        prune_rate, Tuple), 'prune rate must be tuple for structured sparsity.'
    assert prune_rate[0] <= prune_rate[1], (
        'prune_rate[0] must be lower than prune_rate[1] for N:M'
        f' ({prune_rate[0]}:{prune_rate[1]}) sparsity.'
    )
    return get_pruning_n_m_mask(
        inputs,
        n=n_sparsity,
        m=m_sparsity,
        order=sparsity_hparams.order,
        absolute=sparsity_hparams.absolute,
        smallest=sparsity_hparams.smallest)
  elif sparsity_hparams.type == 'UNSTRUCTURED':
    assert (
        isinstance(prune_rate, float) and prune_rate < 1
    ), f'sparsity ratio can not be > 1, provided prune_rate {prune_rate}.'
    return get_pruning_unstruct_mask(
        inputs, prune_rate=prune_rate, smallest=sparsity_hparams.smallest)
  else:
    raise ValueError(f'invalid sparsity type {sparsity_hparams.type}!')


# TODO(ayazdan): Support arrays with length not divisible by `m`.
def prune_inputs_n_m(inputs: jnp.ndarray,
                     *,
                     n: int,
                     m: int,
                     order: str = 'R',
                     offset: int = 0,
                     absolute: bool = True,
                     smallest: bool = True) -> jnp.ndarray:
  """Returns pruned array with N:M (structured) pruning.

  N:M pruning makes at most N values non-zero in each block of M consecutive
  values.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n: Maximum number of non-zero values in each block.
    m: Number of values in each block.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
    `C` -> Column-wise pruning.
    `R` -> Row-wise pruning.
    offset: Indicates the offset between the group of M elements on which
      N:M sparsity is applied. The default is `0` (narrowly-separate),
      indicating that `M` elements are selected from adjacent values in
      the input matrix. Generally, because of the XLA layout
      (lanes 128/sublanes 8), another value for offset would be 128
      (widely-separated).
      If offset > 0, we only support scenarios where the input array size is
      equal to (offset * m).
    absolute: If True, the absolute value of the values are used for sorting.
    smallest: If True, the smallest values in inputs are masked.

  Returns:
    An array with the same shape as inputs pruned with N:M strategy.
  """
  mask = get_pruning_n_m_mask(
      inputs,
      n,
      m,
      order=order,
      offset=offset,
      absolute=absolute,
      smallest=smallest)
  return jnp.where(mask, inputs, jnp.zeros(inputs.shape, inputs.dtype))


# TODO(ayazdan): Add supports for magnitude pruning.
# TODO(ayazdan): Add supports for column-wise pruning.
# TODO(ayazdan): Add for other form of N:M pruning.
# TODO(ayazdan): Add parameter `r` for decaying weights.
def prune_2_4(inputs: jnp.ndarray) -> jnp.ndarray:
  """Returns pruned array w/ 2:4 (structured) pruning (w/ sorting networks).

  Args:
    inputs: Input array for which N:M pruning mask is computed.

  Returns:
    An array with the same shape as inputs pruned with 2:4 strategy.
  """
  def _cswap(a, b, cond):
    """Conditional swap of two vector."""
    return jnp.where(cond, a, b), jnp.where(cond, b, a)

  def _prune(a, n, r):
    """Prunes the first n elements with a decaying factor of (1-r).

    Args:
      a: Input array on which pruning is applied.
      n: The first number of elements on which mask is applied.
      r: The decaying factor for sparsification.

    Returns:
      A sparsified array with the same shape as `a`.
    """
    for i in range(n):
      a[i] = a[i] * (1 - r)
    return a

  def _preprocessing(xin, m=4):
    """Reorganizes the input vector for sorting.

    Example:
      Input:
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)
      Output:
        (0, 0) - (0, 4)
        (0, 1) - (0, 5)
        (0, 2) - (0, 6)
        (0, 3) - (0, 7)

    Args:
      xin: Input array on which the reorganizatin occurs.
      m: Numner of adjacent elements to group.

    Returns:
      A reorganized array with the same total size as input.
      In addition, it returns `start` and `end` indices of each dimension.
      Finally, it returns `red_dim` which indicates the last dimension of input
      array divided by group size (m).
    """
    red_dim = int(jnp.shape(xin)[-1] / m)
    start = [0] * len(jnp.shape(xin))
    end = list(jnp.shape(xin))
    strides = [1] * (len(jnp.shape(xin)) - 1) + [m]
    xs = []
    for i in range(m):
      start_i = start
      start_i[-1] = i
      xs.append(
          jnp.reshape(lax.slice(xin, start_i, end, strides), [-1, red_dim]))
    return xs, start, end, red_dim

  def _postprocessing(xin, start, end, red_dim, shape_in):
    """Reorganizes the input vector back to its original shape.

    Example:
      Input:
        (0, 0) - (0, 4)
        (0, 1) - (0, 5)
        (0, 2) - (0, 6)
        (0, 3) - (0, 7)
      Output:
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)

    Args:
      xin: Input array on which the reorganizatin occurs.
      start: The starting indices of each dimension.
      end: The ending indices of each dimension.
      red_dim: Indicates the last dimension of input array divided
        by group size (m).
      shape_in: The original shape of the input array.

    Returns:
      A reorganized array back to its original shape after sparsification.
    """

    filtered = jax.numpy.reshape(xin, shape_in)
    slices = []
    strides = [1] * (len(shape_in) - 1) + [red_dim]
    for i in range(red_dim):
      start_i = start
      start_i[-1] = i
      slices.append(lax.slice(filtered, start_i, end, strides))
    return jax.numpy.reshape(jax.numpy.stack(slices, axis=-2), shape_in)

  # pre-processing: preprare the input array for sorting network.
  xs, start, end, red_dim = _preprocessing(inputs)
  comparators = [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
  comparator_results = []
  # forward pass through sorting network.
  for x, y in comparators:
    l = jnp.less(xs[x], xs[y])
    comparator_results.append(l)
    xs[x], xs[y] = _cswap(xs[x], xs[y], l)
  # apply pruning.
  pruned = _prune(a=xs, n=2, r=1.0)
  # backward pass through sorting network.
  for xy, cond in reversed(list(zip((comparators), comparator_results))):
    x, y = xy
    pruned[x], pruned[y] = _cswap(pruned[x], pruned[y], cond)
  xs_final = jnp.array(jnp.concatenate(pruned, axis=1), dtype=xs[0].dtype)
  # post-processing: reshape the array to its original form.
  return _postprocessing(xs_final, start, end, red_dim, jax.numpy.shape(inputs))


def get_pruning_n_m_mask(inputs: jnp.ndarray,
                         n: int,
                         m: int,
                         *,
                         order: str = 'R',
                         offset: int = 0,
                         absolute: bool = True,
                         smallest: bool = True) -> jnp.ndarray:
  """Returns a mask for N:M (structured) pruning.

  N:M pruning makes at most N values non-zero in each block of M consecutive
  values.

  Args:
    inputs: Input array for which N:M pruning mask is computed.
    n: Maximum number of non-zero values in each block.
    m: Number of values in each block.
    order: Apply pruning using this index order. Supported values are `C`, `R`.
      `C` and `R` indicate column-wise and row-wise masking, respectively.
      Default is `R` indicating to applying N:M sparsity across rows of
      the input matrix.
    offset: Indicates the offset between the group of M elements on which
      N:M sparsity is applied. The default is `0` (narrowly-separate),
      indicating that `M` elements are selected from adjacent values in
      the input matrix. Generally, because of the XLA layout
      (lanes 128/sublanes 8), another value for offset would be 128
      (widely-separated).
      If offset > 0, we only support scenarios where the input array size is
      equal to (offset * m).
    absolute: If True, the absolute value of the values are used for sorting.
    smallest: If True, the smallest values in inputs are masked.

  Returns:
    A mask that indicates the pruning locations (`0`: no pruning, `1`: pruned).
  """
  if n > m:
    raise ValueError(f'N must be lower than M for N:M ({n}:{m}) sparsity.')
  if order not in ['C', 'R']:
    raise ValueError(f'Index order {order} not supported.')
  if offset < 0:
    raise ValueError('Offset value must be positive. '
                     f'You provided {offset}.')
  if offset != 0 and offset != 128:
    logging.warning(
        'Value %d may not be best optimized for the memory layout. '
        'We suggest a value of `0` or `128` for offset.', offset
    )
  length = jnp.size(inputs)
  # TODO(b/228458062): support m which is not a factor of inputs size.
  if length % m != 0:
    raise ValueError(
        f'inputs size must be divisible by m, provided {length} and {m}')
  group = int(length / m)
  inputs = jnp.abs(inputs) if absolute else inputs
  if order == 'R':
    if offset == 0:
      inputs_temp = inputs.reshape(group, m, order='C')
    else:
      if offset * m != length:
        raise ValueError('When offset > 0, we only support an array size '
                         '(length) equal to (offset * m). '
                         f'Provided offset = {offset}, m = {m}, '
                         f'length = {length}.')
      inputs_temp = jnp.transpose(inputs.reshape(m, offset, order='C'))
  else:
    inputs_temp = jnp.einsum('...ij->...ji', inputs).reshape(
        group, m, order='C')
  mask = jnp.ones(inputs_temp.shape, dtype=bool)
  # Extract the smallest elements and forcefully make them zero.
  _, top_k_indices = jax.lax.top_k(
      -inputs_temp, k=m - n) if smallest else jax.lax.top_k(
          inputs_temp, k=m - n)
  mask = jax.vmap(lambda x, i: x.at[i].set(False))(mask, top_k_indices)
  if order == 'R':
    if offset == 0:
      return mask.reshape(inputs.shape, order='C')
    else:
      return jnp.transpose(mask).reshape(inputs.shape, order='C')
  else:
    if len(inputs.shape) > 2:
      return jnp.einsum('...ij->...ji', mask.reshape(inputs.shape, order='F'))
    else:
      return jnp.einsum('ij->ji', mask).reshape(inputs.shape, order='F')


def get_pruning_unstruct_mask(inputs: jnp.ndarray,
                              *,
                              prune_rate: float = 0.1,
                              smallest: bool = True) -> jnp.ndarray:
  """Returns a mask for pruning according to the prune rate.

  Args:
    inputs: Input array for which pruning mask is computed.
    prune_rate: Pruning rate. The ratio of the elements that are pruned. 0
      meaning no pruning. Defaults to 0.1.
    smallest: If True, the smallest values in inputs are masked.

  Returns:
    A mask that indicates the pruning locations.
  """
  inputs = -jnp.abs(inputs) if smallest else jnp.abs(inputs)
  mask = jnp.ones(inputs.shape, dtype=bool)
  k = int(inputs.size * prune_rate)
  _, idxs = jax.lax.top_k(inputs.reshape(-1), k)
  return mask.reshape(-1).at[idxs].set(False).reshape(inputs.shape)


def prune_inputs_unstruct(inputs: jnp.ndarray,
                          *,
                          prune_rate: float = 0.1,
                          smallest: bool = True) -> jnp.ndarray:
  """Returns unstructured pruning for inputs according to the prune rate.

  Args:
    inputs: Input array which is being pruned.
    prune_rate: Pruning rate. The ratio of the elements that are pruned. 0
      meaning no pruning. Defaults to 0.1.
    smallest: If True, the smallest values in inputs are masked.

  Returns:
    Pruned input.
  """
  mask = get_pruning_unstruct_mask(
      inputs, prune_rate=prune_rate, smallest=smallest)
  return jnp.where(mask, inputs, jnp.zeros(inputs.shape, inputs.dtype))
