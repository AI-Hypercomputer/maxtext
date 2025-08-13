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


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
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
    x: A matrix or batch of matrices to orthogonalize. This function is batch-aware and can process 2D (m, n) or 3D (b, m, n)
  tensors.
    ns_coeffs: Coefficients for the Newton-schulz iterators.
      Must have shape (n, 3) where n is the number of iterations.
    ns_steps: Number of Newton-schulz iterations.
      Ignored if `ns_coeffs` is a 2D array.
    eps: Term added to denominators to improve numerical stability.

  Returns:
    The orthogonalized matrix or batch of matrices.
  """
  # MODIFIED: Support 2D or 3D (batched) tensors.
  if x.ndim != 2 and x.ndim != 3:
    raise ValueError(f"Input must have shape (m, n) or (b, m, n), got {x.shape}")
  if ns_coeffs.ndim > 2 or ns_coeffs.shape[-1] != 3:
    raise ValueError(
        'Newton-Schulz coefficients must have shape (3,) or (n, 3), '
        f'got {ns_coeffs.shape}'
    )

  original_ndim = x.ndim
  if original_ndim == 2:
    x = x[None, ...]  # Add dummy batch dimension

  def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
    # MODIFIED: Use swapaxes for batch-aware transpose.
    a = x @ x.swapaxes(-2, -1)
    b = coeffs[1] * a + coeffs[2] * a @ a
    return coeffs[0] * x + b @ x

  transposed = False
  # MODIFIED: Use negative axes for batch-aware shape checking.
  if x.shape[-2] > x.shape[-1]:
    x = x.swapaxes(-2, -1)
    transposed = True

  # MODIFIED: Normalize each matrix in the batch by its own norm.
  x /= jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps  # Ensure spectral norm is at most 1
  ns_coeffs = ns_coeffs.astype(x.dtype)
  if ns_coeffs.ndim == 1:
    x = jax.lax.fori_loop(
        0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs), x
    )
  else:
    x, _ = jax.lax.scan(
        lambda x, abc: (newton_schulz_iterator(x, abc), None), x, ns_coeffs
    )
  if transposed:
    x = x.swapaxes(-2, -1)

  if original_ndim == 2:
    x = x[0]  # Squeeze dummy batch dimension
  return x


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
    # MODIFIED: Added muon_spec argument.
    muon_spec: Optional[base.Params] = None,
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

    # --- START: ADDED: Reshaping logic using muon_spec. ---
    def _reshape_for_muon(param, spec):
      if spec is None:
        return param, None
      batch_axes, row_axes, col_axes = spec["batch_axes"], spec["rows"], spec["columns"]
      perm = (*batch_axes, *row_axes, *col_axes)
      inv_perm = jnp.argsort(jnp.array(perm))
      param_permuted = jnp.transpose(param, perm)
      batch_dim = jnp.prod(jnp.array([param.shape[i] for i in batch_axes])).astype(int)
      row_dim = jnp.prod(jnp.array([param.shape[i] for i in row_axes])).astype(int)
      col_dim = jnp.prod(jnp.array([param.shape[i] for i in col_axes])).astype(int)
      reshaped_param = param_permuted.reshape((batch_dim, row_dim, col_dim))
      restore_info = (param.shape, inv_perm)
      return reshaped_param, restore_info

    def _restore_from_muon(param, restore_info):
      if restore_info is None:
        return param
      original_shape, inv_perm = restore_info
      permuted_shape = jnp.transpose(jnp.empty(original_shape), jnp.argsort(inv_perm)).shape
      param_permuted = param.reshape(permuted_shape)
      return jnp.transpose(param_permuted, inv_perm)

    # --- END: ADDED: Reshaping logic. ---

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

    # MODIFIED: Reshape before orthogonalization 
    mu_hat, restore_infos = jax.tree.map(_reshape_for_muon, mu_hat, muon_spec)
    # Apply Newton-schulz orthogonalization.
    updates = jax.tree.map(
        lambda x: orthogonalize_via_newton_schulz(
            x, state.ns_coeffs, ns_steps, eps
        ),
        mu_hat,
    )
    # MODIFIED: Restore after orthogonalization
    updates = jax.tree.map(_restore_from_muon, updates, restore_infos)
    if adaptive:
      # Scale the orthogonalized updates by the dual norm of the original
      # updates. See https://arxiv.org/abs/2409.20325 for the derivation.
      updates = jax.tree.map(
        lambda x, y: jnp.einsum("...ij,...ij,...ab->...ab", x, y, y), mu_hat, updates
      )
    updates = jax.tree.map(
        lambda x: jnp.sqrt(jnp.maximum(1, x.shape[-1] / x.shape[-2])) * x,
        updates,
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
    # MODIFIED: Added muon_weight_mask (robert's change)
    muon_weight_mask: Callable[[base.Params], Any] | base.Params | None = None,
    # MODIFIED: Added muon_spec arguments.
    muon_spec: Optional[base.Params] = None,
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
    muon_spec: A PyTree with the same structure as params, where leaves
      are dicts like `{'batch_axes': (0,), 'rows':(1,), 'columns':(2,3)}`
      specifying how to reshape/permute N-D params (N>=3) into 3D for Muon.

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
  elif callable(muon_weight_mask):
    # mask comes first since it can be a prefix tree
    # no-op map over parameters to ensure same structure as params
    param_labels = lambda params: jax.tree.map(lambda m, x: "muon" if m else "adam", muon_weight_mask(params), params)
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
                  # MODIFIED: Pass muon_spec down.
                  muon_spec=muon_spec,
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