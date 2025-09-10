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
from optax.transforms import _masking
import optax.tree

ReshapeFn = Callable[[jax.Array], jax.Array]


class MuonWeightSpec(NamedTuple):
  reduction_axes: tuple[int, ...] | int
  output_axes: tuple[int, ...] | int

is_weight_spec = lambda x: isinstance(x, MuonWeightSpec)


def _normalize_axes(x: jax.Array, spec: MuonWeightSpec
                    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Normalize the axes in a muon spec to two tuples of non-negative ints."""
  if not isinstance(spec.reduction_axes, (list, tuple)):
    spec = spec._replace(reduction_axes=(spec.reduction_axes,))
  reduction_axes = tuple(ax % x.ndim for ax in spec.reduction_axes)

  if not isinstance(spec.output_axes, (list, tuple)):
    spec = spec._replace(output_axes=(spec.output_axes,))
  output_axes = tuple(ax % x.ndim for ax in spec.output_axes)
  return reduction_axes, output_axes


def _compute_muon_reshape(x: jax.Array, spec: MuonWeightSpec
                          ) -> tuple[ReshapeFn, ReshapeFn]:
  """Compute the reshape and inverse functions for an array from a spec."""
  if spec is None:
    return x
  reduction_axes, output_axes = _normalize_axes(x, spec)
  if set(reduction_axes) & set(output_axes):
    raise ValueError(
        'Reduction axes and output axes must be disjoint, got '
        f'{reduction_axes} and {output_axes}')
  batch_axes = tuple(sorted(set(range(x.ndim)) - set(reduction_axes)
                            - set(output_axes)))
  # transpose = batch_axes + output_axes + reduction_axes
  # inv_transpose = tuple(sorted(range(x.ndim), key=lambda i: transpose[i]))
  # axes2shape = lambda axes: tuple(x.shape[ax] for ax in axes)
  # flat_shape = (math.prod(axes2shape(batch_axes)),
  #               math.prod(axes2shape(output_axes)),
  #               math.prod(axes2shape(reduction_axes)))
  # unflat_shape = (axes2shape(batch_axes) + axes2shape(output_axes)
  #                 + axes2shape(reduction_axes))
  # revision 2
  transpose = batch_axes + reduction_axes + output_axes
  inv_transpose = tuple(sorted(range(x.ndim), key=lambda i: transpose[i]))
  axes2shape = lambda axes: tuple(x.shape[ax] for ax in axes)
  flat_shape = (math.prod(axes2shape(batch_axes)),
                math.prod(axes2shape(reduction_axes)),
                math.prod(axes2shape(output_axes)),)
  unflat_shape = (axes2shape(batch_axes) + axes2shape(reduction_axes) + axes2shape(output_axes))
  reshape_fn = lambda x: x.transpose(transpose).reshape(flat_shape)
  inverse_fn = lambda x: x.reshape(unflat_shape).transpose(inv_transpose)
  return reshape_fn, inverse_fn


def _shape_factor(x: jax.Array, spec: MuonWeightSpec) -> float:
  reduction_axes, output_axes = _normalize_axes(x, spec)
  return math.prod(x.shape[ax] for ax in output_axes) / math.prod(
      x.shape[ax] for ax in reduction_axes)


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
    spec: MuonWeightSpec | _masking.MaskedNode | None = None,
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
  if x.ndim != 2 and not isinstance(spec, MuonWeightSpec):
    raise ValueError(f'Input must have shape (m, n), got {x.shape} or the spec'
                     f' must be provided. Got spec {spec}')
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
      x = jax.lax.fori_loop(
          0, ns_steps, lambda _, x: newton_schulz_iterator(x, ns_coeffs_), x)
    else:
      x, _ = jax.lax.scan(
          lambda x, abc: (newton_schulz_iterator(x, abc), None), x, ns_coeffs_
      )
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
      # TODO
      print(weight_specs)
      print(mu_hat)
      updates = jax.tree.map(
          lambda x, spec: orthogonalize_via_newton_schulz(x, state.ns_coeffs,
                                                          ns_steps, eps, spec),
          mu_hat, weight_specs, is_leaf=is_weight_spec)
    else:
      updates = jax.tree.map(lambda x: orthogonalize_via_newton_schulz(
          x, state.ns_coeffs, ns_steps, eps), mu_hat)
    if adaptive:
      # Scale the orthogonalized updates by the dual norm of the original
      # updates. See https://arxiv.org/abs/2409.20325 for the derivation.
      updates = jax.tree.map(
          # lambda x, y: jnp.einsum('ij,ij,ab->ab', x, y, y), mu_hat, updates
          lambda x, y: jnp.sum(x * y) * y, mu_hat, updates
      )
    if weight_specs is not None:
      factors = jax.tree.map(_shape_factor, updates, weight_specs,
                             is_leaf=is_weight_spec)
    else:
      factors = jax.tree.map(lambda x: x.shape[-1] / x.shape[-2], updates)
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
    muon_weight_specs: A tree of `MuonWeightSpec`s, specifying how to reshape
      the parameters for orthogonalization.

  Returns:
    The corresponding `GradientTransformation`.

  References:
    Jordan, `modded-nanogpt: Speedrunning the NanoGPT baseline
    <https://github.com/KellerJordan/modded-nanogpt>`_, 2024

    Bernstein et al., `Old Optimizer, New Norm: An Anthology
    <https://arxiv.org/abs/2409.20325>`_, 2024
  """
  if muon_weight_mask is None:
    param_labels = lambda params: jax.tree.map(
        lambda x: 'muon' if x.ndim == 2 else 'adam', params
    )
  else:
    # mask comes first since it can be a prefix tree
    param_labels = lambda params: jax.tree.map(
        lambda m, x: 'muon' if m else 'adam', muon_weight_mask,
        params)
  # revision 1
  if muon_weight_specs is not None and muon_weight_mask:
    # normalize the specs for combine.partition
    # insert MaskedNode() where muon state will be masked out
    # revision
    muon_weight_specs = jax.tree.map(
        lambda m, spec: spec if m else _masking.MaskedNode(), muon_weight_mask,
        muon_weight_specs, is_leaf=lambda x: x is None or is_weight_spec(x))

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
                  weight_specs=muon_weight_specs,
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



if __name__ == "__main__":

  # python -m MaxText.snapshot5
  
  # def test(params, muon_weight_mask, muon_weight_specs):
  #   optimizer = muon(1e-5, muon_weight_mask=muon_weight_mask, muon_weight_specs=muon_weight_specs)
  #   opt_state = optimizer.init(params)

  #   def f(params):
  #     return jnp.sum(params["p1"]) + jnp.sum(params["p2"])

  #   grad = jax.grad(f)(params)
  #   updates, opt_state = optimizer.update(grad, opt_state, params)


  # # test1: all true, all spec
  # print("test 1")
  # params = {"p1": jnp.ones((1, 2, 3), dtype=jnp.bfloat16), "p2": jnp.ones((2, 3, 4), dtype=jnp.bfloat16)}
  # muon_weight_mask = {"p1": True, "p2": True}
  # muon_weight_specs = {
  #     "p1": MuonWeightSpec(reduction_axes=(1,), output_axes=(2,)),
  #     "p2": MuonWeightSpec(reduction_axes=(1,), output_axes=(2,)),
  # }
  # test(params, muon_weight_mask, muon_weight_specs)

  # # test2: false + true, no spec
  # print("test 2")
  # params = {"p1": jnp.ones((1, 2, 3), dtype=jnp.bfloat16), "p2": jnp.ones((3, 4), dtype=jnp.bfloat16)}
  # muon_weight_mask = {"p1": False, "p2": True}
  # muon_weight_specs = None
  # test(params, muon_weight_mask, muon_weight_specs)

  # # test3: false + true, partial spec
  # print("test 3")
  # params = {"p1": jnp.ones((1, 2, 3), dtype=jnp.bfloat16), "p2": jnp.ones((2, 3, 4), dtype=jnp.bfloat16)}
  # muon_weight_mask = {"p1": False, "p2": True}
  # muon_weight_specs = {"p1": optax.MaskedNode(), "p2": MuonWeightSpec(reduction_axes=(1,), output_axes=(2,))}
  # test(params, muon_weight_mask, muon_weight_specs)


  def test_update(params, muon_weight_mask, muon_weight_specs):
    optimizer = muon(1e-5, muon_weight_mask=muon_weight_mask, muon_weight_specs=muon_weight_specs)
    opt_state = optimizer.init(params)
    # def f(params):
    #   return jnp.sum(params["w"] ** 2) / 2
    # grad = jax.grad(f)(params)
    # chex.assert_trees_all_close(grad, params)
    grad = params
    updates, opt_state = optimizer.update(grad, opt_state, params)
    print(opt_state)
    return updates

  key = jax.random.PRNGKey(4)
  w = jax.random.normal(key, shape=(10, 12))
  # jnp.arange(120).reshape(10, 12)
  params = {"w": w}
  a = test_update(params, muon_weight_mask=None, muon_weight_specs=None)


  # a1 = test_update(params, muon_weight_mask={"w": False}, muon_weight_specs=None)
  
  # need revision?
  print("test 0")
  spec={"w": MuonWeightSpec(reduction_axes=(0,), output_axes=(1,))}
  mask=None
  c = test_update(params, muon_weight_mask=mask, muon_weight_specs=spec)



  print("test 1")
  spec={"w": MuonWeightSpec(reduction_axes=(0,), output_axes=(1,))}
  mask={"w": True}
  b = test_update(params, muon_weight_mask=mask, muon_weight_specs=spec)
  chex.assert_trees_all_close(a, b)



  # spec={"w": MuonWeightSpec(reduction_axes=(0,), output_axes=(1,))}
  # mask=None
  # d = test_update(params, muon_weight_mask=mask, muon_weight_specs=spec)  

  # (2, 12)
  print("test 2")
  reshape = lambda x: x.reshape(10, 3, 1, 4).transpose(3, 2, 0, 1) # (10, 12) -> (x0=4, x1=1, x2=10, x3=3)}
  params1 = {"w": reshape(params["w"])} 
  spec={"w": MuonWeightSpec(reduction_axes=(2,), output_axes=(0,3,)) }
  mask={"w": True}
  d = test_update(params1, muon_weight_mask=mask, muon_weight_specs=spec)  
  chex.assert_trees_all_close(jax.tree.map(reshape, a), d)


  print("test 3") 
  reshape = lambda x: x.reshape(2, 1, 5, 12).transpose(2, 3, 1, 0) # (10, 12) -> (x0=5, x1=12, x2=1, x3=2)}
  params1 = {"w": reshape(params["w"])} 
  spec={"w": MuonWeightSpec(reduction_axes=(0, 3), output_axes=(1,)) }
  mask={"w": True}
  d = test_update(params1, muon_weight_mask=mask, muon_weight_specs=spec)  
  chex.assert_trees_all_close(jax.tree.map(reshape, a), d, rtol=1e-8, atol=1e-8)


  
  
  # print("test 2")
  # spec={"w": MuonWeightSpec(reduction_axes=(1,), output_axes=(0,))}
  # mask={"w": True}
  # c = test_update(params, muon_weight_mask=mask, muon_weight_specs=spec)
  # chex.assert_trees_all_close(a, c)
  
            
 # Assuming your muon implementation is in a file named _muon.py
# import jax
# import jax.numpy as jnp
# import chex
# import optax
# import _muon # Import your muon optimizer file

# 1. Setup: Define target and initial 2D parameters
key = jax.random.PRNGKey(42)
target_key, init_key = jax.random.split(key)

# Our goal is to make our parameter match this target matrix
target_params = {'w': jax.random.normal(target_key, (4, 5))}
# This is our starting point
initial_params = {'w': jax.random.normal(init_key, (4, 5))}

# 2. Loss Function: Mean Squared Error
# This function measures how "close" the current params are to the target
def loss_fn(params):
  # chex.assert_trees_all_equal_shapes(params, target_params)
  return jnp.mean((params['w'] - target_params['w']) ** 2)

# 3. Optimizer: Initialize the Muon optimizer
learning_rate = 1e-3
optimizer = muon(learning_rate=learning_rate)
opt_state = optimizer.init(initial_params)
print(opt_state)

# 4. JIT-compile the update step for performance
@jax.jit
def step(params, opt_state):
  """Performs one optimization step."""
  # Calculate the gradient of the loss function
  grads = jax.grad(loss_fn)(params)
  # Get the updates from the optimizer
  updates, opt_state = optimizer.update(grads, opt_state, params)
  # Apply the updates to the parameters
  params = optax.apply_updates(params, updates)
  return params, opt_state

# --- Run the Optimization ---
print("Starting optimization...")
params = initial_params
for i in range(3000):
  params, opt_state = step(params, opt_state)
  if i % 200 == 0:
    loss = loss_fn(params)
    print(f"Step {i}, Loss: {loss:.6f}")
print(opt_state)

# 5. Verification: Check if the final parameter is close to the target
print("\nOptimization finished.")
print("Verifying results...")
chex.assert_trees_all_close(params, target_params, rtol=1e-3, atol=1e-3)
print("✅ Success! The optimized 2D parameter is very close to the target.")
print("\nFinal Parameters:\n", params['w'])
print("\nTarget Parameters:\n", target_params['w'])