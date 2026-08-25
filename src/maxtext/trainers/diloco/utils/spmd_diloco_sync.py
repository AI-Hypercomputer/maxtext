# Copyright 2023-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collective state synchronization and initialization routines for SPMD DiLoCo."""

from typing import Any

try:
  import drjax
except ImportError:
  drjax = None

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from maxtext.trainers.diloco import diloco
from maxtext.trainers.diloco.utils.fragmenter import FragmentedTreeManipulator
from maxtext.trainers.diloco.utils.nnx_state_utils import replace_nnx_model_params, replace_nnx_model_params_frag
from maxtext.utils.diloco_sharding import add_diloco_to_sharding
import optax


def synchronize_full_state(
    state: Any,
    outer_optimizer: optax.GradientTransformation,
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
  """Synchronizes all parameters across DiLoCo replicas for vanilla SPMD DiLoCo."""
  broadcast_outer_params = drjax.broadcast(state.params, mesh=mesh)
  _, inner_model_params, _ = nnx.split(state.inner_state.model, nnx.Param, ...)
  inner_model_params = inner_model_params.to_pure_dict()

  model_delta = jax.tree.map(lambda x, y: y - x, inner_model_params, broadcast_outer_params)
  averaged_pseudo_grad = drjax.reduce_mean(model_delta)
  updates, new_opt_state = outer_optimizer.update(averaged_pseudo_grad, state.outer_opt_state, state.params)
  new_outer_params = optax.apply_updates(state.params, updates)

  new_inner_state = drjax.map_fn(
      lambda s: replace_nnx_model_params(s, new_outer_params),
      state.inner_state,
      mesh=mesh,
  )
  return state.replace(
      params=new_outer_params,
      outer_opt_state=new_opt_state,
      inner_state=new_inner_state,
  )


def synchronize_fragment_state(
    state: Any,
    manipulator: FragmentedTreeManipulator,
    frag_idx: int,
    outer_optimizer: optax.GradientTransformation,
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
  """Synchronizes a single parameter fragment across DiLoCo replicas in streaming SPMD DiLoCo."""
  # 1. Extract global and local parameters for the fragment
  outer_params_frag = manipulator.get_flat_fragment(state.params, frag_idx, has_replica_dim=False)
  inner_model_params = nnx.filter_state(state.inner_state.model, nnx.Param).to_pure_dict()
  inner_params_frag = manipulator.get_flat_fragment(inner_model_params, frag_idx, has_replica_dim=True)

  # 2. Compute the pseudo-gradient: outer - inner
  broadcast_outer_frag = drjax.broadcast(outer_params_frag, mesh=mesh)
  unreduced_grads = jax.tree.map(lambda x, y: x - y, broadcast_outer_frag, inner_params_frag)

  # 3. Average gradients across replicas
  averaged_pseudo_grad = drjax.reduce_mean(unreduced_grads)

  # 4. Extract outer optimizer state for this fragment (TraceState is (trace, EmptyState))
  trace_frag = manipulator.get_flat_fragment(state.outer_opt_state[0].trace, frag_idx, has_replica_dim=False)
  opt_state_frag = (optax.TraceState(trace=trace_frag), optax.EmptyState())

  # 5. Run outer optimizer on the fragment
  updates_frag, new_opt_state_frag = outer_optimizer.update(
      averaged_pseudo_grad, opt_state_frag, params=outer_params_frag
  )
  new_outer_params_frag = optax.apply_updates(outer_params_frag, updates_frag)

  # 6. Re-merge updated params and optimizer states back to full PyTree
  new_params = manipulator.apply_flat_fragment(state.params, frag_idx, new_outer_params_frag, has_replica_dim=False)
  new_trace = manipulator.apply_flat_fragment(
      state.outer_opt_state[0].trace, frag_idx, new_opt_state_frag[0].trace, has_replica_dim=False
  )
  new_outer_opt_state = (optax.TraceState(trace=new_trace), state.outer_opt_state[1])

  return state.replace(
      params=new_params,
      outer_opt_state=new_outer_opt_state,
  )


def apply_fragment_to_inner_state(
    state: Any,
    manipulator: FragmentedTreeManipulator,
    frag_idx: int,
    alpha: float = 0.0,
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
  """Broadcasts synced outer parameter fragment and updates inner state across replicas."""
  outer_params_frag = manipulator.get_flat_fragment(state.params, frag_idx, has_replica_dim=False)
  broadcast_outer_frag = drjax.broadcast(outer_params_frag, mesh=mesh)

  new_inner_state = drjax.map_fn(
      lambda s, frag: replace_nnx_model_params_frag(s, manipulator, frag_idx, frag, alpha=alpha),
      (state.inner_state, broadcast_outer_frag),
      mesh=mesh,
  )
  return state.replace(inner_state=new_inner_state)


def setup_diloco_initial_state(
    state: Any,
    config: Any,
    mesh: jax.sharding.Mesh,
    state_mesh_shardings: PyTree,
    restored: Any = None,
) -> Any:
  """Builds the full DiLoCoTrainState from the restored or freshly initialized single-replica state."""
  del restored  # Unused, kept for signature compatibility
  if isinstance(state, diloco.DiLoCoTrainState):
    return state

  # 1. Compute per-replica shardings with 'diloco' axis prepended
  inner_state_shardings = add_diloco_to_sharding(state_mesh_shardings)

  # 2. Extract concrete outer model parameters from state.model (or state.params)
  if hasattr(state, "model"):
    _, outer_params, _ = nnx.split(state.model, nnx.Param, ...)
    outer_params = outer_params.to_pure_dict() if hasattr(outer_params, "to_pure_dict") else outer_params
  else:
    outer_params = getattr(state, "params", state)

  # 3. Broadcast single-replica state to multi-replica inner_state across the diloco axis
  def _broadcast_to_replicas(leaf, sharding):
    if hasattr(leaf, "shape"):
      target_shape = (config.num_diloco_replicas, *leaf.shape)
      if isinstance(leaf, jax.ShapeDtypeStruct):
        sharding_arg = sharding if isinstance(sharding, jax.sharding.NamedSharding) else None
        return jax.ShapeDtypeStruct(target_shape, leaf.dtype, sharding=sharding_arg)
      if isinstance(sharding, jax.sharding.NamedSharding):
        return jax.jit(
            lambda x: jnp.broadcast_to(x, target_shape),
            out_shardings=sharding,
        )(leaf)
      return jnp.broadcast_to(leaf, target_shape)
    return leaf

  inner_state = jax.tree_util.tree_map(
      _broadcast_to_replicas,
      state,
      inner_state_shardings,
  )

  # 4. Initialize outer optimizer state with outer SGD momentum
  outer_optimizer = optax.sgd(
      config.diloco_outer_lr,
      momentum=config.diloco_outer_momentum,
      nesterov=True,
  )
  outer_opt_state = outer_optimizer.init(outer_params)

  # 5. Extract global step
  step = getattr(getattr(state, "optimizer", None), "step", jnp.array(0, dtype=jnp.int32))

  return diloco.DiLoCoTrainState(
      inner_state=inner_state,
      params=outer_params,
      outer_opt_state=outer_opt_state,
      step=step,
  )
