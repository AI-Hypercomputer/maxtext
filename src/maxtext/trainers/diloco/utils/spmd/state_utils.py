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

"""Core DiLoCo state initialization, sharding, and synchronization utilities for SPMD."""

from collections.abc import Sequence
from typing import Any

import drjax
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from maxtext.common.train_state_nnx import TrainStateNNX
from maxtext.trainers.diloco import diloco
import optax


def add_diloco_to_sharding(pytree: PyTree) -> PyTree:
  """Recursively traverses a PyTree and prepends 'diloco' to the PartitionSpec
  of any NamedSharding object that contains 'diloco' in its mesh axis names.
  """

  def map_fn(leaf):
    if isinstance(leaf, jax.sharding.NamedSharding):
      if "diloco" not in leaf.mesh.axis_names:
        return leaf
      new_spec = jax.sharding.PartitionSpec("diloco", *leaf.spec)
      return jax.sharding.NamedSharding(mesh=leaf.mesh, spec=new_spec)
    return leaf

  return jax.tree_util.tree_map(map_fn, pytree)


def reshape_first_axis_with_diloco(num_diloco_replicas: int, pytree: PyTree) -> PyTree:
  """Reshapes the first dimension of each array in the PyTree to include a DiLoCo axis."""

  def extend_pspec(
      pspec: jax.sharding.PartitionSpec | Sequence[str | Sequence[str]] = (),
  ) -> jax.sharding.PartitionSpec:
    if pspec and isinstance(pspec[0], (tuple, list)) and len(pspec[0]) > 0 and pspec[0][0] == "diloco":
      remaining = tuple(pspec[0][1:])
      if len(remaining) == 1:
        return jax.sharding.PartitionSpec("diloco", remaining[0], *pspec[1:])
      elif len(remaining) > 1:
        return jax.sharding.PartitionSpec("diloco", remaining, *pspec[1:])
      else:
        return jax.sharding.PartitionSpec("diloco", *pspec[1:])
    return jax.sharding.PartitionSpec("diloco", *pspec)

  def reshape_for_diloco(arr):
    if not hasattr(arr, "shape"):
      return arr
    if (
        hasattr(arr, "ndim")
        and arr.ndim >= 3
        and arr.shape[0] == num_diloco_replicas
        and hasattr(arr, "sharding")
        and isinstance(arr.sharding, jax.sharding.NamedSharding)
        and arr.sharding.spec
        and arr.sharding.spec[0] == "diloco"
        and isinstance(arr.sharding.spec[0], str)
    ):
      return arr
    batch_dim, *example_shape = arr.shape
    if batch_dim % num_diloco_replicas != 0:
      raise ValueError(f"Batch dimension {batch_dim} is not divisible by num_diloco_replicas {num_diloco_replicas}.")
    diloco_shape = (num_diloco_replicas, batch_dim // num_diloco_replicas, *example_shape)
    if hasattr(arr, "sharding") and arr.sharding is not None:
      s = arr.sharding
      s = jax.sharding.NamedSharding(mesh=s.mesh, spec=extend_pspec(s.spec))
      return jax.lax.with_sharding_constraint(jnp.reshape(arr, shape=diloco_shape), s)
    return jnp.reshape(arr, shape=diloco_shape)

  return jax.tree.map(reshape_for_diloco, pytree)


def extract_replica_0(metrics: PyTree) -> PyTree:
  """Extracts metrics from replica 0 across DiLoCo islands."""

  def select_first_replica(x):
    if not hasattr(x, "shape") or len(x.shape) == 0:
      return x
    r = x.shape[0]
    mask = (jnp.arange(r) == 0).reshape((r,) + (1,) * (x.ndim - 1))
    return drjax.reduce_sum(x * mask)

  return jax.tree.map(select_first_replica, metrics)


def extract_per_island_metrics(metrics: PyTree, num_diloco_replicas: int) -> PyTree:
  """Extracts replica 0 metrics and appends per-island loss metrics."""
  default_metrics = extract_replica_0(metrics)
  if isinstance(metrics, dict) and "scalar" in metrics and "learning/loss" in metrics["scalar"]:
    for i in range(num_diloco_replicas):
      mask_i = jnp.arange(num_diloco_replicas) == i
      loss_i = drjax.reduce_sum(metrics["scalar"]["learning/loss"] * mask_i)
      default_metrics["scalar"][f"learning/loss_island_{i}"] = loss_i
  return default_metrics


def replace_nnx_model_params(s, new_params):
  """Replaces model parameters in an NNX TrainState or dictionary structure."""
  s_model = s["model"] if hasattr(s, "keys") else s.model
  s_opt = s["optimizer"] if hasattr(s, "keys") else s.optimizer

  graphdef, _, non_param_state = nnx.split(s_model, nnx.Param, ...)
  new_model = nnx.merge(graphdef, new_params, non_param_state)

  if isinstance(s_model, nnx.State):
    new_model = nnx.state(new_model)
  elif isinstance(s_model, dict):
    new_model = nnx.to_pure_dict(new_model)

  if hasattr(s, "keys"):
    leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(s)
    new_model_iter = iter(jax.tree_util.tree_leaves(new_model))

    def _is_model_leaf(path):
      if not path:
        return False
      k = path[0]
      return getattr(k, "key", None) == "model" or getattr(k, "name", None) == "model"

    new_leaves = [next(new_model_iter) if _is_model_leaf(p) else leaf for p, leaf in leaves_with_paths]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)
  else:
    return TrainStateNNX(new_model, s_opt)


def synchronize_full_state(
    state,
    outer_optimizer: optax.GradientTransformation,
    mesh: jax.sharding.Mesh | None = None,
):
  """Synchronizes all parameters across DiLoCo replicas for vanilla DiLoCo."""
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


def setup_diloco_initial_state(
    state: Any,
    config: Any,
    mesh: jax.sharding.Mesh,
    state_mesh_shardings: PyTree,
    restored: Any = None,
) -> Any:
  """Builds the full DiLoCoTrainState from the restored or freshly initialized single-replica state."""
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
