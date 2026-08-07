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

"""Checkpointing and restoration utilities for SPMD DiLoCo in MaxText."""

from typing import Any
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.common import train_state_nnx
from maxtext.trainers.diloco import diloco
from maxtext.trainers.diloco.utils.spmd import state_utils
import optax


def is_diloco_checkpoint(restored_dict: Any) -> bool:
  """Checks if a restored checkpoint dictionary contains a full multi-replica DiLoCo state."""
  if not isinstance(restored_dict, dict):
    return False
  # A full DiLoCo checkpoint contains 'inner_state' and 'outer_opt_state'
  return "inner_state" in restored_dict and "outer_opt_state" in restored_dict


def to_diloco_checkpoint_dict(state: Any, config: Any = None) -> dict[str, Any]:
  """Packages a DiLoCoTrainState or abstract state into an Orbax-serializable checkpoint dictionary.

  Saves / Serializes:
    1. inner_state: per-replica model weights + Adam first/second moment buffers (m, v) + step
    2. params: outer global model weights
    3. outer_opt_state: outer SGD Nesterov momentum state
    4. step: global step counter

  Args:
    state: The DiLoCoTrainState instance or single-replica state (for abstract restore targets).
    config: MaxText configuration object.

  Returns:
    A dictionary formatted for Orbax saving / restoring.
  """
  num_replicas = getattr(config, "dcn_diloco_parallelism", getattr(config, "num_diloco_replicas", 2))

  if isinstance(state, diloco.DiLoCoTrainState):
    inner_state = state.inner_state
    params = state.params
    outer_opt_state = state.outer_opt_state
    step = state.step
  elif hasattr(state, "inner_state"):
    inner_state = state.inner_state
    params = getattr(state, "params", None)
    outer_opt_state = getattr(state, "outer_opt_state", None)
    step = getattr(state, "step", jnp.array(0, dtype=jnp.int32))
  else:
    # Single-replica state (e.g. abstract TrainStateNNX passed during checkpoint restoration)
    def _add_diloco_dim(leaf):
      if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        new_shape = (num_replicas, *leaf.shape)
        sharding = getattr(leaf, "sharding", None)
        if isinstance(sharding, jax.sharding.NamedSharding) and "diloco" in sharding.mesh.axis_names:
          new_spec = jax.sharding.PartitionSpec("diloco", *sharding.spec)
          sharding = jax.sharding.NamedSharding(mesh=sharding.mesh, spec=new_spec)
        if isinstance(leaf, jax.ShapeDtypeStruct):
          return jax.ShapeDtypeStruct(new_shape, leaf.dtype, sharding=sharding)
        return jnp.broadcast_to(leaf, new_shape)
      return leaf

    inner_state = jax.tree_util.tree_map(_add_diloco_dim, state)

    if hasattr(state, "model"):
      _, params, _ = nnx.split(state.model, nnx.Param, ...)
      params = params.to_pure_dict() if hasattr(params, "to_pure_dict") else params
    elif hasattr(state, "params"):
      params = state.params
    else:
      params = state

    outer_optimizer = optax.sgd(
        getattr(config, "diloco_outer_lr", 0.1),
        momentum=getattr(config, "diloco_outer_momentum", 0.9),
        nesterov=True,
    )
    outer_opt_state = outer_optimizer.init(params)
    step = getattr(getattr(state, "optimizer", None), "step", jnp.array(0, dtype=jnp.int32))

  # 1. Inner state: convert per-replica NNX state to Linen checkpoint layout
  if isinstance(inner_state, (nnx.State, train_state_nnx.TrainStateNNX)):
    inner_state_dict = train_state_nnx.to_checkpoint_dict(inner_state)
  elif hasattr(inner_state, "to_pure_dict"):
    inner_state_dict = inner_state.to_pure_dict()
  elif isinstance(inner_state, dict):
    inner_state_dict = inner_state
  else:
    inner_state_dict = inner_state

  # 2. Outer params: outer global model parameters
  if hasattr(params, "to_pure_dict"):
    params_dict = params.to_pure_dict()
  elif isinstance(params, (nnx.State, nnx.Module)):
    params_dict = nnx.state(params).to_pure_dict()
  elif isinstance(params, dict):
    params_dict = params
  else:
    params_dict = params

  # 3. Outer optimizer state: outer SGD momentum trace
  # 4. Global step
  step_val = step.get_value() if hasattr(step, "get_value") else step

  return {
      "inner_state": inner_state_dict,
      "params": params_dict,
      "outer_opt_state": outer_opt_state,
      "step": step_val,
  }


def from_diloco_checkpoint_dict(
    restored_dict: dict[str, Any],
    abstract_diloco_state: Any,
    config: Any = None,
    mesh: jax.sharding.Mesh | None = None,
) -> Any:
  """Restores a DiLoCoTrainState from an Orbax checkpoint dictionary.

  Supports:
    - Full DiLoCo checkpoints (restores inner_state, outer_opt_state, params, step).
    - Legacy / params-only checkpoints (restores params & step, broadcasts to inner_state,
      and initializes fresh optimizer states).

  Args:
    restored_dict: The dictionary loaded by Orbax.
    abstract_diloco_state: Abstract DiLoCoTrainState or TrainStateNNX with expected shapes.
    config: MaxText configuration object.
    mesh: Device mesh.

  Returns:
    A concrete DiLoCoTrainState instance.
  """
  num_replicas = getattr(config, "dcn_diloco_parallelism", getattr(config, "num_diloco_replicas", 2))

  def _add_diloco_dim(leaf):
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
      new_shape = (num_replicas, *leaf.shape)
      sharding = getattr(leaf, "sharding", None)
      if isinstance(sharding, jax.sharding.NamedSharding) and "diloco" in sharding.mesh.axis_names:
        new_spec = jax.sharding.PartitionSpec("diloco", *sharding.spec)
        sharding = jax.sharding.NamedSharding(mesh=sharding.mesh, spec=new_spec)
      if isinstance(leaf, jax.ShapeDtypeStruct):
        return jax.ShapeDtypeStruct(new_shape, leaf.dtype, sharding=sharding)
      return jnp.broadcast_to(leaf, new_shape)
    return leaf

  if is_diloco_checkpoint(restored_dict):
    # Full multi-replica DiLoCo checkpoint restoration
    inner_dict = restored_dict["inner_state"]
    if isinstance(abstract_diloco_state, diloco.DiLoCoTrainState):
      abstract_inner = abstract_diloco_state.inner_state
    else:
      abstract_inner = jax.tree_util.tree_map(_add_diloco_dim, abstract_diloco_state)

    if isinstance(abstract_inner, nnx.Module):
      abstract_inner = nnx.state(abstract_inner)

    if isinstance(abstract_inner, (nnx.State, train_state_nnx.TrainStateNNX)):
      inner_state = train_state_nnx.linen_items_to_nnx(inner_dict, abstract_inner)
    else:
      inner_state = inner_dict

    params = restored_dict["params"]
    outer_opt_state = restored_dict["outer_opt_state"]
    step = restored_dict["step"]

    return diloco.DiLoCoTrainState(
        inner_state=inner_state,
        params=params,
        outer_opt_state=outer_opt_state,
        step=step,
    )

  # Legacy checkpoint fallback (only params / model and step present)
  raw_params = None
  if "params" in restored_dict:
    raw_params = restored_dict["params"]
    if isinstance(raw_params, dict) and "params" in raw_params:
      raw_params = raw_params["params"]
  elif "model" in restored_dict:
    raw_params = restored_dict["model"]
  else:
    raw_params = restored_dict

  step = restored_dict.get("step", jnp.array(0, dtype=jnp.int32))

  broadcasted_model_params = jax.tree_util.tree_map(_add_diloco_dim, raw_params)

  if isinstance(abstract_diloco_state, diloco.DiLoCoTrainState):
    abstract_inner = abstract_diloco_state.inner_state
  else:
    abstract_inner = jax.tree_util.tree_map(_add_diloco_dim, abstract_diloco_state)

  inner_state = state_utils.replace_nnx_model_params(abstract_inner, broadcasted_model_params)

  outer_optimizer = optax.sgd(
      getattr(config, "diloco_outer_lr", 0.1),
      momentum=getattr(config, "diloco_outer_momentum", 0.9),
      nesterov=True,
  )
  outer_opt_state = outer_optimizer.init(raw_params)

  return diloco.DiLoCoTrainState(
      inner_state=inner_state,
      params=raw_params,
      outer_opt_state=outer_opt_state,
      step=step,
  )
