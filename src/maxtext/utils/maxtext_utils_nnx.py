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
""" Utils for MaxText NNX. """

from functools import partial
from typing import Callable

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxtext.utils import max_logging
from maxtext.configs import pyconfig
from maxtext.common.common_types import MODEL_MODE_TRAIN


def create_nnx_rngs(
    config: pyconfig.HyperParameters, model_mode: str = MODEL_MODE_TRAIN, rng_key: jax.Array | None = None
) -> nnx.Rngs:
  """
  Create NNX Rngs

  Args:
    config: the configuration
    model_mode: the model mode. See maxtext.common.common_types for valid values.
    rng_key: the Rng key

  Returns:
    The NNX Rngs
  """
  if rng_key is None:
    rng_key = jax.random.PRNGKey(config.init_weights_seed)

  if model_mode == MODEL_MODE_TRAIN:
    # Use fold_in to derive independent keys for each stream from a single seed.
    # aqt is needed for quantization-aware training.
    return nnx.Rngs(
        params=jax.random.fold_in(rng_key, 0), dropout=jax.random.fold_in(rng_key, 1), aqt=jax.random.fold_in(rng_key, 2)
    )
  return nnx.Rngs(params=rng_key)  # disable dropout RNG and aqt for inference


def get_named_sharding_nnx(abstract_state: nnx.State) -> nnx.State:
  """Get named sharding from NNX abstract state.

  Args:
    abstract_state: NNX model abstract state created from nnx.get_abstract_model.

  Returns:
    named sharding structure
  """
  # Don't use nnx.get_named_sharding() because it constructs new shardings. Instead, we
  # get the existing sharding from the abstract_state.
  # The state leaf is of type jax.ShapeDtypeStruct(shape, dtype, sharding)
  return jax.tree.map(
      lambda x: x.sharding,
      abstract_state,
      is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
  )


def get_partition_spec_nnx(named_sharding: nnx.State) -> nnx.State:
  """Get mesh partition spec from named sharding.

  Args:
    named_sharding: NNX model named sharding.

  Returns:
    mesh partition spec
  """
  # The leaf is of type NamedSharding.
  return jax.tree.map(
      lambda x: x.spec,
      named_sharding,
      is_leaf=lambda x: isinstance(x, NamedSharding),
  )


def set_named_sharding_nnx(abstract_state: nnx.State, named_sharding: nnx.State) -> nnx.State:
  """Set named sharding to NNX abstract state.

  Args:
    abstract_state: NNX model abstract state created from nnx.get_abstract_model().
    named_sharding: named sharding. It must have the same tree structure with abstract_state.

  Returns:
    updated abstract_state
  """
  return jax.tree.map(lambda x, y: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=y), abstract_state, named_sharding)


def move_memory_to_host(path: tuple[str, ...], x: NamedSharding) -> NamedSharding:
  """
  Change the memory_kind of the NamedSharding to "pinned_host". This function can be
  called by jax.tree_util.tree_map_with_path on a NNX state structure.

  Args:
    path: the tree path tuple
    x: the NamedSharding corresponding to the path

  Returns:
    the NamedSharding with memory_kind set to "pinned_host"
  """
  max_logging.log(f"max_utils.py: Moving {path} to host")
  # Create the new sharding with the target memory kind
  return x.with_memory_kind(kind="pinned_host")


def move_memory_to_device(path: tuple[str, ...], x: NamedSharding) -> NamedSharding:
  """
  Change the memory_kind of the NamedSharding to "device". This function can be
  called by jax.tree_util.tree_map_with_path on a NNX state structure.

  Args:
    path: the tree path tuple
    x: the NamedSharding corresponding to the path

  Returns:
    the NamedSharding with memory_kind set to "device"
  """
  max_logging.log(f"max_utils.py: Moving {path} to device")
  # Create the new sharding with the target memory kind
  return x.with_memory_kind(kind="device")


def create_nnx_sharded_model(
    abstract_model: nnx.Module,
    init_fn: Callable,
    mesh: Mesh | None = None,
    named_sharding: nnx.State | None = None,
) -> nnx.Module:
  """
  Create the model with the given sharding.

  Args:
    abstract_model: the abstract model
    init_fn: the model init function
    mesh: the device mesh
    named_sharding: the given sharding

  Returns:
    The initialized sharded model
  """
  graphdef, abstract_state = nnx.split(abstract_model)
  if named_sharding is None:
    # The state leaf is of type jax.ShapeDtypeStruct(shape, dtype, sharding)
    # we get the sharding directly from it.
    named_sharding = get_named_sharding_nnx(abstract_state)

  if mesh is None:
    mesh = abstract_model.mesh

  # JIT a function that creates the model state with proper sharding from the start.
  # By providing out_shardings, we instruct JAX to produce sharded output directly,
  # avoiding a large intermediate allocation on a single device.
  @partial(jax.jit, out_shardings=named_sharding)
  def create_sharded_state():
    model = init_fn()
    return jax.lax.with_sharding_constraint(nnx.state(model), named_sharding)

  # Create the model with sharded parameters.
  with jax.set_mesh(mesh):
    sharded_state = create_sharded_state()
  return nnx.merge(graphdef, sharded_state)


def nnx_ensure_scan_leading_axis(tree, length):
  """Broadcasts scalar-like variables to have a leading scan axis."""

  def _op(x):
    is_var = isinstance(x, nnx.Variable)
    val = x.get_value() if is_var else x
    if hasattr(val, "shape") and len(val.shape) == 0:
      new_val = jax.numpy.broadcast_to(val, (length,))
      return x.replace(value=new_val) if is_var else new_val
    return x

  return jax.tree.map(_op, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))


# ------------------------------------------------------------------------------
# Metadata Synchronization Helpers for NNX Variables
# ------------------------------------------------------------------------------

def nnx_update_sharding_meta(variable, transform_fn):
  """Generic helper to apply a list transformation to all sharding-related metadata."""
  if not (hasattr(variable, "get_metadata") and hasattr(variable, "replace")):
    return variable

  meta = variable.get_metadata()
  updates = {}

  for key in ["sharding", "out_sharding", "sharding_names"]:
    if (val := meta.get(key)) and isinstance(val, (P, tuple, list)):
      new_list = list(val)
      transformed = transform_fn(new_list)
      updates[key] = P(*transformed) if isinstance(val, P) else tuple(transformed)

  if updates:
    return variable.replace(**updates)
  return variable

def nnx_sync_moveaxis(tree, from_axis, to_axis):
  """Moves an axis in both values and sharding metadata of nnx.Variables."""
  if from_axis == to_axis:
    return tree

  import jax.numpy as jnp
  def _op(x):
    is_var = isinstance(x, nnx.Variable)
    val = x.get_value() if is_var else x
    if not hasattr(val, "shape"):
      return x

    new_val = jnp.moveaxis(val, from_axis, to_axis)
    if not is_var:
      return new_val

    def move_fn(l):
      if len(l) > max(from_axis, to_axis):
        l.insert(to_axis, l.pop(from_axis))
      return l

    return nnx_update_sharding_meta(x.replace(value=new_val), move_fn)

  return jax.tree.map(_op, tree, is_leaf=lambda x: isinstance(x, nnx.Variable) or hasattr(x, "shape"))

def nnx_remove_scan_axis(tree, name="layers"):
  """Removes the given scan axis from the PartitionSpec."""

  def _op(x):
    if not isinstance(x, nnx.Variable):
      return x

    def remove_fn(l):
      if name in l:
        l.remove(name)
      while len(l) > x.get_value().ndim:
        l.pop(0)
      return l

    return nnx_update_sharding_meta(x, remove_fn)

  return jax.tree.map(_op, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))

def nnx_add_scan_axis(tree, name="layers", pos=0):
  """Adds the given scan axis to the PartitionSpec at the specified position."""

  def _op(x):
    if not isinstance(x, nnx.Variable):
      return x

    def add_fn(l):
      if name not in l:
        l.insert(pos, name)
      while len(l) < x.get_value().ndim:
        l.insert(pos, None)
      return l

    return nnx_update_sharding_meta(x, add_fn)

  return jax.tree.map(_op, tree, is_leaf=lambda x: isinstance(x, nnx.Variable))
