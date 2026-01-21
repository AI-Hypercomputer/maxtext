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
from typing import Any, Callable

from flax import nnx
import jax
from jax.sharding import Mesh, NamedSharding

from MaxText import max_logging
from MaxText import pyconfig


def create_nnx_rngs(
    config: pyconfig.HyperParameters, is_training: bool = True, rng_key: jax.Array | None = None
) -> nnx.Rngs:
  """
  Create NNX Rngs

  Args:
    config: the configuration
    is_training: if the Rngs are for training
    rng_key: the Rng key

  Returns:
    The NNX Rngs
  """
  if rng_key is None:
    rng_key = jax.random.PRNGKey(config.init_weights_seed)

  if is_training:
    return nnx.Rngs(
        params=jax.random.fold_in(rng_key, 0), dropout=jax.random.fold_in(rng_key, 1), aqt=jax.random.fold_in(rng_key, 2)
    )
  return nnx.Rngs(params=rng_key)  # disable dropout RNG and aqt for inference


def get_named_sharding_nnx(abstract_state: Any) -> Any:
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


def get_partition_spec_nnx(named_sharding: Any) -> Any:
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


def set_named_sharding_nnx(abstract_state: Any, named_sharding: Any) -> Any:
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


def create_nnx_sharded_model(
    abstract_model: nnx.Module,
    init_fn: Callable,
    mesh: Mesh | None = None,
    named_sharding: Any | None = None,
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
