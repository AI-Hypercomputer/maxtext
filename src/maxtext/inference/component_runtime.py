# Copyright 2026 Google LLC
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

"""NNX runtime state for independently executed model components."""

from dataclasses import dataclass
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp


@dataclass
class ComponentRuntime:
  """A component's graph, parameters, cache template, and materialized auxiliary state."""

  graphdef: Any
  params: Any
  cache: Any
  rest: Any
  config: Any = None

  @classmethod
  def from_model(cls, model, params, loaded_rest):
    """Splits an abstract component, retaining loaded state and initializing only missing RNG state."""
    graphdef, _, cache, abstract_rest = nnx.split(model, nnx.Param, nnx.Cache, ...)
    loaded = dict(nnx.to_flat_state(loaded_rest))
    rest = []
    for path, variable in nnx.to_flat_state(abstract_rest):
      if path in loaded:
        value = loaded[path]
      elif isinstance(variable, nnx.RngState):
        value = jax.tree.map(
            lambda x: jax.random.key(0)
            if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key)
            else jnp.zeros(x.shape, x.dtype),
            variable,
        )
      else:
        raise ValueError(f"Missing loaded non-RNG component state at {path!r}.")
      rest.append((path, value))
    return cls(graphdef, params, cache, nnx.from_flat_state(rest), getattr(model, "config", None))

  def initial_cache(self) -> dict:
    """Returns a fresh zero-filled cache matching the component's own batch and sequence shapes."""
    return jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), nnx.to_pure_dict(self.cache))

  def merge(self, params=None, cache_dict=None):
    """Reconstitutes a component without sharing mutable cache state between calls."""
    cache = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), self.cache)
    if cache_dict is not None:
      nnx.replace_by_pure_dict(cache, cache_dict)
    return nnx.merge(self.graphdef, self.params if params is None else params, cache, self.rest, copy=True)


@dataclass
class GenerationContext:
  """Loaded resources shared with a downstream generator, without exposing engine internals."""

  mesh: Any
  logical_axis_rules: Any
  params: Any
  components: dict[str, ComponentRuntime]
