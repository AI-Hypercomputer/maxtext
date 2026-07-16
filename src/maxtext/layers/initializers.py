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

"""Initializers."""

from typing import Callable

import jax

from flax import linen as nn
from flax import nnx
from aqt.jax.v2 import aqt_tensor

from maxtext.common.common_types import Array, DType, Shape, PRNGKey

Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = int | tuple[int, ...]
NdInitializer = Callable[[PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]

import contextlib

_skip_random_init = False


@contextlib.contextmanager
def skip_random_init():
  """Context manager to temporarily bypass random parameter initialization.

  This is a memory and compile-time optimization. When loading parameters from
  a checkpoint, the initial random weight values are immediately discarded and
  overwritten. Bypassing random number generation avoids allocating massive
  temporary RNG state arrays (e.g., u32 bit generator inputs) on the devices
  and eliminates substantial compiler overhead, preventing CompileTimeHbmOom.
  """
  global _skip_random_init
  orig_skip = _skip_random_init
  _skip_random_init = True

  import jax.nn.initializers as jax_inits
  import flax.linen.initializers as flax_inits

  # Save original initializers
  orig_jax_inits = {}
  orig_flax_inits = {}

  # A zero initializer factory
  def zero_init_factory(*args, **kwargs):
    def init_fn(key, shape, dtype=jax.numpy.float32, *args_inner, **kwargs_inner):
      return jax.nn.initializers.zeros(key, shape, dtype)
    return init_fn

  # List of random initializers to patch
  inits_to_patch = [
      "normal", "uniform", "variance_scaling",
      "glorot_normal", "glorot_uniform",
      "he_normal", "he_uniform",
      "lecun_normal", "lecun_uniform",
      "orthogonal"
  ]

  for name in inits_to_patch:
    if hasattr(jax_inits, name):
      orig_jax_inits[name] = getattr(jax_inits, name)
      setattr(jax_inits, name, zero_init_factory)
    if hasattr(flax_inits, name):
      orig_flax_inits[name] = getattr(flax_inits, name)
      setattr(flax_inits, name, zero_init_factory)

  try:
    yield
  finally:
    _skip_random_init = orig_skip
    for name, orig_fn in orig_jax_inits.items():
      setattr(jax_inits, name, orig_fn)
    for name, orig_fn in orig_flax_inits.items():
      setattr(flax_inits, name, orig_fn)


def _wrapped_embed_init(key, shape, dtype):
  if _skip_random_init:
    return jax.nn.initializers.zeros(key, shape, dtype)
  return nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0)(key, shape, dtype)


default_embed_init = _wrapped_embed_init

default_bias_init = jax.nn.initializers.constant(0.0)
default_scalar_init = jax.nn.initializers.constant(0.01)


def nd_dense_init(scale, mode, distribution):
  """Creates a variance-scaling initializer with dynamic in/out axes.

  This function is a factory that returns an initializer function. The returned
  function is a wrapper around `jax.nn.initializers.variance_scaling` that
  allows the `in_axis` and `out_axis` to be specified at call time, rather
  than at creation time.

  Args:
    scale: The scaling factor for the variance.
    mode: The mode for variance scaling ('fan_in', 'fan_out', 'fan_avg').
    distribution: The distribution to sample from ('normal', 'uniform', etc.).

  Returns:
    A function that takes a PRNG key, shape, dtype, in_axis, and out_axis,
    and returns an initialized array.
  """

  def init_fn(key, shape, dtype, in_axis, out_axis):
    """Initializes an array using variance scaling with specified axes."""
    if _skip_random_init:
      return jax.nn.initializers.zeros(key, shape, dtype)
    fn = jax.nn.initializers.variance_scaling(scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)

  return init_fn


def variable_to_logically_partitioned(variable: nnx.Variable):
  """Wraps an NNX variable's value in `nn.LogicallyPartitioned`.

  This function inspects the metadata of an `nnx.Variable` object. If
  sharding information ('out_sharding', 'sharding' or 'sharding_names') is
  present, it wraps the variable's value in `nn.LogicallyPartitioned` to apply
  the specified sharding constraints.

  It handles special cases for `aqt_tensor.QTensor` and variables of type
  `_overwrite_with_gradient` by returning their values directly without
  wrapping.

  Args:
    variable: The `nnx.Variable` object to process.

  Returns:
    The variable's value, potentially wrapped in `nn.LogicallyPartitioned`.
  """
  val = variable.get_value()
  if isinstance(val, aqt_tensor.QTensor):
    return val

  if variable.type.__name__ == "_overwrite_with_gradient":
    return val

  metadata = variable.get_metadata()
  out_sharding = None
  if "out_sharding" in metadata:
    out_sharding = metadata["out_sharding"]
  elif "sharding_names" in metadata:
    out_sharding = metadata["sharding_names"]
  elif "sharding" in metadata:
    out_sharding = metadata["sharding"]

  if out_sharding is not None:
    if nnx.PARTITION_NAME in metadata:
      partition_name = metadata[nnx.PARTITION_NAME]
      scan_axis = metadata.get("param_scan_axis", 0) if variable.type == nnx.Param else 0

      sharding_list = [out_sharding] if isinstance(out_sharding, str) else list(out_sharding)
      if partition_name not in sharding_list:
        sharding_list.insert(scan_axis, partition_name)

      out_sharding = tuple(sharding_list)

    return nn.LogicallyPartitioned(  # type: ignore[wrong-keyword-args]
        val,
        out_sharding,  # type: ignore[arg-type]
        mesh=metadata.get("mesh"),
        rules=metadata.get("rules"),
    )
  else:
    return val
