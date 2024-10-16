# Copyright 2024 Google LLC
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

"""Implementation of Engine API for MaxText"""
import copy as cp
import dataclasses
import functools
from typing import Any, Optional, Tuple, Callable, Sequence, Mapping

import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax import struct

from layers import models, quantizations

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.topologies import get_topology_desc

import common_types
from jetstream.engine import engine_api
from jetstream.engine import tokenizer_pb2
from jetstream.engine import tokenizer_api
from jetstream.engine import token_utils
import accelerator_to_spec_map

import numpy as np
import max_utils
import inference_utils
import pyconfig
from jax.sharding import Mesh

import warnings

warnings.simplefilter("ignore", category=FutureWarning)

Prefix = Any
Params = Any

@dataclasses.dataclass(slots=True, kw_only=True)
class GridOfRingsPartitionConfig():
  """Creates a target_mesh_shape mesh from a 2D device_mesh input.

    Only two dimensions in target_mesh_shape can be larger than 1, and their
    size must be divisble by two.

    The result is a "grid of rings" in which the inner rings are highly
    efficient for model parallelism (allowing us to use 4 or 8 way model
    parallelism while having wrap arounds), while the outer grid is laid out
    s.t. it is easy to work out how to all-to-all for expert parallelism. This
    will _not_ be the optimal layout for pipelining, which would instead be a
    ring of rings.
    TODO(sholto): Extend this to 2D model axis (e.g. model_q/model_kv edges of
    the inner rings).

    v < < < v < < <
    > > > ^ > > > ^
    v < < < v < < <
    > > > ^ > > > ^

  """
  mesh_axis_names: tuple[str, ...] = ('sequence', 'tensor')
  ici_mesh: Mapping[str, int] = dataclasses.field(default_factory=dict)
  outer_axis_name: str
  inner_axis_name: str

  def _get_snake_horizontal(self,
    x_size: int, y_size: int
  ) -> list[tuple[int, int]] | None:
    """Draws a horizontal snake rectangle-filling loop.

    A snake is a path on the grid that with no self-intersection such that each
    cell is a neighbor of the next cell.

    For example, if `x_size == 4` and `y_size == 3`, we have the following loop:
      a l k
      b i j
      c h g
      d e f
    So the result is:
      [(0, 0), (1, 0), (2, 0), ..., (1, 2), (0, 2), (0, 1)]

    Args:
      x_size: Number of rows, has to be even.
      y_size: Number of columns, has to be at least 2, unless `x_size == 2`.

    Returns:
      A list of tuples (x, y) drawing a snake loop.
      If drawing a horizontal snake is not possible, returns None.
    """
    if x_size % 2 != 0:
      return
    if x_size > 2 and y_size < 2:
      # Note that y_size == 1, x_size == 2 is a legal snake.
      # If x_size > 2 we require at least two columns so we can close the loop.
      return

    snake_indices = []

    # Start with the vetical part of the snake:
    # V * * * * *
    # V * * * * *
    # V * * * * *
    # V * * * * *
    # V * * * * *
    # V * * * * *
    # Where `*` denotes cells left empty for now.
    for row in range(x_size):
      snake_indices.append((row, 0))

    # Add the horizontal part of the snake, from bottom to top:
    # V < < < < <
    # V > > > > ^
    # V ^ < < < <
    # V > > > > ^
    # V ^ < < < <
    # > > > > > ^
    # Note that if `y_size == 1`, this does nothing.
    for row in reversed(range(x_size)):
      columns = range(1, y_size)
      if row % 2 == 0:
        # Even rows return right to left.
        columns = reversed(columns)
      for column in columns:
        snake_indices.append((row, column))

    assert len(snake_indices) == x_size * y_size

    return snake_indices


  def _get_snake(self, x_size: int, y_size: int) -> list[tuple[int, int]] | None:
    """Draws a snake rectangle-filling loop, return None if not possible."""
    if x_size % 2 == 0:
      return self._get_snake_horizontal(x_size, y_size)

    # Construct a snake for the transposed rectangle.
    snake_indices = self._get_snake_horizontal(y_size, x_size)

    if snake_indices is None:
      return None

    # Transpose the result back to the original coordinates.
    return [(x, y) for y, x in snake_indices]

  def _get_physical_tpu_mesh(self, devices: Sequence[Any]) -> np.ndarray:
    """Reshapes JAX devices into their physical topology shape.

    Includes cores dimension.

    Args:
      devices: JAX devices.

    Returns:
      Reshaped JAX devices.
    """
    device_coords = [d.coords for d in devices]
    cores_per_chip = max(d.core_on_chip for d in devices) + 1

    mesh_shape = tuple(d + 1 for d in max(device_coords)) + (cores_per_chip,)
    masked_mesh_shape = tuple(map(min, zip(*device_coords))) + (0,)
    offset_x, offset_y, offset_z, _ = masked_mesh_shape
    actual_mesh_shape = tuple(
        d - e for d, e in zip(mesh_shape, masked_mesh_shape)
    )
    physical_mesh = np.empty(actual_mesh_shape, dtype=object)
    for (x, y, z), d in zip(device_coords, devices):
      physical_mesh[x - offset_x, y - offset_y, z - offset_z, d.core_on_chip] = d

    return physical_mesh

  def _get_ring_dimensions(
      self, axis_length: int, transposed: bool = False
  ) -> tuple[int, int]:
    length_to_ring_dimensions = {
        1: (1, 1),
        2: (2, 1),
        4: (2, 2),
        8: (4, 2),
        16: (4, 4),
        32: (8, 4),
        64: (8, 8),
        128: (16, 8),
        256: (16, 16),
    }
    if axis_length not in length_to_ring_dimensions:
      raise ValueError(
          f'Unsupported axis length {axis_length} for'
          ' GridOfRingsPartitionConfig'
      )
    axes = length_to_ring_dimensions[axis_length]
    return (axes[1], axes[0]) if transposed else axes

  def make_mesh(
      self, devices: Sequence[jax.Device] | None = None
  ) -> jax.sharding.Mesh:
    """Creates a ring-of-rings mesh from a 2D device_mesh input."""
    if devices is None:
      devices = jax.devices()
    device_mesh = self._get_physical_tpu_mesh(devices)
    if len(device_mesh.shape) != 4 or device_mesh.shape[2:] != (1, 1):
      raise ValueError(
          f'Grid-of-rings only works on 2D slices, found {device_mesh.shape}. '
          'Expected shape is (A, B, 1, 1).'
      )

    # We assume the first sharded dimension will be the outer ring.
    outer_size = self.ici_mesh[self.outer_axis_name]
    inner_size = self.ici_mesh[self.inner_axis_name]
    assert outer_size * inner_size == len(devices), (
        f'Outer size: {outer_size}, inner size: {inner_size}, num devices:'
        f' {len(devices)}'
    )
    if outer_size % 2 != 0 or inner_size % 2 != 0:
      raise ValueError(
          'Grid-of-rings logical dimensions must be divisible by two.'
      )

    inner_ring_axes = self._get_ring_dimensions(inner_size)
    outer_ring_axes = (device_mesh.shape[0] // inner_ring_axes[0],
                       device_mesh.shape[1] // inner_ring_axes[1])
    inner_ring_coords = self._get_snake(inner_ring_axes[0], inner_ring_axes[1])

    # Remove the dummy dimensions, we know we are 2D.
    device_mesh = device_mesh.squeeze((2, 3))

    inner_rings = []
    for outer_i in range(outer_ring_axes[0]):
      for outer_j in range(outer_ring_axes[1]):
        inner_ring = []
        for c in inner_ring_coords:
          # Add the inner ring coordinate by coordinate.
          inner_ring.append(
              device_mesh[
                  outer_i * inner_ring_axes[0] + c[0],
                  outer_j * inner_ring_axes[1] + c[1],
              ]
          )
        inner_rings.append(inner_ring)

    final_devices = np.array(inner_rings)
    other_dims = [
        d
        for d in self.mesh_axis_names
        if d not in [self.outer_axis_name, self.inner_axis_name]
    ]
    mesh_axis_names = (
        self.outer_axis_name,
        self.inner_axis_name,
        *other_dims,
    )
    final_devices = final_devices[(...,) + (np.newaxis,) * len(other_dims)]
    return jax.sharding.Mesh(final_devices, mesh_axis_names)

def make_nested_balanced_2d_devices(devices: Sequence[jax.Device], ici_mesh_shape: Sequence[int]) -> Sequence[jax.Device]:
    # Generate a reversed, interleaved sequence of axis indices.
    print(f"\nmake_nested_balanced_2d_devices: {devices=}")
    print(f"\nmake_nested_balanced_2d_devices: {ici_mesh_shape=}")
    log_len = np.array(devices).size.bit_length() - 1
    arr = np.arange(log_len)[::-1]
    midpoint = len(arr) // 2
    first_half = arr[:midpoint]
    second_half = arr[midpoint:]
    print(f"make_nested_balanced_2d_devices: {first_half=}")
    print(f"make_nested_balanced_2d_devices: {second_half=}")

    new_axis_order = []
    for pair in zip(second_half, first_half):
      new_axis_order.extend(pair)
    # Handle odd log_length case: append leftover element if it exists
    if len(arr) % 2 == 1:
      new_axis_order.append(second_half[-1])
    print(f"make_nested_balanced_2d_devices: {new_axis_order=}")

    ordered_flat_devices = sorted(
        np.array(devices).flatten(), key=lambda x: x.id
    )
    # Form a nested, balanced 2D partition with the priority order.
    result = np.reshape(ordered_flat_devices, (2,) * log_len).transpose(new_axis_order[::-1]).reshape(ici_mesh_shape)
    return result

def get_topology_mesh(config):
  """Get the target hardware devices, and create configured mesh with them"""
  target_hardware = accelerator_to_spec_map.get_system_characteristics(config.compile_topology)
  print(f"get_topology_mesh: {target_hardware=}")
  topology_devices = get_topology_desc(
      platform=target_hardware.platform,
      topology_name=target_hardware.topology_name,
      chip_config_name=target_hardware.chip_config_name,
      chips_per_host_bounds=target_hardware.chips_per_host_bounds,
      num_slices=config.compile_topology_num_slices,
      wrap=target_hardware.wrap,
  ).devices
  print(f"get_topology_mesh: {topology_devices=}")
  topology_device_mesh = max_utils.create_device_mesh(config, topology_devices)
  print(f"get_topology_mesh: {topology_device_mesh=}")
  topology_mesh = Mesh(topology_device_mesh, config.mesh_axes)
  print(f"get_topology_mesh: {topology_mesh=}")
  return topology_mesh

@struct.dataclass
class DecodeState:
  """The inputs into a generation step."""

  prefill_cache: jax.Array
  generate_cache: jax.Array
  generate_cache_index: int
  generate_lengths: jax.Array
  generated_token: jax.Array


class MaxEngineConfig:
  """Engine specific config class to allow using multiple MaxEngine instances in an inference run.
  The default pyconfig.config is a global param shared across multiple instances and doesn't
  allow using different config for each MaxEngine instance.
  """

  def __init__(self, keys):
    # self.keys = keys
    self.__dict__["keys"] = keys

  def __getattr__(self, attr):
    if attr not in self.keys:
      raise ValueError(f"Requested key {attr}, not in config")
    return self.keys[attr]

  def __setattr__(self, attr, value):
    raise ValueError

  def get_keys(self):
    return self.keys


class MaxEngine(engine_api.Engine):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  JetStream efficient serving infrastructure.
  """

  def __init__(self, config):
    self.config = config

    ici_parallelism = [
      config.ici_data_parallelism,
      config.ici_pipeline_parallelism,
      config.ici_fsdp_parallelism,
      config.ici_fsdp_transpose_parallelism,
      config.ici_sequence_parallelism,
      config.ici_tensor_parallelism,
      config.ici_expert_parallelism,
      config.ici_autoregressive_parallelism,
    ]

    if config.mesh_type == "balanced_2d":
      # original code: https://source.corp.google.com/piper///depot/google3/learning/gemini/gemax/core/compilation/scheduling.py;l=931;bpv=0;bpt=0
      print("Creating Balanced2DPartitionConfig mesh")
      mesh_axis_names = tuple(config.mesh_axes)
      nested_balanced_2d_devices = make_nested_balanced_2d_devices(jax.devices(), ici_parallelism)
      self._mesh = Mesh(nested_balanced_2d_devices, mesh_axis_names)
    elif config.mesh_type == "balanced_2d_reversed":
      print("Creating Balanced2DPartitionConfig mesh")
      mesh_axis_names = tuple(reversed(config.mesh_axes))
      nested_balanced_2d_devices = make_nested_balanced_2d_devices(jax.devices(), ici_parallelism)
      self._mesh = Mesh(nested_balanced_2d_devices, mesh_axis_names)
    elif config.mesh_type == "grid_of_rings":
      print("Creating GridOfRingsPartitionConfig mesh")
      # original code: https://source.corp.google.com/piper///depot/google3/learning/gemini/gemax/core/compilation/scheduling.py;l=761;bpv=0;bpt=0
      ici_mesh = dict(zip(config.mesh_axes, ici_parallelism))
      grid_of_rings_partition_config = GridOfRingsPartitionConfig(outer_axis_name='tensor', inner_axis_name='sequence')
      grid_of_rings_partition_config.ici_mesh = ici_mesh
      self._mesh = grid_of_rings_partition_config.make_mesh(jax.devices())
    else: 
      print("Creating default mesh")
      devices_array = max_utils.create_device_mesh(config)
      self._mesh = Mesh(devices_array, config.mesh_axes)
    print(f"Created mesh: {self._mesh=}")
      

    # Model and Optimizer definition
    quant = quantizations.configure_quantization(config)
    self.model = models.Transformer(config, mesh=self._mesh, quant=quant)
    self.replicated_sharding = jax.sharding.NamedSharding(self._mesh, P(None))

    self.abstract_params = None
    self.kv_cache_annotations = None
    self.kv_cache_annotations_named = None
    self.kv_cache_shardings = None
    self.state_mesh_annotations = None

  def load_params(self, *args, rng: Optional[jax.random.PRNGKey] = None, **kwargs) -> Params:
    """Load Parameters, typically from GCS"""
    # pylint: disable=unused-argument

    if rng is None:
      rng = jax.random.PRNGKey(0)

    if self.model.quant and self.config.checkpoint_is_quantized:
      print("Loading from the quantized checkpoint...")
      self.model.quant.quant_mode = quantizations.get_quant_mode("serve")

    rng1, rng2, rng3 = jax.random.split(rng, 3)
    state, self.state_mesh_annotations = max_utils.setup_decode_state(self.model, self.config, rng1, self._mesh, None)
    self.abstract_params = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding), state.params
    )
    self.kv_cache_annotations = max_utils.get_kv_cache_annotations(self.model, self.config, rng2, self._mesh)
    self.kv_cache_shardings = jax.tree_util.tree_map(
        lambda x: jax.sharding.NamedSharding(self._mesh, x), self.kv_cache_annotations
    )

    if self.model.quant and not self.config.checkpoint_is_quantized:
      params = self.quantize_params(state, rng3)
    else:
      params = state.params
    max_utils.print_mem_stats("After load_params")
    return params

  def quantize_params(self, state, rng: Optional[jax.random.PRNGKey] = None):
    """Forward pass to quantize decode params."""
    if rng is None:
      rng = jax.random.PRNGKey(0)

    self.model.quant.quant_mode = quantizations.get_quant_mode("convert")

    @jax.jit
    def model_apply(_p, _rng):
      return self.model.apply(
          _p | {"aqt": {}},
          jnp.ones((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          jnp.ones((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          decoder_segment_ids=jnp.zeros((1, self.config.max_prefill_predict_length), dtype=jnp.int32),
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_PREFILL,
          rngs={"params": _rng},
          mutable=True,
      )

    _, new_vars = model_apply(state.params, rng)
    # Remove param values which have corresponding qtensors in aqt to save memory.
    params = {}
    params["aqt"] = new_vars["aqt"]
    params["params"] = quantizations.remove_quantized_params(state.params["params"], new_vars["aqt"])
    self.abstract_params = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding), params
    )
    max_utils.save_quantized_checkpoint_if_configured(self.config, params)
    self.model.quant.quant_mode = quantizations.get_quant_mode("serve")
    return params

  @functools.partial(jax.jit, static_argnums=(0,))
  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: Optional[jax.Array] = None,
      padded_tokens: jax.Array,
      true_length: int,
      sampler: Optional[Callable[[Any], Any]] = None,  # pylint: disable=unused-argument
      rng: Optional[jax.random.PRNGKey] = None,
  ) -> Tuple[Prefix, engine_api.ResultTokens]:
    """Computes a kv-cache for a new generate request.

    Args:
      params: Scalar multiplier.
      existing_prefix: If provided, represents a prefix that has already been
        processed by the underlying model.
      padded_tokens: Logically appended tokens to any existing prefix, this is
        what we compute prefill on.
      true_length: The real length of the tokens, pre-pad.
    Returns:
      kv_cache: For the resulting text.
    """
    if existing_prefix:
      raise ValueError("We don't know what to do with existing_prefix")

    if rng is None:
      rng = jax.random.PRNGKey(0)

    input_tokens = jnp.expand_dims(padded_tokens, 0)  # [BATCH, SEQUENCE]
    positions = jnp.expand_dims(jnp.arange(0, input_tokens.shape[1]), 0)

    zero_to_n = jnp.arange(0, padded_tokens.shape[0])
    ones_to_keep = zero_to_n < true_length
    one_d_output = ones_to_keep * common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    sequence_indicator = jnp.expand_dims(one_d_output, 0)

    rng, new_rng = jax.random.split(rng)
    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      flat_logits, new_vars = self.model.apply(
          params,
          input_tokens,
          positions,
          decoder_segment_ids=sequence_indicator,
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_PREFILL,
          rngs={"params": new_rng},
          mutable=["cache"],
      )

    next_pos = jnp.full((1, 1), true_length, dtype=jnp.int32)
    generated_tokens = jnp.zeros((1, 1), dtype=jnp.int32)
    selected_logits = jax.lax.dynamic_slice(
        flat_logits, (0, true_length - 1, 0), (flat_logits.shape[0], 1, flat_logits.shape[2])
    )
    selected_logits = jax.lax.with_sharding_constraint(selected_logits, self.replicated_sharding)

    # sampling first token
    first_generated_token = inference_utils.sampling(
        selected_logits,
        rng,
        self.config.decode_sampling_strategy,
        topk=self.config.decode_sampling_top_k,
        nucleus_topp=self.config.decode_sampling_nucleus_p,
        temperature=self.config.decode_sampling_temperature,
    )

    all_valid = jnp.ones(first_generated_token.shape, dtype=jnp.int8)
    result = engine_api.ResultTokens(
        data=jnp.concatenate((first_generated_token, all_valid, generated_tokens), axis=1),
        # Tokens are shape [batch, speculations], so when we concatenate
        # tokens, validity and length along their index 1 dimension then they
        # occupy 0:speculations.
        tokens_idx=(0, 1),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(1, 2),
        # And lengths is rank 1.
        length_idx=(2, 3),
        samples_per_slot=1,
    )

    return {
        "logits": selected_logits,
        "cache": new_vars["cache"],
        "next_pos": next_pos,
        "generated_tokens": generated_tokens,
        "tokens": first_generated_token,
    }, result

  @functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2,))
  def generate(
      self,
      params: Params,
      decode_state: DecodeState,
      sampler: Optional[Callable[[Any], Any]] = None,  # pylint: disable=unused-argument
      rng: Optional[jax.random.PRNGKey] = None,
  ) -> Tuple[DecodeState, engine_api.ResultTokens]:
    """Run one generate step"""
    if rng is None:
      rng = jax.random.PRNGKey(0)

    previous_token = decode_state["tokens"]

    rng, new_rng = jax.random.split(rng)
    # run one step generation
    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      out_logits, new_vars = self.model.apply(
          params | {"cache": decode_state["cache"]},
          previous_token,
          decode_state["next_pos"],
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
          rngs={"params": new_rng},
          mutable=["cache"],
      )

    out_logits = jax.lax.with_sharding_constraint(out_logits, self.replicated_sharding)
    new_cache = jax.lax.with_sharding_constraint(new_vars["cache"], self.kv_cache_shardings)

    # sampling tokens
    new_token = inference_utils.sampling(
        out_logits,
        rng,
        self.config.decode_sampling_strategy,
        topk=self.config.decode_sampling_top_k,
        nucleus_topp=self.config.decode_sampling_nucleus_p,
        temperature=self.config.decode_sampling_temperature,
    )

    all_valid = jnp.ones(new_token.shape, dtype=jnp.int8)
    result = engine_api.ResultTokens(
        data=jnp.concatenate((new_token, all_valid, decode_state["generated_tokens"]), axis=1),
        # Tokens are shape [batch, speculations], so when we concatenate
        # tokens, validity and length along their index 1 dimension then they
        # occupy 0:speculations.
        tokens_idx=(0, 1),
        # Validity occupies the same amount of space, but next in line.
        valid_idx=(1, 2),
        # And lengths is rank 1.
        length_idx=(2, 3),
        samples_per_slot=1,
    )

    return {
        "logits": out_logits,
        "cache": new_cache,
        "next_pos": decode_state["next_pos"] + 1,
        "generated_tokens": decode_state["generated_tokens"] + 1,
        "tokens": new_token,
    }, result

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
      donate_argnums=(
          1,
          2,
      ),
  )
  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    """Insert into KV cache"""
    unboxed_prefix = max_utils.unbox_logicallypartioned(prefix)

    def copy(path, partial_cache, full_cache, annotations):
      path_key = path[-1].key
      if path_key in ["cache_ar_index", "cached_ar_key", "cached_ar_value", "cached_ar_key_scale", "cached_ar_value_scale"]:
        return full_cache  # we don't even zero these out because we can mask them out.

      batch_idx = -1
      if "cache_batch" in annotations:
        batch_idx = annotations.index("cache_batch")
      elif "cache_scale_batch" in annotations:
        batch_idx = annotations.index("cache_scale_batch")

      if batch_idx < 0:
        raise ValueError(f"Batch index {batch_idx=} shouldn't be less than zero for {path_key}, got {annotations=}")

      if path_key == "cache_ar_segment_id":
        ### goal: zero this out in case there is existing data
        s = list(full_cache.shape)
        s[batch_idx] = 1
        zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
        return jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
      elif path_key == "cache_prefill_segment_id":
        s = list(full_cache.shape)
        s[batch_idx] = 1
        zeros = jnp.zeros(tuple(s), dtype=jnp.int32)
        ## zero out in case prefill cache is too small to cover
        full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, zeros, slot, batch_idx)
        ## copy prefill cachce
        full_cache = jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
        return full_cache
      elif path_key == "cached_ar_lengths":
        return full_cache.at[slot].set(0)
      elif path_key in [
          "cached_prefill_key",
          "cached_prefill_value",
          "cached_prefill_key_scale",
          "cached_prefill_value_scale",
      ]:
        return jax.lax.dynamic_update_index_in_dim(full_cache, partial_cache, slot, batch_idx)
      else:
        raise ValueError(f"We don't have a strategy for inserting {path_key}")

    inserted_cache = jax.tree_util.tree_map_with_path(
        copy, unboxed_prefix["cache"], decode_state["cache"], self.kv_cache_annotations_named
    )
    inserted_logits = jax.lax.dynamic_update_index_in_dim(decode_state["logits"], unboxed_prefix["logits"], slot, 0)
    inserted_next_pos = jax.lax.dynamic_update_index_in_dim(decode_state["next_pos"], unboxed_prefix["next_pos"], slot, 0)
    inserted_generated_tokens = jax.lax.dynamic_update_index_in_dim(
        decode_state["generated_tokens"], unboxed_prefix["generated_tokens"], slot, 0
    )
    inserted_tokens = jax.lax.dynamic_update_index_in_dim(decode_state["tokens"], unboxed_prefix["tokens"], slot, 0)

    inserted_logits = jax.lax.with_sharding_constraint(inserted_logits, self.replicated_sharding)
    inserted_generated_tokens = jax.lax.with_sharding_constraint(inserted_generated_tokens, self.replicated_sharding)
    inserted_next_pos = jax.lax.with_sharding_constraint(inserted_next_pos, self.replicated_sharding)
    inserted_tokens = jax.lax.with_sharding_constraint(inserted_tokens, self.replicated_sharding)
    inserted_cache = jax.lax.with_sharding_constraint(inserted_cache, self.kv_cache_shardings)

    return {
        "logits": inserted_logits,
        "cache": inserted_cache,
        "next_pos": inserted_next_pos,
        "generated_tokens": inserted_generated_tokens,
        "tokens": inserted_tokens,
    }

  def get_prefix_destination_sharding(self) -> Any:
    return jax.sharding.NamedSharding(mesh=self.mesh, spec=jax.sharding.PartitionSpec())

  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    """Return a protobuf of tokenizer info, callable from Py or C++."""
    return tokenizer_pb2.TokenizerParameters(path=self.config.tokenizer_path, extra_ids=0)

  def build_tokenizer(self, metadata: tokenizer_pb2.TokenizerParameters) -> tokenizer_api.Tokenizer:
    """Return a tokenizer"""
    if "tiktoken" in metadata.path:
      return token_utils.TikToken(metadata)
    else:
      return token_utils.SentencePieceTokenizer(metadata)

  def init_decode_state(
      self,
      *args,  # pylint: disable=unused-argument
      rng: Optional[jax.random.PRNGKey] = None,
      **kwargs,  # pylint: disable=unused-argument
  ) -> DecodeState:
    """Initialises any state which a generation step transforms."""

    if rng is None:
      rng = jax.random.PRNGKey(0)

    # pylint: disable=unused-argument
    def init(abstract_params):
      x = jnp.ones(
          (int(self.config.per_device_batch_size * jax.device_count()), self.config.max_prefill_predict_length),
          dtype=jnp.int32,
      )
      _, cache = self.model.apply(
          abstract_params,
          x,
          x,
          decoder_segment_ids=jnp.zeros(x.shape, dtype=jnp.int32) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR,
          enable_dropout=False,
          model_mode=common_types.MODEL_MODE_PREFILL,
          rngs={"params": rng},
          mutable=["cache"],
      )

      next_pos = jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
      generated_tokens = jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
      tokens = jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1), dtype=jnp.int32)
      return {
          "logits": jnp.zeros((int(self.config.per_device_batch_size * jax.device_count()), 1, self.config.vocab_size)),
          "cache": cache["cache"],
          "next_pos": next_pos,
          "generated_tokens": generated_tokens,
          "tokens": tokens,
      }

    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      abstract_outputs = jax.eval_shape(init, self.abstract_params)
    logical_annotations = nn.get_partition_spec(abstract_outputs)

    with self._mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      mesh_annotations = nn.logical_to_mesh(logical_annotations)

    shardings = jax.tree_util.tree_map(
        lambda mesh_annotation: jax.sharding.NamedSharding(self._mesh, mesh_annotation), mesh_annotations
    )

    @functools.partial(jax.jit, out_shardings=shardings)
    def initialize():
      return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), abstract_outputs)

    cache = initialize()["cache"]

    def is_lp(k):
      return isinstance(k, flax.linen.spmd.LogicallyPartitioned)

    self.kv_cache_annotations_named = jax.tree_util.tree_map(lambda x: tuple(x.names), cache, is_leaf=is_lp)
    del cache
    zeroed = max_utils.unbox_logicallypartioned(initialize())
    return zeroed

  @property
  def max_concurrent_decodes(self) -> int:
    """Free slots."""
    return int(self.config.per_device_batch_size * jax.device_count())

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""
    return int(self.config.max_prefill_predict_length)

  @property
  def samples_per_slot(self) -> int:
    """Number of samples per slot."""
    return 1

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._mesh

  @property
  def colocated_cpus(self) -> None:
    """CPU devices colocated with the engine's accelerators."""
    raise NotImplementedError


def set_engine_vars_from_base_engine(engine: engine_api.Engine, base_engine: engine_api.Engine, rng: jax.random.PRNGKey):
  """Set internal vars from base_engine, which has already loaded the checkpoint and has sharding,
  mesh, and kv cache related vars set.
  """
  engine.model.quant.quant_mode = base_engine.model.quant.quant_mode
  engine.state_mesh_annotations = base_engine.state_mesh_annotations
  engine.abstract_params = base_engine.abstract_params
  engine.kv_cache_annotations = max_utils.get_kv_cache_annotations(engine.model, engine.config, rng, engine._mesh)  # pylint: disable=protected-access
  engine.kv_cache_shardings = jax.tree_util.tree_map(
      lambda x: jax.sharding.NamedSharding(engine._mesh, x), engine.kv_cache_annotations  # pylint: disable=protected-access
  )


def create_engine_from_config_flags(batch_size, max_prefill_predict_length, max_target_length, args_str):
  """Create new MaxEngine instance with given batch_size, prefill and target lengths, and any config
  params provided through `args_str`.
  """
  args = {}
  args["scan_layers"] = "false"
  args["async_checkpointing"] = "false"
  args["ici_fsdp_parallelism"] = "1"
  args["ici_autoregressive_parallelism"] = "1"
  args["ici_tensor_parallelism"] = "-1"
  args["weight_dtype"] = "bfloat16"
  args["attention"] = "dot_product"

  # batch and cache related
  args["max_prefill_predict_length"] = f"{max_prefill_predict_length}"
  args["max_target_length"] = f"{max_target_length}"
  args["per_device_batch_size"] = f"{batch_size}"
  print(f"Command line args: {args_str}")
  cmd_args = args_str.split(" ")
  for cmd_arg in cmd_args:
    k, v = cmd_arg.split("=")
    args[k.strip()] = v.strip()
  assert "load_parameters_path" in args, "load_parameters_path must be defined"
  updated_args = ["MaxText/maxengine_server.py", "../configs/base.yml"]
  for k, v in args.items():
    option = f"{k}={v}"
    updated_args.append(option)
  print(f"Invoking maxengine with args:\n \t{updated_args}")
  pyconfig.initialize(updated_args)
  cfg = MaxEngineConfig(cp.deepcopy(pyconfig._config.keys))  # pylint: disable=protected-access
  engine = MaxEngine(cfg)
  return engine
