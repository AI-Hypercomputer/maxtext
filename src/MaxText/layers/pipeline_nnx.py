# Copyright 2023–2025 Google LLC
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

""" Pipeline layer wrapping a decoder layer(s). Supports circular pipelining """

import functools
from typing import Any, Type

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh
import jax
import jax.ad_checkpoint

from flax import nnx
from flax import linen as nn

from MaxText.common_types import Config, MODEL_MODE_TRAIN
from MaxText.maxtext_utils import all_gather_over_fsdp
from MaxText.layers import initializers
from MaxText.layers import nnx_wrappers


class Pipeline(nnx.Module):
  """Module that implements pipelining across stages.

  This module will loop over microbatches and execute the main body with a vmap for both the inputs and weights.
  This will produce a pipeline pattern if the stage dimension is sharded.

  Supports circular pipelines, and multiple layers per stage are used when a module that executes multiple layers
  is passed as the layers input.

  Attributes:
    config: Configuration object.
    layer_module: The NNX Module *class* to be pipelined (e.g., LlamaDecoderLayer).
    mesh: The device mesh of the system.
    remat_policy: Remat policy to use for the loop iterations.
    layers: A *single* NNX module instance (e.g., LlamaDecoderLayer) whose
            parameters and state are stacked with dimensions
            [num_repeats, num_stages, ...].
    single_layer_graphdef: A GraphDef for a *single* un-stacked layer,
                           used for merging during the vmapped call.
    Attributes:
      config: Importantly contains num_pipeline_microbatches, num_pipeline_repeats.
      layers: A module instance that each stage can execute. It can either be a single layer such as a
        LlamaDecoderLayer instance or scanned/looped set of decoder layers to execute multiple layers per stage.
      mesh:  The device mesh of the system.
      remat_policy: Remat policy to use for the loop iterations
  """

  def __init__(
      self,
      config: Config,
      layer_module: Type[nnx.Module],
      mesh: Mesh,
      remat_policy: Any = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy
    # self.rngs = rngs

    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    self.microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.use_circ_storage = self.need_circ_storage()

    # --- NNX: Eagerly initialize the stacked layers ---

    # 1. Define a factory function to create one layer
    def create_layer(rngs_pytree):
      # --- ROBUST FIX + DEBUG ---
      raw_rngs = {}
      # If rngs_pytree is already an nnx.Rngs object, convert to dict first
      rngs_dict = dict(rngs_pytree) if isinstance(rngs_pytree, nnx.Rngs) else rngs_pytree

      for name, val in rngs_dict.items():
        # If it's an nnx.RngKey or similar wrapper, it holds the raw key in .value
        if hasattr(val, "tag") and hasattr(val, "value"):
          # It's an NNX wrapper (RngKey, RngStream, etc.)
          raw_rngs[name] = val.value
        else:
          # It's likely already a raw JAX array
          raw_rngs[name] = val

      # print(f"DEBUG: create_layer raw_rngs types: {jax.tree.map(type, raw_rngs)}")
      return layer_module(self.config, rngs=nnx.Rngs(**raw_rngs))
      # --------------------------

    # 2. Create the vmapped factory functions
    vmap_over_stages = nnx.vmap(
        create_layer,
        in_axes=0,
        out_axes=0,
        spmd_axis_name="stage",
    )
    if self.config.num_pipeline_repeats > 1:
      vmap_over_repeats = nnx.vmap(
          vmap_over_stages,
          in_axes=0,
          out_axes=0,
          spmd_axis_name="circular_repeats",
      )

    # 3. Prepare the stacked RNGs Pytree
    # Convert input rngs to a pure dict of raw JAX keys first to avoid NNX state issues.
    rngs_pure_dict = {}
    for name, stream in rngs.items():
      # Handle both raw JAX keys and NNX streams/keys in the input
      if hasattr(stream, "value"):
        rngs_pure_dict[name] = stream.value
      elif hasattr(stream, "key") and hasattr(stream.key, "value"):
        rngs_pure_dict[name] = stream.key.value
      else:
        rngs_pure_dict[name] = stream

    def stack_keys(key):
      # Ensure it's a valid JAX key
      if isinstance(key, int):
        key = jax.random.PRNGKey(key)
      elif (
          hasattr(key, "dtype")
          and hasattr(key, "ndim")
          and key.ndim == 0
          and not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
      ):
        key = jax.random.PRNGKey(key)

      # Split using shape tuple to avoid dimension issues
      if self.config.num_pipeline_repeats > 1:
        return jax.random.split(key, (self.config.num_pipeline_repeats, self.num_stages))
      else:
        return jax.random.split(key, self.num_stages)

    init_rngs_pytree = jax.tree.map(stack_keys, rngs_pure_dict)

    # 4. Create the GraphDef for a *single* layer.
    graphdef_rngs_pytree = jax.tree.map(
        lambda x: x[0, 0] if self.config.num_pipeline_repeats > 1 else x[0], init_rngs_pytree
    )

    self.single_layer_graphdef = nnx.graphdef(create_layer(rngs_pytree=graphdef_rngs_pytree))

    # 5. Create the stacked layers
    if self.config.num_pipeline_repeats > 1:
      self.layers = vmap_over_repeats(init_rngs_pytree)
    else:
      self.layers = vmap_over_stages(init_rngs_pytree)
      self.layers = self.layers.add_axis(0, "circular_repeats")

  def need_circ_storage(self):
    return (
        self.config.num_pipeline_repeats > 1
        and self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay
    )

  def iterations_to_complete_first_microbatch_one_repeat(self):
    # Return the number of iterations it takes for microbatch 0 to finish a repeat
    return self.forwarding_delay * (self.num_stages - 1)

  def iterations_to_complete_first_microbatch(self):
    # Return the number of iterations it takes for microbatch 0 to finish the last stage of the last repeat
    return (
        self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1)
        + self.iterations_to_complete_first_microbatch_one_repeat()
    )

  def init_states(self, inputs):
    """Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
    Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

    Returns a dictionary with properties
      shift: zeros shape [num_stages, micro_size, sequence, embed]
      prev_outputs: same shape as shift, only used when pipeline_delay_activation_forwarding is set to true, else None
      state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
      circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed] when needed, else None
      circ_storage_mover: zeros[num_stages, micro_size, sequence, embed] when needed, else None
      loop_iteration: scalar set initially to 0.
    """

    # Shift is used to rotate the output of each pipeline into the input of the next
    # shift has shape [num_stages, micro_size, sequence, embed]
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = nn.with_logical_constraint(
        shift,
        ("activation_stage", "activation_batch", "activation_length", "activation_embed"),
        rules=self.config.logical_axis_rules,
        mesh=self.mesh,
    )

    # Prev outputs has the same shape of the output (and shift)
    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = nn.with_logical_constraint(
          prev_outputs,
          ("activation_stage", "activation_batch", "activation_length", "activation_embed"),
          rules=self.config.logical_axis_rules,
          mesh=self.mesh,
      )
    else:
      prev_outputs = None

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs
    #   as the pipeline runs/finishes
    # state_io has shape [num_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
    # We shard the pipeline_microbatch_size axis by data/fsdp, not num_microbatches since those are looped over.
    state_io = nn.with_logical_constraint(
        state_io,
        ("activation_stage", None, "activation_batch", "activation_length", "activation_embed"),
        rules=self.config.logical_axis_rules,
        mesh=self.mesh,
    )

    # circ_storage is used to hold the final pipeline stage outputs before it is used for the next repeat. It is only
    # needed when num_microbatches > num_stages, else instead the final stage will immediately pass to the first without
    # additional storage.
    # circ_storage has shape [num_stages, microbatches, micro_size, sequence, embed].
    # Note that this shape is a factor of num_stages larger than necessary - each stage holds the global batch, but only
    # stage 0 holds the real activations (since it will use them), the rest hold dummy ones. This amount of storage
    # [global_batch, sequence, embed] is fine as long as there is some amount of additional sharding axes, e.g. FSDP,
    # TP, DP (e.g. there are many devices that shard stage 0)
    # We may look into alternatives using less storage if this becomes an issue (ideas in b/347603101).
    if self.use_circ_storage:
      circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype)
      circ_storage_mover = shift
    else:
      circ_storage = None
      circ_storage_mover = None

    init_loop_state = {
        "state_io": state_io,
        "shift": shift,
        "circ_storage": circ_storage,
        "circ_storage_mover": circ_storage_mover,
        "loop_iteration": 0,
        "prev_outputs": prev_outputs,
    }
    return init_loop_state

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    """
    Construct stages_in: the global array that is operated on for this iteration, shape same as
    shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab a new batch from
    state_io or an old one from circ_storage
    """

    # Setup potential input from state_io, which has a rotating microbatch index (size of microbatches_per_stage)
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:, state_io_batch_idx]

    if self.use_circ_storage:
      # Setup potential input from circ_storage, which also has a rotating index for microbatch,
      # size of num_microbatches
      circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
      circular_stage_in = circ_storage[:, circ_storage_batch_idx]
    else:
      # The last stage immediately flows into the first stage, use this rotated shift instead of circular storage
      circular_stage_in = shift

    # For early loop iterations we grab a new input for stage 0 from the state_io. Once each microbatch has left
    # state_io we instead grab from the last stage's output (possibly buffered when num_microbatches > num_stages, e.g.
    # from circ_storage).
    first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)

    # Note that first_stage_in may correspond to bubble computation during the last few iterations.
    # However, these bubble computation results remain in the shift buffer (do not make it back to state_io) and are
    # thus discarded / not returned.
    # The final returned output is stored in the state_io, which has the appropriate total size of num_microbatches. The
    # state_io will not contain bubble results at the end of the last iteration.

    def select_state_or_input(first_stage_in, shift):
      # Selects input for stage 0, shift for other stages
      return jnp.where(jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, first_stage_in, shift)

    # Selects input (from stream_io) for stage 0, other stages get from shift (the rotated previous output)
    stages_in = select_state_or_input(first_stage_in, shift)
    stages_in = nn.with_logical_constraint(
        stages_in,
        ("activation_stage", "activation_batch", "activation_length", "activation_embed"),
        rules=self.config.logical_axis_rules,
        mesh=self.mesh,
    )
    return stages_in

  def shard_dim_by_stages(self, x, dim: int):
    # Shards a dimension by stages. Currently, the sharding of other dimensions are left up the compiler, alternatively
    # we may want to copy over the sharding from the other input axes.
    dims_mapping = [jax.sharding.PartitionSpec.UNCONSTRAINED] * x.ndim
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(*dims_mapping))
    return jax.lax.with_sharding_constraint(x, sharding)

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    """Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration. Works for both circular and
    non-circular"""
    # Stage 0 has processed one microbatch every loop_iter, but Stage 1 is 1 behind due to bubble, etc for other stages
    microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    repeat_ids = microbatches_processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def vmap_parallel_gather(self, weights, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
    """Use vmap to implement a sharded parallel gather.
    Parallel gather means each stage has its own weights, and gets one slice from it.
    Args:
      weights: Per-stage data to be gathered from.
      repeat_ids: Integer tensor of shape [num_stages], the repeats of the stages.
      repeat_dim_in_weights: The dimension in weights where repeat_ids are applied. The output will not
        have this dimension.
      stages_dim_in_weights: The dimension in weights that represents parallel stages.
    Returns:
      The per-stage gathered values. The shape is weights.shape but with repeat_dim_in_weights
        removed.
    """

    def _gather_one(x, repeat_id):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, repeat_dim_in_weights), repeat_dim_in_weights)

    gathered_weights_stage_dim = 0
    repeat_ids = self.shard_dim_by_stages(repeat_ids, 0)
    weights = self.shard_dim_by_stages(weights, stages_dim_in_weights)
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(
        weights, repeat_ids
    )
    stage_weights = self.shard_dim_by_stages(stage_weights, gathered_weights_stage_dim)
    return stage_weights

  def vmap_gather(self, xs, ids, ids_dim):
    """Use vmap to implement a stage-wise sharded gather.

    The stages share the same input, but they have different offsets.

    Args:
      xs: Data shared by all stages, to be gathered from.
      ids: Integer tensor of shape [num_stages], the offsets of the stages.
      ids_dim: The dimension in xs where ids are applied. In the output, this
        dimension will be [num_stages], since each stage gets one slice.

    Returns:
      The per-stage gathered values. The shape is xs.shape but with ids_dim size
        replaced with [num_stages].
    """

    def _gather_one(x, i):
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, i, 1, ids_dim), ids_dim)

    ids = self.shard_dim_by_stages(ids, 0)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
    return self.shard_dim_by_stages(outs, 0)

  def get_new_loop_state(self, output, loop_state):
    """
    Update the various buffers given the output of the most recent iteration
    * state_io: rotates left/up by 1 (the whole created in the last slot is filled with the most recent pipeline output)
       * Pushing inputs up from top of state_io into first stage of shift
       * Pulling outputs up from last stage of shift into bottom of state_io
    * shift: rotate output (or prev_outputs if using delay) right/down by 1 - we imagine the pipeline moves to
               right/down
    * circ_storage: pushes circ_storage_mover (the output of the previous iteration) into rotating index of circ_storage
    * circ_storage_mover: assigned to rotated output and pushed into circ_storage on the next iteration
    * prev_outputs: is set to the current output
    """

    old_state_io = loop_state["state_io"]
    old_circ_storage = loop_state["circ_storage"]
    old_circ_storage_mover = loop_state["circ_storage_mover"]
    loop_iteration = loop_state["loop_iteration"]
    old_prev_outputs = loop_state["prev_outputs"]

    def _rotate_right(arr):
      # Use lax.slice to avoid generating a gather.
      last = jax.lax.slice_in_dim(arr, self.num_stages - 1, self.num_stages, axis=0)
      except_last = jax.lax.slice_in_dim(arr, 0, self.num_stages - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)

    def _shift_right(arr):
      padding = [[1, 0]] + [[0, 0]] * (arr.ndim - 1)
      # Use lax.slice to guarantee the gradient is a pad.
      return jax.lax.slice(jnp.pad(arr, padding), [0] * arr.ndim, arr.shape)

    # Shift either rotates or shifts depending on if the last stage immediately must send to first or not
    # For non-circular pipelines, the last stage does not need to send to first
    # For circular pipelines with #micro = #stages, last stage immediately sends to first
    # For circular pipelines with #micro > stages (circ_storage), last stage sends to circ storage
    def _update_shift(output_in):
      if self.config.num_pipeline_repeats == 1 or self.use_circ_storage:
        return _shift_right(output_in)  # last stage does not have to send to first immediately
      else:
        return _rotate_right(output_in)  # last stage must immediately send to first

    if self.config.pipeline_delay_activation_forwarding:
      new_shift = _update_shift(old_prev_outputs)
      new_prev_outputs = output
    else:
      new_shift = _update_shift(output)
      new_prev_outputs = None

    if self.use_circ_storage:
      # Insert the circ_storage_mover into new_circ_storage at a microbatch-rotating index.
      # circ_storage_mover still points to the output of PREVIOUS iteration, which should aid in allowing overlapped
      # compute/async transfers
      def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
        rotated = _rotate_right(circ_storage_mover_in)
        rotated = jnp.expand_dims(rotated, 1)
        # We rotate the pushing index into circ storage, and ensure that microbatch 0 lands in index 0
        offset = (
            loop_iteration - self.iterations_to_complete_first_microbatch_one_repeat() - 1
        ) % self.config.num_pipeline_microbatches
        return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)

      new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
      new_circ_storage_mover = output
    else:
      new_circ_storage = None
      new_circ_storage_mover = None

    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]

    def _update_state_io(state_in, stream_slice, output):
      padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
      stream_slice = jax.lax.slice_in_dim(jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
      stream_slice = jnp.where(
          jax.lax.broadcasted_iota("int32", stream_slice.shape, 0) == self.num_stages - 1, output, stream_slice
      )
      stream_slice = jnp.expand_dims(stream_slice, 1)
      return jax.lax.dynamic_update_slice_in_dim(state_in, stream_slice, stream_buf_idx, axis=1)

    new_state = _update_state_io(old_state_io, stream_slice, output)

    new_loop_state = {
        "state_io": new_state,
        "shift": new_shift,
        "circ_storage": new_circ_storage,
        "circ_storage_mover": new_circ_storage_mover,
        "loop_iteration": loop_iteration + 1,
        "prev_outputs": new_prev_outputs,
    }
    return new_loop_state

  def permute_output_micro_per_stage_dim(self, output):
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + microbatch_0_idx) % self.microbatches_per_stage
    output = output[:, permutation]
    return output

  def get_current_repeat_from_stages(self, full_state: nnx.State, loop_iteration: int):
    """
    Gathers the state for the current repeat for each stage.

    Args:
      full_state: The complete state Pytree of self.layers, with arrays
                  shaped [num_repeats, num_stages, ...].
      loop_iteration: The current loop iteration number.

    Returns:
      A state Pytree where arrays are shaped [num_stages, ...].
    """
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    # This function operates on the Pytrees *inside* the nnx.State
    def gather_pytree(path, pytree):
      del path
      return jax.tree.map(
          functools.partial(
              self.vmap_parallel_gather,
              repeat_ids=repeat_ids,
              repeat_dim_in_weights=0,  # Axis 0 is num_repeats
              stages_dim_in_weights=1,  # Axis 1 is num_stages
          ),
          pytree,
      )

    # Apply the gather to all variable collections
    return full_state.map(gather_pytree)

  def run_one_iteration(
      self,
      loop_state,
      positions,
      segment_ids,
      deterministic,
      model_mode,
      # rngs: nnx.Rngs,
  ):
    """
    Run one loop iteration.
    This function will be called by nnx.scan, so it will mutate
    `self.layers` (which is the first 'scan_self' argument to scan).
    """
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
    stages_positions = self.vmap_gather(positions, microbatch_ids, 0) if positions is not None else None
    stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0) if segment_ids is not None else None

    # 1. Split the stacked layers module into its definition and state
    #    `state` is a Pytree of arrays [R, S, ...]
    _, state = nnx.split(self.layers)

    # 2. Get the Pytree for the current stages' repeats
    #    `stage_vars_pytree` is a Pytree of arrays [S, ...]
    stage_vars_pytree = self.get_current_repeat_from_stages(state, loop_iteration)

    # 3. Define the function to run on a *single* stage
    #    This function will be vmapped.
    def run_stage_vmap_fn(
        stage_vars_slice,  # Pytree for one stage [D1, D2, ...]
        stage_input,
        stage_segment_id,
        stage_position,
    ):
      # Merge the single-stage variables with the single-layer GraphDef
      layer_module = nnx.merge(self.single_layer_graphdef, stage_vars_slice)
      # Run the layer
      output = layer_module(stage_input, stage_segment_id, stage_position, deterministic, model_mode)
      # Split to get the (potentially updated) state
      _, new_vars_slice = nnx.split(layer_module)
      return output, new_vars_slice

    # 4. Vmap the stage function over all stages
    vmapped_run_stage = nnx.vmap(
        run_stage_vmap_fn,
        in_axes=(0, 0, 0, 0),  # Map all inputs except rngs
        out_axes=0,
        spmd_axis_name="stage",
        # split_rngs={'dropout': True, 'jitter': True},
    )

    # Run the vmap
    stages_output, new_stage_vars_pytree = vmapped_run_stage(
        stage_vars_pytree,
        stages_inputs,
        stages_segment_ids,
        stages_positions,
    )
    if self.config.scan_layers:
      stages_output = stages_output[0]

    # 5. Scatter the updated state (`new_stage_vars_pytree`) back into the
    #    full state (`state`). We only need to do this for mutable state.
    def update_state(full_state_pytree, new_stage_state_pytree, repeat_ids):
      stage_ids = jnp.arange(self.num_stages)

      def scatter_leaf(full_array, update_slice):
        # full_array: [repeats, stages, ...]
        # update_slice: [stages, ...]
        return full_array.at[repeat_ids, stage_ids].set(update_slice)

      # jax.tree.map will traverse whatever structure full_state has (State, dict, etc.)
      return jax.tree.map(scatter_leaf, full_state_pytree, new_stage_state_pytree)

    new_full_state = update_state(state, new_stage_vars_pytree, repeat_ids)
    nnx.update(self.layers, new_full_state)

    new_loop_state = self.get_new_loop_state(stages_output, loop_state)
    return new_loop_state

  def get_pipeline_remat_policy(self):
    if self.config.remat_policy == "custom":
      return self.remat_policy

    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
    if self.remat_policy is not None:
      remat_policy = jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
    else:
      remat_policy = save_input_policy
    return remat_policy

  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      partition_spec=None,  # This is no longer used for weights
  ) -> jnp.ndarray:

    # Reshape inputs
    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        )
    )
    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))
    if positions is not None:
      positions = jax.lax.with_sharding_constraint(positions, ag_sharding)
      positions = positions.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
    if segment_ids is not None:
      segment_ids = jax.lax.with_sharding_constraint(segment_ids, ag_sharding)
      segment_ids = segment_ids.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )

    loop_state = self.init_states(inputs)

    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    # NNX: No need to all_gather_over_fsdp, NNX/MaxText handles this
    # via sharding annotations on the module itself.

    # This is the function nnx.scan will operate on.
    # Signature: (Module, Carry, X) -> (Module, (NewCarry, Y))
    def run_iteration_scannable(merged_carry):
      scan_self, loop_state = merged_carry
      # scan_self is the Pipeline module
      # loop_state is the Pytree carry
      new_loop_state = scan_self.run_one_iteration(loop_state, positions, segment_ids, deterministic, model_mode)
      # run_one_iteration mutates scan_self.layers in-place
      return (scan_self, new_loop_state), None

    if self.config.set_remat_policy_on_pipeline_iterations:
      run_iteration_rematted = nnx.remat(
          run_iteration_scannable,
          prevent_cse=not self.config.scan_pipeline_iterations,
          policy=self.get_pipeline_remat_policy(),
      )
    else:
      run_iteration_rematted = run_iteration_scannable

    if self.config.scan_pipeline_iterations:
      # nnx.scan will automatically carry mutable state (BatchStat)
      # and broadcast immutable state (Param).
      run_all_iterations_scanned = nnx.scan(
          run_iteration_rematted,
          # We don't need to specify variable_carry or variable_broadcast
          # NNX defaults (BatchStat=carry, Param=broadcast) are correct.
          length=total_iterations,
          in_axes=nnx.Carry,
      )
      # Run the scan
      (updated_self, final_loop_carry), _ = run_all_iterations_scanned((self, loop_state))
      _, new_state = nnx.split(updated_self)
      # Update self with the final state from the scan
      nnx.update(self, new_state)
      # Get the final loop state from the Pytree carry
      loop_state = final_loop_carry

    else:
      # Standard Python loop
      # We need to split/merge to use jax.remat
      graphdef, state = nnx.split(self)

      @functools.partial(
          jax.remat,
          prevent_cse=True,
          policy=self.get_pipeline_remat_policy(),
      )
      def jax_remat_body(state_pytree, loop_state_in, rngs: nnx.Rngs):
        # Re-merge module
        model = nnx.merge(graphdef, state_pytree)
        # Run one iter (this mutates model.layers)
        new_loop_state = model.run_one_iteration(
            loop_state_in,
            positions,
            segment_ids,
            deterministic,
            model_mode,
            rngs=rngs,
        )
        # Split module again to return new state
        _, new_state_pytree = nnx.split(model)
        return new_state_pytree, new_loop_state

      current_state_pytree = state
      rngs = self.rngs
      for _ in range(total_iterations):
        rngs, iter_rngs = rngs.fork()
        current_state_pytree, loop_state = jax_remat_body(current_state_pytree, loop_state, iter_rngs)

      # Update self with final state
      self.update(current_state_pytree)

    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])

    # reshape outputs to match input shape of total batch instead of microbatches [batch, sequence, embed]
    final_output = jnp.reshape(
        final_output, (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim)
    )

    return final_output


PipelineToLinen = nnx_wrappers.to_linen_class(
    Pipeline,
    base_metadata_fn=initializers.variable_to_logically_partitioned,
)
