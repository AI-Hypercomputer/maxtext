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

""" Pipeline layer wrapping a decoder layer(s). Supports circular pipelining """

import functools
from typing import Any

import numpy as np

from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import jax
import jax.ad_checkpoint

from flax.core import meta
from flax import linen as nn
from flax.linen.spmd import LogicallyPartitioned

from maxtext.common.common_types import Config, MODEL_MODE_TRAIN, EP_AS_CONTEXT, ShardMode
from maxtext.utils.sharding import (
    maybe_shard_with_logical,
    maybe_shard_with_name,
    create_sharding,
    logical_to_mesh_axes,
    logical_to_mesh,
)


class Pipeline(nn.Module):
  """Module that implements pipelining across stages.

  This module will loop over microbatches and execute the main body with a vmap for both the inputs and weights.
  This will produce a pipeline pattern if the stage dimension is sharded.

  Supports circular pipelines, and multiple layers per stage are used when a module that executes multiple layers
  is passed as the layers input.
  """

  #: Importantly contains num_pipeline_microbatches, num_pipeline_repeats.
  config: Config
  #: A module instance that each stage can execute. It can either be a single layer such as a LlamaDecoderLayer instance or
  #: scanned/looped set of decoder layers to execute multiple layers per stage. The name of this property (layers) is
  #: reflected in the state pytree and thus also checkpoints.
  layers: nn.Module
  #: The device mesh of the system.
  mesh: Mesh
  #: Remat policy to use for the loop iterations
  remat_policy: Any = None

  def setup(self):  # pylint: disable=missing-function-docstring
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if self.config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = self.config.micro_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.microbatches_per_stage = microbatches_per_stage
    self.use_circ_storage = self.need_circ_storage()

    if self.config.expert_shard_attention_option == EP_AS_CONTEXT:
      self.batch_axis_name = "activation_batch_no_exp"
      self.seq_len_axis_name = "activation_length"
    else:
      self.batch_axis_name = "activation_batch"
      self.seq_len_axis_name = "activation_length_no_exp"

    # TODO(b/470167805): replace self.spmd_axis_name with "stage" when JAX >= 0.8.2.
    self.spmd_axis_name = "stage" if self.config.shard_mode == ShardMode.AUTO else None

    self.stages_in_logical = ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
    self.stages_in_spec = logical_to_mesh_axes(self.stages_in_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.stages_in_sharding = (
        NamedSharding(self.mesh, self.stages_in_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )

    self.state_io_logical = ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed")
    self.state_io_spec = logical_to_mesh_axes(self.state_io_logical, self.mesh, rules=self.config.logical_axis_rules)
    self.state_io_sharding = (
        NamedSharding(self.mesh, self.state_io_spec) if self.config.shard_mode == ShardMode.EXPLICIT else None
    )
    self.input_sharding = (
        create_sharding(
            self.mesh,
            (None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )
    self.output_sharding = (
        create_sharding(
            self.mesh,
            (self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
            rules=self.config.logical_axis_rules,
        )
        if self.config.shard_mode == ShardMode.EXPLICIT
        else None
    )

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

  def _maybe_shard_with_logical(self, inputs, logical_axes):
    """Wrapper of maybe_shard_with_logical"""
    return maybe_shard_with_logical(
        inputs,
        logical_axes,
        shard_mode=self.config.shard_mode,
        mesh=self.mesh,
        rules=self.config.logical_axis_rules,
        debug_sharding=self.config.debug_sharding,
    )

  def _maybe_shard_with_name(self, inputs, sharding_name):
    """Wrapper of maybe_shard_with_name"""
    return maybe_shard_with_name(
        inputs,
        sharding_name,
        shard_mode=self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
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
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

    # Prev outputs has the same shape of the output (and shift)
    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = self._maybe_shard_with_logical(prev_outputs, self.stages_in_logical)
    else:
      prev_outputs = None

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs
    #   as the pipeline runs/finishes
    # state_io has shape [num_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(
        inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:], out_sharding=self.state_io_sharding
    )
    # We shard the pipeline_microbatch_size axis by data/fsdp, not num_microbatches since those are looped over.
    state_io = self._maybe_shard_with_logical(state_io, self.state_io_logical)

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
      circ_storage = jnp.zeros((self.num_stages,) + inputs.shape, dtype=inputs.dtype, out_sharding=self.state_io_sharding)
    else:
      circ_storage = None

    # circ_storage_mover is used to push the microbatches from the pipeline into circ_storage with one buffer iteration
    # of delay circ_storage_mover shape is same as shift: [num_stages, micro_size, sequence, embed]
    if self.use_circ_storage:
      circ_storage_mover = shift
    else:
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
    shift = self._maybe_shard_with_logical(shift, self.stages_in_logical)

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
    first_stage_in = self._maybe_shard_with_logical(first_stage_in, self.stages_in_logical)

    # Note that first_stage_in may correspond to bubble computation during the last few iterations.
    # However, these bubble computation results remain in the shift buffer (do not make it back to state_io) and are
    # thus discarded / not returned.
    # The final returned output is stored in the state_io, which has the appropriate total size of num_microbatches. The
    # state_io will not contain bubble results at the end of the last iteration.

    def select_state_or_input(first_stage_in, shift):
      # Selects input for stage 0, shift for other stages
      return jnp.where(
          jax.lax.broadcasted_iota("int32", shift.shape, 0, out_sharding=self.stages_in_sharding) == 0,
          first_stage_in,
          shift,
      )

    # Selects input (from stream_io) for stage 0, other stages get from shift (the rotated previous output)
    stages_in = select_state_or_input(first_stage_in, shift)
    stages_in = self._maybe_shard_with_logical(stages_in, self.stages_in_logical)
    return stages_in

  def shard_dim_by_stages(self, x, dim: int, physical_partition_spec: P | None, is_stage_weight: bool = False):
    """Shards x using the provided partition_spec, but adds the "stage" mesh axis to the existing sharding at
    the specified dimension."""
    placeholder = None if self.config.shard_mode == ShardMode.EXPLICIT else P.UNCONSTRAINED
    if physical_partition_spec is None:
      dims_mapping = [placeholder] * x.ndim
    else:
      physical_partition_spec = self._remove_fsdp_from_physical_partition_spec(physical_partition_spec)
      dims_mapping = list(physical_partition_spec)
      # If not a stage weight, we handle the repeat dimension offset
      if not is_stage_weight:
        dims_mapping = [placeholder] * (dim + 1) + dims_mapping[dim:]  # inflat one dimension for num_repeats
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    # We add reduced rule only when pspec is given for a stage weight
    if physical_partition_spec and is_stage_weight and self.config.shard_mode == ShardMode.EXPLICIT:
      batch_mesh_axis = ["data", "fsdp"]
      reduced_mark = [mesh_axis for mesh_axis in batch_mesh_axis if self.mesh.shape[mesh_axis] > 1]
      pspec = P(*dims_mapping, reduced=set(reduced_mark))
    else:
      pspec = P(*dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh, pspec)
    return self._maybe_shard_with_name(x, sharding)

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    """Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration. Works for both circular and
    non-circular"""
    # Stage 0 has processed one microbatch every loop_iter, but Stage 1 is 1 behind due to bubble, etc for other stages
    microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    repeat_ids = microbatches_processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def vmap_parallel_gather(
      self, weights, physical_partition_spec, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights
  ):
    """Use vmap to implement a sharded parallel gather.
    Parallel gather means each stage has its own weights, and gets one slice from it.
    Args:
      weights: Per-stage data to be gathered from.
      physical_partition_spec: Physical partition spec of the input weight.
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
    repeat_ids = self.shard_dim_by_stages(repeat_ids, 0, physical_partition_spec=None)
    # num_repeats x num_stages x *param_dim
    weights = self.shard_dim_by_stages(
        weights, stages_dim_in_weights, physical_partition_spec=physical_partition_spec, is_stage_weight=False
    )
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(
        weights, repeat_ids
    )
    # num_stages x *param_dim
    stage_weights = self.shard_dim_by_stages(
        stage_weights, gathered_weights_stage_dim, physical_partition_spec=physical_partition_spec, is_stage_weight=True
    )
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
      idx = tuple(i if d == ids_dim else slice(None) for d in range(x.ndim))
      replicated_sharding = NamedSharding(self.mesh, P())
      return x.at[idx].get(out_sharding=replicated_sharding)

    ids = self.shard_dim_by_stages(ids, 0, physical_partition_spec=None)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
    return self.shard_dim_by_stages(outs, 0, physical_partition_spec=None)

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

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=self.stages_in_spec,
        out_specs=self.stages_in_spec,
        check_vma=True,
    )
    def _rotate_right(arr):
      # we use +1 for right shifting
      stage_size = jax.lax.axis_size("stage")
      perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
      arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
      return arr

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=self.stages_in_spec,
        out_specs=self.stages_in_spec,
        check_vma=True,
    )
    def _shift_right(arr):
      stage_idx = jax.lax.axis_index("stage")
      stage_size = jax.lax.axis_size("stage")
      perm = [(i, (i + 1) % stage_size) for i in range(stage_size)]
      arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
      return jnp.where(stage_idx == 0, jnp.zeros_like(arr), arr)

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
        ) % self.config.num_pipeline_microbatches  # Note extra -1 b/c grabbing from the
        # previous output - using circ_storage_mover before it is updated
        return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)

      new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
      new_circ_storage_mover = output
    else:
      new_circ_storage = None
      new_circ_storage_mover = None

    # Rotate stream_io left/up by 1 on rotating micro/stage index (stream_buf_idx), replacing the last/bottom with the
    # last stage output
    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]

    def _rotate_left(arr, stage_size):
      # we use -1 for left shifting
      perm = [(i, (i - 1) % stage_size) for i in range(stage_size)]
      arr = jax.lax.ppermute(arr, axis_name="stage", perm=perm)
      return arr

    def _shift_left(arr, stage_size, output):
      stage_idx = jax.lax.axis_index("stage")
      arr = _rotate_left(arr, stage_size)
      return jnp.where(stage_idx == stage_size - 1, output, arr)

    @jax.shard_map(
        mesh=self.mesh,
        in_specs=(self.state_io_spec, self.stages_in_spec, self.stages_in_spec, P()),
        out_specs=self.state_io_spec,
    )
    def _update_state_io(state_in, stream_slice, output, stream_buf_idx):
      # Shift the current slice to the left, then fill the last stage with the final output.
      stage_size = jax.lax.axis_size("stage")
      stream_slice = _shift_left(stream_slice, stage_size, output)
      stream_slice = jnp.expand_dims(stream_slice, 1)
      return jax.lax.dynamic_update_slice_in_dim(state_in, stream_slice, stream_buf_idx, axis=1)

    new_state = _update_state_io(old_state_io, stream_slice, output, stream_buf_idx)

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
    # The first real output (microbatch 0) takes a certain amount of loop iterations to finish and be pushed to
    # state_io - it will land on a different index of state_io depending on the number of iterations.
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    permutation = (
        np.arange(self.microbatches_per_stage) + microbatch_0_idx
    ) % self.microbatches_per_stage  # permute so the value in land_idx is moved into idx 0, and (land_idx + 1) appear
    # in idx 1, etc
    output = output[:, permutation]
    return output

  def get_current_stage_weights(self, pipeline_weights, loop_iteration, physical_partition_spec=None):
    """
    Gets the current weights used for one iteration. Outputs a pytree whose arrays have leading dimension of stages, e.g.
    {'mlp': 'wo': [stages, mlp, embed]}. Stage 0 will use the 0th index of this pytree, Stage 1 the 1st index, etc.
    For non-circular pipelines, this simply returns all weights - every weight is used in every iteraiton. However
    for circular pipelines each stage grabs only the weights corresponding to the current repeat.
    """
    if self.config.num_pipeline_repeats > 1:
      return self.get_current_repeat_from_stages(
          pipeline_weights, loop_iteration, physical_partition_spec=physical_partition_spec
      )
    else:
      return pipeline_weights

  def get_current_repeat_from_stages(self, weights, loop_iteration, physical_partition_spec=None):
    """get current repeat from stages"""
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    circular_metadata_params = {
        nn.PARTITION_NAME: "circular_repeats",
        "sub_weight_split_dims_mapping": (None,),
        "is_initializing": self.is_initializing(),
        "x_times": self.config.num_pipeline_repeats,
        "optimizer_dims_mapping": None,
    }
    weights = meta.remove_axis(
        weights, 0, circular_metadata_params
    )  # Remove the circular metadata axis, this axis will be removed when passed to the main vmap, only one circular
    # entry per stage.
    weights = self._remove_logically_partition(weights)

    def gather_weights_for_stages_in(w, spec=None):
      return self.vmap_parallel_gather(
          w,
          repeat_ids=repeat_ids,
          repeat_dim_in_weights=0,
          stages_dim_in_weights=1,
          physical_partition_spec=spec,
      )

    if physical_partition_spec is None:
      weights = jax.tree.map(gather_weights_for_stages_in, weights)
    else:
      weights = jax.tree.map(gather_weights_for_stages_in, weights, physical_partition_spec)
    return weights

  def get_vmap_func_for_init(self):
    """This vmap func is used to initialize the weights only on init."""

    def func_to_vmap(body_instance, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode):
      """nn.vmap requires either a nn.module class or a function whose first argument is a nn.module instance."""
      return body_instance(stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)

    vmap_func = nn.vmap(
        func_to_vmap,
        in_axes=(0, 0, 0, None, None),
        spmd_axis_name=self.spmd_axis_name,  # TODO(b/470167805): replace self.spmd_axis_name with "stage" when JAX >= 0.8.2.
        variable_axes={"params": 0, "_overwrite_with_gradient": 0},
        split_rngs={"params": self.is_initializing(), "dropout": self.config.enable_dropout},
        metadata_params={
            nn.PARTITION_NAME: "layers",
            "sub_weight_split_dims_mapping": (None),
            "is_initializing": self.is_initializing(),
            "x_times": self.num_stages,
        },
    )
    return vmap_func

  def get_main_vmap_func_for_iterations(self):
    """
    Returns main stage function vmapped by number of stages.
    This becomes a vmap over a single layer instance if body_instance is a single layer,
    else a set of layers if body_instance is a set of layers.
    """

    def func_to_vmap(
        body_instance, weights, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode
    ):
      """nn.vmap requires either a nn.module class or a function whose first argument is a nn.module instance."""
      weights = meta.remove_axis(
          weights,
          0,
          {
              nn.PARTITION_NAME: "layers",
              "sub_weight_split_dims_mapping": (None,),
              "is_initializing": self.is_initializing(),
              "x_times": self.num_stages,
          },
      )
      return body_instance.apply(weights, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)

    vmap_func = nn.vmap(
        func_to_vmap,
        in_axes=(0, 0, 0, 0, None, None),
        spmd_axis_name=self.spmd_axis_name,  # TODO(b/470167805): replace self.spmd_axis_name with "stage" when JAX >= 0.8.2.
        variable_axes={"params": 0},
        split_rngs={"params": self.is_initializing(), "dropout": self.config.enable_dropout},
        metadata_params={
            nn.PARTITION_NAME: "layers",
            "sub_weight_split_dims_mapping": (None),
            "is_initializing": self.is_initializing(),
            "x_times": self.num_stages,
        },
    )
    return vmap_func

  def run_one_iteration(
      self,
      loop_state,
      pipeline_weights,
      positions,
      segment_ids,
      deterministic,
      model_mode,
      decoder_layer_instance,
      logical_partition_spec=None,
  ):
    """Run one loop iteration - gets weights and inputs for each stage, run the stages in parallel,
    and update the loop state."""
    state_io = loop_state["state_io"]
    shift = loop_state["shift"]
    circ_storage = loop_state["circ_storage"]
    loop_iteration = loop_state["loop_iteration"]

    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)

    # Convert logical spec to physical spec
    physical_partition_spec = logical_to_mesh(logical_partition_spec, self.mesh, rules=self.config.logical_axis_rules)

    stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
    # We checkpoint stages_inputs since we are grabbing only one slice of the state_io, don't need to save the entire
    # buffer.
    stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, "iteration_input")
    stages_positions = self.vmap_gather(positions, microbatch_ids, 0) if positions is not None else None
    stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0) if segment_ids is not None else None

    vmap_func = self.get_main_vmap_func_for_iterations()

    if self.config.num_pipeline_repeats > 1:
      _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

      def prepare_vars_for_main_vmap(weights, physical_partition_spec=None):

        circular_metadata_params = {
            nn.PARTITION_NAME: "circular_repeats",
            "sub_weight_split_dims_mapping": (None,),
            "is_initializing": self.is_initializing(),
            "x_times": self.config.num_pipeline_repeats,
            "optimizer_dims_mapping": None,
        }
        weights = meta.remove_axis(
            weights, 0, circular_metadata_params
        )  # Remove the circular metadata axis, this axis will be removed when passed to the main vmap, only one
        # circular entry per stage.
        weights = self._remove_logically_partition(weights)

        def gather_weights_for_stages_in(w, spec=None):
          return self.vmap_parallel_gather(
              w, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1, physical_partition_spec=spec
          )

        if physical_partition_spec is None:
          weights = jax.tree.map(gather_weights_for_stages_in, weights)
        else:
          weights = jax.tree.map(gather_weights_for_stages_in, weights, physical_partition_spec)
        return weights

      prepare_vars_for_main_vmap_partial = functools.partial(
          prepare_vars_for_main_vmap, physical_partition_spec=physical_partition_spec
      )
      vmap_func = nn.map_variables(
          vmap_func,
          mapped_collections=["params", "_overwrite_with_gradient", "non_trainable", "summaries", "intermediates"],
          mutable=True,
          trans_in_fn=prepare_vars_for_main_vmap_partial,
      )

    stage_weights = self.get_current_stage_weights(
        pipeline_weights, loop_iteration, physical_partition_spec=physical_partition_spec
    )

    stages_output = vmap_func(
        decoder_layer_instance,
        stage_weights,
        stages_inputs,
        stages_segment_ids,
        stages_positions,
        deterministic,
        model_mode,
    )
    if self.config.scan_layers:
      stages_output = stages_output[0]

    new_state = self.get_new_loop_state(stages_output, loop_state)
    return new_state

  def get_pipeline_remat_policy(self):
    """Returns the pipeline remat policy for this pipeline."""
    # We ensure that the decoder layer inputs are saved, although we leave it to a custom
    # policy if they should be saved to device or offloaded.
    if self.config.remat_policy == "custom":
      return self.remat_policy

    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
    if self.remat_policy is not None:
      remat_policy = jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
    else:
      remat_policy = save_input_policy
    return remat_policy

  def get_weight_sharding(self, *init_args):
    """get weight sharding function for this pipeline."""
    # Returns a partition spec of all weights. Requires passing in arguments to init.
    key = jax.random.PRNGKey(0)
    keys = {"params": key, "dropout": key, "aqt": key}
    weights = self.init(keys, *init_args)

    def get_partition_spec(pytree):
      def _is_leaf(x):
        return isinstance(x, nn.spmd.LogicallyPartitioned)

      def get_partition_spec_leaf(leaf):
        return leaf.get_partition_spec()

      partition_spec_tree = jax.tree.map(get_partition_spec_leaf, pytree, is_leaf=_is_leaf)
      return partition_spec_tree

    partition_spec_with_extra_layer = get_partition_spec(weights)
    logical_partition_spec = {"params": partition_spec_with_extra_layer["params"]["layers"]}
    return logical_partition_spec

  @staticmethod
  def get_logical_spec_repeats_removed(full_logical):
    if full_logical is None:
      return None

    def _remove_from_spec(spec):
      return jax.sharding.PartitionSpec(*[dim for dim in spec if dim != "circular_repeats"])

    return jax.tree.map(_remove_from_spec, full_logical)

  # TODO(chengnuojin) Remove this function and its usage after pipeline nnx migration
  @staticmethod
  def _remove_logically_partition(weights):
    def _remove_logically_partition_leaf(v):
      return getattr(v, "value") if isinstance(v, LogicallyPartitioned) else v

    return jax.tree.map(
        _remove_logically_partition_leaf,
        weights,
        is_leaf=lambda v: isinstance(v, LogicallyPartitioned),
    )

  @staticmethod
  def _remove_fsdp_from_physical_partition_spec(pps):
    """Removes 'fsdp' and 'fsdp_transpose' from a physical PartitionSpec."""
    if isinstance(pps, P):
      new_spec = []
      # Iterate through each axis in the original PartitionSpec.
      for axis in pps:
        if axis is None:
          new_spec.append(None)
        elif isinstance(axis, str):
          # If the axis is 'fsdp', replace it with None to signify replication.
          if axis not in ("fsdp", "fsdp_transpose"):
            new_spec.append(axis)
          else:
            new_spec.append(None)
        elif isinstance(axis, (list, tuple)):
          # If the axis is a collection, filter out 'fsdp'.
          new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
          new_spec.append(tuple(new_axis))
        else:
          raise ValueError(f"Unsupported_axis_type: {type(axis)}")
        # Return a new sharding object with the modified spec.
      return P(*new_spec)
    return pps

  def all_gather_over_fsdp(self, variables, logical_partition_spec):
    physical_partition_spec = logical_to_mesh(
        logical_partition_spec, mesh=self.mesh, rules=self.config.logical_axis_rules
    )
    physical_partition_spec_no_fsdp = jax.tree.map(
        self._remove_fsdp_from_physical_partition_spec, physical_partition_spec
    )
    return jax.tree.map(
        lambda w, p: self._maybe_shard_with_name(w, NamedSharding(self.mesh, p)),
        variables,
        physical_partition_spec_no_fsdp,
    )

  @nn.compact
  def __call__(
      self,
      inputs: jnp.ndarray,
      segment_ids: jnp.ndarray,
      positions: jnp.ndarray,
      deterministic: bool,
      model_mode=MODEL_MODE_TRAIN,
      logical_partition_spec=None,  # Pytree of sharding specifications of the weights (aka self.layers.variables)
  ) -> jnp.ndarray:
    """The main method that maps the series of decoder layer inputs to final layer outputs.
    Has the same signature of a single decoder layer, and expects the same shapes, e.g. the inputs should have shape
    [global_batch], and internally this will be reshapped into microbatches.
    """
    # Reshape inputs of [global_batch, ...] to [microbatches, pipeline_microbatch_sizes, ...]
    inputs = inputs.reshape(
        (
            self.config.num_pipeline_microbatches,
            self.pipeline_microbatch_size,
            self.config.max_target_length,
            self.config.emb_dim,
        ),
        out_sharding=self.input_sharding,
    )

    example_inputs = jax.lax.broadcast(inputs[0], [self.num_stages])  # dummy inputs fed to initialize the module
    ag_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None, None))
    if positions is not None:
      # AG positions
      positions = self._maybe_shard_with_name(positions, ag_sharding)

      positions = positions.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
      example_position = jax.lax.broadcast(positions[0], [self.num_stages])
      position_idx = 0
    else:
      example_position = None
      position_idx = None
    if segment_ids is not None:
      segment_ids = self._maybe_shard_with_name(segment_ids, ag_sharding)
      segment_ids = segment_ids.reshape(
          (self.config.num_pipeline_microbatches, self.pipeline_microbatch_size, self.config.max_target_length)
      )
      example_segmentation = jax.lax.broadcast(segment_ids[0], [self.num_stages])
      segment_idx = 0
    else:
      example_segmentation = None
      segment_idx = None

    loop_state = self.init_states(inputs)

    # Each microbatch should go through each stage (with repeats) - so there is num_micro * (num_stages * repeats)
    # compute to perform
    # Each iteration is vmapped by num_stages, so the number of iterations should be
    # num_micro * num_stages * repeats / num_stages = num_micro * repeats
    # However due to the pipeline bubble some iterations process less than num_stages microbatches. It takes
    # num_micro * repeat iterations for the last microbatch to start the final repeat, then an additional
    # num_stages - 1 to finish the final repeat.
    # Thus the total iterations is num_micro * repeat + num_stages - 1, & we may consider the num_stages - 1 as bubble.
    # The bubble doubles when we use forwarding delay.
    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    if self.is_initializing():
      vmap_func = self.get_vmap_func_for_init()

      if self.config.num_pipeline_repeats > 1:
        # To shard the weights on initialization for the circular pipeline we create weights of
        # shape [num_repeat, num_stages, ...] (e.g. [num_repeat, num_stages, embed, mlp]) and shard the num_stages axis.
        # We wrap the main stage vmap with a num_repeat vmap to generate this axis only for parameter initialization.
        vmap_func = nn.vmap(
            vmap_func,
            in_axes=(0, segment_idx, position_idx, None, None),
            variable_axes={
                "params": 0,
                "_overwrite_with_gradient": 0,
                "non_trainable": 0,
                "hyper_params": 0,
            },
            split_rngs={"params": True, "dropout": self.config.enable_dropout},
            metadata_params={
                nn.PARTITION_NAME: "circular_repeats",
                "sub_weight_split_dims_mapping": (None,),
                "is_initializing": True,
                "x_times": self.config.num_pipeline_repeats,
                "optimizer_dims_mapping": None,
            },
        )

        example_inputs = jax.lax.broadcast(example_inputs, [self.config.num_pipeline_repeats])
        example_segmentation = (
            jax.lax.broadcast(example_segmentation, [self.config.num_pipeline_repeats])
            if example_segmentation is not None
            else None
        )
        example_position = (
            jax.lax.broadcast(example_position, [self.config.num_pipeline_repeats])
            if example_position is not None
            else None
        )
      # We only need to run one set of stages to initialize the variables, instead of looping over all microbatches for
      # the full total_iterations.
      example_inputs = self._maybe_shard_with_logical(example_inputs, (None, None, None, None))
      stage_outputs = vmap_func(
          self.layers, example_inputs, example_segmentation, example_position, deterministic, model_mode
      )
      if self.config.scan_layers:
        stage_outputs = stage_outputs[0]

      # We return something of the correct shape (global_batch, sequence, embed) by reshaping a single stages output
      # which has shape [pipeline_microbatch_size, sequence, embed]
      if self.config.num_pipeline_repeats > 1:
        stage_outputs = stage_outputs[0]  # Remove extra dimension created for the circular vmap
      broadcasted_stage_outpus = jax.lax.broadcast(
          stage_outputs[0], [self.config.micro_batch_size_to_train_on // self.pipeline_microbatch_size]
      )

      return jnp.reshape(
          broadcasted_stage_outpus,
          [self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim],
          out_sharding=self.output_sharding,
      )

    if self.config.pipeline_fsdp_ag_once:
      variables = self._remove_logically_partition(self.layers.variables)
      all_pipeline_weights = self.all_gather_over_fsdp(variables, logical_partition_spec)
    else:
      all_pipeline_weights = self.layers.variables

    logical_partition_spec = self.get_logical_spec_repeats_removed(logical_partition_spec)

    def run_iteration_scannable(model, loop_state, xs):
      # flax transforms like nn.scan and nn.remat can only be applied to nn.module classes or nn.module instances, so we
      # explicitly wrap the run_one_iteration in this method - the 1st argument model (`self`) is a nn.module instance.
      return (
          model.run_one_iteration(
              loop_state,
              all_pipeline_weights,
              positions,
              segment_ids,
              deterministic,
              model_mode,
              model.layers,
              logical_partition_spec=logical_partition_spec,
          ),
          None,
      )

    if self.config.set_remat_policy_on_pipeline_iterations:
      run_iteration_scannable = nn.remat(
          run_iteration_scannable,
          prevent_cse=not self.config.scan_pipeline_iterations,  # prevent_cse not used with scan
          policy=self.get_pipeline_remat_policy(),
      )

    # The scan cannot be used on init since it broadcasts the weights, which aren't yet initialized.
    if self.config.scan_pipeline_iterations:
      variable_carry = []
      variable_broadcast = [
          "params",
          "_overwrite_with_gradient",
      ]  # All loop iterations need the weights for the full pipeline.
      if self.is_mutable_collection("non_trainable"):
        variable_carry.append("non_trainable")
      else:
        variable_broadcast.append("non_trainable")
      run_all_iterations_scanned = nn.scan(
          run_iteration_scannable,
          variable_axes={
              "summaries": 0,
              "aux_loss": 0,
              "intermediates": 0,
              "hyper_params": 0,
          },
          variable_broadcast=variable_broadcast,
          variable_carry=variable_carry,
          # Dropout/aqt keys will be split for each iteration.
          split_rngs={"random": True},
          length=total_iterations,
      )
      loop_state, _ = run_all_iterations_scanned(self, loop_state, None)
    else:
      for _ in range(total_iterations):
        loop_state, _ = run_iteration_scannable(self, loop_state, None)

    # The final output is located in the input/output array, however the output microbatches may be permuted relative to
    # the input
    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])

    # reshape outputs to match input shape of total batch instead of microbatches [batch, sequence, embed]
    final_output = jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
        out_sharding=self.output_sharding,
    )

    return final_output
