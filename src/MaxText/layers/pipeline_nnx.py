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

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx


def with_logical_constraint(x, logical_axis_names, mesh, axis_rules):
  """Replacement for nn.with_logical_constraint in pure JAX/NNX."""
  if mesh is None or axis_rules is None:
    return x

  # Convert logical names (e.g., 'activation_batch') to mesh axes (e.g. 'data')
  # This logic mimics flax.linen.partitioning.logical_to_mesh_sharding
  # Simplified version for the migration:
  partition_spec = []
  for axis_name in logical_axis_names:
    if axis_name is None:
      partition_spec.append(None)
      continue

    # Find the rule mapping logical -> mesh
    mapped = None
    for logical, physical in axis_rules:
      if logical == axis_name:
        mapped = physical
        break
    partition_spec.append(mapped)

  sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*partition_spec))
  return jax.lax.with_sharding_constraint(x, sharding)


class Pipeline(nnx.Module):
  """Module that implements pipelining across stages.

  Attributes:
    config: The model configuration.
    mesh: The device mesh for sharding.
    remat_policy: Optional remat policy for the loop iterations.
  """

  def __init__(self, config, layer_cls, mesh, rngs, remat_policy=None):
    self.config = config
    self.mesh = mesh
    self.remat_policy = remat_policy

    # 1. Calculate Stages (Logic moved from setup)
    self.num_stages = config.ici_pipeline_parallelism * config.dcn_pipeline_parallelism
    self.forwarding_delay = 2 if config.pipeline_delay_activation_forwarding else 1
    self.pipeline_microbatch_size = config.micro_batch_size_to_train_on // config.num_pipeline_microbatches
    self.microbatches_per_stage = config.num_pipeline_microbatches // self.num_stages

    # 2. Setup Axis Names (Logic moved from setup)
    if config.expert_shard_attention_option == "EP_AS_CONTEXT":
      self.batch_axis_name = "activation_batch_no_exp"
      self.seq_len_axis_name = "activation_length"
    else:
      self.batch_axis_name = "activation_batch"
      self.seq_len_axis_name = "activation_length_no_exp"

    # 3. Determine if circular storage is needed
    self.use_circ_storage = self.need_circ_storage()

    # 4. Create the Stack of Layers (The NNX change)
    # We vmap the CONSTRUCTOR of the layer.
    # This creates a module where all params have a leading 'stage' dimension.
    self.layers = nnx.vmap(
        layer_cls,
        in_axes=None,  # Don't vmap over config
        out_axes=0,  # Stack resulting layers on axis 0
        axis_name="stage",
    )(config, rngs=rngs)

  def get_pipeline_remat_policy(self):
    """Returns the pipeline remat policy for this pipeline.

    We ensure that the decoder layer inputs are saved, although we leave it to a custom
    policy if they should be saved to device or offloaded.
    """
    if self.config.remat_policy == "custom":
      return self.remat_policy

    save_input_policy = jax.checkpoint_policies.save_only_these_names("iteration_input", "decoder_layer_input")
    if self.remat_policy is not None:
      remat_policy = jax.checkpoint_policies.save_from_both_policies(self.remat_policy, save_input_policy)
    else:
      remat_policy = save_input_policy
    return remat_policy

  def need_circ_storage(self):
    """Check if circular storage is needed for this pipeline configuration."""
    return (
        self.config.num_pipeline_repeats > 1
        and self.config.num_pipeline_microbatches > self.num_stages * self.forwarding_delay
    )

  def iterations_to_complete_first_microbatch_one_repeat(self):
    """Return the number of iterations it takes for microbatch 0 to finish a repeat."""
    return self.forwarding_delay * (self.num_stages - 1)

  def iterations_to_complete_first_microbatch(self):
    """Return the number of iterations it takes for microbatch 0 to finish the last stage of the last repeat."""
    return (
        self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1)
        + self.iterations_to_complete_first_microbatch_one_repeat()
    )

  def init_states(self, inputs):
    """Initialize components of state: state_io, shift, circular_storage and circular_storage_mover.
    Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

    Returns a dictionary with properties:
      shift: zeros shape [num_stages, micro_size, sequence, embed]
      prev_outputs: same shape as shift, only used when pipeline_delay_activation_forwarding is set, else None
      state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
      circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed] when needed, else None
      circ_storage_mover: zeros [num_stages, micro_size, sequence, embed] when needed, else None
      loop_iteration: scalar set initially to 0.
    """
    # Shift buffer: [num_stages, micro_size, sequence, embed]
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = with_logical_constraint(
        shift,
        ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
        self.mesh,
        self.config.logical_axis_rules,
    )

    # Prev outputs (same shape as shift) - only used when pipeline_delay_activation_forwarding is set
    if self.config.pipeline_delay_activation_forwarding:
      prev_outputs = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
      prev_outputs = with_logical_constraint(
          prev_outputs,
          ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
          self.mesh,
          self.config.logical_axis_rules,
      )
    else:
      prev_outputs = None

    # State IO: [num_stages, microbatches_per_stage, micro_size, sequence, embed]
    state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
    state_io = with_logical_constraint(
        state_io,
        ("activation_stage", None, self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
        self.mesh,
        self.config.logical_axis_rules,
    )

    # Circular storage: [num_stages, num_microbatches, micro_size, sequence, embed]
    # Used to hold the final pipeline stage outputs before it is used for the next repeat.
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

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    """Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration.
    Works for both circular and non-circular pipelines."""
    # Stage 0 has processed one microbatch every loop_iter, but Stage 1 is 1 behind due to bubble, etc
    microbatches_processed = jnp.maximum(loop_iteration - self.forwarding_delay * jnp.arange(self.num_stages), 0)
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    repeat_ids = microbatches_processed // self.config.num_pipeline_microbatches
    return microbatch_ids, repeat_ids

  def shard_dim_by_stages(self, x, dim: int):
    """Shards a dimension by stages.
    Currently, the sharding of other dimensions are left up to the compiler."""
    dims_mapping = [jax.sharding.PartitionSpec.UNCONSTRAINED] * x.ndim
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(*dims_mapping))
    return jax.lax.with_sharding_constraint(x, sharding)

  def vmap_parallel_gather(self, weights, repeat_ids, repeat_dim_in_weights, stages_dim_in_weights):
    """Use vmap to implement a sharded parallel gather.
    Parallel gather means each stage has its own weights, and gets one slice from it.

    Args:
      weights: Per-stage data to be gathered from.
      repeat_ids: Integer tensor of shape [num_stages], the repeats of the stages.
      repeat_dim_in_weights: The dimension in weights where repeat_ids are applied.
        The output will not have this dimension.
      stages_dim_in_weights: The dimension in weights that represents parallel stages.
    Returns:
      The per-stage gathered values. The shape is weights.shape but with
        repeat_dim_in_weights removed.
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

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    """Construct stages_in: the global array that is operated on for this iteration.
    Shape same as shift=[stages, micro_size, sequence, embed].

    This is almost a rotated version of the last outputs, except for the first stage
    which must grab a new batch from state_io or an old one from circ_storage.
    """
    # Setup potential input from state_io, which has a rotating microbatch index
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:, state_io_batch_idx]

    if self.use_circ_storage:
      # Setup potential input from circ_storage, which also has a rotating index
      circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
      circular_stage_in = circ_storage[:, circ_storage_batch_idx]
    else:
      # The last stage immediately flows into the first stage
      circular_stage_in = shift

    # For early iterations we grab new input from state_io. Once each microbatch has left
    # state_io we instead grab from the last stage's output (from circ_storage if buffered).
    first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)

    # Note that first_stage_in may correspond to bubble computation during the last few iterations.
    # However, these bubble computation results remain in the shift buffer and are discarded.

    def select_state_or_input(first_stage_in, shift):
      # Selects input for stage 0, shift for other stages
      return jnp.where(jax.lax.broadcasted_iota("int32", shift.shape, 0) == 0, first_stage_in, shift)

    # Stage 0 gets first_stage_in, Stage 1..N get shift (the rotated previous output)
    stages_in = select_state_or_input(first_stage_in, shift)
    stages_in = with_logical_constraint(
        stages_in,
        ("activation_stage", self.batch_axis_name, self.seq_len_axis_name, "activation_embed"),
        self.mesh,
        self.config.logical_axis_rules,
    )
    return stages_in

  def get_new_loop_state(self, output, loop_state):
    """Update the various buffers given the output of the most recent iteration.

    * state_io: rotates left/up by 1 (fills last slot with recent pipeline output)
    * shift: rotate output (or prev_outputs if using delay) right/down by 1
    * circ_storage: pushes circ_storage_mover into rotating index
    * circ_storage_mover: assigned to rotated output for next iteration
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

    # Update circular storage
    if self.use_circ_storage:

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

    # Rotate stream_io left/up by 1 on rotating micro/stage index, replacing last/bottom with last stage output
    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]

    def _update_state_io(state_in, stream_slice, output):
      # Shift the current slice to the left, then fill the last stage with the final output.
      padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
      stream_slice = jax.lax.slice_in_dim(jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
      stream_slice = jnp.where(
          jax.lax.broadcasted_iota("int32", stream_slice.shape, 0) == self.num_stages - 1,
          output,
          stream_slice,
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
    """Permute output to match original input order.

    The first real output (microbatch 0) takes a certain amount of loop iterations to finish
    and be pushed to state_io - it will land on a different index of state_io depending on
    the number of iterations.
    """
    microbatch_0_idx = self.iterations_to_complete_first_microbatch() % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + microbatch_0_idx) % self.microbatches_per_stage
    # Permute so the value in land_idx is moved into idx 0, and (land_idx + 1) appears in idx 1, etc
    output = output[:, permutation]
    return output

  def run_one_iteration(self, loop_state, positions=None, segment_ids=None, deterministic=True, model_mode="train"):
    """Run one loop iteration - gets inputs for each stage, runs stages in parallel, updates loop state."""
    loop_iteration = loop_state["loop_iteration"]

    # 1. Prepare Inputs
    stages_inputs = self.get_iteration_inputs(
        loop_iteration,
        loop_state["state_io"],
        loop_state["circ_storage"],
        loop_state["shift"],
    )

    # 2. Gather positions and segment_ids for current microbatches
    microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)

    if positions is not None:
      stages_positions = self.vmap_gather(positions, microbatch_ids, 0)
    else:
      stages_positions = None

    if segment_ids is not None:
      stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0)
    else:
      stages_segment_ids = None

    # 3. Run the Layers (Vmapped Call)
    # Since self.layers was created with nnx.vmap, we just call it!
    # It expects inputs to have [stage, ...] shape.
    stages_output = self.layers(
        stages_inputs,
        stages_positions,
        stages_segment_ids,
        deterministic,
        model_mode,
    )

    # 4. Rotate Buffers
    return self.get_new_loop_state(stages_output, loop_state)

  def __call__(self, inputs, segment_ids=None, positions=None, deterministic=True, model_mode="train"):
    """The main method that maps decoder layer inputs to final layer outputs.

    Has the same signature of a single decoder layer, and expects the same shapes, e.g. the inputs
    should have shape [global_batch], and internally this will be reshaped into microbatches.
    """
    # 1. Reshape inputs to microbatches [num_micro, micro_size, ...]
    inputs = inputs.reshape(
        self.config.num_pipeline_microbatches,
        self.pipeline_microbatch_size,
        self.config.max_target_length,
        self.config.emb_dim,
    )

    # Reshape positions and segment_ids similarly [num_micro, micro_size, sequence]
    if positions is not None:
      positions = positions.reshape(
          self.config.num_pipeline_microbatches,
          self.pipeline_microbatch_size,
          self.config.max_target_length,
      )
    if segment_ids is not None:
      segment_ids = segment_ids.reshape(
          self.config.num_pipeline_microbatches,
          self.pipeline_microbatch_size,
          self.config.max_target_length,
      )

    loop_state = self.init_states(inputs)

    # Calculate total iterations (including bubble and repeats)
    # Each microbatch should go through each stage (with repeats) - so there is
    # num_micro * (num_stages * repeats) compute to perform.
    # Due to the pipeline bubble some iterations process less than num_stages microbatches.
    bubble_iterations = self.forwarding_delay * (self.num_stages - 1)
    real_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats
    total_iterations = real_iterations + bubble_iterations

    # 2. Define the Scan Function
    # args: (module, carry, input_element)
    def scan_fn(module, loop_state, _):
      new_state = module.run_one_iteration(loop_state, positions, segment_ids, deterministic, model_mode)
      return new_state, jnp.zeros(())

    # 3. Execute Scan
    # nnx.scan automatically handles the mutable variables in 'module' (self)
    final_state, _ = nnx.scan(scan_fn, length=total_iterations, in_axes=(None, nnx.Carry, None), out_axes=(nnx.Carry, 0))(
        self, loop_state, None
    )

    # 4. Permute and format output
    # The final output is located in the input/output array, however the output microbatches
    # may be permuted relative to the input
    final_output = self.permute_output_micro_per_stage_dim(final_state["state_io"])

    # Reshape outputs to match input shape of total batch instead of microbatches [batch, sequence, embed]
    final_output = jnp.reshape(
        final_output,
        (self.config.micro_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim),
    )

    return final_output
