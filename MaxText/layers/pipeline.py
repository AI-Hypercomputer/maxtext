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

''' Pipeline layer wrapping a decoder layer(s). Supports circular pipelining '''

import jax
import jax.ad_checkpoint
import numpy as np
from jax import numpy as jnp
from flax.core import meta
from flax import linen as nn
import common_types
import functools
from typing import Any

class Pipeline(nn.Module):
  """Module that implements pipelining across stages.
  
  This module will loop over microbatches and execute the main body with a vmap for both the inputs and weights.
  This will produce a pipeline pattern if the stage dimension is sharded.

  Supports circular pipelines, and multiple layers per stage are used when a module that executes multiple layers
  is passed as the layers input.

  Attributes:
    config: Importantly contains num_pipeline_microbatches, num_pipeline_repeats.
    layers: A module instance that each stage can execute. It can either be a single layer such as a LlamaDecoderLayer instance
      or scanned/looped set of decoder layers to execute multiple layers per stage.
    mesh:  The device mesh of the system.
    remat_policy: Remat policy to use for the loop iterations
  """
  
  config: common_types.Config
  layers: nn.Module # The name of this property (layers) is reflected in the state pytree and thus also checkpoints.
  mesh: common_types.Mesh
  remat_policy: Any = None

  def setup(self):
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.use_circ_storage = self.config.num_pipeline_repeats > 1 and self.config.num_pipeline_microbatches > self.num_stages
    self.microbatch_size = self.config.global_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.microbatches_per_stage = microbatches_per_stage

  def init_states(self, inputs):
    '''Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
        Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

        Returns a dictionary with properties
          shift: zeros shape [num_stages, micro_size, sequence, embed]
          state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
          circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed]
          circ_storage_mover: zeros[num_stages, micro_size, sequence, embed]
          loop_iteration: scalar set initially to 0.  
    '''

    # Shift is used to rotate the output of each pipeline into the input of the next
    # shift has shape [num_stages, micro_size, sequence, embed]
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = nn.with_logical_constraint(shift, ("activation_stage", "activation_batch", "activation_length", "activation_embed"),rules=self.config.logical_axis_rules,mesh=self.mesh)

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs as the pipeline runs/finishes
    # state_io has shape [num_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
    # We shard the microbatch_size axis by data/fsdp, not num_microbatches since those are looped over.
    state_io = nn.with_logical_constraint(state_io, ("activation_stage", None, "activation_batch", "activation_length", "activation_embed"),rules=self.config.logical_axis_rules, mesh=self.mesh) 

    # circ_storage is used to hold the final pipeline stage outputs before it is used for the next repeat. It is only needed
    # when num_microbatches > num_stages, else instead the final stage will immediately pass to the first without additional storage.
    # circ_storage has shape [num_stages, microbatches, micro_size, sequence, embed].
    # Note that this shape is a factor of num_stages larger than necessary - each stage holds the global batch, but only stage 0 holds the
    # real activations (since it will use them), the rest hold dummy ones. This amount of storage [global_batch, sequence, embed] is
    # fine as long as there is some amount of additional sharding axes, e.g. FSDP, TP, DP (e.g. there are many devices that shard stage 0)
    # We may look into alternatives using less storage if this becomes an issue (ideas in b/347603101).
    if self.use_circ_storage:
        circ_storage = jnp.zeros((self.num_stages,) + inputs.shape , dtype=inputs.dtype)
    else:
       circ_storage = None

    # circ_storage_mover is used to push the microbatches from the pipeline into circ_storage with one buffer iteration of delay
    # circ_storage_mover shape is same as shift: [num_stages, micro_size, sequence, embed]
    if self.use_circ_storage:
        circ_storage_mover = shift
    else:
       circ_storage_mover = None

    init_loop_state = {
      "state_io": state_io,
      "shift": shift,
      "circ_storage": circ_storage,
      "circ_storage_mover": circ_storage_mover,
      "loop_iteration": 0
    }
    return init_loop_state

  def get_iteration_inputs(self, loop_iteration, state_io, circ_storage, shift):
    '''
    Construct stages_in: the global array that is operated on for this iteration, shape same as shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab a new batch from state_io or an old one from circ_storage
    '''

    # Setup potential input from state_io, which has a rotating microbatch index (size of microbatches_per_stage)
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:,state_io_batch_idx]

    if self.use_circ_storage:
        # Setup potential input from circ_storage, which also has a rotating index for microbatch, size of num_microbatches
        circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
        circular_stage_in = circ_storage[:,circ_storage_batch_idx]
    else:
        # The last stage immediately flows into the first stage, use this rotated shift instead of circular storage
        circular_stage_in = shift
 
    # For early loop iterations we grab a new input for stage 0 from the state_io. Once each microbatch has left state_io
    # we instead grab from the last stage's output (possibly buffered when num_microbatches > num_stages, e.g. from circ_storage).
    first_stage_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circular_stage_in)

    # Note that first_stage_in may correspond to bubble computation during the last few iterations.
    # However these bubble computation results remain in the shift buffer (do not make it back to state_io) and are thus discarded / not returned.
    # The final returned output is stored in the state_io, which has the appropriate total size of num_microbatches. The state_io will not contain bubble results
    # at the end of the last iteration.
    

    def select_state_or_input(first_stage_in, shift):
        # Selects input for stage 0, shift for other stages
        return jnp.where(jax.lax.broadcasted_iota('int32', shift.shape, 0) == 0, first_stage_in, shift)

    # Selects input (from stream_io) for stage 0, other stages get from shift (the rotated previous output)
    stages_in = select_state_or_input(first_stage_in, shift)
    stages_in = nn.with_logical_constraint(stages_in, ("activation_stage", "activation_batch", "activation_length", "activation_embed"), rules=self.config.logical_axis_rules, mesh=self.mesh)
    return stages_in

  def shard_dim_by_stages(self, x, dim: int):
    # Shards a dimension by stages. Currently the sharding of other dimensions are left up the compiler, alternatively
    # we may want to copy over the sharding from the other input axes.
    dims_mapping = [jax.sharding.PartitionSpec.UNCONSTRAINED] * x.ndim
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(*dims_mapping))
    return jax.lax.with_sharding_constraint(x, sharding)

  def get_microbatch_and_repeat_ids(self, loop_iteration):
    '''Gets the microbatch_ids and repeat_ids for all stages on this loop_iteration. Works for both circular and non-circular'''
    # Stage 0 has processed one microbatch every loop_iter, but Stage 1 is one behind due to bubble, etc for other stages
    microbatches_processed = jnp.maximum(loop_iteration - jnp.arange(self.num_stages), 0) 
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
    stage_weights = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=gathered_weights_stage_dim)(weights, repeat_ids)
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
      return jnp.squeeze(
          jax.lax.dynamic_slice_in_dim(x, i, 1, ids_dim), ids_dim)

    ids = self.shard_dim_by_stages(ids, 0)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
    return self.shard_dim_by_stages(outs, 0)

  def get_new_loop_state(self,output, loop_state):
    '''
      Update the various buffers given the output of the most recent iteration
      * state_io: rotates left/up by 1 (the whole created in the last slot is filled with the most recent pipeline output)
         * Pushing inputs up from top of state_io into first stage of shift
         * Pulling outputs up from last stage of shift into bottom of state_io
      * shift: rotate output right/down by 1 - we imagine the pipeline moves to right/down
      * circ_storage: pushes circ_storage_mover (the output of the previous iteration) into rotating index of circ_storage
      * circ_storage_mover: assigned to rotated output and pushed into circ_storage on the next iteration
    '''

    old_state_io = loop_state['state_io']
    old_circ_storage = loop_state["circ_storage"]
    old_circ_storage_mover = loop_state["circ_storage_mover"]
    loop_iteration = loop_state["loop_iteration"]
    # Shift becomes a rotated-right version of the previous output
    def _rotate_right(output_in):
      # Use lax.slice to avoid generating a gather.
      last = jax.lax.slice_in_dim(output_in, self.num_stages - 1, self.num_stages, axis=0)
      except_last = jax.lax.slice_in_dim(output_in, 0, self.num_stages - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)
    new_shift = _rotate_right(output)

    if self.use_circ_storage:
      # Insert the circ_storage_mover into new_circ_storage at a microbatch-rotating index.
      # circ_storage_mover still points to the output of PREVIOUS iteration, which should aid in allowing overlapped compute/async transfers
      def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
          rotated = _rotate_right(circ_storage_mover_in)
          rotated = jnp.expand_dims(rotated, 1)
          # The offset is the previous iterations microbatch ID of the last stage, so that for example microbatch 0 will
          # be placed in index 0 of the num_microbatches axis. 
          offset = (loop_iteration - (self.num_stages - 1) - 1) % self.config.num_pipeline_microbatches # Note extra -1 b/c grabbing from the previous output - using circ_storage_mover before it is updated
          return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)
      new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
      new_circ_storage_mover = output
    else:
       new_circ_storage = None
       new_circ_storage_mover = None

    # Rotate stream_io left/up by 1 on rotating micro/stage index (stream_buf_idx), replacing the last/bottom with the last stage output
    stream_buf_idx = loop_iteration % self.microbatches_per_stage
    stream_slice = old_state_io[:, stream_buf_idx]
    def _update_state_io(state_in, stream_slice, output):
        # Shift the current slice to the left, then fill the last stage with the final output.
        padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
        stream_slice = jax.lax.slice_in_dim(
            jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
        stream_slice = jnp.where(
            jax.lax.broadcasted_iota('int32', stream_slice.shape, 0) == self.num_stages - 1, output,
            stream_slice)
        stream_slice = jnp.expand_dims(stream_slice, 1)
        return jax.lax.dynamic_update_slice_in_dim(
            state_in, stream_slice, stream_buf_idx, axis=1)
    new_state = _update_state_io(old_state_io, stream_slice, output)
    
    new_loop_state = {
      "state_io": new_state,
      "shift": new_shift,
      "circ_storage": new_circ_storage,
      "circ_storage_mover": new_circ_storage_mover,
      "loop_iteration": loop_iteration + 1
    }
    return new_loop_state
   
  def permute_output_micro_per_stage_dim(self, output):
    # The first real output (batch 0) takes a certain amount of loop iterations to finish and be pushed to state_io - it will land on a different index of state_io depending on the number of iterations.
    first_output_num_iters = self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1) + self.num_stages - 1
    # The first term above is a multiple of num_pipeline_microbatches and thus could be ignored since its also a multiple of microbatches_per_stage, but we keep it for clairty
    land_idx = first_output_num_iters % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + land_idx) % self.microbatches_per_stage # permute so the value in land_idx is moved into idx 0, and (land_idx + 1) appear in idx 1, etc
    output = output[:,permutation]
    return output

  def get_main_vmap_func(self):
    def func_to_vmap(body_instance, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode):
      # nn.vmap requires either a nn.module class or a function whose first argument is a nn.module instance.
      return body_instance(stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)

    vmap_func = nn.vmap(
      func_to_vmap,
      in_axes=(0, 0, 0, None, None),
      spmd_axis_name='stage',
      variable_axes={'params': 0},
      split_rngs={'params':  self.is_initializing()},
      metadata_params={
        nn.PARTITION_NAME: "layers",
        'sub_weight_split_dims_mapping': (None),
        "is_initializing": self.is_initializing(),
        "x_times": self.num_stages}
    )
    return vmap_func

  def run_one_iteration(self, loop_state, positions, segment_ids, deterministic, model_mode, decoder_layer_instance):
   '''Run one loop iteration - gets weights and inputs for each stage, run the stages in parallel, and update the loop state.'''
   state_io = loop_state['state_io']
   shift = loop_state["shift"]
   circ_storage = loop_state["circ_storage"]
   loop_iteration = loop_state["loop_iteration"]

   microbatch_ids, _ = self.get_microbatch_and_repeat_ids(loop_iteration)

   stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
   # We checkpoint stages_inputs since we are grabbing only one slice of the state_io, don't need to save the entire buffer.
   stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, 'iteration_input')
   stages_positions = self.vmap_gather(positions, microbatch_ids, 0) if positions is not None else None
   stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0) if segment_ids is not None else None

   vmap_func = self.get_main_vmap_func()

   if self.config.num_pipeline_repeats > 1:
    _, repeat_ids = self.get_microbatch_and_repeat_ids(loop_iteration)

    def prepare_vars_for_main_vmap(weights):
      def gather_weights_for_stages_in(weights):
        return jax.tree.map(
            functools.partial(
                self.vmap_parallel_gather, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1),
            weights)
      circular_metadata_params={
        nn.PARTITION_NAME: "circular_repeats",
        'sub_weight_split_dims_mapping': (None,),
        "is_initializing": self.is_initializing(),
        "x_times": self.config.num_pipeline_repeats,
        'optimizer_dims_mapping': None,
      }
      weights = meta.remove_axis(weights, 0, circular_metadata_params) # Remove the circular metadata axis, this axis will be removed when passed to the main vmap, only one circular entry per stage.
      weights = gather_weights_for_stages_in(weights)
      return weights

    vmap_func = nn.map_variables(
        vmap_func,
        mapped_collections=["params", "non_trainable", "summaries", "intermediates"],
        mutable=True,
        trans_in_fn=prepare_vars_for_main_vmap,
    )

   stages_output = vmap_func(decoder_layer_instance, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)
   if self.config.scan_layers:
     stages_output = stages_output[0]

   new_state = self.get_new_loop_state(stages_output, loop_state)
   return new_state

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, segment_ids: jnp.ndarray, positions:jnp.ndarray, deterministic: bool, model_mode=common_types.MODEL_MODE_TRAIN) -> jnp.ndarray:
    ''' The main method that maps the series of decoder layer inputs to final layer outputs.
    Has the same signature of a single decoder layer, and expects the same shapes, e.g. the inputs should have shape [global_batch], and internally
    this will be reshapped into microbatches.
    '''
    # Reshape inputs of [global_batch, ...] to [microbatches, microbatch_sizes, ...]
    inputs = inputs.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length, self.config.emb_dim))
    example_inputs = jax.lax.broadcast(inputs[0], [self.num_stages]) # dummy inputs fed to initialize the module weights.
    if positions is not None:
      positions = positions.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
      example_position = jax.lax.broadcast(positions[0], [self.num_stages])
      position_idx = 0
    else:
      example_position = None
      position_idx = None
    if segment_ids is not None:
      segment_ids = segment_ids.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
      example_segmentation = jax.lax.broadcast(segment_ids[0], [self.num_stages])
      segment_idx = 0
    else:
      example_segmentation = None
      segment_idx = None

    loop_state = self.init_states(inputs)
    
    # Each microbatch should go through each stage (with repeats) - so there is num_micro * (num_stages * repeats) compute to perform
    # Each iteration is vmapped by num_stages, so the number of iterations should be num_micro * num_stages * repeats / num_stages = num_micro * repeats
    # However due to the pipeline bubble some iterations process less than num_stages microbatches. It takes
    # num_micro * repeat iterations for the last microbatch to start the final repeat, then an additional num_stages - 1 to finish the final repeat.
    # Thus the total iterations is num_micro * repeat + num_stages - 1, and we may consider the num_stages - 1 as bubble. 
    total_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats + self.num_stages  - 1

    if self.is_initializing():     
     vmap_func = self.get_main_vmap_func()

     if self.config.num_pipeline_repeats > 1:
       # To shard the weights on initialization for the circular pipeline we create weights of
       # shape [num_repeat, num_stages, ...] (e.g. [num_repeat, num_stages, embed, mlp]) and shard the num_stages axis.
       # We wrap the main stage vmap with a num_repeat vmap to generate this axis only for parameter initialization.
       vmap_func= nn.vmap(
         vmap_func,
         in_axes=(0, segment_idx, position_idx, None, None),
          variable_axes={
            'params': 0,
            "non_trainable": 0,
            "hyper_params": 0,
          },
          split_rngs={'params': True},
          metadata_params={
            nn.PARTITION_NAME: "circular_repeats",
            'sub_weight_split_dims_mapping': (None,), 
            "is_initializing": True,
            "x_times": self.config.num_pipeline_repeats,
            'optimizer_dims_mapping': None,
          }
        )

       example_inputs = jax.lax.broadcast(example_inputs, [self.config.num_pipeline_repeats])
       example_segmentation = jax.lax.broadcast(example_segmentation, [self.config.num_pipeline_repeats]) if example_segmentation is not None else None
       example_position = jax.lax.broadcast(example_position, [self.config.num_pipeline_repeats]) if example_position is not None else None
     # We only need to run one set of stages to initialize the variables, instead of looping over all microbatches for the full total_iterations.
     stage_outputs = vmap_func(self.layers, example_inputs, example_segmentation, example_position, deterministic, model_mode)
     if self.config.scan_layers:
       stage_outputs = stage_outputs[0]

     # We return something of the correct shape (global_batch, sequence, embed) by reshaping a single stages output which has
     # shape [microbatch_size, sequence, embed]
     if self.config.num_pipeline_repeats > 1:
       stage_outputs = stage_outputs[0] # Remove extra dimension created for the circular vmap
     broadcasted_stage_outpus = jax.lax.broadcast(stage_outputs[0], [self.config.global_batch_size_to_train_on // self.microbatch_size])
     return jnp.reshape(broadcasted_stage_outpus, [self.config.global_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim])

    def run_iteration_scannable(model,loop_state, xs):
       # flax transforms like nn.scan and nn.remat can only be applied to nn.module classes or nn.module instances, so we explicitly wrap
       # the run_one_iteration in this method - the first argument model (i.e. self) is a nn.module instance.
       return model.run_one_iteration(loop_state, positions, segment_ids, deterministic, model_mode, model.layers), None
    if self.remat_policy is not None:
      remat_policy = jax.checkpoint_policies.save_from_both_policies(
          self.remat_policy,
          jax.checkpoint_policies.save_only_these_names('iteration_input')
      )
    else:
      remat_policy = jax.checkpoint_policies.save_only_these_names('iteration_input')
    run_one_iteration_rematted = nn.remat(
      run_iteration_scannable,
      prevent_cse=not self.config.scan_pipeline_iterations, # prevent_cse not used with scan
      policy=remat_policy
    )

    # The scan cannot be used on init since it broadcasts the weights, which aren't yet initialized.
    if self.config.scan_pipeline_iterations:
      variable_carry = []
      variable_broadcast = ["params"] # All loop iterations need the weights for the full pipeline.  
      if self.is_mutable_collection("non_trainable"):
        variable_carry.append("non_trainable")
      else:
        variable_broadcast.append("non_trainable")
      run_all_iterations_scanned = nn.scan(
        run_one_iteration_rematted,
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
        for loop_iteration in range(total_iterations):
            loop_state, _ = run_one_iteration_rematted(self, loop_state, None)

    # The final output is located in the input/output array, however the output microbatches may be permuted relative to the input
    final_output = self.permute_output_micro_per_stage_dim(loop_state["state_io"])

    # reshape outputs to match input shape of total batch instead of microbatches [batch, sequence, embed]
    final_output = jnp.reshape(final_output, (self.config.global_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim))
                               
    return final_output