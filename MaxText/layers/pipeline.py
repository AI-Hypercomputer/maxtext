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

''' Pipeline layer wrapping a decoder layer(s). Does not yet supports circular pipelining '''

import jax
import jax.ad_checkpoint
import numpy as np
from jax import numpy as jnp
from jax import tree_map
from flax.core import meta
from flax import linen as nn
import common_types
import functools
from typing import Any

class Pipeline(nn.Module):
  """Module that implements pipelining across stages.
  
  This module will loop over microbatches and execute the main body with a vmap for both the inputs and weights.
  This will produce a pipeline pattern if the stage dimension is sharded.

  Does not yet support circular pipelines. Multiple
  layers per stage are used when a module that executes multiple layers per stage is passed as the layers input.

  Attributes:
    config: Importantly contains num_pipeline_microbatches.
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
    self.microbatch_size = self.config.global_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    self.microbatches_per_stage = microbatches_per_stage

  def init_states(self, inputs):
    '''Initialize components of state: state_io, shift
        Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

        Returns a dictionary with properties
          shift: zeros shape [num_stages, micro_size, sequence, embed]
          state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
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

    init_loop_state = {
      "state_io": state_io,
      "shift": shift,
      "loop_iteration": 0
    }
    return init_loop_state

  def get_iteration_inputs(self, loop_iteration, state_io, shift):
    '''
    Construct stages_in: the global array that is operated on for this iteration, shape same as shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab a new batch from state_io
    '''

    # Setup potential input from state_io, which has a rotating microbatch index (size of microbatches_per_stage)
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    first_stage_in = state_io[:,state_io_batch_idx]

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

  def get_microbatch_ids(self, loop_iteration):
    '''Gets the microbatch_ids for all stages on this loop_iteration.'''
    # Stage 0 has processed one microbatch every loop_iter, but Stage 1 is one behind due to bubble, etc for other stages
    microbatches_processed = jnp.maximum(loop_iteration - jnp.arange(self.num_stages), 0) 
    microbatch_ids = microbatches_processed % self.config.num_pipeline_microbatches
    return microbatch_ids

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
    '''

    old_state_io = loop_state['state_io']
    loop_iteration = loop_state["loop_iteration"]
    # Shift becomes a rotated-right version of the previous output
    def _rotate_right(output_in):
      # Use lax.slice to avoid generating a gather.
      last = jax.lax.slice_in_dim(output_in, self.num_stages - 1, self.num_stages, axis=0)
      except_last = jax.lax.slice_in_dim(output_in, 0, self.num_stages - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)
    new_shift = _rotate_right(output)

    # Rotate stream_io left/up by 1 on rotating ms index (stream_buf_idx), replacing the last/bottom with the last stage output
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
      "loop_iteration": loop_iteration + 1
    }
    return new_loop_state
   
  def permute_output_ms_dim(self, output):
    # The first real output (batch 0) takes a certain amount of loop iterations to finish and be pushed to state_io - it will land on a different index of state_io depending on the number of iterations.
    first_output_num_iters = self.num_stages - 1
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
   loop_iteration = loop_state["loop_iteration"]

   microbatch_ids = self.get_microbatch_ids(loop_iteration)

   stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, shift)
   # We checkpoint stages_inputs since we are grabbing only one slice of the state_io, don't need to save the entire buffer.
   stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, 'iteration_input')
   stages_positions = self.vmap_gather(positions, microbatch_ids, 0) if positions is not None else None
   stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0) if segment_ids is not None else None

   vmap_func = self.get_main_vmap_func()

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
    else:
      example_position = None
    if segment_ids is not None:
      segment_ids = segment_ids.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length))
      example_segmentation = jax.lax.broadcast(segment_ids[0], [self.num_stages])
    else:
      example_segmentation = None

    loop_state = self.init_states(inputs)
    
    total_iterations = self.config.num_pipeline_microbatches + self.num_stages  - 1 

    if self.is_initializing():     
     vmap_func = self.get_main_vmap_func()

     # We only need to run one set of stages to initialize the variables, instead of looping over all microbatches for the full total_iterations.
     stage_outputs = vmap_func(self.layers, example_inputs, example_segmentation, example_position, deterministic, model_mode)
     if self.config.scan_layers:
       stage_outputs = stage_outputs[0]

     # We return something of the correct shape (global_batch, sequence, embed) by reshaping a single stages output which has
     # shape [microbatch_size, sequence, embed]
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
    if self.config.scan_pipeline_iterations and not self.is_initializing():
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
    final_output = self.permute_output_ms_dim(loop_state["state_io"])

    # reshape outputs to match input shape of total batch instead of microbatches [batch, sequence, embed]
    final_output = jnp.reshape(final_output, (self.config.global_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim))
                               
    return final_output