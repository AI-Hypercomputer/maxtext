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

''' Pipeline layer wrapping a decoder layer. Supports circular pipelining '''

import jax
import jax.ad_checkpoint
import numpy as np
from jax import numpy as jnp
from jax import tree_map
from flax.core import meta
from flax import linen as nn
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from typing import Optional
from layers import quantizations
import common_types
import functools
from typing import Any

PARAMS="params"
NON_TRAINABLE="non_trainable"
SUMMARIES = 'summaries'
INTERMEDIATES = 'intermediates'
RANDOM = 'random'
AUX_LOSS = 'aux_loss'
HYPER_PARAMS = 'hyper_params'


def stack_pytrees(*pytrees):
  """Returns a pytree with identical key structure but whose leaves have a new [num pytree inputs] leading dimension stacking from each input."""
  def stacking_fn(*leaves):
    return jnp.stack(leaves)
  return tree_map(stacking_fn, *pytrees)

# Pipeline is made up of several SimpleDecoderLayers 
class Pipeline(nn.Module):
  
  # TODO: Some properties, (num repeat, num_micro are through the config, some are derived. This makes it annoying to call different properties, some are self.property, some are self.config.property)
  # TODO: Should we declare the derived properties here as well (e.g. num_stages? I think anything declared here becomes required as an input though
  config: common_types.Config
  layers: nn.Module # The name of this property (layers) is reflected in the state pytree and thus also checkpoints.
  mesh: common_types.Mesh
  quant: Optional[quantizations.AqtQuantization] = None
  remat_policy: Any = None

  def setup(self):
    #decoder_layers = [self.decoder_layer_class(config=self.config, mesh=self.mesh, name=f'layers_{lyr}', quant=self.quant) for lyr in range(self.config.num_decoder_layers)]
    #self.decoder_layers = decoder_layers
    self.num_stages = self.config.ici_pipeline_parallelism * self.config.dcn_pipeline_parallelism
    self.layers_per_stage = self.config.num_decoder_layers / (self.num_stages * self.config.num_pipeline_repeats)
    # TODO: should this assert be in this class or in the initial pyconfig check?
    # assert self.layers_per_stage==1,f"Currently only supporting 1 layer per pipeline stage, but {self.config.num_decoder_layers} layers were requested with {self.num_stages * self.config.num_pipeline_repeats} total stages = {self.num_stages} pipeline parallel axes * {self.config.num_pipeline_repeats} circ repeats"
    self.use_circ_storage = self.config.num_pipeline_repeats > 1 and self.config.num_pipeline_microbatches > self.num_stages
    self.microbatch_size = self.config.global_batch_size_to_train_on // self.config.num_pipeline_microbatches
    microbatches_per_stage = self.config.num_pipeline_microbatches // self.num_stages
    # TODO: should this assert be in this class or pyconfig check?
    assert microbatches_per_stage * self.num_stages == self.config.num_pipeline_microbatches, f"Currently the number of microbatches ({self.config.num_pipeline_microbatches}) must be divisible by the number of stages ({self.num_stages})"
    self.microbatches_per_stage = microbatches_per_stage

    
  def S(self, *specs):
    return NamedSharding(self.mesh, PartitionSpec(*specs))

  def init_states(self, inputs):
    '''Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
        Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

        Returns
          shift: zeros shape [num_stages, micro_size, sequence, embed]
          state_io: reshaped inputs [num_stages, microbatches/stages, micro_size, sequence, embed]
          circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed]
          circ_storage_mover: zeros[num_stages, micro_size, sequence, embed]
    
    '''

    # Shift is used to rotate the output of each pipeline into the input of the next
    # shift has shape [num_stages, micro_size, sequence, embed]
    shift = jnp.zeros((self.num_stages,) + inputs.shape[1:], dtype=inputs.dtype)
    shift = nn.with_logical_constraint(shift, ("activation_stage", "activation_batch", "activation_length", "activation_embed"),rules=self.config.logical_axis_rules,mesh=self.mesh)

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs as the pipeline runs/finishes
    # state_io has shape [num_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(inputs, (self.num_stages, self.microbatches_per_stage) + inputs.shape[1:])
    state_io = nn.with_logical_constraint(state_io, ("activation_stage", None, "activation_batch", "activation_length", "activation_embed"),rules=self.config.logical_axis_rules, mesh=self.mesh) 

    # TODO: verify comment below
    # The data/fsdp can shard over microbatch_size, not number of microbatches. The num_microbatches is looped over so should not be sharded.

    # TODO: Consider sharding and/or changing the circ storage. Understand if/when setting microbatches > num_stages is beneficial which is when circ_storage is needed
    # circ_storage is used to hold the final pipeline stage outputs before it is used for the next repeat. It is only needed
    # when num_microbatches > num_stages, else instead the final stage can immediately pass to the first without additional storage.
    # Alternative name is "between_repeats_storage", since this storage is not needed always for circular - but only when microbatches > num_stages
    # circ_storage has shape [num_stages, microbatches, micro_size, sequence, embed] -- this is huge btw, it should be reducible by a factor of num_stages
    if self.use_circ_storage:
        circ_storage = jnp.zeros((self.num_stages,) + inputs.shape , dtype=inputs.dtype)
    else:
       circ_storage = None

    # circ_storage_mover is used to push the microbatches from the pipeline into circ_storage
    # circ_storage_mover shape is same as shift: [num_stages, micro_size, sequence, embed]
    # This mover is one iteration behind before being pushed into storage - which is why we can't just re-use output
    # However shouldn't we be able to keep only the last stage's output instead of all stages?
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
    This is almost a rotated version of the last outputs, except for the first stage which must grab a new batch from either state_io or circ_storage
    '''

    # Setup potential input from state_io, which has a rotating microbatch index (size of micro/stages, state_io_batch_idx below)
    state_io_batch_idx = loop_iteration % self.microbatches_per_stage
    state_io_slice = state_io[:,state_io_batch_idx] 

    if self.use_circ_storage:
        # Setup potential input from circ_storage, which also has a rotating index for microbatch, size of num_microbatches
        circ_storage_batch_idx = loop_iteration % self.config.num_pipeline_microbatches
        circ_storage_slice = circ_storage[:,circ_storage_batch_idx]
    else:
        circ_storage_slice = shift

    stages_in = jnp.where(loop_iteration < self.config.num_pipeline_microbatches, state_io_slice, circ_storage_slice)

    def select_state_or_input(input, shift):
        # Selects input for stage 0, shift for other stages
        return jnp.where(jax.lax.broadcasted_iota('int32', shift.shape, 0) == 0, input, shift)

    # Selects input (from stream_io or circ_slice) for stage 0, other stages get from shift (the rotated previous output)
    stages_in = select_state_or_input(stages_in, shift)
    return stages_in

  # TODO: Instead of unconstrained, should we passing the initial sharding of X?
  # TODO: This can be used instead of shard_leading_dim_by_stages
  def shard_dim_by_stages(self, x, dim: int):
    dims_mapping = [jax.sharding.PartitionSpec.UNCONSTRAINED] * x.ndim
    #dims_mapping = [None] * x.ndim
    dims_mapping[dim] = "stage"
    dims_mapping = tuple(dims_mapping)
    p1 = jax.sharding.PartitionSpec(*dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh,p1)
    return jax.lax.with_sharding_constraint(x, sharding) # maybe PartitionSpec(dims_mapping)
    #return jax.lax.with_sharding_constraint(x, PartitionSpec(*tuple(dims_mapping))) # maybe PartitionSpec(dims_mapping)


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
      dim = repeat_dim_in_weights
      if stages_dim_in_weights < dim:
        dim -= 1
      return jnp.squeeze(jax.lax.dynamic_slice_in_dim(x, repeat_id, 1, dim), dim)

    # TODO: this boolean is always true, out_dim should just be 0
    out_dim = stages_dim_in_weights
    if repeat_dim_in_weights < stages_dim_in_weights:
      out_dim -= 1

    repeat_ids = self.shard_leading_dim_by_stages(repeat_ids)
    weights = self.shard_dim_by_stages(weights, stages_dim_in_weights)
    outs = jax.vmap(_gather_one, in_axes=(stages_dim_in_weights, 0), out_axes=out_dim)(weights, repeat_ids)
    # Also shard outs
    #outs = self.shard_dim_by_stages(outs, out_dim)

    # dims_mapping[dim] = "stage"
    # dims_mapping = tuple(dims_mapping)
    dims_mapping = ("stage", "fsdp", "tensor")
    p1 = jax.sharding.PartitionSpec(*dims_mapping)
    sharding = jax.sharding.NamedSharding(self.mesh,p1)

    print(f"Shape of outs is {outs.shape}", flush=True)
    outs = jax.lax.with_sharding_constraint(outs, p1)
    return outs

  def get_microbatch_id(self, stage_idx, loop_iteration):
    '''Gets the microbatch_id on this loop_iteration for this stage. Works for both circular and non-circular'''
    return (loop_iteration - stage_idx) % self.config.num_pipeline_microbatches

  def shard_leading_dim_by_stages_old(self, x):
    stage_sharding_constraint = ('layers',) + tuple([None] * (x.ndim -1))
    # return jax.lax.with_sharding_constraint(x, ('stage',) + tuple([None] * (x.ndim -1)))
    return nn.with_logical_constraint(x, stage_sharding_constraint, mesh=self.mesh, rules=self.config.logical_axis_rules)

  def shard_leading_dim_by_stages(self, x):
    partition_spec_list = [jax.sharding.PartitionSpec.UNCONSTRAINED] * x.ndim
    partition_spec_list[0] = jax.sharding.PartitionSpec("stage")
    partition_spec = jax.sharding.PartitionSpec(*partition_spec_list)
    return jax.lax.with_sharding_constraint(x, partition_spec)

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

    ids = self.shard_leading_dim_by_stages(ids)
    outs = jax.vmap(_gather_one, in_axes=(None, 0), out_axes=ids_dim)(xs, ids)
    return self.shard_leading_dim_by_stages(outs)

  def get_microbatches_for_stages(self, microbatched_array, loop_iteration):
    '''
    Returns an array of leading dimension stages grabbing the current microbatch for each stage.
    TODO: This is not actually used to get the microbatches, but the position and segment IDs, so probably should change method name
    
    Input:
        microbatched_array: Array to grab from, should have leading dimension num_microbatches
        loop_iteration: Integer of loop index
    Returns:
        One microbatch from microbatched_array for each stage. Array same shape as microbatched_array except the leading dimension is replaced by num_stages
    '''

    microbatched_stages_list = [microbatched_array[self.get_microbatch_id(stage_idx, loop_iteration)] for stage_idx in range(self.num_stages)]
    stages_array = jnp.concatenate(microbatched_stages_list, axis=0)
    stages_array = jnp.reshape(stages_array, (self.num_stages,) + microbatched_array.shape[1:])
    return stages_array

  def get_new_loop_state(self,output, loop_state):
    '''
      Update the various buffers given the output of the most recent iteration
      * state_io: rotates left/up by 1 (replace last element with last stage output) - we are pushing inputs up into the pipeline
      * shift: rotate output right/down by 1 - we imagine the pipeline moves to right/down
      * circ_storage: push latest circ_mover (e.g. FULL outputs) into rotating index -- why are we pushing full outputs, why not just last stage?
      * circ_mover gets FULL? rotated output -- I think it should only need the last stage of output
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
    #new_shift = _rotate_right(output) #TODO(big):file a bug or ping again on jax chat, why do we need to jit here
    jit_rotate_right = jax.jit(_rotate_right)
    new_shift = jit_rotate_right(output)

    if self.use_circ_storage:
        # Insert the circ_storage_mover into new_circ_storage at a microbatch-rotating index.
        # circ_storage_mover still points to the output of PREVIOUS iteration, which should aid in allowing overlapped compute/async transfers
        def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
            rotated = _rotate_right(circ_storage_mover_in)
            rotated = jnp.expand_dims(rotated, 1)
            # The offset is the last stage's last microbatch ID. 
            offset = (loop_iteration - (self.num_stages - 1) - 1) % self.config.num_pipeline_microbatches # Note extra -1 b/c grabbing from the previous output - circ_storage_mover is one iter behind
            return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)
        new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
        new_circ_storage_mover = output
    else:
       new_circ_storage = None
       new_circ_storage_mover = None

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
    #new_state = _update_state_io(old_state_io, stream_slice, output)# TODO(medium):same sharding/jit issue
    jit_update_state_io = jax.jit(_update_state_io)
    new_state = jit_update_state_io(old_state_io, stream_slice, output) 
    
    new_loop_state = {
      "state_io": new_state,
      "shift": new_shift,
      "circ_storage": new_circ_storage,
      "circ_storage_mover": new_circ_storage_mover,
      "loop_iteration": loop_iteration + 1
    }
    return new_loop_state
   
  def permute_output_ms_dim(self, output):
    '''
    TODO: Likely need to match http://google3/third_party/py/praxis/layers/pipeline.py;l=935;rcl=577941721 to avoid all gathers
    '''

    # The first real output (batch 0) takes a certain amount of loop iterations to finish and be pushed to state_io - it will land on a different index of state_io depending on the number of iters
    first_output_num_iters = self.config.num_pipeline_microbatches * (self.config.num_pipeline_repeats - 1) + self.num_stages - 1
    # The first term above is a multiple of num_pipeline_microbatches and thus could be ignored since its also a multiple of microbatches_per_stage, but we keep it for clairty
    land_idx = first_output_num_iters % self.microbatches_per_stage
    permutation = (np.arange(self.microbatches_per_stage) + land_idx) % self.microbatches_per_stage # make the value in land_idx actually appear in idx 0, and (land_idx + 1) appear in spot 1, etc
    output = output[:,permutation]
    return output

  def get_main_vmap_func(self, segment_stage_idx, positions_stage_idx):
      # With magic weight via name gathering
    def func_to_vmap(body_instance,stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode):
      return body_instance(stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)

    # TODO: this can probably be removed
    mutable_func_to_vmap = nn.map_variables(
      func_to_vmap,
      mapped_collections=True,
      mutable=True
    )
    vmap_func = nn.vmap(
      mutable_func_to_vmap,
      in_axes=(0, segment_stage_idx, positions_stage_idx, None, None),
      spmd_axis_name='stage',
      variable_axes={'params': 0},
      # TODO: params:self.is_initializing instead of always true
      split_rngs={'params': True},
      metadata_params={
        nn.PARTITION_NAME: "layers",
        'sub_weight_split_dims_mapping': (None),
        "is_initializing": self.is_initializing(),
        "x_times": self.num_stages}
    )
    return vmap_func

  def run_one_iteration(self, loop_state, positions, segment_ids, deterministic, model_mode, decoder_layer_instance):
   '''Run one loop iteration - gets weights and inputs for each stage, run the stages in parallel, and update the various state buffers'''
   state_io = loop_state['state_io']
   shift = loop_state["shift"]
   circ_storage = loop_state["circ_storage"]
   circ_storage_mover = loop_state["circ_storage_mover"]
   loop_iteration = loop_state["loop_iteration"]

   microbatch_ids = jnp.maximum(loop_iteration - jnp.arange(self.num_stages), 0)
   microbatch_ids = microbatch_ids % self.config.num_pipeline_microbatches

  # TODO: try sharding this again explicitly over stages
   stages_inputs = self.get_iteration_inputs(loop_iteration, state_io, circ_storage, shift)
   # We checkpoint stages_inputs since we are grabbing only one slice of the state_io, don't need to save the entire buffer.
   stages_inputs = jax.ad_checkpoint.checkpoint_name(stages_inputs, 'iteration_input')

   if positions is not None:
    stages_positions = self.vmap_gather(positions, microbatch_ids, 0)
    positions_stage_idx = 0
   else:
     stages_positions = None
     positions_stage_idx = 0 # can be 0 or None? TODO
   if segment_ids is not None:
    stages_segment_ids = self.vmap_gather(segment_ids, microbatch_ids, 0)
    segment_stage_idx = 0
   else:
    stages_segment_ids = None
    segment_stage_idx = 0 # can be 0 or None? TODO

   vmap_func = self.get_main_vmap_func(segment_stage_idx, positions_stage_idx)

   if self.config.num_pipeline_repeats > 1:
    microbatch_ids = jnp.maximum(loop_iteration - jnp.arange(self.num_stages), 0) # not a great name, this is really something like microbatch_id * repeat idx
    repeat_ids = microbatch_ids // self.config.num_pipeline_microbatches

    metadata_params={
      nn.PARTITION_NAME: "circular_repeats",
      'sub_weight_split_dims_mapping': (None,), #(None,), # Maybe -1? 
      "is_initializing": True,
      "x_times": self.config.num_pipeline_repeats,
      'optimizer_dims_mapping': None,
    }
    def prepare_vars_for_main_vmap(weights):
      def gather_weights_for_stages_in(weights):
        return jax.tree_map(
            functools.partial(
                self.vmap_parallel_gather, repeat_ids=repeat_ids, repeat_dim_in_weights=0, stages_dim_in_weights=1),
            weights)
      # IDea: also need to remove axis from metadata "pipeline_Repeats"
      #breakpoint()
      weights = meta.remove_axis(weights, 0, metadata_params) # Remove the circular_repeats axis annotation, we will select only one circular_repeat per stage and remove this axis
      #weights = meta.remove_axis(weights, 0, metadata_params) # Remove the circular_repeats axis annotation, we will select only one circular_repeat per stage and remove this axis
      weights = gather_weights_for_stages_in(weights)
      #weights = meta.remove_axis(weights, 0, metadata_params) # Remove the circular_repeats axis annotation, we will select only one circular_repeat per stage and remove this axis
      return weights

    vmap_func = nn.map_variables(
        vmap_func,
        mapped_collections=[PARAMS, NON_TRAINABLE, SUMMARIES, INTERMEDIATES],
        mutable=True,
        trans_in_fn=prepare_vars_for_main_vmap,
    )
   stages_output, _ = vmap_func(decoder_layer_instance, stages_inputs, stages_segment_ids, stages_positions, deterministic, model_mode)

   new_state = self.get_new_loop_state(stages_output, loop_state)
   return new_state

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, segment_ids: jnp.ndarray, positions:jnp.ndarray, deterministic: bool, model_mode=common_types.MODEL_MODE_TRAIN) -> jnp.ndarray:
    # Reshape inputs of [global_batch, ...] to [microbatches, microbatch_sizes, ...]
    # TODO: Shard inputs after reshape along num_microbatches?
    inputs = inputs.reshape((self.config.num_pipeline_microbatches, self.microbatch_size, self.config.max_target_length, self.config.emb_dim))
    example_inputs = jax.lax.broadcast(inputs[0], [self.num_stages])
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
    total_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats + self.num_stages  - 1 

    def func_to_scan(model,loop_state, xs):
       return model.run_one_iteration(loop_state, positions, segment_ids, deterministic, model_mode, model.layers), None
    
    use_scan = False
    variable_carry = []
    variable_broadcast = ["params"]
    NON_TRAINABLE="non_trainable"
    if self.is_mutable_collection(NON_TRAINABLE):
      variable_carry.append(NON_TRAINABLE)
    else:
      variable_broadcast.append(NON_TRAINABLE)

    if self.is_initializing():
     # TODO(possible): praxis is using the _scan_Fn, possibly having the real fwd use scan but not the initial causes issues
     # TODO: to call this need to reshape segments and positions to num_layers probably
     # We only need to run one set of stages to initialize the variables, instead of looping over all microbatches
     vmap_func = self.get_main_vmap_func(segment_idx, position_idx)
     if self.config.num_pipeline_repeats > 1:
       vmap_func= nn.vmap(
         vmap_func,
         in_axes=(0, segment_idx, position_idx, None, None),
          variable_axes={
            'params': 0,
            NON_TRAINABLE: 0,
            "hyper_params": 0,
          },
          # TODO: params:self.is_initializing instead of always true
          split_rngs={'params': True},
          #spmd_axis_name="stage",
          metadata_params={
            nn.PARTITION_NAME: "circular_repeats",
            'sub_weight_split_dims_mapping': (None,), #(None,), # Maybe -1? 
            "is_initializing": True,
            "x_times": self.config.num_pipeline_repeats,
            'optimizer_dims_mapping': None,
          }
        )
       
       example_inputs = jax.lax.broadcast(example_inputs, [self.config.num_pipeline_repeats])

       example_segmentation = jax.lax.broadcast(example_segmentation, [self.config.num_pipeline_repeats]) if example_segmentation is not None else None
       example_position = jax.lax.broadcast(example_position, [self.config.num_pipeline_repeats]) if example_position is not None else None
       # To shard weight (both for initialization and at rest) for the circular pipeline
       # we create weights of shape [num_repeat, num_stages, ...] (e.g. [num_repeat, num_stages, embed, mlp])
       # and shard the num_stages  we wrap the main stage vmap with a num_repeat vmap to generate this axis only for parameter initialization
     stage_outputs, _ = vmap_func(self.layers, example_inputs, example_segmentation, example_position, deterministic, model_mode)
     # We return something of the correct shape (global_batch, sequence, embed) by reshaping a single stages output which has
     # shape [microbatch_size, sequence, embed]
     if self.config.num_pipeline_repeats > 1:
       stage_outputs = stage_outputs[0]
     broadcasted_stage_outpus = jax.lax.broadcast(stage_outputs[0], [self.config.global_batch_size_to_train_on // self.microbatch_size])
     return jnp.reshape(broadcasted_stage_outpus, [self.config.global_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim])


    # The scan cannot be used on init since it broadcasts the state. Thus the state must be independent of the loop body,
    # but on init the loop body will initialize the params.
    if self.config.scan_pipeline_iterations and not self.is_initializing():
        
        if self.remat_policy is not None:
          remat_policy = jax.checkpoint_policies.save_from_both_policies(
              self.remat_policy,
              jax.checkpoint_policies.save_only_these_names('iteration_input')
          )
        else:
          remat_policy = jax.checkpoint_policies.save_only_these_names('iteration_input')
        remat_fn = nn.remat(
          func_to_scan,
          prevent_cse=False, # prevent_cse not used with scan
          policy=remat_policy
        )
        scan_func = nn.scan(
          remat_fn,
          variable_axes={
            SUMMARIES: 0,
            AUX_LOSS: 0,
            INTERMEDIATES: 0,
            HYPER_PARAMS: 0,
          },
          variable_broadcast=variable_broadcast,
          variable_carry=variable_carry,
          # Dropout/aqt keys will be split for each iteration.
          split_rngs={RANDOM: True},
          length=total_iterations,
          )
        loop_state, _ = scan_func(self, loop_state, None)
    else:
        for loop_iteration in range(total_iterations):
            loop_state = self.run_one_iteration(loop_state, positions, segment_ids, deterministic, model_mode, self.layers)

    # The final output is located in the input/output array, however the output microbatches may be permuted relative to the input
    final_output = self.permute_output_ms_dim(loop_state["state_io"])

    # reshape outputs to match input shape of total batch instead of microbatches [batch, sequence, embed]
    final_output = jnp.reshape(final_output, (self.config.global_batch_size_to_train_on, self.config.max_target_length, self.config.emb_dim))
                               
    return final_output