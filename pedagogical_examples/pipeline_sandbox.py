import jax

from jax import numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
import os
import argparse
import pipeline_utils
import functools
from flax import linen as nn

def S(mesh, *specs):
    return NamedSharding(mesh, PartitionSpec(*specs))

def shard_dim_by_stages(x, mesh):
   '''Assumes the stages dimension is leading and the mesh has name stages.'''
   specs = ['stage'] + [None] * (x.ndim - 1)
   stage_sharding = S(mesh, *specs)
   return jax.lax.with_sharding_constraint(x, stage_sharding)

def get_weights_and_inputs(batch_size, sequence, features, n_layers):
    '''Get random weights, random inputs, and random targets

        Returns
            weights: [n_layers, features, features]
            inputs: [global_batch, sequence, features]
            targets: [global_batch, sequence, features]
    '''
    weights_shape = jnp.array([n_layers, features, features]) # pytree in real cases instead of single array
    k = jax.random.PRNGKey(1)
    weights = jax.random.normal(k,weights_shape, dtype=jnp.float32)

    # we pass in input with global batch, its up to the pipeline function to reshape to microbatches
    input_shape = [batch_size, sequence, features]
    k = jax.random.PRNGKey(2)
    inputs = jax.random.normal(k,input_shape, dtype=jnp.float32)
    
    # dummy targets same shape as inputs to use for a dummy loss funciton to check gradient correctness
    k = jax.random.PRNGKey(3)
    dummy_targets = jax.random.normal(k,input_shape, dtype=jnp.float32)

    return weights, inputs, dummy_targets

def init_states(inputs, n_stages, use_circ_storage, mesh):
    '''Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
        Assumes input has already been reshaped into microbatches: [num_micro_batches, micro_batch_size, sequence, embed]

        Returns
          shift: zeros shape [n_stages, micro_size, sequence, embed]
          state_io: reshaped inputs [n_stages, microbatches/stages, micro_size, sequence, embed]
          circ_storage: zeros [num_stages, microbatches, micro_size, sequence, embed]
          circ_storage_mover: zeros[n_stages, micro_size, sequence, embed]
    
    '''

    n_microbatches = inputs.shape[0]

    # Shift is used to rotate the output of each pipeline into the input of the next
    # shift has shape [n_stages, micro_size, sequence, embed]
    shift = jnp.zeros((n_stages,) + inputs.shape[1:])
    shift = shard_dim_by_stages(shift, mesh)
    shift = jax.lax.with_sharding_constraint(shift, S(mesh, 'stage', 'data', None, 'tensor'))

    # state_io (state input output) at first holds all of the input batches, but also will hold the outputs as the pipeline runs/finishes
    # state_io has shape [n_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(inputs, (n_stages, n_microbatches // n_stages) + inputs.shape[1:])
    #state_io = shard_dim_by_stages(state_io, mesh)
    state_io = jax.lax.with_sharding_constraint(state_io, S(mesh, 'stage', None, 'data', None, 'tensor'))
    # shard over microbatch_size, not number of microbatches. The num_microbatches is looped over so should not be sharded.

    # circ_storage is used to hold the final pipeline stage outputs before it is used for the next repeat. It is only needed
    # when num_microbatches > num_stages, else instead the final stage can immediately pass to the first without additional storage.
    # Alternative name is "between_repeats_storage"
    # circ_storage has shape [num_stages, microbatches, micro_size, sequence, embed] -- this is huge btw, it should be reducible by a factor of num_stages
    if use_circ_storage:
        circ_storage = jnp.zeros((n_stages,) + inputs.shape )
    else:
       circ_storage = None

    # circ_storage_mover is used to push the microbatches from the pipeline into circ_storage
    # circ_storage_mover shape is same as shift: [n_stages, micro_size, sequence, embed]
    # This mover is one iteration behind before being pushed into storage - which is why we can't just re-use output
    # However shouldn't we be able to keep only the last stage's output instead of all stages?
    if use_circ_storage:
        circ_storage_mover = shift
    else:
       circ_storage_mover = None

    return state_io, shift, circ_storage, circ_storage_mover

def stage(weights, x):
  x = layer(weights, x) # To support multiple layers per stage we could add a for loop here instead of one layer
  return x

def layer(weights, stages_in):
    outputs = jnp.einsum('bse,eh->bsh',stages_in,weights) # The leading stage dimensions of weights and stages_in is missing because it is vmapped out
    outputs = jnp.tanh(outputs)
    return outputs

def get_weights_stage(weights, loop_iteration, n_stages, n_microbatches):
    '''
    Get the weights for each stage used for this loop itereation. 
    
    Input:
        Weights of shape [num_layers, embed, embed]
    Returns:
        Weights for stages of shape [stage, embed, embed].

    For non-circular pipelines this would just be stacked [weights_layer_0; weights_layer1; etc],
    but for circular the stages need a repeat_idx to determine what layer weights to grab, e.g. on iteration 5 with 4 stages
    the repeat indexes are [1;1;0;0] so need weights [4,5,2,3]
    '''
    # We use numpy instead of jnp so these indexes are not traced
    microbatch_ids = np.maximum(loop_iteration - np.arange(n_stages), 0) # not a great name, this is really batch_id * repeat idx
    repeat_ids = microbatch_ids // n_microbatches
    layer_ids = np.arange(n_stages) + repeat_ids * n_stages
    #layer_ids goes out of bounds on the last bubble, we cap it within range.
    layer_ids= np.minimum(layer_ids, weights.shape[0] - 1)
    # slice_in_dim avoids executing an all gather
    to_stack = [jax.lax.slice_in_dim(weights,layer_ids[stage], layer_ids[stage] + 1, axis=0) for stage in range(n_stages)]
    weights_stage = jnp.concatenate(to_stack, axis=0)
    desired_shape = (n_stages,) + weights.shape[1:]
    weights_stage = jnp.reshape(weights_stage, desired_shape) # This reshape fleshes out singleton axes that are flattened in concatenate
    return weights_stage

def get_iteration_inputs(loop_iteration, microbatches, num_stages, state_io, circ_storage, shift, use_circ_storage):
    '''
    Construct stages_in: the global array that is operated on for this iteration, shape same as shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab from state_io or circ_storage
    '''

    # Setup potential input from state_io. state_io has a rotating microbatch index (size of micro/stages, stream_buf_idx below)
    state_io_batch_idx = loop_iteration % (microbatches // num_stages)
    state_io_slice = state_io[:,state_io_batch_idx] 

    if use_circ_storage:
        # Setup potential input from circ_slice, which also has a rotating index for microbatch
        circ_storage_batch_idx = loop_iteration % microbatches
        circ_storage_slice = circ_storage[:,circ_storage_batch_idx]
    else:
        circ_storage_slice = shift # TODO: confirm correct, rename circ_storage_slice

    stages_in = jnp.where(loop_iteration < microbatches, state_io_slice, circ_storage_slice)

    def select_state_or_input(input, shift):
        # Selects input for stage 0, shift for other stages
        return jnp.where(jax.lax.broadcasted_iota('int32', shift.shape, 0) == 0, input, shift)

    # Selects input (from stream_io or circ_slice) for stage 0, other stages get from shift (the rotated previous output)
    stages_in = select_state_or_input(stages_in, shift)
    return stages_in

def get_new_loop_state(output, old_state_io, old_circ_storage, old_circ_storage_mover, loop_iteration, use_circ_storage):
    '''
      Update the various buffers given the output of the most recent iteration
      * state_io: rotates left/up by 1 (replace last element with last stage output) - we are pushing inputs up into the pipeline
      * shift: rotate output right/down by 1 - we imagine the pipeline moves to right/down
      * circ_storage: push latest circ_mover (e.g. FULL outputs) into rotating index -- why are we pushing full ouputs, why not just last stage?
      * circ_mover gets FULL? rotated output -- I think it should only need the last stage of output
    '''
    
    n_stages = old_state_io.shape[0]
    n_microbatches = old_state_io.shape[0] * old_state_io.shape[1]

    # Shift becomes a rotated-right version of the previous output
    def _rotate_right(output_in):
      # Use lax.slice to avoid generating a gather.
      last = jax.lax.slice_in_dim(output_in, n_stages - 1, n_stages, axis=0)
      except_last = jax.lax.slice_in_dim(output_in, 0, n_stages - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)
    new_shift = _rotate_right(output)

    if use_circ_storage:
        # Insert the circ_storage_mover into new_circ_storage at a microbatch-rotating index.
        # circ_storage_mover still points to the output of PREVIOUS iteration, I believe this is intentional to help with async transfers
        # The fact that its the previous iteration isn't a problem - we just need to store the last pipeline stage output at some point.
        def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
            rotated = _rotate_right(circ_storage_mover_in)
            rotated = jnp.expand_dims(rotated, 1)
            # The offset is the last stage's last microbatch ID. 
            offset = (loop_iteration - (n_stages - 1) - 1) % n_microbatches # we need extra -1 b/c grabbing from un-updated circ_storage_mover (one iter behind)
            return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)
        new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
        new_circ_storage_mover = output
    else:
       new_circ_storage = None
       new_circ_storage_mover = None

    # Rotate stream_io left/up by 1 on rotating ms index (stream_buf_idx), replacing the last/bottom with the last stage output
    stream_buf_idx = loop_iteration % (n_microbatches // n_stages)
    stream_slice = old_state_io[:, stream_buf_idx]
    def _update_state_io(state_in, stream_slice, output):
        # Shift the current slice to the left, then fill the last stage with
        # the final output.
        padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
        stream_slice = jax.lax.slice_in_dim(
            jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
        stream_slice = jnp.where(
            jax.lax.broadcasted_iota('int32', stream_slice.shape, 0) == n_stages - 1, output,
            stream_slice)
        stream_slice = jnp.expand_dims(stream_slice, 1)
        return jax.lax.dynamic_update_slice_in_dim(
            state_in, stream_slice, stream_buf_idx, axis=1)
    new_state = _update_state_io(old_state_io, stream_slice, output)

    return new_state, new_shift, new_circ_storage, new_circ_storage_mover

def run_one_iteration(state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights, use_circ_storage):
   '''
      Run one loop iteration - sending inputs and specifying weights for each pipeline stage, run the pipeline, and update the various state buffers
   '''
   n_stages = state_io.shape[0]
   n_microbatches = state_io.shape[0] * state_io.shape[1]
   stages_in = get_iteration_inputs(loop_iteration, n_microbatches, n_stages, state_io, circ_storage, shift, use_circ_storage)
   weights_stage = get_weights_stage(weights, loop_iteration, n_stages, n_microbatches)
   output = jax.vmap(stage, in_axes=0, out_axes=0,
                        spmd_axis_name='stage')(weights_stage, stages_in)
   new_state_io, new_shift, new_circ_storage, new_circ_storage_mover = get_new_loop_state(output, state_io, circ_storage, circ_storage_mover, loop_iteration, use_circ_storage)
   return new_state_io, new_shift, new_circ_storage, new_circ_storage_mover

def permute_output_ms_dim(output):
    '''
    Although re-using the same array for both input and output is cute,
    The final outputs turn out permuted compared to the inputs. Worringly I don't see this function in praxis
    '''

    n_stages = output.shape[0]
    ms_size = output.shape[1]
    # The first real output takes a certain amount of loop iterations to finish and be pushed to state_io - it will land on a different index of state_io depending on this 
    # Really (n_microbatches * (n_repeat - 1) + n_stages - 1) % ms_size but ms_size divides n_microbatches
    land_idx = (n_stages - 1) % ms_size # first_finish % ms_size
    permutation = (np.arange(ms_size) + land_idx) % ms_size # make the value in land_idx actually appear in idx 0, and (land_idx + 1) appear in spot 1, etc
    output = output[:,permutation]
    return output

def run_pipeline(weights, inputs, n_stages, n_microbatches, n_repeat, use_circ_storage, mesh):
    '''
    Runs all iterations of the pipeline. Takes input formmated as regular batching (not microbatched). Will reshape the inputs
    internally for microbatching and reshape the outputs to match the original input

    Inputs:
        weights: [layers, features, features]
        inputs: [global batch, sequence, features]
        n_stages: int
        n_micorbatches: int (must divide n_stages)
        n_repeat: int
    '''

    global_batch_size, sequence, features = inputs.shape[0], inputs.shape[1], inputs.shape[2]
    microbatch_size = global_batch_size // n_microbatches

    # Reshape from [global_batch, sequence, embed] to [num_micro_batches, micro_batch_size, sequence, embed]
    inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))

    state_io, shift, circ_storage, circ_storage_mover = init_states(inputs, n_stages, use_circ_storage, mesh)

    total_iterations = n_microbatches * n_repeat + n_stages  - 1 
    for loop_iteration in range(total_iterations):
       state_io, shift, circ_storage, circ_storage_mover = run_one_iteration(state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights, use_circ_storage)

    # The final output is located in the input/output array, however the microbatches may be permuted
    final_output = permute_output_ms_dim(state_io)

    # reshape state to match input shape of total batch instead of microbatches [batch, sequence, embed]
    #final_output = jnp.reshape(final_output, (args.n_microbatches,) + state_io.shape[2:])
    final_output = jnp.reshape(final_output, (global_batch_size, sequence, features))
                               
    return final_output

def create_mesh(n_stages, tp_axis, dp_axis):
  devices = mesh_utils.create_device_mesh((n_stages, tp_axis, dp_axis))
  mesh = Mesh(devices, axis_names=('stage', 'tensor', 'data'))
  return mesh

def get_logical_axis_rules():
    return (
        ('pipeline_stages', 'stage'),
        ('activation_length', 'sequence'),
        ('activation_embed', 'tensor'),
        ('activation_mlp', 'tensor'),
        ('embed', 'sequence'),
        ('mlp', 'tensor'),
    )

def get_pipelint_jit(n_stages, dp_axis, mesh):
  # Configure shardings passed to in and out_sharding. Possibly this can be refactored somewhere different

  # Fully sharded
  weight_sharding = S(mesh, 'stage', 'data', 'tensor') # weight sharded over stage -- this is actually highly inefficinet for circular pipelining, we should reshape first with repeat_idx
  input_sharding = S(mesh, 'data', None, 'tensor')   # inputs sharded over batch -- unsure about how inputs should be sharded into PP
  result_sharding = S(mesh, 'data', None, 'tensor')  # output sharded over batch -- unsure about how outputs should be sharded out of PP

  # Only stage sharded
  #weight_sharding = S(mesh, 'stage', None, None) # weight sharded over stage -- this is actually highly inefficinet for circular pipelining, we should reshape first with repeat_idx
  #input_sharding = S(mesh, 'data', None, None)   # inputs sharded over batch
  #result_sharding = S(mesh, 'data', None, None)  # output sharded over batch
  

  pipeline_with_mesh = functools.partial(run_pipeline, mesh=mesh)
  jitted_pipeline = jax.jit(run_pipeline,
              in_shardings=((weight_sharding, input_sharding)),
              out_shardings=result_sharding,
              static_argnums=[2,3,4,5,6])
  return jitted_pipeline

def main() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    parser = argparse.ArgumentParser(description='Pipeline Parallelism Options')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--n_stages', type=int, default=4)
    parser.add_argument('--n_microbatches', type=int, default=8)
    parser.add_argument('--dp_axis', type=int, default=1)
    parser.add_argument('--tp_axis', type=int, default=1)
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--sequence', type=int, default=16)
    parser.add_argument('--n_repeat', type=int, default=2)

    args = parser.parse_args()
    args.microbatch_size = args.batch_size // args.n_microbatches
    args.layers = args.n_stages * args.n_repeat
    use_circ_storage = args.n_repeat > 1 and args.n_microbatches > args.n_stages

    # Necessary artifacts for the fun stuff
    mesh = create_mesh(args.n_stages, args.tp_axis, args.dp_axis)
    pipeline_func = get_pipelint_jit(args.n_stages, args.dp_axis, mesh)
    weights, inputs, targets = get_weights_and_inputs(args.batch_size, args.sequence, args.features, args.layers)

    # The fun stuff
    #pipeline_utils.assert_same_output_and_grad(pipeline_utils.reg_matmuls, pipeline_func, targets, weights, inputs,f2_extra_inputs=[args.n_stages, args.n_microbatches, args.n_repeat, use_circ_storage, mesh])

    pipeline_utils.simple_timeit(pipeline_func, weights, inputs, args.n_stages, args.n_microbatches, args.n_repeat, use_circ_storage, mesh, tries = 3, task = 'circular_pipeline')

if __name__ == "__main__":
  main()