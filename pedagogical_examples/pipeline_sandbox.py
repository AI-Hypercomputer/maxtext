import jax

from jax import numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
import os
import argparse


def S(mesh,*specs):
  return NamedSharding(mesh, PartitionSpec(*specs))

def get_iteration_inputs(loop_iteration, microbatches, num_stages, state_io, circ_storage, shift):
    '''
    Construct stages_in: what each stage will operate on of global shape same as shift=[stages, micro_size, sequence, embed]
    This is almost a rotated version of the last outputs, except for the first stage which must grab from state_io or circ_storage
    '''

    # Grab input from state_io - rotate through the microbatch dimension
    stream_buf_idx = loop_iteration % (microbatches // num_stages)
    stream_slice = state_io[:,stream_buf_idx] 

    # For early iterations we grab the new inputs from state_io, but once those have all passed into shift
    # we grab from circular_storage
    circ_slice = circ_storage[:,loop_iteration % microbatches]
    stages_in = jnp.where(loop_iteration < microbatches, stream_slice, circ_slice)

    def select_state_or_input(input, shift):
    # Selects input for stage 0, state for other stages
        return jnp.where(jax.lax.broadcasted_iota('int32', shift.shape, 0) == 0, input, shift)

    stages_in = select_state_or_input(stages_in, shift)
    return stages_in

def get_new_loop_state(output, old_state_io, old_circ_storage, old_circ_storage_mover, loop_iteration):
    '''
      Update the various buffers given the output of the most recent iteration
      * Rotates state_io up by 1 (replace last element with last stage output)
      * Shift down by 1 (of the output)
      * Rotate circ_storage down by 1, 
      * circ mover gets FULL? output -- I think it should only need the last stage of output
    '''
    
    # Rotate shift right by 1 (rotation of the output, not the previous shift)
    def _rotate_right(output_in):
      # Use lax.slice to avoid generating a gather.
      last = jax.lax.slice_in_dim(output_in, args.n_stages - 1, args.n_stages, axis=0)
      except_last = jax.lax.slice_in_dim(output_in, 0, args.n_stages - 1, axis=0)
      return jnp.concatenate([last, except_last], axis=0)
    new_shift = _rotate_right(output)

    # Rotate circular storage right by 1, inserting latest output at specified index
    def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
        rotated = _rotate_right(circ_storage_mover_in)
        rotated = jnp.expand_dims(rotated, 1)
        # The offset is the last stage's last microbatch ID. 
        offset = (loop_iteration - (args.n_stages - 1) - 1) % args.n_microbatches # # we need extar -1 b/c grabbing from un-updated circ_storage_mover
        return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=1)
    new_circ_storage = _rotate_right_and_update(old_circ_storage_mover, old_circ_storage)
    new_circ_storage_mover = output

    # Rotate stream_io left (up by 1), replacing the bottom with the latest output
    stream_buf_idx = loop_iteration % (args.n_microbatches // args.n_stages)
    stream_slice = old_state_io[:, stream_buf_idx]
    def _update_state_io(state_in, stream_slice, output):
        # Shift the current slice to the left, then fill the last stage with
        # the final output.
        padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
        stream_slice = jax.lax.slice_in_dim(
            jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
        stream_slice = jnp.where(
            jax.lax.broadcasted_iota('int32', stream_slice.shape, 0) == args.n_stages - 1, output,
            stream_slice)
        stream_slice = jnp.expand_dims(stream_slice, 1)
        return jax.lax.dynamic_update_slice_in_dim(
            state_in, stream_slice, stream_buf_idx, axis=1)
    new_state = _update_state_io(old_state_io, stream_slice, output)

    return new_state, new_shift, new_circ_storage, new_circ_storage_mover

def stage(weights, x):
  x = layer(weights, x) # To support multiple layers per stage we could add a for loop here instead of one layer
  return x

def layer(weights, stages_in):
    outputs = jnp.einsum('bse,eh->bsh',x,weights) # The leading stage dimensions of weights and x is missing because it is vmapped out
    outputs = jnp.tanh(outputs)
    return outputs

def get_weights_stage(weights, loop_iteration):
    microbatch_ids = jnp.maximum(loop_iteration - jnp.arange(args.n_stages), 0) # not a great name, really this is like batch_id * repeat idx
    repeat_ids = microbatch_ids // args.n_microbatches
    layer_ids = jnp.arange(args.n_stages) + repeat_ids * args.n_stages
    # layer_idx actauly goes out of bounds on the last bubble, but jax pulls it back to last idx
    # since its the bubble we don't care that its randomly clipped to the last, but should probably change this
    # TODO: Maybe use lax.dynamic slice instead of indexing?
    to_stack = [weights[layer_ids[stage],:,:] for stage in range(args.n_stages)]
    weights_stage = jnp.concatenate(to_stack, axis=0)
    desired_shape = (args.n_stages,) + weights.shape[1:]
    weights_stage = jnp.reshape(weights_stage, desired_shape) # This reshape fleshes out singleton axes that are flattened in concatenate
    return weights_stage

def run_one_iteration(state, shift, circ_storage, circ_storage_mover, loop_iteration, weights):
   stages_in = get_iteration_inputs(loop_iteration, args.n_microbatches, args.n_stages, state, circ_storage, shift)
   weights_stage = get_weights_stage(weights, loop_iteration)
   output = jax.vmap(stage, in_axes=0, out_axes=0,
                        spmd_axis_name='stage')(weights_stage, stages_in)
   new_state_io, new_shift, new_circ_storage, new_circ_storage_mover = get_new_loop_state(output, state, circ_storage, circ_storage_mover, loop_iteration)
   return new_state_io, new_shift, new_circ_storage, new_circ_storage_mover

def permute_output_ms_dim(output):
    '''
    Although re-using the same array for both input and output is cute,
    The final outputs turn out permuted compared to the inputs. Worringly I don't see this function in praxis
    '''

    ms_size = output.shape[1]
    # More accurately land_idx = microbatches * (r - 1) + num_stages - 1 % ms, but ms | microbatches
    land_idx = (args.n_stages - 1) % ms_size # first_finish % ms_size (really first_finish - 1 is the idx we care about)
    permutation = (np.arange(ms_size) + land_idx) % ms_size
    output = output[:,permutation]
    return output

def init_states(inputs):
    # Initialize components of state: state_io, shift, circular_storage and circular_storage_mover
    # shift is [n_stages, micro_size, sequence, embed]
    shift = jnp.zeros((args.n_stages,) + inputs.shape[1:]) # equivalently inputs.shape[1:] is microshape

    # state_io is [n_stages, microbatches/stages, micro_size, sequence, embed]
    state_io = jnp.reshape(inputs, (args.n_stages, args.n_microbatches // args.n_stages) + inputs.shape[1:])

    # circ_stoage is [num_stages, microbatches, micro_size, sequence, embed]
    circ_storage = jnp.zeros((args.n_stages,) + inputs.shape ) # This is huge, is this size really what is in the pax code, and do we need this large?

    # circ_storage_mover is same as shift: [n_stages, micro_size, sequence, embed]
    circ_storage_mover = shift
    return state_io, shift, circ_storage, circ_storage_mover

def run_pipeline(weights, inputs):
    state_io, shift, circ_storage, circ_storage_mover = init_states(inputs)

    #total_iterations = microbatches + num_repeat * num_stages  - 1
    total_iterations = args.n_microbatches * args.num_repeat + args.n_stages  - 1 # What? Shoulnd't this be num_stages * num_repeat + micro - 1
    #breakpoint()
    for loop_iteration in range(total_iterations):
       my_print(f"Starting loop {loop_iteration}")
       my_print(f"shift:{jnp.ravel(shift)}")
       #my_print(f"state: {jnp.ravel(state)}")
       if yes_print:
        ss = jnp.reshape(state, [4,2])
        my_print(f"ss: {ss}")
        ras = jnp.reshape(circ_storage, [4,8])
        my_print(f" as: {ras}")
        my_print(f"circ_storage_mover: {jnp.ravel(circ_storage_mover)}")
       state_io, shift, circ_storage, circ_storage_mover = run_one_iteration(state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights)

    my_print("Final output")
    my_print(f"shift:{jnp.ravel(shift)}")
    if yes_print:
        my_print(f"state: {jnp.reshape(jnp.ravel(state),[4,2])}")
    # reshape state to match input shape
    #state = jnp.transpose(state, axes=(0,2,1,3,4)) # holy crap
    #qqq = jnp.transpose(state, axes=(2,3,4,1,0))
    final_output = permute_output_ms_dim(state_io)

    final_output = jnp.reshape(final_output, (args.n_microbatches,) + state_io.shape[2:])
    return final_output


######################     Begin main      #################


def get_weights_random():
    # Assuming layer_i looks like output = inputs[micro_id,:,:,:] * weights[i,:,:] --> x_out = jnp.einsum('bse,eh->bsh',x,weights)
    weights_random_shape = jnp.array([args.n_stages * args.num_repeat, args.features, args.features]) # more realistic:  layers x embed x hidden, etc
    k = jax.random.PRNGKey(1)
    return jax.random.normal(k,weights_random_shape, dtype=jnp.float32)

def get_weights_debug_unique():
    weights = list()
    weights_debug_shape = [args.n_stages * num_repeat, args.microbatch_size, args.features, args.features]
    for i in range(jnp.prod(weights_debug_shape)):
       weights.append((i+1) * 10**(i+1))
    weights = jnp.array(weights, dtype=jnp.float32)
    weights = jnp.reshape(weights, weights_debug_shape)
    return weights

def get_inputs_random():
    micro_shape = [args.microbatch_size, args.sequence, args.features] # realistic
    test_input_shape = [args.n_microbatches] + micro_shape # [microbatches, microbatch_size, seq_len, model_dim]
    k = jax.random.PRNGKey(2)
    return jax.random.normal(k,test_input_shape, dtype=jnp.float32)

def get_weights_debug():
   # Assuming layer_i looks like output = inputs[micro_id,:,:,:] + weights[i,:,:,:] --> x_out = x_out = inputs + weights
    weights_debug_shape = [args.n_stages * args.num_repeat, args.microbatch_size, args.features, args.features]
    return 100 + jnp.zeros(weights_debug_shape, dtype=jnp.float32)
   
def get_inputs_debug():
    test_inputs_shape = jnp.array([args.n_microbatches] + micro_shape)
    test_inputs = jnp.reshape(jnp.arange(jnp.prod(test_inputs_shape), dtype=jnp.float32), test_inputs_shape)
   

if 0:
    # Sizes
    num_stages = 4
    microbatches = 8
    microbatch_size = 5
    seq_len = 2048
    model_dim = 2560
    total_batch = microbatches * microbatch_size
    num_repeat = 3

    yes_print = False
    sum_layer = False

    micro_shape = [microbatch_size, seq_len, model_dim] # realistic
    #micro_shape = [microbatch_size] # great for debugging state transformations
    #micro_shape = [microbatch_size, model_dim] # middle ground for debugging running with weights

    k = jax.random.PRNGKey(1)

    test_inputs = get_inputs_random()
    weights = get_weights_random()

    # Configure sharding
    pipeline_axis = 4
    dp_axis = 1
    devices = mesh_utils.create_device_mesh((pipeline_axis, dp_axis))
    mesh = Mesh(devices, axis_names=('stage', 'data'))

    weight_sharding = S('stage', None, None) # weight sharded over stage
    input_sharding = S('data', None, None, None)   # inputs sharded over batch
    result_sharding = S('data', None, None, None)  # output sharded over batch

    #weights = jax.device_put(weights, weight_sharding)
    #jax.debug.visualize_array_sharding(weights)


####### Start testing ###########

# Test get_weights_stage
if 0:
    ws = get_weights_stage(weights, 0)


# Test run_one_iteration
# Initialize shift and state
if 0:
    shift = jnp.zeros((args.n_stages,) + test_inputs.shape[1:]) # equivalently inputs.shape[1:] is microshape
    state = jnp.reshape(test_inputs, (args.n_stages, args.n_microbatches // args.n_stages) + test_inputs.shape[1:])
    new_state, new_shift = run_one_iteration(state, shift, 0, weights)


# Test get_iteration input + select_state = stages_in
if 0:
    loop_iteration = 0
    state, shift, circ_storage, circ_storage_mover = init_states(1.0 + test_inputs)
    stages_in = get_iteration_inputs(loop_iteration, args.n_microbatches, args.n_stages, state, circ_storage)
    stages_in = select_state_or_input(stages_in, shift)


# Test get_new_loop_state
if 0:
    loop_iteration = 0
    state, shift, circ_storage, circ_storage_mover = init_states(1.0 + test_inputs)
    output = shift
    new_state, new_shift, new_circ_storage, new_circ_storage_mover = get_new_loop_state(output, state, circ_storage, circ_storage_mover, loop_iteration)
    assert new_state.shape == state.shape
    assert new_shift.shape == shift.shape
    assert new_circ_storage.shape == circ_storage.shape
    assert new_circ_storage_mover.shape == circ_storage_mover.shape

# Test get_weights_stage
if 0:
   weights = jnp.reshape(jnp.arange(8),weights_shape)
   for loop_iteration in range(19):
      weights_stage = get_weights_stage(weights, loop_iteration)
      print(f"iter {loop_iteration}: weights {jnp.ravel(weights_stage)}")

# Test run pipeline (no jit)
if 0:

    weights = get_weights_debug()
    inputs = get_inputs_debug()
    print(f"weights: {jnp.ravel(weights)}")
    print(f"inputs: {jnp.ravel(weights)}")

    outputs = run_pipeline(weights, test_inputs)
    #print(f"{outputs=}")

# Test jitted E2E
def rawr():
    if 1:
        devices = mesh_utils.create_device_mesh((args.n_stages, 1))
        mesh = Mesh(devices, axis_names=('stage', 'data'))
        weight_sharding = S(mesh,'stage', None, None) # weight sharded over stage
        input_sharding = S(mesh,'data', None, None, None)   # inputs sharded over batch
        result_sharding = S(mesh,'data', None, None, None)  # output sharded over batch

        weights = get_weights_random()
        test_inputs = get_inputs_random()

        output_jit = jax.jit(run_pipeline,
                    in_shardings=((weight_sharding, input_sharding)),
                    out_shardings=result_sharding)

        output_pipeline = output_jit(weights, test_inputs)
        # [Microbatch, microsize, seq embed] -> [Batch, Seq, Embed]
        output_pipeline= jnp.reshape(output_pipeline, (args.batch_size,) + output_pipeline.shape[2:])

        def reg_layer(weights, input):
            for layer_idx in range(weights.shape[0]):
                input = layer(weights[layer_idx,:,:], input)
            return input

        # Reshape batched_inputs from [micro,micro_size,...] to [batch,...]
        batched_inputs = jnp.reshape(test_inputs, (args.batch_size,) + test_inputs.shape[2:])
        regular_output = reg_layer(weights, batched_inputs)

        diff_norm = jnp.linalg.norm(output_pipeline - regular_output)
        print(f"{diff_norm=}")

        regular_norm = jnp.linalg.norm(regular_output)
        print(f"{regular_norm=}")

        output_pipeline_norm = jnp.linalg.norm(output_pipeline)
        print(f"{output_pipeline_norm=}")

        yes_print = False
        my_print(f"regular {jnp.ravel(regular_output)}")
        my_print(f"pipeline {jnp.ravel(regular_output)}")

# Test 1 stage of vmap
if 0:
    sum_layer = False
    weights = get_weights_random()
    inputs = get_inputs_random()
    print(f"{weights=}")
    print(f"{inputs=}")

    batched_inputs = jnp.reshape(inputs, (args.batch_size,) + test_inputs.shape[2:])
    layer_1_reg = layer(weights[0,:,:], batched_inputs)
    #print(f"{layer_1_reg=}")

    loop_iteration = 0
    state, shift, circ_storage, circ_storage_mover = init_states(inputs)
    stages_in = get_iteration_inputs(loop_iteration, args.n_microbatches, args.n_stages, state, circ_storage)
    stages_in = select_state_or_input(stages_in, shift)
    #my_print(f"Stages in: {jnp.ravel(stages_in)}")
    weights_stage = get_weights_stage(weights, loop_iteration)
    pipeline_output = jax.vmap(stage, in_axes=0, out_axes=0,
                        spmd_axis_name='stage')(weights_stage, stages_in)
    print(f"{pipeline_output=}")

    
    batched_state = batched_inputs
    for layer_idx in range(weights.shape[0]):
       batched_state = layer(weights[layer_idx,:,:], batched_state)
       print(f"layer {layer_idx}, batched_state: {jnp.ravel(batched_state)}")


    total_iterations = args.n_microbatches * args.num_repeat + args.n_stages
    for loop_iteration in range(total_iterations):
       my_print(f"Starting loop {loop_iteration}")
       my_print(f"shift:{jnp.ravel(shift)}")
       #my_print(f"state: {jnp.ravel(state)}")
       if yes_print:
        ss = jnp.reshape(state, [4,2])
        my_print(f"ss: {ss}")
        ras = jnp.reshape(circ_storage, [4,8])
        my_print(f" as: {ras}")
        my_print(f"circ_storage_mover: {jnp.ravel(circ_storage_mover)}")
       state, shift, circ_storage, circ_storage_mover = run_one_iteration(state, shift, circ_storage, circ_storage_mover, loop_iteration, weights)

    
       

if 0:
    import timing_util
    weights = get_weights_random()
    inputs = get_inputs_random()
    print(f"weights: {jnp.ravel(weights)}")
    print(f"inputs: {jnp.ravel(inputs)}")

    timing_util.simple_timeit(run_pipeline, weights, test_inputs, task = "run_pipeline")
    #outputs = run_pipeline(weights, test_inputs)
    #print(f"{outputs=}")


def main() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    parser = argparse.ArgumentParser(description='Pipeline Parallelism Options')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_stages', type=int, default=4)
    parser.add_argument('--n_microbatches', type=int, default=4)
    parser.add_argument('--pipeline_axis', type=int, default=4)
    parser.add_argument('--dp_axis', type=int, default=1)
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--sequence', type=int, default=2048)
    parser.add_argument('--num_repeat', type=int, default=1)

    global args
    args = parser.parse_args()
    args.microbatch_size = args.batch_size // args.n_microbatches

    global yes_print
    yes_print=False
    # Necessary artifacts for the good stuff
    #pipeline_func = get_pipelint_jit()
    #weights, inputs, targets = get_weights_and_inputs()

    rawr()


    #assert_same_output_and_grad(reg_matmuls, pipeline_func, targets, weights, inputs)

    #timing_util.simple_timeit(pipeline_func, weights, inputs, tries = 3, task = 'basic_pp')



if __name__ == "__main__":
  main()