import jax

from jax import numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils

def S(*specs):
  return NamedSharding(mesh, PartitionSpec(*specs))

# Initialize shift and state
# shift = jnp.zeros([num_stages] + micro_shape)
# state = jnp.reshape(test_inputs, (num_stages, microbatches // num_stages) + test_inputs.shape[1:])

# Construct stages in
def get_iteration_inputs(loop_iteration, microbatches, num_stages, state):
    stream_buf_idx = loop_iteration % (microbatches // num_stages)
    return state[:,stream_buf_idx] # equivalent to state[:,stream_buf_idx,:,:]

def select_state_or_input(input, state):
    # Selects input for stage 0, state for other stages
    return jnp.where(jax.lax.broadcasted_iota('int32', state.shape, 0) == 0, input, state)

# run model
def get_new_loop_state(output, state, shift, loop_iteration):
    # Shift state to the right by 1.
    
    def _shift_right(shift_in):
        padding = [[1, 0]] + [[0, 0]] * (shift_in.ndim - 1)
        # Use lax.slice to guarantee the gradient is a pad.
        return jax.lax.slice(jnp.pad(shift_in, padding), [0] * shift_in.ndim, shift_in.shape)
    shift = _shift_right(output)

    stream_buf_idx = loop_iteration % (microbatches // num_stages)
    stream_slice = state[:, stream_buf_idx]

    def _update_state(state, stream_slice, output):
        # Shift the current slice to the left, then fill the last stage with
        # the final output.
        padding = [[0, 1]] + [[0, 0]] * (stream_slice.ndim - 1)
        stream_slice = jax.lax.slice_in_dim(
            jnp.pad(stream_slice, padding), 1, stream_slice.shape[0] + 1, axis=0)
        stream_slice = jnp.where(
            jax.lax.broadcasted_iota('int32', stream_slice.shape, 0) == num_stages - 1, output,
            stream_slice)
        stream_slice = jnp.expand_dims(stream_slice, 1)
        return jax.lax.dynamic_update_slice_in_dim(
            state, stream_slice, stream_buf_idx, axis=1)
    state = _update_state(state, stream_slice, output)
    return state, shift

def stage(weights, x):
  #for i in range(weights.shape[0]): # this was used if each stage had multiple layers (we would need to reshape weights to stages, layers/stage, embed, embed)
  x = layer(weights, x)
  return x

def layer(weights, x):
  #breakpoint()
  x_out = jnp.einsum('bse,eh->bsh',x,weights) # The leading stage dimension is missing because it is vmapped out
  #x_out = jnp.tanh(x_out)
  #x = jnp.tanh(jnp.dot(x, w))
  return x_out

def run_one_iteration(state, shift, loop_iteration, weights):
   stages_in = get_iteration_inputs(loop_iteration, microbatches, num_stages, state)
   stages_in = select_state_or_input(stages_in, shift)
   output = jax.vmap(stage, in_axes=0, out_axes=0,
                        spmd_axis_name='stage')(weights, stages_in)
   new_state, new_shift = get_new_loop_state(output, state, shift, loop_iteration)
   return new_state, new_shift

def permute_ms_dim(state):
    # This is for real?
    ms_size = state.shape[1]
    land_idx = (num_stages - 1) % ms_size
    permutation = (np.arange(ms_size) + land_idx) % ms_size
    state = state[:,permutation]
    return state

def run_pipeline(weights, inputs):
    

    # Initialize shift and state
    shift = jnp.zeros((num_stages,) + inputs.shape[1:]) # equivalently inputs.shape[1:] is microshape
    state = jnp.reshape(inputs, (num_stages, microbatches // num_stages) + inputs.shape[1:])

    total_iterations = microbatches + num_stages - 1
    for loop_iteration in range(total_iterations):
       state, shift = run_one_iteration(state, shift,loop_iteration, weights)

    # reshape state to match input shape
    #state = jnp.transpose(state, axes=(0,2,1,3,4)) # holy crap
    #qqq = jnp.transpose(state, axes=(2,3,4,1,0))
    #breakpoint()
    state_perm = permute_ms_dim(state)
    #breakpoint()
    state = jnp.reshape(state_perm, (microbatches,) + state.shape[2:])
    return state # this can be reshaped to match input at some point


######################     Begin main      #################

# Sizes
num_stages = 4
microbatches = 36
microbatch_size = 1
seq_len = 1
model_dim = 1
total_batch = microbatches * microbatch_size

micro_shape = [microbatch_size, seq_len, model_dim] # realistic
#micro_shape = [microbatch_size] # great for debugging state transformations
#micro_shape = [microbatch_size, model_dim] # middle ground for debugging running with weights

test_inputs = np.ones([microbatches] + micro_shape, dtype=jnp.float32)
test_inputs_shape = jnp.array([microbatches] + micro_shape)
test_inputs = jnp.reshape(jnp.arange(jnp.prod(test_inputs_shape), dtype=jnp.float32), test_inputs_shape)

weights_shape = jnp.array([num_stages, model_dim, model_dim]) # ideally  layers x embed x hidden,
weights = jnp.ones(weights_shape, dtype=jnp.float32)

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

# Test run_one_iteration
# Initialize shift and state
# shift = jnp.zeros((num_stages,) + test_inputs.shape[1:]) # equivalently inputs.shape[1:] is microshape
# state = jnp.reshape(test_inputs, (num_stages, microbatches // num_stages) + test_inputs.shape[1:])
# new_state, new_shift = run_one_iteration(state, shift, 0, weights)


# Test run_pipeline unjitted
# outputs = run_pipeline(weights, test_inputs)
# print(f"{outputs=}")

# Test jit run_pipeline
# initialize
#test_inputs = np.ones([microbatches] + micro_shape, dtype=jnp.float32)
#test_inputs_shape = jnp.array([microbatches] + micro_shape)
k = jax.random.PRNGKey(1)
test_inputs = jax.random.normal(k, test_inputs_shape, dtype=jnp.float32)

#weights_shape = jnp.array([num_stages, model_dim, model_dim]) # ideally  layers x embed x hidden,
weights = jax.random.normal(k,weights_shape, dtype=jnp.float32)


output_jit = jax.jit(run_pipeline,
             in_shardings=((weight_sharding, input_sharding)),
             out_shardings=result_sharding)

output_pipeline = output_jit(weights, test_inputs)
# [Microbatch, microsize, seq embed] -> [Batch, Seq, Embed]
output_pipeline= jnp.reshape(output_pipeline, (total_batch,) + output_pipeline.shape[2:])

def reg_matmuls(weights, input):
  for layer_idx in range(num_stages):
    input = layer(weights[layer_idx,:,:], input)
  return input

# Reshape batched_inputs from [micro,micro_size,...] to [batch,...]
batched_inputs = jnp.reshape(test_inputs, (total_batch,) + test_inputs.shape[2:])
regular_output = reg_matmuls(weights, batched_inputs)

diff_norm = jnp.linalg.norm(output_pipeline - regular_output)
print(f"{diff_norm=}")

regular_norm = jnp.linalg.norm(regular_output)
print(f"{regular_norm=}")

output_pipeline_norm = jnp.linalg.norm(output_pipeline)
print(f"{output_pipeline_norm=}")

#print(f"{output_pipeline=}")
#print(f"{regular_output=}")

#breakpoint()



# new_state, new_shift = run_one_iteration(state, shift, 0)

# Test getting stages_in
# loop_iteration = 0
# stages_in = get_iteration_inputs(loop_iteration, microbatches, num_stages, state)
# stages_in = select_state_or_input(stages_in, shift)

# Test get_new_loop_state
# shift = jnp.zeros((num_stages,) + test_inputs.shape[1:],dtype=jnp.float32) # equivalently inputs.shape[1:] is microshape
# state = jnp.reshape(test_inputs, (num_stages, microbatches // num_stages) + test_inputs.shape[1:])
# test_outputs_shape = jnp.array([num_stages] + micro_shape)
# test_outputs = 100 + jnp.reshape(jnp.arange(jnp.prod(test_outputs_shape),dtype=jnp.float32), test_outputs_shape)
# #breakpoint()
# new_state, new_shift = get_new_loop_state(test_outputs, state, shift, 0)




# for i in range(n_microbatches + n_stages - 1):
#     vmap_in = get_vmap_in(state)
#     update_state(state)
#     state = shift_right_and_insert_input(state, x[i])
#     state = jax.vmap(stage, in_axes=0, out_axes=0,
#                         spmd_axis_name='stage')(w, state)

