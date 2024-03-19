import jax

from jax import numpy as jnp
import numpy as np

microbatches = 8
microbatch_size = 9
seq_len = 7
model_dim = 3

test_inputs = np.ones([microbatches, microbatch_size, seq_len, model_dim])

num_stages = 4

# Initialize shift and state
shift = jnp.zeros([num_stages, microbatch_size, seq_len, model_dim])
state = jnp.reshape(test_inputs, (num_stages, microbatches // num_stages) + test_inputs.shape[1:])

# Construct stages in
def get_iteration_inputs(loop_iteration, microbatches, num_stages, state):
    stream_buf_idx = loop_iteration % (microbatches // num_stages)
    return state[:,stream_buf_idx] # equivalent to state[:,stream_buf_idx,:,:]

def select_state_or_input(input, state):
    # Selects input for stage 0, state for other stages
    return jnp.where(jax.lax.broadcasted_iota('int32', state.shape, 0) == 0, input, state)


loop_iteration = 0
stages_in = get_iteration_inputs(loop_iteration, microbatches, num_stages, state)
stages_in = select_state_or_input(stages_in, shift)



