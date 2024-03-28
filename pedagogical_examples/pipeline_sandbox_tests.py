import pipeline_sandbox
import os
import argparse

def test_get_weights(weights):
    ws = pipeline_sandbox.get_weights_stage(weights, 0)



def main() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    parser = argparse.ArgumentParser(description='Pipeline Parallelism Options')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_stages', type=int, default=4)
    parser.add_argument('--n_microbatches', type=int, default=8)
    parser.add_argument('--pipeline_axis', type=int, default=4)
    parser.add_argument('--dp_axis', type=int, default=1)
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--sequence', type=int, default=16)
    parser.add_argument('--num_repeat', type=int, default=2)

    global args
    args = parser.parse_args()
    args.microbatch_size = args.batch_size // args.n_microbatches

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs()

    test_get_weights(weights)

if __name__ == "__main__":
  main()




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