import pipeline_sandbox
import os
import argparse
import jax
from jax import numpy as jnp


def test_get_weights():
    n_stages = 4
    n_repeat = 2
    n_layers = n_stages * n_repeat
    sequence = 8
    features = 6
    batch_size = 24
    n_microbatches = 8

    weights, _, _ = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)

    ws = pipeline_sandbox.get_weights_stage(weights, 0, n_stages, n_microbatches)
    assert jnp.allclose(weights[0,:,:],ws[0,:,:])

    # after one loop through the num_batches=8, the stages should be one repeat (num_stages=4) higher
    ws = pipeline_sandbox.get_weights_stage(weights, 11, n_stages, n_microbatches)
    assert jnp.allclose(weights[5,:,:],ws[1,:,:])

def test_get_init_states():
    n_stages = 4
    n_repeat = 2
    n_layers = n_stages * n_repeat
    sequence = 8
    features = 6
    batch_size = 24
    n_microbatches = 8
    microbatch_size = batch_size // n_microbatches
    loop_iteration = 0

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    state_io, shift, circ_storage, circ_storage_mover = pipeline_sandbox.init_states(inputs, n_stages)

def test_get_iteration_inputs():
    n_stages = 4
    n_repeat = 2
    n_layers = n_stages * n_repeat
    sequence = 8
    features = 6
    batch_size = 24
    n_microbatches = 8
    microbatch_size = batch_size // n_microbatches
    loop_iteration = 0

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    state_io, shift, circ_storage, circ_storage_mover = pipeline_sandbox.init_states(inputs, n_stages)

    stages_in = pipeline_sandbox.get_iteration_inputs(loop_iteration, n_microbatches, n_stages, state_io, circ_storage, shift)


def test_get_new_loop_state():
    n_stages = 4
    n_repeat = 2
    n_layers = n_stages * n_repeat
    sequence = 8
    features = 6
    batch_size = 24
    n_microbatches = 8
    microbatch_size = batch_size // n_microbatches
    loop_iteration = 0

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    state_io, shift, circ_storage, circ_storage_mover = pipeline_sandbox.init_states(inputs, n_stages)

    output = 100.0 + shift # Create fake outputs
    pnew_state, new_shift, new_circ_storage, new_circ_storage_mover = pipeline_sandbox.get_new_loop_state(output, state_io, circ_storage, circ_storage_mover, loop_iteration)

def test_run_one_iteration():
    n_stages = 4
    n_repeat = 2
    n_layers = n_stages * n_repeat
    sequence = 8
    features = 6
    batch_size = 24
    n_microbatches = 8
    microbatch_size = batch_size // n_microbatches
    loop_iteration = 0

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    state_io, shift, circ_storage, circ_storage_mover = pipeline_sandbox.init_states(inputs, n_stages)

    new_state_io, new_shift, new_circ_storage, new_circ_storage_mover = pipeline_sandbox.run_one_iteration(state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights)

def test_run_pipeline():
    n_stages = 4
    n_repeat = 2
    n_layers = n_stages * n_repeat
    sequence = 8
    features = 6
    batch_size = 24
    n_microbatches = 8
    microbatch_size = batch_size // n_microbatches

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    #inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))

    final_output = pipeline_sandbox.run_pipeline(weights, inputs, n_stages, n_microbatches, n_repeat)

def test_jit_pipeline():
    n_stages = 4
    n_repeat = 2
    n_layers = n_stages * n_repeat
    sequence = 8
    features = 6
    batch_size = 24
    n_microbatches = 8
    microbatch_size = batch_size // n_microbatches

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    #inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    
    dp_axis = 1
    pipeline_jit = pipeline_sandbox.get_pipelint_jit(n_stages, dp_axis)

    output = pipeline_jit(weights, inputs, n_stages, n_microbatches, n_repeat)

def main() -> None:

    test_get_weights()

    test_get_init_states()

    test_get_iteration_inputs()

    test_get_new_loop_state()

    test_run_one_iteration()

    test_run_pipeline()

    test_jit_pipeline()



if __name__ == "__main__":
  main()





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
