import pipeline_sandbox
import os
import argparse
import jax
from jax import numpy as jnp

def test_get_weights(weights, n_stages, n_microbatches):
    ws = pipeline_sandbox.get_weights_stage(weights, 0, n_stages, n_microbatches)
    assert jnp.allclose(weights[0,:,:],ws[0,:,:])

    # after one loop through the num_batches=8, the stages should be one repeat (num_stages=4) higher
    ws = pipeline_sandbox.get_weights_stage(weights, 11, n_stages, n_microbatches)
    assert jnp.allclose(weights[5,:,:],ws[1,:,:])

def test_run_one_iteration(state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights):
    new_state_io, new_shift, new_circ_storage, new_circ_storage_mover = pipeline_sandbox.run_one_iteration(state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights)


def main() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    parser = argparse.ArgumentParser(description='Pipeline Parallelism Options')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--n_stages', type=int, default=4)
    parser.add_argument('--n_microbatches', type=int, default=8)
    parser.add_argument('--dp_axis', type=int, default=1)
    parser.add_argument('--features', type=int, default=16)
    parser.add_argument('--sequence', type=int, default=16)
    parser.add_argument('--n_repeat', type=int, default=2)

    args = parser.parse_args()
    args.microbatch_size = args.batch_size // args.n_microbatches
    args.layers = args.n_stages * args.n_repeat

    # Necessary artifacts for the fun stuff
    pipeline_func = pipeline_sandbox.get_pipelint_jit(args.n_stages, args.dp_axis)
    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(args.batch_size, args.sequence, args.features, args.layers)
    inputs = inputs.reshape((args.n_microbatches, args.microbatch_size, args.sequence, args.features))
    state_io, shift, circ_storage, circ_storage_mover = pipeline_sandbox.init_states(inputs, args.n_stages)

    test_get_weights(weights, args.n_stages, args.n_microbatches)

    loop_iteration = 0
    test_run_one_iteration(state_io, shift, circ_storage, circ_storage_mover, loop_iteration, weights)



if __name__ == "__main__":
  main()







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
