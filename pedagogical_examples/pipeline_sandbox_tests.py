import pipeline_sandbox
import os
import argparse
import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec

def S(mesh, *specs):
    return NamedSharding(mesh, PartitionSpec(*specs))

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
    dp_axis = 1

    
    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    use_circ_storage = True
    mesh = pipeline_sandbox.create_mesh(n_stages, dp_axis)
    state_io, shift, circ_storage, circ_storage_mover = pipeline_sandbox.init_states(inputs, n_stages, use_circ_storage, mesh)

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
    dp_axis = 1
    use_circ_storage = True

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    mesh = pipeline_sandbox.create_mesh(n_stages, dp_axis)
    state_io, shift, circ_storage, circ_storage_mover = pipeline_sandbox.init_states(inputs, n_stages, use_circ_storage, mesh)

    stages_in = pipeline_sandbox.get_iteration_inputs(loop_iteration, n_microbatches, n_stages, state_io, circ_storage, shift, use_circ_storage)


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
    dp_axis = 1
    use_circ_storage = True

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    mesh = pipeline_sandbox.create_mesh(n_stages, dp_axis)
    state_io, shift, circ_storage, circ_storage_mover = pipeline_sandbox.init_states(inputs, n_stages, use_circ_storage, mesh)

    output = 100.0 + jnp.zeros(shape=shift.shape, dtype=shift.dtype) # Create fake outputs
    circ_storage_mover = output


    # def shard_dim_by_stages(x, mesh):
    #     '''Assumes the stages dimension is leading and the mesh has name stages.'''
    #     specs = ['stage'] + [None] * (x.ndim - 1)
    #     stage_sharding = S(mesh, *specs)
    #output = jax.lax.with_sharding_constraint(output, pipeline_sandbox.S(mesh,None,None,None,None))
    #breakpoint()



    # Unsure if these are actually sharded like this - we only specify state_io, but the others are derived
    # via shift which is also specified like this
    output_sharding = S(mesh, 'stage', None, None, None)
    state_io_sharding = S(mesh, 'stage', None, None, None)
    circ_storage_sharding = S(mesh, 'stage', None, None, None, None)
    circ_storage_mover_sharding = S(mesh, 'stage', None, None, None)

    # Input: output, state_io, circ_storage, circ_storage_mover, loop_iteration, use_circ_storage
    # Output: new_state, new_shift, new_circ_storage, new_circ_storage_mover
    jit_new_loop_state = jax.jit(
        pipeline_sandbox.get_new_loop_state, 
        in_shardings=((output_sharding, state_io_sharding, circ_storage_sharding, circ_storage_mover_sharding)),
        out_shardings=((output_sharding, state_io_sharding, circ_storage_sharding, circ_storage_mover_sharding)),
        static_argnums=[4,5]
    )
    pnew_state, new_shift, new_circ_storage, new_circ_storage_mover = jit_new_loop_state(output, state_io, circ_storage, circ_storage_mover, loop_iteration, use_circ_storage)

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
    dp_axis = 1
    use_circ_storage = True

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    mesh = pipeline_sandbox.create_mesh(n_stages, dp_axis)
    state_io, shift, circ_storage, circ_storage_mover = pipeline_sandbox.init_states(inputs, n_stages, use_circ_storage, mesh)

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
    use_circ_storage = True

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    #inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))

    dp_axis = 1
    mesh = pipeline_sandbox.create_mesh(n_stages, dp_axis)

    final_output = pipeline_sandbox.run_pipeline(weights, inputs, n_stages, n_microbatches, n_repeat, use_circ_storage, mesh)

def test_jit_pipeline():
    n_stages = 4
    n_repeat = 2
    n_layers = n_stages * n_repeat
    sequence = 8
    features = 6
    batch_size = 24
    n_microbatches = 8
    microbatch_size = batch_size // n_microbatches
    use_circ_storage = True

    weights, inputs, targets = pipeline_sandbox.get_weights_and_inputs(batch_size, sequence, features, n_layers)
    #inputs = inputs.reshape((n_microbatches, microbatch_size, sequence, features))
    
    dp_axis = 1
    mesh = pipeline_sandbox.create_mesh(n_stages, dp_axis)
    pipeline_jit = pipeline_sandbox.get_pipelint_jit(n_stages, dp_axis, mesh)

    #pipeline_sandbox.run_pipeline(weights, inputs, n_stages, n_microbatches, n_repeat, use_circ_storage, mesh)
    output = pipeline_jit(weights, inputs, n_stages, n_microbatches, n_repeat, use_circ_storage, mesh)

def main() -> None:

    test_get_weights()

    test_get_init_states()

    test_get_iteration_inputs()

    test_get_new_loop_state()

    #test_run_one_iteration()

    #test_run_pipeline()

    test_jit_pipeline()

if __name__ == "__main__":
  main()

