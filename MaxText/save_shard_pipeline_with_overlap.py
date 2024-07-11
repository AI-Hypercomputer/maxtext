import jax
from jax import numpy as jnp
from functools import partial

from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils

import timing_util

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

global mesh
mesh = Mesh(jax.devices(), ('stages',))

def predict(params, inputs):
  for layer in params:
    inputs = jnp.dot(inputs, layer)
    inputs = jax.nn.relu(inputs)
  return inputs

def loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  print(f"{predictions=}")
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))

def init_layer(key, embed_size):
    W = jax.random.normal(key, (embed_size, embed_size)) / jnp.sqrt(embed_size)
    return W

def init(key_init, num_layers, embed_size, batch_size):
    keys = jax.random.split(key_init, num_layers)
    params = [init_layer(key, embed_size) for key in keys]

    input_key, target_key = jax.random.split(key_init, 2)
    inputs = jax.random.normal(input_key, (batch_size, embed_size))
    targets = jax.random.normal(target_key, (batch_size, embed_size))

    return params, (inputs, targets)

def stage_fn(layer, inputs):
  inputs = jnp.dot(inputs, layer)
  return jax.nn.relu(inputs)

def predict_pp(params, inputs):
  if args.use_overlapped:
    outputs = spmd_pipeline_overlapped(stage_fn, params, inputs)
  else:
    outputs = spmd_pipeline(stage_fn, params, inputs)
  return outputs

@partial(shard_map, mesh=mesh, in_specs=((P('stages')), P('stages')),
         out_specs=P())
def loss_pp(params, batch):
  inputs, targets = batch
  predictions = predict_pp(params, inputs.reshape(args.microbatches_per_stage, args.microbatch_size, -1)).reshape(args.microbatches_per_stage * args.microbatch_size, -1)
  local_loss = jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))
  return jax.lax.pmean(local_loss, 'stages')

def get_real_permute_pairs(loop_iteration, num_stages):
  if loop_iteration >= num_stages:
    # If pipeline is full, loop through all
    return [(i, (i+1) % num_stages) for i in range(num_stages)]
  else:
    return [(i, (i+1) % num_stages) for i in range(loop_iteration + 1)]

# overlapped
# non-overallped
def spmd_pipeline_overlapped(fn, stage_params, inputs):
  stage = jax.lax.axis_index('stages')
  outputs = jnp.zeros_like(inputs) * jnp.nan
  #state = jnp.zeros((args.microbatch_size, args.embed_size)) * jnp.nan
  current_input = jnp.zeros((args.microbatch_size, args.embed_size)) * jnp.nan
  prev_output = jnp.zeros((args.microbatch_size, args.embed_size)) * jnp.nan



  num_total_iterations = args.num_microbatches * args.num_repeats + 2 * (args.num_stages - 1) # double bubble
  # TODO (this should be re-written into its own method so it can be scanned, inputs including circ_storage)
  for loop_iter in range(num_total_iterations):
    # Push new input into stage 0
    current_input = current_input.at[:].set(jnp.where(stage == 0, inputs[loop_iter % args.microbatches_per_stage], current_input[:]))
    inputs = shift_inputs(loop_iter, inputs)

    # compute
    # new_previous_output = compute(current_input)
    # We want to pass something of shape [embed, embed] instead, so we index away the first unit axis.
    # Fake/ poorly named argument, just repeating the same matmul per layer
    new_previous_output = current_input
    for _ in range(args.num_layers_per_stage):
      # Shard map is rank preserving, so the params have shape [1,embed,embed] inside each shard
      new_previous_output = fn(stage_params[0], new_previous_output)

    # Store outputs
    output_offset = loop_iter - 2 * (args.num_stages - 1) # regular just loop_iter - (args.num_stages - 1)
    outputs = outputs.at[output_offset % args.microbatches_per_stage].set(jnp.where(stage == args.num_stages-1, new_previous_output, outputs[output_offset % args.microbatches_per_stage]))


    # communicate (permute)
    # next_input = communicate_collective_permte(previous_output)
    # state, inputs, outputs = shift(loop_iter, state, inputs, outputs)

    # Split the 3 rotations into their own function for easier xprof code tracing
    next_input = shift_stages(loop_iter, prev_output)
    # inputs = shift_inputs(loop_iter, inputs)
    outputs = shift_outputs(loop_iter, outputs)
    #next_input, inputs, outputs = shift(loop_iter, prev_output, inputs, outputs)

    current_input, prev_output = next_input, new_previous_output

  outputs = jax.lax.ppermute(outputs, 'stages', [(i, (i+1) % args.num_stages) for i in range(args.num_stages)])
  return outputs

# non-overallped
def spmd_pipeline(fn, stage_params, inputs):
  stage = jax.lax.axis_index('stages')
  outputs = jnp.zeros_like(inputs) * jnp.nan
  state = jnp.zeros((args.microbatch_size, args.embed_size)) * jnp.nan
  # Each stage has their own circ_storage, don't need leading num_stages
  #circ_storage = jnp.zeros((args.num_stages, args.num_microbatches, args.microbatch_size, args.embed_size))
  if args.use_circ_storage:
    circ_storage_mover = jnp.zeros_like(state) * jnp.nan
    circ_storage = jnp.zeros((args.num_microbatches, args.microbatch_size, args.embed_size))
  else:
    circ_storage_mover, circ_storage = None, None
  # TODO: wrap this in a function that can be scanned
  #total_iterations = self.config.num_pipeline_microbatches * self.config.num_pipeline_repeats + self.num_stages  - 1
  num_total_iterations = args.num_microbatches * args.num_repeats + args.num_stages - 1 # regular

  loop_state = {"loop_iter": 0, "state": state, "inputs":inputs, "outputs": outputs, "circ_storage": circ_storage, "circ_storage_mover": circ_storage_mover}
  def run_iteration_scannable(loop_state, xs):
    # loop state consists of:
    # loop_iteration,
    # state,
    # circ_storage
    # circ_storage_mover
    loop_iter = loop_state["loop_iter"]
    state = loop_state["state"]
    inputs = loop_state["inputs"]
    outputs = loop_state["outputs"]
    circ_storage = loop_state["circ_storage"]
    circ_storage_mover = loop_state["circ_storage_mover"]

    # grab new input from either inputs or circ_storage
    state_io_batch_idx = loop_iter % args.microbatches_per_stage
    state_io_slice = inputs[state_io_batch_idx]
    if args.use_circ_storage:
        # Setup potential input from circ_storage, which also has a rotating index for microbatch, size of num_microbatches
        circ_storage_batch_idx = loop_iter % args.num_microbatches
        circular_stage_in = circ_storage[circ_storage_batch_idx]
    else:
        # The last stage immediately flows into the first stage, use this rotated shift instead of circular storage
        circular_stage_in = state
 
    # For early loop iterations we grab a new input for stage 0 from the state_io. Once each microbatch has left state_io
    # we instead grab from the last stage's output (possibly buffered when num_microbatches > num_stages, e.g. from circ_storage).
    first_stage_in = jnp.where(loop_iter < args.num_microbatches, state_io_slice, circular_stage_in)

    state = state.at[:].set(jnp.where(stage == 0, first_stage_in, state[:]))
    if args.shift_io:
      inputs = shift_inputs(loop_iter, inputs)

     # Fake/ poorly named argument, just repeating the same matmul per layer to increase model AI
    for _ in range(args.num_layers_per_stage):
      # Shard map is rank preserving, so the params have shape [1,embed,embed] inside each shard
      # We want to pass something of shape [embed, embed] instead, so we index away the first unit axis.
      state = fn(stage_params[0], state)

    # Updates
    if args.use_circ_storage:
      def _rotate_right_and_update(circ_storage_mover_in, circ_storage_in):
        rotated = shift_stages(loop_iter, circ_storage_mover_in) # TODO (this can remove some comms - is this correct for circ_storage_mover?)
        rotated = jnp.expand_dims(rotated, 0) # Don't think we need this
        # The offset is the previous iterations microbatch ID of the last stage, so that for example microbatch 0 will
        # be placed in index 0 of the num_microbatches axis. 
        offset = (loop_iter - (args.num_stages - 1) - 1) % args.num_microbatches # Note extra -1 b/c grabbing from the previous output - using circ_storage_mover before it is updated
        #breakpoint()
        return jax.lax.dynamic_update_slice_in_dim(circ_storage_in, rotated, offset, axis=0)
      circ_storage = _rotate_right_and_update(circ_storage_mover, circ_storage)
      circ_storage_mover = state

    # Is this equiv to permute output ms dim?
    output_offset = args.num_stages - 1
    # originally used num_layers where I believe we should use num_stages
    #outputs = outputs.at[(loop_iter-args.num_layers+1) % args.microbatches_per_stage].set(jnp.where(stage == args.num_stages-1, state, outputs[(loop_iter-args.num_layers+1) % args.microbatches_per_stage]))
    outputs = outputs.at[(loop_iter - output_offset) % args.microbatches_per_stage].set(jnp.where(stage == args.num_stages-1, state, outputs[(loop_iter - output_offset) % args.microbatches_per_stage]))
    state = shift_stages(loop_iter, state)
    # inputs = shift_inputs(loop_iter, inputs)
    if args.shift_io:
      outputs = shift_outputs(loop_iter, outputs) # Please uncomment me
    #state, inputs, outputs = shift(loop_iter, state, inputs, outputs)
    loop_state = {"loop_iter": loop_iter + 1, "state": state, "inputs":inputs, "outputs": outputs, "circ_storage": circ_storage, "circ_storage_mover": circ_storage_mover}
    return loop_state, None
  
  loop_state_final, _ = jax.lax.scan(run_iteration_scannable, loop_state,length=num_total_iterations)
  # run_iteration_scanned = jax.lax.scan(run_iteration_scannable, loop_state,length=num_total_iterations)
  # loop_state_final = run_iteration_scanned(loop_state, None)
  outputs = jax.lax.ppermute(loop_state_final["outputs"], 'stages', [(i, (i+1) % args.num_stages) for i in range(args.num_stages)])
  return outputs

def shift(i, state, inputs, outputs):
  sh = lambda x, d: jax.lax.ppermute(x, 'stages', [(i, (i+d) % args.num_stages) for i in range(args.num_stages)])
  if args.remove_dummy_comms:
    permute_pairs = get_real_permute_pairs(i, args.num_stages)
    state_sh = lambda x, d: jax.lax.ppermute(x, 'stages', permute_pairs)
  else:
    state_sh = sh
  
  state = state_sh(state, +1)
  # In pax code we roll the specific ms index every loop iteration
  # Instead we could roll every ms index after every`` K=|ms| loop iterations
  # This permutation should be done in full regardless of loop iteration - e.g. use all
  # stage pairs.
  if (i % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage):
    inputs = sh(inputs, +1)
  if ((i-args.num_layers+1) % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage):
    outputs = sh(outputs, +1)
  return state, inputs, outputs

def shift_stages(i, state):
  sh = lambda x, d: jax.lax.ppermute(x, 'stages', [(i, (i+d) % args.num_stages) for i in range(args.num_stages)])
  if args.remove_dummy_comms:
    permute_pairs = get_real_permute_pairs(i, args.num_stages)
    state_sh = lambda x, d: jax.lax.ppermute(x, 'stages', permute_pairs)
  else:
    state_sh = sh
  
  state = state_sh(state, +1)
  return state

def shift_inputs(i, inputs):
  sh = lambda x, d: jax.lax.ppermute(x, 'stages', [(i, (i+d) % args.num_stages) for i in range(args.num_stages)])
  inputs = jnp.where((i % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage), sh(inputs, +1), inputs)
  # if (i % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage):
  #   inputs = sh(inputs, +1)
  return inputs

def shift_outputs(i, outputs):
  sh_outputs = lambda x, d: jax.lax.ppermute(x, 'stages', [(i, (i+d) % args.num_stages) for i in range(args.num_stages)])
  outputs = jnp.where(((i-args.num_layers+1) % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage), sh_outputs(outputs, +1), outputs)
  # if ((i-args.num_layers+1) % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage):
  #   outputs = sh(outputs, +1)
  return outputs

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Sharding and size settings')
  parser.add_argument('--num_stages', type=int, default=4)
  parser.add_argument('--num_layers', type=int, default=4)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--embed_size', type=int, default=2048)
  parser.add_argument('--num_microbatches', type=int, default=4)
  parser.add_argument('--remove_dummy_comms', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--use_overlapped', action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument('--num_layers_per_stage', type=int, default=1) # Fake/ poorly named argument, just repeating the same matmul per layer to increase the model AI
  parser.add_argument('--shift_io', action=argparse.BooleanOptionalAction, default=True) # Needs to be true for correctness, but these shifts do not appear hidden on trace (they should be), and add complexity to HLO
  global args
  args = parser.parse_args()
    
  args.microbatches_per_stage = args.num_microbatches // args.num_stages
  assert args.num_layers % (args.num_stages * args.num_layers_per_stage) == 0, "Number of repeats must be an integer"
  args.num_repeats = args.num_layers // (args.num_stages * args.num_layers_per_stage)
  if args.num_repeats > 1 and args.num_microbatches > args.num_stages:
    args.use_circ_storage = True
  else:
    args.use_circ_storage = False

  params, batch = init(jax.random.PRNGKey(0), args.num_layers, args.embed_size, args.batch_size)
  print(f"input size is {batch[0].shape}")

  # assert args.num_layers == args.num_stages, "Number of layers must equal the number of stages"
  

  microbatch_size = args.batch_size // args.num_microbatches
  args.microbatch_size = microbatch_size
  microbatches_per_stage = args.num_microbatches // args.num_stages
  args.microbatches_per_stage = microbatches_per_stage

  print(f'{args.num_stages} stages, {args.num_layers // args.num_stages} layer(s) per stage, {args.num_layers} pipelined layers total')
  print(f'{args.microbatch_size} examples per microbatch, {args.num_microbatches} microbatches total')

  params_stacked = jnp.stack(params)
  params_sharded = jax.device_put(params_stacked, NamedSharding(mesh, P('stages')))
  batch_sharded = jax.device_put(batch, NamedSharding(mesh, P('stages')))

  print(f"regular loss {jax.jit(loss)(params, batch)}")
  print(f"pipeline loss {jax.jit(loss_pp)(params_sharded, batch_sharded)}")
  jit_pipeline = jax.jit(loss_pp)

  timing_util.simple_timeit(jit_pipeline, params_sharded, batch_sharded, tries = 3, task = 'shard_pp')
  

main()