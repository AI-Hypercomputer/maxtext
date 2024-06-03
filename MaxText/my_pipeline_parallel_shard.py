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


  # TODO: double the bubble stages
  for loop_iter in range(args.num_microbatches+args.num_layers-1):
    # Push new input into stage 0
    current_input = current_input.at[:].set(jnp.where(stage == 0, inputs[loop_iter % args.microbatches_per_stage], current_input[:]))



    # compute
    # new_previous_output = compute(current_input)
    # We want to pass something of shape [embed, embed] instead, so we index away the first unit axis.
    # Fake/ poorly named argument, just repeating the same matmul per layer
    new_previous_output = current_input
    for _ in range(args.num_layers_per_stage):
      # Shard map is rank preserving, so the params have shape [1,embed,embed] inside each shard
      new_previous_output = fn(stage_params[0], new_previous_output)

    # Store outputs
    # This stores the last stages output, may need to be modified because larger initial bubble
    outputs = outputs.at[(loop_iter-args.num_layers+1) % args.microbatches_per_stage].set(jnp.where(stage == args.num_stages-1, new_previous_output, outputs[(loop_iter-args.num_layers+1) % args.microbatches_per_stage]))


    # communicate (permute)
    # next_input = communicate_collective_permte(previous_output)
    # state, inputs, outputs = shift(loop_iter, state, inputs, outputs)

    # Split the 3 rotations into their own function for easier xprof code tracing
    next_input = shift_stages(loop_iter, prev_output)
    inputs = shift_inputs(loop_iter, inputs)
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
  for loop_iter in range(args.num_microbatches+args.num_layers-1):
    state = state.at[:].set(jnp.where(stage == 0, inputs[loop_iter % args.microbatches_per_stage], state[:]))
    inputs = shift_inputs(loop_iter, inputs)
     # Fake/ poorly named argument, just repeating the same matmul per layer
    for _ in range(args.num_layers_per_stage):
      # Shard map is rank preserving, so the params have shape [1,embed,embed] inside each shard
      # We want to pass something of shape [embed, embed] instead, so we index away the first unit axis.
      state = fn(stage_params[0], state)
    outputs = outputs.at[(loop_iter-args.num_layers+1) % args.microbatches_per_stage].set(jnp.where(stage == args.num_stages-1, state, outputs[(loop_iter-args.num_layers+1) % args.microbatches_per_stage]))

    state = shift_stages(loop_iter, state)
    # inputs = shift_inputs(loop_iter, inputs)
    outputs = shift_outputs(loop_iter, outputs)
    #state, inputs, outputs = shift(loop_iter, state, inputs, outputs)
  outputs = jax.lax.ppermute(outputs, 'stages', [(i, (i+1) % args.num_stages) for i in range(args.num_stages)])
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
  if (i % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage):
    inputs = sh(inputs, +1)
  return inputs

def shift_outputs(i, outputs):
  sh = lambda x, d: jax.lax.ppermute(x, 'stages', [(i, (i+d) % args.num_stages) for i in range(args.num_stages)])
  if ((i-args.num_layers+1) % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage):
    outputs = sh(outputs, +1)
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
  global args
  args = parser.parse_args()
    
  params, batch = init(jax.random.PRNGKey(0), args.num_layers, args.embed_size, args.batch_size)
  print(f"input size is {batch[0].shape}")

  assert args.num_layers == args.num_stages, "Number of layers must equal the number of stages"

  microbatch_size = args.batch_size // args.num_microbatches
  args.microbatch_size = microbatch_size
  microbatches_per_stage = args.num_microbatches // args.num_stages
  args.microbatches_per_stage = microbatches_per_stage

  print(f'{args.num_stages} stages, {args.num_layers // args.num_stages} layer(s) per stage, {args.num_layers} pipelined layers total')
  print(f'{args.microbatch_size} examples per microbatch, {args.num_microbatches} microbatches total')

  params_stacked = jnp.stack(params)
  params_sharded = jax.device_put(params_stacked, NamedSharding(mesh, P('stages')))
  batch_sharded = jax.device_put(batch, NamedSharding(mesh, P('stages')))

  print(jax.jit(loss)(params, batch))
  print(jax.jit(loss_pp)(params_sharded, batch_sharded))
  jit_pipeline = jax.jit(loss_pp)

  timing_util.simple_timeit(jit_pipeline, params_sharded, batch_sharded, tries = 3, task = 'shard_pp')
  

main()