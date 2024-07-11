import jax
from jax import numpy as jnp
from functools import partial

from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

global mesh
mesh = Mesh(jax.devices(), ('stages',))

def predict(params, inputs):
  # predict function for non-pipeline (used to assert correctness)
  for layer in params:
    for _ in range(args.matmul_repeats):
      inputs = jnp.dot(inputs, layer)
      inputs = jax.nn.relu(inputs)
  return inputs

def loss(params, batch):
  # loss function for non-pipeline (used to assert correctness)
  inputs, targets = batch
  predictions = predict(params, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))

def init_layer(key, embed_size):
    # Initialize parameters for a layer
    W = jax.random.normal(key, (embed_size, embed_size)) / jnp.sqrt(embed_size)
    return W

def init(key_init, num_layers, embed_size, batch_size):
    # Initialize all model parameters, inputs and targets
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

def spmd_pipeline(fn, stage_params, inputs):
  stage = jax.lax.axis_index('stages')
  outputs = jnp.zeros_like(inputs) * jnp.nan
  state = jnp.zeros((args.microbatch_size, args.embed_size)) * jnp.nan
  # Each stage has their own circ_storage, don't need leading num_stages
  if args.use_circ_storage:
    circ_storage_mover = jnp.zeros_like(state) * jnp.nan
    circ_storage = jnp.zeros((args.num_microbatches, args.microbatch_size, args.embed_size))
  else:
    circ_storage_mover, circ_storage = None, None

  def run_iteration_scannable(loop_state, xs):
    # loop state components:
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
      # This is needed for correctness - this pushes new inputs to the top to be moved into the first pipeline
      inputs = shift_inputs(loop_iter, inputs)

     # matmul_repeats just repeats the same matmul per layer to increase model AI - doesn't actually use new weights
    for _ in range(args.matmul_repeats):
      # Shard map is rank preserving, so the params have shape [1,embed,embed] inside each shard
      # We want to pass something of shape [embed, embed] instead, so we index away the first unit axis.
      state = fn(stage_params[0], state)

    # Updates
    if args.use_circ_storage:
      # push new inputs and rotate circ_storage_mover from old circ_storage
      # update circ_storage to be the current outputs
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

    # Push last pipeline stage to output. In order to keep the same permutation order of outputs as inputs we push to a specific microbatches_per_stage index (so that first real output lands on idx 0)
    output_offset = args.num_stages - 1
    outputs = outputs.at[(loop_iter - output_offset) % args.microbatches_per_stage].set(jnp.where(stage == args.num_stages-1, state, outputs[(loop_iter - output_offset) % args.microbatches_per_stage]))
    state = shift_stages(loop_iter, state)
    if args.shift_io:
      # This is needed for correctness, we are rotating outputs down over stages so they are spread evenly across stages
      outputs = shift_outputs(loop_iter, outputs)
    loop_state = {"loop_iter": loop_iter + 1, "state": state, "inputs":inputs, "outputs": outputs, "circ_storage": circ_storage, "circ_storage_mover": circ_storage_mover}
    return loop_state, None
  
  num_total_iterations = args.num_microbatches * args.num_repeats + args.num_stages - 1
  loop_state = {"loop_iter": 0, "state": state, "inputs": inputs, "outputs": outputs, "circ_storage": circ_storage, "circ_storage_mover": circ_storage_mover}

  if args.scan_iterations:
    loop_state_final, _ = jax.lax.scan(run_iteration_scannable, loop_state,length=num_total_iterations)
  else:
    for loop_iter in range(num_total_iterations):
      loop_state, _ = run_iteration_scannable(loop_state, None)
    loop_state_final = loop_state
  # outputs needs one more permute
  outputs = jax.lax.ppermute(loop_state_final["outputs"], 'stages', [(i, (i+1) % args.num_stages) for i in range(args.num_stages)])
  return outputs

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
  # basically the same as shift_stages, except no option to remove dummy comms (although removing dummy comms should also work)
  sh = lambda x, d: jax.lax.ppermute(x, 'stages', [(i, (i+d) % args.num_stages) for i in range(args.num_stages)])
  inputs = jnp.where((i % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage), sh(inputs, +1), inputs)
  return inputs

def shift_outputs(i, outputs):
  # This is the same method as shift_inputs (also basically the same as shift_stages)
  sh_outputs = lambda x, d: jax.lax.ppermute(x, 'stages', [(i, (i+d) % args.num_stages) for i in range(args.num_stages)])
  outputs = jnp.where(((i-args.num_layers+1) % args.microbatches_per_stage) == (-1 % args.microbatches_per_stage), sh_outputs(outputs, +1), outputs)
  return outputs

import datetime
import jax
import random
import string
def simple_timeit(f, *args, tries=10, task=None):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  trace_name = f"t_{task}_" + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  trace_dir = f"gs://mattdavidow-br/{trace_name}"

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  jax.profiler.start_trace(trace_dir)

  for _ in range(tries):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())
  jax.profiler.stop_trace()

  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"{task}: average time milliseconds: {average_time_ms:.2f}, trace {trace_dir}")
  return average_time_ms

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Sharding and size settings')
  parser.add_argument('--num_stages', type=int, default=4)
  parser.add_argument('--num_layers', type=int, default=8)
  parser.add_argument('--microbatch_size', type=int, default=4096)
  parser.add_argument('--embed_size', type=int, default=8192) # The arithmetic intensity (ease of overlap) is proportional to this since the matmul FLOPs are batch * embed^2 and then communication is of size batch * embed (ratio of embed)
  parser.add_argument('--num_microbatches', type=int, default=8)
  parser.add_argument('--remove_dummy_comms', action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument('--matmul_repeats', type=int, default=1) # Imitating/simplification of multiple layers per stage - this allows us to increase the model AI (more compute per stage, easier to hide collective permutes) (e.g. AI is proportional to embed * matmul_repeats)
  parser.add_argument('--scan_iterations', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--shift_io', action=argparse.BooleanOptionalAction, default=True) # Needs to be true for correctness, but these shifts add complexity to HLO, (collective permutes which should be easily overlapped)
  parser.add_argument('--check_correctness', action=argparse.BooleanOptionalAction, default=False)
  parser.add_argument('--run_timing_script', action=argparse.BooleanOptionalAction, default=True)
  global args
  args = parser.parse_args()
    
  assert args.num_stages == len(jax.devices()), "Number of stages must be equal to the number of devices"
  assert args.num_microbatches % args.num_stages==0, "Number of microbatches must be a multiple of number stages"
  args.microbatches_per_stage = args.num_microbatches // args.num_stages
  assert args.num_layers % args.num_stages==0, "Number of layers must be a multiple of number of stages"
  args.num_repeats = args.num_layers // args.num_stages
  assert not args.scan_iterations or not args.remove_dummy_comms, "Removing dummy comms does not work with scanning, that is a large part of what we need to fix!"
  print(f"Pipelining using {args.num_stages} stages, {args.num_repeats} repeats, {args.num_microbatches} microbatches.")
  if args.num_repeats > 1 and args.num_microbatches > args.num_stages:
    args.use_circ_storage = True
  else:
    args.use_circ_storage = False

  global_batch_size = args.microbatch_size * args.num_microbatches
  params, batch = init(jax.random.PRNGKey(0), args.num_layers, args.embed_size, global_batch_size)
  params_stacked = jnp.stack(params)
  params_sharded = jax.device_put(params_stacked, NamedSharding(mesh, P('stages')))
  batch_sharded = jax.device_put(batch, NamedSharding(mesh, P('stages')))

  if args.check_correctness:
    print(f"regular loss {jax.jit(loss)(params, batch)}")
    print(f"pipeline loss {jax.jit(loss_pp)(params_sharded, batch_sharded)}")
  
  if args.run_timing_script:
    jit_pipeline = jax.jit(loss_pp)
    simple_timeit(jit_pipeline, params_sharded, batch_sharded, tries = 3, task = 'shard_pp')

main()

