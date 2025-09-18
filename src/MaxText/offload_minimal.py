import jax
from jax import numpy as jnp
from functools import partial

from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils

from jax.ad_checkpoint import checkpoint_name

import datetime


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

# Hardcode 1D data parallel mesh for this simple toy
# Meshes are for squares
global mesh
mesh = Mesh(jax.devices(), ('dp',))

# Print the key path and shapes of each pytree leaf, using jax.tree.map
def print_pytree_shapes(pytree):
    def get_key_paths(pytree, parent_path=()):
        paths = []
        if isinstance(pytree, dict):
            for k, v in pytree.items():
                paths.extend(get_key_paths(v, parent_path + (k,)))
        elif isinstance(pytree, (list, tuple)):
            for i, v in enumerate(pytree):
                paths.extend(get_key_paths(v, parent_path + (i,)))
        else:
            paths.append((parent_path, pytree.shape))
        return paths

    key_paths_and_shapes = get_key_paths(pytree)
    for path, shape in key_paths_and_shapes:
        path_str = ".".join(map(str, path))
        print(f"Path: {path_str}, Shape: {shape}")
pps = print_pytree_shapes # short name to be used with pdb

def simple_timeit(f, *args, tries=10, task=None, enable_profile=True):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  trace_name = f"{task}"  # + '_' ]+ ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  trace_dir = f"gs://mattdavidow-maxtext-br/{trace_name}"
  #trace_dir = "/tmp/where-my-trace"
  print(trace_dir)

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  if enable_profile:
    jax.profiler.start_trace(trace_dir)
  for _ in range(tries):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())
  if enable_profile:
    jax.profiler.stop_trace()
  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"Average time ms for mm for {task} is {round(average_time_ms, 3)}")
  return average_time_ms / 1000


def init_layer(key, embed_size, mlp_size):
    # returns a dictionary with two keys of size {"w_in": [embed, mlp] and "w_out":[mlp, embed]}
    keys = jax.random.split(key, 2)
    w_in = jax.random.normal(keys[0], (embed_size, mlp_size)) / jnp.sqrt(embed_size)
    w_out = jax.random.normal(keys[1], (mlp_size, embed_size)) / jnp.sqrt(mlp_size)
    params = {"w_in": w_in, "w_out": w_out}
    return params

def init_model_and_inputs(key_init, num_layers, embed_size, mlp_size, batch_size):
    keys = jax.random.split(key_init, num_layers)
    params = [init_layer(key, embed_size, mlp_size) for key in keys]
    params = jax.tree.map(lambda *args: jnp.stack(args), *params) # params are now pyree of "w_in"[layers, embed, mlp] amd "w_out [layers, , mlp, embed]
     
    # init inputs and targets
    input_key, target_key = jax.random.split(key_init, 2)
    inputs = jax.random.normal(input_key, (batch_size, embed_size)).astype(jnp.bfloat16)
    targets = jax.random.normal(target_key, (batch_size, embed_size)).astype(jnp.bfloat16)

    return params, inputs, targets

def layer_fn(inputs, single_layer_params):
    # single_layer_params is pytree with two leaves of shape [embed, mlp] and [mlp, embed]
    # Inputs have shape [global_batch, embed]
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    intermediate = jnp.dot(inputs, jnp.astype(single_layer_params['w_in'], jnp.bfloat16))
    intermediate = jax.nn.relu(intermediate)
    intermediate = checkpoint_name(intermediate, "intermediate")
    output = jnp.dot(intermediate, jnp.astype(single_layer_params['w_out'], jnp.bfloat16))
    return output, None # scan expect two outputs, carries to next layer and per layer outputs
        
def predict(params, inputs):
  # Params is pytree with two leaves of shape [layers, embed, mlp] and [layers, mlp, embed]
  # Inputs have shape [global_batch, embed]
  num_layers = params['w_in'].shape[0] #w_in is [layers, embed, mlp]

  layer_fn_remat = jax.checkpoint(layer_fn, policy=remat_policy())
  output, _ = jax.lax.scan(layer_fn_remat, inputs, params,length=num_layers) # scan returns final carry (final layer outputs here) and per layer outputs (none here)
  return output

def remat_policy():
    names_to_save_hbm = ("intermediate",)
    names_to_offload = ("decoder_layer_input",)
    policy = jax.checkpoint_policies.save_and_offload_only_these_names(
        names_which_can_be_saved=names_to_save_hbm,
        names_which_can_be_offloaded=names_to_offload,
        offload_src="device",
        offload_dst="pinned_host",
    )
    return policy

def loss(params, inputs, targets):
  # Params is pytree with two leaves of shape [layers, embed, mlp] and [layers, mlp, embed]
  # Inputs and targets both have shape [global_batch, embed]
  predictions = predict(params, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))
    

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Sharding and size settings')
  parser.add_argument('--global_batch_size', type=int, default=32768)
  parser.add_argument('--embed_size', type=int, default=2048)
  parser.add_argument('--mlp_size', type=int, default=8192)
  parser.add_argument('--num_layers', type=int, default=8)
  global args
  args = parser.parse_args()
    
  params, inputs, targets = init_model_and_inputs(jax.random.PRNGKey(0), args.num_layers, args.embed_size, args.mlp_size, args.global_batch_size)

  # shard params and inputs
  params_sharded = jax.device_put(params, NamedSharding(mesh, P())) # params fully replicated (data parallel)
  inputs_sharded = jax.device_put(inputs, NamedSharding(mesh, P('dp'))) # inputs  is [batch, embed]
  targets_sharded = jax.device_put(inputs, NamedSharding(mesh, P('dp'))) # targets  is [batch, embed]

  grad_fn = jax.value_and_grad(loss)
  jit_grad_fn = jax.jit(grad_fn)
  
  simple_timeit(jit_grad_fn, params_sharded, inputs_sharded, targets_sharded, tries = 3, task = 'toy_model')
  

main()