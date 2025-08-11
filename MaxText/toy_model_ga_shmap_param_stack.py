import jax
from jax import numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from jax.tree_util import tree_map, tree_all

from functools import partial
import datetime
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"

# Utilities
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


###### Start main script ######

# Hardcode 1D data parallel mesh for this simple toy
global mesh
mesh = Mesh(jax.devices(), ('dp',))
replicated_sharding = NamedSharding(mesh, P()) # used for params - replicated for DP
dp_sharding = NamedSharding(mesh, P('dp')) # use for inputs

def init_layer(key, embed_size, mlp_size):
    # returns a dictionary with two keys of sizes {"w_in": [embed, mlp] and "w_out":[mlp, embed]}
    keys = jax.random.split(key, 2)
    w_in = jax.random.normal(keys[0], (embed_size, mlp_size)) / jnp.sqrt(embed_size)
    w_out = jax.random.normal(keys[1], (mlp_size, embed_size)) / jnp.sqrt(mlp_size)
    params = {"w_in": w_in, "w_out": w_out}
    return params

def init_model_and_inputs(key_init, num_layers, embed_size, mlp_size, batch_size):
    # initializes weights and with random values
    # params are now pytree with two keys: "w_in" : [layers, embed, mlp] amd "w_out [layers, mlp, embed]
    # returns inputs and targets of size [batch_size, embed]
    keys = jax.random.split(key_init, num_layers)
    params = [init_layer(key, embed_size, mlp_size) for key in keys]
    params = jax.tree.map(lambda *args: jnp.stack(args), *params) 
     
    # init inputs and targets
    input_key, target_key = jax.random.split(key_init, 2)
    inputs = jax.random.normal(input_key, (batch_size, embed_size)).astype(jnp.bfloat16)
    targets = jax.random.normal(target_key, (batch_size, embed_size)).astype(jnp.bfloat16)
    return params, inputs, targets

def layer_fn(inputs, single_layer_params):
    # input shape and output are the same: [batch, embed]
    # outputs = inputs @ [embed, mlp] @ [mlp, embed] (with some nonlinearity)
    intermediate = jnp.dot(inputs.astype(jnp.bfloat16), single_layer_params['w_in'].astype(jnp.bfloat16))
    intermediate = jax.nn.relu(intermediate)
    output = jnp.dot(intermediate.astype(jnp.bfloat16), single_layer_params['w_out'].astype(jnp.bfloat16))
    return output, None # scan expect two outputs, carries to next layer (output) and per layer outputs (None in this case)
        
def predict(params, inputs):
  # Params is pytree with two leaves of shape [layers, embed, mlp] and [layers, mlp, embed]
  # Inputs have shape [global_batch, embed]
  num_layers = params['w_in'].shape[0] #w_in is [layers, embed, mlp]
  # jax.lax.scan: https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
  # Takes as input (function_to_scan, initial_input, inputs_each_layer, length)
  output, _ = jax.lax.scan(layer_fn, inputs, params,length=num_layers) # scan returns final carry (final layer outputs here) and per layer outputs (none here)
  return output

def loss_fn(params, inputs, targets):
  # Params is pytree with two leaves of shape [layers, embed, mlp] and [layers, mlp, embed]
  # Inputs and targets both have shape [global_batch, embed]
  predictions = predict(params, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))

@partial(jax.experimental.shard_map.shard_map, mesh=mesh, in_specs=((P('dp')), P('dp'), P('dp')),
        out_specs=P('dp'))
def per_device_losses_and_grads(params, inputs, targets):
    # remove initial singleton dim from params
    params = jax.tree_util.tree_map(lambda x: x[0], params)
    grad_func = jax.value_and_grad(loss_fn)
    losses, grads = grad_func(params, inputs, targets)
    losses = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), losses)
    grads = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), grads)
    
    return losses, grads
  

def loss_and_grad_ga(params, inputs, targets, num_microbatches):
    # Params is pytree with two leaves of shape [layers, embed, mlp] and [layers, mlp, embed]
    # Inputs and targets both have shape [global_batch, embed]
    # Outputs is params, same size and sharding as params

    num_dp_replicas = jax.device_count()
    microbatch_size = inputs.shape[0] // num_microbatches
    embed = inputs.shape[1]
    def reshape_inputs_for_microbatches(inputs):
        inputs = inputs.reshape([microbatch_size, num_microbatches, embed])
        inputs = inputs.transpose([1, 0, 2]) 
        # This ensures the previous dp sharding of 2D inputs does not need any communication to reshape to microbatches
        input_sharding_constraint = NamedSharding(mesh, P(None, "dp", None)) # [microbatches, microbatch_size, embed] microbatches is leading since it will be scanned over
        return jax.lax.with_sharding_constraint(inputs, input_sharding_constraint)
    inputs = reshape_inputs_for_microbatches(inputs)
    targets = reshape_inputs_for_microbatches(targets)
    params_replicated = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_dp_replicas), params) # our hack around shmap + grad

    #grad_func = jax.value_and_grad(loss_fn)
    grad_func = per_device_losses_and_grads
    def accumulate_gradient(acc_grad_and_loss, inputs_and_targets):
        inputs, targets= inputs_and_targets['inputs'], inputs_and_targets['targets']
        cur_batch_loss, cur_batch_gradient = grad_func(params_replicated, inputs, targets)
        acc_grad_and_loss["loss"] += cur_batch_loss
        acc_grad_and_loss["grad"] = jax.tree_util.tree_map(lambda x, y: x + y, cur_batch_gradient, acc_grad_and_loss["grad"])
        return acc_grad_and_loss, None # We will scan this so need to return None

    init_grad = jax.tree_util.tree_map(jnp.zeros_like, params_replicated)
    # init_loss is a jax array of length num_dp_replicas of zeros
    init_loss = jnp.zeros(num_dp_replicas)
    init_grad_and_loss = {"loss": init_loss, "grad": init_grad}
    inputs_and_targets = {"inputs": inputs,"targets": targets}
    grad_and_loss, _ = jax.lax.scan(accumulate_gradient, init_grad_and_loss, inputs_and_targets, length=num_microbatches)
    grads, loss = grad_and_loss["grad"], grad_and_loss["loss"]
    # Average the accumulated gradients and loss to match the global batch calculation
    # grads = jax.tree_util.tree_map(
    #     lambda g: g / num_microbatches, grads
    # )
    # loss = loss / num_microbatches

    # grads = jax.lax.pmean(grads, axis_name='dp')
    # # TODO(mattdavidow: Why do we need an additional divison by num_dp_replicas here? We already took average above?)
    # grads = jax.tree_util.tree_map(
    #     lambda g: g / num_dp_replicas, grads
    # )
    # loss = jax.lax.pmean(loss, axis_name='dp')
    # return loss, grads
    return loss, grads

 

def main():
  import argparse
  parser = argparse.ArgumentParser(description='Sharding and size settings')
  parser.add_argument('--global_batch_size', type=int, default=262144) # 262144 #2097152
  parser.add_argument('--embed_size', type=int, default=2048)
  parser.add_argument('--mlp_size', type=int, default=8192)
  parser.add_argument('--num_layers', type=int, default=2)
  parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
  global args
  args = parser.parse_args()
    


  jit_initialize_inputs = jax.jit(init_model_and_inputs, static_argnums=(1,2,3,4),out_shardings=(replicated_sharding, dp_sharding, dp_sharding))
  params, inputs, targets = jit_initialize_inputs(jax.random.PRNGKey(0), args.num_layers, args.embed_size, args.mlp_size, args.global_batch_size)

  # non-ga, regular run
  grad_fn = jax.value_and_grad(loss_fn)
  jit_grad_fn = jax.jit(grad_fn)
  loss_non_ga, non_ga_grad = jit_grad_fn(params, inputs, targets)
  
  # ga run
  jit_loss_and_grad_ga = jax.jit(loss_and_grad_ga, static_argnums=(3,)) # num_microbathces is static
  loss_ga, ga_grad = jit_loss_and_grad_ga(params, inputs, targets, args.gradient_accumulation_steps)

  # tree map sum ga_grad along leading axis
  ga_grad_avg = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), ga_grad)
  loss_ga = jnp.sum(loss_ga, axis=0)

  print(f"{loss_ga=}")
  print(f"{loss_non_ga=}")
  
  nga=non_ga_grad['w_in'][0:3,0:3,0]
  ga=ga_grad['w_in'][0:3,0:3,0]
  rawr = ga/nga
  print(rawr)

  # assert non_ga_grad and ga_grad are similar
  def allclose(a, b):
    return tree_all(tree_map(partial(jnp.allclose, atol=1e-2, rtol=1e-2), a, b))
  assert allclose(non_ga_grad, ga_grad)
  
  # profile + time the ga run
  simple_timeit(jit_loss_and_grad_ga, params, inputs, targets, args.gradient_accumulation_steps, tries = 3, task = 'toy_model_ga')

  # profile + time the non-ga run
  #simple_timeit(jit_grad_fn, params, inputs, targets, tries = 3, task = 'toy_model_non_ga')

main()