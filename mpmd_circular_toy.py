# Author: chengnuojin@
from typing import Any, Sequence
from functools import partial

import pathwaysutils  # pylint: disable=unused-import
from absl import app
import datetime
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.lax.parallel import all_gather_invariant
from shardy.integrations.python.jax import mpmd

pathwaysutils.initialize()

# Multi-stage hyperparameters
NUM_LAYERS = 32
EMB_DIM = 2048
MLP_DIM = 8192
GBS = 2048 * 16
NUM_STAGES = 4
NUM_MICROBATCHES = 16
dtype = jnp.float32


def init_layer(key, embed_size, mlp_size):
  keys = jax.random.split(key, 2)
  w_in = jax.random.normal(keys[0], (embed_size, mlp_size)) / jnp.sqrt(embed_size)
  w_out = jax.random.normal(keys[1], (mlp_size, embed_size)) / jnp.sqrt(mlp_size)
  return {"w_in": w_in, "w_out": w_out}


def init_model_and_inputs(key_init, num_layers, embed_size, mlp_size, batch_size, num_stages, num_microbatch):
  keys = jax.random.split(key_init, num_stages)
  num_repeats = num_layers // num_stages
  params = {}
  for i in range(num_stages):
    stage_keys = jax.random.split(keys[i], num_repeats)
    p = [init_layer(key, embed_size, mlp_size) for key in stage_keys]
    params[f"stage{i}"] = jax.tree.map(lambda *args: jnp.stack(args), *p)

  microbatch_size = batch_size // num_microbatch
  input_key, target_key = jax.random.split(key_init, 2)
  inputs = jax.random.normal(input_key, (num_microbatch, microbatch_size, embed_size))
  targets = jax.random.normal(target_key, (num_microbatch, microbatch_size, embed_size))
  return params, inputs, targets


def create_custom_mpmd_in_shardings(params, name2mesh, topology):
  params_shardings = {}
  w_in_spec = P(None, "fsdp", None)  # [repeats, embed, fsdp]
  w_out_spec = P(None, "fsdp", None)  # [repeats, mlp, embed]

  for stage_name in params.keys():
    mesh_name = name2mesh[stage_name]
    stage_mesh = topology[mesh_name]
    params_shardings[stage_name] = {
        "w_in": NamedSharding(stage_mesh, w_in_spec),
        "w_out": NamedSharding(stage_mesh, w_out_spec),
    }

  return params_shardings

# TODO: can we refactor so mpmd calls this instead of duplicating?
# TODO: Currently no sharding
def layer_fn_regular(inputs, single_layer_params):
    intermediate = jnp.dot(inputs, single_layer_params["w_in"])
    intermediate = jax.nn.relu(intermediate)
    output = jnp.dot(intermediate, single_layer_params["w_out"])
    return output, None

def predict_regular(params, inputs):
    num_repeats = params["w_in"].shape[0]
    output, _ = jax.lax.scan(layer_fn_regular, inputs, params, length=num_repeats)
    return output
  
def build_model_fns(mpmd_config):
  # Encapsulated so that shard_map wrappers bind to the correct mesh dynamically
  # created after Pathways service connects.

  def layer_fn_mpmd(inputs, single_layer_params):
    inputs = jax.lax.with_sharding_constraint(inputs, NamedSharding(mpmd_config.sharding_mesh, P("fsdp", None)))
    intermediate = jnp.dot(inputs, single_layer_params["w_in"])
    intermediate = jax.nn.relu(intermediate)
    intermediate = jax.lax.with_sharding_constraint(
        intermediate, NamedSharding(mpmd_config.sharding_mesh, P("fsdp", None))
    )
    output = jnp.dot(intermediate, single_layer_params["w_out"])
    output = jax.lax.with_sharding_constraint(output, NamedSharding(mpmd_config.sharding_mesh, P("fsdp", None)))
    return output, None

  def predict_mpmd(params, inputs):
    num_repeats = params["w_in"].shape[0]
    output, _ = jax.lax.scan(layer_fn_mpmd, inputs, params, length=num_repeats)
    return output

  def multi_stage_predict_mpmd(params, inputs):
    num_stages = len(params.keys())
    for i in range(num_stages):
      inputs = mpmd.named_computation(predict_mpmd, name=f"stage{i}")(
          params[f"stage{i}"],
          inputs,
      )
    return inputs

  # Matt question: Which stage does this run on? - the jnp.mean part?
  def loss_fn_mpmd(params, inputs, targets):
    predictions = multi_stage_predict_mpmd(params, inputs)
    return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))

  @partial(
      jax.shard_map,
      mesh=mpmd_config.sharding_mesh,
      in_specs=P(None, "fsdp", None),
      out_specs=P(),
      check_vma=True,
  )
  def gather_params_globally(p):
    return all_gather_invariant(p, axis_name="fsdp", axis=1, tiled=True)

  @partial(
      jax.shard_map,
      mesh=mpmd_config.sharding_mesh,
      in_specs=P(),
      out_specs=P(None, "fsdp", None),
      check_vma=True,
  )
  def scatter_grads(g):
    return jax.lax.psum_scatter(g, axis_name="fsdp", scatter_dimension=1, tiled=True)

  def microbatch_loss_and_grad(params, inputs, targets):
    grad_acc = jax.tree.map(lambda x: jnp.zeros_like(x), params)
    loss_acc = jnp.array(0.0)
    carry = (loss_acc, grad_acc)

    num_microbatch = len(inputs)
    gathered_params = jax.tree.map(lambda p: gather_params_globally(p), params)

    def microbatch_step(carry, params, inputs, targets):
      loss, grads = jax.value_and_grad(loss_fn_mpmd)(params, inputs, targets)
      loss_acc, grad_acc = carry
      loss_acc += loss
      grad_acc = jax.tree.map(jax.lax.add, grad_acc, grads)
      return loss_acc, grad_acc

    for i in range(num_microbatch):
      micro_inputs = inputs[i]
      micro_targets = targets[i]
      carry = mpmd.call(microbatch_step, call_counter=i)(carry, gathered_params, micro_inputs, micro_targets)

    loss_acc, grads_acc = carry
    sharded_grads = jax.tree.map(lambda g: scatter_grads(g), grads_acc)
    return loss_acc / num_microbatch, sharded_grads

  return microbatch_loss_and_grad


def run_mpmd(params, inputs, targets):
  # Must pull devices here so it registers the Pathways TPU cluster properly
  devices = jax.devices()
  device_per_stage = len(devices) // NUM_STAGES

  topology = {
      f"m{i}": Mesh(np.array(devices[device_per_stage * i : device_per_stage * (i + 1)]), ("fsdp",))
      for i in range(NUM_STAGES)
  }
  name2mesh = {f"stage{i}": f"m{i}" for i in range(NUM_STAGES)}

  # Shardy MPMD config
  mpmd_config = mpmd.make_config(
      topology,
      name2mesh,
      partitioning_options={
          "mpmd_infer_cross_mesh_reductions": True,
          "mpmd_infer_transfers": True,
          #"mpmd_pipeline_schedule": "GPipe",
          #"mpmd_pipeline_schedule": "1F1B",
          "mpmd_pipeline_schedule": "Circular",
      },
  )

  # Initialize variables
  print("Initializing logic and devices...")
  params_sharding = create_custom_mpmd_in_shardings(params, name2mesh, topology)
  inputs_sharding = NamedSharding(
      topology["m0"],
      P(
          None,
          "fsdp",
      ),
  )
  targets_sharding = NamedSharding(
      topology[f"m{NUM_STAGES-1}"],
      P(
          None,
          "fsdp",
      ),
  )

  in_shardings = (
      params_sharding,
      inputs_sharding,
      targets_sharding,
  )

  microbatch_loss_and_grad = build_model_fns(mpmd_config)

  jit_loss_and_grad = mpmd.jit(
      microbatch_loss_and_grad,
      mpmd_config=mpmd_config,
      in_shardings=in_shardings,
  )

  # Device Put
  params = jax.tree.map(lambda p, sharding: jax.device_put(p, sharding), params, params_sharding)
  inputs = jax.device_put(inputs, inputs_sharding)
  targets = jax.device_put(targets, targets_sharding)

  print("Lowering computation...")
  fwd_bwd_lowered = jit_loss_and_grad.lower(params, inputs, targets)
  print("Lowering successful!")

  print("Compiling logic...")
  compiled = fwd_bwd_lowered.compile()
  print("Compilation successful!")

  print("Running step...")
  simple_timeit(compiled, params, inputs, targets, tries=1, task="mpmd-train-demo")
  print("Run successful!")

  # print("Running step...")
  # simple_timeit(jit_loss_and_grad, params, inputs, targets, tries=1, task="mpmd-train-demo")
  # print("Run successful!")

def run_regular(params, inputs, targets):
  breakpoint()

def main(argv: Sequence[str]) -> None:
  params, inputs, targets = init_model_and_inputs(
      jax.random.PRNGKey(0),
      NUM_LAYERS,
      EMB_DIM,
      MLP_DIM,
      GBS,
      NUM_STAGES,
      NUM_MICROBATCHES,
  )
  #run_mpmd(params, inputs, targets)
  run_regular(params, inputs, targets)


def simple_timeit(f, *args, tries=1, task=None, enable_profile=True):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  trace_name = f"{task}"  # + '_' ]+ ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  trace_dir = f"gs://mattdavidow-maxtext-br/mintext-mpmd/a8/{trace_name}"

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  import time
  time.sleep(5) #profile is messed up?
  if enable_profile:
    jax.profiler.start_trace(trace_dir)
    print(f"{trace_dir=}")
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


if __name__ == "__main__":
  app.run(main)

