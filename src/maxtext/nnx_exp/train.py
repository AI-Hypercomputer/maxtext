"""Simplified training loop for NNX experimental track in MaxText."""

from datetime import datetime

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from maxtext.nnx_exp.models import Llama, LlamaConfig
from maxtext.nnx_exp.sharding import LlamaSharding, create_mesh, LlamaShardingHook, shard_model_parameters
from maxtext.nnx_exp.infra import (
    apply_remat,
    maybe_quantize,
    to_host,
)

def cross_entropy_loss(logits, targets):
  from jax.sharding import NamedSharding, PartitionSpec as P, reshard
  mesh = jax.sharding.get_abstract_mesh()
  if mesh is not None:
    logits = reshard(logits, NamedSharding(mesh, P(("dp", "pp"), None, None)))
    targets = reshard(targets, NamedSharding(mesh, P(("dp", "pp"), None)))
  loss = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), targets)
  return jnp.mean(loss)


@jax.jit
def train_step(state, tokens, positions, mask, targets):
  def loss_fn(params):
    logits = nnx.merge(state.graphdef, params)(tokens, positions, mask=mask)
    return cross_entropy_loss(logits, targets)

  loss, grads = jax.value_and_grad(loss_fn)(state.params)
  return state.apply_gradients(grads=grads), loss


def main():
  # Simple config for prototype
  config = LlamaConfig(dtype="bfloat16")
  
  # Mock cluster config for single-host TPU or local CPU/GPU
  cluster_cfg = {
    "ici_dp_parallelism": 1,
    "ici_fsdp_parallelism": -1,
    "ici_tensor_parallelism": 1,
    "ici_sequence_parallelism": 1,
  }
  
  mesh = create_mesh(cluster_cfg)
  print(f"Total devices: {jax.device_count()}")
  print(f"Mesh: {mesh}")
  
  with jax.set_mesh(mesh):
    sharding = LlamaSharding()
    rngs = nnx.Rngs(42)
    
    # Generate mock data (needed for quantization tracing)
    batch_size = 2
    seq_len = 4096
    
    tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    targets = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
    mask = None
    
    # Place inputs on devices with explicit sharding
    tokens, targets, positions, mask = sharding.place_inputs(tokens, targets, positions, mask)

    # Initialize model with sharding callback hook (handles both Initializer-Time weight sharding and Activation sharding)
    sharding_hook = LlamaShardingHook(sharding)
    model = Llama(config, rngs=rngs, sharding_hook=sharding_hook, scan=True)
    print(f"Embedding sharding: {model.embed.embedding.sharding}")
    
    # 1. Apply Rematerialization
    print("Applying rematerialization...")
    apply_remat(model, policy="full")
    
    # 2. Apply Quantization (INT8)
    print("Applying quantization...")
    model = maybe_quantize(model, "", tokens, positions)
    
    # Split model into graphdef and params
    gdef, params = nnx.split(model, nnx.Param)
    
    # Create optimizer
    #tx = optax.adam(learning_rate=3e-4)
    tx = optax.sgd(learning_rate=3e-4, momentum=0.9)
    
    # Create TrainState
    state = nnx.TrainState.create(gdef, params=params, tx=tx)
    
    # 3. Apply Host Offloading to Optimizer State (Disabled due to JAX memory space mismatch)
    # print("Offloading optimizer state to host...")
    # state = state.replace(opt_state=to_host(state.opt_state))


    
    print("Starting training loop...")
    num_steps = 10
    profile_dir = "gs://bvandermoon-multipod-maxtext/m3_profile_traces/" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    
    loss = None
    for step in range(num_steps):
      if step == 5:
        if loss is not None:
          loss.block_until_ready()
        print(f"Starting profiler at step {step}...")
        jax.profiler.start_trace(profile_dir)
        
      state, loss = train_step(state, tokens, positions, mask, targets)
      print(f"Step {step} loss: {loss}")
      
      if step == 7:
        loss.block_until_ready()
        print(f"Stopping profiler at step {step}...")
        jax.profiler.stop_trace()
        print(f"Access profile at {profile_dir}")

if __name__ == "__main__":
  main()
