"""Simplified training loop for NNX experimental track in MaxText."""

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from maxtext.nnx_exp.models import Llama, LlamaConfig
from maxtext.nnx_exp.sharding import LlamaSharding, create_mesh
from maxtext.nnx_exp.infra import (
    apply_remat,
    maybe_quantize,
    to_host,
)


def _target_distribution(logits, targets):
  target_dist = jax.nn.one_hot(targets, logits.shape[-1], dtype=jnp.float32)
  try:
    out_sharding = logits.sharding.spec
  except AttributeError:
    out_sharding = jax.typeof(logits).sharding.spec
  return jnp.reshape(target_dist, target_dist.shape, out_sharding=out_sharding)


def cross_entropy_loss(logits, targets):
  log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
  return -jnp.sum(log_probs * _target_distribution(logits, targets)) / targets.size


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
    seq_len = 128
    
    tokens = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    targets = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch_size, seq_len))
    mask = None
    
    # Place inputs on devices with explicit sharding
    tokens, targets, positions, mask = sharding.place_inputs(tokens, targets, positions, mask)

    # Initialize model
    model = Llama(config, rngs=rngs, sharding=sharding)
    print(f"Embedding sharding: {model.embed.embedding.sharding}")
    
    # 1. Apply Rematerialization
    print("Applying rematerialization...")
    apply_remat(model, policy="full")
    
    # 2. Apply Quantization (INT8)
    print("Applying quantization...")
    model = maybe_quantize(model, "int8", tokens, positions)
    
    # Split model into graphdef and params
    gdef, params = nnx.split(model, nnx.Param)
    
    # Create optimizer
    tx = optax.adam(learning_rate=3e-4)
    
    # Create TrainState
    state = nnx.TrainState.create(gdef, params=params, tx=tx)
    
    # 3. Apply Host Offloading to Optimizer State (Disabled due to JAX memory space mismatch)
    # print("Offloading optimizer state to host...")
    # state = state.replace(opt_state=to_host(state.opt_state))


    
    print("Starting dummy training step...")
    state, loss = train_step(state, tokens, positions, mask, targets)
    print(f"Step 1 loss: {loss}")

if __name__ == "__main__":
  main()
