import jax
import functools
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import partitioning
from flax.training import train_state
from jax.sharding import PartitionSpec, Mesh, NamedSharding
from flax.linen import partitioning as nn_partitioning
import optax
from jax.experimental import mesh_utils
import datetime
import jax.profiler
import gzip


from typing import Sequence, Optional, Callable, Any

# Reduced model size and added bfloat16 for memory efficiency
class Config:
  d_model: int = 2560
  d_ff: int = 10240
  num_layers: int = 8
  num_heads: int = 32
  vocab_size: int = 128256
  seq_len: int = 1024
  dtype: Any = jnp.bfloat16

class FeedForward(nn.Module):
  config: Config
  dtype: Any = jnp.bfloat16

  @nn.compact
  def __call__(self, x: jax.Array):
    # All weights are sharded on their first dimension to be compatible with host offloading.
    # This is less performant than standard tensor parallelism but is required to avoid the compiler error.
    w1_kernel = partitioning.param_with_axes(
        'w1_kernel',
        nn.initializers.normal(),
        (self.config.d_model, self.config.d_ff),
        axes=PartitionSpec('model', None),
        dtype=self.dtype
    )
    w3_kernel = partitioning.param_with_axes(
        'w3_kernel',
        nn.initializers.normal(),
        (self.config.d_model, self.config.d_ff),
        axes=PartitionSpec('model', None),
        dtype=self.dtype
    )
    w2_kernel = partitioning.param_with_axes(
        'w2_kernel',
        nn.initializers.normal(),
        (self.config.d_ff, self.config.d_model),
        axes=PartitionSpec('model', None),
        dtype=self.dtype
    )

    w1 = x @ w1_kernel
    w3 = x @ w3_kernel
    y = nn.silu(w1) * w3
    w2 = y @ w2_kernel
    
    # The output of this sequence of operations with this sharding results in a replicated
    # activation. It must be explicitly re-sharded to match the expectation of the residual connection.
    return jax.lax.with_sharding_constraint(w2, PartitionSpec(None, None, 'model'))

class SelfAttention(nn.Module):
  config: Config
  dtype: Any = jnp.bfloat16

  @nn.compact
  def __call__(self, x: jax.Array, *, mask: Optional[jax.Array] = None):
    batch_size, seq_len, _ = x.shape
    qkv_features = self.config.d_model
    num_heads = self.config.num_heads
    if qkv_features % num_heads != 0:
      raise ValueError("d_model must be divisible by num_heads")
    head_dim = qkv_features // num_heads

    # All weights are sharded on their first dimension ('model') for offload compatibility.
    qkv_kernel_sharding = PartitionSpec('model', None)
    out_kernel_sharding = PartitionSpec('model', None)
    
    query_kernel = partitioning.param_with_axes(
        'query_kernel', nn.initializers.normal(), (self.config.d_model, qkv_features), axes=qkv_kernel_sharding, dtype=self.dtype
    )
    key_kernel = partitioning.param_with_axes(
        'key_kernel', nn.initializers.normal(), (self.config.d_model, qkv_features), axes=qkv_kernel_sharding, dtype=self.dtype
    )
    value_kernel = partitioning.param_with_axes(
        'value_kernel', nn.initializers.normal(), (self.config.d_model, qkv_features), axes=qkv_kernel_sharding, dtype=self.dtype
    )

    query = x @ query_kernel
    key = x @ key_kernel
    value = x @ value_kernel

    query = query.reshape(batch_size, seq_len, num_heads, head_dim)
    key = key.reshape(batch_size, seq_len, num_heads, head_dim)
    value = value.reshape(batch_size, seq_len, num_heads, head_dim)

    # Shard activations on the 'heads' dimension for efficient attention computation
    query = jax.lax.with_sharding_constraint(query, PartitionSpec(None, None, 'model', None))
    key = jax.lax.with_sharding_constraint(key, PartitionSpec(None, None, 'model', None))
    value = jax.lax.with_sharding_constraint(value, PartitionSpec(None, None, 'model', None))

    attn_values = nn.dot_product_attention(query, key, value, mask=mask)
    attn_values = attn_values.reshape(batch_size, seq_len, qkv_features)
    
    out_kernel = partitioning.param_with_axes(
        'out_kernel', nn.initializers.normal(), (qkv_features, self.config.d_model), axes=out_kernel_sharding, dtype=self.dtype
    )
    attn_output = attn_values @ out_kernel
    return attn_output

# THE FIX: @nn.remat is removed to avoid compiler conflicts with offloading and this sharding scheme.
class TransformerBlock(nn.Module):
  config: Config
  dtype: Any = jnp.bfloat16

  @nn.compact
  def __call__(self, x: jax.Array, *, mask: Optional[jax.Array] = None):
    x = jax.lax.with_sharding_constraint(x, PartitionSpec(None, None, 'model'))

    x_norm = nn.LayerNorm(name='ln1', dtype=self.dtype)(x)
    attn_output = SelfAttention(config=self.config, name='attention', dtype=self.dtype)(x_norm, mask=mask)
    x = x + attn_output

    x_norm = nn.LayerNorm(name='ln2', dtype=self.dtype)(x)
    ff_output = FeedForward(config=self.config, name='feed_forward', dtype=self.dtype)(x_norm)
    x = x + ff_output
    return x

class Embedding(nn.Module):
  config: Config
  dtype: Any = jnp.bfloat16

  @nn.compact
  def __call__(self, x: jax.Array):
    embedding = partitioning.param_with_axes(
        'embedding',
        nn.initializers.normal(),
        (self.config.vocab_size, self.config.d_model),
        axes=PartitionSpec('model', None), # Shard on vocab
        dtype=self.dtype
    )
    return embedding[x]

class Transformer(nn.Module):
  config: Config
  dtype: Any = jnp.bfloat16

  @nn.compact
  def __call__(self, x: jax.Array):
    
    input_tokens = x
    x = Embedding(config=self.config, name='embed_in', dtype=self.dtype)(input_tokens)
    x = jax.lax.with_sharding_constraint(x, PartitionSpec(None, None, 'model'))

    mask = nn.make_causal_mask(input_tokens)

    for i in range(self.config.num_layers):
      x = TransformerBlock(config=self.config, name=f'transformer_block_{i}', dtype=self.dtype)(x, mask=mask)

    x = nn.LayerNorm(name='final_ln', dtype=self.dtype)(x)
    
    embed_out_kernel = partitioning.param_with_axes(
        'embed_out_kernel',
        nn.initializers.normal(),
        (self.config.d_model, self.config.vocab_size),
        axes=PartitionSpec('model', None), # Shard on d_model
        dtype=self.dtype
    )
    logits = x @ embed_out_kernel
    return logits

key = jax.random.PRNGKey(0)
config = Config()

dp_size = 1
mp_size = jax.device_count()
assert mp_size > 1, "This script requires multiple devices for model parallelism."

devices = mesh_utils.create_device_mesh((dp_size, mp_size))
mesh = Mesh(devices, axis_names=('data', 'model'))
print(f"Using {mp_size}-way tensor parallelism.")

dummy_input = jax.random.randint(key, (1, config.seq_len), 0, config.vocab_size)
dummy_target = jax.random.randint(key, (1, config.seq_len), 0, config.vocab_size)

model = Transformer(config)
tx = optax.adam(1e-3)

def create_train_state(rngs, model_cls, tx_cls, dummy_input):
    """Creates initial `TrainState`."""
    variables = model_cls.init(rngs, dummy_input)
    return train_state.TrainState.create(
        apply_fn=model_cls.apply,
        params=variables['params'],
        tx=tx_cls,
    )

with mesh:
    abstract_state = jax.eval_shape(lambda: create_train_state(key, model, tx, dummy_input))

params_spec = nn.get_partition_spec(abstract_state.params)

def get_sharding(spec_tree, memory_kind='device'):
    return jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec, memory_kind=memory_kind), spec_tree)

opt_state_sharding = (
    optax.ScaleByAdamState(
        count=NamedSharding(mesh, PartitionSpec()),
        mu=get_sharding(params_spec),
        nu=get_sharding(params_spec),
    ),
    optax.EmptyState()
)
opt_state_sharding_host = (
    optax.ScaleByAdamState(
        count=NamedSharding(mesh, PartitionSpec()),
        mu=get_sharding(params_spec, memory_kind='pinned_host'),
        nu=get_sharding(params_spec, memory_kind='pinned_host'),
    ),
    optax.EmptyState()
)

state_sharding = train_state.TrainState(
    step=NamedSharding(mesh, PartitionSpec()),
    apply_fn=abstract_state.apply_fn,
    params=get_sharding(params_spec),
    tx=abstract_state.tx,
    opt_state=opt_state_sharding
)
state_sharding_host = train_state.TrainState(
    step=NamedSharding(mesh, PartitionSpec()),
    apply_fn=abstract_state.apply_fn,
    params=get_sharding(params_spec, memory_kind='pinned_host'),
    tx=abstract_state.tx,
    opt_state=opt_state_sharding_host
)
data_sharding = NamedSharding(mesh, PartitionSpec()) 

jit_create_train_state = jax.jit(
    create_train_state,
    static_argnums=(1, 2),
    in_shardings=NamedSharding(mesh, PartitionSpec()),
    out_shardings=state_sharding_host
)

with mesh:
  state = jit_create_train_state(key, model, tx, dummy_input)


@functools.partial(jax.jit, 
                   in_shardings=(state_sharding_host, data_sharding, data_sharding), 
                   out_shardings=(state_sharding_host, NamedSharding(mesh, PartitionSpec())))
def train_step(state, x, y):
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, x)
    # Cast logits to float32 for stable loss calculation
    logits = logits.astype(jnp.float32)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    return loss
  # Move state from host to device for computation
  state = jax.device_put(state, state_sharding)
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  # Move state back to host
  state = jax.device_put(state, state_sharding_host)
  return state, loss

print("Starting training...")
with mesh:
  dummy_input = jax.device_put(dummy_input, data_sharding)
  dummy_target = jax.device_put(dummy_target, data_sharding)

  # Run a few steps to demonstrate it works
  for step in range(5):
    state, loss = train_step(state, dummy_input, dummy_target)
    print(f"Step={state.step}, Loss: {loss}")
    
  # Profiling section
  jax.profiler.start_trace("/tmp/tensorboard")
  for step in range(5, 10):
    state, loss = train_step(state, dummy_input, dummy_target)
    print(f"Step={state.step}, Loss: {loss}")
  jax.profiler.stop_trace()

print("\nTraining finished successfully with host offloading enabled.")
