import jax
import functools
import numpy as np
import jax.numpy as jnp
from flax import nnx
from flax.linen import partitioning
from flax.training import train_state
from jax.sharding import PartitionSpec, Mesh, NamedSharding
import optax
from jax.experimental import mesh_utils
import datetime
import jax.profiler
import gzip


from typing import Sequence, Optional, Callable

# 4B Parameter Transformer Model
class Config:
  d_model: int = 3072
  d_ff: int = 10752
  num_layers: int = 8
  num_heads: int = 32
  vocab_size: int = 128256
  seq_len: int = 1024

class FeedForward(nnx.Module):
  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    # kernel_axes={'kernel': ('data', None)} for w1, w3
    self.w1 = nnx.Linear(
      config.d_model, config.d_ff, use_bias=False,
      kernel_init=nnx.with_metadata(nnx.initializers.normal(), sharding=PartitionSpec('data', None)),
      rngs=rngs
    )
    self.w3 = nnx.Linear(
      config.d_model, config.d_ff, use_bias=False,
      kernel_init=nnx.with_metadata(nnx.initializers.normal(), sharding=PartitionSpec('data', None)),
      rngs=rngs
    )
    # kernel_axes={'kernel': (None, 'data')} for w2
    self.w2 = nnx.Linear(
      config.d_ff, config.d_model, use_bias=False,
      kernel_init=nnx.with_metadata(nnx.initializers.normal(), sharding=PartitionSpec(None, 'data')),
      rngs=rngs
    )

  def __call__(self, x: jax.Array):
    return self.w2(nnx.silu(self.w1(x)) * self.w3(x))

class SelfAttention(nnx.Module):
  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    self.qkv_features = self.config.d_model
    self.num_heads = self.config.num_heads
    if self.qkv_features % self.num_heads != 0:
      raise ValueError("d_model must be divisible by num_heads")
    self.head_dim = self.qkv_features // self.num_heads

    # kernel_axes={'kernel': ('data', None)} for query, key, value
    qkv_kernel_sharding = PartitionSpec('data', None)
    self.query = nnx.Linear(config.d_model, self.qkv_features, use_bias=False, kernel_init=nnx.with_metadata(nnx.initializers.normal(), sharding=qkv_kernel_sharding), rngs=rngs)
    self.key   = nnx.Linear(config.d_model, self.qkv_features, use_bias=False, kernel_init=nnx.with_metadata(nnx.initializers.normal(), sharding=qkv_kernel_sharding), rngs=rngs)
    self.value = nnx.Linear(config.d_model, self.qkv_features, use_bias=False, kernel_init=nnx.with_metadata(nnx.initializers.normal(), sharding=qkv_kernel_sharding), rngs=rngs)

    # kernel_axes={'kernel': (None, 'data')} for out
    out_kernel_sharding = PartitionSpec(None, 'data')
    self.out   = nnx.Linear(self.qkv_features, config.d_model, use_bias=False, kernel_init=nnx.with_metadata(nnx.initializers.normal(), sharding=out_kernel_sharding), rngs=rngs)

  def __call__(self, x: jax.Array, *, mask: Optional[jax.Array] = None):
    batch_size, seq_len, _ = x.shape

    query = self.query(x)
    key = self.key(x)
    value = self.value(x)

    query = query.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    key = key.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    value = value.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

    attn_values = nnx.dot_product_attention(query, key, value, mask=mask)

    attn_values = attn_values.reshape(batch_size, seq_len, self.qkv_features)
    attn_output = self.out(attn_values)
    return attn_output

class TransformerBlock(nnx.Module):
  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    # LayerNorm scale/bias are typically replicated, so no sharding spec added.
    self.ln1 = nnx.LayerNorm(num_features=config.d_model, rngs=rngs)
    self.attention = SelfAttention(config, rngs=rngs)
    self.ln2 = nnx.LayerNorm(num_features=config.d_model, rngs=rngs)
    self.feed_forward = FeedForward(config, rngs=rngs)

  def __call__(self, x: jax.Array, *, mask: Optional[jax.Array] = None):
    # Activation sharding: (batch, seq, model_dim)
    x = jax.lax.with_sharding_constraint(x, PartitionSpec(None, None, 'data'))

    x_norm = self.ln1(x)
    attn_output = self.attention(x_norm, mask=mask)
    x = x + attn_output

    x_norm = self.ln2(x)
    ff_output = self.feed_forward(x_norm)
    x = x + ff_output
    return x

class Embedding(nnx.Module):
  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    embedding_shape = (self.config.vocab_size, self.config.d_model)
    # axes=(None, 'data')
    sharding = PartitionSpec(None, 'data')
    self.embedding = nnx.Param(
        nnx.with_metadata(nnx.initializers.normal(), sharding=sharding)(rngs.params(), embedding_shape)
    )

  def __call__(self, x: jax.Array):
    return self.embedding.value[x]

class Transformer(nnx.Module):
  def __init__(self, config: Config, *, rngs: nnx.Rngs):
    self.config = config
    self.embed_in = Embedding(config, rngs=rngs)

    keys = jax.random.split(rngs.params(), config.num_layers)
    block_rngs = [nnx.Rngs(params=key) for key in keys]
    self.transformer_blocks = [
        TransformerBlock(config, rngs=block_rngs[i]) for i in range(config.num_layers)
    ]
    self.final_ln = nnx.LayerNorm(num_features=config.d_model, rngs=rngs)

    # embed_out: features=self.config.vocab_size, kernel_axes={'kernel': ('data', None)})
    self.embed_out = nnx.Linear(
        config.d_model, config.vocab_size, use_bias=False,
        kernel_init=nnx.with_metadata(nnx.initializers.normal(), sharding=PartitionSpec('data', None)),
        rngs=rngs
    )

  def __call__(self, x: jax.Array):
    
    input_tokens = x
    x = self.embed_in(input_tokens)
    # Activation sharding: (batch, seq, model_dim)
    x = jax.lax.with_sharding_constraint(x, PartitionSpec(None, None, 'data'))

    mask = nnx.make_causal_mask(input_tokens)

    for block in self.transformer_blocks:
      x = block(x, mask=mask)

    x = self.final_ln(x)
    logits = self.embed_out(x)
    return logits

key = jax.random.key(0)
rngs = nnx.Rngs(params=key)
config = Config()
devices = mesh_utils.create_device_mesh((jax.device_count(),))
print(f"Devices: {devices}")
devices = np.array(jax.devices())
mesh = Mesh(devices, axis_names=('data',))

dummy_input = jax.random.randint(key, (1, config.seq_len), 0, config.vocab_size)
dummy_target = jax.random.randint(key, (1, config.seq_len), 0, config.vocab_size)

# # Define a simple loss function
# def loss_fn(params, apply_fn, x, y):
#     logits = apply_fn({'params': params}, x)
#     # Mean of per-token cross-entropy loss
#     return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), {}



# Create dummy objects to infer sharding specs

model_template = Transformer(config, rngs=rngs)
optimizer_template = nnx.Optimizer(model_template, optax.adam(1e-3), wrt=nnx.Param)
output_template = (model_template, optimizer_template)

# Get the PartitionSpec pytree from the dummy objects
partition_specs_model = nnx.spmd.get_partition_spec(model_template)
partition_specs_optimizer = nnx.spmd.get_partition_spec(optimizer_template)

# Create the out_shardings pytree
# Note: None in partition_specs means replicated, which corresponds to an empty PartitionSpec()
sharding_model = jax.tree_util.tree_map(
    lambda spec: NamedSharding(mesh, spec),
    partition_specs_model
)
sharding_optimizer = jax.tree_util.tree_map(
    lambda spec: NamedSharding(mesh, spec),
    partition_specs_optimizer
)


@functools.partial(nnx.jit, out_shardings=(sharding_model, sharding_optimizer))
def create_model_and_optimizer(rngs):
  model = Transformer(config, rngs=rngs)
  optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
  return model, optimizer

# with mesh:
model, optimizer = create_model_and_optimizer(rngs)
breakpoint()
compressed_profile = jax.profiler.device_memory_profile()

# Decompress the byte string
decompressed_profile = gzip.decompress(compressed_profile)

# Decode from bytes to a string and print
print(decompressed_profile.decode())

breakpoint()

@nnx.jit
def train_step(model, optimizer, x, y):
  def loss_fn(model):
    y_pred = model(x)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y_pred, y))

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(model, grads)  # In place updates.

  return loss

print("Starting training...")
with mesh:
  for step in range(5):
    loss = train_step(model, optimizer, dummy_input, dummy_target)
    print(f"Step={optimizer.step.value}, Loss: {loss}")

# tx = optax.adamw(learning_rate=0.01)
# def init_state(model):
#     params = model.init(key, dummy_input)['params']
#     state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)  # type: ignore
#     return state

# init_state_partial = functools.partial(init_state, model)
# init_state_partial.__name__ = "initialize_state"
# logical_axis_rules = nnx.spmd.get_logical_axis_rules(mesh)

# sharding = NamedSharding(mesh, partition_spec)

# abstract_state = jax.eval_shape(init_state_partial)
# with nn.partitioning.axis_rules(config.logical_axis_rules):
#     state_logical_annotations = nn.get_partition_spec(abstract_state)
# state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh)
# replicated_sharding = NamedSharding(mesh, PartitionSpec())

# change state's sharding to memory_kind pinned_host before initialization
# state_mesh_shardings = state_mesh_shardings.replace(params = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="device"), state_mesh_shardings.params),
#                                                     opt_state = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="device"), state_mesh_shardings.opt_state))

# state = jax.jit(init_state_partial, out_shardings=state_mesh_shardings)()

# def move_state(target_kind, state, state_shardings):
#   dest_shardings = jax.tree_util.tree_map(lambda s: s.with_memory_kind(target_kind), state_shardings)

#   new_params = jax.tree_util.tree_map(
#       lambda x, s: jax.device_put(x, s),
#       state.params,
#       dest_shardings.params
#   )
#   new_opt_state = jax.tree_util.tree_map(
#       lambda x, s: jax.device_put(x, s),
#       state.opt_state,
#       dest_shardings.opt_state
#   )

#   return state.replace(params=new_params, opt_state=new_opt_state)

# def p_train_step_creator(state_mesh_shardings):
#   def train_step(state, dummy_input, dummy_target):
#       moved_state = move_state('device', state, state_mesh_shardings)
#       (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(moved_state.params, state.apply_fn, dummy_input, dummy_target)
#       new_state_on_device = moved_state.apply_gradients(grads=grads)
#       final_state = move_state('pinned_host', new_state_on_device, state_mesh_shardings)
#       return final_state, loss

#   return jax.jit(
#       train_step,
#       in_shardings=(state_mesh_shardings, replicated_sharding, replicated_sharding),
#       out_shardings=(state_mesh_shardings, replicated_sharding),
#       donate_argnums=(0,),
#   )

# p_train_step = p_train_step_creator(state_mesh_shardings)
# last_step_completion = datetime.datetime.now()
# print("Starting training...")
# for step in range(5):
#   state, _ = p_train_step(state, dummy_input, dummy_target)
#   step_time_delta = datetime.datetime.now() - last_step_completion
#   last_step_completion = datetime.datetime.now()
#   print(f"Step {step} completed in {step_time_delta.total_seconds()} seconds")

# jax.profiler.start_trace("/tmp/tensorboard")
# for _ in range(3):
#     state, _ = p_train_step(state, dummy_input, dummy_target)
# state.step.block_until_ready()
# jax.profiler.stop_trace()