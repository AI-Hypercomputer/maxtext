import jax
import functools
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from jax.sharding import PartitionSpec, Mesh, NamedSharding
import optax
from jax.experimental import mesh_utils
import datetime
import jax.profiler

from typing import Sequence

# 8B Parameter Transformer Model
class Config:
    d_model: int = 4096
    d_ff: int = 14336
    num_layers: int = 32
    num_heads: int = 32
    vocab_size: int = 128256
    seq_len: int = 1024


class FeedForward(nn.Module):
    config: Config
    @nn.compact
    def __call__(self, x):
        w1 = nn.Dense(self.config.d_ff, use_bias=False)(x)
        w3 = nn.Dense(self.config.d_ff, use_bias=False)(x)
        w2 = nn.Dense(self.config.d_model, use_bias=False)(nn.silu(w1) * w3)
        return w2

class TransformerBlock(nn.Module):
    config: Config
    @nn.compact
    def __call__(self, x, mask=None):
        attn_output = nn.SelfAttention(
            num_heads=self.config.num_heads,
            qkv_features=self.config.d_model,
            use_bias=False
        )(x, mask)
        x = x + attn_output
        x = nn.LayerNorm()(x)
        
        ff_output = FeedForward(config=self.config)(x)
        x = x + ff_output
        x = nn.LayerNorm()(x)
        return x

class Transformer(nn.Module):
    config: Config
    @nn.compact
    def __call__(self, x, train=True):
        input_tokens = x
        x = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.d_model,
                     embedding_init=nn.initializers.normal(),
                     name='embed_in')(input_tokens)
        mask = nn.make_causal_mask(input_tokens)
        for _ in range(self.config.num_layers):
            x = TransformerBlock(config=self.config)(x, mask)
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.config.vocab_size, use_bias=False,
                          kernel_init=nn.initializers.normal(),
                          name='embed_out')(x)
        return logits

    def get_partition_spec(self):
        return {
            'params': {
                'embed_in': {'embedding': PartitionSpec('data', None)},
                'embed_out': {'kernel': PartitionSpec(None, 'data')},
                'TransformerBlock_0': {
                    'SelfAttention_0': {
                        'query': {'kernel': PartitionSpec(None, 'data')},
                        'key': {'kernel': PartitionSpec(None, 'data')},
                        'value': {'kernel': PartitionSpec(None, 'data')},
                        'out': {'kernel': PartitionSpec('data', None)},
                    },
                    'FeedForward_0': {
                        'w1': {'kernel': PartitionSpec(None, 'data')},
                        'w3': {'kernel': PartitionSpec(None, 'data')},
                        'w2': {'kernel': PartitionSpec('data', None)},
                    },
                },
            }
        }

key = jax.random.PRNGKey(0)
config = Config()
model = Transformer(config)
dummy_input = jax.random.randint(key, (1, config.seq_len), 0, config.vocab_size)
dummy_target = jax.random.randint(key, (1, config.seq_len), 0, config.vocab_size)

# Define a simple loss function
def loss_fn(params, apply_fn, x, y):
    logits = apply_fn(params, x)
    # Mean of per-token cross-entropy loss
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), {}

tx = optax.adamw(learning_rate=0.01)
def init_state(model):
    params = model.init(key, dummy_input)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)  # type: ignore
    return state

init_state_partial = functools.partial(init_state, model)

devices = mesh_utils.create_device_mesh((jax.device_count(),))
partition_spec = PartitionSpec()
mesh = Mesh(devices, ("data",))
sharding = NamedSharding(mesh, partition_spec)

abstract_state = jax.eval_shape(init_state_partial)
state_logical_annotations = model.get_partition_spec()
state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh, [])
replicated_sharding = NamedSharding(mesh, PartitionSpec())

state = init_state_partial()
# change state's sharding to memory_kind pinned_host
state_mesh_shardings = state_mesh_shardings.replace(params = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="pinned_host"), state_mesh_shardings.params),
                                                    opt_state = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="pinned_host"), state_mesh_shardings.opt_state))

state = jax.device_put(state, state_mesh_shardings)

def move_state(target_kind, state, state_shardings):
    dest_shardings = jax.tree_util.tree_map(lambda s: s.with_memory_kind(target_kind), state_shardings)

    new_params = jax.tree_util.tree_map(
        lambda x, s: jax.device_put(x, s),
        state.params,
        dest_shardings.params
    )
    new_opt_state = jax.tree_util.tree_map(
        lambda x, s: jax.device_put(x, s),
        state.opt_state,
        dest_shardings.opt_state
    )

    return state.replace(params=new_params, opt_state=new_opt_state)

def p_train_step_creator(state_mesh_shardings):
    def train_step(state, dummy_input, dummy_target):
        moved_state = move_state('device', state, state_mesh_shardings)
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(moved_state.params, state.apply_fn, dummy_input, dummy_target)
        new_state_on_device = moved_state.apply_gradients(grads=grads)
        final_state = move_state('pinned_host', new_state_on_device, state_mesh_shardings)
        return final_state, loss

    return jax.jit(
        train_step,
        in_shardings=(state_mesh_shardings, replicated_sharding, replicated_sharding),
        out_shardings=(state_mesh_shardings, replicated_sharding),
        donate_argnums=(0,),
    )

p_train_step = p_train_step_creator(state_mesh_shardings)

for _ in range(5):
    state, _ = p_train_step(state, dummy_input, dummy_target)

jax.profiler.start_trace("/tmp/tensorboard")
for _ in range(5):
    state, _ = p_train_step(state, dummy_input, dummy_target)
state.step.block_until_ready()
jax.profiler.stop_trace()