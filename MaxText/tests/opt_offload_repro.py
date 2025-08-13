import jax
import functools
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from jax._src.sharding_impls import TransferToMemoryKind
from jax.sharding import PartitionSpec, Mesh, NamedSharding
import optax
from jax.experimental import mesh_utils
import datetime

# Simple Dummy Model
class DummyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=(4), use_bias=False)(x)  # Simple dense layer

key = jax.random.PRNGKey(0)
dummy_model = DummyModel()
dummy_input = jnp.ones((1, 4))  # Dummy input shape (1,4)
dummy_target = jnp.ones((1, 4))

# Define a simple loss function
def loss_fn(params, apply_fn, x, y):
    preds = apply_fn(params, x)  # Forward pass
    return jnp.mean((preds - y) ** 2), {}  # MSE loss

tx = optax.adamw(learning_rate=0.01)
def init_state(model):
    params = model.init(key, dummy_input)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)  # type: ignore
    return state

init_state_partial = functools.partial(init_state, dummy_model)

devices = mesh_utils.create_device_mesh((jax.device_count(),))
partition_spec = PartitionSpec()
mesh = Mesh(devices, ("data",))
sharding = NamedSharding(mesh, partition_spec)

abstract_state = jax.eval_shape(init_state_partial)
state_logical_annotations = nn.get_partition_spec(abstract_state)
state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh, [])
replicated_sharding = NamedSharding(mesh, PartitionSpec())

state = init_state_partial()
# change state's sharding to memory_kind pinned_host
state_mesh_shardings = state_mesh_shardings.replace(params = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="pinned_host"), state_mesh_shardings.params),
                                                    opt_state = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="pinned_host"), state_mesh_shardings.opt_state))

state = jax.device_put(state, state_mesh_shardings)

def move_state(sharding_memory_kind, state):
    def my_scan(xs):
        def body(i, data):
            def put(arr):
                start_indices = (i,) + (0,) * (arr.ndim - 1)
                slice_sizes = (1,) + arr.shape[1:]
                if arr.ndim < 2:
                    return jax.device_put(arr, sharding_memory_kind)
                y = jax.device_put(jax.lax.dynamic_slice(arr, start_indices, slice_sizes), sharding_memory_kind)
                return jax.lax.dynamic_update_slice(arr, y, start_indices)

            return jax.tree_util.tree_map(put, data)
        return jax.lax.fori_loop(0, 4, body, xs, unroll=False)
    params, opt_state = my_scan((state.params, state.opt_state))
    state = state.replace(params = params, opt_state = opt_state)
    return state

def train_step(state, dummy_input, dummy_target):
    state = move_state(TransferToMemoryKind('device'), state)
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, state.apply_fn, dummy_input, dummy_target)
    new_state = state.apply_gradients(grads=grads)
    new_state = move_state(TransferToMemoryKind('pinned_host'), new_state)
    return new_state, loss

p_train_step = jax.jit(
    train_step,
    in_shardings=(state_mesh_shardings, replicated_sharding, replicated_sharding),
    out_shardings=(state_mesh_shardings, replicated_sharding),
    donate_argnums=(0,),
)

for _ in range(5):
    state, _ = p_train_step(state, dummy_input, dummy_target)