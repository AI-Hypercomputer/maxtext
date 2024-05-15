import jax
from jax import numpy as jnp
from functools import partial

from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils


def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jax.nn.relu(outputs)
  return outputs

def loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))

def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = jax.random.normal(k2, (n_out,))
    return W, b

def init(key, layer_sizes, batch_size):
    key, *keys = jax.random.split(key, len(layer_sizes))
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))

    key, *keys = jax.random.split(key, 3)
    inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))

    return params, (inputs, targets)

layer_sizes = [784, 128, 128, 128, 128, 128, 128, 128, 8]
batch_size = 32

params, batch = init(jax.random.PRNGKey(0), layer_sizes, batch_size)
print(f"input size is {batch[0].shape}")

### Start pipeline parallelism ###

L = len(params) - 2        # num layers, excluding first and last
N = batch_size             # batch size
F = params[0][0].shape[1]  # num features

# choose some pipeline parameters
S = 2      # number of stages
B = 8      # size of each microbatch
assert L % S == 0, "S (number of stages) must divide L (number of inner layers)"

# compute some useful quantities
M, ragged = divmod(N, B)  # M is number of microbatches
assert not ragged, "B (size of each microbatch) must divide total batch size"
K, ragged = divmod(M, S)  # K is microbatches per stage
assert not ragged, "S (number of stages) must divide number of microbatches"
print(f'{S} stages, {L // S} layer(s) per stage, {L} pipelined layers total')
print(f'{B} examples per microbatch, {M} microbatches total')


mesh = Mesh(jax.devices()[:S], ('stages',))

def predict_pp(params, inputs):
  (W_first, b_first), inner_params, (W_last, b_last) = params
  inputs = jax.nn.relu(jnp.dot(inputs, W_first) + b_first)
  inputs = spmd_pipeline(lambda Wb, x: jax.nn.relu(x @ Wb[0] + Wb[1]),
                        inner_params, inputs)
  outputs = jnp.dot(inputs, W_last) + b_last
  return outputs

@partial(shard_map, mesh=mesh, in_specs=((P(), P('stages'), P()), P('stages')),
         out_specs=P())
def loss_pp(params, batch):
  inputs, targets = batch
  predictions = predict_pp(params, inputs.reshape(K, B, -1)).reshape(K * B, -1)
  local_loss = jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))
  return jax.lax.pmean(local_loss, 'stages')

def spmd_pipeline(fn, stage_params, inputs):
  stage = jax.lax.axis_index('stages')
  outputs = jnp.zeros_like(inputs) * jnp.nan
  state = jnp.zeros((L // S, B, F)) * jnp.nan
  for i in range(M+L-1):
    # Using jnp.where with axis_index creates a float32[2,8,128;stages:2] object - 
    # I believe this means each device actually has different data - its neither
    # sharded nor replicated
    state = state.at[0].set(jnp.where(stage == 0, inputs[i % K], state[0]))
    #breakpoint()
    # This is vmapping over layers per stage?
    state = jax.vmap(fn)(stage_params, state)
    outputs = outputs.at[(i-L+1) % K].set(jnp.where(stage == S-1, state[-1], outputs[(i-L+1) % K]))
    state, inputs, outputs = shift(i, state, inputs, outputs)
  outputs = jax.lax.ppermute(outputs, 'stages', [(i, (i+1) % S) for i in range(S)])
  return outputs

def shift(i, state, inputs, outputs):
  sh = lambda x, d: jax.lax.ppermute(x, 'stages', [(i, (i+d) % S) for i in range(S)])
  # jnp or np.roll shifts elements of an array, e.g. [0,1,2] -> [2,0,1]
  rolled = jnp.roll(state, +1, axis=0)
  rolled_0 = jnp.roll(state, +1, axis=0).at[0]
  state = jnp.roll(state, +1, axis=0).at[0].set(sh(state[-1], +1))
  # In pax code we roll the specific ms index every loop iteration
  # Instead we could roll every ms index after every K=|ms| loop iterations
  if (i % K) == (-1 % K):
    inputs = sh(inputs, +1)
  if ((i-L+1) % K) == (-1 % K):
    outputs = sh(outputs, +1)
  return state, inputs, outputs

first_params, *inner_params, last_params = params
Ws, bs = zip(*inner_params)
params_stacked = jnp.stack(Ws), jnp.stack(bs)
first_params = jax.device_put(first_params, NamedSharding(mesh, P()))
params_stacked = jax.device_put(params_stacked, NamedSharding(mesh, P('stages')))
last_params = jax.device_put(last_params, NamedSharding(mesh, P()))
params_ = first_params, params_stacked, last_params

batch_ = jax.device_put(batch, NamedSharding(mesh, P('stages')))

print(jax.jit(loss)(params, batch))
print(jax.jit(loss_pp)(params_, batch_))