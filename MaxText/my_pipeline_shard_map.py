import jax
from jax import numpy as jnp
from functools import partial

from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils


def predict(params, inputs):
  for layer in params:
    inputs = jnp.dot(inputs, layer)
    inputs = jax.nn.relu(inputs)
  return inputs

def loss(params, batch):
  inputs, targets = batch
  predictions = predict(params, inputs)
  return jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))

def init_layer(key, embed_size):
    W = jax.random.normal(key, (embed_size, embed_size)) / jnp.sqrt(embed_size)
    return W

def init(key_init, num_layers, embed_size, batch_size):
    keys = jax.random.split(key_init, num_layers)
    params = [init_layer(key, embed_size) for key in keys]

    input_key, target_key = jax.random.split(key_init, 2)
    inputs = jax.random.normal(input_key, (batch_size, embed_size))
    targets = jax.random.normal(target_key, (batch_size, embed_size))

    return params, (inputs, targets)

num_layers = 4
embed_size = 1024
batch_size = 32

params, batch = init(jax.random.PRNGKey(0), num_layers, embed_size, batch_size)
print(f"input size is {batch[0].shape}")

### Start pipeline parallelism ###

L = len(params)       # num layers
N = batch_size             # batch size
F = embed_size

# choose some pipeline parameters
S = 4      # number of stages
B = 8      # size of each microbatch
assert L == S, "Number of layers must equal the number of stages"
#assert L % S == 0, "S (number of stages) must divide L (number of inner layers)"

# compute some useful quantities
M, ragged = divmod(N, B)  # M is number of microbatches
assert not ragged, "B (size of each microbatch) must divide total batch size"
K, ragged = divmod(M, S)  # K is microbatches per stage
assert not ragged, "S (number of stages) must divide number of microbatches"
print(f'{S} stages, {L // S} layer(s) per stage, {L} pipelined layers total')
print(f'{B} examples per microbatch, {M} microbatches total')


mesh = Mesh(jax.devices(), ('stages',))

def stage_fn(layer, inputs):
  inputs = jnp.dot(inputs, layer)
  return jax.nn.relu(inputs)

def predict_pp(params, inputs):
  outputs = spmd_pipeline(stage_fn, params, inputs)
  return outputs

# @partial(shard_map, mesh=mesh, in_specs=((P(), P('stages'), P()), P('stages')),
#          out_specs=P())
@partial(shard_map, mesh=mesh, in_specs=((P('stages')), P('stages')),
         out_specs=P())
def loss_pp(params, batch):
  inputs, targets = batch
  predictions = predict_pp(params, inputs.reshape(K, B, -1)).reshape(K * B, -1)
  local_loss = jnp.mean(jnp.sum((predictions - targets)**2, axis=-1))
  return jax.lax.pmean(local_loss, 'stages')

def spmd_pipeline(fn, stage_params, inputs):
  stage = jax.lax.axis_index('stages')
  outputs = jnp.zeros_like(inputs) * jnp.nan
  #state = jnp.zeros((L // S, B, F)) * jnp.nan
  state = jnp.zeros((B, F)) * jnp.nan
  for i in range(M+L-1):
    # Using jnp.where with axis_index creates a float32[2,8,128;stages:2] object - 
    # I believe this means each device actually has different data - its neither
    # sharded nor replicated
    # TODO simplifying this API
    #state = state.at[0].set(jnp.where(stage == 0, inputs[i % K], state[0]))
    state = state.at[:,:].set(jnp.where(stage == 0, inputs[i % K], state[:,:]))
    # This is vmapping over layers per stage? I don't like this so I'm keeping one layer per stage for now
    #state = jax.vmap(fn)(stage_params, state)
    state = fn(stage_params[0], state) # state = fn(stage_params, state)
    outputs = outputs.at[(i-L+1) % K].set(jnp.where(stage == S-1, state, outputs[(i-L+1) % K]))
    state, inputs, outputs = shift(i, state, inputs, outputs)
  outputs = jax.lax.ppermute(outputs, 'stages', [(i, (i+1) % S) for i in range(S)])
  return outputs

def shift(i, state, inputs, outputs):
  sh = lambda x, d: jax.lax.ppermute(x, 'stages', [(i, (i+d) % S) for i in range(S)])
  # jnp or np.roll shifts elements of an array, e.g. [0,1,2] -> [2,0,1]
  # Roll moves the activations from within-stage layer i to layer i+1 on the same device
  # The shift moves the last layer on the device to the first layer on the next device
  # The order of operations is a bit hard to parse, but the shift is on the pre-rolled state, so that
  # the activations from the last stage are still on the last stage instead of rolled around dummily to the first

  #state = jnp.roll(state, +1, axis=0).at[0].set(sh(state[-1], +1))
  state = sh(state, +1)
  # In pax code we roll the specific ms index every loop iteration
  # Instead we could roll every ms index after every`` K=|ms| loop iterations
  if (i % K) == (-1 % K):
    inputs = sh(inputs, +1)
  if ((i-L+1) % K) == (-1 % K):
    outputs = sh(outputs, +1)
  return state, inputs, outputs

params_stacked = jnp.stack(params)
params_sharded = jax.device_put(params_stacked, NamedSharding(mesh, P('stages')))
batch_sharded = jax.device_put(batch, NamedSharding(mesh, P('stages')))

print(jax.jit(loss)(params, batch))
print(jax.jit(loss_pp)(params_sharded, batch_sharded))