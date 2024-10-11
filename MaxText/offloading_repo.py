# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import jax
import jax.sharding
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
import numpy as np

import argparse
from flax import linen as nn
from jax.sharding import PartitionSpec as P
from flax.training import train_state
import optax
import tensorflow as tf
from multihost_dataloading import MultiHostDataLoadIterator

def with_memory_kind(t, memory_kind):
  return jax.tree_util.tree_map(
      lambda x: x.with_memory_kind(kind=memory_kind), t
  )

dtype = jnp.bfloat16

def cast_dtype_from_to(nest, src, dst):
  """All items in nest with dtype src are casted to dtype dst."""
  return jax.tree_util.tree_map(
      lambda t: t.astype(dst) if t.dtype == src else t, nest
  )

def cast_to_bf16(params):
    return cast_dtype_from_to(params, np.float32, jnp.bfloat16)

class DummyModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        # A simple linear layer
        for i in range(10):
          x = nn.Dense(features=12376)(x)
        return x

def get_abstract_state(model, tx, mesh, rng):
    init_state_partial = functools.partial(init_intial_state, model, tx)
    abstract_state = jax.eval_shape(init_state_partial, rng)
    state_logical_annotations = nn.get_partition_spec(abstract_state)
    state_mesh_shardings = nn.logical_to_mesh_sharding(
        state_logical_annotations, mesh
    )
    return state_mesh_shardings

def init_intial_state(model, tx, rng):
    dummy_input = jnp.arange(0, 12376, dtype = dtype)
    # Initialize parameters
    params = model.init(rng, dummy_input)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return state

def create_train_state(model, optimizer, mesh, rng):

    state_mesh_shardings = get_abstract_state(model, optimizer, mesh, rng)
    # Create the TrainState
    init_state_partial = functools.partial(init_intial_state, model, optimizer)
    state = jax.jit(
        init_state_partial,
        in_shardings=None,
        out_shardings=state_mesh_shardings,
    )(rng)
    return state, state_mesh_shardings


def create_random_global_array(rng, global_shape, sharding, dtype):
  local_tensor_shape = sharding.shard_shape(global_shape)
  local_tensor = jax.random.normal(rng, shape=local_tensor_shape, dtype=dtype)
  random_global_array = jax.make_array_from_single_device_arrays(
      global_shape,
      sharding,
      [jax.device_put(local_tensor, d) for d, index in sharding.addressable_devices_indices_map(global_shape).items()],
  ).astype(dtype)
  return random_global_array

# train_loop is called from main

def train_loop(output_path, offload):
    # setting  up model and mesh
    rng = jax.random.PRNGKey(0)
    model = DummyModel()

    grid = np.array(jax.devices()).reshape((16, 16))
    mesh = jax.sharding.Mesh(grid, ('data', 'model'))

    # creating train_state
    # Define an optimizer
    learning_rate = 0.001
    weight_decay = 1e-4
    optimizer = optax.adam(learning_rate=1e-3)
    state, state_mesh_shardings = create_train_state(model, optimizer, mesh, rng)
    if offload:
        params = jax.device_put(
            state.params, with_memory_kind(state_mesh_shardings.params, 'pinned_host')
        )
        opt_state = jax.device_put(
            state.opt_state, with_memory_kind(state_mesh_shardings.opt_state, 'pinned_host')
        )
        state = state.replace(params=params, opt_state=opt_state)

    data_pspec = P(('data','model',))
    data_sharding = jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), data_pspec)
    local_batch_size = 1
    data = create_random_global_array(
        jax.random.PRNGKey(0),
        global_shape=(local_batch_size * len(jax.devices()), 12376),
        sharding=data_sharding,
        dtype=jnp.bfloat16,
    )
    data = jax.device_put(data, with_memory_kind(data_sharding, 'device'))

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def optimizer_apply(params, opt_state, grads):
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state


    def train_step(model, state, batch, offload):
        cast_params = cast_dtype_from_to(state.params, np.float32, jnp.bfloat16)
        if offload:
            cast_params = jax.device_put(
                cast_params, with_memory_kind(state_mesh_shardings.params, 'device')
            )
        else:
            cast_params = state.params
        def loss_fn(model, batch, params):
            predictions = model.apply(
                params,
                batch
            )
            return jnp.mean((predictions - batch))
        grad_func = jax.value_and_grad(loss_fn, argnums=2)
        _, grad = grad_func(model, batch, cast_params)
        if offload:
            params = jax.device_put(
                state.params, with_memory_kind(state_mesh_shardings.params, 'device')
            )
            opt_state = jax.device_put(
                state.opt_state, with_memory_kind(state_mesh_shardings.opt_state, 'device')
            )
        params, opt_state = optimizer_apply(params, opt_state, grad)
        if offload:
            params = jax.device_put(
                params, with_memory_kind(state_mesh_shardings.params, 'pinned_host')
            )
            opt_state = jax.device_put(
                opt_state, with_memory_kind(state_mesh_shardings.opt_state, 'pinned_host')
            )

        new_state = state.replace(params=params, opt_state=opt_state)
        return new_state

    p_train_step = jax.jit(train_step,
            in_shardings=(state_mesh_shardings, data_sharding),
            out_shardings=state_mesh_shardings,
            donate_argnums=(1,),
            static_argnums=(0,3))


    for step in range(10):
        if step == 5:
            jax.profiler.start_trace(output_path)
        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            state = p_train_step(model, state, data, offload)

    jax.profiler.stop_trace()
    print(f"Profile saved at {output_path}")

if __name__ == "__main__":
    jax.config.update('jax_enable_memories', True)
    parser = argparse.ArgumentParser()

    parser.add_argument("--offload", type=bool, default=True)
    parser.add_argument("--output_path", type=str, required=True)
    test_args, _ = parser.parse_known_args()

    train_loop(test_args.output_path, test_args.offload)
