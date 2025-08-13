"""A simple transformer training program using host offload."""

import functools
from typing import Any, Mapping
import argparse
from absl import flags

from absl import logging
import jax
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
import numpy as np
import optax

import transformer

def with_memory_kind(t, memory_kind):
  return jax.tree_util.tree_map(
      lambda x: x.with_memory_kind(kind=memory_kind), t
  )


def cast_dtype_from_to(nest, src, dst):
  """All items in nest with dtype src are casted to dtype dst."""
  return jax.tree_util.tree_map(
      lambda t: t.astype(dst) if t.dtype == src else t, nest
  )

def test_training(offload_state: bool, offload_compute: bool):
  config = transformer.SMALL_TRANSFORMER_CONFIG
  model = transformer.Transformer(cfg=config)
  logging.info('Param count = %d', config.count_params_wo_embedding())
  fake_data = transformer.FakeData(seed=25235235, batch_size=256, cfg=config)
  data = next(fake_data)
  params = jax.jit(model.init)(
      jax.random.PRNGKey(seed=25235235),
      data['observation'],
      data['input_mask'],
  )
  freeze_embeddings = False

  def optimizable(ps: Mapping[str, Mapping[str, Any]]):
    if freeze_embeddings:
      return {
          'params': {
              k: v for k, v in ps['params'].items() if k != 'embeddings'
          }
      }
    else:
      return ps

  def merge_frozen_params(
      new_params: Mapping[str, Mapping[str, Any]],
      old_params: Mapping[str, Mapping[str, Any]],
  ):
    if freeze_embeddings:
      return dict(
          params=dict(
              embeddings=old_params['params']['embeddings'],
              **new_params['params'],
          )
      )
    return new_params

  optimizer = optax.adam(learning_rate=1e-3)
  opt_state = optimizer.init(optimizable(params))

  # This example has a naive sharding that always splits
  # model parallel on the last axis. This is pretty unrealistic but
  # at least provides a model parallelism example for the test.
  grid = np.array(jax.devices()).reshape((2, 2))
  mesh = jax.sharding.Mesh(grid, ('data', 'model'))
  print(f"Mesh: {mesh}")
  def make_param_sharding(x):
    if not x.shape:
      return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    ps = [None] * len(x.shape)
    ps[-1] = 'model'
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*ps))

  param_sharding = jax.tree_util.tree_map(make_param_sharding, params)
  opt_state_sharding = jax.tree_util.tree_map(make_param_sharding, opt_state)
  optimizable_param_sharding = jax.tree_util.tree_map(
      make_param_sharding, optimizable(params)
  )

  if offload_state:
    param_sharding = with_memory_kind(param_sharding, 'pinned_host')
    opt_state_sharding = with_memory_kind(opt_state_sharding, 'pinned_host')

  params = jax.device_put(params, param_sharding)
  opt_state = jax.device_put(opt_state, opt_state_sharding)

  def optimizer_update(params, opt_state, grads):
    updates, opt_state = optimizer.update(grads, opt_state)
    updated_params = optax.apply_updates(optimizable(params), updates)
    params = merge_frozen_params(updated_params, params)
    return params, opt_state

  def cast_to_bf16(params):
    return cast_dtype_from_to(params, np.float32, jnp.bfloat16)

  def update(params, opt_state, data):
    if offload_state and not offload_compute:
      device_params = jax.device_put(
          params, with_memory_kind(param_sharding, 'device')
      )
      cast_params = cast_to_bf16(device_params)
    elif offload_state:
      cast_params = compute_on('device_host')(jax.jit(cast_to_bf16))(params)
      cast_params = jax.device_put(
          cast_params, with_memory_kind(param_sharding, 'device')
      )
    else:
      cast_params = params
    loss, grads = jax.value_and_grad(transformer.loss_fn(model))(
        cast_params, data
    )
    grads = optimizable(grads)
    if offload_state and offload_compute:
      grads = jax.device_put(
          grads, with_memory_kind(optimizable_param_sharding, 'pinned_host')
      )
    elif offload_state:
      params = jax.device_put(
          params, with_memory_kind(param_sharding, 'device')
      )
      opt_state = jax.device_put(
          opt_state, with_memory_kind(opt_state_sharding, 'device')
      )
    if offload_compute:
      params, opt_state = compute_on('device_host')(
          jax.jit(optimizer_update)
      )(params, opt_state, grads)
    else:
      params, opt_state = optimizer_update(params, opt_state, grads)
    if offload_state and not offload_compute:
      params = jax.device_put(
          params, with_memory_kind(param_sharding, 'pinned_host')
      )
      opt_state = jax.device_put(
          opt_state, with_memory_kind(opt_state_sharding, 'pinned_host')
      )
    return params, opt_state, loss

  data_sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec('data')
  )
  jitted_update = jax.jit(
      update,
      in_shardings=(param_sharding, opt_state_sharding, data_sharding),
      out_shardings=(param_sharding, opt_state_sharding, None),
      # donate_argnums=(0, 1),
  )

  num_steps = 5
  for i in range(num_steps):
    print(f'Step {i}')
    data = next(fake_data)
    data = jax.device_put(data, data_sharding)
    params, opt_state, loss = jax.block_until_ready(jitted_update(params, opt_state, data))


if __name__ == "__main__":
  jax.config.update('jax_enable_memories', True)
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_path", type=str, required=True)
  test_args, _ = parser.parse_known_args()
  jax.profiler.start_trace(test_args.output_path)
  test_training(offload_state=True, offload_compute=False)
  jax.profiler.stop_trace()
  print(f"Profile saved at {test_args.output_path}")