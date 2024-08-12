# python MaxText/attentions_test.py MaxText/configs/base.yml global_parameter_scale=128 per_device_batch_size=4

import jax
import os
import pyconfig
from layers import attentions
import max_utils
import max_logging
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import common_types
import numpy as np
from typing import Sequence
import profiler
from absl import app
import datetime
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from layers import quantizations
import math
import jax.numpy as jnp


Mesh = jax.sharding.Mesh
AttentionOp = attentions.AttentionOp
AxisNames = common_types.AxisNames


BATCH = common_types.BATCH
KV_BATCH = common_types.KV_BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
D_KV = common_types.D_KV
KV_HEAD_DIM = common_types.KV_HEAD_DIM


def calculate_attention_forward_tflops_per_device(config):
  """Calculates theoretical TFLOPs per device for attention forward pass."""
  attention_flops = (
      4  # 2 for Multiply-accumulate operations per element and 2 for qk and wv product
      * config.per_device_batch_size
      * config.max_target_length**2  # Quadratic complexity
      * config.num_query_heads
      * config.head_dim
      / 10**12  # Convert to TFLOPs
  )
  return attention_flops


def create_random_global_array(rng, global_shape, sharding, dtype):
  local_tensor_shape = sharding.shard_shape(global_shape)
  local_tensor = jax.random.normal(rng, shape=local_tensor_shape, dtype=jnp.float32)
  random_global_array = jax.make_array_from_single_device_arrays(
      global_shape,
      sharding,
      [jax.device_put(local_tensor, d) for d, index in sharding.addressable_devices_indices_map(global_shape).items()],
  ).astype(dtype)
  return random_global_array


def get_train_iter(config, mesh):
  """Generates an infinite stream of random query, key, value batches."""
  query_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  key_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  value_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  decoder_segment_ids_axis_names: AxisNames = (KV_BATCH, LENGTH)
  rng = jax.random.PRNGKey(0)
  query = create_random_global_array(
      rng,
      global_shape=(config.global_batch_size_to_train_on, config.max_target_length, config.num_query_heads, config.head_dim),
      sharding=NamedSharding(mesh, nn.logical_to_mesh_axes(query_axis_names)),
      dtype=config.dtype,
  )
  key = create_random_global_array(
      rng,
      global_shape=(config.global_batch_size_to_train_on, config.max_target_length, config.num_kv_heads, config.head_dim),
      sharding=NamedSharding(mesh, nn.logical_to_mesh_axes(key_axis_names)),
      dtype=config.dtype,
  )
  value = create_random_global_array(
      rng,
      global_shape=(config.global_batch_size_to_train_on, config.max_target_length, config.num_kv_heads, config.head_dim),
      sharding=NamedSharding(mesh, nn.logical_to_mesh_axes(value_axis_names)),
      dtype=config.dtype,
  )

  decoder_segment_ids = create_random_global_array(
      rng,
      global_shape=(config.global_batch_size_to_train_on, config.max_target_length),
      sharding=NamedSharding(mesh, nn.logical_to_mesh_axes(decoder_segment_ids_axis_names)),
      dtype=jnp.int32,
  )
  while True:
    yield query, key, value, decoder_segment_ids


def test_tpu_flash_attention(attention_op, query, key, value, decoder_segment_ids):
  out = attention_op(query, key, value, decoder_segment_ids, common_types.MODEL_MODE_TRAIN)
  return out


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  pyconfig.initialize(argv)
  max_utils.print_system_information()
  config = pyconfig.config
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  train_iter = get_train_iter(config, mesh)
  start_step = 0
  prof = profiler.Profiler(config)
  first_profiling_step = start_step + config.skip_first_n_steps_for_profiler
  last_profiling_step = np.clip(first_profiling_step + config.profiler_steps - 1, first_profiling_step, config.steps - 1)
  per_device_tflops = calculate_attention_forward_tflops_per_device(config)

  last_step_completion = datetime.datetime.now()

  attention_op = AttentionOp(
      mesh=mesh,
      attention_kernel=config.attention,
      max_target_length=config.max_target_length,
      quant=quantizations.configure_quantization(config),
      num_query_heads=config.num_query_heads,
      num_kv_heads=config.num_kv_heads,
      dropout_rate=config.dropout_rate,
      dtype=config.dtype,
  )
  # empty variable states in pure flash attention
  vars = {}
  jitted = jax.jit(nn.apply(test_tpu_flash_attention, attention_op))
  for step in np.arange(start_step, config.steps):
    if step == first_profiling_step:
      prof.activate()
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        query, key, value, decoder_segment_ids = next(train_iter)
        out = jitted(vars, query, key, value, decoder_segment_ids)
    new_time = datetime.datetime.now()
    step_time_delta = new_time - last_step_completion
    max_logging.log(
      f"out: {jnp.sum(out)}; "
      f"step: {step}; "
      f"perf/step_time_seconds: {step_time_delta.total_seconds()}; "
      f"perf/per_device_tflops_per_sec: { per_device_tflops / step_time_delta.total_seconds()}"
    )
    last_step_completion = new_time
    if step == last_profiling_step:
      prof.deactivate()


if __name__ == "__main__":
  app.run(main)