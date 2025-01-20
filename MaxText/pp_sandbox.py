"""
Copyright 2024 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from absl import app
import jax
from jax.sharding import Mesh
from typing import Sequence
import random

import pyconfig


from layers import pipeline, simple_layer
import jax
from jax import numpy as jnp
from jax.sharding import Mesh

import common_types
import max_utils

import jax.numpy as jnp
import tensorflow as tf
import string
import datetime


def get_inputs(batch_size, sequence, features):
    """Get random inputs, and random dummy targets
    Returns
        inputs: [batch_size, sequence, features]
        targets: [batch_size, sequence, features]
        positions: [batch_size, sequence]
        segmentations: [batch_size, segmentation]
    """
    input_shape = [batch_size, sequence, features]
    inputs = jax.random.normal(jax.random.PRNGKey(2), input_shape, dtype=jnp.float32)

    # dummy targets same shape as inputs to use for a dummy loss function to check gradient correctness
    dummy_targets = jax.random.normal(jax.random.PRNGKey(3), input_shape, dtype=jnp.float32)

    inputs_position = jnp.array([jnp.arange(sequence, dtype=jnp.int32) for _ in range(batch_size)], dtype=jnp.int32)
    inputs_segmentation = jnp.ones((batch_size, sequence), dtype=jnp.int32)
    return inputs, dummy_targets, inputs_position, inputs_segmentation

# Create a dummy scalar loss function so we may take the gradient wrt weights
def pipeline_parallelism_dummy_loss(
    params, pipeline, inputs, inputs_position, inputs_segmentation, deterministic, model_mode, dummy_targets
):
    outputs = pipeline.apply(params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
    loss = jnp.linalg.norm(outputs - dummy_targets)
    return loss

def simple_timeit(f, *args, tries=10, task=None):
  """Simple utility to time a function for multiple runs"""
  assert task is not None

  trace_name = f"t_{task}_" + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
  trace_dir = f"gs://mattdavidow-br/{trace_name}"

  outcomes_ms = []
  jax.block_until_ready(f(*args))  # warm it up!
  jax.profiler.start_trace(trace_dir)

  for _ in range(tries):
    s = datetime.datetime.now()
    jax.block_until_ready(f(*args))
    e = datetime.datetime.now()
    outcomes_ms.append(1000 * (e - s).total_seconds())
  jax.profiler.stop_trace()

  average_time_ms = sum(outcomes_ms) / len(outcomes_ms)
  print(f"{task}: average time milliseconds: {average_time_ms:.2f}, trace {trace_dir}")
  return average_time_ms

def main(argv: Sequence[str]):
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    tf.config.set_visible_devices([], "GPU")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
        os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
    pyconfig.initialize(argv)
    config = pyconfig.config

    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    inputs, dummy_targets, inputs_position, inputs_segmentation = get_inputs(
    config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim
    )
    deterministic = True
    model_mode = common_types.MODEL_MODE_TRAIN
    # We use a simpler single matmul decoder layer for fast compilation in these tests.
    #single_pipeline_stage = simple_layer.SimpleDecoderLayer(config=config, mesh=mesh)
    single_pipeline_stage = simple_layer.SimpleMlpDecoderLayer(config=config, mesh=mesh)
    my_pipeline = pipeline.Pipeline(config=config, layers=single_pipeline_stage, mesh=mesh)
    init_pipeline_params = my_pipeline.init(
    jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode
    )

    # This only computes the loss, e.g. only the forward pass. To get the backward pass we should call jax.value_and_grad
    jit_dummy_loss_fn=jax.jit(pipeline_parallelism_dummy_loss, static_argnums=(1,5,6))
    simple_timeit(jit_dummy_loss_fn, init_pipeline_params, my_pipeline, inputs, inputs_position, inputs_segmentation, deterministic, model_mode, dummy_targets, task='simple_pp')
    

if __name__ == "__main__":
  app.run(main)