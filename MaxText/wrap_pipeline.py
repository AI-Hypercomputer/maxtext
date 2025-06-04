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

import functools
import os.path
import sys
import unittest

import pytest

import jax
from jax.sharding import Mesh
import jax.numpy as jnp

from flax.core import meta
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.globals import PKG_DIR
from MaxText.layers import pipeline
from MaxText.layers import simple_layer
from MaxText.train import main as train_main
from MaxText.layers import deepseek


def test_simple():
    # 4 stages, 8 layers (2 repeats, 1 layer per stage), 8 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="circular_moe",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4,
        ici_expert_parallelism=2,
        base_num_decoder_layers=8,
        num_pipeline_microbatches=8,
        per_device_batch_size=4,
        decoder_block="simple",
    )
    
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

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

    data_logical = (("activation_batch", "activation_length", "activation_embed"))
    data_physical = nn.logical_to_mesh_axes(data_logical, rules=config.logical_axis_rules)
    data_shardings = jax.sharding.NamedSharding(mesh, data_physical)

    token_logical = (("activation_batch", "activation_length"))
    token_physical = nn.logical_to_mesh_axes(token_logical, rules=config.logical_axis_rules)
    token_shardings = jax.sharding.NamedSharding(mesh, token_physical)
    jit_inputs = jax.jit(get_inputs, in_shardings=(), out_shardings=(data_shardings, data_shardings, token_shardings, token_shardings), static_argnums=(0,1,2))

    inputs, dummy_targets, inputs_position, inputs_segmentation = jit_inputs(config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim)
    breakpoint()
    deterministic = True
    model_mode = MODEL_MODE_TRAIN




    def get_params():
        my_pipeline = pipeline.Pipeline(config=config, layers=single_pipeline_stage, mesh=mesh)
        init_pipeline_params = my_pipeline.init(
            jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode
        )
        partition_spec = my_pipeline.get_weight_sharding(inputs, inputs_position, inputs_segmentation, deterministic, model_mode)

    # Create a dummy scalar loss function so we may take the gradient wrt weights
    def pipeline_parallelism_dummy_loss_extra(
        params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode, dummy_targets, partition_spec=None
    ):
        outputs = my_pipeline.apply(
            params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode, partition_spec=partition_spec
        )
        loss = jnp.linalg.norm(outputs - dummy_targets)
        return loss

    pipeline_parallelism_dummy_loss = functools.partial(pipeline_parallelism_dummy_loss_extra, partition_spec=partition_spec)

    jit_pp = jax.jit(pipeline_parallelism_dummy_loss)
    f2_value, f2_grad = jax.value_and_grad(f2)(*inputs)

if __name__ == "__main__":
  test_simple()
