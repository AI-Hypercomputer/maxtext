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

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.globals import PKG_DIR
from MaxText.layers import pipeline
from MaxText.layers import simple_layer
from MaxText.train import main as train_main
from MaxText.layers import deepseek


def assert_same_output_and_grad(f1, f2, *inputs):
  """check that the output and gradient are the same"""
  f1_value, f1_grad = jax.value_and_grad(f1)(*inputs)
  f2_value, f2_grad = jax.value_and_grad(f2)(*inputs)

  def pytree_ravel(pytree):
    ravelled_tree = jax.tree.map(jnp.ravel, pytree)
    ravelled_leaves, _ = jax.tree_util.tree_flatten(ravelled_tree)
    return jnp.concatenate(ravelled_leaves)

  f1_grad = pytree_ravel(f1_grad)
  f2_grad = pytree_ravel(f2_grad)
  assert jax.numpy.allclose(f1_value, f2_value, rtol=1e-2, equal_nan=False)
  assert jax.numpy.allclose(f1_grad, f2_grad, rtol=1e-1, equal_nan=False)


class PipelineParallelismTest(unittest.TestCase):

  def assert_pipeline_same_output_and_grad(self, config, single_pipeline_stage_class=None):
    """check that the output and gradient are the same"""
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    if single_pipeline_stage_class is None:
      single_pipeline_stage = simple_layer.SimpleDecoderLayer(config=config, mesh=mesh)
    else:
      single_pipeline_stage = single_pipeline_stage_class(config=config, mesh=mesh)

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

    inputs, dummy_targets, inputs_position, inputs_segmentation = get_inputs(
        config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim
    )
    deterministic = True
    model_mode = MODEL_MODE_TRAIN
    # We use a simpler single matmul decoder layer for fast compilation in these tests.
    #single_pipeline_stage = simple_layer.SimpleDecoderLayer(config=config, mesh=mesh)
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

    def regular_sequential_layers(params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode):
      def get_cur_layer_params(params, layer_idx):
        def get_cur_layer_params_arr(leaf):
          # Reshape layers into a linear list of layers, e.g. [repeat, stage] into [layers]
          if config.num_pipeline_repeats > 1 and config.num_layers_per_pipeline_stage == 1:
            new_shape = (leaf.shape[0] * leaf.shape[1],) + leaf.shape[2:]
            leaf = jnp.reshape(leaf, new_shape)  # [repeat, stage] -> [layers]
          elif config.num_pipeline_repeats > 1 and config.num_layers_per_pipeline_stage > 1:
            new_shape = (leaf.shape[0] * leaf.shape[1] * leaf.shape[2],) + leaf.shape[3:]
            leaf = jnp.reshape(leaf, new_shape)  # [repeat, stage, layers_per_stage] -> [layers]
          elif config.num_pipeline_repeats == 1 and config.num_layers_per_pipeline_stage > 1:
            new_shape = (leaf.shape[0] * leaf.shape[1],) + leaf.shape[2:]
            leaf = jnp.reshape(leaf, new_shape)  # [stage, layers_per_stage] -> [layers]
          return leaf[layer_idx]

        return jax.tree.map(get_cur_layer_params_arr, params)

      reg_layer_activations = inputs
      for layer in range(config.num_decoder_layers):
        cur_layer_params = get_cur_layer_params(params, layer)
        cur_layer_params["params"] = cur_layer_params["params"]["layers"]
        if config.num_pipeline_repeats > 1 and config.num_layers_per_pipeline_stage > 1:
          cur_layer_params["params"] = meta.remove_axis(
              cur_layer_params["params"], 0, {nn.PARTITION_NAME: "circular_repeats"}
          )
          cur_layer_params["params"] = meta.remove_axis(cur_layer_params["params"], 0, {nn.PARTITION_NAME: "layers"})
        reg_layer_activations, _ = single_pipeline_stage.apply(
            cur_layer_params, reg_layer_activations, inputs_position, inputs_segmentation, deterministic, model_mode
        )
      return reg_layer_activations

    def regular_sequential_layers_dummy_loss(
        params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode, dummy_targets
    ):
      outputs = regular_sequential_layers(params, inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
      loss = jnp.linalg.norm(outputs - dummy_targets)
      return loss

    assert_same_output_and_grad(
        regular_sequential_layers_dummy_loss,
        pipeline_parallelism_dummy_loss,
        init_pipeline_params,
        inputs,
        inputs_segmentation,
        inputs_position,
        deterministic,
        model_mode,
        dummy_targets,
    )


  @pytest.mark.tpu_only
  def test_ra2a(self):
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
        num_experts=4,
        num_experts_per_tok=2,
        megablox=False,
        sparse_matmul=False,
        capacity_factor=1.0,
        attention_type="mla",
        decoder_block="deepseek",
    )
    self.assert_pipeline_same_output_and_grad(config, single_pipeline_stage_class=deepseek.DeepSeekMoELayer)


  # @pytest.mark.tpu_only
  # def test_ra2a(self):
  #   # 4 stages, 8 layers (2 repeats, 1 layer per stage), 8 microbatches
  #   config = pyconfig.initialize(
  #       [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
  #       enable_checkpointing=False,
  #       enable_goodput_recording=False,
  #       run_name="circular_moe",
  #       max_target_length=128,
  #       base_emb_dim=28,
  #       ici_pipeline_parallelism=4,
  #       ici_expert_parallelism=2,
  #       base_num_decoder_layers=8,
  #       num_pipeline_microbatches=8,
  #       per_device_batch_size=4,
  #       num_experts=4,
  #       num_experts_per_tok=2,
  #       megablox=True,
  #       sparse_matmul=True,
  #       attention_type="mla",
  #       decoder_block="deepseek",
  #   )
  #   self.assert_pipeline_same_output_and_grad(config, single_pipeline_stage_class=deepseek.DeepSeekMoELayer)


if __name__ == "__main__":
  unittest.main()
