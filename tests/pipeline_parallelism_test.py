# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for pipeline parallelism."""

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
from flax import nnx

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import MODEL_MODE_TRAIN, Config
from MaxText.globals import MAXTEXT_PKG_DIR, MAXTEXT_ASSETS_ROOT
from MaxText.layers import pipeline
from MaxText.layers import pipeline_nnx
from MaxText.layers import simple_layer
from MaxText.layers.simple_layer import SimpleDecoderLayer, SimpleDecoderLayerNnx
from MaxText.train import main as train_main
from MaxText.layers import deepseek


def assert_same_output_and_grad(f1, f1_inputs, f2, f2_inputs):
  """check that the output and gradient are the same for two functions with potentially different inputs"""
  f1_value, f1_grad = jax.value_and_grad(f1, argnums=0, allow_int=True)(*f1_inputs)
  f2_value, f2_grad = jax.value_and_grad(f2, argnums=0, allow_int=True)(*f2_inputs)

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
    model_mode = MODEL_MODE_TRAIN
    if single_pipeline_stage_class is None:
      single_pipeline_stage = simple_layer.SimpleDecoderLayer(config=config, mesh=mesh, model_mode=model_mode)
    else:
      single_pipeline_stage = single_pipeline_stage_class(config=config, mesh=mesh, model_mode=model_mode)

    def get_inputs(batch_size, sequence, features):
      """Get random inputs, and random dummy targets"""
      input_shape = [batch_size, sequence, features]
      inputs = jax.random.normal(jax.random.PRNGKey(2), input_shape, dtype=jnp.float32)
      dummy_targets = jax.random.normal(jax.random.PRNGKey(3), input_shape, dtype=jnp.float32)
      inputs_position = jnp.array([jnp.arange(sequence, dtype=jnp.int32) for _ in range(batch_size)], dtype=jnp.int32)
      inputs_segmentation = jnp.ones((batch_size, sequence), dtype=jnp.int32)
      return inputs, dummy_targets, inputs_position, inputs_segmentation

    inputs, dummy_targets, inputs_position, inputs_segmentation = get_inputs(
        config.global_batch_size_to_train_on, config.max_target_length, config.emb_dim
    )
    deterministic = True
    layer_class_for_pipeline = simple_layer.SimpleDecoderLayerNnx
    if single_pipeline_stage_class is not None:
      layer_class_for_pipeline = single_pipeline_stage_class

    # Create two identical Rngs objects to ensure both models initialize the same way.
    pipeline_rngs = nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1))
    sequential_rngs = nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1))

    # Define the sequential model with unique layers
    class SequentialModel(nnx.Module):

      def __init__(self, config: Config, *, rngs: nnx.Rngs):
        self.layers = nnx.List(
            [
                simple_layer.SimpleDecoderLayerNnx(config=config, rngs=rngs.fork())
                for _ in range(config.num_decoder_layers)
            ]
        )

      def __call__(self, inputs, inputs_position, inputs_segmentation, deterministic, model_mode):
        for layer in self.layers:
          inputs, _ = layer(inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
        return inputs

    sequential_model = SequentialModel(config=config, rngs=sequential_rngs)
    my_pipeline = pipeline_nnx.Pipeline(
        config=config, layer_module=layer_class_for_pipeline, mesh=mesh, rngs=pipeline_rngs
    )

    def pipeline_parallelism_dummy_loss(
        pipeline_module, inputs, inputs_position, inputs_segmentation, deterministic, model_mode, dummy_targets
    ):
      outputs = pipeline_module(inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
      return jnp.linalg.norm(outputs - dummy_targets)

    def regular_sequential_layers_dummy_loss(
        seq_model, inputs, inputs_position, inputs_segmentation, deterministic, model_mode, dummy_targets
    ):
      reg_layer_activations = seq_model(inputs, inputs_position, inputs_segmentation, deterministic, model_mode)
      return jnp.linalg.norm(reg_layer_activations - dummy_targets)

    f1_inputs = (sequential_model, inputs, inputs_position, inputs_segmentation, deterministic, model_mode, dummy_targets)
    f2_inputs = (my_pipeline, inputs, inputs_position, inputs_segmentation, deterministic, model_mode, dummy_targets)

    assert_same_output_and_grad(
        regular_sequential_layers_dummy_loss,
        f1_inputs,
        pipeline_parallelism_dummy_loss,
        f2_inputs,
    )

  @pytest.mark.tpu_only
  def test_circular_minimum_microbatches_same_output_and_grad(self):
    # 4 stages, 8 layers (2 repeats, 1 layer per stage), 4 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="circular_minimum_microbatches",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4,
        base_num_decoder_layers=8,
        num_pipeline_microbatches=4,
        per_device_batch_size=4,
    )
    self.assert_pipeline_same_output_and_grad(config)

  @pytest.mark.tpu_only
  def test_circular_extra_microbatches_same_output_and_grad(self):
    # 4 stages, 8 layers (2 repeats, 1 layer per stage), 8 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="circular_extra_microbatches",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4,
        base_num_decoder_layers=8,
        num_pipeline_microbatches=8,
        per_device_batch_size=4,
    )
    self.assert_pipeline_same_output_and_grad(config)

  @pytest.mark.tpu_only
  def test_circular_deepseek_megablox_same_output_and_grad(self):
    # 4 stages, 8 layers (2 repeats, 1 layer per stage), 8 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="circular_moe",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4,
        base_num_decoder_layers=8,
        num_pipeline_microbatches=8,
        per_device_batch_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        megablox=False,
        sparse_matmul=False,
        capacity_factor=1,
        decoder_block="deepseek",
    )
    self.assert_pipeline_same_output_and_grad(config, single_pipeline_stage_class=deepseek.DeepSeekMoELayer)

  @pytest.mark.tpu_only
  def test_circular_ag_once(self):
    # 2 stages, 8 microbatches, all gather once
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="circular_ag_once",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=2,
        base_num_decoder_layers=8,
        num_pipeline_microbatches=8,
        per_device_batch_size=4,
        pipeline_fsdp_ag_once=True,
        log_config=False,
    )
    self.assert_pipeline_same_output_and_grad(config)

  @pytest.mark.tpu_only
  def test_non_circular_same_output_and_grad(self):
    # 4 stages, 4 layers (no circular repeats, 1 layer per stage), 4 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        run_name="non_circular",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4,
        base_num_decoder_layers=4,
        num_pipeline_microbatches=4,
        per_device_batch_size=4,
    )
    self.assert_pipeline_same_output_and_grad(config)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_full_train_circular(self):
    # Run a full train.py call with 4 stages, 32 layers (2 layers per stage, 4 circular repeats), 8 microbatches
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_pipeline_parallelism_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=32",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            "num_layers_per_pipeline_stage=2",
            "num_pipeline_microbatches=8",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "scan_layers_per_stage=False",  # We see better performance only scanning the pipeline iterations.
            "log_config=False",
        ]
    )

  @pytest.mark.tpu_only
  def test_delay_activation_forwarding_same_output_and_grad(self):
    # 4 stages, delayed activation forwarding, 8 layers (2 repeats, 1 layer per stage), 8 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="activation_forwarding",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=4,
        base_num_decoder_layers=8,
        num_pipeline_microbatches=8,
        per_device_batch_size=4,
        pipeline_delay_activation_forwarding=True,
    )
    self.assert_pipeline_same_output_and_grad(config)

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_full_train_non_circular(self):
    # Run a full train.py call with 4 stages, 32 layers (8 layers per stage), 8 microbatches
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_pipeline_parallelism_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=32",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            "num_layers_per_pipeline_stage=8",
            "num_pipeline_microbatches=8",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "scan_layers_per_stage=False",  # We see better performance only scanning the pipeline iterations.
        ]
    )

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_subset_layers(self):
    # Run a full train.py call with 4 stages, 16 layers - 8 in pipeline, 8 ran outside of pipeline
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_pipeline_parallelism_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=16",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            "num_layers_per_pipeline_stage=1",
            "num_pipeline_repeats=2",
            "pipeline_parallel_layers=8",
            "num_pipeline_microbatches=8",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "scan_layers_per_stage=False",  # We see better performance only scanning the pipeline iterations.
            "log_config=False",
        ]
    )

  @pytest.mark.integration_test
  def test_full_train_fp8(self):
    # Run a full train.py call with fp8 quantization, which adds extra
    # variable collections that need to be handled
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_pipeline_parallelism_fp8_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=4",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "quantization=fp8",
            "scan_layers_per_stage=False",
            "attention=dot_product",
            "log_config=False",
        ]
    )

  @pytest.mark.integration_test
  def test_full_train_nanoo_fp8(self):
    # Run a full train.py call with NANOO fp8 quantization, which adds extra
    # variable collections that need to be handled
    train_main(
        [
            None,
            os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
            "base_output_directory=gs://runner-maxtext-logs",
            "run_name=runner_pipeline_parallelism_nanoo_fp8_test",
            "dataset_path=gs://maxtext-dataset",
            "base_emb_dim=28",
            "base_num_query_heads=4",
            "base_num_kv_heads=4",
            "base_mlp_dim=32",
            "base_num_decoder_layers=4",
            "head_dim=128",
            "per_device_batch_size=2",
            "max_target_length=1024",
            "vocab_size=32",
            "dataset_type=synthetic",
            "steps=3",
            "enable_checkpointing=False",
            "enable_goodput_recording=False",
            "ici_pipeline_parallelism=4",
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
            "quantization=nanoo_fp8",
            "scan_layers_per_stage=False",
            "attention=dot_product",
        ]
    )


if __name__ == "__main__":
  unittest.main()
