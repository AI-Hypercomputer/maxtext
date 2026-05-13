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

from flax import linen as nn
from flax import nnx
from flax.core import meta
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.layers import nnx_wrappers
from maxtext.layers import pipeline_0506 as pipeline
from maxtext.models import deepseek
from maxtext.models import simple_layer
from maxtext.utils import maxtext_utils
from maxtext.trainers.pre_train.train import main as train_main
from tests.utils.test_helpers import get_test_config_path, get_test_dataset_path, get_test_base_output_directory
import pytest


# Helper to fix pipeline parallelism in test_full_train_fp8 and test_full_train_nanoo_fp8
def _adapt_parallelism(args, pipeline_stages=4):
  dc = jax.device_count()
  # In decoupled mode with limited devices, adjust pipeline stages to device count
  if is_decoupled() and dc < pipeline_stages:
    pipeline_stages = dc
  args.append(f"ici_pipeline_parallelism={pipeline_stages}")
  if dc >= pipeline_stages:
    data_par = dc // pipeline_stages
    if data_par > 1:
      args.append(f"ici_data_parallelism={data_par}")


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

  g_diff = jnp.abs(f1_grad - f2_grad)
  v_diff = jnp.abs(f1_value - f2_value)

  value_close = bool(jax.numpy.allclose(f1_value, f2_value, rtol=1e-2, atol=1e-1, equal_nan=False))
  grad_close = bool(jax.numpy.allclose(f1_grad, f2_grad, rtol=1e-1, atol=1.0, equal_nan=False))
  assert value_close, f"value mismatch: f1={float(f1_value)} vs f2={float(f2_value)}, " f"abs_diff={float(v_diff)}"
  assert grad_close, (
      f"grad mismatch: abs_diff_max={float(g_diff.max())}, "
      f"abs_diff_mean={float(g_diff.mean())}, "
      f"f1_grad_norm={float(jnp.linalg.norm(f1_grad))}, "
      f"f2_grad_norm={float(jnp.linalg.norm(f2_grad))}, "
      f"grad_size={f1_grad.size}, "
      f"tolerance=rtol=1e-1+atol=1.0"
  )


class PipelineParallelismTest(unittest.TestCase):
  decoupled = is_decoupled()
  base_output_directory = get_test_base_output_directory()
  dataset_path = get_test_dataset_path()

  def assert_pipeline_same_output_and_grad(self, config, single_pipeline_stage_class=None):
    """check that the output and gradient are the same"""
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    model_mode = MODEL_MODE_TRAIN

    # `single_pipeline_stage_class` (when provided, e.g. `deepseek.DeepSeekMoELayerToLinen`
    # for the deepseek test) controls BOTH the pipeline and the per-layer reference path.
    rngs = nnx.Rngs(params=0)
    if single_pipeline_stage_class is None:
      single_pipeline_stage = simple_layer.SimpleDecoderLayerToLinen(
          config=config, mesh=mesh, model_mode=model_mode, rngs=rngs
      )
      raw_stage_class = simple_layer.SimpleDecoderLayer
    elif issubclass(single_pipeline_stage_class, nnx_wrappers.ToLinen):
      single_pipeline_stage = single_pipeline_stage_class(config=config, mesh=mesh, model_mode=model_mode, rngs=rngs)
      # `to_linen_class` stores the wrapped NNX class as the `module_class` class attribute
      # (see `nnx_wrappers.py:to_linen_class` -> `ToLinenPartial.module_class = base_nnx_class`).
      raw_stage_class = single_pipeline_stage_class.module_class
    else:
      single_pipeline_stage = single_pipeline_stage_class(config=config, mesh=mesh, model_mode=model_mode, rngs=rngs)
      raw_stage_class = single_pipeline_stage_class

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

    def stage_factory(stage_rngs):
      return raw_stage_class(config=config, mesh=mesh, model_mode=model_mode, rngs=stage_rngs)

    my_pipeline = pipeline.create_pipeline(config=config, layers=stage_factory, mesh=mesh)
    init_pipeline_params = my_pipeline.init(
        jax.random.PRNGKey(0), inputs, inputs_position, inputs_segmentation, deterministic, model_mode
    )
    # `get_weight_sharding` is a compact method on `PipelineLinen` (callable as
    # `my_pipeline.get_weight_sharding(...)` directly) but on the ToLinen-wrapped NNX
    # pipeline it must be invoked inside a bound module context. Use `bind` so the
    # same call shape works on both paths.
    logical_partition_spec = my_pipeline.bind(init_pipeline_params).get_weight_sharding(
        inputs, inputs_position, inputs_segmentation, deterministic, model_mode
    )

    # Create a dummy scalar loss function so we may take the gradient wrt weights
    def pipeline_parallelism_dummy_loss_extra(
        params,
        inputs,
        inputs_position,
        inputs_segmentation,
        deterministic,
        model_mode,
        dummy_targets,
        logical_partition_spec=None,
    ):
      outputs = my_pipeline.apply(
          params,
          inputs,
          inputs_position,
          inputs_segmentation,
          deterministic,
          model_mode,
          logical_partition_spec=logical_partition_spec,
      )
      loss = jnp.linalg.norm(outputs - dummy_targets)
      return loss

    pipeline_parallelism_dummy_loss = functools.partial(
        pipeline_parallelism_dummy_loss_extra, logical_partition_spec=logical_partition_spec
    )

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
  def test_circular_minimum_microbatches_same_output_and_grad(self):
    # 4 stages, 8 layers (2 repeats, 1 layer per stage), 4 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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
        [sys.argv[0], get_test_config_path()],
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
    # 4 stages, 8 layers (2 repeats, 1 layer per stage), 8 microbatches.
    # DeepSeek's MoE block (`moe.RoutedAndSharedMoE`) constructs a `shared_experts`
    # `MlpBlock` with `intermediate_dim = config.shared_experts * moe_mlp_dim`, so
    # `shared_experts >= 1` is required to avoid a zero-dim DenseGeneral kernel
    # (`ZeroDivisionError` in `_compute_fans`). DeepSeek self-attention is
    # `attention_mla.MLA`, which asserts `config.attention_type == "mla"` in
    # `_init_projections` (`src/maxtext/layers/attention_mla.py:718-721`).
    # MLA layer supplies sane defaults for the head-dim / lora-rank fields.
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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
        base_moe_mlp_dim=1024,
        base_mlp_dim=1024,
        attention_type="mla",
        shared_experts=1,
    )
    self.assert_pipeline_same_output_and_grad(config, single_pipeline_stage_class=deepseek.DeepSeekMoELayerToLinen)

  @pytest.mark.tpu_only
  def test_circular_ag_once(self):
    # 2 stages, 8 microbatches, all gather once
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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
    )
    self.assert_pipeline_same_output_and_grad(config)

  @pytest.mark.tpu_only
  def test_circular_pipeline_ag_per_repeat(self):
    # 2 stages, 8 microbatches, enable pipeline ag per repeat
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="circular_ag_per_repeat",
        max_target_length=128,
        base_emb_dim=28,
        ici_pipeline_parallelism=2,
        base_num_decoder_layers=8,
        num_pipeline_microbatches=8,
        per_device_batch_size=4,
        pipeline_fsdp_ag_per_repeat=True,
    )
    self.assert_pipeline_same_output_and_grad(config)

  @pytest.mark.tpu_only
  def test_non_circular_same_output_and_grad(self):
    # 4 stages, 4 layers (no circular repeats, 1 layer per stage), 4 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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
            get_test_config_path(),
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_pipeline_parallelism_test",
            f"dataset_path={self.dataset_path}",
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
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "scan_layers_per_stage=False",  # We see better performance only scanning the pipeline iterations.
        ]
    )

  @pytest.mark.integration_test
  @pytest.mark.tpu_only
  def test_full_train_circular_pipeline_ag_per_repeat(self):
    # Run a full train.py call with 4 stages, 32 layers (2 layers per stage, 4 circular repeats),
    # 8 microbatches and using pipeline ag per repeat
    train_main(
        [
            None,
            get_test_config_path(),
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_pipeline_parallelism_test",
            f"dataset_path={self.dataset_path}",
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
            "ici_pipeline_parallelism=2",
            "num_layers_per_pipeline_stage=1",
            "num_pipeline_microbatches=4",
            "pipeline_fsdp_ag_per_repeat=True",
            (rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}"),
        ]
    )

  @pytest.mark.tpu_only
  def test_delay_activation_forwarding_same_output_and_grad(self):
    # 4 stages, delayed activation forwarding, 8 layers (2 repeats, 1 layer per stage), 8 microbatches
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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
            get_test_config_path(),
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_pipeline_parallelism_test",
            f"dataset_path={self.dataset_path}",
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
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
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
            get_test_config_path(),
            f"base_output_directory={self.base_output_directory}",
            "run_name=runner_pipeline_parallelism_test",
            f"dataset_path={self.dataset_path}",
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
            rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
            "scan_layers_per_stage=False",  # We see better performance only scanning the pipeline iterations.
        ]
    )

  @pytest.mark.skipif(is_decoupled(), reason="Pipeline parallelism not supported in decoupled mode")
  @pytest.mark.integration_test
  def test_full_train_fp8(self):
    # Run a full train.py call with fp8 quantization, which adds extra
    # variable collections that need to be handled
    args = [
        None,
        get_test_config_path(),
        f"base_output_directory={self.base_output_directory}",
        "run_name=runner_pipeline_parallelism_fp8_test",
        f"dataset_path={self.dataset_path}",
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
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
        "quantization=fp8",
        "scan_layers_per_stage=False",
        "attention=dot_product",
    ]
    _adapt_parallelism(args, pipeline_stages=4)
    train_main(args)

  @pytest.mark.skipif(is_decoupled(), reason="Pipeline parallelism not supported in decoupled mode")
  @pytest.mark.integration_test
  def test_full_train_nanoo_fp8(self):
    # Run a full train.py call with NANOO fp8 quantization, which adds extra
    # variable collections that need to be handled
    args = [
        None,
        get_test_config_path(),
        f"base_output_directory={self.base_output_directory}",
        "run_name=runner_pipeline_parallelism_nanoo_fp8_test",
        f"dataset_path={self.dataset_path}",
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
        rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
        "quantization=nanoo_fp8",
        "scan_layers_per_stage=False",
        "attention=dot_product",
    ]
    _adapt_parallelism(args, pipeline_stages=4)
    train_main(args)


if __name__ == "__main__":
  unittest.main()
