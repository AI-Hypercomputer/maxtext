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
import json
import os
import os.path
import sys
import tempfile
import unittest

from flax import linen as nn
from flax import nnx
from flax.core import meta
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, PartitionSpec as P, reshard
import numpy as np
from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from maxtext.common.common_types import MODEL_MODE_TRAIN, ShardMode
from maxtext.common.gcloud_stub import is_decoupled
from maxtext.layers import nnx_wrappers
from maxtext.models import simple_layer
from maxtext.utils import maxtext_utils
from maxtext.utils import pipeline_utils
from maxtext.trainers.pre_train.train import main as train_main
from tests.utils.test_helpers import get_test_config_path, get_test_dataset_path, get_test_base_output_directory
from tests.utils import linen_wrappers
import pytest

pytestmark = [pytest.mark.integration_test, pytest.mark.tpu_only]


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

  value_close = bool(jax.numpy.allclose(f1_value, f2_value, rtol=1e-2, atol=1e-2, equal_nan=False))
  grad_close = bool(jax.numpy.allclose(f1_grad, f2_grad, rtol=1e-1, atol=2e-1, equal_nan=False))
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

  def assert_pipeline_matches_sequential_output_and_grad(self, config, single_pipeline_stage_class=None):
    """check that the output and gradient are the same"""
    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)
    model_mode = MODEL_MODE_TRAIN

    # `single_pipeline_stage_class` (when provided, e.g. `linen_wrappers.DeepSeekMoELayerToLinen`
    # for the deepseek test) controls BOTH the pipeline and the per-layer reference path.
    rngs = nnx.Rngs(params=0)
    if single_pipeline_stage_class is None:
      single_pipeline_stage = linen_wrappers.SimpleDecoderLayerToLinen(
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

    my_pipeline = linen_wrappers.create_pipeline(config=config, layers=stage_factory, mesh=mesh)
    with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
      init_pipeline_params = my_pipeline.init(
          jax.random.PRNGKey(0), inputs, inputs_segmentation, inputs_position, deterministic, model_mode
      )
      # `get_weight_sharding` is a compact method on `PipelineLinen` (callable as
      # `my_pipeline.get_weight_sharding(...)` directly) but on the ToLinen-wrapped NNX
      # pipeline it must be invoked inside a bound module context. Use `bind` so the
      # same call shape works on both paths.
      logical_partition_spec = my_pipeline.bind(init_pipeline_params).get_weight_sharding(
          inputs, inputs_segmentation, inputs_position, deterministic, model_mode
      )

    # Create a dummy scalar loss function so we may take the gradient wrt weights
    def pipeline_parallelism_dummy_loss_extra(
        params,
        inputs,
        inputs_segmentation,
        inputs_position,
        deterministic,
        model_mode,
        dummy_targets,
        logical_partition_spec=None,
    ):
      outputs = my_pipeline.apply(
          params,
          inputs,
          inputs_segmentation,
          inputs_position,
          deterministic,
          model_mode,
          logical_partition_spec=logical_partition_spec,
      )
      loss = jnp.linalg.norm(outputs - dummy_targets)
      return loss

    pipeline_parallelism_dummy_loss = functools.partial(
        pipeline_parallelism_dummy_loss_extra, logical_partition_spec=logical_partition_spec
    )

    def regular_sequential_layers(params, inputs, inputs_segmentation, inputs_position, deterministic, model_mode):
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
        if config.num_pipeline_repeats > 1:
          cur_layer_params["params"] = meta.remove_axis(
              cur_layer_params["params"], 0, {nn.PARTITION_NAME: "circular_repeats"}
          )
        cur_layer_params["params"] = meta.remove_axis(cur_layer_params["params"], 0, {nn.PARTITION_NAME: "layers"})
        reg_layer_activations, _ = single_pipeline_stage.apply(
            cur_layer_params, reg_layer_activations, inputs_segmentation, inputs_position, deterministic, model_mode
        )
      return reg_layer_activations

    def regular_sequential_layers_dummy_loss(
        params, inputs, inputs_segmentation, inputs_position, deterministic, model_mode, dummy_targets
    ):
      outputs = regular_sequential_layers(params, inputs, inputs_segmentation, inputs_position, deterministic, model_mode)
      loss = jnp.linalg.norm(outputs - dummy_targets)
      return loss

    with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
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
    self.assert_pipeline_matches_sequential_output_and_grad(config)

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
    self.assert_pipeline_matches_sequential_output_and_grad(config)

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
    self.assert_pipeline_matches_sequential_output_and_grad(
        config, single_pipeline_stage_class=linen_wrappers.DeepSeekMoELayerToLinen
    )

  @pytest.mark.scheduled_only
  def test_deepseek_ragged_a2a_ep_same_output_and_grad(self):
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        enable_checkpointing=False,
        enable_goodput_recording=False,
        run_name="deepseek_ragged_a2a_ep",
        dtype="bfloat16",
        weight_dtype="bfloat16",
        matmul_precision="high",
        max_target_length=128,
        base_emb_dim=256,
        ici_pipeline_parallelism=2,
        ici_expert_parallelism=2,
        allow_split_physical_axes=True,
        base_num_decoder_layers=2,
        num_pipeline_microbatches=4,
        per_device_batch_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_experts=1,
        megablox=True,
        sparse_matmul=True,
        capacity_factor=-1,
        decoder_block="deepseek",
        attention_type="mla",
        base_moe_mlp_dim=256,
        base_mlp_dim=256,
    )
    self.assert_pipeline_matches_sequential_output_and_grad(
        config,
        single_pipeline_stage_class=linen_wrappers.DeepSeekMoELayerToLinen,
    )

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
    self.assert_pipeline_matches_sequential_output_and_grad(config)

  def test_circular_pipeline_ag_per_repeat(self):
    # 2 stages, 8 microbatches, weight prefetching (BSW). Covers the re-gather of the previous repeat's weights (the
    # default) and the gathered slot carried across the repeat scan, each on the plain schedule and on the shifted
    # phase-split schedule (prologue before the repeat scan, single-slot phase first, carried w_curr bound to the
    # params by custom_vjp).
    for phase_split in (False, True):
      for bsw_no_carry in (True, False):
        with self.subTest(pipeline_bsw_phase_split=phase_split, pipeline_bsw_no_carry=bsw_no_carry):
          config = pyconfig.initialize(
              [sys.argv[0], get_test_config_path()],
              enable_checkpointing=False,
              enable_goodput_recording=False,
              run_name=f"circular_ag_per_repeat_phase_split_{phase_split}_no_carry_{bsw_no_carry}",
              max_target_length=128,
              base_emb_dim=28,
              ici_pipeline_parallelism=2,
              base_num_decoder_layers=8,
              num_pipeline_microbatches=8,
              per_device_batch_size=4,
              pipeline_fsdp_ag_per_repeat=True,
              pipeline_bsw_phase_split=phase_split,
              pipeline_bsw_no_carry=bsw_no_carry,
          )
          self.assert_pipeline_matches_sequential_output_and_grad(config)

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
    self.assert_pipeline_matches_sequential_output_and_grad(config)

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
    self.assert_pipeline_matches_sequential_output_and_grad(config)

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

  def test_explicit_stage_vmap_runs_shard_map_gather(self):
    """A shard_map in the stages gathering from an unbatched table at token-dependent indices, as the MoE does.

    Under explicit sharding, vmapped over an Explicit 'stage' axis, JAX replicates the stage dim into the shard_map
    and cannot batch the gather; call_with_stage_axis_auto runs it per stage as under shard_mode=auto.
    """
    if jax.device_count() < 4:
      self.skipTest("needs 4 devices")
    # No size-1 axes: train_main may have set jax_remove_size_one_mesh_axis_from_type, which drops them from types.
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("stage", "fsdp"), axis_types=(AxisType.Explicit,) * 2)
    stage_mesh = pipeline_utils.pipeline_stage_mesh(mesh, ShardMode.EXPLICIT)
    table = jnp.arange(4, dtype=jnp.float32)

    def stage_fn(x, w):
      y = jnp.einsum("bd,df->bf", x, w, out_sharding=P("fsdp", None))

      @jax.shard_map(mesh=stage_mesh, in_specs=P("fsdp", None), out_specs=P("fsdp", None))
      def route(y_local):
        return y_local * table[(y_local > 0).astype(jnp.int32) + (y_local > 1).astype(jnp.int32)]

      return route(y)

    def run(x, w):
      return pipeline_utils.call_with_stage_axis_auto(jax.vmap(stage_fn, spmd_axis_name="stage"), stage_mesh, x, w)

    x_host = np.random.default_rng(0).normal(size=(2, 8, 4)).astype(np.float32)
    w_host = np.random.default_rng(1).normal(size=(2, 4, 16)).astype(np.float32)

    def reference(x, w):
      y = jnp.einsum("sbd,sdf->sbf", x, w, precision=jax.lax.Precision.HIGHEST)
      return y * table[(y > 0).astype(jnp.int32) + (y > 1).astype(jnp.int32)]

    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
      x = reshard(jnp.asarray(x_host), P("stage", "fsdp", None))
      w = reshard(jnp.asarray(w_host), P("stage", "fsdp", None))
      out = jax.jit(run)(x, w)
      grad = jax.jit(jax.grad(lambda x, w: (run(x, w) ** 2).sum(), argnums=1))(x, w)
    self.assertEqual(out.sharding.spec, P("stage", "fsdp", None))
    self.assertEqual(grad.sharding.spec, P("stage", "fsdp", None))
    np.testing.assert_allclose(np.asarray(out), reference(x_host, w_host), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.asarray(grad),
        jax.grad(lambda w: (reference(x_host, w) ** 2).sum())(w_host),
        rtol=1e-4,
        atol=1e-4,
    )

  def _train_losses(self, run_name, extra_args):
    """Trains a tiny model for three steps and returns its per-step losses."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      metrics_file = os.path.join(tmp_dir, "metrics.txt")
      train_main(
          [
              None,
              get_test_config_path(),
              f"base_output_directory={self.base_output_directory}",
              f"run_name={run_name}",
              f"dataset_path={self.dataset_path}",
              f"metrics_file={metrics_file}",
              "dataset_type=synthetic",
              "steps=3",
              "max_target_length=128",
              "enable_checkpointing=False",
              "enable_goodput_recording=False",
              rf"tokenizer_path={os.path.join(MAXTEXT_ASSETS_ROOT, 'tokenizers', 'tokenizer.llama2')}",
          ]
          + list(extra_args)
      )
      with open(metrics_file, "rt", encoding="utf8") as f:
        return [json.loads(line)["learning/loss"] for line in f if line.strip()]

  def _explicit_and_auto_losses(self, name, args):
    auto_losses = self._train_losses(f"pp_{name}_auto", args + ["shard_mode=auto"])
    explicit_losses = self._train_losses(f"pp_{name}_explicit", args + ["shard_mode=explicit"])
    print(f"[{name}] auto losses: {auto_losses}", flush=True)
    print(f"[{name}] explicit losses: {explicit_losses}", flush=True)
    self.assertTrue(auto_losses, "auto run produced no metrics")
    return explicit_losses, auto_losses

  def test_explicit_sharding_matches_auto(self):
    """Explicit sharding only changes how the pipeline expresses its layouts, so the losses must not move.

    Covers NNXPipeline circular (the FSDP-sharded weights of each repeat gathered inside the stages) and non-circular
    with the FSDP all-gather hoisted out of the loop, and NNXCircularPipeline (weight prefetching) with the gathered
    slot carried across repeats and on the shifted schedule with the default re-gather. Every case shards the batch over
    FSDP besides the stages.
    """
    tiny_model = [
        "base_emb_dim=28",
        "base_num_query_heads=4",
        "base_num_kv_heads=4",
        "base_mlp_dim=32",
        "head_dim=128",
        "vocab_size=32",
        "ici_pipeline_parallelism=2",
    ]
    cases = {
        "circular": ["base_num_decoder_layers=8", "per_device_batch_size=2", "num_pipeline_microbatches=4"],
        "non_circular_ag_once": [
            "base_num_decoder_layers=4",
            "num_layers_per_pipeline_stage=2",
            "per_device_batch_size=2",
            "num_pipeline_microbatches=4",
            "pipeline_fsdp_ag_once=True",
        ],
        "bsw_carry": [
            "base_num_decoder_layers=8",
            "per_device_batch_size=2",
            "num_pipeline_microbatches=4",
            "pipeline_fsdp_ag_per_repeat=True",
            "pipeline_bsw_no_carry=False",
        ],
        "bsw_phase_split": [
            "base_num_decoder_layers=8",
            "per_device_batch_size=4",
            "num_pipeline_microbatches=8",
            "pipeline_fsdp_ag_per_repeat=True",
            "pipeline_bsw_phase_split=True",
        ],
    }
    for name, case_args in cases.items():
      with self.subTest(case=name):
        explicit_losses, auto_losses = self._explicit_and_auto_losses(name, tiny_model + case_args)
        np.testing.assert_allclose(explicit_losses, auto_losses, rtol=1e-5, atol=0.0)

  def test_deepseek_explicit_sharding_matches_auto(self):
    """DeepSeek under explicit sharding with pipeline and expert parallelism.

    The sparse-matmul MoE runs a shard_map inside the stage vmap; the dense layer before the pipeline and the MTP
    block run under the PP-as-DP rules; the routing biases are updated from the per-layer sown counts. The first loss
    is pure forward pass and matches to rounding. After an update the two modes' reassociated gradient reductions can
    move near-tied tokens to other experts, so the later losses drift by ~1e-4 and the bound is loose.
    """
    args = [
        "model_name=deepseek3-tiny",
        "override_model_config=True",
        "base_num_decoder_layers=5",
        "first_num_dense_layers=1",
        "vocab_size=256",
        "num_experts=4",
        "num_experts_per_tok=2",
        "mtp_num_layers=1",
        "mtp_loss_scaling_factor=0.1",
        "per_device_batch_size=2",
        "ici_pipeline_parallelism=2",
        "ici_expert_parallelism=2",
        "ici_fsdp_parallelism=1",
        "num_pipeline_microbatches=4",
    ]
    explicit_losses, auto_losses = self._explicit_and_auto_losses("deepseek", args)
    np.testing.assert_allclose(explicit_losses[0], auto_losses[0], rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(explicit_losses, auto_losses, rtol=1e-3, atol=0.0)

  @pytest.mark.skipif(is_decoupled(), reason="Pipeline parallelism not supported in decoupled mode")
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
        "max_target_length=128",
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
        "max_target_length=128",
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
