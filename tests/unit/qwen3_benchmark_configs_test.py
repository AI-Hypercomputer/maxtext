# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Qwen3-235B v7x benchmark configurations and XLA flags."""

import argparse

from absl.testing import absltest
from absl.testing import parameterized
from benchmarks import benchmark_runner
from benchmarks import maxtext_v7x_model_configs
from benchmarks import xla_flags_library
from maxtext.common.common_types import CustomRule, ShardMode
from maxtext.configs import pyconfig
from maxtext.configs.types import RematLocation
from tests.utils.test_helpers import get_test_config_path

_MODELS = maxtext_v7x_model_configs.v7x_model_dict
_RECIPE_BASELINE = "qwen3_235b_a22b_v7x_bf16_baseline"
_RECIPE_PREFETCH = "qwen3_235b_a22b_v7x_bf16_fsdp_prefetch"
_RECIPE_ROE = "qwen3_235b_a22b_v7x_bf16_roe"
_EXPECTED_RECIPES = (_RECIPE_BASELINE, _RECIPE_PREFETCH, _RECIPE_ROE)
_SC_AG_FLAG = "--xla_tpu_enable_sparse_core_collective_offload_all_gather=true"
_ASYNC_RAGGED_A2A_FLAG = "--xla_tpu_enable_async_ragged_all_to_all=true"


class Qwen3BenchmarkConfigsTest(parameterized.TestCase):
  """Validates Qwen3-235B v7x benchmark recipes and paired negative configs."""

  def setUp(self) -> None:
    super().setUp()
    self._argv = [None, get_test_config_path("base.yml")]

  def test_qwen3_recipes_registered_and_wired_in_runner(self) -> None:
    """Verifies all three Qwen3-235B recipes are in v7x_model_dict and runner."""
    all_model_dicts = (
        benchmark_runner.trillium_model_dict,
        benchmark_runner.v5p_model_dict,
        benchmark_runner.v5e_model_dict,
        benchmark_runner.v7x_model_dict,
        benchmark_runner.c4_pretrain_model_dict,
    )
    all_keys = [name for model_dict in all_model_dicts for name in model_dict]
    self.assertEqual(len(all_keys), len(set(all_keys)))
    ctk_parser = argparse.ArgumentParser()
    benchmark_runner.add_ctk_runner_arguments(ctk_parser)
    on_device_parser = argparse.ArgumentParser()
    benchmark_runner.add_on_device_runner_arguments(on_device_parser)
    for recipe_name in _EXPECTED_RECIPES:
      self.assertIn(recipe_name, _MODELS)
      model = _MODELS[recipe_name]
      self.assertEqual(model.model_type, "qwen3-235b-a22b")
      self.assertNotEmpty(model.xla_flags)
      ctk_args = [
          "--project=test-project",
          "--cluster_name=test-cluster",
          "--base_output_directory=gs://test",
          "--device_type=v7x-128",
          f"--model_name={recipe_name}",
      ]
      ctk_parsed = ctk_parser.parse_args(ctk_args)
      self.assertEqual(ctk_parsed.model_name, recipe_name)
      dev_args = [
          "--base_output_directory=gs://test",
          f"--model_name={recipe_name}",
      ]
      parsed = on_device_parser.parse_args(dev_args)
      self.assertEqual(parsed.model_name, recipe_name)

  def test_baseline_recipe_initialization(self) -> None:
    """Verifies qwen3_235b_a22b_v7x_bf16_baseline passes pyconfig.initialize."""
    model = _MODELS[_RECIPE_BASELINE]
    params = model.tuning_params
    cfg = pyconfig.initialize(self._argv, model_name=model.model_type, **params)
    self.assertEqual(cfg.model_name, "qwen3-235b-a22b")
    self.assertEqual(cfg.per_device_batch_size, 16.0)
    self.assertEqual(cfg.max_target_length, 4096)
    self.assertEqual(cfg.dataset_type, "synthetic")
    self.assertFalse(cfg.use_random_routing)
    self.assertEqual(cfg.ici_fsdp_parallelism, -1)
    self.assertEqual(cfg.ici_expert_parallelism, 1)
    self.assertEqual(cfg.decoder_block.value, "qwen3_moe")
    self.assertEqual(cfg.num_experts, 128)
    self.assertEqual(cfg.num_experts_per_tok, 8)
    self.assertIn("--xla_tpu_scoped_vmem_limit_kib=65536", model.xla_flags)

  def test_fsdp_prefetch_track_a_recipe_initialization(self) -> None:
    """Verifies qwen3_235b_a22b_v7x_bf16_fsdp_prefetch passes initialize."""
    model = _MODELS[_RECIPE_PREFETCH]
    params = model.tuning_params
    cfg = pyconfig.initialize(self._argv, model_name=model.model_type, **params)
    self.assertEqual(cfg.model_name, "qwen3-235b-a22b")
    self.assertEqual(cfg.per_device_batch_size, 16.0)
    self.assertEqual(cfg.max_target_length, 4096)
    self.assertEqual(cfg.dataset_type, "synthetic")
    self.assertFalse(cfg.use_random_routing)
    self.assertEqual(cfg.ici_fsdp_parallelism, -1)
    self.assertEqual(cfg.ici_expert_parallelism, 1)
    self.assertTrue(cfg.moe_flat_fsdp_weights)
    self.assertTrue(cfg.prefetch_fsdp_weights)
    self.assertTrue(cfg.prefetch_fsdp_defer_grad_reduce_scatter)
    self.assertTrue(cfg.use_tokamax_gmm)
    self.assertTrue(cfg.use_gmm_v2)
    self.assertFalse(cfg.shard_embed_moe_on_fsdp)
    self.assertEqual(cfg.shard_mode, ShardMode.EXPLICIT)
    self.assertTrue(cfg.moe_gmm_v2_dlhs_transpose_rhs)
    self.assertTrue(cfg.moe_use_direct_token_gather)
    self.assertEqual(cfg.remat_policy, "custom")
    self.assertEqual(cfg.decoder_layer_input, RematLocation.DEVICE)
    self.assertIn(_SC_AG_FLAG, model.xla_flags)

  def test_roe_track_b_recipe_initialization(self) -> None:
    """Verifies qwen3_235b_a22b_v7x_bf16_roe passes pyconfig.initialize."""
    model = _MODELS[_RECIPE_ROE]
    params = model.tuning_params
    cfg = pyconfig.initialize(self._argv, model_name=model.model_type, **params)
    self.assertEqual(cfg.model_name, "qwen3-235b-a22b")
    self.assertEqual(cfg.per_device_batch_size, 16.0)
    self.assertEqual(cfg.max_target_length, 4096)
    self.assertEqual(cfg.dataset_type, "synthetic")
    self.assertTrue(cfg.use_random_routing)
    self.assertEqual(cfg.ici_fsdp_parallelism, -1)
    self.assertEqual(cfg.ici_expert_parallelism, 16)
    self.assertTrue(cfg.use_ring_of_experts)
    self.assertTrue(cfg.use_ragged_sort)
    self.assertTrue(cfg.moe_tc_ragged_sort)
    self.assertTrue(cfg.moe_tc_ragged_3d_dispatch)
    self.assertTrue(cfg.moe_tc_ragged_3d_gmm)
    self.assertTrue(cfg.moe_tc_ragged_weights_on_activation)
    self.assertTrue(cfg.moe_tc_window_routing)
    self.assertEqual(cfg.ragged_buffer_factor, 2.5)
    self.assertTrue(cfg.use_tokamax_gmm)
    self.assertTrue(cfg.use_gmm_v2)
    self.assertEqual(
        cfg.custom_mesh_and_rule,
        CustomRule.FSDP_AS_DP_FOR_ATTN_RF,
    )
    self.assertEqual(cfg.remat_policy, "custom")
    self.assertEqual(cfg.decoder_layer_input, RematLocation.DEVICE)
    self.assertEqual(cfg.moe_routing_maps, RematLocation.DEVICE)
    self.assertEqual(cfg.moe_tc_routing_checkpoint, RematLocation.DEVICE)
    self.assertEqual(cfg.wi_tile_fwd_batch_seq, 512)
    self.assertEqual(cfg.wi_tile_dlhs_batch_seq, 256)
    self.assertEqual(cfg.wi_tile_dlhs_embed_dim, 1536)
    self.assertEqual(cfg.wi_tile_drhs_batch_seq, 1024)
    self.assertEqual(cfg.wi_tile_drhs_embed_dim, 1024)
    self.assertEqual(cfg.wo_tile_dlhs_batch_seq, 256)
    self.assertEqual(cfg.wo_tile_dlhs_embed_dim, 4096)
    self.assertEqual(cfg.wo_tile_drhs_batch_seq, 1024)
    self.assertEqual(cfg.wo_tile_drhs_embed_dim, 1024)
    self.assertIn(_ASYNC_RAGGED_A2A_FLAG, model.xla_flags)

  def test_xla_flags_library_v7x_helpers(self) -> None:
    """Verifies v7x XLA flag constants and helper functions."""
    vmem_flag = xla_flags_library.V7X_MOE_VMEM_LIMIT_FLAG
    moe_async = xla_flags_library.V7X_MOE_ASYNC_COLLECTIVES
    sc_async = xla_flags_library.V7X_SPARSECORE_ASYNC_COLLECTIVES
    self.assertIn("--xla_tpu_scoped_vmem_limit_kib=65536", vmem_flag)
    self.assertIn(_ASYNC_RAGGED_A2A_FLAG, moe_async)
    self.assertIn("--xla_tpu_force_async_all_to_all=true", moe_async)
    self.assertIn(_SC_AG_FLAG, sc_async)
    baseline_flags = xla_flags_library.get_v7x_qwen3_235b_flags("baseline")
    self.assertIn("--xla_tpu_scoped_vmem_limit_kib=65536", baseline_flags)
    self.assertIn("--xla_enable_async_all_gather=true", baseline_flags)
    prefetch_flags = xla_flags_library.get_v7x_qwen3_235b_flags("fsdp_prefetch")
    self.assertIn(_SC_AG_FLAG, prefetch_flags)
    roe_flags = xla_flags_library.get_v7x_qwen3_235b_flags("roe")
    self.assertIn(_ASYNC_RAGGED_A2A_FLAG, roe_flags)
    with self.assertRaisesRegex(ValueError, "Unknown track: invalid_track"):
      xla_flags_library.get_v7x_qwen3_235b_flags("invalid_track")

  @parameterized.named_parameters(
      (
          "use_random_routing_when_ep_rank_is_1",
          _RECIPE_PREFETCH,
          {"use_random_routing": True},
          r"When EP rank is 1, use_random_routing must be False",
      ),
      (
          "shard_embed_moe_on_fsdp_in_unquantized_bf16",
          _RECIPE_PREFETCH,
          {"shard_embed_moe_on_fsdp": True},
          r"shard_embed_moe_on_fsdp requires quantization",
      ),
      (
          "moe_tc_ragged_sort_requires_ring_of_experts",
          _RECIPE_ROE,
          {"use_ring_of_experts": False},
          r"moe_tc_ragged_sort=True requires use_ring_of_experts=True",
      ),
      (
          "moe_tc_3d_dispatch_requires_3d_gmm",
          _RECIPE_ROE,
          {"moe_tc_ragged_3d_gmm": False},
          r"moe_tc_ragged_3d_dispatch=True requires moe_tc_ragged_sort=True",
      ),
      (
          "moe_tc_window_routing_requires_ragged_sort",
          _RECIPE_ROE,
          {
              "moe_tc_ragged_sort": False,
              "moe_tc_ragged_3d_dispatch": False,
              "moe_tc_routing_checkpoint": "remat",
          },
          r"moe_tc_window_routing=True requires moe_tc_ragged_sort=True",
      ),
      (
          "ragged_buffer_factor_without_ragged_sort",
          _RECIPE_ROE,
          {"use_ragged_sort": False},
          r"Ragged buffer factor is currently only supported with",
      ),
  )
  def test_paired_negative_invalid_flag_combinations(
      self, recipe_name: str, overrides: dict[str, object], error_regex: str
  ) -> None:
    """Verifies invalid flag overrides on registered recipes raise ValueError."""
    model = _MODELS[recipe_name]
    params = {**model.tuning_params, **overrides}
    with self.assertRaisesRegex(ValueError, error_regex):
      pyconfig.initialize(self._argv, model_name=model.model_type, **params)


if __name__ == "__main__":
  absltest.main()
