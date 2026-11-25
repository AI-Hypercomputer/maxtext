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

"""Tests for the new pydantic-based configuration system."""

import math
import os
import sys
import unittest.mock

from absl.testing import absltest
from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.layers import quantizations
from maxtext.utils import globals as maxtext_globals
from tests.utils.test_helpers import get_test_config_path
import pydantic

# Path to the base.yml config.
_BASE_CONFIG_PATH = os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml")


class ConfigTest(absltest.TestCase):
  """Tests for the new pydantic-based configuration system."""

  def test_basic_config_loading(self):
    """Tests that a basic config loads and we can access a value."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "steps=1"]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.run_name, "test")
    self.assertEqual(config.steps, 1)
    self.assertIsInstance(config, pyconfig.HyperParameters)

  def test_type_conversion(self):
    """Tests that CLI arguments are converted to the correct types."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "per_device_batch_size=3.5",
        "enable_checkpointing=false",
        "steps=50",
    ]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.per_device_batch_size, 3.5)
    self.assertIsInstance(config.per_device_batch_size, float)
    self.assertEqual(config.enable_checkpointing, False)
    self.assertIsInstance(config.enable_checkpointing, bool)
    self.assertEqual(config.steps, 50)
    self.assertIsInstance(config.steps, int)

  def test_model_override(self):
    """Tests that model-specific configs override base.yml."""
    argv = ["", _BASE_CONFIG_PATH, "model_name=llama3.1-8b", "run_name=test"]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.base_emb_dim, 4096)  # From llama3.1-8b.yml
    self.assertEqual(config.base_num_decoder_layers, 32)  # From llama3.1-8b.yml
    self.assertEqual(config.decoder_block, types.DecoderBlockType.LLAMA2)  # from llama3.1-8b.yml
    self.assertEqual(config.steps, 150001)  # From base.yml, not overridden

  def test_derived_values(self):
    """Tests that derived values are calculated correctly."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "global_parameter_scale=4",
        "per_device_batch_size=8",
        "gradient_accumulation_steps=2",
    ]
    # Mock jax.devices() to be deterministic
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    # global_parameter_scale=4 -> emb_scale=1, num_head_scale=1, mlp_dim_scale=1, layer_scale=0
    # base_emb_dim=2048, base_num_query_heads=16, base_mlp_dim=7168
    self.assertEqual(config.emb_dim, 2048 * (2**1))
    self.assertEqual(config.num_query_heads, 16 * (2**1))
    self.assertEqual(config.mlp_dim, 7168 * (2**1))

    # global_batch_size_to_train_on = per_device_batch_size * num_devices * gradient_accumulation_steps
    # num_devices is mocked to 8
    self.assertEqual(config.global_batch_size_to_train_on, 8 * 8 * 2)

  def test_unavailable_device_count_raises(self):
    """Config validation raises instead of assuming 1 device, which would derive the wrong global batch size."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "num_slices=1"]
    with unittest.mock.patch("jax.devices", side_effect=RuntimeError("backend not initialized")):
      with self.assertRaisesRegex(ValueError, "Could not determine the number of devices"):
        pyconfig.initialize(argv)

  def test_validation_error(self):
    """Tests that a validation error is raised for invalid config."""
    # A negative number for steps should trigger a ValidationError in the pydantic model.
    argv = ["", _BASE_CONFIG_PATH, "steps=-5"]
    with self.assertRaises(pydantic.ValidationError):
      pyconfig.initialize(argv)

  def test_mtp_norm_main_hidden_state_rejects_unsupported_combinations(self):
    """`mtp_norm_main_hidden_state` is rejected with vocab tiling or expanded mHC streams."""
    base = ["", _BASE_CONFIG_PATH, "run_name=test", "mtp_num_layers=1", "mtp_norm_main_hidden_state=True"]
    pyconfig.initialize(base)
    with self.assertRaisesRegex(ValueError, "num_vocab_tiling"):
      pyconfig.initialize(base + ["num_vocab_tiling=2"])
    with self.assertRaisesRegex(ValueError, "mhc_expansion_rate"):
      pyconfig.initialize(base + ["mhc_expansion_rate=2"])

  def test_enable_mllog_without_eval_warns(self):
    """enable_mllog with eval_interval <= 0 is allowed (for measuring eval impact) and logs a warning."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "steps=1", "enable_mllog=true", "eval_interval=-1"]
    with unittest.mock.patch("maxtext.utils.max_logging.warning") as mock_warning:
      config = pyconfig.initialize(argv)
    self.assertTrue(config.enable_mllog)
    self.assertTrue(
        any("enable_mllog=True with eval_interval=" in str(call.args[0]) for call in mock_warning.call_args_list)
    )

  def test_splash_diag_skip_rejects_invalid_grid(self):
    for grid in (1, 3, 6):
      with self.subTest(grid=grid):
        argv = ["", _BASE_CONFIG_PATH, "run_name=test", f"sa_qk_diag_grid={grid}"]
        with self.assertRaisesRegex(pydantic.ValidationError, "sa_qk_diag_grid must be a power of 2"):
          pyconfig.initialize(argv)

  def test_splash_diag_skip_rejects_non_square_eval_blocks(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "use_tokamax_splash=True",
        "sa_qk_diag_skip=True",
        "custom_mesh_and_rule_for_eval=pure-fsdp",
        "eval_sa_block_q=512",
        "eval_sa_block_kv=512",
        "eval_sa_block_kv_compute=256",
    ]
    with self.assertRaisesRegex(pydantic.ValidationError, "require square eval blocks"):
      pyconfig.initialize(argv)

  def test_splash_diag_skip_warns_when_disabled(self):
    common = ["", _BASE_CONFIG_PATH, "run_name=test", "use_tokamax_splash=True", "sa_sv_diag_skip=True"]
    # (overrides, expected warning substring or None)
    cases = (
        ([], None),
        (["attention_type=full"], "attention_type='full'"),
        (["ici_context_parallelism=2", "context_parallel_load_balance=True"], "load-balanced all_gather"),
        (["ici_context_parallelism=2", "context_parallel_load_balance=False"], None),
    )
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    for overrides, expected in cases:
      with self.subTest(overrides=overrides):
        with (
            unittest.mock.patch("jax.devices", return_value=mock_devices),
            unittest.mock.patch("maxtext.utils.max_logging.warning") as mock_warning,
        ):
          pyconfig.initialize(common + overrides)
        messages = [str(call.args[0]) for call in mock_warning.call_args_list if "sa_sv_diag_skip" in str(call.args[0])]
        if expected is None:
          self.assertEqual(messages, [])
        else:
          self.assertTrue(any(expected in m for m in messages), messages)

  def test_enable_mllog_accepted_with_eval(self):
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "steps=1", "enable_mllog=true", "eval_interval=1"]
    config = pyconfig.initialize(argv)
    self.assertTrue(config.enable_mllog)

  def test_te_moe_block_rejects_unsupported_options_during_config_validation(self):
    common_config = {
        "run_name": "test",
        "num_experts": 2,
        "base_moe_mlp_dim": 7168,
        "first_num_dense_layers": 1,
        "sparse_matmul": True,
        "prefuse_moe_weights": True,
        "te_moe_block": True,
        "te_gmm_quantization": "te_no_quant",
        # Prevent the general EP validation from masking the more specific
        # TE-specific incompatibilities under test.
        "override_logical_axis_rules": True,
    }
    invalid_configs = (
        (
            {"norm_topk_prob": True},
            "te_moe_block=True does not currently support norm_topk_prob=True "
            "when routed_score_func is not 'sqrtsoftplus' or 'sigmoid'.",
        ),
        (
            {"decoder_block": types.DecoderBlockType.LLAMA4},
            "te_moe_block=True does not currently support Llama4 routing semantics.",
        ),
        (
            {"te_gmm_quantization": ""},
            "te_gmm_quantization must be specified when te_moe_block=True.",
        ),
    )

    for overrides, expected_error in invalid_configs:
      with self.subTest(overrides=overrides):
        with self.assertRaises(pydantic.ValidationError) as context:
          types.MaxTextConfig(**{**common_config, **overrides})
        self.assertIn(expected_error, str(context.exception))

  def test_te_moe_block_accepts_random_routing(self):
    config = types.MaxTextConfig(
        run_name="test",
        num_experts=2,
        base_moe_mlp_dim=7168,
        first_num_dense_layers=1,
        sparse_matmul=True,
        prefuse_moe_weights=True,
        te_moe_block=True,
        te_gmm_quantization="te_no_quant",
        override_logical_axis_rules=True,
        use_random_routing=True,
        load_balance_loss_weight=0.0,
        routed_bias_update_rate=0.0,
        routed_bias=False,
    )
    self.assertTrue(config.te_moe_block)
    self.assertTrue(config.use_random_routing)

  def test_te_moe_disable_cudnn_gmm_fusion_requires_te_moe_block(self):
    common_config = {
        "run_name": "test",
        "num_experts": 2,
        "base_moe_mlp_dim": 7168,
        "first_num_dense_layers": 1,
        "sparse_matmul": True,
        "prefuse_moe_weights": True,
        "te_gmm_quantization": "te_no_quant",
        "override_logical_axis_rules": True,
    }

    self.assertFalse(types.MaxTextConfig(**common_config).te_moe_disable_cudnn_gmm_fusion)
    self.assertTrue(
        types.MaxTextConfig(
            **common_config, te_moe_block=True, te_moe_disable_cudnn_gmm_fusion=True
        ).te_moe_disable_cudnn_gmm_fusion
    )
    with self.assertRaisesRegex(
        pydantic.ValidationError, "te_moe_disable_cudnn_gmm_fusion=True requires te_moe_block=True"
    ):
      types.MaxTextConfig(**common_config, te_moe_disable_cudnn_gmm_fusion=True)

  def test_te_moe_quantized_weight_gather_validation(self):
    common = {
        "run_name": "test",
        "num_experts": 2,
        "base_moe_mlp_dim": 7168,
        "first_num_dense_layers": 1,
        "sparse_matmul": True,
        "prefuse_moe_weights": True,
        "te_moe_block": True,
        "te_gmm_quantization": "te_mxfp8",
        "te_moe_quantize_before_fsdp_all_gather": True,
        "override_logical_axis_rules": True,
    }
    self.assertTrue(types.MaxTextConfig(**common).te_moe_quantize_before_fsdp_all_gather)
    self.assertFalse(
        types.MaxTextConfig(
            **{**common, "te_moe_quantize_before_fsdp_all_gather": False}
        ).te_moe_quantize_before_fsdp_all_gather
    )
    for overrides in (
        {"te_gmm_quantization": "te_no_quant"},
        {"use_2d_fsdp_sharding": True},
        {"te_moe_quantize_before_fsdp_all_gather": "invalid"},
    ):
      with self.subTest(overrides=overrides):
        with self.assertRaises(pydantic.ValidationError):
          types.MaxTextConfig(**{**common, **overrides})

  def test_te_moe_block_uses_ragged_buffer_factor_validation(self):
    common_config = {
        "run_name": "test",
        "num_experts": 2,
        "base_moe_mlp_dim": 7168,
        "first_num_dense_layers": 1,
        "sparse_matmul": True,
        "prefuse_moe_weights": True,
        "te_moe_block": True,
        "te_gmm_quantization": "te_no_quant",
    }

    # TE accepts the native worst-case sentinel and a finite factor even when
    # EP is not enabled in the MaxText logical rules.
    for factor in (-1.0, 0.0, 1.0, 1.5):
      with self.subTest(factor=factor):
        config = types.MaxTextConfig(**common_config, ragged_buffer_factor=factor)
        self.assertEqual(config.ragged_buffer_factor, factor)

    with self.assertRaises(pydantic.ValidationError) as context:
      types.MaxTextConfig(**common_config, ragged_buffer_factor=0.5)
    self.assertIn("te_moe_block=True requires ragged_buffer_factor >= 1.0", str(context.exception))

    # Native ring-of-experts keeps its existing ragged-sort restriction, while
    # TE MoEBlock bypasses that native implementation and its validation.
    native_ring_config = {
        **common_config,
        "te_moe_block": False,
        "te_gmm_quantization": "",
        "override_logical_axis_rules": True,
        "use_ring_of_experts": True,
        "ragged_buffer_factor": 1.5,
    }
    with self.assertRaises(pydantic.ValidationError) as context:
      types.MaxTextConfig(**native_ring_config)
    self.assertIn("Ragged buffer factor is currently only supported with", str(context.exception))

    te_ring_config = {**native_ring_config, "te_moe_block": True, "te_gmm_quantization": "te_no_quant"}
    config = types.MaxTextConfig(**te_ring_config)
    self.assertEqual(config.ragged_buffer_factor, 1.5)

  def test_moe_dropless_fallback_modes(self):
    common_config = {
        "run_name": "test",
        "num_experts": 8,
        "base_mlp_dim": 64,
        "base_moe_mlp_dim": 64,
        "override_logical_axis_rules": True,
        "use_ring_of_experts": True,
        "use_ragged_sort": True,
        "ragged_buffer_factor": 1.5,
    }
    for mode in (None, "step", "layer"):
      with self.subTest(moe_dropless_fallback=mode):
        config = types.MaxTextConfig(**common_config, moe_dropless_fallback=mode)
        self.assertEqual(config.moe_dropless_fallback, mode)

    # YAML/CLI "none" reaches pydantic as None, so it must mean "off".
    for raw in ("none", "None"):
      with self.subTest(raw=raw):
        raw_keys = {**common_config, "moe_dropless_fallback": raw}
        kwargs = pyconfig._prepare_for_pydantic(raw_keys)  # pylint: disable=protected-access
        self.assertIsNone(types.MaxTextConfig(**kwargs).moe_dropless_fallback)

    with self.assertRaises(pydantic.ValidationError):
      types.MaxTextConfig(**common_config, moe_dropless_fallback="both")

    # The ragged all-to-all path (no ring of experts) supports the in-layer fallback, not the step replay.
    a2a_config = {**common_config, "use_ring_of_experts": False}
    self.assertEqual(types.MaxTextConfig(**a2a_config, moe_dropless_fallback="layer").moe_dropless_fallback, "layer")
    with self.assertRaises(pydantic.ValidationError) as context:
      types.MaxTextConfig(**a2a_config, moe_dropless_fallback="step")
    self.assertIn("only supported with use_ring_of_experts=True", str(context.exception))

  def test_tpu_tokamax_ring_config_validation_accepts_initial_config(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=2",
        "ring_scan_unroll=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertEqual(config.context_parallel_strategy, "ring")
    self.assertEqual(config.ici_context_parallelism, 2)
    self.assertEqual(config.ring_scan_unroll, 2)
    self.assertEqual(config.attention, "flash")
    self.assertTrue(config.use_tokamax_splash)

  def test_tpu_tokamax_ring_config_validation_accepts_load_balance(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=True",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertTrue(config.context_parallel_load_balance)

  def test_tpu_tokamax_ring_config_validation_accepts_mla(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "attention_type=mla",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertEqual(config.attention_type, "mla")

  def test_tpu_tokamax_ring_config_validation_accepts_packing(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=True",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertTrue(config.packing)

  def test_tpu_tokamax_ring_config_validation_accepts_packed_load_balance(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=True",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=True",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertTrue(config.context_parallel_load_balance)
    self.assertTrue(config.packing)

  def test_tpu_tokamax_ring_config_validation_accepts_indexer(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "attention_type=mla",
        "use_indexer=True",
        "q_lora_rank=1",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertTrue(config.use_indexer)
    self.assertEqual(config.attention_type, "mla")

  def test_indexer_all_gather_context_parallelism_rejects_attention_sink(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "attention_type=mla",
        "use_indexer=True",
        "q_lora_rank=1",
        "attention_sink=True",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=all_gather",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      with self.assertRaisesRegex(ValueError, "does not support attention sinks"):
        pyconfig.initialize(argv)

  def test_tpu_tokamax_ring_config_validation_rejects_unsupported_configs(self):
    base_args = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ring",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    cases = [
        (["ici_context_parallelism=1"], ["ici_context_parallelism=2"], "context_parallel_size > 1"),
        (["context_sharding=expert", "ici_expert_parallelism=2"], [], "context_sharding"),
        (["dq_reduction_steps=2"], [], "dq_reduction_steps"),
        (["ring_scan_unroll=-1"], [], "ring_scan_unroll"),
        (["max_target_length=2050"], [], "context_parallel_size squared"),
        (["attention=dot_product"], ["attention=flash"], "attention=flash"),
        (["use_tokamax_splash=False"], ["use_tokamax_splash=True"], "use_tokamax_splash"),
        (["use_jax_splash=True"], ["use_jax_splash=False"], "use_jax_splash"),
        (["attention_type=full"], [], "attention_type"),
        (["attention_type=local_sliding", "sliding_window_size=128"], [], "attention_type"),
        (["attention_type=chunk", "chunk_attn_window_size=128"], [], "attention_type"),
        (["attention_type=compressed"], [], "attention_type"),
        (["attention_type=mla", "packing=True"], ["packing=False"], "packing"),
        (["attention_type=mla", "use_batch_split_schedule=True"], [], "batch-split"),
        (["attention_type=block_diffusion"], [], "attention_type"),
        (
            [
                "context_parallel_load_balance=True",
                "ici_context_parallelism=3",
                "max_target_length=2304",
            ],
            ["context_parallel_load_balance=False", "ici_context_parallelism=2"],
            "even context_parallel_size",
        ),
        (
            [
                "context_parallel_load_balance=True",
                "mtp_num_layers=1",
            ],
            ["context_parallel_load_balance=False"],
            "MTP",
        ),
        (["use_ragged_attention=True"], [], "ragged attention"),
        (["attention_sink=True"], [], "attention sinks"),
        (["use_chunked_prefill=True"], [], "chunked prefill"),
        (["moba=True"], [], "MoBA"),
        (["use_multimodal=True"], [], "multimodal"),
        (["use_qk_clip=True"], [], "QK-Clip"),
        (["dropout_rate=0.1"], [], "dropout"),
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    for bad_args, args_to_remove, expected_regex in cases:
      with self.subTest(bad_args=bad_args):
        argv = [arg for arg in base_args if arg not in args_to_remove]
        argv.extend(bad_args)
        with unittest.mock.patch("jax.devices", return_value=mock_devices):
          with self.assertRaisesRegex((ValueError, pydantic.ValidationError), expected_regex):
            pyconfig.initialize(argv)

  def test_compressed_attention_rejects_context_parallelism(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "attention_type=compressed",
        "compress_ratios=[0, 128]",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      with self.assertRaisesRegex(ValueError, "Context parallelism .* is not supported with attention_type='compressed'"):
        pyconfig.initialize(argv)

  def test_compressed_attention_allows_context_parallelism_when_all_ratios_are_zero(self):
    """A compress_ratio of 0 downgrades the layer to local sliding, so CP stays legal."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "attention_type=compressed",
        "compress_ratios=[0, 0]",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)
    self.assertEqual(config.attention_type, "compressed")

  def test_compressed_attention_rejects_unsupported_dq_reduction_steps(self):
    base_args = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "attention_type=compressed",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      with self.assertRaisesRegex(ValueError, "requires dq_reduction_steps to be 0 or 3"):
        pyconfig.initialize(base_args + ["dq_reduction_steps=2"])
      # 0 (unset) and 3 both remain valid.
      for valid in ("dq_reduction_steps=0", "dq_reduction_steps=3"):
        with self.subTest(valid=valid):
          pyconfig.initialize(base_args + [valid])

  def test_tpu_ulysses_config_validation_accepts_initial_config(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ulysses",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=4",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertEqual(config.context_parallel_strategy, "ulysses")
    self.assertEqual(config.ici_context_parallelism, 4)
    self.assertFalse(config.context_parallel_load_balance)

  def test_tpu_ulysses_config_validation_accepts_fewer_kv_heads_than_context_ranks(
      self,
  ):
    """1 (MQA) or 2 KV heads at context 4 or 8 are repeated across the context ranks."""
    for base_num_kv_heads in (1, 2):
      for ici_context_parallelism in (4, 8):
        with self.subTest(
            base_num_kv_heads=base_num_kv_heads,
            ici_context_parallelism=ici_context_parallelism,
        ):
          argv = [
              "",
              _BASE_CONFIG_PATH,
              "run_name=test",
              "attention=flash",
              "use_tokamax_splash=True",
              "use_jax_splash=False",
              "context_parallel_strategy=ulysses",
              "context_parallel_load_balance=False",
              f"ici_context_parallelism={ici_context_parallelism}",
              "base_num_query_heads=32",
              f"base_num_kv_heads={base_num_kv_heads}",
              "hardware=tpu",
              "packing=True",
              "skip_jax_distributed_system=True",
          ]
          mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
          with unittest.mock.patch("jax.devices", return_value=mock_devices):
            config = pyconfig.initialize(argv)
          self.assertEqual(config.num_kv_heads, base_num_kv_heads)
          self.assertEqual(config.context_parallel_strategy, "ulysses")

  def test_tpu_ulysses_config_validation_accepts_packing(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ulysses",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=4",
        "hardware=tpu",
        "packing=True",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertTrue(config.packing)

  def test_tpu_ulysses_config_validation_accepts_full_attention(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "attention_type=full",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ulysses",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=4",
        "hardware=tpu",
        "packing=True",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertEqual(config.attention_type, "full")
    self.assertTrue(config.packing)

  def test_context_parallel_strategy_is_normalized(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=Ulysses",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=4",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertEqual(config.context_parallel_strategy, "ulysses")

  def test_tpu_ulysses_config_validation_rejects_unsupported_configs(self):
    base_args = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=ulysses",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=4",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    cases = [
        (["context_parallel_load_balance=True"], ["context_parallel_load_balance=False"], "load_balance"),
        (["base_num_query_heads=18"], [], "requires num_query_heads"),
        (["base_num_kv_heads=10"], [], "requires num_kv_heads"),
        (
            ["base_num_query_heads=24", "base_num_kv_heads=6"],
            [],
            r"requires num_kv_heads \(6\).*or to divide it",
        ),
        (["attention_type=mla"], [], "global causal or full attention"),
        (["attention_type=local_sliding", "sliding_window_size=128"], [], "global causal or full attention"),
        (["attention_type=chunk", "chunk_attn_window_size=128"], [], "global causal or full attention"),
        (["attention_type=compressed"], [], "global causal or full attention"),
        (["use_qk_clip=True"], [], "QK-Clip"),
        (["dq_reduction_steps=2"], [], "dq_reduction_steps"),
        (["attention=dot_product"], ["attention=flash"], "attention=flash"),
        (["use_tokamax_splash=False"], ["use_tokamax_splash=True"], "use_tokamax_splash"),
        (["use_jax_splash=True"], ["use_jax_splash=False"], "use_jax_splash"),
        (["max_target_length=2050"], [], "divisible by context_parallel_size"),
        (["ici_context_parallelism=-1"], ["ici_context_parallelism=4"], "explicit positive"),
        (["dcn_context_parallelism=-1"], [], "explicit positive"),
        (["dcn_context_parallelism=2"], [], "dcn context parallelism"),
        (
            ["ici_context_parallelism=-1", "dcn_context_parallelism=-1"],
            ["ici_context_parallelism=4"],
            "explicit positive",
        ),
        (["ici_context_parallelism=1"], ["ici_context_parallelism=4"], "context_parallel_size > 1"),
        (["context_sharding=expert"], [], "context_sharding"),
        (["use_ragged_attention=True"], [], "ragged attention"),
        (["attention_sink=True"], [], "attention sinks"),
        (["use_indexer=True", "attention_type=mla", "q_lora_rank=1"], [], "sparse indexer"),
        (["use_chunked_prefill=True"], [], "chunked prefill"),
        (["moba=True"], [], "MoBA"),
        (["use_multimodal=True"], [], "multimodal"),
        (["dropout_rate=0.1"], [], "dropout"),
        (["context_parallel_strategy=ulysess"], ["context_parallel_strategy=ulysses"], "context_parallel_strategy"),
        (["hardware=gpu"], ["hardware=tpu"], "only supported on TPU"),
        (["hardware=gpu_multiprocess"], ["hardware=tpu"], "only supported on TPU"),
        (["hardware=cpu"], ["hardware=tpu"], "only supported on TPU"),
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    for bad_args, args_to_remove, expected_regex in cases:
      with self.subTest(bad_args=bad_args):
        argv = [arg for arg in base_args if arg not in args_to_remove]
        argv.extend(bad_args)
        with unittest.mock.patch("jax.devices", return_value=mock_devices):
          with self.assertRaisesRegex((ValueError, pydantic.ValidationError), expected_regex):
            pyconfig.initialize(argv)

  def test_tpu_usp_config_validation_accepts_initial_config(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=usp",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=2",
        "ici_context_usp_ulysses_parallelism=2",
        "ring_scan_unroll=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertEqual(config.context_parallel_strategy, "usp")
    self.assertEqual(config.ici_context_parallelism, 2)
    self.assertEqual(config.ici_context_usp_ulysses_parallelism, 2)
    self.assertEqual(config.ring_scan_unroll, 2)
    self.assertEqual(config.ulysses_context_sharding, "context_usp_ulysses")
    context_usp_ulysses_index = config.mesh_axes.index("context_usp_ulysses")
    self.assertEqual(context_usp_ulysses_index, config.mesh_axes.index("context") + 1)
    self.assertEqual(config.ici_parallelism[context_usp_ulysses_index], 2)
    self.assertEqual(types.infer_cp_axes(config.logical_axis_rules), ("context", "context_usp_ulysses"))
    self.assertEqual(types.infer_cp_axes(config.logical_axis_rules_for_eval), ("context", "context_usp_ulysses"))

  def test_tpu_usp_config_validation_accepts_packing(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=usp",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=2",
        "ici_context_usp_ulysses_parallelism=2",
        "hardware=tpu",
        "packing=True",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertTrue(config.packing)

  def test_tpu_usp_config_validation_accepts_load_balance(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=usp",
        "context_parallel_load_balance=True",
        "ici_context_parallelism=2",
        "ici_context_usp_ulysses_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertTrue(config.context_parallel_load_balance)

  def test_context_usp_ulysses_parallelism_requires_usp(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "ici_context_usp_ulysses_parallelism=2",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      with self.assertRaisesRegex(
          (ValueError, pydantic.ValidationError), "only supported when context_parallel_strategy='usp'"
      ):
        pyconfig.initialize(argv)

  def test_tpu_usp_config_validation_rejects_unsupported_configs(self):
    base_args = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "attention=flash",
        "use_tokamax_splash=True",
        "use_jax_splash=False",
        "context_parallel_strategy=usp",
        "context_parallel_load_balance=False",
        "ici_context_parallelism=2",
        "ici_context_usp_ulysses_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    cases = [
        (
            ["context_parallel_load_balance=True", "ici_context_parallelism=3"],
            ["context_parallel_load_balance=False", "ici_context_parallelism=2"],
            "even ici_context_parallelism",
        ),
        (["attention=dot_product"], ["attention=flash"], "attention=flash"),
        (["use_tokamax_splash=False"], ["use_tokamax_splash=True"], "use_tokamax_splash"),
        (["use_jax_splash=True"], ["use_jax_splash=False"], "use_jax_splash"),
        (["attention_type=mla"], [], "global causal attention"),
        (["use_ragged_attention=True"], [], "ragged attention"),
        (["attention_sink=True"], [], "attention sinks"),
        (["use_indexer=True", "attention_type=mla", "q_lora_rank=1"], [], "sparse indexer"),
        (["use_chunked_prefill=True"], [], "chunked prefill"),
        (["use_multimodal=True"], [], "multimodal"),
        (["dropout_rate=0.1"], [], "dropout"),
        (["dq_reduction_steps=2"], [], "dq_reduction_steps"),
        (["use_qk_clip=True"], [], "QK-Clip"),
        (["context_sharding=expert"], [], "context_sharding"),
        (["ulysses_context_sharding=expert"], [], "ulysses_context_sharding"),
        (["custom_mesh_and_rule=pure-fsdp"], [], "mesh axis 'context' in"),
        (["custom_mesh_and_rule=cp-as-ep"], [], "mesh axis 'context_usp_ulysses' in"),
        (["logical_axis_rules=[['activation_length',['context']]]"], [], r"in logical_axis_rules\."),
        (["custom_mesh_and_rule_for_eval=pure-fsdp"], [], "logical_axis_rules_for_eval"),
        (["ici_context_parallelism=1"], ["ici_context_parallelism=2"], "ring dimension"),
        (["ici_context_usp_ulysses_parallelism=1"], ["ici_context_usp_ulysses_parallelism=2"], "Ulysses dimension"),
        (["ici_context_parallelism=-1"], ["ici_context_parallelism=2"], "explicit positive"),
        (["ici_context_usp_ulysses_parallelism=-1"], ["ici_context_usp_ulysses_parallelism=2"], "explicit positive"),
        (["dcn_context_parallelism=2"], [], "dcn context parallelism"),
        (["dcn_context_usp_ulysses_parallelism=2"], [], "dcn context parallelism"),
        (["dcn_context_parallelism=-1"], [], "explicit positive"),
        (["dcn_context_usp_ulysses_parallelism=-1"], [], "explicit positive"),
        (["mtp_num_layers=1"], [], "multi-token prediction"),
        (["sa_bwd_dkv_megacore=True"], [], "sa_bwd_dkv_megacore"),
        (["max_target_length=2050"], [], "total context parallelism"),
        (["ici_context_parallelism=4", "max_target_length=2056"], ["ici_context_parallelism=2"], "squared"),
        (
            ["base_num_query_heads=18", "ici_context_usp_ulysses_parallelism=4"],
            ["ici_context_usp_ulysses_parallelism=2"],
            "requires num_query_heads",
        ),
        (["base_num_kv_heads=1"], [], "MQA"),
        (
            ["base_num_kv_heads=10", "ici_context_usp_ulysses_parallelism=4"],
            ["ici_context_usp_ulysses_parallelism=2"],
            "requires num_kv_heads",
        ),
        (["hardware=gpu"], ["hardware=tpu"], "only supported on TPU"),
        (["hardware=cpu"], ["hardware=tpu"], "only supported on TPU"),
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    for bad_args, args_to_remove, expected_regex in cases:
      with self.subTest(bad_args=bad_args):
        argv = [arg for arg in base_args if arg not in args_to_remove]
        argv.extend(bad_args)
        with unittest.mock.patch("jax.devices", return_value=mock_devices):
          with self.assertRaisesRegex((ValueError, pydantic.ValidationError), expected_regex):
            pyconfig.initialize(argv)

  def test_load_balanced_chunk_context_parallel_config(self):
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "steps=1",
        "attention_type=chunk",
        "chunk_attn_window_size=256",
        "context_parallel_load_balance=True",
        "ici_context_parallelism=2",
        "hardware=tpu",
        "packing=False",
        "dataset_type=synthetic",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)

    self.assertEqual(config.attention_type, "chunk")
    self.assertTrue(config.context_parallel_load_balance)

  def test_block_diffusion_attention_config(self):
    config = pyconfig.initialize(
        [
            "",
            _BASE_CONFIG_PATH,
            "run_name=test",
            "steps=1",
            "attention=dot_product",
            "attention_type=block_diffusion",
            "causal_block_size=7",
            "vocab_size=256",
            "max_target_length=2048",
            "hardware=cpu",
            "packing=False",
            "skip_jax_distributed_system=True",
        ]
    )

    self.assertEqual(config.attention_type, "block_diffusion")
    self.assertEqual(config.causal_block_size, 7)
    self.assertNotEqual(config.max_target_length % config.causal_block_size, 0)

  def test_block_diffusion_attention_rejects_unsupported_configs(self):
    base_overrides = {
        "run_name": "test",
        "steps": 1,
        "attention": "dot_product",
        "attention_type": "block_diffusion",
        "causal_block_size": 32,
        "vocab_size": 256,
        "max_target_length": 2048,
        "hardware": "cpu",
        "packing": False,
        "skip_jax_distributed_system": True,
    }
    cases = (
        {"causal_block_size": 0},
        {"packing": True},
        {"attention": "cudnn_flash_te"},
        {"attention": "flash", "hardware": "gpu"},
    )
    for overrides in cases:
      with self.subTest(overrides=overrides):
        values = base_overrides | overrides
        argv = ["", _BASE_CONFIG_PATH, *(f"{key}={value}" for key, value in values.items())]
        with self.assertRaises((ValueError, pydantic.ValidationError)):
          pyconfig.initialize(argv)

  def test_full_attention_rejects_cudnn_flash_kernels(self):
    for kernel in ("cudnn_flash_te", "cudnn_flash_jax"):
      with self.subTest(attention=kernel):
        argv = [
            "",
            _BASE_CONFIG_PATH,
            "run_name=test",
            "steps=1",
            "attention_type=full",
            f"attention={kernel}",
            "hardware=gpu",
            "skip_jax_distributed_system=True",
        ]
        with self.assertRaisesRegex(pydantic.ValidationError, f"attention={kernel!r}"):
          pyconfig.initialize(argv)

  def test_default_attention_remains_global(self):
    config = pyconfig.initialize(["", _BASE_CONFIG_PATH, "run_name=test", "steps=1"])

    self.assertEqual(config.attention_type, "global")
    self.assertEqual(config.training_objective, "causal_lm")
    self.assertEqual(config.block_diffusion_mask_id, -1)

  def test_cpu_hardware_disables_megablox(self):
    """Tests that megablox is automatically disabled when hardware=cpu."""
    config = pyconfig.initialize(
        [
            "",
            _BASE_CONFIG_PATH,
            "run_name=test",
            "hardware=cpu",
            "megablox=True",
            "skip_jax_distributed_system=True",
        ]
    )
    self.assertFalse(config.megablox)

  def test_block_diffusion_pretraining_config(self):
    config = pyconfig.initialize(
        [
            "",
            _BASE_CONFIG_PATH,
            "run_name=test",
            "steps=1",
            "training_objective=block_diffusion",
            "attention=dot_product",
            "attention_type=block_diffusion",
            "causal_block_size=7",
            "block_diffusion_mask_id=100",
            "block_diffusion_min_noise=0.05",
            "vocab_size=256",
            "max_target_length=2048",
            "packing=False",
            "dataset_type=hf",
            "hf_path=parquet",
            "hardware=cpu",
        ]
    )

    self.assertEqual(config.training_objective, "block_diffusion")
    self.assertEqual(config.block_diffusion_mask_id, 100)
    self.assertEqual(config.block_diffusion_min_noise, 0.05)
    self.assertEqual(config.block_diffusion_logit_alignment, "same_position")
    self.assertEqual(config.block_diffusion_canvas_policy, "all_masked")

  def test_block_diffusion_pretraining_rejects_incompatible_config(self):
    base_overrides = {
        "run_name": "test",
        "steps": 1,
        "training_objective": "block_diffusion",
        "attention": "dot_product",
        "attention_type": "block_diffusion",
        "causal_block_size": 32,
        "block_diffusion_mask_id": 100,
        "block_diffusion_min_noise": 0.05,
        "vocab_size": 256,
        "max_target_length": 2048,
        "packing": False,
        "dataset_type": "hf",
        "hf_path": "parquet",
        "hardware": "cpu",
    }
    cases = (
        ({"attention_type": "global"}, "attention_type='block_diffusion'"),
        ({"block_diffusion_mask_id": -1}, "block_diffusion_mask_id"),
        ({"block_diffusion_mask_id": 256}, "block_diffusion_mask_id"),
        ({"block_diffusion_min_noise": 0.0}, "block_diffusion_min_noise"),
        ({"packing": True}, "packing=False"),
        ({"mtp_num_layers": 1}, "MTP"),
        ({"num_vocab_tiling": 2}, "vocabulary tiling"),
        ({"dataset_type": "grain"}, "dataset_type='hf'"),
        ({"use_dpo": True}, "DPO"),
        ({"use_multimodal": True}, "text-only"),
        ({"block_diffusion_logit_alignment": "shifted"}, "seed_and_mask"),
        ({"block_diffusion_canvas_policy": "seed_and_mask"}, "same_position/all_masked"),
        (
            {
                "causal_block_size": 1,
                "block_diffusion_logit_alignment": "shifted",
                "block_diffusion_canvas_policy": "seed_and_mask",
            },
            "causal_block_size >= 2",
        ),
    )
    for overrides, expected_regex in cases:
      with self.subTest(overrides=overrides):
        values = base_overrides | overrides
        argv = ["", _BASE_CONFIG_PATH, *(f"{key}={value}" for key, value in values.items())]
        with self.assertRaisesRegex((ValueError, pydantic.ValidationError), expected_regex):
          pyconfig.initialize(argv)

  def test_block_diffusion_sft_config(self):
    config = pyconfig.initialize(
        [
            "",
            _BASE_CONFIG_PATH,
            "run_name=test",
            "steps=1",
            "training_objective=block_diffusion",
            "attention=dot_product",
            "attention_type=block_diffusion",
            "causal_block_size=8",
            "block_diffusion_mask_id=100",
            "vocab_size=256",
            "packing=False",
            "dataset_type=hf",
            "hf_path=parquet",
            "use_sft=True",
            "sft_train_on_completion_only=True",
            "hardware=cpu",
        ]
    )

    self.assertTrue(config.use_sft)
    self.assertTrue(config.sft_train_on_completion_only)

  def test_shifted_block_diffusion_requires_seeded_canvas(self):
    config = pyconfig.initialize(
        [
            "",
            _BASE_CONFIG_PATH,
            "run_name=test",
            "steps=1",
            "training_objective=block_diffusion",
            "attention=dot_product",
            "attention_type=block_diffusion",
            "causal_block_size=8",
            "block_diffusion_mask_id=100",
            "block_diffusion_logit_alignment=shifted",
            "block_diffusion_canvas_policy=seed_and_mask",
            "vocab_size=256",
            "packing=False",
            "dataset_type=hf",
            "hf_path=parquet",
            "hardware=cpu",
        ]
    )

    self.assertEqual(config.block_diffusion_logit_alignment, "shifted")
    self.assertEqual(config.block_diffusion_canvas_policy, "seed_and_mask")

  def test_rampup_batch_size_validation(self):
    """Unusable rampup settings must raise rather than silently changing the schedule.

    The schedule is derived behind guards that fall through to `rampup_end_step = 0`,
    so without validation a bad combination either disables rampup with no diagnostic
    or, when the batch-size change is not a multiple of the increment, ramps to a
    different final batch size than the one requested.
    """
    base_args = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "steps=1",
        "enable_rampup_batch_size=True",
        "per_device_batch_size_start=4",
        "per_device_batch_size=16",
        "per_device_batch_size_increment=4",
        "global_rampup_samples=500",
        "skip_jax_distributed_system=True",
    ]
    cases = [
        (["per_device_batch_size_increment=0"], "per_device_batch_size_increment must be positive"),
        (["per_device_batch_size_start=0"], "per_device_batch_size_start must be positive"),
        (["global_rampup_samples=0"], "global_rampup_samples must be positive"),
        (["per_device_batch_size=4"], "must be greater than per_device_batch_size_start"),
        (["per_device_batch_size=2"], "must be greater than per_device_batch_size_start"),
        (["per_device_batch_size=15"], "divisible by per_device_batch_size_increment"),
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    for bad_args, expected_regex in cases:
      with self.subTest(bad_args=bad_args):
        argv = base_args + bad_args
        with unittest.mock.patch("jax.devices", return_value=mock_devices):
          with self.assertRaisesRegex((ValueError, pydantic.ValidationError), expected_regex):
            pyconfig.initialize(argv)

    # A well-formed rampup config is still accepted and still produces a schedule.
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(base_args)
    self.assertTrue(config.enable_rampup_batch_size)
    # With 8 devices, expansion_factor_real_data=-1 (i.e. disabled) and
    # gradient_accumulation_steps=1: global_batch_size_to_load goes 32 -> 128
    # in 3 increments of 32, so 500 rampup samples split into ~166.67/increment,
    # ceil'd per stage at batch sizes 32/64/96 gives 6 + 3 + 2 = 11 rampup steps.
    self.assertEqual(config.rampup_end_step, 11)

  def test_rampup_validation_skipped_when_disabled(self):
    """The checks must not fire when rampup is off, whatever the rampup fields hold."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "steps=1",
        "enable_rampup_batch_size=False",
        "per_device_batch_size_start=0",
        "per_device_batch_size_increment=0",
        "skip_jax_distributed_system=True",
    ]
    mock_devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(8)]
    with unittest.mock.patch("jax.devices", return_value=mock_devices):
      config = pyconfig.initialize(argv)
    self.assertEqual(config.rampup_end_step, 0)

  @unittest.mock.patch.dict(os.environ, {pyconfig.yaml_key_to_env_key("steps"): "123"})
  def test_env_override(self):
    """Tests that environment variables override YAML values."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test"]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.steps, 123)

  @unittest.mock.patch.dict(os.environ, {pyconfig.yaml_key_to_env_key("steps"): "123"})
  def test_cli_overrides_env_is_disallowed(self):
    """Tests that CLI arguments overriding environment variables fails."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "steps=456"]
    # The new config logic explicitly forbids overriding the same key
    # from both CLI and environment variables to prevent ambiguity.
    with self.assertRaises(ValueError):
      pyconfig.initialize(argv)

  def test_llama3_tokenizer_correction(self):
    """Tests that tokenizer_type is forced to 'tiktoken' for llama3."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "model_name=llama3.1-8b",
        "tokenizer_path=assets/tokenizer_llama3.tiktoken",
        "run_name=test",
    ]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.tokenizer_type, "tiktoken")

  def test_initialize_pydantic_bad_keys(self):
    """Test that `pydantic.ValidationError` is raised on keys not in MaxTextConfig"""
    with self.assertRaises(ValueError):
      pyconfig.initialize_pydantic(
          [
              "",
              _BASE_CONFIG_PATH,
              "tokenizer_path=assets/tokenizer_llama3.tiktoken",
              "NOT_A_VALID_KEY=test",
          ]
      )

  def test_safetensors_dynamic_disallows_single_controller(self):
    """Tests that source_checkpoint_layout=safetensors_dynamic disallows enable_single_controller=True."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "source_checkpoint_layout=safetensors_dynamic",
        "enable_single_controller=true",
    ]
    with self.assertRaises(pydantic.ValidationError):
      pyconfig.initialize(argv)

  def test_elastic_backup_kind_validation(self):
    """Tests that elastic_backup_kind must be either 'snapshot' or 'checkpoint'."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "elastic_backup_kind=invalid_backup_kind",
    ]
    with self.assertRaises(pydantic.ValidationError):
      pyconfig.initialize(argv)

  def test_elastic_snapshot_mode_config(self):
    """Snapshot mode is the default, takes an interval, and is rejected where it can't work."""
    config = pyconfig.initialize(["", _BASE_CONFIG_PATH, "run_name=test"])
    self.assertEqual(config.elastic_backup_kind, "snapshot")
    self.assertEqual(config.elastic_snapshot_interval, 10)
    # Without elastic training the kind is inert, whatever the data pipeline.
    pyconfig.initialize(["", _BASE_CONFIG_PATH, "run_name=test", "dataset_type=tfds"])
    # Elastic runs outside the snapshot envelope opt out explicitly.
    config = pyconfig.initialize(
        [
            "",
            _BASE_CONFIG_PATH,
            "run_name=test",
            "elastic_enabled=true",
            "enable_single_controller=true",
            "dataset_type=tfds",
            "elastic_backup_kind=checkpoint",
        ]
    )
    self.assertEqual(config.elastic_backup_kind, "checkpoint")

    config = pyconfig.initialize(
        ["", _BASE_CONFIG_PATH, "run_name=test", "elastic_backup_kind=snapshot", "elastic_snapshot_interval=25"]
    )
    self.assertEqual(config.elastic_backup_kind, "snapshot")
    self.assertEqual(config.elastic_snapshot_interval, 25)

    with self.assertRaises(pydantic.ValidationError):
      pyconfig.initialize(["", _BASE_CONFIG_PATH, "run_name=test", "elastic_snapshot_interval=0"])

    snapshot_argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "elastic_enabled=true",
        "enable_single_controller=true",
        "elastic_backup_kind=snapshot",
    ]
    with self.assertRaisesRegex(pydantic.ValidationError, "not supported with DiLoCo"):
      pyconfig.initialize(snapshot_argv + ["enable_diloco=true"])
    with self.assertRaisesRegex(pydantic.ValidationError, "enable_rampup_batch_size"):
      pyconfig.initialize(
          snapshot_argv
          + [
              "enable_rampup_batch_size=true",
              "per_device_batch_size=4",
              "per_device_batch_size_start=1",
              "per_device_batch_size_increment=1",
              "global_rampup_samples=100",
          ]
      )
    # Synthetic data needs no iterator position; real data needs grain's ElasticIterator.
    pyconfig.initialize(snapshot_argv + ["dataset_type=synthetic"])
    with self.assertRaisesRegex(pydantic.ValidationError, "grain_use_elastic_iterator"):
      pyconfig.initialize(snapshot_argv + ["dataset_type=tfds"])
    with self.assertRaisesRegex(pydantic.ValidationError, "grain_use_elastic_iterator"):
      pyconfig.initialize(snapshot_argv + ["dataset_type=grain", "grain_train_files=/x/*"])
    pyconfig.initialize(
        snapshot_argv
        + [
            "dataset_type=grain",
            "grain_train_files=/x/*",
            "grain_use_elastic_iterator=true",
            "grain_file_type=arrayrecord",
            "packing=false",
            "colocated_python_data_input=true",
        ]
    )

  def test_te_moe_dispatch_remat_location(self):
    """Dispatch outputs participate in the custom remat policy."""
    name = "te_moe_dispatch_remat_location"
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "remat_policy=custom"]
    default_config = pyconfig.initialize(argv)
    self.assertEqual(getattr(default_config, name), types.RematLocation.REMAT)
    for location in ("remat", "device", "offload"):
      with self.subTest(location=location):
        config = pyconfig.initialize(argv + [f"{name}={location}"])
        self.assertEqual(name in config.tensors_on_device, location == "device")
        self.assertEqual(name in config.tensors_to_offload, location == "offload")
    with self.assertRaises(pydantic.ValidationError):
      pyconfig.initialize(argv + [f"{name}=invalid"])

  def test_indexer_cutoff_threshold_remat_policy(self):
    """Tests custom remat policy and validation for indexer_cutoff_threshold."""
    # 1. Verify custom remat policy puts indexer_cutoff_threshold on device
    argv_device = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "use_indexer=true",
        "attention_type=mla",
        "q_lora_rank=1536",
        "attention=dot_product",
        "remat_policy=custom",
        "indexer_cutoff_threshold=device",
    ]
    config_device = pyconfig.initialize(argv_device)
    self.assertIn("indexer_cutoff_threshold", config_device.tensors_on_device)

    # 2. Verify custom remat policy puts indexer_cutoff_threshold on offload
    argv_offload = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "use_indexer=true",
        "attention_type=mla",
        "q_lora_rank=1536",
        "attention=dot_product",
        "remat_policy=custom",
        "indexer_cutoff_threshold=offload",
    ]
    config_offload = pyconfig.initialize(argv_offload)
    self.assertIn("indexer_cutoff_threshold", config_offload.tensors_to_offload)

    # 3. Verify validation error when use_indexer=False and indexer_cutoff_threshold != 'remat'
    argv_invalid = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "use_indexer=false",
        "indexer_cutoff_threshold=device",
    ]
    with self.assertRaises(ValueError):
      pyconfig.initialize(argv_invalid)

  def test_sliced_mla_proj_disallows_non_qwix_quantization(self):
    """Tests that use_sliced_mla_proj=True is incompatible with module-level (non-Qwix) quantization."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "use_sliced_mla_proj=true",
        "quantization=int8",
    ]
    with self.assertRaisesRegex(pydantic.ValidationError, "only supported via Qwix interception"):
      pyconfig.initialize(argv)

  def test_sliced_mla_proj_allows_qwix_quantization(self):
    """Tests that use_sliced_mla_proj=True is allowed with Qwix interception quantization."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "use_sliced_mla_proj=true",
        "quantization=int8",
        "use_qwix_quantization=true",
    ]
    config = pyconfig.initialize(argv)
    self.assertTrue(config.use_sliced_mla_proj)
    self.assertTrue(config.use_qwix_quantization)

  def test_sliced_mla_proj_disallows_qwix_quantization_with_batch_split(self):
    """Tests that use_sliced_mla_proj=True is incompatible with batch-split Qwix quantization."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "use_sliced_mla_proj=true",
        "quantization=fp8_full",
        "use_qwix_quantization=true",
        "use_batch_split_schedule=true",
    ]
    with self.assertRaisesRegex(pydantic.ValidationError, "only supported via Qwix interception"):
      pyconfig.initialize(argv)

  def test_serve_fp8_weight_accepts_valid_config(self):
    """Tests valid serve_fp8_weight configuration."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "enable_checkpointing=false",
        "quantization=serve_fp8_weight",
        "weight_dtype=float8_e4m3fn",
    ]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.quantization, "serve_fp8_weight")

  def test_serve_fp8_weight_threads_act_calibration_method(self):
    """Tests that serve_fp8_weight picks up act_quantization_calibration_method."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "enable_checkpointing=false",
        "quantization=serve_fp8_weight",
        "weight_dtype=float8_e4m3fn",
        "act_quantization_calibration_method=fixed,-224,224",
    ]
    quant = quantizations.configure_quantization(pyconfig.initialize(argv))
    self.assertIsInstance(quant, quantizations.ServeFp8WeightQuantization)
    self.assertEqual(quant.act_calibration_method, "fixed,-224,224")

  def test_serve_fp8_weight_requires_fp8_weight_dtype(self):
    """Tests that serve_fp8_weight requires an FP8 weight_dtype."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "enable_checkpointing=false",
        "quantization=serve_fp8_weight",
        "weight_dtype=bfloat16",
    ]
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "requires weight_dtype to be an FP8 dtype"):
      pyconfig.initialize(argv)

  def test_serve_fp8_weight_rejects_qwix_quantization(self):
    """Tests that serve_fp8_weight rejects use_qwix_quantization=True."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "enable_checkpointing=false",
        "quantization=serve_fp8_weight",
        "weight_dtype=float8_e4m3fn",
        "use_qwix_quantization=true",
    ]
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "not supported with use_qwix_quantization=True"):
      pyconfig.initialize(argv)

  def test_load_mesh_config_from_package_resources_when_not_on_disk(self):
    """Tests custom_mesh_and_rule falls back to pkgutil.get_data in archives."""
    # pylint: disable=protected-access
    with unittest.mock.patch("os.path.exists", return_value=False):
      mesh_cfg = types.MaxTextConfig._load_mesh_config_from_yaml("ep-as-dp")
      self.assertEqual(
          mesh_cfg.get("mesh_axes"),
          ["data", "fsdp", "fsdp_transpose", "expert"],
      )
      with self.assertRaisesRegex(ValueError, "Custom mesh config file not found"):
        types.MaxTextConfig._load_mesh_config_from_yaml("nonexistent-rule")

  def _elastic_config(self, *extra_argv, num_live_devices=16):
    """Initializes an elastic config on five 4-device slices, by default with one of them down."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "enable_checkpointing=false", "enable_single_controller=true"]
    all_devices = [unittest.mock.MagicMock(slice_index=i // 4) for i in range(20)]
    with (
        unittest.mock.patch("jax.devices", return_value=all_devices),
        unittest.mock.patch.object(types.elastic_utils, "ensure_elastic_manager_initialized"),
        unittest.mock.patch.object(types.elastic_utils, "live_devices", return_value=all_devices[:num_live_devices]),
    ):
      return pyconfig.initialize(argv + ["elastic_enabled=true", *extra_argv])

  def test_elastic_constant_padded_mode_scaling_and_rounding(self):
    for pdbs, logical, physical in ((5, 100, 112), (4, 80, 80)):
      with self.subTest(per_device_batch_size=pdbs):
        cfg = self._elastic_config("elastic_global_batch_size_mode=constant_padded", f"per_device_batch_size={pdbs}")
        self.assertEqual(
            (
                cfg.logical_global_batch_size_to_load,
                cfg.logical_global_batch_size_to_train_on,
                cfg.global_batch_size_to_load,
                cfg.global_batch_size_to_train_on,
                cfg.eval_per_device_batch_size,
                cfg.global_batch_size_to_load_eval,
            ),
            (logical, logical, physical, physical, pdbs, 16 * pdbs),
        )

    # Plain ceil(3 * 20 / 16) = 4 fails (4 * 16) % 6 == 0; vocab step=3 rounds per_device_batch_size up to 6.
    vocab_cfg = self._elastic_config(
        "elastic_global_batch_size_mode=constant_padded",
        "per_device_batch_size=3",
        "max_target_length=16",
        "num_vocab_tiling=6",
    )
    self.assertEqual((vocab_cfg.logical_global_batch_size_to_train_on, vocab_cfg.per_device_batch_size), (60, 6.0))
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "divisible by the number of vocab tiles"):
      self._elastic_config(
          "elastic_global_batch_size_mode=constant_padded",
          "per_device_batch_size=5",
          "max_target_length=16",
          "num_vocab_tiling=6",
      )

    # Pipeline step=2 from non_batch_devices=4 rounds 5 -> 6 on 16 live devices (stage=4, data=4) and 7 -> 8 on 12 live
    # devices, while keeping eval_per_device_batch_size synchronized.
    pp_flags = (
        "elastic_global_batch_size_mode=constant_padded",
        "per_device_batch_size=4",
        "ici_pipeline_parallelism=4",
        "pipeline_delay_activation_forwarding=true",
        "base_num_decoder_layers=8",
        "eval_interval=5",
    )
    for live_devs, expected_pdbs in ((16, 6.0), (12, 8.0)):
      pp_cfg = self._elastic_config(*pp_flags, num_live_devices=live_devs)
      self.assertEqual(
          (
              pp_cfg.logical_global_batch_size_to_train_on,
              pp_cfg.per_device_batch_size,
              pp_cfg.eval_per_device_batch_size,
              pp_cfg.micro_batch_size_to_train_on,
              pp_cfg.micro_batch_size_to_eval_on,
          ),
          (80, expected_pdbs, expected_pdbs, 96, 96),
      )

    # 20 job devices (stage=4, data=5) with per_device_batch_size=1 and 5 microbatches cannot shard over data=5.
    with self.assertRaisesRegex(
        (ValueError, pydantic.ValidationError), "must be divisible by the activation_batch sharding size"
    ):
      self._elastic_config(
          "elastic_global_batch_size_mode=constant_padded",
          "per_device_batch_size=1",
          "ici_pipeline_parallelism=4",
          "num_pipeline_microbatches=5",
          "base_num_decoder_layers=8",
      )

  def test_elastic_constant_padded_mode_default_no_live_slice_and_aot_compile(self):
    default_cfg = self._elastic_config("per_device_batch_size=5")
    self.assertEqual(
        (
            default_cfg.elastic_global_batch_size_mode,
            default_cfg.logical_global_batch_size_to_load,
            default_cfg.global_batch_size_to_load,
        ),
        (types.ElasticGlobalBatchSizeMode.SCALED, None, 80),
    )
    with self.assertRaises(pydantic.ValidationError):
      self._elastic_config("elastic_global_batch_size_mode=invalid_mode")

    no_live = self._elastic_config(
        "elastic_global_batch_size_mode=constant_padded", "per_device_batch_size=5", num_live_devices=0
    )
    self.assertEqual((no_live.logical_global_batch_size_to_load, no_live.per_device_batch_size), (100, 5))

    for aot_flags in (
        ("compile_topology=v5e-32", "compile_topology_num_slices=2"),
        ("internal_compile=true", "internal_compile_num_devices=64"),
    ):
      aot = self._elastic_config("elastic_global_batch_size_mode=constant_padded", "per_device_batch_size=5", *aot_flags)
      self.assertEqual(
          (aot.num_target_devices, aot.logical_global_batch_size_to_load, aot.global_batch_size_to_load),
          (64, 320, 320),
      )

  def test_elastic_constant_padded_mode_rejects_unsupported_configs(self):
    keep = ("elastic_backup_kind=checkpoint", "elastic_global_batch_size_mode=constant_padded")
    for extra, error_regex in (
        ("per_device_batch_size=0.5", "requires an integer `per_device_batch_size >= 1`"),
        ("per_device_batch_size=1.5", "requires an integer `per_device_batch_size >= 1`"),
        ("dataset_type=tfds", "requires synthetic data"),
        ("dataset_type=grain grain_train_files=/x/*", "requires synthetic data"),
        ("subslice_shape=2,2", "not supported with `subslice_shape`"),
        ("expansion_factor_real_data=2", "requires `expansion_factor_real_data` to be -1 or 1"),
        ("expansion_factor_real_data=0.5", "requires `expansion_factor_real_data` to be -1 or 1"),
        (
            "ici_pipeline_parallelism=4 base_num_decoder_layers=8 eval_interval=5"
            " per_device_batch_size=4 eval_per_device_batch_size=6",
            "requires `eval_per_device_batch_size` to match `per_device_batch_size`",
        ),
        (
            "enable_rampup_batch_size=true per_device_batch_size=4 per_device_batch_size_start=1"
            " per_device_batch_size_increment=1 global_rampup_samples=100",
            "not supported with `enable_rampup_batch_size=True`",
        ),
        ("enable_diloco=true", "not supported with DiLoCo"),
        ("use_dpo=true packing=false", "not supported with DPO"),
        # MoE is supported through RoutedMoE's token_mask; only the MoE implementations that bypass it are rejected.
        ("num_experts=4 te_moe_block=true sparse_matmul=true", "not supported with MoE and `te_moe_block=True`"),
        (
            "num_experts=4 use_batch_split_schedule=true decoder_block=deepseek attention_type=mla",
            "not supported with MoE and `use_batch_split_schedule=True`",
        ),
        ("num_experts=4 use_lineage=true", "not supported with MoE and `use_lineage=True`"),
        ("num_experts=4 use_m3_model=true", "not supported with MoE and `use_m3_model=True`"),
        (
            "num_experts=4 decoder_block=kimi_linear packing=false",
            "not supported with MoE and `decoder_block=kimi_linear`",
        ),
        (
            "use_indexer=true attention_type=mla q_lora_rank=1536 attention=dot_product"
            " indexer_topk=8 max_target_length=64 indexer_loss_scaling_factor=0.1",
            "not supported with `indexer_loss_scaling_factor > 0`",
        ),
        ("use_qk_clip=true attention_type=mla decoder_block=deepseek", "not supported with QK-Clip"),
        (
            "use_tunix_gradient_accumulation=true gradient_accumulation_steps=2",
            "not supported with `use_tunix_gradient_accumulation=True` when `gradient_accumulation_steps > 1`",
        ),
    ):
      with self.subTest(extra=extra), self.assertRaisesRegex((ValueError, pydantic.ValidationError), error_regex):
        self._elastic_config(*keep, *extra.split())

    for extra in (
        "use_tunix_gradient_accumulation=true gradient_accumulation_steps=1 per_device_batch_size=5",
        # A plain MoE (RoutedMoE routes with a token_mask) is accepted.
        "num_experts=4 num_experts_per_tok=2 base_mlp_dim=256 base_moe_mlp_dim=256 per_device_batch_size=5",
        "num_experts=4 num_experts_per_tok=2 base_mlp_dim=256 base_moe_mlp_dim=256 sparse_matmul=false capacity_factor=1.25"
        " per_device_batch_size=5",
        "dataset_type=grain grain_train_files=/x/* grain_use_elastic_iterator=true grain_file_type=arrayrecord"
        " packing=false colocated_python_data_input=false per_device_batch_size=5",
        "dataset_type=grain grain_train_files=/x/* grain_use_elastic_iterator=true grain_file_type=arrayrecord"
        " packing=false colocated_python_data_input=true per_device_batch_size=5",
    ):
      cfg = self._elastic_config(*keep, *extra.split())
      self.assertEqual((cfg.logical_global_batch_size_to_load, cfg.global_batch_size_to_load), (100, 112))

  def test_moe_exclude_padding_from_routing_rejects_unsupported_configs(self):
    """The flag needs routed experts and an MoE implementation that takes RoutedMoE's token_mask."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "enable_checkpointing=false", "moe_exclude_padding_from_routing=true"]
    moe = ["num_experts=4", "num_experts_per_tok=2", "base_moe_mlp_dim=256", "base_mlp_dim=256"]
    for extra, error_regex in (
        (["num_experts=1"], "requires num_experts > 1"),
        (moe + ["decoder_block=kimi_linear", "packing=false"], "not supported with decoder_block=kimi_linear"),
        (moe + ["use_m3_model=true"], "not supported with use_m3_model=True"),
    ):
      with self.subTest(extra=extra), self.assertRaisesRegex((ValueError, pydantic.ValidationError), error_regex):
        pyconfig.initialize(argv + extra)
    self.assertTrue(pyconfig.initialize(argv + moe).moe_exclude_padding_from_routing)

  def test_remat_policy_none_rejects_prefetch_fsdp_weights_and_pipeline_parallelism(self):
    """Tests that remat_policy=none is rejected with prefetch_fsdp_weights=True and pipeline parallelism."""
    base_argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "enable_checkpointing=false",
        "base_num_decoder_layers=2",
        "per_device_batch_size=2",
    ]
    # Baseline: default remat_policy='full' permits FSDP weight prefetch and pipeline parallelism.
    self.assertTrue(pyconfig.initialize([*base_argv, "prefetch_fsdp_weights=true"]).prefetch_fsdp_weights)
    self.assertTrue(
        pyconfig.initialize(
            [*base_argv, "compile_topology=v5e-16", "compile_topology_num_slices=1", "ici_pipeline_parallelism=2"]
        ).using_pipeline_parallelism
    )
    self.assertTrue(
        pyconfig.initialize(
            [*base_argv, "compile_topology=v5e-16", "compile_topology_num_slices=2", "dcn_pipeline_parallelism=2"]
        ).using_pipeline_parallelism
    )

    # Baseline: remat_policy=none succeeds when neither prefetch_fsdp_weights nor pipeline parallelism is enabled.
    for remat_flag in ("remat_policy=none", "remat_policy=NONE"):
      cfg = pyconfig.initialize([*base_argv, remat_flag])
      self.assertIsNone(cfg.remat_policy)
    direct_ok = types.MaxTextConfig(run_name="test", enable_checkpointing=False, remat_policy="none")
    self.assertEqual(direct_ok.remat_policy, types.RematPolicy.NONE)

    # pyconfig.initialize (normalizes 'none'/'NONE' -> None before Pydantic validation).
    for extra_flags, error_regex in (
        (["remat_policy=none", "prefetch_fsdp_weights=true"], "not supported with prefetch_fsdp_weights=True"),
        (["remat_policy=NONE", "prefetch_fsdp_weights=true"], "not supported with prefetch_fsdp_weights=True"),
        (
            ["remat_policy=none", "compile_topology=v5e-16", "ici_pipeline_parallelism=2"],
            "not supported with pipeline parallelism",
        ),
        (
            [
                "remat_policy=none",
                "compile_topology=v5e-16",
                "compile_topology_num_slices=2",
                "dcn_pipeline_parallelism=2",
            ],
            "not supported with pipeline parallelism",
        ),
    ):
      with self.subTest(loader="pyconfig", extra_flags=extra_flags):
        with self.assertRaisesRegex((ValueError, pydantic.ValidationError), error_regex):
          pyconfig.initialize([*base_argv, *extra_flags])

    # Direct MaxTextConfig instantiation (preserves 'none' / RematPolicy.NONE string without pyconfig normalization).
    for remat_val in ("none", types.RematPolicy.NONE, None):
      for bad_kwargs, error_regex in (
          ({"prefetch_fsdp_weights": True}, "not supported with prefetch_fsdp_weights=True"),
          ({"ici_pipeline_parallelism": 2}, "not supported with pipeline parallelism"),
          ({"dcn_pipeline_parallelism": 2}, "not supported with pipeline parallelism"),
      ):
        with self.subTest(loader="direct", remat_val=remat_val, bad_kwargs=bad_kwargs):
          with self.assertRaisesRegex(pydantic.ValidationError, error_regex):
            types.MaxTextConfig(
                run_name="test",
                enable_checkpointing=False,
                remat_policy=remat_val,
                **bad_kwargs,
            )

  def test_plan_constant_global_batch(self):
    """`plan_constant_global_batch` picks the least-padded (B', K') within the per-device bound, else rounds B up."""
    # (per_device_batch_size, gradient_accumulation_steps, job_devices, live_devices, alignment, max) -> (B', K').
    for args, expected in (
        # Legacy (-1): keep K, round B up to cover the logical batch. 4 -> 3 slices of 32 chips.
        ((1, 16, 128, 96, 1, -1), (2, 16)),
        ((8, 1, 128, 96, 1, -1), (11, 1)),
        # Bound == B: keep B, grow K; padding is less than one microbatch.
        ((1, 16, 128, 96, 1, 1), (1, 22)),  # 2048 logical -> 2112 physical (3%), not 3072 (33%).
        ((8, 1, 128, 96, 1, 8), (8, 2)),  # 1024 -> 1536: with K=1 nominal the padding equals the legacy B'=11 case.
        ((1, 8, 256, 224, 1, 1), (1, 10)),  # 8 -> 7 slices: 2048 -> 2240.
        ((1, 8, 256, 160, 1, 1), (1, 13)),  # 8 -> 5 slices: 2048 -> 2080.
        # Bound > B: the search may raise B when that pads less (1024 on 96 devices: B'=11, K'=1 pads 32 rows).
        ((8, 1, 128, 96, 1, 16), (11, 1)),
        # Alignment: B' must be a multiple of 2, even when the bound equals B (96 < 100 rows, so K' = 2).
        ((5, 1, 20, 16, 2, 5), (6, 2)),
        ((4, 1, 20, 16, 2, 4), (4, 2)),
        # Zero padding whenever the logical batch divides by the live microbatch.
        ((1, 16, 20, 16, 1, 1), (1, 20)),
        # The search never lowers B' below B: with B=8, K=2 on 128 -> 96 devices (logical 2048) the bound keeps B'=8,
        # K'=3 (2304 rows) rather than B'=2, K'=11 (2112 rows) so the microbatch shapes and scan count stay fixed.
        ((8, 2, 128, 96, 1, 8), (8, 3)),
    ):
      with self.subTest(args=args):
        self.assertEqual(types.plan_constant_global_batch(*args), expected)
    # Padding ties (B'=1, K'=22 and B'=2, K'=11 both pad 64 rows) go to the smaller B' (less activation memory).
    self.assertEqual(types.plan_constant_global_batch(1, 16, 128, 96, 1, 4), (1, 22))
    # With the bound equal to B (`constant_grad_accum`), B' = B and K' = ceil(logical / (live * B)) exactly.
    for b, k, job, live in ((1, 16, 128, 96), (8, 1, 128, 96), (3, 4, 256, 224), (5, 1, 20, 16)):
      with self.subTest(b=b, k=k, job=job, live=live):
        self.assertEqual(
            types.plan_constant_global_batch(b, k, job, live, 1, b), (b, math.ceil(b * k * job / (live * b)))
        )

  def test_elastic_constant_padded_mode_bounded_per_device_batch(self):
    """With `elastic_max_per_device_batch_size`, the per-device batch stays bounded and accumulation steps grow."""
    keep = ("elastic_global_batch_size_mode=constant_padded",)
    # 20 job devices, 16 live: logical 5 * 20 = 100 is covered by B'=5 on 16 devices with K'=2 (160 physical rows).
    cfg = self._elastic_config(*keep, "per_device_batch_size=5", "elastic_max_per_device_batch_size=5")
    self.assertEqual(
        (
            cfg.per_device_batch_size,
            cfg.gradient_accumulation_steps,
            cfg.logical_global_batch_size_to_train_on,
            cfg.global_batch_size_to_train_on,
            cfg.micro_batch_size_to_train_on,
            cfg.global_batch_size_to_load,
        ),
        (5.0, 2, 100, 160, 80, 160),
    )
    # Nominal K=16 with B=1: logical 320 = 16 live devices * 20 steps, zero padding.
    cfg = self._elastic_config(
        *keep, "per_device_batch_size=1", "gradient_accumulation_steps=16", "elastic_max_per_device_batch_size=1"
    )
    self.assertEqual(
        (cfg.per_device_batch_size, cfg.gradient_accumulation_steps, cfg.global_batch_size_to_train_on), (1.0, 20, 320)
    )
    # A looser bound lets the planner raise B when that pads less: logical 80 = 5 * 16 exactly.
    cfg = self._elastic_config(*keep, "per_device_batch_size=4", "elastic_max_per_device_batch_size=8")
    self.assertEqual((cfg.per_device_batch_size, cfg.gradient_accumulation_steps), (5.0, 1))
    # Pipeline alignment (step 2 on 16 live devices) is honoured without changing B=4; K grows instead.
    cfg = self._elastic_config(
        *keep,
        "per_device_batch_size=4",
        "elastic_max_per_device_batch_size=4",
        "ici_pipeline_parallelism=4",
        "pipeline_delay_activation_forwarding=true",
        "base_num_decoder_layers=8",
        "eval_interval=5",
    )
    self.assertEqual(
        (cfg.per_device_batch_size, cfg.eval_per_device_batch_size, cfg.gradient_accumulation_steps), (4.0, 4.0, 2)
    )
    # With every slice live (or none yet), the bound changes nothing.
    for live in (20, 0):
      cfg = self._elastic_config(
          *keep, "per_device_batch_size=5", "elastic_max_per_device_batch_size=5", num_live_devices=live
      )
      self.assertEqual((cfg.per_device_batch_size, cfg.gradient_accumulation_steps), (5.0, 1))
    # The legacy default is unchanged, and records the nominal per-device batch for the HBM budget check.
    cfg = self._elastic_config(*keep, "per_device_batch_size=5")
    self.assertEqual(
        (cfg.per_device_batch_size, cfg.gradient_accumulation_steps, cfg.elastic_nominal_per_device_batch_size),
        (7.0, 1, 5),
    )
    # When the per-device batch is unchanged there is nothing to fall back to.
    cfg = self._elastic_config(*keep, "per_device_batch_size=5", "elastic_max_per_device_batch_size=5")
    self.assertIsNone(cfg.elastic_nominal_per_device_batch_size)
    # logical 320 on 16 devices: B'=1 K'=20 and B'=2 K'=10 both pad nothing; the tie goes to the smaller B'.
    cfg = self._elastic_config(
        *keep, "per_device_batch_size=1", "gradient_accumulation_steps=16", "elastic_max_per_device_batch_size=2"
    )
    self.assertEqual((cfg.per_device_batch_size, cfg.gradient_accumulation_steps), (1.0, 20))

    with self.assertRaisesRegex(
        (ValueError, pydantic.ValidationError),
        "requires `elastic_max_per_device_batch_size` to be -1 or >= `per_device_batch_size`",
    ):
      self._elastic_config(*keep, "per_device_batch_size=5", "elastic_max_per_device_batch_size=3")
    # Rejected even when no resize is pending (all 20 devices live), since a later resize could change K.
    with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "may change `gradient_accumulation_steps`"):
      self._elastic_config(
          *keep,
          "per_device_batch_size=5",
          "elastic_max_per_device_batch_size=5",
          "use_tunix_gradient_accumulation=true",
          num_live_devices=20,
      )
    # FP8 quantization asserts `gradient_accumulation_steps == 1` at train time, so a bound that may grow the steps on
    # a later resize is rejected up front, even with every device live.
    for quantization in ("fp8", "nanoo_fp8", "fp8_gpu", "te_fp8_delayedscaling"):
      with self.subTest(quantization=quantization):
        with self.assertRaisesRegex(
            (ValueError, pydantic.ValidationError), "may change `gradient_accumulation_steps`.*FP8 quantization"
        ):
          self._elastic_config(
              *keep,
              "per_device_batch_size=5",
              "elastic_max_per_device_batch_size=5",
              f"quantization={quantization}",
              num_live_devices=20,
          )

  def test_elastic_constant_grad_accum_mode(self):
    """`constant_grad_accum` keeps B and grows K to `ceil(GBS / (live * B))`; the per-device bound is implied."""
    keep = ("elastic_global_batch_size_mode=constant_grad_accum",)
    # 20 job devices, 16 live. (B, K) -> (B, ceil(B * 20 * K / (16 * B))) with the physical batch B * 16 * K'.
    for b, k, expected_steps in ((5, 1, 2), (1, 16, 20), (4, 1, 2), (3, 2, 3)):
      with self.subTest(b=b, k=k):
        cfg = self._elastic_config(*keep, f"per_device_batch_size={b}", f"gradient_accumulation_steps={k}")
        self.assertEqual(expected_steps, math.ceil(b * 20 * k / (16 * b)))
        self.assertEqual(
            (
                cfg.per_device_batch_size,
                cfg.gradient_accumulation_steps,
                cfg.logical_global_batch_size_to_train_on,
                cfg.global_batch_size_to_train_on,
                cfg.micro_batch_size_to_train_on,
            ),
            (float(b), expected_steps, b * 20 * k, b * 16 * expected_steps, b * 16),
        )
    # Same plan as `constant_padded` with the bound set to B by hand.
    padded = self._elastic_config(
        "elastic_global_batch_size_mode=constant_padded", "per_device_batch_size=5", "elastic_max_per_device_batch_size=5"
    )
    grad_accum = self._elastic_config(*keep, "per_device_batch_size=5")
    self.assertEqual(
        (padded.per_device_batch_size, padded.gradient_accumulation_steps, padded.global_batch_size_to_load),
        (grad_accum.per_device_batch_size, grad_accum.gradient_accumulation_steps, grad_accum.global_batch_size_to_load),
    )
    # Pipeline alignment (step 2 on 16 live devices) never changes B; K grows instead.
    cfg = self._elastic_config(
        *keep,
        "per_device_batch_size=4",
        "ici_pipeline_parallelism=4",
        "pipeline_delay_activation_forwarding=true",
        "base_num_decoder_layers=8",
        "eval_interval=5",
    )
    self.assertEqual(
        (cfg.per_device_batch_size, cfg.eval_per_device_batch_size, cfg.gradient_accumulation_steps), (4.0, 4.0, 2)
    )
    # With every slice live (or none yet) nothing changes.
    for live in (20, 0):
      cfg = self._elastic_config(*keep, "per_device_batch_size=5", num_live_devices=live)
      self.assertEqual((cfg.per_device_batch_size, cfg.gradient_accumulation_steps), (5.0, 1))
    # An explicit bound is rejected: it is implied by the mode.
    for bound in (5, 8):
      with self.subTest(bound=bound):
        with self.assertRaisesRegex(
            (ValueError, pydantic.ValidationError),
            "`elastic_global_batch_size_mode='constant_grad_accum'`.*`elastic_max_per_device_batch_size` is implied"
            " and must stay -1",
        ):
          self._elastic_config(*keep, "per_device_batch_size=5", f"elastic_max_per_device_batch_size={bound}")
    # The steps may change on a resize, so Tunix gradient accumulation and FP8 are rejected even with every device live.
    with self.assertRaisesRegex(
        (ValueError, pydantic.ValidationError), "constant_grad_accum.*may change `gradient_accumulation_steps`"
    ):
      self._elastic_config(*keep, "per_device_batch_size=5", "use_tunix_gradient_accumulation=true", num_live_devices=20)
    with self.assertRaisesRegex(
        (ValueError, pydantic.ValidationError), "may change `gradient_accumulation_steps`.*FP8 quantization"
    ):
      self._elastic_config(*keep, "per_device_batch_size=5", "quantization=fp8", num_live_devices=20)
    # The shared constant-mode restrictions apply with the mode's own name in the message.
    with self.assertRaisesRegex(
        (ValueError, pydantic.ValidationError),
        "`elastic_global_batch_size_mode='constant_grad_accum'` requires an integer `per_device_batch_size >= 1`",
    ):
      self._elastic_config(*keep, "per_device_batch_size=0.5")

  def test_elastic_simulate_job_devices_plans_without_pathways(self):
    """`elastic_simulate_job_devices` sizes the logical batch for a pretended job on the present devices."""
    argv = ["", _BASE_CONFIG_PATH, "run_name=test", "enable_checkpointing=false"]
    devices = [unittest.mock.MagicMock(slice_index=0) for _ in range(4)]
    with unittest.mock.patch.object(types.jax, "devices", return_value=devices, autospec=True):
      cfg = pyconfig.initialize(
          argv
          + [
              "elastic_global_batch_size_mode=constant_padded",
              "elastic_simulate_job_devices=6",
              "per_device_batch_size=3",
              "elastic_max_per_device_batch_size=3",
          ]
      )
      self.assertEqual(
          (
              cfg.elastic_enabled,
              cfg.logical_global_batch_size_to_train_on,
              cfg.per_device_batch_size,
              cfg.gradient_accumulation_steps,
              cfg.global_batch_size_to_train_on,
          ),
          (False, 18, 3.0, 2, 24),
      )
      legacy = pyconfig.initialize(
          argv
          + [
              "elastic_global_batch_size_mode=constant_padded",
              "elastic_simulate_job_devices=6",
              "per_device_batch_size=3",
          ]
      )
      self.assertEqual((legacy.per_device_batch_size, legacy.global_batch_size_to_train_on), (5.0, 20))
      # `constant_grad_accum` plans the same simulated resize without a bound: B stays 3, K grows to 2.
      grad_accum = pyconfig.initialize(
          argv
          + [
              "elastic_global_batch_size_mode=constant_grad_accum",
              "elastic_simulate_job_devices=6",
              "per_device_batch_size=3",
          ]
      )
      self.assertEqual(
          (
              grad_accum.elastic_enabled,
              grad_accum.logical_global_batch_size_to_train_on,
              grad_accum.per_device_batch_size,
              grad_accum.gradient_accumulation_steps,
              grad_accum.global_batch_size_to_train_on,
          ),
          (False, 18, 3.0, 2, 24),
      )
      with self.assertRaisesRegex(
          (ValueError, pydantic.ValidationError), "`elastic_max_per_device_batch_size` is implied"
      ):
        pyconfig.initialize(
            argv
            + [
                "elastic_global_batch_size_mode=constant_grad_accum",
                "elastic_simulate_job_devices=6",
                "per_device_batch_size=3",
                "elastic_max_per_device_batch_size=3",
            ]
        )
      # The simulation also enables the constant_padded validation.
      with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "not supported with MoE and `te_moe_block"):
        pyconfig.initialize(
            argv
            + [
                "elastic_global_batch_size_mode=constant_padded",
                "elastic_simulate_job_devices=6",
                "num_experts=4",
                "te_moe_block=true",
                "sparse_matmul=true",
            ]
        )
      # Negative values and a pretended job smaller than the present devices are rejected.
      with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "`elastic_simulate_job_devices` must be >= 0"):
        pyconfig.initialize(argv + ["elastic_global_batch_size_mode=constant_padded", "elastic_simulate_job_devices=-1"])
      with self.assertRaisesRegex(
          (ValueError, pydantic.ValidationError), r"\(2\) must be >= the number of present devices \(4\)"
      ):
        pyconfig.initialize(argv + ["elastic_global_batch_size_mode=constant_padded", "elastic_simulate_job_devices=2"])


class Qwen3StackConfigValidationTest(absltest.TestCase):
  """Cross-field validators of the Qwen3-235B v7x training knobs."""

  _ERRORS = (ValueError, pydantic.ValidationError)

  def _initialize(self, *overrides):
    return pyconfig.initialize(["", _BASE_CONFIG_PATH, "run_name=test", "enable_checkpointing=false", *overrides])

  def test_sa_context_split_requires_custom_remat(self):
    for overrides in (["sa_context_split_device_heads=4"], ["sa_context_offload_fp8=true"]):
      with self.subTest(overrides=overrides):
        with self.assertRaisesRegex(self._ERRORS, "require remat_policy='custom'"):
          self._initialize("remat_policy=full", *overrides)

  def test_sa_context_split_registers_context_tensors(self):
    config = self._initialize("remat_policy=custom", "sa_context_split_device_heads=4", "sa_context_offload_fp8=true")
    self.assertIn("context_device", config.tensors_on_device)
    self.assertIn("context_offload", config.tensors_to_offload)
    # Without the context split, the custom policy keeps its plain tensor lists.
    config = self._initialize("remat_policy=custom")
    self.assertNotIn("context_device", config.tensors_on_device)
    self.assertNotIn("context_offload", config.tensors_to_offload)

  def test_sa_num_stacked_q_heads_validation(self):
    with self.assertRaisesRegex(self._ERRORS, "sa_num_stacked_q_heads must be >= 1"):
      self._initialize("sa_num_stacked_q_heads=0")
    # 16 query heads over 4 KV heads: a GQA group of 4 queries per KV head.
    gqa = ["base_num_query_heads=16", "base_num_kv_heads=4"]
    with self.assertRaisesRegex(self._ERRORS, "must divide"):
      self._initialize(*gqa, "sa_num_stacked_q_heads=3")
    self.assertEqual(self._initialize(*gqa, "sa_num_stacked_q_heads=2").sa_num_stacked_q_heads, 2)

  def test_qwix_disable_channelwise_axes_requirements(self):
    with self.assertRaisesRegex(self._ERRORS, "requires quantization='fp8_full' and use_qwix_quantization=True"):
      self._initialize("qwix_disable_channelwise_axes=true")
    fp8_full = ["quantization=fp8_full", "use_qwix_quantization=true"]
    with self.assertRaisesRegex(self._ERRORS, "requires fixed weight_quantization_calibration_method"):
      self._initialize(*fp8_full, "weight_quantization_calibration_method=absmax", "qwix_disable_channelwise_axes=true")
    config = self._initialize(
        *fp8_full, "weight_quantization_calibration_method=fixed,-224,224", "qwix_disable_channelwise_axes=true"
    )
    self.assertTrue(config.qwix_disable_channelwise_axes)

  def test_moe_fused_gmm3d_wgrad_schedule_mode_bounds(self):
    for mode in (-1, 18):
      with self.subTest(mode=mode):
        with self.assertRaises(pydantic.ValidationError):
          self._initialize(f"moe_fused_gmm3d_wgrad_schedule_mode={mode}")
    self.assertEqual(self._initialize("moe_fused_gmm3d_wgrad_schedule_mode=17").moe_fused_gmm3d_wgrad_schedule_mode, 17)

  def test_moe_tc_routing_offload_free_prerequisites(self):
    with self.assertRaisesRegex(self._ERRORS, "requires moe_packed_routing_sort=True"):
      self._initialize("moe_tc_routing_offload_free=true")
    config = self._initialize("moe_packed_routing_sort=true", "moe_tc_routing_offload_free=true")
    self.assertTrue(config.moe_tc_routing_offload_free)
    # The ring-of-experts ragged sort needs expert parallelism > 1, so check that rule on the validator directly.
    ring_config = unittest.mock.Mock(
        moe_tc_routing_offload_free=True, moe_packed_routing_sort=True, use_ring_of_experts=True, use_ragged_sort=True
    )
    with self.assertRaisesRegex(ValueError, "does not support use_ring_of_experts=True"):
      types.MaxTextConfig.validate_moe_tc_routing_offload_free(ring_config)

  def test_moe_use_fused_gmm3d_prerequisites(self):
    # A fully-MoE model: base_moe_mlp_dim must match base_mlp_dim (7168 in base.yml).
    moe = ["num_experts=8", "num_experts_per_tok=2", "base_moe_mlp_dim=7168"]
    with self.assertRaisesRegex(self._ERRORS, r"moe_use_fused_gmm3d=True requires: .*use_tokamax_gmm=True"):
      self._initialize(*moe, "moe_use_fused_gmm3d=true")
    config = self._initialize(
        *moe,
        "sparse_matmul=true",
        "megablox=true",
        "use_tokamax_gmm=true",
        "use_gmm_v2=true",
        "moe_flat_fsdp_weights=true",
        "capacity_factor=-1",
        "moe_use_fused_gmm3d=true",
    )
    self.assertTrue(config.moe_use_fused_gmm3d)

  def test_sa_diag_band_skip_prerequisites(self):
    with self.assertRaisesRegex(self._ERRORS, "sa_diag_band_skip requires use_tokamax_splash=True"):
      self._initialize("sa_diag_band_skip=true")
    splash = ["attention=flash", "use_tokamax_splash=true", "sa_qk_diag_skip=true", "sa_diag_band_skip=true"]
    with self.assertRaisesRegex(self._ERRORS, "only takes effect with dq_reduction_steps=1"):
      self._initialize(*splash, "dq_reduction_steps=0")
    self.assertTrue(self._initialize(*splash, "dq_reduction_steps=1").sa_diag_band_skip)

  def test_sa_ignore_segment_ids_accepts_single_segment_synthetic_data(self):
    base = ["attention=flash", "dataset_type=synthetic", "sa_ignore_segment_ids=true"]
    self.assertTrue(self._initialize(*base, "packing=true", "max_segments_per_seq=1").sa_ignore_segment_ids)
    with self.assertRaisesRegex(self._ERRORS, "sa_ignore_segment_ids requires"):
      self._initialize(*base, "packing=true", "max_segments_per_seq=2")

  def test_rl_config_sa_context_split_requires_custom_remat(self):
    # Tiny-model overrides that let RLConfig validate quickly on CPU.
    overrides = {
        "enable_checkpointing": False,
        "base_num_decoder_layers": 1,
        "attention": "dot_product",
        "max_target_length": 8,
        "base_emb_dim": 128,
        "base_num_query_heads": 2,
        "base_num_kv_heads": 2,
        "base_mlp_dim": 256,
        "max_prefill_predict_length": 4,
        "model_name": "llama3.1-8b",
        "override_model_config": True,
        "weight_dtype": "bfloat16",
        "tokenizer_path": "meta-llama/Llama-3.1-8B",
    }

    def rl_config(*cli_args):
      return pyconfig.initialize_pydantic(
          [sys.argv[0], get_test_config_path("post_train/rl.yml"), *cli_args],
          config_class=types.RLConfig,
          run_name="rl_sa_context_split_test",
          **overrides,
      )

    with self.assertRaisesRegex(self._ERRORS, "require remat_policy='custom'"):
      rl_config("remat_policy=full", "sa_context_split_device_heads=2")
    config = rl_config("remat_policy=custom", "sa_context_split_device_heads=2")
    self.assertIn("context_device", config.tensors_on_device)
    self.assertIn("context_offload", config.tensors_to_offload)

    config_qkv_dev = rl_config("remat_policy=custom", "ulysses_qkv=device")
    self.assertIn("ulysses_qkv", config_qkv_dev.tensors_on_device)
    self.assertNotIn("ulysses_qkv", config_qkv_dev.tensors_to_offload)

    config_qkv_off = rl_config("remat_policy=custom", "ulysses_qkv=offload")
    self.assertNotIn("ulysses_qkv", config_qkv_off.tensors_on_device)
    self.assertIn("ulysses_qkv", config_qkv_off.tensors_to_offload)

    self.assertEqual(rl_config().ulysses_pin_row_major_layout, "auto")
    for mode in ("auto", "always", "never"):
      self.assertEqual(rl_config(f"ulysses_pin_row_major_layout={mode}").ulysses_pin_row_major_layout, mode)
    with self.assertRaises(pydantic.ValidationError):
      rl_config("ulysses_pin_row_major_layout=invalid")


class MMapDatasetConfigTest(absltest.TestCase):
  """Tests for mmap-specific configuration defaults and accepted values."""

  def test_default_is_25_preserving_prior_behavior(self):
    config = types.MMapDataset()
    self.assertEqual(config.packing_max_segments_per_sample, 25)

  def test_custom_value_is_accepted(self):
    config = types.MMapDataset(packing_max_segments_per_sample=64)
    self.assertEqual(config.packing_max_segments_per_sample, 64)

  def test_zero_disables_merging_round_trip(self):
    config = types.MMapDataset(packing_max_segments_per_sample=0)
    self.assertEqual(config.packing_max_segments_per_sample, 0)


class GrainReadOptionsConfigTest(absltest.TestCase):
  """Tests pyconfig validation for Grain ReadOptions."""

  def _initialize(self, *overrides):
    return pyconfig.initialize(
        ["", _BASE_CONFIG_PATH, "run_name=test", "enable_checkpointing=false", "dataset_type=grain", *overrides]
    )

  def test_read_options_bounds(self):
    cfg = self._initialize(
        "grain_train_files=/tmp/train*",
        "grain_num_threads=0",
        "grain_prefetch_buffer_size=0",
        "grain_num_threads_eval=0",
        "grain_prefetch_buffer_size_eval=0",
    )
    self.assertEqual((cfg.grain_num_threads, cfg.grain_prefetch_buffer_size), (0, 0))
    with self.assertRaises(pydantic.ValidationError):
      self._initialize("grain_train_files=/tmp/train*", "grain_num_threads=-1")
    for file_type in ("parquet", "tfrecord"):
      with self.assertRaisesRegex((ValueError, pydantic.ValidationError), "grain_num_threads.*must be > 0"):
        self._initialize(f"grain_file_type={file_type}", "grain_train_files=/tmp/train*", "grain_num_threads=0")


if __name__ == "__main__":
  absltest.main()
