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

import os
import unittest.mock

from absl.testing import absltest
from maxtext.configs import pyconfig
from maxtext.configs import types
from maxtext.utils import globals as maxtext_globals
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
    argv = ["", _BASE_CONFIG_PATH, "model_name=llama2-7b", "run_name=test"]
    config = pyconfig.initialize(argv)
    self.assertEqual(config.base_emb_dim, 4096)  # From llama2-7b.yml
    self.assertEqual(config.base_num_decoder_layers, 32)  # From llama2-7b.yml
    self.assertEqual(config.decoder_block, types.DecoderBlockType.LLAMA2)  # from llama2-7b.yml
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

  def test_validation_error(self):
    """Tests that a validation error is raised for invalid config."""
    # A negative number for steps should trigger a ValidationError in the pydantic model.
    argv = ["", _BASE_CONFIG_PATH, "steps=-5"]
    with self.assertRaises(pydantic.ValidationError):
      pyconfig.initialize(argv)

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
        (["base_num_kv_heads=1"], [], "MQA"),
        (["base_num_query_heads=18"], [], "requires num_query_heads"),
        (["base_num_kv_heads=10"], [], "requires num_kv_heads"),
        (["attention_type=mla"], [], "global causal attention"),
        (["attention_type=local_sliding", "sliding_window_size=128"], [], "global causal attention"),
        (["attention_type=chunk", "chunk_attn_window_size=128"], [], "global causal attention"),
        (["attention_type=full"], [], "global causal attention"),
        (["attention_type=compressed"], [], "global causal attention"),
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
        (["context_parallel_load_balance=True"], ["context_parallel_load_balance=False"], "load_balance"),
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

  def test_default_attention_remains_global(self):
    config = pyconfig.initialize(["", _BASE_CONFIG_PATH, "run_name=test", "steps=1"])

    self.assertEqual(config.attention_type, "global")
    self.assertEqual(config.training_objective, "causal_lm")
    self.assertEqual(config.block_diffusion_mask_id, -1)

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
        ({"use_sft": True}, "pre-training only"),
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
        "model_name=llama3-8b",
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

  def test_sliced_mla_proj_disallows_quantization(self):
    """Tests that use_sliced_mla_proj=True is incompatible with quantization."""
    argv = [
        "",
        _BASE_CONFIG_PATH,
        "run_name=test",
        "use_sliced_mla_proj=true",
        "quantization=int8",
    ]
    with self.assertRaises(pydantic.ValidationError):
      pyconfig.initialize(argv)


if __name__ == "__main__":
  absltest.main()
