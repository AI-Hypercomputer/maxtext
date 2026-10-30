# Copyright 2026 Google LLC
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

"""Unit tests for Workstream 1: `TrainingConfig`, Model Garden presets, CLI parser, and `from_pyconfig`."""

import json
import os
from unittest import mock

from absl.testing import absltest
import jax
from pydantic import ValidationError

from maxtext.configs import pyconfig
from maxtext.m3 import (
    DataPipelineConfig,
    MeshConfig,
    ModelConfig,
    TrainingConfig,
    TransformConfig,
)
from maxtext.m3.configs import (
    ConfigMigrationError,
    M3ArgumentParser,
    from_pyconfig,
)
from maxtext.m3.models.qwen3.qwen3_config import DeepSeekV3Config, Qwen3Config
from tests.utils.test_helpers import get_test_config_path


class TrainingConfigTest(absltest.TestCase):
  """Verifies pure value-object invariants, schema validation, CLI reflection, and `from_pyconfig`."""

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path

  def test_zero_jax_devices_side_effects_on_instantiation(self):
    with mock.patch.object(jax, "devices", side_effect=AssertionError("jax.devices() must not be called")):
      arch = ModelConfig.from_preset("qwen3-0.6b")
      cfg = TrainingConfig(
          steps=20,
          per_device_batch_size=2.0,
          scan_layers=False,
          profiler="xplane",
          hf_access_token="hf_secret_12345",
      )
    self.assertIsInstance(arch, ModelConfig)
    self.assertIsNotNone(MeshConfig)
    self.assertIsNotNone(TransformConfig)
    self.assertIsNotNone(DataPipelineConfig)
    self.assertEqual(arch.hidden_size, 1024)
    self.assertEqual(arch.emb_dim, 1024)
    self.assertFalse(cfg.transforms.scan_layers)
    self.assertEqual(cfg.profiling.profiler, "xplane")

    dumped = cfg.to_json()
    self.assertNotIn("hf_secret_12345", dumped)
    self.assertIn("**********", dumped)

  def test_forbids_unknown_fields_and_is_frozen(self):
    with self.assertRaises(ValidationError):
      TrainingConfig(nonexistent_knob=123)  # type: ignore[call-arg]
    with self.assertRaises(ValidationError):
      Qwen3Config(unknown_arch_field=7)  # type: ignore[call-arg]

    cfg = TrainingConfig(steps=10)
    with self.assertRaises(ValidationError):
      cfg.steps = 99  # type: ignore[misc]

  def test_from_hf_config_and_deepseek_preset(self):
    hf_json = os.path.join(self.temp_dir, "config.json")
    with open(hf_json, "w", encoding="utf-8") as f:
      json.dump(
          {
              "architectures": ["Qwen3ForCausalLM"],
              "vocab_size": 512,
              "hidden_size": 128,
              "intermediate_size": 256,
              "num_hidden_layers": 4,
              "num_attention_heads": 8,
              "num_key_value_heads": 4,
              "rms_norm_eps": 1e-6,
          },
          f,
      )
    loaded = Qwen3Config.from_hf_config(hf_json, tie_word_embeddings=False)
    self.assertEqual(loaded.vocab_size, 512)
    self.assertEqual(loaded.head_dim, 16)
    self.assertFalse(loaded.logits_via_embedding)

    ds_cfg = ModelConfig.from_preset("deepseek3-mini")
    self.assertIsInstance(ds_cfg, DeepSeekV3Config)
    self.assertEqual(ds_cfg.n_routed_experts, 8)

  def test_m3_argument_parser_cli_and_json_roundtrip(self):
    out_json = os.path.join(self.temp_dir, "provenance.json")
    parser = M3ArgumentParser()
    arch, train_cfg = parser.parse_args(
        [
            "--preset",
            "qwen3-mini",
            "--steps",
            "12",
            "--learning_rate",
            "1e-3",
            "--transforms.scan_layers",
            "false",
            "--mesh.ici",
            '{"fsdp": -1}',
            "--model.vocab_size",
            "320",
            "--save_config_json",
            out_json,
        ]
    )
    self.assertEqual(arch.vocab_size, 320)
    self.assertEqual(train_cfg.steps, 12)
    self.assertAlmostEqual(train_cfg.learning_rate, 1e-3)
    self.assertFalse(train_cfg.transforms.scan_layers)
    self.assertTrue(os.path.isfile(out_json))

  def test_from_pyconfig_translates_valid_flags_and_rejects_unsupported(self):
    py_cfg = pyconfig.initialize_pydantic(
        [
            "",
            get_test_config_path(),
            "model_name=qwen3-0.6b",
            "use_m3_model=true",
            "override_model_config=true",
            "base_emb_dim=64",
            "base_mlp_dim=128",
            "base_num_decoder_layers=2",
            "base_num_query_heads=4",
            "base_num_kv_heads=2",
            "head_dim=16",
            "vocab_size=256",
            "steps=5",
            "num_vocab_tiling=4",
            "scan_layers=true",
            "skip_jax_distributed_system=true",
        ]
    )
    arch, train_cfg = from_pyconfig(py_cfg)
    self.assertEqual(arch.hidden_size, 64)
    self.assertEqual(arch.num_hidden_layers, 2)
    self.assertEqual(train_cfg.steps, 5)
    self.assertEqual(train_cfg.num_vocab_tiles, 4)
    self.assertTrue(train_cfg.transforms.scan_layers)

    bad_cfg = pyconfig.initialize_pydantic(
        [
            "",
            get_test_config_path(),
            "model_name=qwen3-0.6b",
            "use_m3_model=true",
            "enable_diloco=true",
            "skip_jax_distributed_system=true",
        ]
    )
    with self.assertRaisesRegex(ConfigMigrationError, "enable_diloco"):
      from_pyconfig(bad_cfg)


if __name__ == "__main__":
  absltest.main()
