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

"""Unit tests for the zero-YAML `m3` CLI `ArgumentParser`."""

import sys
from unittest import mock

from absl.testing import absltest
from maxtext.configs import types as config_types
from maxtext.m3.configs.cli_parser import ArgumentParser
from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5ModelConfig
from maxtext.m3.models.granitemoe.granitemoe_config import GraniteMoeModelConfig
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config
from maxtext.m3.models.qwen3_8_flash_next.qwen3_8_config import Qwen3_8FlashNextConfig
from maxtext.m3.models.qwen3_vl.qwen3_vl_config import Qwen3VLConfig
from pydantic import ValidationError


class CliParserTest(absltest.TestCase):
  """Tests `ArgumentParser` flag parsing, scoping, type handling, and error messages."""

  def setUp(self):
    super().setUp()
    self.parser = ArgumentParser()

  def test_model_name_lookup_and_hyphen_normalization(self):
    model_cfg, train_cfg = self.parser.parse_args(
        [
            "train.py",
            "--model-name",
            "qwen3-0.6b",
            "--steps",
            "42",
            "--max-target-length=8192",
            "--eval_interval",
            "-1",
            "--per_device_batch_size=0.5",
        ]
    )
    self.assertIsInstance(model_cfg, Qwen3Config)
    self.assertEqual(model_cfg, Qwen3Config.qwen3_0_6b())
    self.assertEqual(train_cfg.steps, 42)
    self.assertEqual(train_cfg.max_target_length, 8192)
    self.assertEqual(train_cfg.eval_interval, -1)
    self.assertEqual(train_cfg.per_device_batch_size, 0.5)

    # `-Instruct` suffix normalizes cleanly.
    inst_cfg, _ = self.parser.parse_args(["--model_name=qwen3-0.6b-Instruct"])
    self.assertEqual(inst_cfg, model_cfg)

    # Reads `sys.argv[1:]` when `argv=None`.
    with mock.patch.object(sys, "argv", ["prog.py", "--model_name=granitemoe", "--steps=5"]):
      gm_cfg, gm_train = self.parser.parse_args()
    self.assertIsInstance(gm_cfg, GraniteMoeModelConfig)
    self.assertEqual(gm_train.steps, 5)

  def test_string_fields_preserve_raw_strings_and_optional_none(self):
    _, train_cfg = self.parser.parse_args(
        [
            "/usr/local/bin/train",
            "--model_name=qwen3-0.6b",
            "--run_name=20261124",
            "--base_output_directory=",
            "--metrics.metrics_file=none",
            "--data.dataset_name=nan",
            "--data.hf_access_token=hf_super_secret",
        ]
    )
    self.assertEqual(train_cfg.run_name, "20261124")
    self.assertEqual(train_cfg.base_output_directory, "")
    self.assertIsNone(train_cfg.metrics.metrics_file)
    self.assertEqual(train_cfg.data.dataset_name, "nan")
    self.assertEqual(train_cfg.data.hf_access_token.get_secret_value(), "hf_super_secret")

    # Values ending in `.yml` remain valid string values for flags.
    _, eq_cfg = self.parser.parse_args(["--model_name=qwen3-0.6b", "--run_name=foo.yml"])
    _, sp_cfg = self.parser.parse_args(["--model_name", "qwen3-0.6b", "--run_name", "foo.yml"])
    self.assertEqual(eq_cfg.run_name, "foo.yml")
    self.assertEqual(sp_cfg.run_name, "foo.yml")

  def test_nested_subconfigs_model_overrides_and_json_containers(self):
    model_cfg, train_cfg = self.parser.parse_args(
        [
            "--model_name=qwen3-vl-2b",
            "--model.num_hidden_layers=2",
            "--model.mrope_section=[24, 20, 20]",
            "--model.rope_theta=1e6",
            "--transforms.scan_layers=false",
            "--transforms.remat_policy=none",
            "--transforms.quantization=",
            '--transforms.remat_save_names=["attn", "mlp"]',
            "--transforms.named_scopes",
            '--mesh.ici={"fsdp": -1, "tensor": 2}',
            "--optimizer.learning_rate=1e-3",
            "--checkpoint.max_num_checkpoints_to_keep=none",
            "--profiling.profiler=xplane",
        ]
    )
    self.assertIsInstance(model_cfg, Qwen3VLConfig)
    self.assertEqual(model_cfg.num_hidden_layers, 2)
    self.assertEqual(model_cfg.mrope_section, (24, 20, 20))
    self.assertEqual(model_cfg.rope_theta, 1_000_000.0)
    self.assertFalse(train_cfg.transforms.scan_layers)
    self.assertEqual(train_cfg.transforms.remat_policy, config_types.RematPolicy.NONE)
    self.assertEqual(train_cfg.transforms.quantization, config_types.QuantizationType.NONE)
    self.assertEqual(train_cfg.transforms.remat_save_names, ("attn", "mlp"))
    self.assertTrue(train_cfg.transforms.named_scopes)
    self.assertEqual(train_cfg.mesh.ici, {"fsdp": -1, "tensor": 2})
    self.assertEqual(train_cfg.optimizer.learning_rate, 1e-3)
    self.assertIsNone(train_cfg.checkpoint.max_num_checkpoints_to_keep)
    self.assertEqual(train_cfg.profiling.profiler, config_types.ProfilerType.XPLANE)

    # Full sub-config object passed as JSON merges cleanly with later dotted flags.
    _, obj_train = self.parser.parse_args(
        [
            "--model_name=glm5.3-flash",
            '--transforms={"scan_layers": false, "remat_policy": "none"}',
            "--transforms.named_scopes=false",
        ]
    )
    self.assertFalse(obj_train.transforms.scan_layers)
    self.assertEqual(obj_train.transforms.remat_policy, config_types.RematPolicy.NONE)
    self.assertFalse(obj_train.transforms.named_scopes)

    # Spot-check remaining registered families with tuple/list model overrides.
    glm_cfg, _ = self.parser.parse_args(
        [
            "--model_name=glm5.3-flash",
            "--model.num_hidden_layers=2",
            '--model.layer_types=["linear_attention", "full_attention"]',
        ]
    )
    self.assertIsInstance(glm_cfg, Glm5ModelConfig)
    self.assertEqual(glm_cfg.layer_types, ("linear_attention", "full_attention"))

    q38_cfg, _ = self.parser.parse_args(["--model_name=qwen3.8-flash-next", "--model.num_hidden_layers=8"])
    self.assertIsInstance(q38_cfg, Qwen3_8FlashNextConfig)
    self.assertEqual(q38_cfg.num_hidden_layers, 8)

  def test_shared_dtype_flags_and_scoped_override_precedence(self):
    model_cfg, train_cfg = self.parser.parse_args(
        [
            "--model_name=qwen3-0.6b",
            "--compute_dtype=float32",
            "--param_dtype=bfloat16",
        ]
    )
    self.assertEqual(model_cfg.compute_dtype, config_types.DType.FLOAT32)
    self.assertEqual(train_cfg.compute_dtype, config_types.DType.FLOAT32)
    self.assertEqual(model_cfg.param_dtype, config_types.DType.BFLOAT16)
    self.assertEqual(train_cfg.param_dtype, config_types.DType.BFLOAT16)

    # `--model.compute_dtype` after `--compute_dtype` only overrides `ModelConfig`.
    m_split, t_split = self.parser.parse_args(
        [
            "--model_name=qwen3-0.6b",
            "--compute_dtype=float32",
            "--model.compute_dtype=bfloat16",
        ]
    )
    self.assertEqual(t_split.compute_dtype, config_types.DType.FLOAT32)
    self.assertEqual(m_split.compute_dtype, config_types.DType.BFLOAT16)

  def test_errors_and_validation_failures(self):
    with self.assertRaisesRegex(ValueError, "Missing required `--model_name"):
      self.parser.parse_args(["--steps=10"])
    with self.assertRaisesRegex(ValueError, "requires a non-empty value"):
      self.parser.parse_args(["--model_name"])
    with self.assertRaisesRegex(ValueError, "requires a non-empty value"):
      self.parser.parse_args(["--model_name="])
    with self.assertRaises(KeyError):
      self.parser.parse_args(["--model_name=does-not-exist"])
    with self.assertRaisesRegex(ValueError, "Expected `--flag`"):
      self.parser.parse_args(["bogus_arg", "--model_name=qwen3-0.6b"])
    with self.assertRaisesRegex(ValueError, "Expected `--flag`"):
      self.parser.parse_args(["base.yml", "--model_name=qwen3-0.6b"])
    with self.assertRaisesRegex(ValueError, "Empty flag name"):
      self.parser.parse_args(["--=foo"])
    with self.assertRaisesRegex(ValueError, "requires a value"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--steps"])

    # Unscoped nested leaf flags fail fast with helpful dotted suggestions.
    with self.assertRaisesRegex(ValueError, "--transforms.scan_layers"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--scan_layers=false"])
    with self.assertRaisesRegex(ValueError, "--model.num_hidden_layers"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--num_hidden_layers=4"])
    with self.assertRaisesRegex(ValueError, "Unrecognized CLI flag"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--totally_unknown=1"])

    # Bad dotted flags and unknown nested fields.
    with self.assertRaisesRegex(ValueError, "Unknown dotted CLI argument"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--bogus.field=1"])
    with self.assertRaisesRegex(ValueError, "Unknown dotted CLI argument"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--a.b.c=1"])
    with self.assertRaisesRegex(ValueError, "Unknown field 'bogus' on `transforms`"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--transforms.bogus=1"])
    with self.assertRaisesRegex(ValueError, "Unknown field 'bogus' on `model`"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--model.bogus=1"])
    with self.assertRaisesRegex(ValueError, "Invalid JSON value"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--mesh.ici={bad}"])
    with self.assertRaisesRegex(ValueError, "expects a JSON object"):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--transforms=bad"])

    # Strict Pydantic & cross-config validation.
    with self.assertRaises(ValidationError):
      self.parser.parse_args(["--model_name=qwen3-0.6b", "--steps=true"])
    with self.assertRaises(ValidationError):
      self.parser.parse_args(["--model_name=qwen3-0.6b", '--mesh.ici={"fsdp": true}'])
    with self.assertRaisesRegex(ValueError, "num_vocab_tiling"):
      self.parser.parse_args(
          [
              "--model_name=qwen3-0.6b",
              "--per_device_batch_size=1",
              "--max_target_length=10",
              "--num_vocab_tiling=4",
          ]
      )


if __name__ == "__main__":
  absltest.main()
