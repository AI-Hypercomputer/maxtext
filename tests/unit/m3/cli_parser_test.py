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

"""Unit tests for the m3 CLI parser."""

import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from maxtext.configs import types as config_types
from maxtext.m3.configs import cli_parser
from maxtext.m3.configs.cli_parser import parse_args
from maxtext.m3.configs.training_config import TrainingConfig, TransformConfig
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config

_QWEN3 = "--model_name=qwen3-0.6b"


class _OptionalSubconfigTrainingConfig(TrainingConfig):
  """`TrainingConfig` with an optional sub-config, a shape no real field has yet."""

  extra_transforms: TransformConfig | None = None


class CliParserTest(parameterized.TestCase):
  """Tests `parse_args` flag parsing, scoping, type handling, and error messages."""

  def test_flag_forms_and_hyphen_normalization(self):
    model_cfg, train_cfg = parse_args(
        [
            "--model-name",
            "qwen3-0.6b",
            "--steps",
            "42",
            "--max-target-length=8192",
            "--eval_interval",
            "none",
            "--data.grain_worker_count",
            "-1",  # A negative value after a space is a value, not a flag.
            "--base_output_directory=",
        ]
    )
    self.assertEqual(model_cfg, Qwen3Config.qwen3_0_6b())
    self.assertEqual((train_cfg.steps, train_cfg.max_target_length), (42, 8192))
    self.assertIsNone(train_cfg.eval_interval)
    self.assertEqual(train_cfg.data.grain_worker_count, -1)
    self.assertEqual(train_cfg.base_output_directory, "")

    with mock.patch.object(sys, "argv", ["prog.py", _QWEN3, "--steps=5"]):
      self.assertEqual(parse_args()[1].steps, 5)  # Reads `sys.argv[1:]` when `argv=None`.

  def test_none_only_applies_to_optional_fields(self):
    _, train_cfg = parse_args([_QWEN3, "--metrics.metrics_file=none", "--data.hf_access_token=none"])
    self.assertEqual(train_cfg.metrics.metrics_file, "none")  # `str` keeps "none" literally.
    self.assertIsNone(train_cfg.data.hf_access_token)  # `SecretStr | None` maps it to None.

  def test_nested_fields_and_json_values(self):
    model_cfg, train_cfg = parse_args(
        [
            "--model_name=qwen3-vl-2b",
            "--model.num_hidden_layers=2",
            "--model.mrope_section=[24, 20, 20]",
            "--transforms.scan_layers=false",
            "--transforms.quantization=",
            '--transforms.remat_save_names=["attn", "mlp"]',
            "--transforms.named_scopes",
            "--mesh.ici.tensor=2",
            "--optimizer.learning_rate=1e-3",
            "--checkpoint.max_num_checkpoints_to_keep=none",
        ]
    )
    self.assertEqual((model_cfg.num_hidden_layers, model_cfg.mrope_section), (2, (24, 20, 20)))
    transforms = train_cfg.transforms
    self.assertFalse(transforms.scan_layers)
    self.assertEqual(transforms.quantization, config_types.QuantizationType.NONE)
    self.assertEqual(transforms.remat_save_names, ("attn", "mlp"))
    self.assertTrue(transforms.named_scopes)
    self.assertEqual((train_cfg.mesh.ici.fsdp, train_cfg.mesh.ici.tensor), (-1, 2))
    self.assertEqual(train_cfg.optimizer.learning_rate, 1e-3)
    self.assertIsNone(train_cfg.checkpoint.max_num_checkpoints_to_keep)

  def test_optional_subconfig_takes_dotted_flags_json_and_none(self):
    with mock.patch.object(cli_parser, "TrainingConfig", _OptionalSubconfigTrainingConfig):
      _, dotted = parse_args([_QWEN3, "--extra_transforms.scan_layers=false"])
      _, json_obj = parse_args([_QWEN3, '--extra_transforms={"named_scopes": false}'])
      _, unset = parse_args([_QWEN3, "--extra_transforms=none"])
    self.assertFalse(dotted.extra_transforms.scan_layers)
    self.assertFalse(json_obj.extra_transforms.named_scopes)
    self.assertIsNone(unset.extra_transforms)

  def test_json_objects_merge_with_dotted_flags(self):
    model_cfg, train_cfg = parse_args(
        [
            "--model_name=glm5.3-flash",
            '--model={"num_hidden_layers": 2}',
            "--mesh.ici.fsdp=4",
            '--mesh={"ici": {"tensor": 2}, "dcn": {"data": 2}}',
            '--transforms={"scan_layers": false}',
            "--transforms.named_scopes=false",
        ]
    )
    self.assertEqual(model_cfg.num_hidden_layers, 2)
    self.assertEqual((train_cfg.mesh.ici.fsdp, train_cfg.mesh.ici.tensor, train_cfg.mesh.dcn.data), (4, 2, 2))
    self.assertEqual((train_cfg.transforms.scan_layers, train_cfg.transforms.named_scopes), (False, False))

  @parameterized.named_parameters(
      ("missing_model_name", ["--steps=10"], "`--model_name` exactly once"),
      ("repeated_model_name", [_QWEN3, "--model_name=granitemoe"], "`--model_name` exactly once"),
      ("bare_model_name", ["--model_name"], "requires a non-empty value"),
      ("empty_model_name", ["--model_name="], "requires a non-empty value"),
      ("model_name_under_model", [_QWEN3, "--model.model_name=foo"], "not `--model.model_name`"),
      ("positional_token", ["train.py", _QWEN3], "Expected `--flag`"),
      ("empty_flag_name", ["--=foo"], "Empty flag name"),
      ("missing_value", [_QWEN3, "--steps"], "requires a value"),
      ("hint_model_field", [_QWEN3, "--compute_dtype=float32"], "--model.compute_dtype"),
      ("hint_nested_field", [_QWEN3, "--tensor=2"], "--mesh.ici.tensor"),
      ("hint_partial_path", [_QWEN3, "--ici.tensor=2"], "--mesh.ici.tensor"),
      ("unknown_root", [_QWEN3, "--bogus.field=1"], "Unrecognized CLI flag"),
      ("unknown_subconfig_field", [_QWEN3, "--transforms.bogus=1"], "Unknown field 'bogus' on `transforms`"),
      ("unknown_model_field", [_QWEN3, "--model.bogus=1"], "Unknown field 'bogus' on `model`"),
      ("path_through_scalar", [_QWEN3, "--steps.x=1"], "`steps` is not a sub-config"),
      ("invalid_json", [_QWEN3, "--mesh.ici={bad}"], "Invalid JSON value"),
      ("subconfig_not_json", [_QWEN3, "--transforms=bad"], "expects a JSON object"),
      ("bool_for_int", [_QWEN3, "--steps=true"], "valid integer"),
      (
          "cross_config_validation",
          [_QWEN3, "--per_device_batch_size=1", "--max_target_length=10", "--num_vocab_tiling=4"],
          "num_vocab_tiling",
      ),
  )
  def test_errors(self, argv, message):
    with self.assertRaisesRegex(ValueError, message):  # `pydantic.ValidationError` is a `ValueError`.
      parse_args(argv)


if __name__ == "__main__":
  absltest.main()
