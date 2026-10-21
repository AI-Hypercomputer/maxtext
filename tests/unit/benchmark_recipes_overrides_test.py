# Copyright 2023–2026 Google LLC
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

"""Unit tests for MaxText training overrides exposed by benchmarks/recipes."""

import argparse
import contextlib
import dataclasses
import io
import unittest

from benchmarks.recipes import parser_utils
from benchmarks.recipes import runner_utils
from benchmarks.recipes import user_configs


def _parse(argv):
  """Parses argv the same way the recipes do (USER_CONFIG values as defaults)."""
  parser = argparse.ArgumentParser()
  parser_utils.add_arguments(parser)
  parser.set_defaults(
      **{
          f.name: getattr(user_configs.USER_CONFIG, f.name)
          for f in dataclasses.fields(user_configs.USER_CONFIG)
          if f.init
      }
  )
  return parser.parse_args(argv)


def _user_config(argv):
  return user_configs.UserConfig(**vars(_parse(argv)))


class RecipeOverridesTest(unittest.TestCase):
  """Tests for typed MaxText flags and --maxtext_override."""

  def test_typed_flags_parse(self):
    cfg = _user_config(
        [
            "--elastic_enabled=true",
            "--elastic_min_slice_count=1",
            "--checkpoint_period=10",
            "--dataset_type=grain",
            "--grain_train_files=gs://b/*.array_record",
        ]
    )
    self.assertEqual(
        cfg.tuning_param_overrides(),
        {
            "elastic_enabled": True,
            "elastic_min_slice_count": 1,
            "checkpoint_period": 10,
            "dataset_type": "grain",
            "grain_train_files": "gs://b/*.array_record",
        },
    )

  def test_unset_flags_not_emitted(self):
    self.assertEqual(_user_config([]).tuning_param_overrides(), {})

  def test_maxtext_override_repeatable_and_unset(self):
    cfg = _user_config(["--maxtext_override", "profiler=", "--maxtext_override", "remat_policy=full"])
    self.assertEqual(cfg.maxtext_overrides, {"profiler": None, "remat_policy": "full"})
    for bad in ("no_equals_sign", "=value", "bad key=1"):
      with self.subTest(bad=bad):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
          _parse(["--maxtext_override", bad])

  def test_set_defaults_dict_not_mutated(self):
    _user_config(["--maxtext_override", "profiler="])
    self.assertEqual(user_configs.USER_CONFIG.maxtext_overrides, {})

  def test_conflict_raises(self):
    with self.assertRaisesRegex(ValueError, "checkpoint_period"):
      _user_config(["--checkpoint_period=10", "--maxtext_override", "checkpoint_period=20"])

  def test_same_value_ok(self):
    cfg = _user_config(["--enable_checkpointing=true", "--maxtext_override", "enable_checkpointing=true"])
    self.assertEqual(cfg.tuning_param_overrides(), {"enable_checkpointing": True})

  def test_same_value_different_spelling_ok(self):
    # Overrides are compared after parsing with the typed flag's own type, so equivalent spellings agree.
    for typed_arg, override in (
        ("--enable_checkpointing=true", "enable_checkpointing=True"),
        ("--enable_checkpointing=false", "enable_checkpointing=0"),
        ("--checkpoint_period=10", "checkpoint_period=010"),
        ("--dataset_type=grain", "dataset_type=grain"),
    ):
      with self.subTest(typed_arg=typed_arg, override=override):
        key = override.split("=", 1)[0]
        cfg = _user_config([typed_arg, "--maxtext_override", override])
        self.assertIn(key, cfg.tuning_param_overrides())

  def test_unparseable_or_unset_override_conflicts(self):
    for override in ("enable_checkpointing=maybe", "enable_checkpointing="):
      with self.subTest(override=override):
        with self.assertRaisesRegex(ValueError, "Conflicting values"):
          _user_config(["--enable_checkpointing=true", "--maxtext_override", override])

  def test_dataclasses_replace_keeps_overrides(self):
    cfg = dataclasses.replace(_user_config(["--elastic_enabled=true"]), max_restarts=10)
    self.assertEqual(cfg.tuning_param_overrides(), {"elastic_enabled": True})

  def test_flag_specs_match_userconfig_fields(self):
    field_names = {f.name for f in dataclasses.fields(user_configs.UserConfig)}
    self.assertEqual(len(parser_utils.MAXTEXT_FLAG_NAMES), len(set(parser_utils.MAXTEXT_FLAG_NAMES)))
    self.assertLessEqual(set(parser_utils.MAXTEXT_FLAG_NAMES), field_names)
    self.assertIn("maxtext_overrides", field_names)

  def test_workload_config_receives_overrides(self):
    cfg = _user_config(["--selected_model_framework=mcjax", "--elastic_enabled=true", "--maxtext_override", "profiler="])
    framework, models = next((k, v) for k, v in cfg.models.items() if v)
    wl_config = runner_utils._create_workload_config(  # pylint: disable=protected-access
        framework, models[0], 2, cfg, 20, "medium"
    )
    self.assertEqual(wl_config.extra_tuning_params, {"profiler": None, "elastic_enabled": True})

  def test_workload_config_without_overrides_is_none(self):
    cfg = _user_config(["--selected_model_framework=mcjax"])
    framework, models = next((k, v) for k, v in cfg.models.items() if v)
    wl_config = runner_utils._create_workload_config(  # pylint: disable=protected-access
        framework, models[0], 2, cfg, 20, "medium"
    )
    self.assertIsNone(wl_config.extra_tuning_params)


if __name__ == "__main__":
  unittest.main()
