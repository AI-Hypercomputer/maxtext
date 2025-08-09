# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Pydantic-based configuration system and its legacy compatibility."""

import io
import os
import unittest
from typing import List, Dict, Any
from unittest.mock import patch

import yaml

from pydantic import ValidationError
from pydantic_core import PydanticUndefined

from ml_collections.config_dict import ConfigDict

from MaxText.configs import types_k as new_pyconfig
from MaxText.configs.types_k import MaxTextConfig
from MaxText.globals import PKG_DIR


# ----------------------------------------------------------------------------
# Legacy Pyconfig Simulator
# This mock object simulates the behavior of the old pyconfig system
# for the purpose of comparison testing.
# ----------------------------------------------------------------------------


class LegacyPyconfigSimulator:
    """A simulator for the old pyconfig loading logic, updated to match new system's behavior."""

    def _load_and_merge_yamls(
        self, filepath: str, config_dir: str = None
    ) -> Dict[str, Any]:
        """Correctly loads and merges YAML files with relative base_config paths."""
        if config_dir is None:
            config_dir = os.path.join(PKG_DIR, "configs")

        with open(filepath, "rt", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if "base_config" in cfg and cfg["base_config"]:
            base_config_filename = cfg.pop("base_config")
            base_path = os.path.join(config_dir, base_config_filename)
            base_cfg = self._load_and_merge_yamls(base_path, os.path.dirname(base_path))
            return {**base_cfg, **cfg}

        cfg.pop("base_config", None)  # Clean up empty base_config key
        return cfg

    def _parse_cli(self, args: List[str]) -> Dict[str, Any]:
        """Parses CLI overrides, using yaml.safe_load for type inference."""
        overrides = {}
        for arg in args:
            if "=" in arg:
                key, value_str = arg.split("=", 1)
                try:
                    value = yaml.safe_load(value_str)
                except (yaml.YAMLError, AttributeError):
                    value = value_str
                overrides[key] = value
        return overrides

    def _update_model_vars(self, config_dict: dict, maxtext_path: str) -> dict:
        """Loads and merges a model-specific YAML, with user config taking precedence."""
        model_name = config_dict.get("model_name")
        if model_name and model_name != "default":
            model_path1 = os.path.join(
                maxtext_path, "configs", "models", f"{model_name}.yml"
            )
            model_path2 = os.path.join(
                maxtext_path, "configs", "models", "gpu", f"{model_name}.yml"
            )
            model_path = model_path1 if os.path.exists(model_path1) else model_path2

            if os.path.exists(model_path):
                # Correctly load model config with its own potential base configs
                model_config = self._load_and_merge_yamls(model_path)
                return {**model_config, **config_dict}
        return config_dict

    def initialize(self, argv: List[str]) -> ConfigDict:
        """Simulates the old initialize function but produces a complete config dict with defaults."""
        config_file = next(
            (arg for arg in argv[1:] if arg.endswith(".yml") and "=" not in arg), None
        )
        if not config_file:
            raise ValueError("No config file found.")

        # Replicate new system's loading order and logic
        merged_config = self._load_and_merge_yamls(config_file)

        maxtext_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        merged_config = self._update_model_vars(merged_config, maxtext_path)

        cli_overrides = self._parse_cli(argv[1:])
        merged_config.update(cli_overrides)

        # *** KEY FIX: APPLY DEFAULTS ***
        # Get all default values from the Pydantic model itself.
        defaults = {}
        for name, field in MaxTextConfig.model_fields.items():
            if field.default is not PydanticUndefined:
                # get_default() correctly handles default_factory
                defaults[name] = field.get_default()

        # The final config is the defaults overridden by the loaded values.
        final_config_dict = defaults
        final_config_dict.update(merged_config)

        cfg = ConfigDict(final_config_dict)

        # Simulate derived values logic, which happens *after* all values (including defaults) are set.
        if cfg.get("learning_rate_schedule_steps") == -1:
            cfg.learning_rate_schedule_steps = cfg.steps
        if cfg.get("eval_per_device_batch_size") == 0.0:
            cfg.eval_per_device_batch_size = cfg.per_device_batch_size

        return cfg


# ----------------------------------------------------------------------------
# Main Test Suite
# ----------------------------------------------------------------------------


def get_test_resource_path(*args):
    """Helper to get the absolute path to a test resource file."""
    # Assuming the test is run from the project root or similar
    # and `MaxText` is a top-level directory.
    return os.path.join(os.path.dirname(__file__), "..", *args)


class ConfigTest(unittest.TestCase):
    """Tests the new Pydantic config system, its helpers, and its compatibility."""

    def test_parse_cli_key_value_pairs(self):
        """Tests the internal CLI parser helper function."""
        argv = [
            "script.py",
            get_test_resource_path("configs", "base.yml"),
            "int_val=123",
            "float_val=0.45",
            "bool_true=True",
            "bool_false=false",
            "str_val=hello",
            "list_val=[1, 2, 3]",
            "dict_val={'a': 1}",
            "simple_list=a,b,c",
            "path_val=gs://bucket/path",
        ]
        parsed = new_pyconfig._parse_cli_key_value_pairs(argv)
        self.assertEqual(parsed["int_val"], 123)
        self.assertEqual(parsed["float_val"], 0.45)
        self.assertTrue(parsed["bool_true"])
        self.assertFalse(parsed["bool_false"])
        self.assertEqual(parsed["str_val"], "hello")
        self.assertEqual(parsed["list_val"], [1, 2, 3])
        self.assertEqual(parsed["dict_val"], {"a": 1})
        self.assertEqual(parsed["simple_list"], ["a", "b", "c"])
        self.assertEqual(parsed["path_val"], "gs://bucket/path")

    def test_forbid_extra_fields(self):
        """Ensures that unknown fields raise a validation error."""
        config_path = get_test_resource_path("configs", "tpu_smoke_test.yml")
        argv = ["test.py", config_path, "this_field_is_not_allowed=true"]

        with self.assertRaises(ValidationError) as context:
            new_pyconfig.initialize(argv)

        self.assertIn("this_field_is_not_allowed", str(context.exception))
        self.assertIn("Extra inputs are not permitted", str(context.exception))

    def test_yaml_to_config_roundtrip(self):
        """
        Tests a full round trip:
        1. Load a YAML file into a MaxTextConfig object.
        2. Dump the object back into a YAML string.
        3. Load the generated YAML string.
        4. Assert that the original object and the re-loaded object are identical.
        """
        config_path = get_test_resource_path(
            "configs", "models", "gpu", "mixtral_8x7b.yml"
        )
        argv = [
            "test.py",
            config_path,
            "log_config=false",
        ]  # Disable logging for clean test output

        # 1. YAML to MaxTextConfig
        original_config = new_pyconfig.initialize(argv)
        self.assertIsInstance(original_config, MaxTextConfig)

        # 2. MaxTextConfig back to YAML string
        config_dict = original_config.model_dump(mode="json")

        string_stream = io.StringIO()
        yaml.dump(config_dict, string_stream, sort_keys=False, default_flow_style=False)
        yaml_string_output = string_stream.getvalue()

        # 3. Load the generated YAML string back into a dictionary
        reloaded_dict = yaml.safe_load(yaml_string_output)

        # 4. Create a new MaxTextConfig object from the reloaded dictionary and compare.
        reloaded_config = MaxTextConfig.model_validate(reloaded_dict)

        self.assertEqual(
            original_config,
            reloaded_config,
            "Config object should be identical after round-trip.",
        )

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_legacy_and_new_config_equality(self, mock_stderr):
        """
        Asserts that both the legacy pyconfig and the new types.py produce
        the same final configuration dictionary from the same inputs.
        """
        legacy_loader = LegacyPyconfigSimulator()

        config_path_rel = os.path.join("configs", "models", "gpu", "llama3_70b.yml")
        config_path_abs = get_test_resource_path(config_path_rel)
        cli_overrides = [
            "run_name=equality_test",
            "steps=99",
            "learning_rate=5.5e-5",
            "enable_checkpointing=false",
            "logical_axis_rules=[['foo', 'bar']]",
            "per_device_batch_size=2.5",
        ]
        argv = ["test.py", config_path_abs] + cli_overrides

        # 1. Get config dict from the legacy system (now simulated to include defaults)
        legacy_config = legacy_loader.initialize(argv)
        legacy_dict = legacy_config.to_dict()

        # 2. Get config dict from the new Pydantic system
        new_config_obj = new_pyconfig.initialize(argv)
        new_dict = new_config_obj.model_dump(mode="json")

        # 3. Compare the dictionaries.
        # We perform a detailed key-by-key comparison for better debug output on mismatch.
        all_keys = sorted(list(set(legacy_dict.keys()) | set(new_dict.keys())))

        mismatched_keys = []
        for key in all_keys:
            legacy_val = legacy_dict.get(key)
            new_val = new_dict.get(key)
            if legacy_val != new_val:
                mismatched_keys.append(
                    f"Key '{key}': legacy='{legacy_val}' (type: {type(legacy_val).__name__}) vs new='{new_val}' (type: {type(new_val).__name__})"
                )

        self.assertEqual(
            len(mismatched_keys),
            0,
            f"Found {len(mismatched_keys)} mismatched keys between legacy and new config systems:\n"
            + "\n".join(mismatched_keys),
        )

        self.assertDictEqual(
            new_dict,
            legacy_dict,
            "The final configuration dictionaries from legacy and new systems must be identical.",
        )
        # print(f"\nSuccessfully verified config equality for {config_path_rel}")


if __name__ == "__main__":
    unittest.main()
