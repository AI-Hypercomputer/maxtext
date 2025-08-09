"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pytype: skip-file
# pylint: disable=missing-module-docstring, bare-except, consider-using-generator, missing-function-docstring

import datetime
import os
import sys
from typing import Any, Dict, List, Optional

import jax
from jax.experimental.compilation_cache import compilation_cache
from pydantic import ValidationError
import yaml

from MaxText import max_utils
from MaxText.configs import types_k
from MaxText.globals import PKG_DIR

def _load_and_merge_yamls(filepath: str) -> Dict[str, Any]:
    """Recursively loads `base_config` files and merges them."""
    with open(filepath, "rt", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "base_config" in cfg and cfg["base_config"]:
        parent_config_filename = cfg.pop("base_config")
        if not os.path.isabs(parent_config_filename):
            # Assumes parent is relative to the current file's directory
            base_config_path = os.path.join(os.path.dirname(filepath), parent_config_filename)
        else:
            base_config_path = parent_config_filename

        # Fallback to the global configs dir if not found relative to the file.
        if not os.path.isfile(base_config_path):
            base_config_path = os.path.join(PKG_DIR, "configs", parent_config_filename)

        base_cfg = _load_and_merge_yamls(base_config_path)
        # Higher-level config (cfg) overrides base_cfg
        return {**base_cfg, **cfg}

    cfg.pop("base_config", None)
    return cfg

def _parse_cli_key_value_pairs(args: List[str]) -> Dict[str, Any]:
    """Parses key=value pairs from a list of strings."""
    overrides = {}
    for arg in args:
        if "=" in arg:
            key, value_str = arg.split("=", 1)
            key = key.strip()
            value_str = value_str.strip()
            try:
                # Attempt to parse as a number first
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    # Handle boolean strings
                    if value_str.lower() in ("true", "t", "1"):
                        value = True
                    elif value_str.lower() in ("false", "f", "0"):
                        value = False
                    # Handle comma-separated lists (if not already a YAML list)
                    elif "," in value_str and "[" not in value_str:
                        value = [item.strip() for item in value_str.split(",")]
                    else:
                        # Fallback to YAML parsing for lists, dicts, etc.
                        try:
                            value = yaml.safe_load(value_str)
                        except (yaml.YAMLError, AttributeError):
                            value = value_str
            overrides[key] = value
    return overrides

def _update_with_model_vars(config_dict: dict, base_config_path: str) -> dict:
    """Loads and merges a model-specific YAML based on `model_name`."""
    model_name = config_dict.get("model_name")
    if model_name and model_name != "default":
        # First check for model config relative to the base config path
        config_dir = os.path.dirname(base_config_path)
        model_paths_to_try = [
            os.path.join(config_dir, "models", f"{model_name}.yml"),
            os.path.join(config_dir, "models", "gpu", f"{model_name}.yml"),
            # Fallback to package config directory
            os.path.join(PKG_DIR, "configs", "models", f"{model_name}.yml"),
            os.path.join(PKG_DIR, "configs", "models", "gpu", f"{model_name}.yml"),
        ]
        model_path = next((path for path in model_paths_to_try if os.path.exists(path)), None)

        if model_path:
            print(f"Applying model configuration from: {model_path}", file=sys.stderr)
            model_config = _load_and_merge_yamls(model_path)
            # The model config is the new base, and the existing config_dict overrides it.
            return {**model_config, **config_dict}

    return config_dict

class HyperParameters:
    """A wrapper class to provide a read-only-like interface to the configuration
    that is compatible with the legacy pyconfig system."""

    def __init__(self, config: types_k.MaxTextConfig):
        # Use object.__setattr__ to avoid a recursive loop with our __setattr__.
        object.__setattr__(self, "_config", config)

    def __getattr__(self, attr):
        # Get the attribute from the wrapped pydantic config object.
        if hasattr(self._config, attr):
            return getattr(self._config, attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __setattr__(self, attr, value):
        # Allow mutation to match the behavior of the new Pydantic-based system
        # and not break any code that might rely on modifying config post-initialization.
        setattr(self._config, attr, value)

    def get_keys(self):
        """Provides a list of all configuration keys for compatibility."""
        return self._config.model_dump().keys()

def initialize(argv: List[str], **kwargs) -> HyperParameters:
    """
    Initializes the configuration by parsing YAML files, command-line arguments,
    and keyword arguments, returning a validated, ready-to-use config object.

    The precedence for configuration values is:
    0. Keyword arguments to this function (highest).
    1. Command-line arguments in `key=value` format.
    2. The user-provided YAML file.
    3. The `model_name` specific YAML file.
    4. The `base_config` YAML file(s).
    5. Default values defined in the Pydantic model (lowest).
    """
    config_file = next((arg for arg in argv[1:] if arg.endswith((".yml", ".yaml")) and "=" not in arg), None)
    if not config_file:
        raise ValueError("No YAML configuration file specified in `argv`.")

    # Peek at overrides to initialize JAX distributed system early if needed.
    cli_overrides = _parse_cli_key_value_pairs(argv[1:])
    temp_merged_config = _load_and_merge_yamls(config_file)
    temp_merged_config.update(cli_overrides)
    temp_merged_config.update(kwargs)

    max_utils.maybe_initialize_jax_distributed_system(temp_merged_config)
    if temp_merged_config.get("jax_cache_dir"):
        compilation_cache.set_cache_dir(os.path.expanduser(temp_merged_config["jax_cache_dir"]))

    # 1. Load base YAML configuration.
    print(f"Loading configuration from: {config_file}", file=sys.stderr)
    merged_config = _load_and_merge_yamls(config_file)

    # 2. Merge model-specific configuration.
    merged_config = _update_with_model_vars(merged_config, config_file)

    # Replicate run_name logic from the old pyconfig.
    if not merged_config.get("run_name"):
        run_name = os.environ.get("JOBSET_NAME")
        if not run_name:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d-%H-%M")
            model_name = merged_config.get("model_name", "model")
            run_name = f"{model_name}_{timestamp}"
        merged_config["run_name"] = run_name

    # 3. Merge overrides from both CLI strings and keyword arguments.
    final_overrides = {**cli_overrides, **kwargs}
    merged_config.update(final_overrides)

    # 4. Validate and instantiate the Pydantic model.
    try:
        config_instance = types_k.MaxTextConfig.model_validate(merged_config)
    except ValidationError as e:
        print(f"Configuration validation error:\n{e}", file=sys.stderr)
        print("\n--- Merged Config that Failed Validation ---", file=sys.stderr)
        print(yaml.dump(merged_config, sort_keys=False, default_flow_style=False, indent=2), file=sys.stderr)
        print("--- End of Failed Config ---", file=sys.stderr)
        raise

    # 5. Log the final configuration if requested.
    if config_instance.log_config:
        print("\nFinal Configuration:", file=sys.stderr)
        # Using model_dump() with exclude_defaults makes the output cleaner.
        config_dict = config_instance.model_dump(exclude_defaults=True)
        print(yaml.dump(config_dict, sort_keys=True, default_flow_style=False, indent=2), file=sys.stderr)

    # 6. Return a compatibility wrapper.
    return HyperParameters(config_instance)
