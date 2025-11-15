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

# pytype: skip-file
"""Pydantic-based configuration management for MaxText."""
import logging
import os
import sys
from typing import Any

import jax
import jax.numpy as jnp

import omegaconf

from MaxText import max_utils
from MaxText import pyconfig_deprecated
from MaxText.common_types import DecoderBlockType, ShardMode
from MaxText.configs import types
from MaxText.inference_utils import str2bool

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

_BASE_CONFIG_ATTR = "base_config"
_MAX_PREFIX = "M_"
_yaml_types_to_parser = {str: str, int: int, float: float, bool: str2bool}


def yaml_key_to_env_key(s: str) -> str:
  return _MAX_PREFIX + s.upper()


def resolve_config_path(param: str) -> str:
  """Resolve config path to auto rewrite to use new src folder."""
  return param if os.path.isfile(param) else os.path.join("src", param)


def _load_config(config_name: str) -> omegaconf.DictConfig:
  """Loads a YAML file and its base_configs recursively using OmegaConf."""
  cfg = omegaconf.OmegaConf.load(config_name)
  if _BASE_CONFIG_ATTR in cfg:
    base_path = cfg[_BASE_CONFIG_ATTR]
    if not os.path.isabs(base_path):
      # Search relative to current config, then in the default configs folder
      loaded_parent_config_filename = os.path.join(os.path.dirname(config_name), base_path)
      if not os.path.isfile(loaded_parent_config_filename):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        loaded_parent_config_filename = os.path.join(dir_path, "configs", base_path)
    else:
      loaded_parent_config_filename = base_path

    base_cfg = _load_config(loaded_parent_config_filename)
    cfg = omegaconf.OmegaConf.merge(base_cfg, cfg)
  return cfg


def _tuples_to_lists(l: list | tuple | Any) -> list | Any:
  """Recursively converts nested tuples to lists for Pydantic compatibility."""
  return [_tuples_to_lists(x) for x in l] if isinstance(l, (list, tuple)) else l


def _lists_to_tuples(l: list | Any) -> tuple | Any:
  """Recursively converts nested lists to tuples for JAX compatibility."""
  return tuple(_lists_to_tuples(x) for x in l) if isinstance(l, list) else l


def _prepare_for_pydantic(raw_keys: dict[str, Any]) -> dict[str, Any]:
  """Prepares the raw dictionary for Pydantic model instantiation."""
  pydantic_kwargs = {}
  valid_fields = types.MaxTextConfig.model_fields.keys()

  # This is a workaround for tests that use `dataset_type='hf'` but do not
  # specify `tokenizer_type='huggingface'`, which they should.
  if raw_keys.get("dataset_type") == "hf" and "tokenizer_type" not in raw_keys:
    raw_keys["tokenizer_type"] = "huggingface"

  for key, value in raw_keys.items():
    if key not in valid_fields:
      logger.warning("Ignoring invalid/unsupported field from YAML/CLI: %s", repr(key))
      continue

    new_value = value
    if isinstance(new_value, str) and new_value.lower() == "none":
      new_value = None

    # Pydantic validates enums from their values, so string is fine.
    # It also handles type coercion for simple types.
    if key in ("logical_axis_rules", "data_sharding"):
      if isinstance(new_value, tuple):
        new_value = _tuples_to_lists(new_value)
      if key == "data_sharding" and isinstance(new_value, list) and new_value and isinstance(new_value[0], str):
        new_value = [new_value]

    if key in ("run_name", "hf_train_files", "hf_eval_files") and new_value is None:
      new_value = ""

    pydantic_kwargs[key] = new_value

  return pydantic_kwargs


class HyperParameters:
  """
  Wrapper class to expose the configuration in a read-only manner,
  maintaining backward compatibility with attribute-style access and JAX object types.
  """

  def __init__(self, pydantic_config: types.MaxTextConfig):
    object.__setattr__(self, "_pydantic_config", pydantic_config)

    final_dict = pydantic_config.model_dump()
    final_dict["dtype"] = jnp.dtype(final_dict["dtype"])
    final_dict["grad_dtype"] = jnp.dtype(final_dict["grad_dtype"])
    final_dict["weight_dtype"] = jnp.dtype(final_dict["weight_dtype"])
    final_dict["mu_dtype"] = (
        final_dict["weight_dtype"] if not final_dict["mu_dtype"] else jnp.dtype(final_dict["mu_dtype"])
    )

    final_dict["logical_axis_rules"] = _lists_to_tuples(final_dict["logical_axis_rules"])
    final_dict["data_sharding"] = _lists_to_tuples(final_dict["data_sharding"])

    final_dict["decoder_block"] = DecoderBlockType(final_dict["decoder_block"])
    final_dict["shard_mode"] = ShardMode(final_dict["shard_mode"])

    object.__setattr__(self, "_flat_config", final_dict)

  def __getattr__(self, attr: str) -> Any:
    """Provides attribute-style access to the final configuration dictionary."""
    if attr in self._flat_config:
      return self._flat_config[attr]
    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

  def __setattr__(self, attr: str, value: Any) -> None:
    """Makes the configuration object read-only."""
    raise ValueError("Configuration is read-only and cannot be modified after initialization.")

  def get_keys(self) -> dict[str, Any]:
    """Returns the configuration as a flat dictionary for backward compatibility."""
    return self._flat_config


def initialize(argv: list[str], **kwargs) -> HyperParameters:
  """Initializes the configuration by loading YAML files, and applying CLI, env, and kwarg overrides."""
  # 1. Load base and inherited configs from file(s)
  config_path = resolve_config_path(argv[1])
  base_yml_config = _load_config(config_path)

  # 2. Get overrides from CLI and kwargs
  cli_cfg = omegaconf.OmegaConf.from_cli(argv[2:])
  kwargs_cfg = omegaconf.OmegaConf.create(kwargs)
  overrides_cfg = omegaconf.OmegaConf.merge(cli_cfg, kwargs_cfg)

  # 3. Handle model-specific config
  temp_cfg = omegaconf.OmegaConf.merge(base_yml_config, overrides_cfg)
  model_name = temp_cfg.get("model_name", "default")
  model_cfg = {}
  if model_name != "default":
    # First try relative to base config path
    model_config_path = os.path.join(os.path.dirname(config_path), "models", f"{model_name}.yml")
    if not os.path.isfile(model_config_path):
      # Fallback to default location within package
      dir_path = os.path.dirname(os.path.realpath(__file__))
      model_config_path = os.path.join(dir_path, "configs", "models", f"{model_name}.yml")

    if os.path.exists(model_config_path):
      model_loaded_cfg = omegaconf.OmegaConf.load(model_config_path)
      # if override_model_config=True, only apply model configs for keys not present in overrides.
      if temp_cfg.get("override_model_config"):
        model_cfg = {k: v for k, v in model_loaded_cfg.items() if k not in overrides_cfg}
      else:
        model_cfg = model_loaded_cfg
    else:
      logger.warning("Model config for '%s' not found at %s", model_name, model_config_path)

      # 4. Final merge (base, model, then overrides)
  final_config = omegaconf.OmegaConf.merge(base_yml_config, model_cfg, overrides_cfg)
  raw_keys_dict = omegaconf.OmegaConf.to_container(final_config, resolve=True)

  # 5. Handle environment variable overrides
  cli_keys = set(omegaconf.OmegaConf.to_container(cli_cfg, resolve=True).keys())
  kwargs_keys = set(kwargs.keys())
  for k in list(raw_keys_dict.keys()):
    env_key = yaml_key_to_env_key(k)
    if env_key in os.environ:
      if k in cli_keys or k in kwargs_keys:
        raise ValueError(
            f"Key '{k}' is overridden by both CLI/kwargs and environment variable '{env_key}'. This is not allowed."
        )

      new_proposal = os.environ.get(env_key)
      original_value = raw_keys_dict.get(k)
      parser = None
      if isinstance(original_value, bool):
        parser = _yaml_types_to_parser[bool]
      elif isinstance(original_value, (str, int, float)):
        parser = type(original_value)

      if parser is None:
        raise TypeError(f"Type {type(original_value)} for key '{k}' not supported for ENV override.")

      try:
        raw_keys_dict[k] = parser(new_proposal)
      except (ValueError, KeyError) as e:
        raise ValueError(f"Couldn't parse value from ENV '{new_proposal}' for key '{k}'") from e

  pydantic_kwargs = _prepare_for_pydantic(raw_keys_dict)

  # Initialize JAX distributed system before device backend is initialized.
  if pydantic_kwargs.get("jax_debug_log_modules"):
    jax.config.update("jax_debug_log_modules", pydantic_kwargs["jax_debug_log_modules"])
  # Do not initialize jax distributed system during pytest runs.
  if "pytest" not in sys.modules:
    max_utils.maybe_initialize_jax_distributed_system(pydantic_kwargs)
  if pydantic_kwargs.get("jax_cache_dir"):
    from jax.experimental.compilation_cache import compilation_cache  # pylint: disable=import-outside-toplevel

    compilation_cache.set_cache_dir(os.path.expanduser(pydantic_kwargs["jax_cache_dir"]))

  pydantic_config = types.MaxTextConfig(**pydantic_kwargs)
  config = HyperParameters(pydantic_config)

  if config.log_config:
    for k, v in sorted(config.get_keys().items()):
      if k != "hf_access_token":
        logger.info("Config param %s: %s", k, v)

  return config


# Shim for backward compatibility with pyconfig_deprecated_test.py
validate_and_update_keys = pyconfig_deprecated.validate_and_update_keys
__all__ = ["initialize"]
