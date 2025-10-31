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
from typing import Any

import pydantic

from MaxText import pyconfig_deprecated
from MaxText.common_types import DecoderBlockType
from MaxText.configs import types
from MaxText.pyconfig_deprecated import validate_and_update_keys

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))


def _tuples_to_lists(t: tuple | list | Any) -> list | Any:
  """Recursively converts nested tuples to lists."""
  return [_tuples_to_lists(x) for x in t] if isinstance(t, (tuple, list)) else t


def pyconfig_to_pydantic(raw_keys: dict[str, Any]) -> types.MaxTextConfig:
  """
  Converts a raw pyconfig dict to a pydantic MaxTextConfig object
  This involves "dehydrating" custom objects, filtering unknown keys,
  and cleaning data to pass Pydantic's stricter validation.
  """
  pydantic_kwargs = {}
  valid_fields = types.MaxTextConfig.model_fields

  for key, value in raw_keys.items():
    if key not in valid_fields:
      logger.warning("Ignoring invalid|unsupported field: %s", repr(key))
      continue

    new_value = value

    match key:
      case "run_name" | "hf_train_files" | "hf_eval_files":
        if new_value is None:
          new_value = ""

      case "dtype" | "grad_dtype" | "weight_dtype" | "mu_dtype":
        if new_value is not None and not isinstance(new_value, str):
          new_value = str(new_value)

      case "decoder_block":
        if isinstance(new_value, DecoderBlockType):
          new_value = new_value.value

      case "logical_axis_rules":
        if isinstance(new_value, tuple):
          new_value = _tuples_to_lists(new_value)

      case "data_sharding":
        if isinstance(new_value, tuple):
          new_value = _tuples_to_lists(new_value)

        if isinstance(new_value, list) and new_value and isinstance(new_value[0], str):
          new_value = [new_value]

      case _:
        pass
        # logger.info("Doing nothing special with %s -> %s of type %s", repr(key), repr(value), type(value))

    pydantic_kwargs[key] = new_value

  return types.MaxTextConfig(**pydantic_kwargs)


def pydantic_to_pyconfig(cfg: types.MaxTextConfig) -> dict[str, Any]:
  """
  Converts a pydantic MaxTextConfig object back to a flat dictionary of primitive types.
  """
  return cfg.model_dump()


class HyperParameters:
  """
  Wrapper class to expose the configuration in a read-only manner,
  maintaining backward compatibility with attribute-style access and JAX object types.
  """

  def __init__(self, pydantic_config: types.MaxTextConfig, raw_keys_dict: dict[str, Any]):
    object.__setattr__(self, "_pydantic_config", pydantic_config)

    # The single source of truth is the fully-processed dictionary from pyconfig_deprecated.py.
    # We instantiate Pydantic primarily to validate the config against the schema, but we
    # do not use its re-derived values, as that could break backward compatibility.
    # By using raw_keys_dict directly, we ensure all tests pass without modification.
    object.__setattr__(self, "_flat_config", raw_keys_dict)

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
  """
  Initializes the configuration by running the legacy logic from pyconfig_deprecated.py,
  then converts the result into the new pydantic-based configuration object,
  and wraps it for read-only access.
  """
  # pylint: disable=protected-access
  _config = pyconfig_deprecated._HyperParameters(argv, **kwargs)
  raw_keys_dict = _config.keys

  try:
    pydantic_config = pyconfig_to_pydantic(raw_keys_dict)
    config = HyperParameters(pydantic_config, raw_keys_dict)
  except (TypeError, pydantic.ValidationError) as e:
    logger.warning("Please report this failure of pydantic to parse MaxText config")
    if "GITHUB_ACTIONS" in os.environ and os.environ.get("GITHUB_REPOSITORY", "").rpartition("/")[2] == "maxtext":
      raise e
    logger.warning(e, exc_info=True)
    config = _config
  return config


__all__ = ["initialize", "validate_and_update_keys"]
