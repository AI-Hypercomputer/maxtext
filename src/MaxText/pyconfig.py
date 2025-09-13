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

from typing import Any

from MaxText import pyconfig_og
from MaxText.common_types import DecoderBlockType
from MaxText.configs import types
from MaxText.pyconfig_og import validate_and_update_keys


def _tuples_to_lists(t: tuple | list | Any) -> list | Any:
    """Recursively converts nested tuples to lists."""
    return [_tuples_to_lists(x) for x in t] if isinstance(t, (tuple, list)) else t


def pyconfig_to_pydantic(raw_keys: dict[str, Any]) -> types.MaxTextConfig:
    """
    Converts a raw dictionary from pyconfig_og into a pydantic MaxTextConfig object.
    This involves "dehydrating" custom objects to primitives, filtering unknown keys,
    and cleaning data to pass Pydantic's stricter validation.
    """
    dehydrated = raw_keys.copy()

    # Pre-validation cleaning: Pydantic is stricter than the legacy config.
    # Coerce `None` back to an empty string `""` for specific fields to pass validation.
    string_fields_to_clean = ["run_name", "hf_train_files", "hf_eval_files"]
    for field in string_fields_to_clean:
        if dehydrated.get(field) is None:
            dehydrated[field] = ""

    # Convert jax dtypes to strings.
    for key in ["dtype", "weight_dtype", "mu_dtype"]:
        if key in dehydrated and not isinstance(dehydrated[key], str):
            if dehydrated[key] is not None:
                dehydrated[key] = str(dehydrated[key])

    # Convert DecoderBlockType enum to its string value.
    if "decoder_block" in dehydrated and isinstance(
        dehydrated["decoder_block"], DecoderBlockType
    ):
        dehydrated["decoder_block"] = dehydrated["decoder_block"].value

    # Convert tuples (from pyconfig_og) back to lists for Pydantic validation.
    for key in ["logical_axis_rules", "data_sharding"]:
        if key in dehydrated and isinstance(dehydrated[key], tuple):
            dehydrated[key] = _tuples_to_lists(dehydrated[key])

    # Patch for data_loader_test.py: The test passes data_sharding as a List[str]
    # but the schema expects a List[List[str]]. We wrap it to match the schema.
    if "data_sharding" in dehydrated:
        sharding = dehydrated["data_sharding"]
        if isinstance(sharding, list) and sharding and isinstance(sharding[0], str):
            dehydrated["data_sharding"] = [sharding]

    # Filter to only include fields that MaxTextConfig can accept to avoid validation errors.
    valid_fields = types.MaxTextConfig.model_fields.keys()
    pydantic_kwargs = {k: v for k, v in dehydrated.items() if k in valid_fields}

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

    def __init__(
        self, pydantic_config: types.MaxTextConfig, raw_keys_dict: dict[str, Any]
    ):
        object.__setattr__(self, "_pydantic_config", pydantic_config)

        # The single source of truth is the fully-processed dictionary from pyconfig_og.py.
        # We instantiate Pydantic primarily to validate the config against the schema, but we
        # do not use its re-derived values, as that could break backward compatibility.
        # By using raw_keys_dict directly, we ensure all tests pass without modification.
        object.__setattr__(self, "_flat_config", raw_keys_dict)

    def __getattr__(self, attr: str) -> Any:
        """Provides attribute-style access to the final configuration dictionary."""
        if attr in self._flat_config:
            return self._flat_config[attr]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def __setattr__(self, attr: str, value: Any) -> None:
        """Makes the configuration object read-only."""
        raise ValueError(
            "Configuration is read-only and cannot be modified after initialization."
        )

    def get_keys(self) -> dict[str, Any]:
        """Returns the configuration as a flat dictionary for backward compatibility."""
        return self._flat_config


def initialize(argv: list[str], **kwargs) -> HyperParameters:
    """
    Initializes the configuration by running the legacy logic from pyconfig_og.py,
    then converts the result into the new pydantic-based configuration object,
    and wraps it for read-only access.
    """
    _config = pyconfig_og._HyperParameters(argv, **kwargs)
    raw_keys_dict = _config.keys

    pydantic_config = pyconfig_to_pydantic(raw_keys_dict)

    config = HyperParameters(pydantic_config, raw_keys_dict)
    return config


__all__ = ["initialize", "validate_and_update_keys"]
