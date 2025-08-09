"""
Copyright 2025 Google LLC

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

import io
from enum import Enum

import yaml

"""Utility functions for MaxText.configs"""

from collections import OrderedDict
from typing import Any, Dict, TypeVar, Mapping, Sequence

from pydantic import BaseModel

from deepmerge import always_merger

from ruamel.yaml import YAML

from MaxText.configs.types import (
    MaxTextConfig,
    CoreConfig,
    ModelConfig,
    CheckpointConfig,
    OptimizerConfig,
    DatasetConfig,
    TokenizerConfig,
    ParallelismConfig,
    InferenceConfig,
)

T = TypeVar("T", bound=BaseModel)


# https://github.com/pydantic/pydantic/discussions/3416#discussioncomment-12267413
def merge_pydantic_models(base: T, nxt: T) -> T:
    """Merge two Pydantic model instances.

    The attributes of 'base' and 'nxt' that weren't explicitly set are dumped into dicts
    using '.model_dump(exclude_unset=True)', which are then merged using 'deepmerge',
    and the merged result is turned into a model instance using '.model_validate'.

    For attributes set on both 'base' and 'nxt', the value from 'nxt' will be used in
    the output result.
    """
    base_dict = base.model_dump(exclude_unset=True)
    nxt_dict = nxt.model_dump(exclude_unset=True)
    merged_dict = always_merger.merge(base_dict, nxt_dict)
    return base.model_validate(merged_dict)


"""
Utilities for serializing MaxTextConfig pydantic objects to YAML files
matching the original pyconfig YAML formatting style.
"""


def _get_final_value_for_flat_dict(value: Any) -> Any:
    """
    Processes values before they are inserted into the flat dictionary.
    Converts Enums to their string values and recursively processes lists/dicts.
    """
    if isinstance(value, Enum):
        return value.value
    elif isinstance(
        value, list
    ):  # Handles lists of primitives, Enums, or further nested dicts/lists
        return [_get_final_value_for_flat_dict(item) for item in value]
    elif isinstance(
        value, dict
    ):  # Handles dicts that are field values (not sub-models)
        return {
            k_nested: _get_final_value_for_flat_dict(v_nested)
            for k_nested, v_nested in value.items()
        }
    # Primitives (int, float, str, bool, None) are returned as is.
    # Pydantic models are handled by the caller (_flatten_model_into_dict).
    return value


# This is a helper, you can place it above config_to_flat_dict
def _flatten_model_into_dict(model_instance: BaseModel, target_dict: Dict[str, Any]):
    """
    Recursively flattens a Pydantic model instance into a target dictionary.
    Nested BaseModel fields are flattened by merging their fields into the target_dict.
    """
    # model_dump(mode='python') converts Enums to values, Decimals to floats, etc.
    # exclude_none=False ensures that fields explicitly set to None are included (will become null).
    # by_alias=False uses the actual field names.
    dumped_data = model_instance.model_dump(
        mode="python", exclude_none=False, by_alias=False
    )

    for field_name, dumped_value in dumped_data.items():
        # Get the actual attribute from the model instance to check if it's a Pydantic sub-model
        actual_value = getattr(
            model_instance, field_name, None
        )  # Use getattr for safety

        if isinstance(actual_value, BaseModel):
            # If the field's value is another Pydantic model, recurse.
            # Its fields will be added directly to the target_dict, achieving flattening.
            _flatten_model_into_dict(actual_value, target_dict)
        else:
            # If it's not a nested Pydantic model (e.g., it's a primitive, list, dict of primitives,
            # or an Enum that model_dump already converted), process its dumped value.
            # The _get_final_value_for_flat_dict handles any Enums or nested lists/dicts
            # within this dumped_value that might need further simple conversion.
            processed_value = _get_final_value_for_flat_dict(dumped_value)

            # Handle potential field name overwrites:
            # If a field name from a sub-model clashes with one from a parent or another sub-model,
            # the one processed later (typically deeper in nesting or later in field order) will win.
            # For MaxTextConfig, usually fields are uniquely named or defaults in sub-models
            # are intended to be overridden if specified at a higher level that flattens later.
            # If a key exists and the new value is None but old one wasn't, prefer old one.
            # This ensures that if a sub-model field is None by default, it doesn't overwrite
            # a potentially set parent value if there was a name clash (unlikely with good design).
            if (
                field_name in target_dict
                and processed_value is None
                and target_dict[field_name] is not None
            ):
                pass  # Keep existing non-None value
            else:
                target_dict[field_name] = processed_value


# This is the main function you need in utils.py
def config_to_flat_dict(config: BaseModel) -> Dict[str, Any]:
    """
    Converts a Pydantic BaseModel instance (e.g., MaxTextConfig) into a
    flat dictionary. Nested Pydantic models have their fields merged into
    the top-level dictionary.

    Args:
        config: The Pydantic BaseModel instance to convert.

    Returns:
        An OrderedDict with keys sorted alphabetically, representing the
        flattened configuration.
    """
    if not isinstance(config, BaseModel):
        # The type hint for 'config' in your existing utils.py is MaxTextConfig.
        # We can make it more general to BaseModel here.
        raise TypeError(
            f"Input 'config' must be a Pydantic BaseModel instance, got {type(config)}"
        )

    flat_dict_accumulator: Dict[str, Any] = {}
    _flatten_model_into_dict(config, flat_dict_accumulator)

    # Sort the final flat dictionary by keys for consistent output order.
    # json.dumps(..., sort_keys=True) will also sort, but doing it here makes
    # the returned dict itself ordered, which can be useful.
    return OrderedDict(sorted(flat_dict_accumulator.items()))


def convert_pydantic_to_flat_dict(config: BaseModel) -> dict:
    """
    Converts a Pydantic model instance (e.g., MaxTextConfig) into a flattened dictionary.
    """
    return config_to_flat_dict(config)


__all__ = [
    "config_to_flat_dict",
    "convert_pydantic_to_flat_dict",
    "dump_config_to_yaml_file",
    "merge_pydantic_models",
]
