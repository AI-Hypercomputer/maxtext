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

"""This file contains common utilities for MaxText benchmarking.

It includes:
  - The `MaxTextModel` dataclass for defining model configurations.
  - Helper functions like `str2bool` for parsing arguments.
"""

import dataclasses
import typing

from enum import Enum


class Framework(Enum):
  PATHWAYS = "pathways"
  MCJAX = "mcjax"


def str2bool(v: str) -> bool:
  """Convert a string of truth to True or False.

  Args:
    - v (str):
      - True values are 'y', 'yes', 't', 'true', and '1';
      - False values are 'n', 'no', 'f', 'false', and '0'.

  Returns:
    bool: True or False

  Raises:
    ValueError if v is anything else.
  """
  v = v.lower()
  true_values = ["y", "yes", "t", "true", "1"]
  false_values = ["n", "no", "f", "false", "0"]
  if v in true_values:
    return True
  elif v in false_values:
    return False
  else:
    raise ValueError(f"Invalid value '{v}'!")


@dataclasses.dataclass
class MaxTextModel:
  """A dataclass for representing a MaxText model configuration for benchmarking.

  Attributes:
    model_name: The user-facing name of the model configuration.
    model_type: The specific model variant to be run (e.g., '7b', '13b').
    tuning_params: A dictionary of hyperparameters and settings to override
      the base configuration.
    xla_flags: A string of XLA flags to be used for the model compilation.
    pathways_tuning_params: An optional dictionary of tuning parameters specific
      to Pathways execution.
    pathways_xla_flag_options: An optional dictionary to customize XLA flags
      for Pathways, allowing for additions or removals.
  """

  model_name: str
  model_type: str
  tuning_params: dict[str, typing.Any]
  xla_flags: str

  # Additional pathways tuning params as necessary. Adding
  # enable_single_controller=True to pathways_tuning_params is not necessary.
  pathways_tuning_params: dict[str, typing.Any] = None

  # XLA flags for pathways, if different from the default. Some flags may not
  # be supported by pathways e.g. "--2a886c8_chip_config_name".
  pathways_xla_flag_options: dict[str, typing.Any] = None


# Run this for new definitions that should be part of the library.
def _add_to_model_dictionary(model_dictionary: dict[str, MaxTextModel], maxtext_model: MaxTextModel) -> MaxTextModel:
  model_dictionary[maxtext_model.model_name.replace("-", "_")] = maxtext_model
  return maxtext_model
