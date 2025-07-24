"""
 Copyright 2024 Google LLC

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

import dataclasses
import typing


@dataclasses.dataclass
class MaxTextModel:
  """A dataclass to store all configurations for a given MaxText model.

  This structure encapsulates the necessary parameters for defining and running
  a model, including its name, type, specific tuning parameters, and XLA
  compiler flags for both standard and Pathways execution environments.

  Attributes:
    model_name: The unique identifier for the model.
    model_type: The category or architecture of the model (e.g., 'llama2-7b').
    tuning_params: A dictionary of hyperparameters and settings specific to
      the model's training and inference behavior.
    xla_flags: A string of command-line flags to pass to the XLA compiler for
      standard execution.
    pathways_tuning_params: An optional dictionary of tuning parameters
      specific to the Pathways execution environment.
    pathways_xla_flag_options: An optional dictionary of XLA flags for Pathways
      if they differ from the default `xla_flags`.
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
  """Adds a MaxTextModel configuration to a dictionary with a standardized key.

  This helper function modifies the provided dictionary in-place by adding the
  `maxtext_model`. The dictionary key is derived from the model's name by
  replacing hyphens with underscores.

  Args:
    model_dictionary: The dictionary to which the model configuration will be
      added.
    maxtext_model: The `MaxTextModel` object to add to the dictionary.

  Returns:
    The `MaxTextModel` object that was added to the dictionary.
  """
  print(maxtext_model.model_name.replace("-", "_"))
  model_dictionary[maxtext_model.model_name.replace("-", "_")] = maxtext_model
  return maxtext_model
