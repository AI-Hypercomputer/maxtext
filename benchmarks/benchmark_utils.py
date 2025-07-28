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
import math
import typing

@dataclasses.dataclass
class MaxTextModel:
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
