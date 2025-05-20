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
def _add_to_model_dictionary(
    model_dictionary: dict[str, MaxTextModel], maxtext_model: MaxTextModel
) -> MaxTextModel:
  print(maxtext_model.model_name.replace("-", "_"))
  model_dictionary[maxtext_model.model_name.replace("-", "_")] = maxtext_model
  return maxtext_model
