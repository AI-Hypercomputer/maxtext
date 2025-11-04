
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional, Union

from absl import logging

# Assuming get_relative_import_files is in the same module or a utils module
from .dynamic_module_utils import get_relative_import_files


def custom_object_save(
    obj: Any, folder: Union[str, os.PathLike], config: Optional[dict] = None
) -> list[str]:
  """Save the modeling files corresponding to a custom model/configuration/tokenizer etc. in a given folder.

  Optionally adds the proper fields in a config.

  Args:
    obj: The object for which to save the module files.
    folder: The folder where to save.
    config: A PretrainedConfig or dictionary, optional, a config in which to
      register the auto_map corresponding to this custom object.

  Returns:
    The list of files saved.
  """
  if obj.__module__ == "__main__":
    logging.warning(
        "We can't save the code defining %s in %s as it's been defined in"
        " __main__. You should put this code in a separate module so we can"
        " include it in the saved folder and make it easier to share.",
        obj,
        folder,
    )
    return

  def _set_auto_map_in_config(_config):
    module_name = obj.__class__.__module__
    last_module = module_name.split(".")[-1]
    full_name = f"{last_module}.{obj.__class__.__name__}"
    # Special handling for tokenizers
    if "Tokenizer" in full_name:
      slow_tokenizer_class = None
      fast_tokenizer_class = None
      if obj.__class__.__name__.endswith("Fast"):
        # Fast tokenizer: we have the fast tokenizer class and we may have the
        # slow one has an attribute.
        fast_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"
        if getattr(obj, "slow_tokenizer_class", None) is not None:
          slow_tokenizer = getattr(obj, "slow_tokenizer_class")
          slow_tok_module_name = slow_tokenizer.__module__
          last_slow_tok_module = slow_tok_module_name.split(".")[-1]
          slow_tokenizer_class = (
              f"{last_slow_tok_module}.{slow_tokenizer.__name__}"
          )
      else:
        # Slow tokenizer: no way to have the fast class
        slow_tokenizer_class = f"{last_module}.{obj.__class__.__name__}"

      full_name = (slow_tokenizer_class, fast_tokenizer_class)

    if isinstance(_config, dict):
      auto_map = _config.get("auto_map", {})
      auto_map[obj._auto_class] = full_name
      _config["auto_map"] = auto_map
    elif getattr(_config, "auto_map", None) is not None:
      _config.auto_map[obj._auto_class] = full_name
    else:
      _config.auto_map = {obj._auto_class: full_name}

  # Add object class to the config auto_map
  if isinstance(config, (list, tuple)):
    for cfg in config:
      _set_auto_map_in_config(cfg)
  elif config is not None:
    _set_auto_map_in_config(config)

  result = []
  # Copy module file to the output folder.
  object_file = sys.modules[obj.__module__].__file__
  dest_file = Path(folder) / (Path(object_file).name)
  shutil.copy(object_file, dest_file)
  result.append(str(dest_file))

  # Gather all relative imports recursively and make sure they are copied as
  # well.
  for needed_file in get_relative_import_files(object_file):
    dest_file = Path(folder) / (Path(needed_file).name)
    shutil.copy(needed_file, dest_file)
    result.append(str(dest_file))

  return result

import functools
from typing import Any


def get_module_from_name(module: Any, tensor_name: str) -> tuple[Any, str]:
  if "." in tensor_name:
    module_name, tensor_name = tensor_name.rsplit(".", 1)
    module = functools.reduce(getattr, module_name.split("."), module)
  return module, tensor_name

from typing import TYPE_CHECKING

from jax import Array

# Used from: generated_code.Qwen3MoeForCausalLM.dynamic_module_utils.get_module_from_name
from ..dynamic_module_utils import get_module_from_name

if TYPE_CHECKING:
    from .modeling_flax_utils import FlaxPreTrainedModel


def _load_parameter_into_model(model: "FlaxPreTrainedModel", param_name: str, tensor: Array):
    """Cast a single parameter `param_name` into the `model`, with value `tensor`."""
    module, param_type = get_module_from_name(model, param_name)
    # This will check potential shape mismatch if skipped before
    setattr(module, param_type, tensor)
