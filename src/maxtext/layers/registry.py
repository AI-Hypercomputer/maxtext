# Copyright 2023–2026 Google LLC
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

"""Registry for model plugins in MaxText."""

import dataclasses
from typing import Dict, List, Type, Any, TYPE_CHECKING

if TYPE_CHECKING:
  from maxtext.layers import strategies


@dataclasses.dataclass
class ModelPlugin:
  """Container for a model plugin's components."""
  layer_classes: List[Type[Any]]  # List of nn.Module or nnx.Module classes
  strategy_class: Type["strategies.ModelStrategy"]


_MODEL_PLUGIN_REGISTRY: Dict[str, ModelPlugin] = {}


def register_model(name: str, strategy_class: Type["strategies.ModelStrategy"]):
  """Decorator to register a model's layer classes and its strategy.

  This enables "Top-Level Model" definitions by associating execution logic (strategy)
  directly with the model's layers, reducing the need for hardcoded logic in decoders.py.

  Args:
    name: The name of the model (e.g. from DecoderBlockType).
    strategy_class: The ModelStrategy class to use for this model.
  """
  def decorator(cls_or_classes):
    # Support both single class and list/tuple of classes
    if isinstance(cls_or_classes, (list, tuple)):
      classes = list(cls_or_classes)
    else:
      classes = [cls_or_classes]

    _MODEL_PLUGIN_REGISTRY[name] = ModelPlugin(
        layer_classes=classes,
        strategy_class=strategy_class
    )
    return cls_or_classes

  return decorator


def get_model_plugin(name: str) -> ModelPlugin:
  """Retrieves a registered model plugin by name.

  Args:
    name: The name of the model plugin to retrieve.

  Returns:
    The registered ModelPlugin.

  Raises:
    ValueError: If the model name is not registered.
  """
  if name not in _MODEL_PLUGIN_REGISTRY:
    raise ValueError(f"Model plugin '{name}' not found in registry. "
                     f"Available plugins: {list_registered_models()}")
  return _MODEL_PLUGIN_REGISTRY[name]


def list_registered_models() -> List[str]:
  """Returns a list of names of all registered models."""
  return list(_MODEL_PLUGIN_REGISTRY.keys())
