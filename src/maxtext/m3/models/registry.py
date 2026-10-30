# Copyright 2026 Google LLC
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

"""Static Python-first Model Garden configuration registry."""

from collections.abc import Callable
from typing import Any

from maxtext.m3.models.glm5_3_flash.glm5_config import Glm5ModelConfig
from maxtext.m3.models.model_config import ModelConfig
from maxtext.m3.models.qwen3.qwen3_config import Qwen3Config

ModelConfigFactory = Callable[..., ModelConfig]

MODEL_GARDEN_REGISTRY: dict[str, ModelConfigFactory] = {
    "glm5.3-flash": Glm5ModelConfig.glm5_3_flash,
    "qwen3-0.6b": Qwen3Config.qwen3_0_6b,
}


def normalize_model_name(model_name: str) -> str:
  """Strips the `-Instruct` suffix matching `pyconfig.py` lookup behavior."""
  return model_name.replace("-Instruct", "")


def get_model_config(
    model_name: str,
    expected_cls: type[ModelConfig] = ModelConfig,
    **overrides: Any,
) -> ModelConfig:
  """Returns a frozen `ModelConfig` for `model_name`, applying strictly validated `overrides`."""
  resolved = normalize_model_name(model_name)
  if resolved not in MODEL_GARDEN_REGISTRY:
    available = ", ".join(sorted(MODEL_GARDEN_REGISTRY))
    raise KeyError(f"Unknown m3 model {model_name!r}. Available models: {available}.")

  factory = MODEL_GARDEN_REGISTRY[resolved]
  if expected_cls is not ModelConfig:
    base_cfg = factory()
    if not isinstance(base_cfg, expected_cls):
      raise TypeError(
          f"Model {model_name!r} resolves to {type(base_cfg).__name__}, which is not a {expected_cls.__name__}."
      )
    if not overrides:
      return base_cfg
  return factory(**overrides)


def get_pyconfig_model_dict(model_name: str) -> dict[str, Any] | None:
  """Returns transitional `MaxTextConfig` field overrides for a YAML-free registered model, or `None`."""
  resolved = normalize_model_name(model_name)
  if resolved not in MODEL_GARDEN_REGISTRY:
    return None
  return MODEL_GARDEN_REGISTRY[resolved]().to_pyconfig_dict()
