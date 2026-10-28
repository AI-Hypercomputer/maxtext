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

"""m3 model garden: one self-contained directory per model family.

Each family owns its architecture end to end:

    models/
        qwen3/
            modeling_qwen3.py  # attention, MLP, decoder layer, model

This module holds the single dispatch point (create_model) and the model
registry. There is no shared Decoder, no shared attention class, and no
model-name branching.
"""

from collections.abc import Callable
from maxtext.m3.models.qwen3.modeling_qwen3 import create_qwen3_model  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next import qwen3_8_flops  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_8_flash_next.modeling_qwen3_8_flash_next import create_qwen3_8_flash_next_model  # pylint: disable=no-name-in-module
from maxtext.m3.models.qwen3_vl.modeling_qwen3_vl import create_qwen3_vl_model  # pylint: disable=no-name-in-module


def create_glm5_3_flash_model(config, mesh=None, *, rngs, **kwargs):
  """Lazy factory for GLM-5.3-Flash to avoid eager tokamax import on CI."""
  # pylint: disable=import-outside-toplevel
  if rngs is None:
    raise ValueError("rngs must be provided as a non-None nnx.Rngs instance.")
  from maxtext.m3.models.glm5_3_flash import modeling_glm5_3_flash  # pylint: disable=g-import-not-at-top

  return modeling_glm5_3_flash.create_glm5_3_flash_model(config, mesh=mesh, rngs=rngs, **kwargs)


MODEL_REGISTRY: dict[str, Callable] = {
    "glm5.3-flash": create_glm5_3_flash_model,
    "qwen3-0.6b": create_qwen3_model,
    "qwen3-vl-2b": create_qwen3_vl_model,
    "qwen3.8-flash-next": create_qwen3_8_flash_next_model,
    # Debug-only mini variant of qwen3.8-flash-next: same architecture and constructor, shrunk to
    # 4 layers / 1 cycle (configs/models/qwen3.8-flash-next-mini.yml). Used only by CPU/CI unit tests.
    "qwen3.8-flash-next-mini": create_qwen3_8_flash_next_model,
}


def create_model(config, mesh, **kwargs):
  """Dispatches to a registered m3 constructor using driver configuration."""
  if config.model_name not in MODEL_REGISTRY:
    raise ValueError(
        f"Model {config.model_name!r} does not support the m3 backend (use_m3_model=True). "
        f"Registered models: {', '.join(sorted(MODEL_REGISTRY))}."
    )
  return MODEL_REGISTRY[config.model_name](config, mesh, **kwargs)


def calculate_glm5_3_flash_tflops_per_device(config, log: bool = True):
  """Lazy wrapper for GLM-5.3-Flash TFLOPs calculation."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from maxtext.m3.models.glm5_3_flash import glm5_flops

  return glm5_flops.calculate_glm5_3_flash_tflops_per_device(config, log=log)


MODEL_FLOPS_REGISTRY: dict[str, Callable] = {
    "glm5.3-flash": calculate_glm5_3_flash_tflops_per_device,
    "qwen3.8-flash-next": qwen3_8_flops.calculate_qwen3_8_flash_next_tflops_per_device,
    "qwen3.8-flash-next-mini": qwen3_8_flops.calculate_qwen3_8_flash_next_tflops_per_device,
}


def calculate_tflops_per_device(config, log: bool = True):
  """Calculates per-device training TFLOPs for registered m3 models."""
  if config.model_name not in MODEL_FLOPS_REGISTRY:
    raise ValueError(
        "TFLOPs calculation not implemented for m3 model"
        f" {config.model_name!r}. Registered models:"
        f" {', '.join(sorted(MODEL_FLOPS_REGISTRY))}."
    )
  return MODEL_FLOPS_REGISTRY[config.model_name](config, log=log)


def configure_glm5_3_flash_engine(engine):
  """Lazy wrapper for GLM-5.3-Flash training engine configuration."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from maxtext.m3.models.glm5_3_flash import glm5_train

  return glm5_train.configure_glm5_engine(engine)


MODEL_ENGINE_CONFIG_REGISTRY: dict[str, Callable] = {
    "glm5.3-flash": configure_glm5_3_flash_engine,
}


def configure_engine(engine):
  """Applies model-specific MaxTextTrainingEngine configuration for registered m3 models."""
  model_name = engine._config.model_name  # pylint: disable=protected-access
  if model_name in MODEL_ENGINE_CONFIG_REGISTRY:
    return MODEL_ENGINE_CONFIG_REGISTRY[model_name](engine)
  return engine
