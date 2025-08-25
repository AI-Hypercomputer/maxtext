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

__author__ = "Google LLC"
__version__ = "2025.07.24"
__description__ = (
    "MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax and "
    "targeting Google Cloud TPUs and GPUs for training and **inference."
)


# maxtext/__init__.py

from collections.abc import Sequence
from typing import overload

from flax import nnx
from MaxText.common_types import MODEL_MODE_TRAIN
from MaxText.layers import nnx_wrappers
from MaxText import maxtext_utils
from MaxText import train_utils
from MaxText import pyconfig
from MaxText.layers import models
from MaxText import checkpointing as checkpointing  # pylint: disable=useless-import-alias
import jax
from jax.sharding import Mesh

Transformer = models.Transformer
transformer_as_linen = models.transformer_as_linen

@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
) -> nnx_wrappers.ToLinen | models.TransformerLinenPure: ...
@overload
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs,
) -> Transformer: ...
def from_config(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
    *,
    model_mode: str = MODEL_MODE_TRAIN,
    rngs: nnx.Rngs | None = None,
) -> nnx_wrappers.ToLinen | models.TransformerLinenPure | Transformer:
  """Instantiate a MaxText model.

  This function creates a model instance from config but does not load any states.

  Args:
      config: Config object.

  Returns:
      Transformer: The loaded model instance (only the model)

  Example:
      model = from_config(config)
  """
  devices_array = maxtext_utils.create_device_mesh(config, devices)
  mesh = Mesh(devices_array, config.mesh_axes)
  model = train_utils.create_model(config, mesh, model_mode=model_mode, rngs=rngs)

  # Return only the model
  return model
