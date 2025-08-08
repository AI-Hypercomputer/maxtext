# SPDX-License-Identifier: Apache-2.0

__author__ = "Google LLC"
__version__ = "2025.07.24"
__description__ = (
    "MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax and "
    "targeting Google Cloud TPUs and GPUs for training and **inference."
)


# maxtext/__init__.py

from collections.abc import Sequence

from MaxText import maxtext_utils
from MaxText import train_utils
from MaxText import pyconfig
from MaxText.layers import models
import jax
from jax.sharding import Mesh

Transformer = models.Transformer


def from_pretrained(
    config: pyconfig.HyperParameters,
    devices: Sequence[jax.Device] | None = None,
) -> Transformer:
  """Load a pretrained MaxText model from checkpoint.

  This function loads a model from a checkpoint.

  Args:
      config: Config object.

  Returns:
      Transformer: The loaded model instance (only the model)

  Example:
      model = from_pretrained(config)
  """
  devices_array = maxtext_utils.create_device_mesh(config, devices)
  mesh = Mesh(devices_array, config.mesh_axes)
  model = train_utils.create_model(config, mesh)

  # Return only the model
  return model
