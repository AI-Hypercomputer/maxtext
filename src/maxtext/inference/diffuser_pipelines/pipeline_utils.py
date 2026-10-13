# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-agnostic diffusion pipeline utilities and base classes."""

from typing import List, Union

from diffusers.utils import BaseOutput
import flax
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

INDEX_FILE = "diffusion_flax_model.bin"


@flax.struct.dataclass
class FlaxImagePipelineOutput(BaseOutput):
  """
  Output class for image pipelines.

  Args:
      images (`List[PIL.Image.Image]` or `np.ndarray`)
          List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
          num_channels)`.
  """

  images: Union[List[Image.Image], np.ndarray]


class FlaxDiffusionPipeline:
  r"""
  Base class for Flax-based pipelines.

  Components (models, schedulers, processors) are loaded by each pipeline from the converted MaxText
  Orbax checkpoint, so this base class only provides shared helpers:

      - enable/disable the progress bar for the denoising iteration
      - convert NumPy images to PIL
  """

  @staticmethod
  def numpy_to_pil(images):
    """
    Convert a NumPy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
      images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
      # special case for grayscale (single channel) images
      pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
      pil_images = [Image.fromarray(image) for image in images]

    return pil_images

  # TODO: make it compatible with jax.lax
  def progress_bar(self, iterable):
    if not hasattr(self, "_progress_bar_config"):
      self._progress_bar_config = {}
    elif not isinstance(self._progress_bar_config, dict):
      raise ValueError(f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}.")

    return tqdm(iterable, **self._progress_bar_config)

  def set_progress_bar_config(self, **kwargs):
    self._progress_bar_config = kwargs
