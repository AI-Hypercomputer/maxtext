"""
Copyright 2025 Google LLC

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

"""Utils needed by multimodal pipelines for image processing."""

import os
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image


_GEMMA_DEFAULT_IMAGE_SIZE = 896
_GEMMA_IMAGE_MEAN = (127.5,) * 3
_GEMMA_IMAGE_STD = (127.5,) * 3
_NUM_IMAGE_CHANNELS = 3


def load_image_from_path(image_path):
  """Loads an image from a given file path and returns a jnp.array."""
  if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found at path {image_path}. Please specify a valid image path")
  try:
    image = Image.open(image_path).convert("RGB")
    image.load()  # Load image data to catch errors early
    return jnp.asarray(np.array(image))
  except (IOError, OSError) as e:
    raise IOError(f"Error loading image from {image_path}")


def _normalize_images(images, mean, std):
  """Normalize the image to zero mean and unit variance.
  Change the image mean and std based on parameters mean and std.
  Args:
    images: The images to normalize.
    mean: tuple[float, float, float].
    std: tuple[float, float, float].
  Returns:
    The normalized images.
  """
  images -= jnp.asarray(mean)
  images /= jnp.asarray(std)
  return images


def pre_process_gemma3_image(image):
  """Performs a bi-linear resize (with anti-aliasing) and normalizes the image."""
  image_shape = (_GEMMA_DEFAULT_IMAGE_SIZE, _GEMMA_DEFAULT_IMAGE_SIZE, _NUM_IMAGE_CHANNELS)
  image = jax.image.resize(
      image,
      shape=image_shape,
      method="bilinear",
      antialias=True,
  )
  image = _normalize_images(image, mean=_GEMMA_IMAGE_MEAN, std=_GEMMA_IMAGE_STD)
  image = jnp.clip(image, -1, 1)
  return image


def pre_process_image(image, model_name):
  """Pre-process image according to different model's requirements.
  Args:
    image: The jnp.array image [H, W, C] to pre-process.
    model_name: The config.model_name that specifies the image preprocess ways.
  Returns:
    The pre-processed image in jnp.array [H, W, C].
  """
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    return pre_process_gemma3_image(image)
  else:
    raise ValueError(f"Model {model_name} does not support multimodal inference.")
