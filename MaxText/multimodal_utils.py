import os
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image


_GEMMA_DEFAULT_IMAGE_SIZE = 896
_GEMMA_IMAGE_MEAN = (127.5,) * 3
_GEMMA_IMAGE_STD = (127.5,) * 3


def load_image_from_path(image_path):
  """Loads an image from a given file path and returns a jnp.array."""
  if not os.path.isfile(image_path):
    print(f"Error: Image not found at path '{image_path}'")
    return None
  try:
    image = Image.open(image_path).convert("RGB")
    image.load()  # Load image data to catch errors early
    return jnp.asarray(np.array(image))
  except (IOError, OSError) as e:
    print(f"Error loading image from '{image_path}': {e}")
    return None


def _normalize_images(images):
  """Normalize the image to zero mean and unit variance.

  In order to change the image mean and std, we need to change the _IMAGE_MEAN
  and _IMAGE_STD global constants in this file.

  Args:
    images: The images to normalize.

  Returns:
    The normalized images.
  """
  images -= jnp.asarray(_GEMMA_IMAGE_MEAN)
  images /= jnp.asarray(_GEMMA_IMAGE_STD)
  return images


def pre_process_gemma3_image(image):
  """Performs a bi-linear resize (with anti-aliasing) and normalizes the image."""
  image_shape = (_GEMMA_DEFAULT_IMAGE_SIZE, _GEMMA_DEFAULT_IMAGE_SIZE, 3)
  image = jax.image.resize(
      image,
      shape=image_shape,
      method="bilinear",
      antialias=True,
  )
  image = _normalize_images(image)
  image = jnp.clip(image, -1, 1)
  return image


def pre_process_image(
    image,
    model_name
):
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
