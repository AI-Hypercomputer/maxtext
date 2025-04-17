from PIL import Image
import numpy as np
import jax
import jax.numpy as jnp


_DEFAULT_IMAGE_SIZE = 896

def load_PIL_image(image_path):
  """Loads an image into a PIL image."""
  image = Image.open(image_path).convert("RGB")
  return image


def pre_process_image(
    image,
    *,
    image_shape: tuple[int, int, int],
):
  """Pre-process image.

  Performs a bi-linear resize (with anti-aliasing) and normalizes the image.

  Args:
    image: The image to pre-process.
    image_shape: The target shape (h, w, c) of the image (default to (896, 896,
      3)).

  Returns:
    The pre-processed image.
  """
  # TODO(epot): All inputs are expected to have been jpeg encoded with
  # TensorFlow.
  # tf.image.decode_jpeg(tf.io.encode_jpeg(image), channels=3)

  image = jax.image.resize(
      image,
      shape=image_shape,
      method="bilinear",
      antialias=True,
  )
  image = _normalize_images(image)
  image = jnp.clip(image, -1, 1)
  return image