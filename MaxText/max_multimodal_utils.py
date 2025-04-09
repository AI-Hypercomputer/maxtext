import numpy as np
from PIL import Image

_IMAGE_MEAN = (127.5,) * 3
_IMAGE_STD = (127.5,) * 3
_DEFAULT_IMAGE_SIZE = 896  # SigLip expected input image size
_DEFAULT_PATCH_SIZE = 14  # SigLip expected patch size


def normalize_images(images):
  """Normalize the image to zero mean and unit variance.

  In order to change the image mean and std, we need to change the _IMAGE_MEAN
  and _IMAGE_STD global constants in this file.

  Args:
    images: The images to normalize.

  Returns:
    The normalized images.
  """
  images = np.array(images, dtype=np.float32)
  images -= np.asarray(_IMAGE_MEAN)
  images /= np.asarray(_IMAGE_STD)
  return images


def pre_process_image(
    image,
    image_height: int = _DEFAULT_IMAGE_SIZE,
    image_width: int = _DEFAULT_IMAGE_SIZE,
):
  """Pre-process image.

  Performs a bi-linear resize (with anti-aliasing) and normalizes the image.

  Args:
    image: The image to pre-process in PIL obj format.
    image_height: The height of the image (default to 896).
    image_width: The width of the image (default to 896).

  Returns:
    The pre-processed image as np array.
  """
  if not isinstance(image, Image.Image):
    raise ValueError("Input image must be a PIL Image object.")
  
  image = image.convert("RGB")
  image = image.resize((image_width, image_height), Image.Resampling.BILINEAR)
  image = np.array(image)
  image = normalize_images(image)
  image = np.clip(image, -1, 1)
  # return image.tolist()
  return image


def load_image_rgb(image_path):
  """Loads an image from the given path using PIL and converts it to RGB format.
  Args:
    image_path: The path to the image file.
  Returns:
    A PIL Image object in RGB format, or None if an error occurred.
  """
  try:
    img = Image.open(image_path)
    img_rgb = img.convert("RGB")
    return img_rgb
  except FileNotFoundError:
    print(f"Error: Image not found at path: {image_path}")
    return None
  except Exception as e:
    print(f"Error loading or converting image: {e}")
    return None


