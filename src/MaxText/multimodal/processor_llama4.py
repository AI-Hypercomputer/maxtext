from dataclasses import dataclass

import numpy as np
from PIL import Image

from MaxText.multimodal import utils as mm_utils

# Constants for Gemma3-specific processing
GEMMA_DEFAULT_IMAGE_SIZE = 896
GEMMA_IMAGE_MEAN = (127.5,) * 3
GEMMA_IMAGE_STD = (127.5,) * 3


@dataclass
class Gemma3PreprocessorOutput(mm_utils.PreprocessorOutput):
  """Holds the output of Gemma3 image preprocessor.

  Attributes:
    Inherited from `mm_utils.PreprocessorOutput`.
  """

  # Image attributes.
  num_images: int = 0
  pixel_values: None | np.ndarray = None
  pixel_mask: None | np.ndarray = None


def preprocess_mm_data_gemma3(config):
  """Preprocesses multimodal data for Gemma3 models."""
  images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]

  model_name = config.model_name
  print(f"Preprocessing multimodal data for model: {model_name}")

  """Performs a bi-linear resize (with anti-aliasing) and normalizes the image."""
  target_size = (GEMMA_DEFAULT_IMAGE_SIZE, GEMMA_DEFAULT_IMAGE_SIZE)

  images_in, images_out = [], []
  if isinstance(images, np.ndarray):
    images_in.append(images)
  else:
    images_in.extend(images)

  for img in images_in:
    pil_img = Image.fromarray(img)
    resample_method = Image.Resampling.BILINEAR

    # Use a higher quality downsampling filter to approximate antialias=True
    if pil_img.size[0] > target_size[0] or pil_img.size[1] > target_size[1]:
      resample_method = Image.Resampling.LANCZOS

    resized_pil_img = pil_img.resize(target_size, resample=resample_method)
    img = np.asarray(resized_pil_img, dtype=np.float32)
    img = mm_utils.normalize_images(img, mean=GEMMA_IMAGE_MEAN, std=GEMMA_IMAGE_STD)
    img = np.clip(img, -1, 1)
    images_out.append(img)

  processor_output = Gemma3PreprocessorOutput(
      num_images=len(images),
      pixel_values=np.stack(images_out, axis=0).astype(np.float32),  # (N, H, W, C)
  )
  processor_output.num_images = len(images)
  return processor_output
