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

"""Multimodal data preprocessor router."""

import numpy as np
from PIL import Image

from MaxText import multimodal_utils  # TODO(hengtaoguo): deprecate this file and refactor to MaxText/multimodal/utils.py


def _normalize_images_to_numpy(images):
  """Convert images to numpy arrays and ensure RGB format.

  Handles both PIL Images and numpy arrays from the data pipeline.
  All model-specific preprocessing functions expect numpy arrays.

  Args:
    images: Single image (PIL or numpy) or list of images

  Returns:
    List of numpy arrays in RGB format
  """
  # Ensure images is a list
  if not isinstance(images, list):
    images = [images]

  normalized_images = []
  for img in images:
    # Convert PIL Image to numpy array
    if isinstance(img, Image.Image):
      # Ensure RGB mode
      img = multimodal_utils.convert_to_RGB(img)
      # Convert to numpy array
      img = np.asarray(img)
    elif isinstance(img, np.ndarray):
      # Already numpy array, no conversion needed
      pass
    else:
      raise TypeError(f"Unexpected image type: {type(img)}. Expected PIL.Image or np.ndarray")

    normalized_images.append(img)

  return normalized_images


def preprocess_mm_data(
    config,
    images=None,
    image_path: str | None = None,
    video_path: str | None = None,
    audio_path: str | None = None,
):
  """Preprocesses multimodal data based on the provided configuration.
  Routes to the appropriate preprocessing function based on the model name.

  Args:
    config: A config object containing model-specific parameters (patch sizes, etc.)
    images: PIL Image(s) or numpy array(s) from the data pipeline
    image_path: Optional path to image file(s), comma-separated for multiple images
    video_path: Optional path to video file
    audio_path: Optional path to audio file

  Returns:
    A `PreprocessorOutput` object containing the processed multimodal data.
  """
  processor_outputs = multimodal_utils.PreprocessorOutput()

  # Handle image loading and normalization (common for all models)
  if images is not None or image_path:
    # Load images from path if not provided
    if not images and image_path:
      images = [multimodal_utils.load_image_from_path(p) for p in image_path.split(",")]

    # Normalize images to numpy arrays (convert PIL to numpy, ensure RGB)
    if images is not None:
      images = _normalize_images_to_numpy(images)
  elif not video_path and not audio_path:
    # No multimodal data provided at all
    return processor_outputs

  # Route to model-specific preprocessing
  if config.model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    processor_outputs = multimodal_utils.pre_process_gemma3_image(images)
  elif config.model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    processor_outputs = multimodal_utils.pre_process_llama4_image(images)
  elif config.model_name in ["qwen3-omni-30b-a3b"]:
    from MaxText.multimodal.qwen3_omni_processor import preprocess_mm_data_qwen3_omni  # pylint: disable=import-outside-toplevel

    processor_outputs = preprocess_mm_data_qwen3_omni(
        config=config,
        images=images,
        video_path=video_path,
        audio_path=audio_path,
    )
  else:
    raise ValueError(f"Model {config.model_name} not supported for multimodal preprocessing.")

  return processor_outputs
