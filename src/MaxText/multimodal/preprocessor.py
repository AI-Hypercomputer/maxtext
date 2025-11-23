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

from MaxText import multimodal_utils  # TODO(hengtaoguo): deprecate this file and refactor to MaxText/multimodal/utils.py


def preprocess_mm_data(
    model_name: str,
    config,
    image_path: str | None = None,
    video_path: str | None = None,
    audio_path: str | None = None,
):
  """Preprocesses multimodal data based on the provided configuration.
  Routes to the appropriate preprocessing function based on the model name.

  Args:
    model_name: Name of the model (e.g., "qwen3-omni-30b-a3b")
    config: A config object containing model-specific parameters (patch sizes, etc.)
    image_path: Optional path to image file(s), comma-separated for multiple images
    video_path: Optional path to video file
    audio_path: Optional path to audio file

  Returns:
    A `PreprocessorOutput` object containing the processed multimodal data.
  """
  processor_outputs = multimodal_utils.PreprocessorOutput()

  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    if not image_path:
      return processor_outputs
    images = [multimodal_utils.load_image_from_path(p) for p in image_path.split(",")]
    processor_outputs = multimodal_utils.pre_process_gemma3_image(images)
  elif model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    if not image_path:
      return processor_outputs
    images = [multimodal_utils.load_image_from_path(p) for p in image_path.split(",")]
    processor_outputs = multimodal_utils.pre_process_llama4_image(images)
  elif model_name in ["qwen3-omni-30b-a3b"]:
    from MaxText.multimodal.qwen3_omni_processor import preprocess_mm_data_qwen3_omni  # pylint: disable=import-outside-toplevel

    processor_outputs = preprocess_mm_data_qwen3_omni(
        config=config,
        image_path=image_path,
        video_path=video_path,
        audio_path=audio_path,
    )
  else:
    raise ValueError(f"Model {model_name} not supported for multimodal preprocessing.")

  return processor_outputs
