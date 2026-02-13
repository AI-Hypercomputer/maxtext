# Copyright 2023–2026 Google LLC
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

from maxtext.multimodal import utils as mm_utils


def preprocess_mm_data(config):
  """Preprocesses multimodal data based on the provided configuration.
  Routes to the appropriate preprocessing function based on the model name.

  Args:
    config: A `pyconfig.Config` object containing configuration parameters.

  Returns:
    A `PreprocessorOutput` object containing the processed multimodal data.
  """
  processor_outputs = mm_utils.PreprocessorOutput()

  if config.model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    from maxtext.multimodal.processor_gemma3 import preprocess_mm_data_gemma3  # pylint: disable=import-outside-toplevel

    images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]
    processor_outputs = preprocess_mm_data_gemma3(images)
  elif config.model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    from maxtext.multimodal.processor_llama4 import preprocess_mm_data_llama4  # pylint: disable=import-outside-toplevel

    images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]
    processor_outputs = preprocess_mm_data_llama4(images)
  elif config.model_name in ["qwen3-omni-30b-a3b"]:
    from maxtext.multimodal.processor_qwen3_omni import preprocess_mm_data_qwen3_omni  # pylint: disable=import-outside-toplevel

    processor_outputs = preprocess_mm_data_qwen3_omni(config)
  else:
    raise ValueError(f"Model {config.model_name} not supported for multimodal preprocessing.")

  return processor_outputs


def preprocess_image_for_training(image, model_name):
  """Preprocesses a single image for training based on the model name."""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    from maxtext.multimodal.processor_gemma3 import preprocess_mm_data_gemma3  # pylint: disable=import-outside-toplevel

    return preprocess_mm_data_gemma3(image)
  elif model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    from maxtext.multimodal.processor_llama4 import preprocess_mm_data_llama4  # pylint: disable=import-outside-toplevel

    return preprocess_mm_data_llama4(image)
  else:
    raise ValueError(f"Model {model_name} not supported for image preprocessing.")


def get_image_offsets(model_name, processor_output: mm_utils.PreprocessorOutput | None):
  """Get the increase in total token count after inserting image token placeholders"""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    from maxtext.multimodal.processor_gemma3 import get_image_offsets_gemma3  # pylint: disable=import-outside-toplevel

    return get_image_offsets_gemma3(processor_output)
  elif model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    from maxtext.multimodal.processor_llama4 import get_image_offsets_llama4  # pylint: disable=import-outside-toplevel

    return get_image_offsets_llama4(processor_output)
  else:
    return 0


def reformat_prompt(prompt, image_placeholder, model_name, num_images):
  """Reformat prompt for different models."""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    from maxtext.multimodal.processor_gemma3 import reformat_prompt_gemma3  # pylint: disable=import-outside-toplevel

    return reformat_prompt_gemma3(prompt, image_placeholder, num_images)
  elif model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    from maxtext.multimodal.processor_llama4 import reformat_prompt_llama4  # pylint: disable=import-outside-toplevel

    return reformat_prompt_llama4(prompt, image_placeholder, num_images)
  else:
    return prompt


def reformat_response(response, model_name):
  """Reformat response for different models."""
  if model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    formatted_response = f"{response}<|eot|>"
    return formatted_response
  elif model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    formatted_response = f"{response}<end_of_turn>"
    return formatted_response
  else:
    return response


def prepare_text_for_image_fusion(texts, model_name, processor_output=None):
  """Prepare text by adding extra tokens for image fusion based on the model."""
  if model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    from maxtext.multimodal.processor_gemma3 import add_extra_tokens_for_images_gemma3  # pylint: disable=import-outside-toplevel

    return add_extra_tokens_for_images_gemma3(texts, max_num_images=processor_output.num_images)
  elif model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    from maxtext.multimodal.processor_llama4 import add_extra_tokens_for_images_llama4  # pylint: disable=import-outside-toplevel

    return add_extra_tokens_for_images_llama4(texts, processor_output)
  else:
    raise ValueError(f"Model {model_name} does not support multimodal inference.")


def get_dummy_image_shape_for_init(model_name, batch_size=1, num_image_per_sequence=1):
  """Return the shape of the dummy image for specific model's initialization."""
  image_shape = ()
  if model_name.startswith("gemma3"):
    from maxtext.multimodal.processor_gemma3 import get_dummy_image_shape_for_init_gemma3  # pylint: disable=import-outside-toplevel

    image_shape = get_dummy_image_shape_for_init_gemma3(batch_size, num_image_per_sequence)
  elif model_name.startswith("llama4"):
    from maxtext.multimodal.processor_llama4 import get_dummy_image_shape_for_init_llama4  # pylint: disable=import-outside-toplevel

    image_shape = get_dummy_image_shape_for_init_llama4(batch_size, num_image_per_sequence)
  elif model_name.startswith("qwen3-omni-30b-a3b"):
    from maxtext.multimodal.processor_qwen3_omni import get_dummy_image_shape_for_init_qwen3_omni  # pylint: disable=import-outside-toplevel

    image_shape = get_dummy_image_shape_for_init_qwen3_omni(batch_size)
  return image_shape


def get_dummy_audio_shape_for_init(config):
  """Return the shape of the dummy audio for specific model's initialization.

  Args:
    config: Model configuration containing audio parameters

  Returns:
    Tuple representing audio shape: (batch, num_mel_bins, audio_length)
    Returns empty tuple if audio is not configured for the model
  """
  audio_shape = ()
  if config.model_name.startswith("qwen3-omni"):
    from maxtext.multimodal.processor_qwen3_omni import get_dummy_audio_shape_for_init_qwen3_omni  # pylint: disable=import-outside-toplevel

    audio_shape = get_dummy_audio_shape_for_init_qwen3_omni(config)

  return audio_shape


def get_bidirectional_mask_vision(config, decoder_input_tokens):
  """Get the bidirectional mask for specific models."""
  bidirectional_mask_vision = None
  if config.model_name in ["gemma3-4b", "gemma3-12b", "gemma3-27b"]:
    from maxtext.multimodal.processor_gemma3 import GEMMA_TOKEN_PLACEHOLDER  # pylint: disable=import-outside-toplevel

    bidirectional_mask_vision = decoder_input_tokens == GEMMA_TOKEN_PLACEHOLDER
  elif config.model_name in ["llama4-17b-16e", "llama4-17b-128e"]:
    from maxtext.multimodal.processor_llama4 import LLAMA4_PATCH_TOKEN  # pylint: disable=import-outside-toplevel

    bidirectional_mask_vision = decoder_input_tokens == LLAMA4_PATCH_TOKEN
  elif config.model_name in ["qwen3-omni-30b-a3b"]:
    from maxtext.multimodal.processor_qwen3_omni import QWEN3_OMNI_IMAGE_TOKEN, QWEN3_OMNI_VIDEO_TOKEN  # pylint: disable=import-outside-toplevel

    # Create bidirectional_mask for vision/video token merging
    bidirectional_mask_vision = (decoder_input_tokens == QWEN3_OMNI_IMAGE_TOKEN) | (
        decoder_input_tokens == QWEN3_OMNI_VIDEO_TOKEN
    )
    # Create image/video mask for deepstack visual embedding injection
  return bidirectional_mask_vision


def get_bidirectional_mask_audio(config, decoder_input_tokens):
  """Get the bidirectional mask for specific models."""
  bidirectional_mask_audio = None
  if config.model_name in ["qwen3-omni-30b-a3b"]:
    from maxtext.multimodal.processor_qwen3_omni import QWEN3_OMNI_AUDIO_TOKEN  # pylint: disable=import-outside-toplevel

    # Create bidirectional_mask for audio token merging
    bidirectional_mask_audio = decoder_input_tokens == QWEN3_OMNI_AUDIO_TOKEN
  return bidirectional_mask_audio
