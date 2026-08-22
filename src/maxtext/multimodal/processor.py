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

# model_name -> (vision_encoder_block, decoder_block) mapping
_MODEL_TO_BLOCKS = {
    # Gemma 3
    "gemma3-4b": ("gemma3", "gemma3"),
    "gemma3-12b": ("gemma3", "gemma3"),
    "gemma3-27b": ("gemma3", "gemma3"),
    # Gemma 4
    "gemma4-26b": ("gemma4", "gemma4"),
    "gemma4-31b": ("gemma4", "gemma4"),
    "gemma4-e2b": ("gemma4", "gemma4_small"),
    "gemma4-e4b": ("gemma4", "gemma4_small"),
    # Llama 4
    "llama4-17b-16e": ("llama4", "llama4"),
    "llama4-17b-128e": ("llama4", "llama4"),
    # Qwen 3 & 3.5
    "qwen3-omni-30b-a3b": ("qwen3_omni", "qwen3_moe"),
    "qwen3-vl-2b": ("qwen3_vl", "qwen3"),
    "qwen3-vl-4b": ("qwen3_vl", "qwen3"),
    "qwen3-vl-30b-a3b": ("qwen3_vl", "qwen3_moe"),
    "qwen3.5-35b-a3b": ("qwen3_5", "qwen3_5"),
    "qwen3.5-397b-a17b": ("qwen3_5", "qwen3_5"),
}


def _get_vision_block(config_or_name):
  """Extract vision encoder architecture from config."""
  # If input is config, extract vision_encoder name
  if hasattr(config_or_name, "vision_encoder_block"):
    block = config_or_name.vision_encoder_block
    block_val = getattr(block, "value", str(block)).lower()
    if block_val != "none":
      return block_val
  # If input is model_name (backward compatibility), find its corresponding encoder
  elif isinstance(config_or_name, str):
    if config_or_name.lower() in _MODEL_TO_BLOCKS:
      return _MODEL_TO_BLOCKS[config_or_name.lower()][0]
  # If non-vision model, return None
  return None


def _get_decoder_block(config_or_name):
  """Extract decoder architecture from config."""
  # If input is config, extract decoder name
  if hasattr(config_or_name, "decoder_block"):
    block = config_or_name.decoder_block
    block_val = getattr(block, "value", str(block)).lower()
    if block_val != "default":
      return block_val
  # If input is model_name (backward compatibility), find its corresponding decoder
  elif isinstance(config_or_name, str):
    if config_or_name.lower() in _MODEL_TO_BLOCKS:
      return _MODEL_TO_BLOCKS[config_or_name.lower()][1]
  # If decoder_block not found or is default, return model_name/default
  return str(getattr(config_or_name, "model_name", config_or_name)).lower()


def preprocess_mm_data(config):
  """Preprocesses multimodal data based on the provided configuration.
  Routes to the appropriate preprocessing function based on the vision architecture.

  Args:
    config: A `pyconfig.Config` object containing configuration parameters.

  Returns:
    A `PreprocessorOutput` object containing the processed multimodal data.
  """
  processor_outputs = mm_utils.PreprocessorOutput()
  vision_block = _get_vision_block(config)

  if vision_block in ["gemma3"]:
    from maxtext.multimodal.processor_gemma3 import preprocess_mm_data_gemma3  # pylint: disable=import-outside-toplevel

    images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]
    processor_outputs = preprocess_mm_data_gemma3(images)
  elif vision_block in ["gemma4"]:
    from maxtext.multimodal.processor_gemma4 import preprocess_mm_data_gemma4  # pylint: disable=import-outside-toplevel

    images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]
    processor_outputs = preprocess_mm_data_gemma4(images)
  elif vision_block in ["llama4"]:
    from maxtext.multimodal.processor_llama4 import preprocess_mm_data_llama4  # pylint: disable=import-outside-toplevel

    images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]
    processor_outputs = preprocess_mm_data_llama4(images)
  elif vision_block in ["qwen3_omni", "qwen3_vl", "qwen3_5"]:
    from maxtext.multimodal.processor_qwen3_omni import preprocess_mm_data_qwen3_omni  # pylint: disable=import-outside-toplevel

    processor_outputs = preprocess_mm_data_qwen3_omni(config)
  else:
    raise ValueError(
        f"Model {config.model_name} (vision block {vision_block}) not supported for multimodal preprocessing."
    )

  return processor_outputs


def preprocess_image_for_training(image, config):
  """Preprocesses a single image for training based on the vision architecture."""
  vision_block = _get_vision_block(config)
  if vision_block in ["gemma3"]:
    from maxtext.multimodal.processor_gemma3 import preprocess_mm_data_gemma3  # pylint: disable=import-outside-toplevel

    return preprocess_mm_data_gemma3(image)
  elif vision_block in ["gemma4"]:
    from maxtext.multimodal.processor_gemma4 import preprocess_mm_data_gemma4  # pylint: disable=import-outside-toplevel

    return preprocess_mm_data_gemma4(image)
  elif vision_block in ["llama4"]:
    from maxtext.multimodal.processor_llama4 import preprocess_mm_data_llama4  # pylint: disable=import-outside-toplevel

    return preprocess_mm_data_llama4(image)
  elif vision_block in ["qwen3_omni", "qwen3_vl", "qwen3_5"]:
    from maxtext.multimodal.processor_qwen3_omni import preprocess_mm_data_qwen3_omni_for_training  # pylint: disable=import-outside-toplevel

    return preprocess_mm_data_qwen3_omni_for_training(image, config)
  else:
    raise ValueError(f"Model {config.model_name} (vision block {vision_block}) not supported for image preprocessing.")


def get_image_offsets(config, processor_output: mm_utils.PreprocessorOutput | None):
  """Get the increase in total token count after inserting image token placeholders"""
  vision_block = _get_vision_block(config)

  if vision_block in ["gemma3"]:
    from maxtext.multimodal.processor_gemma3 import get_image_offsets_gemma3  # pylint: disable=import-outside-toplevel

    return get_image_offsets_gemma3(processor_output)
  elif vision_block in ["gemma4"]:
    from maxtext.multimodal.processor_gemma4 import get_image_offsets_gemma4  # pylint: disable=import-outside-toplevel

    return get_image_offsets_gemma4(processor_output)
  elif vision_block in ["llama4"]:
    from maxtext.multimodal.processor_llama4 import get_image_offsets_llama4  # pylint: disable=import-outside-toplevel

    return get_image_offsets_llama4(processor_output)
  elif vision_block in ["qwen3_omni", "qwen3_vl", "qwen3_5"]:
    from maxtext.multimodal.processor_qwen3_omni import get_mm_offsets_qwen3_omni  # pylint: disable=import-outside-toplevel

    return get_mm_offsets_qwen3_omni(config, processor_output)
  else:
    return 0


def reformat_prompt(
    prompt,
    image_placeholder,
    model_name,
    num_images,
    video_placeholder="<|video|>",
    num_videos=0,
    num_image_tokens=None,
    num_video_tokens=None,
):
  """Reformat prompt for different models."""
  vision_block = _get_vision_block(model_name)
  if vision_block is None:
    return prompt

  decoder_block = _get_decoder_block(model_name)

  if decoder_block in ["gemma3"]:
    from maxtext.multimodal.processor_gemma3 import reformat_prompt_gemma3  # pylint: disable=import-outside-toplevel

    return reformat_prompt_gemma3(prompt, image_placeholder, num_images)
  elif decoder_block in ["gemma4", "gemma4_small"]:
    from maxtext.multimodal.processor_gemma4 import reformat_prompt_gemma4  # pylint: disable=import-outside-toplevel

    return reformat_prompt_gemma4(prompt, image_placeholder, num_images)
  elif decoder_block in ["llama4"]:
    from maxtext.multimodal.processor_llama4 import reformat_prompt_llama4  # pylint: disable=import-outside-toplevel

    return reformat_prompt_llama4(prompt, image_placeholder, num_images)
  elif decoder_block in ["qwen3", "qwen3_moe", "qwen3_5"]:
    from maxtext.multimodal.processor_qwen3_omni import reformat_prompt_qwen3_omni  # pylint: disable=import-outside-toplevel

    return reformat_prompt_qwen3_omni(
        prompt=prompt,
        image_placeholder=image_placeholder,
        num_images=num_images,
        video_placeholder=video_placeholder,
        num_videos=num_videos,
        num_image_tokens=num_image_tokens,
        num_video_tokens=num_video_tokens,
    )
  else:
    return prompt


def reformat_response(response, model_name):
  """Reformat response for different models."""
  vision_block = _get_vision_block(model_name)
  if vision_block is None:
    return response

  decoder_block = _get_decoder_block(model_name)

  if decoder_block in ["llama4"]:
    formatted_response = f"{response}<|eot|>"
    return formatted_response
  elif decoder_block in ["gemma3"]:
    formatted_response = f"{response}<end_of_turn>"
    return formatted_response
  elif decoder_block in ["gemma4", "gemma4_small"]:
    formatted_response = f"{response}<turn|>"
    return formatted_response
  elif decoder_block in ["qwen3", "qwen3_moe", "qwen3_5"]:
    formatted_response = f"{response}<|im_end|>"
    return formatted_response
  else:
    return response


def prepare_text_for_image_fusion(tokens, config, processor_output=None):
  """Prepare text by adding extra tokens for image fusion based on the model."""
  vision_block = _get_vision_block(config)
  if vision_block in ["gemma3"]:
    from maxtext.multimodal.processor_gemma3 import add_extra_tokens_for_images_gemma3  # pylint: disable=import-outside-toplevel

    return add_extra_tokens_for_images_gemma3(
        tokens, max_num_images=processor_output.num_images  # pyrefly: ignore[missing-attribute]
    )  # pyrefly: ignore[missing-attribute]
  elif vision_block in ["gemma4"]:
    from maxtext.multimodal.processor_gemma4 import add_extra_tokens_for_images_gemma4  # pylint: disable=import-outside-toplevel

    return add_extra_tokens_for_images_gemma4(
        tokens, max_num_images=processor_output.num_images  # pyrefly: ignore[missing-attribute]
    )  # pyrefly: ignore[missing-attribute]
  elif vision_block in ["llama4"]:
    from maxtext.multimodal.processor_llama4 import add_extra_tokens_for_images_llama4  # pylint: disable=import-outside-toplevel

    return add_extra_tokens_for_images_llama4(tokens, processor_output)  # pyrefly: ignore[bad-argument-type]
  elif vision_block in ["qwen3_omni", "qwen3_vl", "qwen3_5"]:
    from maxtext.multimodal.processor_qwen3_omni import add_extra_tokens_for_qwen3_omni  # pylint: disable=import-outside-toplevel

    return add_extra_tokens_for_qwen3_omni(tokens, config, processor_output)
  else:
    raise ValueError(f"Model {config.model_name} (vision block {vision_block}) does not support multimodal inference.")


def get_dummy_image_shape_for_init(model_name, batch_size=1, num_image_per_sequence=1):
  """Return the shape of the dummy image for specific model's initialization."""
  image_shape = ()
  vision_block = _get_vision_block(model_name)
  if vision_block in ["gemma3"]:
    from maxtext.multimodal.processor_gemma3 import get_dummy_image_shape_for_init_gemma3  # pylint: disable=import-outside-toplevel

    image_shape = get_dummy_image_shape_for_init_gemma3(batch_size, num_image_per_sequence)
  elif vision_block in ["gemma4"]:
    from maxtext.multimodal.processor_gemma4 import get_dummy_image_shape_for_init_gemma4  # pylint: disable=import-outside-toplevel

    image_shape = get_dummy_image_shape_for_init_gemma4(batch_size, num_image_per_sequence)
  elif vision_block in ["llama4"]:
    from maxtext.multimodal.processor_llama4 import get_dummy_image_shape_for_init_llama4  # pylint: disable=import-outside-toplevel

    image_shape = get_dummy_image_shape_for_init_llama4(batch_size, num_image_per_sequence)
  elif vision_block in ["qwen3_omni", "qwen3_vl", "qwen3_5"]:
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


def get_bidirectional_mask_vision(config, decoder_input_tokens, is_video: bool = False):
  """Get the bidirectional mask for specific models."""
  bidirectional_mask_vision = None

  vision_block = _get_vision_block(config)
  if vision_block is None:
    return bidirectional_mask_vision

  decoder_block = _get_decoder_block(config)

  if decoder_block in ["gemma3"]:
    from maxtext.multimodal.processor_gemma3 import GEMMA_TOKEN_PLACEHOLDER  # pylint: disable=import-outside-toplevel

    bidirectional_mask_vision = decoder_input_tokens == GEMMA_TOKEN_PLACEHOLDER
  elif decoder_block in ["gemma4", "gemma4_small"]:
    from maxtext.multimodal.processor_gemma4 import GEMMA4_TOKEN_PLACEHOLDER  # pylint: disable=import-outside-toplevel

    bidirectional_mask_vision = decoder_input_tokens == GEMMA4_TOKEN_PLACEHOLDER
  elif decoder_block in ["llama4"]:
    from maxtext.multimodal.processor_llama4 import LLAMA4_PATCH_TOKEN  # pylint: disable=import-outside-toplevel

    bidirectional_mask_vision = decoder_input_tokens == LLAMA4_PATCH_TOKEN
  elif decoder_block in ["qwen3", "qwen3_moe", "qwen3_5"]:
    from maxtext.multimodal.processor_qwen3_omni import QwenTokens  # pylint: disable=import-outside-toplevel

    tokens = QwenTokens(config)

    if is_video:
      bidirectional_mask_vision = decoder_input_tokens == tokens.video_pad
    else:
      bidirectional_mask_vision = decoder_input_tokens == tokens.image_pad
  return bidirectional_mask_vision


def get_bidirectional_mask_audio(config, decoder_input_tokens):
  """Get the bidirectional mask for specific models."""
  bidirectional_mask_audio = None
  if config.model_name in ["qwen3-omni-30b-a3b"]:
    from maxtext.multimodal.processor_qwen3_omni import QwenTokens  # pylint: disable=import-outside-toplevel

    tokens = QwenTokens(config)

    # Create bidirectional_mask for audio token merging
    bidirectional_mask_audio = decoder_input_tokens == tokens.audio_pad
  return bidirectional_mask_audio


def downsample_video_mask_to_tokens(video_mask, config):
  """Routes video-mask reduction to the model-specific multimodal processor."""
  if video_mask is None:
    return None
  if config.model_name.startswith(("qwen3")):
    from maxtext.multimodal.processor_qwen3_omni import (  # pylint: disable=import-outside-toplevel
        downsample_video_mask_to_tokens as downsample_qwen3_video_mask,
    )

    return downsample_qwen3_video_mask(video_mask, config)
  raise ValueError(f"Model {config.model_name} does not support padded video-mask reduction.")
