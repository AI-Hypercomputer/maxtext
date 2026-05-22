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

"""Multimodal data preprocessor router.

This module routes all multimodal preprocessing, prompt formatting, and token 
manipulation calls to the appropriate model-specific processor class.

Routing Mappings:
  - Model: "gemma3-*" -> processor_gemma3.Gemma3Processor
  - Model: "gemma4-*" -> processor_gemma4.Gemma4Processor
  - Model: "llama4-*" -> processor_llama4.Llama4Processor
  - Model: "qwen3-omni-*" or "qwen3.5-*" -> processor_qwen3_omni.Qwen3OmniProcessor
"""

from maxtext.multimodal import utils as mm_utils
from maxtext.multimodal.utils import BaseMultimodalProcessor
from maxtext.utils import max_logging
from maxtext.multimodal.processor_gemma3 import Gemma3Processor
from maxtext.multimodal.processor_gemma4 import Gemma4Processor
from maxtext.multimodal.processor_llama4 import Llama4Processor
from maxtext.multimodal.processor_qwen3_omni import Qwen3OmniProcessor

_REGISTERED_PROCESSORS = [
    Gemma3Processor,
    Gemma4Processor,
    Llama4Processor,
    Qwen3OmniProcessor,
]


def get_processor(model_name: str) -> BaseMultimodalProcessor:
  """Returns the appropriate processor instance for the given model name.

  If no specific processor matches, returns a fallback BaseMultimodalProcessor
  instance.
  """
  for processor_cls in _REGISTERED_PROCESSORS:
    if processor_cls.supports_model(model_name):
      max_logging.log(f"Multimodal Router: Instantiating {processor_cls.__name__} for model {model_name}")
      return processor_cls(model_name)
  max_logging.log(
      f"Multimodal Router: No specific processor found for {model_name}. Falling back to BaseMultimodalProcessor."
  )
  return BaseMultimodalProcessor(model_name)


def preprocess_mm_data(config, **kwargs) -> mm_utils.PreprocessorOutput:
  """Preprocesses multimodal data based on the provided configuration."""
  return get_processor(config.model_name).preprocess_mm_data(config, **kwargs)


def preprocess_image_for_training(image, model_name, **kwargs) -> mm_utils.PreprocessorOutput:
  """Preprocesses a single image for training based on the model name."""
  return get_processor(model_name).preprocess_image_for_training(image, **kwargs)


def get_image_offsets(config, processor_output: mm_utils.PreprocessorOutput | None, **kwargs) -> int:
  """Get the increase in total token count after inserting image tokens."""
  return get_processor(config.model_name).get_image_offsets(config, processor_output, **kwargs)


def reformat_prompt(
    prompt,
    image_placeholder,
    model_name,
    num_images,
    **kwargs,
) -> str:
  """Reformat prompt for different models."""
  return get_processor(model_name).reformat_prompt(
      prompt=prompt,
      image_placeholder=image_placeholder,
      num_images=num_images,
      **kwargs,
  )


def reformat_response(response, model_name, **kwargs) -> str:
  """Reformat response for different models."""
  return get_processor(model_name).reformat_response(response, **kwargs)


def prepare_text_for_image_fusion(
    tokens,
    config,
    processor_output=None,
    **kwargs,
):
  """Prepare text by adding extra tokens for image fusion based on the model."""
  return get_processor(config.model_name).prepare_text_for_image_fusion(tokens, config, processor_output, **kwargs)


def get_dummy_image_shape_for_init(
    model_name,
    batch_size=1,
    num_image_per_sequence=1,
    **kwargs,
) -> tuple:
  """Return the shape of the dummy image for specific model's initialization."""
  return get_processor(model_name).get_dummy_image_shape_for_init(
      batch_size=batch_size,
      num_image_per_sequence=num_image_per_sequence,
      **kwargs,
  )


def get_dummy_audio_shape_for_init(config, **kwargs) -> tuple:
  """Return the shape of the dummy audio for specific model's initialization."""
  return get_processor(config.model_name).get_dummy_audio_shape_for_init(config, **kwargs)


def get_bidirectional_mask_vision(config, decoder_input_tokens, **kwargs):
  """Get the bidirectional mask for vision tokens."""
  return get_processor(config.model_name).get_bidirectional_mask_vision(config, decoder_input_tokens, **kwargs)


def get_bidirectional_mask_audio(config, decoder_input_tokens, **kwargs):
  """Get the bidirectional mask for audio tokens."""
  return get_processor(config.model_name).get_bidirectional_mask_audio(config, decoder_input_tokens, **kwargs)
