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

"""MaxText vLLM adapter package."""

import os
from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model
from .adapter import MaxTextForCausalLM


logger = init_logger(__name__)


def register(config=None):
  """Register MaxTextForCausalLM model with tpu_inference and vllm.

  Note, this function is invoked directly by the vLLM engine during startup. As such,
  it leverages vLLM logging to report its status.
  """
  model_name = os.environ.get("MAXTEXT_MODEL_NAME")
  if config:
    model_name = config.model_name
    os.environ["MAXTEXT_MODEL_NAME"] = model_name

  if model_name and model_name.startswith("qwen3-vl"):
    MaxTextForCausalLM.supports_multimodal = True
    logger.info("Setting supports_multimodal = True for %s", model_name)

  register_model("MaxTextForCausalLM", MaxTextForCausalLM)

  # Dynamically apply KVCacheManager patch when registering the adapter
  # pylint: disable=import-outside-toplevel
  from .adapter import patch_kv_cache_manager

  patch_kv_cache_manager()

  if model_name and model_name.startswith("qwen3-vl"):
    try:
      from vllm.multimodal import MULTIMODAL_REGISTRY
      from vllm.model_executor.models.qwen3_vl import (
          Qwen3VLMultiModalProcessor,
          Qwen3VLProcessingInfo,
          Qwen3VLDummyInputsBuilder,
      )

      logger.info("Registering Qwen3VLMultiModalProcessor for MaxTextForCausalLM.")
      MULTIMODAL_REGISTRY.register_processor(
          Qwen3VLMultiModalProcessor,
          info=Qwen3VLProcessingInfo,
          dummy_inputs=Qwen3VLDummyInputsBuilder,
      )(MaxTextForCausalLM)
    except ImportError as e:
      logger.warning("Failed to register Qwen3VLMultiModalProcessor: %s", e)

  logger.info("Successfully registered MaxTextForCausalLM model.")
