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
from .multimodal import get_multimodal_handler


logger = init_logger(__name__)


def register(config=None):
  """Register MaxTextForCausalLM model with tpu_inference and vllm.

  Note, this function is invoked directly by the vLLM engine during startup. As such,
  it leverages vLLM logging to report its status.
  """
  logger.info("Registering MaxTextForCausalLM model with tpu_inference and vllm.")
  model_name = os.environ.get("MAXTEXT_MODEL_NAME")
  if config is not None:
    model_name = config.model_name
    os.environ["MAXTEXT_MODEL_NAME"] = model_name

  multimodal_handler = get_multimodal_handler(model_name)
  MaxTextForCausalLM.supports_multimodal = multimodal_handler is not None
  if multimodal_handler is not None:
    logger.info("Setting supports_multimodal = True for %s", model_name)

  register_model("MaxTextForCausalLM", MaxTextForCausalLM)

  # Dynamically apply KVCacheManager patch when registering the adapter
  # pylint: disable=import-outside-toplevel
  from .adapter import patch_kv_cache_manager

  patch_kv_cache_manager()

  if multimodal_handler is not None:
    multimodal_handler.register_processor(MaxTextForCausalLM)

  logger.info("Successfully registered MaxTextForCausalLM model.")
