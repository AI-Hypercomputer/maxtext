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

"""MaxText vLLM adapter package."""

from tpu_inference.logger import init_logger
from tpu_inference.models.common.model_loader import register_model
from .adapter import MaxTextForCausalLM


logger = init_logger(__name__)


def register():
  """Register MaxTextForCausalLM model with tpu_inference and vllm.

  Note, this function is invoked directly by the vLLM engine during startup. As such,
  it leverages vLLM logging to report its status.
  """
  logger.info("Registering MaxTextForCausalLM model with tpu_inference and vllm.")
  register_model("MaxTextForCausalLM", MaxTextForCausalLM)
  logger.info("Successfully registered MaxTextForCausalLM model.")
