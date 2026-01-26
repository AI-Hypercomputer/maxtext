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

"""Factory for creating parameter mapper strategies."""

from MaxText.utils.ckpt_conversion.strategies.base import ParamMapperStrategy
from MaxText.utils.ckpt_conversion.strategies.deepseek import DeepSeekMapper
from MaxText.utils.ckpt_conversion.strategies.gemma import Gemma2Mapper, Gemma3Mapper
from MaxText.utils.ckpt_conversion.strategies.gpt_oss import GptOssMapper
from MaxText.utils.ckpt_conversion.strategies.llama import LlamaMapper
from MaxText.utils.ckpt_conversion.strategies.mixtral import MixtralMapper
from MaxText.utils.ckpt_conversion.strategies.qwen import QwenMapper, QwenOmniMoeMapper


class ModelMapperFactory:
  """Factory class to instantiate the appropriate parameter mapping strategy."""

  @staticmethod
  def get_strategy(model_name: str) -> ParamMapperStrategy:
    """Selects the correct strategy based on model name substrings.

    Args:
      model_name: The model identifier string (e.g. "gemma2-2b", "qwen3-4b").

    Returns:
      An instance of a concrete ParamMapperStrategy.

    Raises:
      ValueError: If no strategy is registered for the given model_name.
    """
    if model_name.startswith("gemma3"):
      return Gemma3Mapper()
    if model_name.startswith("gemma2") or model_name.startswith("gemma"):
      return Gemma2Mapper()
    if model_name.startswith("llama3.1"):
      return LlamaMapper()
    if model_name.startswith("qwen3"):
      if "omni" in model_name:
        return QwenOmniMoeMapper()
      return QwenMapper()
    if model_name.startswith("deepseek3"):
      return DeepSeekMapper()
    if model_name.startswith("gpt-oss"):
      return GptOssMapper()
    if model_name.startswith("mixtral"):
      return MixtralMapper()

    raise ValueError(f"No mapping strategy found for model: {model_name}")
