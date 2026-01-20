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

"""Tests for ckpt_conversion strategies."""

import unittest

from MaxText.utils.ckpt_conversion.strategies.deepseek import DeepSeekMapper
from MaxText.utils.ckpt_conversion.strategies.factory import ModelMapperFactory
from MaxText.utils.ckpt_conversion.strategies.gemma import Gemma2Mapper, Gemma3Mapper
from MaxText.utils.ckpt_conversion.strategies.gpt_oss import GptOssMapper
from MaxText.utils.ckpt_conversion.strategies.llama import LlamaMapper
from MaxText.utils.ckpt_conversion.strategies.mixtral import MixtralMapper
from MaxText.utils.ckpt_conversion.strategies.qwen import QwenMapper, QwenOmniMoeMapper


class StrategyFactoryTest(unittest.TestCase):

  def test_gemma_strategies(self):
    strategy = ModelMapperFactory.get_strategy("gemma-2b")
    self.assertIsInstance(strategy, Gemma2Mapper)
    strategy = ModelMapperFactory.get_strategy("gemma2-9b")
    self.assertIsInstance(strategy, Gemma2Mapper)
    strategy = ModelMapperFactory.get_strategy("gemma3-4b")
    self.assertIsInstance(strategy, Gemma3Mapper)

  def test_llama3_strategy(self):
    strategy = ModelMapperFactory.get_strategy("llama3.1-8b")
    self.assertIsInstance(strategy, LlamaMapper)

  def test_qwen_strategies(self):
    strategy = ModelMapperFactory.get_strategy("qwen3-0.6b")
    self.assertIsInstance(strategy, QwenMapper)
    strategy = ModelMapperFactory.get_strategy("qwen3-omni-30b")
    self.assertIsInstance(strategy, QwenOmniMoeMapper)

  def test_legacy_models(self):
    strategy = ModelMapperFactory.get_strategy("gpt-oss-20b")
    self.assertIsInstance(strategy, GptOssMapper)

    strategy = ModelMapperFactory.get_strategy("deepseek3-671b")
    self.assertIsInstance(strategy, DeepSeekMapper)

    strategy = ModelMapperFactory.get_strategy("mixtral-8x7b")
    self.assertIsInstance(strategy, MixtralMapper)

  def test_factory_failure(self):
    with self.assertRaisesRegex(ValueError, "No mapping strategy found"):
      ModelMapperFactory.get_strategy("unknown-model-123b")


if __name__ == "__main__":
  unittest.main()
