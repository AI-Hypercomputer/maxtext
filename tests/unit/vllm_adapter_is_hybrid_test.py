# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests the vLLM `is_hybrid` flag and `MaxTextModelConfig` hook per model."""

import dataclasses
import os
from types import SimpleNamespace
import unittest
from unittest import mock

try:
  from maxtext.integration.vllm.maxtext_vllm_adapter.adapter import (
      MaxTextForCausalLM,
      MaxTextModelConfig,
      is_hybrid_maxtext_model,
  )
except ImportError:  # vLLM / tpu_inference not installed.
  MaxTextForCausalLM = None
  MaxTextModelConfig = None
  is_hybrid_maxtext_model = None


@dataclasses.dataclass(frozen=True)
class _DummyModelInfo:
  is_hybrid: bool = False


@unittest.skipIf(is_hybrid_maxtext_model is None, "requires vllm and tpu_inference")
class IsHybridMaxTextModelTest(unittest.TestCase):
  """Only attention + GatedDeltaNet decoder blocks are vLLM hybrid models."""

  def test_gdn_models_are_hybrid(self):
    # vLLM only turns on mamba prefix caching ("align" mode) for hybrid
    # models; without the flag a prefix-cached Qwen3.5 rollout dies in the
    # scheduler's MambaManager.
    self.assertTrue(is_hybrid_maxtext_model("qwen3.5-35b-a3b"))
    self.assertTrue(is_hybrid_maxtext_model("qwen3.5-397b-a17b"))
    self.assertTrue(is_hybrid_maxtext_model("qwen3-next-80b-a3b"))

  def test_dense_and_moe_attention_models_are_not_hybrid(self):
    self.assertFalse(is_hybrid_maxtext_model("qwen3-8b"))
    self.assertFalse(is_hybrid_maxtext_model("qwen3-30b-a3b"))

  def test_unknown_or_missing_model_is_not_hybrid(self):
    self.assertFalse(is_hybrid_maxtext_model(None))
    self.assertFalse(is_hybrid_maxtext_model(""))
    self.assertFalse(is_hybrid_maxtext_model("no-such-model"))

  def test_maxtext_model_config_resolves_hybrid_from_additional_config(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("MAXTEXT_MODEL_NAME", None)
      MaxTextForCausalLM.is_hybrid = False
      model_config = SimpleNamespace(_model_info=_DummyModelInfo(is_hybrid=False))
      cache_config = SimpleNamespace(
          enable_prefix_caching=True,
          mamba_cache_mode="none",
          block_size=256,
          mamba_block_size=None,
      )
      vllm_config = SimpleNamespace(
          model_config=model_config,
          cache_config=cache_config,
          additional_config={"maxtext_config": {"model_name": "qwen3.5-397b-a17b"}},
      )

      MaxTextModelConfig.verify_and_update_config(vllm_config)

      self.assertTrue(MaxTextForCausalLM.is_hybrid)
      self.assertTrue(model_config._model_info.is_hybrid)
      self.assertEqual(cache_config.mamba_cache_mode, "align")
      self.assertEqual(cache_config.mamba_block_size, 256)
      self.assertEqual(os.environ.get("MAXTEXT_MODEL_NAME"), "qwen3.5-397b-a17b")

  def test_maxtext_model_config_leaves_non_hybrid_mamba_cache_mode_none(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("MAXTEXT_MODEL_NAME", None)
      MaxTextForCausalLM.is_hybrid = True
      model_config = SimpleNamespace(_model_info=_DummyModelInfo(is_hybrid=True))
      cache_config = SimpleNamespace(
          enable_prefix_caching=True,
          mamba_cache_mode="none",
          block_size=256,
          mamba_block_size=None,
      )
      vllm_config = SimpleNamespace(
          model_config=model_config,
          cache_config=cache_config,
          additional_config={"maxtext_config": {"model_name": "qwen3-30b-a3b"}},
      )

      MaxTextModelConfig.verify_and_update_config(vllm_config)

      self.assertFalse(MaxTextForCausalLM.is_hybrid)
      self.assertFalse(model_config._model_info.is_hybrid)
      self.assertEqual(cache_config.mamba_cache_mode, "none")


if __name__ == "__main__":
  unittest.main()
