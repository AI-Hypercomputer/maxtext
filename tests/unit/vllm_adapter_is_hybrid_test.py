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

"""Tests the vLLM `is_hybrid` flag the MaxText adapter sets per model."""

import unittest

try:
  from maxtext.integration.vllm.maxtext_vllm_adapter.adapter import is_hybrid_maxtext_model
except ImportError:  # vLLM / tpu_inference not installed.
  is_hybrid_maxtext_model = None


@unittest.skipIf(is_hybrid_maxtext_model is None, "requires vllm and tpu_inference")
class IsHybridMaxTextModelTest(unittest.TestCase):
  """Only attention + GatedDeltaNet decoder blocks are vLLM hybrid models."""

  def test_gdn_models_are_hybrid(self):
    # vLLM only turns on mamba prefix caching ("align" mode) for hybrid
    # models; without the flag a prefix-cached Qwen3.5 rollout dies in the
    # scheduler's MambaManager.
    self.assertTrue(is_hybrid_maxtext_model("qwen3.5-35b-a3b"))
    self.assertTrue(is_hybrid_maxtext_model("qwen3-next-80b-a3b"))

  def test_dense_and_moe_attention_models_are_not_hybrid(self):
    self.assertFalse(is_hybrid_maxtext_model("qwen3-8b"))
    self.assertFalse(is_hybrid_maxtext_model("qwen3-30b-a3b"))

  def test_unknown_or_missing_model_is_not_hybrid(self):
    self.assertFalse(is_hybrid_maxtext_model(None))
    self.assertFalse(is_hybrid_maxtext_model(""))
    self.assertFalse(is_hybrid_maxtext_model("no-such-model"))


if __name__ == "__main__":
  unittest.main()
