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

"""Tests for the MaxText vLLM multimodal adapter helper layer.

Splits into two suites:

* `DummyMMInputInfoTest` — exercises `get_dummy_mm_input_info` and the
  `_hf_to_maxtext_model_name` resolver in isolation. Does not require vLLM.
* `MaxTextProcessingInfoTest` — exercises the vLLM-side classes
  (`MaxTextProcessingInfo`, `MaxTextDummyInputsBuilder`). Skipped when vLLM
  is not installed in the test environment.
"""

from __future__ import annotations

import importlib.util
import types
import unittest
from unittest import mock

import numpy as np


_HAS_VLLM = importlib.util.find_spec("vllm") is not None


def _make_fake_model_config(
    *,
    architecture: str,
    hidden_size: int | None = None,
    num_local_experts: int | None = None,
    explicit_model_name: str | None = None,
    mm_limits: dict | None = None,
):
  """Build a stand-in for vLLM's `ModelConfig` (only the attrs we read)."""
  text_cfg = types.SimpleNamespace(
      hidden_size=hidden_size, num_local_experts=num_local_experts
  )
  hf_config = types.SimpleNamespace(architectures=[architecture], text_config=text_cfg)
  additional: dict = {}
  if explicit_model_name:
    additional["maxtext_config"] = {"model_name": explicit_model_name}
  if mm_limits is not None:
    additional["maxtext_mm_limits"] = mm_limits
  return types.SimpleNamespace(hf_config=hf_config, additional_config=additional)


class DummyMMInputInfoTest(unittest.TestCase):
  """Tests the per-model helper independently of vLLM."""

  def test_gemma3_info(self):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        get_dummy_mm_input_info,
    )

    info = get_dummy_mm_input_info("gemma3-4b")
    self.assertEqual(info.image_shape, (896, 896, 3))
    self.assertEqual(info.image_pil_size, (896, 896))
    self.assertEqual(info.max_images, 1)
    self.assertEqual(info.image_placeholder_id, 262144)
    self.assertEqual(info.tokens_per_image, 260)  # 256 soft + 4 special
    self.assertIsNone(info.audio_shape)
    self.assertIsNone(info.audio_placeholder_id)

  def test_gemma4_info(self):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        get_dummy_mm_input_info,
    )

    info = get_dummy_mm_input_info("gemma4-26b")
    self.assertEqual(info.image_shape, (672, 960, 3))
    self.assertEqual(info.image_pil_size, (960, 672))
    self.assertEqual(info.image_placeholder_id, 258880)
    # (672/16)*(960/16)/9 = 280 soft tokens + 2 extras = 282
    self.assertEqual(info.tokens_per_image, 282)

  def test_llama4_info(self):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        get_dummy_mm_input_info,
    )

    info = get_dummy_mm_input_info("llama4-17b-16e")
    self.assertEqual(info.image_shape, (20, 3, 336, 336))
    self.assertEqual(info.image_placeholder_id, 200092)
    # 4x4 worst case: 1 begin + 4*(4*144 + 3 sep + 1 sep) + 1 fake + 144 + 1 end
    # = 1 + 4*580 + 1 + 144 + 1 = 2467
    self.assertEqual(info.tokens_per_image, 2467)
    self.assertIsNone(info.audio_shape)

  def test_qwen3_omni_info(self):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        get_dummy_mm_input_info,
    )

    info = get_dummy_mm_input_info("qwen3-omni-30b-a3b")
    self.assertEqual(info.image_shape, (3, 2, 768, 768))
    self.assertEqual(info.image_placeholder_id, 151655)
    # (768/16/2)^2 = 24^2 = 576 visual tokens.
    self.assertEqual(info.tokens_per_image, 576)
    self.assertEqual(info.max_images, 4)
    self.assertIsNotNone(info.audio_shape)
    self.assertEqual(info.audio_placeholder_id, 151675)
    self.assertEqual(info.max_audios, 2)
    self.assertEqual(info.audio_sample_rate, 16000)

  def test_overrides_apply_to_max_images_and_audios(self):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        get_dummy_mm_input_info,
    )

    info = get_dummy_mm_input_info(
        "qwen3-omni-30b-a3b", {"max_images": 8, "max_audios": 1}
    )
    self.assertEqual(info.max_images, 8)
    self.assertEqual(info.max_audios, 1)

  def test_unknown_model_raises(self):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        get_dummy_mm_input_info,
    )

    with self.assertRaises(ValueError):
      get_dummy_mm_input_info("not-a-real-model")


@unittest.skipUnless(_HAS_VLLM, "vLLM not installed; skipping resolver tests")
class HfToMaxtextResolverTest(unittest.TestCase):
  """Tests the HF-architecture → MaxText model_name dispatch."""

  def _resolve(self, **kwargs):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        _hf_to_maxtext_model_name,
    )

    return _hf_to_maxtext_model_name(_make_fake_model_config(**kwargs))

  def test_explicit_override_wins(self):
    name = self._resolve(
        architecture="Gemma3ForConditionalGeneration",
        hidden_size=2560,
        explicit_model_name="gemma3-27b",
    )
    self.assertEqual(name, "gemma3-27b")

  def test_gemma3_dispatch_by_hidden_size(self):
    self.assertEqual(
        self._resolve(architecture="Gemma3ForConditionalGeneration", hidden_size=2560),
        "gemma3-4b",
    )
    self.assertEqual(
        self._resolve(architecture="Gemma3ForConditionalGeneration", hidden_size=5376),
        "gemma3-27b",
    )

  def test_llama4_dispatch_by_num_experts(self):
    self.assertEqual(
        self._resolve(
            architecture="Llama4ForConditionalGeneration", num_local_experts=128
        ),
        "llama4-17b-128e",
    )

  def test_qwen3_omni(self):
    self.assertEqual(
        self._resolve(architecture="Qwen3OmniMoeForConditionalGeneration"),
        "qwen3-omni-30b-a3b",
    )

  def test_unknown_architecture_raises(self):
    with self.assertRaises(ValueError):
      self._resolve(architecture="SomeOtherModel")


@unittest.skipUnless(_HAS_VLLM, "vLLM not installed; skipping vLLM class tests")
class MaxTextProcessingInfoTest(unittest.TestCase):
  """Tests `MaxTextProcessingInfo` and `MaxTextDummyInputsBuilder`."""

  def _make_info(self, model_name: str):
    """Construct a `MaxTextProcessingInfo` with mocked context."""
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        MaxTextProcessingInfo,
    )

    model_config = _make_fake_model_config(
        architecture="Gemma3ForConditionalGeneration",
        explicit_model_name=model_name,
    )
    tokenizer = mock.MagicMock()
    tokenizer.decode = lambda ids: f"<|tok{ids[0]}|>"
    ctx = mock.MagicMock()
    ctx.model_config = model_config
    # BaseProcessingInfo.get_tokenizer() reads from ctx; stub it directly.
    info = MaxTextProcessingInfo(ctx)
    info.get_tokenizer = lambda: tokenizer  # type: ignore[assignment]
    return info, tokenizer

  def test_gemma3_supported_mm_limits(self):
    info, _ = self._make_info("gemma3-4b")
    limits = info.get_supported_mm_limits()
    self.assertEqual(limits, {"image": 1})

  def test_qwen3_supported_mm_limits_includes_audio(self):
    info, _ = self._make_info("qwen3-omni-30b-a3b")
    limits = info.get_supported_mm_limits()
    self.assertEqual(limits, {"image": 4, "audio": 2})

  def test_max_tokens_per_item_text_only(self):
    info, _ = self._make_info("gemma4-26b")
    out = info.get_mm_max_tokens_per_item(seq_len=2048, mm_counts={"image": 1})
    self.assertEqual(out, {"image": 282})

  def test_max_tokens_per_item_with_audio(self):
    info, _ = self._make_info("qwen3-omni-30b-a3b")
    out = info.get_mm_max_tokens_per_item(seq_len=2048, mm_counts={"image": 1, "audio": 1})
    self.assertIn("image", out)
    self.assertIn("audio", out)

  def test_dummy_builder_returns_pil_images(self):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        MaxTextDummyInputsBuilder,
    )

    info, _ = self._make_info("gemma3-4b")
    builder = MaxTextDummyInputsBuilder(info)
    out = builder.get_dummy_mm_data(seq_len=512, mm_counts={"image": 1})
    self.assertIn("image", out)
    self.assertEqual(out["image"][0].size, (896, 896))

  def test_dummy_builder_returns_audio_for_qwen(self):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        MaxTextDummyInputsBuilder,
    )

    info, _ = self._make_info("qwen3-omni-30b-a3b")
    builder = MaxTextDummyInputsBuilder(info)
    out = builder.get_dummy_mm_data(seq_len=512, mm_counts={"image": 1, "audio": 2})
    self.assertEqual(len(out["audio"]), 2)
    waveform, sample_rate = out["audio"][0]
    self.assertEqual(sample_rate, 16000)
    self.assertEqual(waveform.dtype, np.float32)

  def test_dummy_text_uses_placeholder_tokens(self):
    from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal_adapter import (
        MaxTextDummyInputsBuilder,
    )

    info, tokenizer = self._make_info("gemma3-4b")
    builder = MaxTextDummyInputsBuilder(info)
    text = builder.get_dummy_text({"image": 2})
    self.assertIn("<|tok262144|>", text)


if __name__ == "__main__":
  unittest.main()
