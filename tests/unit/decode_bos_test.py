# Copyright 2026 Google LLC
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

"""Tests for whether decode and the API server ask the tokenizer to prepend BOS to the prompt."""

import os
from types import SimpleNamespace
import unittest
from unittest import mock

import pytest

from benchmarks.api_server import maxtext_generator
from maxtext.inference import decode

pytestmark = pytest.mark.cpu_only


class _StopDecode(Exception):
  """Raised by the fake tokenizer to end `decode.main` right after the prompt is encoded."""


class _Tokenizer:
  """Stand-in for a JetStream tokenizer that records the `is_bos` it is asked for.

  `hf=True` mimics the Hugging Face wrapper, whose `.tokenizer` may carry a chat template; the SentencePiece and
  tiktoken wrappers have no `.tokenizer`.
  """

  def __init__(self, chat_template=None, bos_id=2, hf=True):
    if hf:
      self.tokenizer = SimpleNamespace(chat_template=chat_template)
    self.bos_id = bos_id
    self.is_bos = None

  def encode(self, text, is_bos, prefill_lengths):
    del text, prefill_lengths
    self.is_bos = is_bos
    raise _StopDecode


def _config(add_bos=True, use_chat_template=False, use_multimodal=False, model_name="test-model"):
  return SimpleNamespace(
      add_bos=add_bos,
      use_chat_template=use_chat_template,
      use_multimodal=use_multimodal,
      load_full_state_path="",
      shardy=True,
      prompt="I love to",
      max_prefill_predict_length=8,
      model_name=model_name,
      image_placeholder="<|image|>",
      video_placeholder="<|video|>",
      audio_placeholder="<|audio|>",
  )


class DecodeBosTest(unittest.TestCase):
  """`is_bos` passed to `tokenizer_model.encode` by `decode.main`."""

  def _is_bos(self, config, tokenizer):
    """Runs `decode.main` up to the prompt encoding and returns the `is_bos` it used."""
    engine = mock.MagicMock()
    engine.build_tokenizer.return_value = tokenizer
    with (
        mock.patch.object(decode, "engine_api", SimpleNamespace()),
        mock.patch.object(decode, "_token_params_ns", SimpleNamespace()),
        mock.patch.object(decode.jax.config, "update"),
        mock.patch.dict(os.environ),
        mock.patch.object(decode.pyconfig, "initialize", return_value=config),
        mock.patch.object(decode.max_utils, "print_system_information"),
        mock.patch.object(decode.maxengine, "MaxEngine", return_value=engine),
        mock.patch.object(decode.profiler, "Profiler"),
        mock.patch.object(decode.mm_processor, "preprocess_mm_data"),
        mock.patch.object(decode.mm_processor, "get_image_offsets", return_value=0),
        mock.patch.object(decode.mm_processor, "reformat_prompt", return_value="<formatted prompt>"),
        self.assertRaises(_StopDecode),
    ):
      decode.main([])
    return tokenizer.is_bos

  def test_text_prompt_gets_bos_when_tokenizer_has_chat_template(self):
    # Gemma 3/4 and gpt-oss tokenizers ship a chat template, which is not applied to a text prompt.
    self.assertTrue(self._is_bos(_config(), _Tokenizer(chat_template="{{ messages }}")))

  def test_text_prompt_gets_bos_without_chat_template(self):
    self.assertTrue(self._is_bos(_config(), _Tokenizer()))

  def test_text_prompt_gets_bos_from_sentencepiece_or_tiktoken_tokenizer(self):
    self.assertTrue(self._is_bos(_config(), _Tokenizer(hf=False)))

  def test_no_bos_when_tokenizer_has_no_bos_token(self):
    # Qwen3 tokenizers have a chat template but no BOS token.
    self.assertFalse(self._is_bos(_config(), _Tokenizer(chat_template="{{ messages }}", bos_id=None)))

  def test_no_bos_when_add_bos_is_false(self):
    self.assertFalse(self._is_bos(_config(add_bos=False), _Tokenizer(chat_template="{{ messages }}")))

  def test_no_bos_when_chat_template_is_applied(self):
    config = _config(use_chat_template=True)
    self.assertFalse(self._is_bos(config, _Tokenizer(chat_template="{{ messages }}")))

  def test_text_prompt_gets_bos_with_use_chat_template_on_sentencepiece_or_tiktoken_tokenizer(self):
    # These tokenizers cannot apply a chat template, so `use_chat_template` leaves the prompt as plain text.
    self.assertTrue(self._is_bos(_config(use_chat_template=True), _Tokenizer(hf=False)))

  def test_multimodal_prompt_skips_bos_when_tokenizer_has_chat_template(self):
    # `reformat_prompt` writes BOS into the text for these models.
    config = _config(use_multimodal=True)
    self.assertFalse(self._is_bos(config, _Tokenizer(chat_template="{{ messages }}")))

  def test_multimodal_prompt_gets_bos_without_chat_template(self):
    self.assertTrue(self._is_bos(_config(use_multimodal=True), _Tokenizer()))

  def test_gemma3_multimodal_prompt_gets_bos_when_tokenizer_has_chat_template(self):
    # Gemma 3's format leaves BOS to the token ids (`mm_processor.reformatted_prompt_needs_bos`).
    config = _config(use_multimodal=True, model_name="gemma3-4b")
    self.assertTrue(self._is_bos(config, _Tokenizer(chat_template="{{ messages }}")))


class _ServerTokenizer:
  """Stand-in for the API server's JetStream tokenizer that records the `is_bos` of each `encode` call.

  `hf=False` mimics the SentencePiece and tiktoken wrappers, which have no `.tokenizer`.
  """

  def __init__(self, chat_template="{{ messages }}", bos_id=2, hf=True):
    if hf:
      self.tokenizer = SimpleNamespace(chat_template=chat_template)
    self.bos_id = bos_id
    self.is_bos = []

  def encode(self, text, is_bos, prefill_lengths):
    del text, prefill_lengths
    self.is_bos.append(is_bos)
    return [], 0


class ApiServerBosTest(unittest.TestCase):
  """`is_bos` passed to the tokenizer by the API server's `MaxTextGenerator`."""

  def _generator(self, tokenizer, add_bos=True, use_chat_template=False, use_multimodal=False):
    """Returns a `MaxTextGenerator` with only the attributes that prompt preprocessing reads."""
    generator = object.__new__(maxtext_generator.MaxTextGenerator)
    generator.config = SimpleNamespace(
        add_bos=add_bos,
        use_chat_template=use_chat_template,
        use_multimodal=use_multimodal,
        prompt="Describe the image.",
        image_placeholder="<|image|>",
        model_name="test-model",
        max_prefill_predict_length=8,
    )
    generator.tokenizer = tokenizer
    generator.has_chat_template = bool(getattr(getattr(tokenizer, "tokenizer", None), "chat_template", None))
    return generator

  def _is_bos(self, tokenizer, is_chat=False, image_path=None, **config):
    """Preprocesses one prompt and returns the `is_bos` it was encoded with."""
    generator = self._generator(tokenizer, use_multimodal=image_path is not None, **config)
    mm_processor = maxtext_generator.mm_processor
    with (
        mock.patch.object(mm_processor, "preprocess_mm_data"),
        mock.patch.object(mm_processor, "get_image_offsets", return_value=0),
        mock.patch.object(mm_processor, "reformat_prompt", return_value="<formatted prompt>"),
        mock.patch.object(mm_processor, "prepare_text_for_image_fusion"),
    ):
      generator._preprocess_inputs("I love to", 8, image_path, is_chat=is_chat)  # pylint: disable=protected-access
    return tokenizer.is_bos[-1]

  def test_completion_prompt_gets_bos_when_tokenizer_has_chat_template(self):
    # /v1/completions passes the prompt text as is (e.g. lm-eval's loglikelihood tasks).
    self.assertTrue(self._is_bos(_ServerTokenizer()))

  def test_chat_prompt_gets_no_bos(self):
    # /v1/chat/completions formats the messages with the chat template, which decides whether the prompt starts
    # with BOS: Gemma 3's template writes it, gpt-oss's has none although its tokenizer has a BOS token.
    self.assertFalse(self._is_bos(_ServerTokenizer(), is_chat=True))

  def test_completion_prompt_gets_bos_when_bos_id_is_zero(self):
    # DeepSeek V3's BOS id is 0.
    self.assertTrue(self._is_bos(_ServerTokenizer(bos_id=0)))

  def test_completion_prompt_gets_bos_from_sentencepiece_or_tiktoken_tokenizer(self):
    self.assertTrue(self._is_bos(_ServerTokenizer(hf=False)))

  def test_no_bos_when_tokenizer_has_no_bos_token(self):
    # Qwen3 tokenizers have a chat template but no BOS token.
    self.assertFalse(self._is_bos(_ServerTokenizer(bos_id=None)))

  def test_no_bos_when_add_bos_is_false(self):
    self.assertFalse(self._is_bos(_ServerTokenizer(), add_bos=False))

  def test_no_bos_when_tokenizer_applies_chat_template(self):
    self.assertFalse(self._is_bos(_ServerTokenizer(), use_chat_template=True))

  def test_completion_prompt_gets_bos_with_use_chat_template_on_sentencepiece_or_tiktoken_tokenizer(self):
    # These tokenizers cannot apply a chat template, so `use_chat_template` leaves the prompt as plain text.
    self.assertTrue(self._is_bos(_ServerTokenizer(hf=False), use_chat_template=True))

  def test_multimodal_prompt_follows_decode(self):
    # Same rule as decode.py: no BOS when the tokenizer has a chat template, BOS otherwise.
    self.assertFalse(self._is_bos(_ServerTokenizer(), image_path="image.jpg"))
    self.assertTrue(self._is_bos(_ServerTokenizer(chat_template=None), image_path="image.jpg"))

  def test_is_chat_flags_reach_each_prompt(self):
    tokenizer = _ServerTokenizer()
    generator = self._generator(tokenizer)
    generator._initialize_streams_and_state(  # pylint: disable=protected-access
        ["I love to", "<chat prompt>", "I love to"], [None] * 3, seed=0, is_chat=[False, True, False]
    )
    self.assertEqual(tokenizer.is_bos, [True, False, True])


if __name__ == "__main__":
  unittest.main()
