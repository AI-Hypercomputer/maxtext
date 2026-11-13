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


def _config(add_bos=True, use_chat_template=False, use_multimodal=False):
  return SimpleNamespace(
      add_bos=add_bos,
      use_chat_template=use_chat_template,
      use_multimodal=use_multimodal,
      load_full_state_path="",
      shardy=True,
      prompt="I love to",
      max_prefill_predict_length=8,
      model_name="test-model",
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
    mm_processor = mock.MagicMock()
    mm_processor.get_image_offsets.return_value = 0
    mm_processor.reformat_prompt.return_value = "<formatted prompt>"
    with (
        mock.patch.object(decode, "engine_api", SimpleNamespace()),
        mock.patch.object(decode, "_token_params_ns", SimpleNamespace()),
        mock.patch.object(decode.jax.config, "update"),
        mock.patch.dict(os.environ),
        mock.patch.object(decode.pyconfig, "initialize", return_value=config),
        mock.patch.object(decode.max_utils, "print_system_information"),
        mock.patch.object(decode.maxengine, "MaxEngine", return_value=engine),
        mock.patch.object(decode.profiler, "Profiler"),
        mock.patch.object(decode, "mm_processor", mm_processor),
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

  def test_multimodal_prompt_skips_bos_when_tokenizer_has_chat_template(self):
    # `reformat_prompt` writes BOS into the text for these models.
    config = _config(use_multimodal=True)
    self.assertFalse(self._is_bos(config, _Tokenizer(chat_template="{{ messages }}")))

  def test_multimodal_prompt_gets_bos_without_chat_template(self):
    self.assertTrue(self._is_bos(_config(use_multimodal=True), _Tokenizer()))


class _ServerTokenizer:
  """Stand-in for the API server's JetStream tokenizer that records the `is_bos` it is asked for.

  `bos_token` is the BOS text of the Hugging Face tokenizer (`None` for Qwen3, which has no BOS token); `hf=False`
  mimics the SentencePiece and tiktoken wrappers, which have no `.tokenizer`.
  """

  def __init__(self, bos_token="<bos>", bos_id=2, hf=True):
    if hf:
      self.tokenizer = SimpleNamespace(bos_token=bos_token, chat_template="{{ messages }}")
    self.bos_id = bos_id
    self.is_bos = None

  def encode(self, text, is_bos, prefill_lengths):
    del text, prefill_lengths
    self.is_bos = is_bos
    return [], 0


class ApiServerBosTest(unittest.TestCase):
  """`is_bos` passed to the tokenizer by the API server's `MaxTextGenerator._preprocess_inputs`."""

  def _is_bos(self, text, tokenizer, add_bos=True, use_chat_template=False, reformatted_prompt=None):
    """Encodes `text`, or an image prompt that `reformat_prompt` turns into `reformatted_prompt`; returns `is_bos`."""
    multimodal = reformatted_prompt is not None
    generator = object.__new__(maxtext_generator.MaxTextGenerator)
    generator.config = SimpleNamespace(
        add_bos=add_bos,
        use_chat_template=use_chat_template,
        use_multimodal=multimodal,
        prompt=text,
        image_placeholder="<|image|>",
        model_name="test-model",
    )
    generator.tokenizer = tokenizer
    generator.has_chat_template = bool(getattr(getattr(tokenizer, "tokenizer", None), "chat_template", False))
    mm_processor = mock.MagicMock()
    mm_processor.reformat_prompt.return_value = reformatted_prompt
    mm_processor.get_image_offsets.return_value = 0
    with mock.patch.object(maxtext_generator, "mm_processor", mm_processor):
      generator._preprocess_inputs(text, 8, "image.jpg" if multimodal else None)  # pylint: disable=protected-access
    return tokenizer.is_bos

  def test_completion_prompt_gets_bos_when_tokenizer_has_chat_template(self):
    # /v1/completions passes the prompt text as is (e.g. lm-eval's loglikelihood tasks).
    self.assertTrue(self._is_bos("I love to", _ServerTokenizer()))

  def test_chat_prompt_that_starts_with_bos_gets_no_second_bos(self):
    # /v1/chat/completions applies the chat template, which writes BOS into the text (here Gemma 3's).
    text = "<bos><start_of_turn>user\nI love to<end_of_turn>\n<start_of_turn>model\n"
    self.assertFalse(self._is_bos(text, _ServerTokenizer()))

  def test_completion_prompt_gets_bos_when_bos_id_is_zero(self):
    # DeepSeek V3's BOS id is 0.
    self.assertTrue(self._is_bos("I love to", _ServerTokenizer(bos_token="<s>", bos_id=0)))

  def test_completion_prompt_gets_bos_from_sentencepiece_or_tiktoken_tokenizer(self):
    self.assertTrue(self._is_bos("I love to", _ServerTokenizer(hf=False)))

  def test_no_bos_when_tokenizer_has_no_bos_token(self):
    # Qwen3 tokenizers have a chat template but no BOS token.
    self.assertFalse(self._is_bos("I love to", _ServerTokenizer(bos_token=None, bos_id=None)))

  def test_no_bos_when_add_bos_is_false(self):
    self.assertFalse(self._is_bos("I love to", _ServerTokenizer(), add_bos=False))

  def test_no_bos_when_tokenizer_applies_chat_template(self):
    self.assertFalse(self._is_bos("I love to", _ServerTokenizer(), use_chat_template=True))

  def test_multimodal_prompt_gets_bos_when_reformat_prompt_does_not_write_it(self):
    # Gemma 3's `reformat_prompt` format has no BOS text.
    reformatted = "<start_of_turn>user\n<start_of_image>Describe.<end_of_turn>\n<start_of_turn>model\n"
    self.assertTrue(self._is_bos("Describe.", _ServerTokenizer(), reformatted_prompt=reformatted))

  def test_multimodal_prompt_gets_no_second_bos_when_reformat_prompt_writes_it(self):
    # Gemma 4's `reformat_prompt` format starts with `<bos>`.
    reformatted = "<bos><|turn>user\n<|image|>Describe.<turn|>\n<|turn>model\n"
    self.assertFalse(self._is_bos("Describe.", _ServerTokenizer(), reformatted_prompt=reformatted))


if __name__ == "__main__":
  unittest.main()
