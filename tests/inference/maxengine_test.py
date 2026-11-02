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

"""Unit tests for `maxengine.build_hf_tokenizer`.

The chat-template path of JetStream's `HuggingFaceTokenizer.encode` breaks on
transformers >= 5 (`apply_chat_template` returns a `BatchEncoding`, which has no
`.squeeze()`); `build_hf_tokenizer` wraps it so `use_chat_template=True` decoding
works on both transformers 4 and 5. A toy word-level tokenizer with a chat
template stands in for a real HF tokenizer, so nothing is downloaded.
"""

import tempfile
import unittest

import numpy as np

from maxtext.common.gcloud_stub import jetstream
from maxtext.inference.maxengine import maxengine

_, _, token_utils, _, _token_params_ns = jetstream()

_VOCAB = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "user": 3, "assistant": 4, "hello": 5, "world": 6}
_CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
)


def _save_toy_tokenizer(path):
  """Writes a word-level HF tokenizer with a chat template to `path`."""
  from tokenizers import Tokenizer, models, pre_tokenizers  # pylint: disable=import-outside-toplevel
  from transformers import PreTrainedTokenizerFast  # pylint: disable=import-outside-toplevel

  backend = Tokenizer(models.WordLevel(_VOCAB, unk_token="<unk>"))
  backend.pre_tokenizer = pre_tokenizers.Whitespace()
  tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>")
  tokenizer.chat_template = _CHAT_TEMPLATE
  tokenizer.save_pretrained(path)


def _metadata(path, use_chat_template):
  tokenizer_type = _token_params_ns.TokenizerType.DESCRIPTOR.values_by_name["huggingface"].number
  return _token_params_ns.TokenizerParameters(
      path=path, tokenizer_type=tokenizer_type, access_token="", use_chat_template=use_chat_template, extra_ids=0
  )


@unittest.skipIf(getattr(token_utils, "_IS_STUB", False), "JetStream is not installed")
class BuildHfTokenizerTest(unittest.TestCase):
  """`build_hf_tokenizer` must encode chat prompts on the installed transformers version."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._tmpdir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    _save_toy_tokenizer(cls._tmpdir.name)

  @classmethod
  def tearDownClass(cls):
    cls._tmpdir.cleanup()
    super().tearDownClass()

  def test_chat_template_encode_pads_the_templated_prompt(self):
    tokenizer = maxengine.build_hf_tokenizer(_metadata(self._tmpdir.name, use_chat_template=True))
    tokens, true_length = tokenizer.encode("hello world", is_bos=True, prefill_lengths=[8])
    # "user hello world assistant" through the template; no BOS, since the template owns the prompt.
    expected = [_VOCAB["user"], _VOCAB["hello"], _VOCAB["world"], _VOCAB["assistant"]]
    self.assertEqual(true_length, len(expected))
    self.assertEqual(tokens.shape, (8,))
    np.testing.assert_array_equal(np.asarray(tokens)[:true_length], expected)
    np.testing.assert_array_equal(np.asarray(tokens)[true_length:], _VOCAB["<pad>"])

  def test_plain_encode_matches_jetstream(self):
    metadata = _metadata(self._tmpdir.name, use_chat_template=False)
    ours, ours_length = maxengine.build_hf_tokenizer(metadata).encode("hello world", prefill_lengths=[8])
    theirs, theirs_length = token_utils.HuggingFaceTokenizer(metadata).encode("hello world", prefill_lengths=[8])
    self.assertEqual(ours_length, theirs_length)
    np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))


if __name__ == "__main__":
  unittest.main()
