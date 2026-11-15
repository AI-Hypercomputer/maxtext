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
"""Tests that a one-message `document` conversation is trained as plain text in SFT."""

import unittest

import numpy as np

from maxtext.input_pipeline import input_pipeline_utils


class _ByteTokenizer:
  """Tokenizes text to its UTF-8 bytes; it has a chat template that must not be used for documents."""

  eos_token_id = 1000

  def __call__(self, text, add_special_tokens=True):
    return {"input_ids": list(text.encode("utf-8"))}

  def get_chat_template(self):
    raise AssertionError("a document must not be templated")

  def apply_chat_template(self, *args, **kwargs):
    raise AssertionError("a document must not be templated")


class SFTDocumentRoleTest(unittest.TestCase):

  def test_document_is_one_trained_segment_ending_in_eos(self):
    text = "Long-context replay document.\nSecond line."
    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": [{"role": "document", "content": text}]},
        tokenizer_model=_ByteTokenizer(),
        data_column_name="messages",
    )
    self.assertEqual(processed["is_prompt"], [False])
    self.assertEqual(processed["messages"], [list(text.encode("utf-8")) + [_ByteTokenizer.eos_token_id]])

  def test_every_document_token_is_a_target(self):
    text = "abc"
    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": [{"role": "document", "content": text}]},
        tokenizer_model=_ByteTokenizer(),
        data_column_name="messages",
    )
    out = input_pipeline_utils.SFTPromptMasking("messages", completion_only=True, max_target_length=16).map(processed)
    np.testing.assert_array_equal(out["inputs"], [97, 98, 99, 1000])
    np.testing.assert_array_equal(out["targets"], out["inputs"])


if __name__ == "__main__":
  unittest.main()
