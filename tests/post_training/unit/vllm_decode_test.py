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

"""Unit tests for vllm_decode helpers."""

import types
import unittest

import pytest

pytest.importorskip("vllm")
pytest.importorskip("tunix")

pytestmark = pytest.mark.post_training

from maxtext.inference.vllm_decode import build_chat_messages
from maxtext.integration.vllm.maxtext_vllm_adapter.multimodal import get_multimodal_handler


def _config(prompt: str, system_prompt: str, use_multimodal: bool = False, image_path: str = ""):
  return types.SimpleNamespace(
      prompt=prompt,
      system_prompt=system_prompt,
      use_multimodal=use_multimodal,
      image_path=image_path,
  )


class BuildChatMessagesTest(unittest.TestCase):
  """Chat-message construction for the vllm_decode CLI."""

  def test_user_only_when_no_system_prompt(self):
    messages = build_chat_messages(_config("What is 2+2?", ""))
    self.assertEqual(messages, [{"role": "user", "content": "What is 2+2?"}])

  def test_system_prompt_prepended(self):
    messages = build_chat_messages(_config("Who was Albert Einstein?", "You are a helpful assistant."))
    self.assertEqual(
        messages,
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who was Albert Einstein?"},
        ],
    )


class MultimodalHandlerTest(unittest.TestCase):
  """Model-family selection for vLLM multimodal handling."""

  def test_handler_selection(self):
    handler = get_multimodal_handler("qwen3-vl-2b")

    self.assertIsNotNone(handler)
    self.assertEqual(handler.placeholder_token_ids(types.SimpleNamespace(image_token_id=42)), [42])
    self.assertIsNone(get_multimodal_handler("qwen3-30b-a3b"))


class MultimodalChatMessagesTest(unittest.TestCase):
  """Multimodal chat-message construction for the vllm_decode CLI."""

  def test_multimodal_content_contains_one_item_per_image(self):
    messages = build_chat_messages(_config("Compare these.", "", True, "first.jpg,second.jpg"))
    self.assertEqual(
        messages,
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": "Compare these."},
                ],
            }
        ],
    )


if __name__ == "__main__":
  unittest.main()
