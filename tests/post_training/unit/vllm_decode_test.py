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

from maxtext.inference.vllm_decode import build_chat_messages


def _config(prompt: str, system_prompt: str):
  return types.SimpleNamespace(prompt=prompt, system_prompt=system_prompt)


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


if __name__ == "__main__":
  unittest.main()
