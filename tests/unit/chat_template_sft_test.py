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
"""Tests for chat template and SFT masking functionality."""

import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

from absl.testing import parameterized
import datasets
import numpy as np
import pytest
from transformers import AutoTokenizer
import yaml

from maxtext.configs import types as config_types
from maxtext.input_pipeline import grain_data_processing
from maxtext.input_pipeline import hf_data_processing
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.utils import globals as maxtext_globals
from tests.utils.test_helpers import ensure_tokenizer_downloaded

GEMMA4_TOKENIZER_PATH = os.path.join(maxtext_globals.MAXTEXT_ASSETS_ROOT, "tokenizers", "gemma4-tokenizer")
_QWEN3_BLAZE_PATH = os.path.join(maxtext_globals.MAXTEXT_PKG_DIR, "assets", "tokenizers", "qwen3-tokenizer")
_QWEN3_REPO_PATH = os.path.join(maxtext_globals.MAXTEXT_ASSETS_ROOT, "tokenizers", "qwen3-tokenizer")
QWEN3_TOKENIZER_PATH = _QWEN3_REPO_PATH if os.path.isdir(_QWEN3_REPO_PATH) else _QWEN3_BLAZE_PATH


def _get_gemma4_tokenizer_or_skip():
  """Loads the local Gemma 4 tokenizer, downloading or skipping when unavailable."""
  target_path = GEMMA4_TOKENIZER_PATH
  if os.path.isdir(target_path) and os.listdir(target_path):
    return AutoTokenizer.from_pretrained(target_path)
  if shutil.which("gcloud") is None:
    raise unittest.SkipTest("Skipping test: gcloud CLI is unavailable to download gemma4-tokenizer")
  parent_dir = os.path.dirname(target_path)
  if not os.path.isdir(parent_dir) or not os.access(parent_dir, os.W_OK):
    tmp_root = os.environ.get("TEST_TMPDIR") or tempfile.gettempdir()
    target_path = os.path.join(tmp_root, "gemma4-tokenizer")
  ensure_tokenizer_downloaded("gemma4-tokenizer", target_path, skip_test_on_failure=True)
  return AutoTokenizer.from_pretrained(target_path)


@pytest.mark.external_training
class ChatTemplateGemma4ThinkingTest(parameterized.TestCase):
  """Tests chat template formatting and thinking channel boundaries for Gemma 4."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tokenizer = _get_gemma4_tokenizer_or_skip()
    cls.unk_id = cls.tokenizer.unk_token_id if cls.tokenizer.unk_token_id is not None else 0

  def test_single_turn_with_thinking(self):
    """Verifies that thinking trace is cleanly placed inside <|channel>thought without duplicate channel tags."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are a TPU optimization expert."},
            {"role": "user", "content": "Optimize this JAX function."},
            {
                "role": "assistant",
                "reasoning": "We will reformulate this using a single vector reduction pass.",
                "content": "```python\nimport jax.numpy as jnp\n```",
            },
        ]
    }

    # Ground truth from tokenizer
    true_full_str = self.tokenizer.apply_chat_template(sample["messages"], add_generation_prompt=False, tokenize=False)

    # Process via MaxText input pipeline
    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": list(sample["messages"])},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )

    self.assertEqual(len(processed["messages"]), 2)
    self.assertEqual(processed["is_prompt"], [True, False])

    prompt_seg = processed["messages"][0]
    completion_seg = processed["messages"][1]

    # Check that concatenation exactly reproduces full chat template
    reconstructed_str = prompt_seg + completion_seg
    self.assertEqual(reconstructed_str, true_full_str)

    # Check boundaries
    self.assertTrue(prompt_seg.endswith("<|turn>model\n<|channel>thought\n"))
    self.assertTrue(prompt_seg.startswith("<bos><|turn>system\n"))
    self.assertTrue(completion_seg.startswith("We will reformulate"))
    self.assertIn("<channel|>```python\nimport jax.numpy as jnp\n```<turn|>\n", completion_seg)

    # Verify no premature <channel|> tag in the prompt
    self.assertNotIn("<channel|>", prompt_seg)

  @parameterized.named_parameters(
      ("absent_reasoning_field", "OMITTED"),
      ("empty_string_reasoning", ""),
      ("none_reasoning", None),
  )
  def test_turn_without_thinking(self, reasoning_val):
    """Verifies that omitted, empty string, or None reasoning produces clean output without thought tags."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
    }
    if reasoning_val != "OMITTED":
      sample["messages"][2]["reasoning"] = reasoning_val

    true_full_str = self.tokenizer.apply_chat_template(sample["messages"], add_generation_prompt=False, tokenize=False)

    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": list(sample["messages"])},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )

    self.assertEqual(len(processed["messages"]), 2)
    self.assertEqual(processed["is_prompt"], [True, False])

    prompt_seg = processed["messages"][0]
    completion_seg = processed["messages"][1]

    self.assertEqual(prompt_seg + completion_seg, true_full_str)
    self.assertTrue(prompt_seg.endswith("<|turn>model\n"))
    self.assertEqual(completion_seg, "The capital of France is Paris.<turn|>\n")

    # Verify that no thinking channel tags (<|channel>thought\n or <channel|>) are injected
    self.assertNotIn("<|channel>", prompt_seg)
    self.assertNotIn("<channel|>", prompt_seg)
    self.assertNotIn("<|channel>", completion_seg)
    self.assertNotIn("<channel|>", completion_seg)
    self.assertNotIn("thought", prompt_seg)
    self.assertNotIn("thought", completion_seg)

  def test_multi_turn_with_thinking(self):
    """Verifies whole-conversation multi-turn formatting matches Gemma 4's template without duplicate <bos>."""
    turn1 = [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Step 1: Write a function."},
        {"role": "assistant", "reasoning": "Thinking step 1...", "content": "def step1(): pass"},
    ]
    turn2 = [
        {"role": "user", "content": "Step 2: Add logging."},
        {"role": "assistant", "reasoning": "Thinking step 2...", "content": "def step2(): print('done')"},
    ]
    sample = {"messages": turn1 + turn2}

    true_full_str = self.tokenizer.apply_chat_template(sample["messages"], add_generation_prompt=False, tokenize=False)

    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": list(sample["messages"])},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )

    self.assertEqual(len(processed["messages"]), 4)
    self.assertEqual(processed["is_prompt"], [True, False, True, False])
    self.assertEqual("".join(processed["messages"]), true_full_str)

    # Turn 1: Gemma 4's template omits prior-turn thought traces in multi-turn history.
    self.assertTrue(processed["messages"][0].startswith("<bos><|turn>system\n"))
    self.assertTrue(processed["messages"][0].endswith("<|turn>model\n"))
    self.assertEqual(processed["messages"][1], "def step1(): pass<turn|>\n")

    # Turn 2: no duplicate <bos>; final assistant turn preserves thought channel.
    self.assertFalse(processed["messages"][2].startswith("<bos>"))
    self.assertTrue(processed["messages"][2].endswith("<|turn>model\n<|channel>thought\n"))
    self.assertTrue(processed["messages"][3].startswith("Thinking step 2..."))
    self.assertIn("<channel|>def step2(): print('done')<turn|>\n", processed["messages"][3])

  def test_multi_turn_with_tool_and_trailing_user(self):
    """Verifies Gemma 4 tool_calls/tool responses and trailing user turns are preserved and masked."""
    sample = {
        "messages": [
            {"role": "user", "content": "Check weather in NYC."},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "NYC"}}}
                ],
                "content": "",
            },
            {"role": "tool", "tool_call_id": "c1", "content": "72F sunny"},
            {"role": "assistant", "content": "It is 72F and sunny."},
            {"role": "user", "content": "Thanks!"},
        ]
    }
    true_full_str = self.tokenizer.apply_chat_template(sample["messages"], add_generation_prompt=False, tokenize=False)
    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": list(sample["messages"])},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )
    self.assertEqual("".join(processed["messages"]), true_full_str)
    self.assertEqual(processed["is_prompt"], [True, False, True, False, True])
    self.assertIn("call:get_weather", processed["messages"][1])
    self.assertIn("72F sunny", processed["messages"][2])
    self.assertEqual(processed["messages"][3], "It is 72F and sunny.<turn|>\n")
    self.assertIn("Thanks!", processed["messages"][4])
    self.assertEqual("".join(processed["messages"]).count("<bos>"), 1)

  def test_no_system_message(self):
    """Verifies handling when no system message is provided."""
    sample = {
        "messages": [
            {"role": "user", "content": "Direct question."},
            {
                "role": "assistant",
                "reasoning": "Direct thinking...",
                "content": "Direct answer.",
            },
        ]
    }

    true_full_str = self.tokenizer.apply_chat_template(sample["messages"], add_generation_prompt=False, tokenize=False)

    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": list(sample["messages"])},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )

    self.assertEqual(len(processed["messages"]), 2)
    self.assertEqual(processed["is_prompt"], [True, False])
    self.assertEqual(processed["messages"][0] + processed["messages"][1], true_full_str)

  def test_invalid_system_message_position(self):
    """Verifies that system message not at index 0 raises ValueError."""
    sample = {
        "messages": [
            {"role": "user", "content": "User first"},
            {"role": "system", "content": "System second (invalid)"},
            {"role": "assistant", "content": "Response"},
        ]
    }
    with self.assertRaises(ValueError) as ctx:
      input_pipeline_utils.apply_chat_template(
          example={"messages": sample["messages"]},
          tokenizer_model=self.tokenizer,
          data_column_name="messages",
      )
    self.assertIn("System message found at index 1", str(ctx.exception))


@pytest.mark.external_training
class SFTPromptMaskingEdgeCasesTest(parameterized.TestCase):
  """Tests end-to-end tokenization and loss masking behavior."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tokenizer = _get_gemma4_tokenizer_or_skip()
    cls.unk_id = cls.tokenizer.unk_token_id if cls.tokenizer.unk_token_id is not None else 0

  def test_sft_prompt_masking_completion_only_true(self):
    """Ensures that when completion_only=True, prompt tokens are masked to unk_id and targets only contain completion."""
    sample = {
        "messages": [
            {"role": "system", "content": "System instructions."},
            {"role": "user", "content": "Compute sum."},
            {
                "role": "assistant",
                "reasoning": "Let us add x + y.",
                "content": "sum = x + y",
            },
        ]
    }

    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": sample["messages"]},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )

    tok_example = input_pipeline_utils.tokenization(
        example=processed,
        hf_tokenizer=self.tokenizer,
        truncation=False,
        max_length=4096,
        column_names=["messages"],
    )

    masker = input_pipeline_utils.SFTPromptMasking(
        text_column_name="messages",
        completion_only=True,
        max_target_length=4096,
        unk_id=self.unk_id,
    )

    sft_out = masker.map(tok_example)
    inputs = sft_out["inputs"]
    targets = sft_out["targets"]

    self.assertEqual(len(inputs), len(targets))

    # All prompt tokens must be masked in targets
    prompt_len = len(tok_example["messages"][0])
    for i in range(prompt_len):
      self.assertEqual(targets[i], self.unk_id, f"Prompt token at {i} was not masked!")

    # All completion tokens must be unmasked (equal to inputs)
    for i in range(prompt_len, len(inputs)):
      self.assertEqual(targets[i], inputs[i], f"Completion token at {i} was incorrectly masked!")

    # Decode unmasked targets and verify thinking + code content
    trained_ids = [int(t) for t in targets if t != self.unk_id]
    trained_text = self.tokenizer.decode(trained_ids)
    self.assertTrue(trained_text.startswith("Let us add x + y."))
    self.assertIn("<channel|>sum = x + y<turn|>\n", trained_text)

  def test_sft_prompt_masking_completion_only_false(self):
    """Ensures that when completion_only=False, targets are identical to inputs."""
    sample = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
    }

    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": sample["messages"]},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )

    tok_example = input_pipeline_utils.tokenization(
        example=processed,
        hf_tokenizer=self.tokenizer,
        truncation=False,
        max_length=4096,
        column_names=["messages"],
    )

    masker = input_pipeline_utils.SFTPromptMasking(
        text_column_name="messages",
        completion_only=False,
        max_target_length=4096,
        unk_id=self.unk_id,
    )

    sft_out = masker.map(tok_example)
    np.testing.assert_array_equal(sft_out["inputs"], sft_out["targets"])

  @parameterized.named_parameters(
      ("omitted_reasoning_field", "OMITTED"),
      ("empty_string_reasoning", ""),
      ("none_reasoning", None),
  )
  def test_sft_prompt_masking_non_thinking(self, reasoning_val):
    """Ensures that for non-thinking conversations, no thought tags exist and targets only contain plain completion."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
    }
    if reasoning_val != "OMITTED":
      sample["messages"][2]["reasoning"] = reasoning_val

    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": sample["messages"]},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )

    tok_example = input_pipeline_utils.tokenization(
        example=processed,
        hf_tokenizer=self.tokenizer,
        truncation=False,
        max_length=4096,
        column_names=["messages"],
    )

    masker = input_pipeline_utils.SFTPromptMasking(
        text_column_name="messages",
        completion_only=True,
        max_target_length=4096,
        unk_id=self.unk_id,
    )

    sft_out = masker.map(tok_example)
    inputs = sft_out["inputs"]
    targets = sft_out["targets"]

    self.assertEqual(len(inputs), len(targets))

    # All prompt tokens must be masked in targets
    prompt_len = len(tok_example["messages"][0])
    for i in range(prompt_len):
      self.assertEqual(targets[i], self.unk_id, f"Prompt token at {i} was not masked!")

    # Decode unmasked targets and verify plain response without any thought channel tokens
    trained_ids = [int(t) for t in targets if t != self.unk_id]
    trained_text = self.tokenizer.decode(trained_ids)
    self.assertEqual(trained_text, "The capital of France is Paris.<turn|>\n")
    self.assertNotIn("<|channel>", trained_text)
    self.assertNotIn("<channel|>", trained_text)
    self.assertNotIn("thought", trained_text)


class _SyntheticWholeConversationTokenizer:
  """Minimal offline tokenizer for whole-conversation chat template regression tests.

  Models templates that inject a default system prompt once at turn 0 and emit an
  EOS token only after the final message (e.g., OLMo-3 Instruct).
  """

  def __init__(
      self,
      *,
      support_generation_blocks: bool = False,
      support_offsets: bool = True,
      merge_tokens: tuple[str, ...] = (),
      ignore_content_when_tool_calls: bool = False,
      diverge_user_before_assistant: bool = False,
      always_strip_earlier_turn: bool = False,
  ):
    self._support_generation_blocks = support_generation_blocks
    self._support_offsets = support_offsets
    self._merge_tokens = tuple(sorted(merge_tokens, key=len, reverse=True))
    self._ignore_content_when_tool_calls = ignore_content_when_tool_calls
    self._diverge_user_before_assistant = diverge_user_before_assistant
    self._always_strip_earlier_turn = always_strip_earlier_turn
    self.bos_token = "<|endoftext|>"
    self.eos_token = "<|endoftext|>"
    self.pad_token = "<|pad|>"
    self.eos_token_id = 1
    self.bos_token_id = 1
    self.pad_token_id = 0
    self.unk_token_id = 0
    self.special_tokens_map = {
        "bos_token": self.bos_token,
        "eos_token": self.eos_token,
        "pad_token": self.pad_token,
    }
    if support_generation_blocks:
      self.chat_template = (
          "{% if not messages or messages[0]['role'] != 'system' %}"
          "<|im_start|>system\nDefault system.<|im_end|>\n"
          "{% endif %}"
          "{% for message in messages %}"
          "<|im_start|>{{ message['role'] }}\n"
          "{% if message['role'] == 'assistant' %}"
          "{% generation %}{{ message['content'] }}"
          "{% if loop.last and not add_generation_prompt %}<|im_end|>{{ eos_token }}{% else %}<|im_end|>\n{% endif %}"
          "{% endgeneration %}"
          "{% else %}"
          "{{ message['content'] }}<|im_end|>\n"
          "{% endif %}"
          "{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
      )
    else:
      self.chat_template = None
    self._special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]
    self._vocab = {
        "<|pad|>": 0,
        "<|endoftext|>": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
    }
    self._inv = {v: k for k, v in self._vocab.items()}

  def get_chat_template(self):
    return self.chat_template

  def _tokenize_with_offsets(self, text: str):
    """Tokenizes text into integer IDs and character offset spans."""
    ids = []
    offsets = []
    i = 0
    n = len(text)
    while i < n:
      matched = None
      for sp in self._special:
        if text.startswith(sp, i):
          matched = sp
          break
      if matched is None:
        for merged in self._merge_tokens:
          if text.startswith(merged, i):
            matched = merged
            break
      if matched is not None:
        tok = matched
        end = i + len(matched)
      else:
        tok = text[i]
        end = i + 1
      if tok not in self._vocab:
        idx = len(self._vocab)
        self._vocab[tok] = idx
        self._inv[idx] = tok
      ids.append(self._vocab[tok])
      offsets.append((i, end))
      i = end
    return ids, offsets

  def encode(self, text: str):
    return self._tokenize_with_offsets(text)[0]

  def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_kwargs):
    if return_offsets_mapping and not self._support_offsets:
      raise NotImplementedError("Slow tokenizer does not support return_offsets_mapping.")
    if isinstance(text, list):
      all_ids = [self._tokenize_with_offsets(t)[0] for t in text]
      return {"input_ids": all_ids}
    ids, offsets = self._tokenize_with_offsets(text)
    out = {"input_ids": ids}
    if return_offsets_mapping:
      out["offset_mapping"] = offsets
    return out

  def decode(self, ids, skip_special_tokens=False):
    if skip_special_tokens:
      return "".join(self._inv[i] for i in ids if self._inv[i] not in self._special)
    return "".join(self._inv[i] for i in ids)

  def apply_chat_template(
      self,
      messages,
      add_generation_prompt=False,
      tokenize=False,
      return_dict=False,
      return_assistant_tokens_mask=False,
  ):
    """Renders messages with optional default system prompt, EOS, and generation mask."""
    pieces = []
    mask_spans = []
    pos = 0
    has_sys = bool(messages and messages[0]["role"] == "system")
    if not has_sys:
      default_sys = "<|im_start|>system\nDefault system.<|im_end|>\n"
      pieces.append(default_sys)
      pos += len(default_sys)
    for idx, msg in enumerate(messages):
      header = f"<|im_start|>{msg['role']}\n"
      pieces.append(header)
      pos += len(header)
      body_start = pos
      if self._ignore_content_when_tool_calls and msg.get("tool_calls"):
        body = f"<tool>{msg['tool_calls'][0]['name']}</tool>"
      else:
        body = msg.get("content") or ""
      if self._diverge_user_before_assistant and msg["role"] == "user":
        has_following_asst = any(m["role"] == "assistant" for m in messages[idx + 1 :])
        body = f"[hist]{body}" if has_following_asst else body
      if self._always_strip_earlier_turn and idx == 0 and len(messages) > 2:
        body = f"[stripped]{body}"
      if msg["role"] == "assistant" and idx == len(messages) - 1 and not add_generation_prompt:
        closer = "<|im_end|><|endoftext|>"
      else:
        closer = "<|im_end|>\n"
      chunk = body + closer
      pieces.append(chunk)
      pos += len(chunk)
      if msg["role"] == "assistant":
        mask_spans.append((body_start, pos))
    if add_generation_prompt:
      gen = "<|im_start|>assistant\n"
      pieces.append(gen)
      pos += len(gen)
    rendered = "".join(pieces)
    if not tokenize:
      return rendered
    ids, offsets = self._tokenize_with_offsets(rendered)
    if return_dict:
      out = {"input_ids": ids}
      if return_assistant_tokens_mask and self._support_generation_blocks:
        mask = [1 if any(s <= start and end <= e for s, e in mask_spans) else 0 for start, end in offsets]
        out["assistant_tokens_mask"] = mask
      return out
    return ids


class _PrefixStableCounterTokenizer(_SyntheticWholeConversationTokenizer):
  """Prefix-stable tokenizer that counts tokenize=True invocations."""

  def __init__(self):
    super().__init__()
    self.tokenize_calls = 0

  def apply_chat_template(
      self,
      messages,
      add_generation_prompt=False,
      tokenize=False,
      return_dict=False,
      return_assistant_tokens_mask=False,
  ):
    """Renders prefix-stable chat text and tracks tokenize=True calls."""
    del return_dict, return_assistant_tokens_mask
    if tokenize:
      self.tokenize_calls += 1
    rendered = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    if add_generation_prompt:
      rendered += "<|im_start|>assistant\n"
    if not tokenize:
      return rendered
    return self._tokenize_with_offsets(rendered)[0]


class ChatTemplateWholeConversationTest(parameterized.TestCase):
  """Offline regression tests for whole-conversation chat template splitting."""

  def test_multi_turn_default_system_and_end_only_eos(self):
    """Default system prompt and final EOS must each appear once in multi-turn conversations."""
    tok = _SyntheticWholeConversationTokenizer()
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    expected_full = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages")

    self.assertEqual(processed["is_prompt"], [True, False, True, False])
    self.assertEqual("".join(processed["messages"]), expected_full)
    self.assertEqual("".join(processed["messages"]).count("Default system."), 1)
    self.assertEqual("".join(processed["messages"]).count("<|endoftext|>"), 1)
    self.assertEqual(processed["messages"][1], "A1<|im_end|>\n")
    self.assertEqual(processed["messages"][3], "A2<|im_end|><|endoftext|>")

  def test_tool_and_trailing_user_messages_are_preserved_and_masked(self):
    """Non-user/non-assistant roles (e.g. tool) and trailing user turns stay intact and masked."""
    tok = _SyntheticWholeConversationTokenizer()
    messages = [
        {"role": "user", "content": "Run calc"},
        {"role": "assistant", "content": "calc(1+1)"},
        {"role": "tool", "content": "2"},
        {"role": "assistant", "content": "The answer is 2."},
        {"role": "user", "content": "Bye"},
    ]
    expected_full = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages")

    self.assertEqual(processed["is_prompt"], [True, False, True, False, True])
    self.assertEqual("".join(processed["messages"]), expected_full)
    self.assertEqual(processed["messages"][1], "calc(1+1)<|im_end|>\n")
    self.assertIn("<|im_start|>tool\n2<|im_end|>\n", processed["messages"][2])
    self.assertEqual(processed["messages"][3], "The answer is 2.<|im_end|>\n")
    self.assertIn("<|im_start|>user\nBye<|im_end|>", processed["messages"][4])

  def test_generation_blocks_mask_is_used_when_supported(self):
    """Templates with {% generation %} blocks use assistant_tokens_mask directly."""
    tok = _SyntheticWholeConversationTokenizer(support_generation_blocks=True)
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "More"},
        {"role": "assistant", "content": "Sure."},
    ]
    expected_full = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages")

    self.assertEqual(processed["is_prompt"], [True, False, True, False])
    self.assertEqual("".join(processed["messages"]), expected_full)
    self.assertEqual(processed["messages"][1], "Hello!<|im_end|>\n")
    self.assertEqual(processed["messages"][3], "Sure.<|im_end|><|endoftext|>")

  def test_document_role_passthrough_trains_full_text_plus_eos(self):
    """Single-message role='document' bypasses chat template and trains on raw text + EOS."""
    tok = _SyntheticWholeConversationTokenizer()
    processed = input_pipeline_utils.apply_chat_template(
        {"messages": [{"role": "document", "content": "Raw text."}]}, tok, "messages"
    )
    self.assertEqual(processed["is_prompt"], [False])
    expected_ids = tok("Raw text.", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    self.assertEqual(processed["messages"], ["Raw text.<|endoftext|>"])
    self.assertEqual(processed[input_pipeline_utils.SFT_TOKEN_IDS_COLUMN], [expected_ids])
    # Also verify batched tokenization passes pre-tokenized document IDs through unchanged.
    batched = input_pipeline_utils.tokenization(
        {
            "messages": [processed["messages"]],
            input_pipeline_utils.SFT_TOKEN_IDS_COLUMN: [processed[input_pipeline_utils.SFT_TOKEN_IDS_COLUMN]],
        },
        hf_tokenizer=tok,
        truncation=False,
        max_length=64,
        column_names=["messages"],
    )
    self.assertEqual(batched["messages"], [[expected_ids]])

  def test_boundary_merge_preserves_exact_ids_across_hf_and_grain(self):
    """Boundary-merged tokens keep exact whole-conversation IDs in both HF and Grain SFT pipelines."""
    tok = _SyntheticWholeConversationTokenizer(merge_tokens=("\nA",))
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]
    full_ids = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages")
    self.assertEqual(processed["is_prompt"], [True, False])
    self.assertEqual(sum(processed[input_pipeline_utils.SFT_TOKEN_IDS_COLUMN], []), full_ids)

    # HF tokenization + SFTPromptMasking
    hf_tok = input_pipeline_utils.tokenization(
        dict(processed), hf_tokenizer=tok, truncation=False, max_length=128, column_names=["messages"]
    )
    masker = input_pipeline_utils.SFTPromptMasking(
        text_column_name="messages", completion_only=True, max_target_length=128, unk_id=tok.pad_token_id
    )
    hf_out = masker.map(hf_tok)
    self.assertEqual(hf_out["inputs"].tolist(), full_ids)
    prompt_len = len(processed[input_pipeline_utils.SFT_TOKEN_IDS_COLUMN][0])
    self.assertTrue(all(t == tok.pad_token_id for t in hf_out["targets"][:prompt_len]))
    self.assertEqual(hf_out["targets"][prompt_len:].tolist(), full_ids[prompt_len:])

    # Grain _tokenize_sft_chunks (pre-tokenized SFT column and direct ID-list passthrough)
    grain_elem = grain_data_processing._tokenize_sft_chunks(  # pylint: disable=protected-access
        dict(processed), text_column_name="messages", tokenizer_model=tok
    )
    self.assertEqual(grain_elem["messages"], processed[input_pipeline_utils.SFT_TOKEN_IDS_COLUMN])
    self.assertNotIn(input_pipeline_utils.SFT_TOKEN_IDS_COLUMN, grain_elem)
    grain_out = masker.map(grain_elem)
    np.testing.assert_array_equal(grain_out["inputs"], hf_out["inputs"])
    np.testing.assert_array_equal(grain_out["targets"], hf_out["targets"])

    # Direct mixed chunk passthrough (strings + pre-tokenized IDs without SFT_TOKEN_IDS_COLUMN)
    direct = grain_data_processing._tokenize_sft_chunks(  # pylint: disable=protected-access
        {"messages": ["Hi", [10, 11]]}, text_column_name="messages", tokenizer_model=tok
    )
    self.assertEqual(direct["messages"], [tok.encode("Hi"), [10, 11]])

  def test_mixed_chat_document_and_boundary_merge_dataset_map_arrow_safe(self):
    """Map-style PyArrow Dataset.map handles mixed chat, document, and boundary-merge rows cleanly."""
    tok = _SyntheticWholeConversationTokenizer(merge_tokens=("\nA",))
    ds = datasets.Dataset.from_dict(
        {
            "messages": [
                [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "B1"}],
                [{"role": "document", "content": "Raw doc"}],
                [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}],
            ]
        }
    )
    mapped = ds.map(
        input_pipeline_utils.apply_chat_template,
        fn_kwargs={"tokenizer_model": tok, "data_column_name": "messages"},
    )
    tokenized = mapped.map(
        input_pipeline_utils.tokenization,
        batched=True,
        fn_kwargs={"hf_tokenizer": tok, "truncation": False, "max_length": 128, "column_names": ["messages"]},
    )
    self.assertEqual(tokenized[0]["is_prompt"], [True, False])
    self.assertEqual(tokenized[1]["is_prompt"], [False])
    self.assertEqual(tokenized[2]["is_prompt"], [True, False])
    self.assertEqual(
        sum(tokenized[1]["messages"], []),
        tok("Raw doc", add_special_tokens=False)["input_ids"] + [tok.eos_token_id],
    )
    self.assertEqual(
        sum(tokenized[2]["messages"], []),
        tok.apply_chat_template(
            [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}],
            add_generation_prompt=False,
            tokenize=True,
        ),
    )

  def test_generation_blocks_slow_tokenizer_fallback(self):
    """Slow tokenizers without return_offsets_mapping succeed in _generation_block_segments."""
    tok = _SyntheticWholeConversationTokenizer(
        support_generation_blocks=True, support_offsets=False, merge_tokens=("\nA",)
    )
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Next"},
        {"role": "assistant", "content": "A2"},
    ]
    expected_full_ids = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
    processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages")
    self.assertEqual(processed["is_prompt"], [True, False, True, False])
    self.assertEqual(sum(processed[input_pipeline_utils.SFT_TOKEN_IDS_COLUMN], []), expected_full_ids)
    self.assertIn("A1<|im_end|>\n", processed["messages"][1])
    self.assertIn("A2<|im_end|><|endoftext|>", processed["messages"][3])

  def test_empty_content_tool_calls_turn_does_not_leak_subsequent_turns(self):
    """Empty-content tool_calls turns in non-prefix-stable templates never leak later turns into loss."""
    messages = [
        {"role": "user", "content": "Check weather"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "get_weather"}]},
        {"role": "tool", "content": "72F sunny"},
        {"role": "assistant", "content": "It is 72F."},
    ]
    # Recoverable non-prefix-stable case (OLMo-3 style last-turn EOS): lcp_thru locates turn end cleanly.
    tok = _SyntheticWholeConversationTokenizer(ignore_content_when_tool_calls=True)
    processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages")
    self.assertEqual(processed["is_prompt"], [True, False, True, False])
    self.assertIn("<tool>get_weather</tool><|im_end|>", processed["messages"][1])
    self.assertNotIn("72F sunny", processed["messages"][1])
    self.assertIn("72F sunny", processed["messages"][2])
    self.assertEqual(processed["messages"][3], "It is 72F.<|im_end|><|endoftext|>")

    # Unrecoverable case (earlier turn also diverged): masks whole conversation instead of leaking tool/user turns.
    tok_diverged = _SyntheticWholeConversationTokenizer(
        ignore_content_when_tool_calls=True, always_strip_earlier_turn=True
    )
    masked = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok_diverged, "messages")
    self.assertEqual(masked["is_prompt"], [True])

    # Direct check on _assistant_span_is_clean when content is empty/None.
    rendered = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    self.assertFalse(
        input_pipeline_utils._assistant_span_is_clean(  # pylint: disable=protected-access
            messages, 1, rendered, "<tool>get_weather</tool><|im_end|>\n<|im_start|>tool\n72F sunny"
        )
    )

  def test_assistant_first_turn_masks_leading_prefix(self):
    """A conversation starting directly with assistant still masks the default system prompt and role header."""
    tok = _SyntheticWholeConversationTokenizer()
    messages = [{"role": "assistant", "content": "Hello!"}]
    expected_full = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages")
    self.assertEqual("".join(processed["messages"]), expected_full)
    self.assertEqual(processed["is_prompt"], [True, False])
    self.assertTrue(processed["messages"][0].endswith("<|im_start|>assistant\n"))
    self.assertEqual(processed["messages"][1], "Hello!<|im_end|><|endoftext|>")

  def test_diverged_preceding_user_turn_recovers_start(self):
    """Divergence inside a user turn before the first assistant turn recovers cleanly via _find_turn_start_char."""
    tok = _SyntheticWholeConversationTokenizer(diverge_user_before_assistant=True)
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    expected_full = tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages")
    self.assertEqual("".join(processed["messages"]), expected_full)
    self.assertEqual(processed["is_prompt"], [True, False])
    self.assertTrue(processed["messages"][0].endswith("<|im_start|>assistant\n"))
    self.assertEqual(processed["messages"][1], "Answer<|im_end|><|endoftext|>")

  def test_prefix_stable_boundary_merge_does_not_overshoot_next_turn(self):
    """A boundary merge between turns in a prefix-stable template never overshoots into the next prompt."""

    class _PrefixStableMergeTokenizer(_SyntheticWholeConversationTokenizer):
      """Tokenizer with plain delimiters that can merge across turn boundaries."""

      def apply_chat_template(
          self,
          messages,
          add_generation_prompt=False,
          tokenize=False,
          return_dict=False,
          return_assistant_tokens_mask=False,
      ):
        """Renders messages with plain delimiters that allow boundary merges."""
        del return_dict, return_assistant_tokens_mask
        # Plain delimiters with no special tokens so merge_tokens can fuse across turn boundaries.
        parts = [f"{m['role']}:{m['content']}." for m in messages]
        if add_generation_prompt:
          parts.append("assistant:")
        rendered = "".join(parts)
        if not tokenize:
          return rendered
        return self._tokenize_with_offsets(rendered)[0]

    tok = _PrefixStableMergeTokenizer(merge_tokens=(".user:",))
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages")
    self.assertEqual(processed["is_prompt"], [True, False, True, False])
    self.assertNotIn("Q2", processed["messages"][1])
    self.assertIn("Q2", processed["messages"][2])
    self.assertEqual(processed["messages"][3], "A2.")

  def test_hf_causal_shift_bos_masking_policy(self):
    """HF causal shift ignores distinct BOS targets, while preserving EOS when BOS is None or equals EOS."""
    cfg = types.SimpleNamespace(training_objective="causal_lm")
    with_bos = hf_data_processing._get_training_objective_transform(  # pylint: disable=protected-access
        cfg,
        shift=True,
        use_dpo=False,
        use_sft=False,
        packing=True,
        pad_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    without_bos = hf_data_processing._get_training_objective_transform(  # pylint: disable=protected-access
        cfg,
        shift=True,
        use_dpo=False,
        use_sft=False,
        packing=True,
        pad_id=0,
        bos_token_id=None,
    )
    bos_equals_eos = hf_data_processing._get_training_objective_transform(  # pylint: disable=protected-access
        cfg,
        shift=True,
        use_dpo=False,
        use_sft=True,
        packing=True,
        pad_id=0,
        bos_token_id=100257,
        eos_token_id=100257,
    )
    self.assertIsInstance(with_bos, input_pipeline_utils.ShiftData)
    self.assertEqual(with_bos.ignored_ids, [0, 1])
    self.assertIsInstance(without_bos, input_pipeline_utils.ShiftData)
    self.assertEqual(without_bos.ignored_ids, [0])
    self.assertIsInstance(bos_equals_eos, input_pipeline_utils.ShiftData)
    self.assertEqual(bos_equals_eos.ignored_ids, [0])

  def test_short_later_turn_matching_closer_is_not_masked(self):
    """Short later turns like 'im', 'd', 'end', or 'turn' do not trigger false leak detection."""
    qwen_tok = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER_PATH)
    for short_reply in ("im", "d", "e", "end", "turn"):
      messages = [
          {"role": "user", "content": "Q1"},
          {"role": "assistant", "reasoning_content": "t", "content": "A1"},
          {"role": "user", "content": short_reply},
          {"role": "assistant", "content": "A2"},
      ]
      expected_full_ids = input_pipeline_utils.extract_token_ids(
          qwen_tok.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
      )
      processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, qwen_tok, "messages")
      self.assertEqual(processed["is_prompt"], [True, False, True, False])
      self.assertEqual(sum(processed[input_pipeline_utils.SFT_TOKEN_IDS_COLUMN], []), expected_full_ids)
      self.assertEqual(processed["messages"][1], "A1<|im_end|>\n")
      self.assertIn(f"<|im_start|>user\n{short_reply}<|im_end|>", processed["messages"][2])
      self.assertEqual(processed["messages"][3], "<think>\n\n</think>\n\nA2<|im_end|>\n")

    # Slow tokenizer (no offsets) with a short later turn also avoids false positives while catching real leaks.
    slow_tok = _SyntheticWholeConversationTokenizer(support_offsets=False)
    slow_msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "im"},
        {"role": "assistant", "content": "A2"},
    ]
    slow_out = input_pipeline_utils.apply_chat_template({"messages": list(slow_msgs)}, slow_tok, "messages")
    self.assertEqual(slow_out["is_prompt"], [True, False, True, False])
    leak_msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "t", "content": "A1"},
        {"role": "user", "content": "im"},
        {"role": "assistant", "content": "A2"},
    ]
    rendered = qwen_tok.apply_chat_template(leak_msgs, add_generation_prompt=False, tokenize=False)
    self.assertFalse(
        input_pipeline_utils._assistant_span_is_clean(  # pylint: disable=protected-access
            leak_msgs,
            1,
            rendered,
            "A1<|im_end|>\n<|im_start|>user\nim",
            closer="<|im_end|>\n",
        )
    )
    self.assertFalse(
        input_pipeline_utils._assistant_span_is_clean(  # pylint: disable=protected-access
            leak_msgs,
            1,
            rendered,
            "A1<|im_end|>\n",
            span_char_end=100,
            next_char_start=90,
        )
    )

  def test_generation_block_terminator_outside_falls_back_to_prefix(self):
    """Terminator placed after {% endgeneration %} falls back to prefix masking with warning + counter."""
    qwen_tok = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER_PATH)
    qwen_tok.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }}{% endgeneration %}<|im_end|>\n"
        "{% else %}{{ message['content'] }}<|im_end|>\n{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    with mock.patch.dict(input_pipeline_utils._SFT_FALLBACK_COUNTS, {}, clear=True):  # pylint: disable=protected-access
      with mock.patch.object(input_pipeline_utils.max_logging, "warning") as warn_mock:
        processed = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, qwen_tok, "messages")
      after = input_pipeline_utils.get_sft_masking_fallback_counts_for_test().get(
          "terminator_outside_generation_block", 0
      )
      self.assertEqual(after, 1)
      warn_mock.assert_called_once()
      self.assertEqual(processed["is_prompt"], [True, False, True, False])
      self.assertEqual(processed["messages"][1], "A1<|im_end|>\n")
      self.assertEqual(processed["messages"][3], "A2<|im_end|>\n")

    # Empty generation blocks on a conversation with assistant turns logs a warning and increments counter.
    qwen_tok.chat_template = (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% if false %}{% generation %}{% endgeneration %}{% endif %}"
        "{% endfor %}"
    )
    with mock.patch.dict(input_pipeline_utils._SFT_FALLBACK_COUNTS, {}, clear=True):  # pylint: disable=protected-access
      with mock.patch.object(input_pipeline_utils.max_logging, "warning") as warn_mock:
        masked = input_pipeline_utils.apply_chat_template({"messages": list(messages)}, qwen_tok, "messages")
      empty_after = input_pipeline_utils.get_sft_masking_fallback_counts_for_test().get("empty_generation_blocks", 0)
      self.assertEqual(empty_after, 1)
      warn_mock.assert_called_once()
      self.assertEqual(masked["is_prompt"], [True])

  def test_document_role_raises_when_eos_token_id_is_none(self):
    """Document passthrough fails fast when tokenizer.eos_token_id is None."""
    tok = _SyntheticWholeConversationTokenizer()
    tok.eos_token_id = None
    with self.assertRaisesRegex(ValueError, "eos_token_id is None"):
      input_pipeline_utils.apply_chat_template(
          {"messages": [{"role": "document", "content": "Raw text"}]}, tok, "messages"
      )

  def test_sft_label_mode_config_validation_and_dispatch(self):
    """sft_label_mode is validated, overrides generation blocks when 'prefix', and matches base/sft YAML."""
    tok = _SyntheticWholeConversationTokenizer(support_generation_blocks=True)
    messages = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
    with mock.patch.object(
        input_pipeline_utils,
        "_generation_block_segments",
        wraps=input_pipeline_utils._generation_block_segments,  # pylint: disable=protected-access
    ) as gen_mock:
      prefix_out = grain_data_processing._format_chat_template_grain(  # pylint: disable=protected-access
          {"messages": list(messages)},
          data_columns=["messages"],
          tokenizer_model=tok,
          label_mode="prefix",
      )
      gen_mock.assert_not_called()
    self.assertEqual(prefix_out["is_prompt"], [True, False])

    with self.assertRaisesRegex(ValueError, "Unsupported sft_label_mode"):
      input_pipeline_utils.apply_chat_template({"messages": list(messages)}, tok, "messages", label_mode="bad-mode")

    with open(os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "base.yml"), "r", encoding="utf-8") as f:
      base_cfg = yaml.safe_load(f)
    with open(os.path.join(maxtext_globals.MAXTEXT_CONFIGS_DIR, "post_train", "sft.yml"), "r", encoding="utf-8") as f:
      sft_cfg = yaml.safe_load(f)
    self.assertEqual(base_cfg["sft_label_mode"], "generation_blocks")
    self.assertEqual(sft_cfg["sft_label_mode"], "generation_blocks")
    self.assertEqual(config_types.MaxTextConfig(run_name="test").sft_label_mode, "generation_blocks")
    self.assertEqual(config_types.MaxTextConfig(run_name="test", sft_label_mode="prefix").sft_label_mode, "prefix")
    with self.assertRaisesRegex(ValueError, "sft_label_mode"):
      config_types.MaxTextConfig(run_name="test", sft_label_mode="generation-blocks")

    dataset = mock.MagicMock()
    dataset.map.return_value = dataset
    raw_tok = types.SimpleNamespace(pad_token_id=50256, eos_token_id=50256, chat_template=None)
    wrapped_tok = types.SimpleNamespace(tokenizer=raw_tok, pad_id=50256)
    cfg = types.SimpleNamespace(
        sft_train_on_completion_only=True,
        max_target_length=16,
        sft_label_mode="prefix",
        model_name="qwen3-0.6b",
    )
    with (
        mock.patch.object(grain_data_processing.data_processing_utils, "parse_and_keep_features", return_value=dataset),
        mock.patch.object(
            grain_data_processing.data_processing_utils, "get_tokenizer_and_pad_id", return_value=(wrapped_tok, 50256)
        ),
        mock.patch.object(grain_data_processing.data_processing_utils, "validate_and_configure_sft_columns"),
        mock.patch.object(grain_data_processing.data_processing_utils, "get_local_batch_size", return_value=1),
        mock.patch.object(grain_data_processing.data_processing_utils, "format_and_batch", return_value=dataset),
        mock.patch.object(
            grain_data_processing.data_processing_utils, "apply_multiprocessing_and_prefetch", return_value=dataset
        ),
        mock.patch.object(grain_data_processing.max_logging, "warning") as grain_warn_mock,
    ):
      grain_data_processing.sft_preprocessing_pipeline(
          dataset,
          cfg,
          data_columns=["messages"],
          tokenize=False,
          grain_worker_count=0,
          grain_per_worker_buffer_size=1,
      )
    grain_warn_mock.assert_called_once()
    format_fn = grain_data_processing._format_chat_template_grain  # pylint: disable=protected-access
    format_call = next(
        call.args[0]
        for call in dataset.map.call_args_list
        if call.args and getattr(call.args[0], "func", None) is format_fn
    )
    self.assertEqual(format_call.keywords["label_mode"], "prefix")

  def test_prefix_path_avoids_per_turn_retokenization_and_handles_long_qwen3_turns(self):
    """Prefix-stable avoids per-turn tokenize=True calls and long multi-turn Qwen3 completes with O(1) renders."""
    tok_stable = _PrefixStableCounterTokenizer()
    long_conv = []
    for i in range(20):
      long_conv.append({"role": "user", "content": f"Question {i}"})
      long_conv.append({"role": "assistant", "content": f"Answer {i}"})
    stable_out = input_pipeline_utils.apply_chat_template({"messages": list(long_conv)}, tok_stable, "messages")
    self.assertEqual(stable_out["is_prompt"], [True, False] * 20)
    self.assertEqual(tok_stable.tokenize_calls, 1)

    qwen_tok = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER_PATH)
    qwen_conv = []
    user_body = " ".join(["question"] * 30)
    asst_body = " ".join(["answer"] * 60)
    for i in range(60):
      qwen_conv.append({"role": "user", "content": f"u{i} {user_body}"})
      qwen_conv.append({"role": "assistant", "content": f"a{i} {asst_body}"})
    with mock.patch.object(qwen_tok, "apply_chat_template", wraps=qwen_tok.apply_chat_template) as tmpl_spy:
      qwen_out = input_pipeline_utils.apply_chat_template({"messages": list(qwen_conv)}, qwen_tok, "messages")
    self.assertLessEqual(tmpl_spy.call_count, 12)
    tokenize_calls = sum(1 for call in tmpl_spy.call_args_list if call.kwargs.get("tokenize"))
    self.assertEqual(tokenize_calls, 1)
    expected_full_ids = input_pipeline_utils.extract_token_ids(
        qwen_tok.apply_chat_template(qwen_conv, add_generation_prompt=False, tokenize=True)
    )
    self.assertEqual(qwen_out["is_prompt"], [True, False] * 60)
    self.assertEqual(sum(qwen_out[input_pipeline_utils.SFT_TOKEN_IDS_COLUMN], []), expected_full_ids)


if __name__ == "__main__":
  parameterized.absltest.main()
