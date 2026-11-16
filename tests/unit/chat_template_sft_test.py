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
from absl.testing import parameterized
import numpy as np
import pytest
from transformers import AutoTokenizer

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import ensure_tokenizer_downloaded

GEMMA4_TOKENIZER_PATH = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "gemma4-tokenizer")


@pytest.mark.external_training
class ChatTemplateGemma4ThinkingTest(parameterized.TestCase):
  """Tests chat template formatting and thinking channel boundaries for Gemma 4."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    ensure_tokenizer_downloaded("gemma4-tokenizer", GEMMA4_TOKENIZER_PATH, skip_test_on_failure=True)
    cls.tokenizer = AutoTokenizer.from_pretrained(GEMMA4_TOKENIZER_PATH)
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
    ensure_tokenizer_downloaded("gemma4-tokenizer", GEMMA4_TOKENIZER_PATH, skip_test_on_failure=True)
    cls.tokenizer = AutoTokenizer.from_pretrained(GEMMA4_TOKENIZER_PATH)
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

  def __init__(self, *, support_generation_blocks: bool = False):
    self._support_generation_blocks = support_generation_blocks
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
        if support_generation_blocks
        else None
    )
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

  def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **_kwargs):
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
      body = msg.get("content", "")
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
        mask = [
            1 if any(s <= start and end <= e for s, e in mask_spans) else 0
            for start, end in offsets
        ]
        out["assistant_tokens_mask"] = mask
      return out
    return ids


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
    self.assertEqual(processed["messages"], [expected_ids])
    # Also verify batched tokenization passes pre-tokenized document IDs through unchanged.
    batched = input_pipeline_utils.tokenization(
        {"messages": [processed["messages"]]},
        hf_tokenizer=tok,
        truncation=False,
        max_length=64,
        column_names=["messages"],
    )
    self.assertEqual(batched["messages"], [processed["messages"]])


if __name__ == "__main__":
  parameterized.absltest.main()
