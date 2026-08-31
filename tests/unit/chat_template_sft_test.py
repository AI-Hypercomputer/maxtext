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

import os
import unittest
import numpy as np
import pytest
from transformers import AutoTokenizer

from maxtext.input_pipeline import input_pipeline_utils
from maxtext.utils.globals import MAXTEXT_ASSETS_ROOT
from tests.utils.test_helpers import ensure_tokenizer_downloaded

GEMMA4_TOKENIZER_PATH = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "gemma4-tokenizer")


@pytest.mark.external_training
class ChatTemplateGemma4ThinkingTest(unittest.TestCase):
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

  def test_single_turn_without_thinking(self):
    """Verifies standard conversational turn without reasoning field."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
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

  def test_absent_reasoning_field_does_not_add_thought_channel_tags(self):
    """Explicitly verifies that when the reasoning field is not present, no thought tags are added."""
    sample = {
        "messages": [
            {"role": "user", "content": "Calculate 2 + 2"},
            {"role": "assistant", "content": "4"},
        ]
    }

    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": list(sample["messages"])},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )

    prompt_seg = processed["messages"][0]
    completion_seg = processed["messages"][1]

    # Prompt must end at model turn boundary without entering thought channel
    self.assertEqual(prompt_seg, "<bos><|turn>user\nCalculate 2 + 2<turn|>\n<|turn>model\n")
    self.assertEqual(completion_seg, "4<turn|>\n")

    self.assertNotIn("<|channel>", prompt_seg)
    self.assertNotIn("<channel|>", prompt_seg)
    self.assertNotIn("<|channel>", completion_seg)
    self.assertNotIn("<channel|>", completion_seg)

  def test_multi_turn_with_thinking(self):
    """Verifies multi-turn conversation with independent turn formatting."""
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

    processed = input_pipeline_utils.apply_chat_template(
        example={"messages": list(sample["messages"])},
        tokenizer_model=self.tokenizer,
        data_column_name="messages",
    )

    self.assertEqual(len(processed["messages"]), 4)
    self.assertEqual(processed["is_prompt"], [True, False, True, False])

    # Turn 1
    self.assertTrue(processed["messages"][0].endswith("<|turn>model\n<|channel>thought\n"))
    self.assertTrue(processed["messages"][1].startswith("Thinking step 1..."))
    self.assertIn("<channel|>def step1(): pass<turn|>\n", processed["messages"][1])

    # Turn 2
    self.assertTrue(processed["messages"][2].endswith("<|turn>model\n<|channel>thought\n"))
    self.assertTrue(processed["messages"][3].startswith("Thinking step 2..."))
    self.assertIn("<channel|>def step2(): print('done')<turn|>\n", processed["messages"][3])

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

  def test_empty_or_none_reasoning(self):
    """Verifies that empty string or None reasoning behaves like non-thinking without injecting thought tags."""
    for reasoning_val in ["", None]:
      sample = {
          "messages": [
              {"role": "user", "content": "Hi"},
              {"role": "assistant", "reasoning": reasoning_val, "content": "Hello!"},
          ]
      }
      if reasoning_val is None:
        del sample["messages"][1]["reasoning"]

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
      self.assertEqual(completion_seg, "Hello!<turn|>\n")

      # Explicitly verify no thought tags exist in prompt or completion
      self.assertNotIn("<|channel>", prompt_seg)
      self.assertNotIn("<channel|>", prompt_seg)
      self.assertNotIn("<|channel>", completion_seg)
      self.assertNotIn("<channel|>", completion_seg)
      self.assertNotIn("thought", prompt_seg)
      self.assertNotIn("thought", completion_seg)

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
class SFTPromptMaskingEdgeCasesTest(unittest.TestCase):
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

  def test_sft_prompt_masking_without_reasoning(self):
    """Ensures that for conversations without reasoning, no thought tags exist and targets only contain the plain completion."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
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

    # Decode unmasked targets and verify plain response without any thought channel tokens
    trained_ids = [int(t) for t in targets if t != self.unk_id]
    trained_text = self.tokenizer.decode(trained_ids)
    self.assertEqual(trained_text, "The capital of France is Paris.<turn|>\n")
    self.assertNotIn("<|channel>", trained_text)
    self.assertNotIn("<channel|>", trained_text)
    self.assertNotIn("thought", trained_text)

  def test_sft_prompt_masking_empty_reasoning(self):
    """Ensures that when reasoning field is present but empty (''), no thought tags are added and targets only contain completion."""
    sample = {
        "messages": [
            {"role": "user", "content": "What is the capital of Japan?"},
            {"role": "assistant", "reasoning": "", "content": "Tokyo"},
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

    prompt_len = len(tok_example["messages"][0])
    for i in range(prompt_len):
      self.assertEqual(targets[i], self.unk_id)

    trained_ids = [int(t) for t in targets if t != self.unk_id]
    trained_text = self.tokenizer.decode(trained_ids)
    self.assertEqual(trained_text, "Tokyo<turn|>\n")
    self.assertNotIn("<|channel>", trained_text)
    self.assertNotIn("<channel|>", trained_text)
    self.assertNotIn("thought", trained_text)


if __name__ == "__main__":
  unittest.main()
