# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Omni multimodal evaluation runner (eval_sft_omni.py)."""

from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image

from maxtext.experimental.omni_poc import eval_sft_omni


class TestEvalSftOmni(unittest.TestCase):
  """Unit tests for dataset parsing, prompt construction, and output cleanup."""

  def setUp(self):
    self.dummy_image = Image.new("RGB", (100, 100), color="blue")
    self.config = SimpleNamespace(
        image_placeholder="<image>",
        model_name="maxtext-omni-gemma3-qwen3",
        vision_encoder_block="gemma3",
        decoder_block="qwen3",
        use_multimodal=True,
    )

  def test_parse_dataset_example_chartqa(self):
    raw_example = {
        "query": "What is the highest value in the chart?",
        "image": self.dummy_image,
        "label": ["42.5"],
    }
    parsed = eval_sft_omni.parse_dataset_example(raw_example, "HuggingFaceM4/ChartQA")
    self.assertEqual(parsed.question, "What is the highest value in the chart?")
    self.assertEqual(parsed.answer, "42.5")
    self.assertEqual(parsed.image_np.shape, (100, 100, 3))

  def test_parse_dataset_example_resize(self):
    raw_example = {
        "query": "Describe the trend",
        "image": self.dummy_image,
        "label": "Increasing",
    }
    parsed = eval_sft_omni.parse_dataset_example(raw_example, "HuggingFaceM4/ChartQA", image_resize=64)
    self.assertEqual(parsed.image_np.shape, (64, 64, 3))
    self.assertEqual(parsed.answer, "Increasing")

  def test_construct_prompt_sft(self):
    parsed = eval_sft_omni.ParsedDatasetExample(
        question="What is the average sales?",
        image_np=np.zeros((100, 100, 3), dtype=np.uint8),
        answer="150",
    )
    prompt = eval_sft_omni.construct_prompt(parsed, self.config, ckpt_type="sft")
    self.assertIn("<|im_start|>user\n", prompt)
    self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", prompt)
    self.assertIn("What is the average sales?", prompt)
    self.assertIn("<|im_end|>\n<|im_start|>assistant\n", prompt)

  def test_construct_prompt_base(self):
    parsed = eval_sft_omni.ParsedDatasetExample(
        question="What is the peak month?",
        image_np=np.zeros((100, 100, 3), dtype=np.uint8),
        answer="July",
    )
    prompt = eval_sft_omni.construct_prompt(parsed, self.config, ckpt_type="base")
    self.assertIn("<answer></answer>", prompt)
    self.assertIn("What is the peak month?", prompt)
    self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", prompt)

  def test_clean_model_output(self):
    output_with_im_end = "42.5<|im_end|>"
    self.assertEqual(eval_sft_omni.clean_model_output(output_with_im_end), "42.5")

    output_with_answer_tag = "<answer>Paris</answer>"
    self.assertEqual(eval_sft_omni.clean_model_output(output_with_answer_tag), "<answer>Paris")

    output_with_turn = "Increasing trend<end_of_turn>"
    self.assertEqual(eval_sft_omni.clean_model_output(output_with_turn), "Increasing trend")

  def test_resolve_checkpoint_path_explicit(self):
    explicit = "gs://my-bucket/checkpoint/items"
    resolved = eval_sft_omni.resolve_checkpoint_path(self.config, explicit_path=explicit)
    self.assertEqual(resolved, explicit)


if __name__ == "__main__":
  unittest.main()
