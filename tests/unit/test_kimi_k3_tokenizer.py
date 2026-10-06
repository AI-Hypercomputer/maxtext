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

"""Unit tests for Kimi-K3 Tokenizer."""

import os
import unittest
import transformers

from maxtext.input_pipeline.tokenizer import (
    KimiTikTokenTokenizer,
    KIMI_K3_TIKTOKEN_PAT_STR,
    build_tokenizer,
)
from tests.utils.kimi_k3_reference import requires_kimi_k3_reference


class KimiK3TokenizerTest(unittest.TestCase):
  """Tests validating Kimi-K3 tokenizer pattern and parity against reference."""

  def setUp(self):
    super().setUp()
    self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    self.ref_dir = os.path.join(self.base_dir, "kimi-k3-hf-reference")
    self.vocab_file = os.path.join(self.ref_dir, "tiktoken.model")

  def test_kimi_k3_tokenizer_regex_pattern(self):
    """Verifies that KIMI_K3_TIKTOKEN_PAT_STR contains Han character splitting."""
    self.assertIn(r"[\p{Han}]+", KIMI_K3_TIKTOKEN_PAT_STR)
    self.assertIn(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*", KIMI_K3_TIKTOKEN_PAT_STR)

  @requires_kimi_k3_reference
  def test_kimi_k3_tokenizer_parity_with_reference(self):
    """Verifies complete encode/decode parity between MaxText KimiTikTokenTokenizer and HF AutoTokenizer."""
    if not os.path.exists(self.vocab_file):
      self.skipTest(f"Vocab file not found: {self.vocab_file}")

    # Load HF AutoTokenizer reference
    hf_tok = transformers.AutoTokenizer.from_pretrained(self.ref_dir, trust_remote_code=True)

    # Instantiate MaxText KimiTikTokenTokenizer
    maxtext_tok = KimiTikTokenTokenizer(self.vocab_file)
    maxtext_tok_via_build = build_tokenizer(self.vocab_file, "tiktoken", False, False, None)

    test_sentences = [
        "Hello, world! This is a test for Kimi-K3 tokenizer.",
        "你好世界！这是一段用于测试Kimi-K3分词器汉字切分特性的中文文本。",
        "混合测试 Mixed English 1234567890 and 特殊符号 !@#$%^&*()_+{}[]:;'<>,.?/",
        "def forward(self, x: torch.Tensor) -> torch.Tensor:\n    return self.linear_attn(x) * 0.5\n",
        "     多个空格   \t\t\n\n\r换行符测试   ",
        "大语言模型与混合专家架构（MoE）的前沿技术发展分析报告。",
    ]

    for text in test_sentences:
      hf_encoded = hf_tok.encode(text)
      mt_encoded = maxtext_tok.encode(text)
      mt_build_encoded = maxtext_tok_via_build.encode(text)

      # Verify token sequence parity
      self.assertEqual(mt_encoded, hf_encoded, f"Token mismatch on text: {text}")
      self.assertEqual(mt_build_encoded, hf_encoded, f"Build tokenizer mismatch on text: {text}")

      # Verify round-trip decode
      mt_decoded = maxtext_tok.decode(mt_encoded)
      self.assertEqual(mt_decoded, text, f"Decoded text mismatch: {mt_decoded} != {text}")


if __name__ == "__main__":
  unittest.main()
