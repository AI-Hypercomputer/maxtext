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

"""Unit tests for the Kimi-K3 tiktoken tokenizer.

The hermetic tests build a tiny byte-level BPE vocabulary in a temp dir. The parity test compares against the HF
remote-code tokenizer and is skipped unless the Kimi-K3 reference checkout and transformers are available.
"""

import base64
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import regex

from maxtext.input_pipeline import tokenizer as tokenizer_lib
from maxtext.input_pipeline.tokenizer import (
    KIMI_K3_TIKTOKEN_PAT_STR,
    KimiTikTokenTokenizer,
    TikTokenTokenizer,
    build_tokenizer,
)
from tests.utils.kimi_k3_reference import KIMI_K3_REFERENCE_DIR, requires_kimi_k3_reference

NUM_BASE = 256  # tiny vocab: every single byte.

# Deliberately different from the built-in fallback table so the test can tell which one was used.
TINY_ADDED_TOKENS = {
    NUM_BASE + 0: "[BOS]",
    NUM_BASE + 1: "[EOS]",
    NUM_BASE + 2: "<|end_of_msg|>",
    NUM_BASE + 3: "[EOT]",
    NUM_BASE + 4: "<|im_user|>",
    NUM_BASE + 10: "[UNK]",
    NUM_BASE + 11: "[PAD]",
}


def _write_tiny_bpe(path):
  with open(path, "w", encoding="utf-8") as f:
    for i in range(NUM_BASE):
      f.write(f"{base64.b64encode(bytes([i])).decode()} {i}\n")


class KimiK3TokenizerHermeticTest(unittest.TestCase):
  """Hermetic tests on a tiny byte-level vocabulary."""

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, self.tmp)
    self.model_path = os.path.join(self.tmp, "tiktoken.model")
    _write_tiny_bpe(self.model_path)

  def _write_config(self):
    cfg = {"added_tokens_decoder": {str(i): {"content": t, "special": True} for i, t in TINY_ADDED_TOKENS.items()}}
    with open(os.path.join(self.tmp, "tokenizer_config.json"), "w", encoding="utf-8") as f:
      json.dump(cfg, f)

  def test_regex_splits_han_from_latin(self):
    # The pattern uses character-class intersection (`&&`), supported by tiktoken's fancy-regex and by `regex` V1.
    pat = regex.compile(KIMI_K3_TIKTOKEN_PAT_STR, flags=regex.V1)
    self.assertEqual(pat.findall("测试Kimi"), ["测试", "Kimi"])
    self.assertEqual(pat.findall("Hello世界 123"), ["Hello", "世界", " ", "123"])
    # The Llama-3 regex keeps Han and Latin letters in one pre-token.
    self.assertEqual(regex.findall(TikTokenTokenizer.pat_str, "测试Kimi"), ["测试Kimi"])

  def test_special_token_ids_from_config(self):
    self._write_config()
    tok = KimiTikTokenTokenizer(self.model_path, add_bos=False, add_eos=False)
    self.assertEqual(tok.bos_id, NUM_BASE + 0)
    self.assertEqual(tok.eos_id, NUM_BASE + 1)
    self.assertEqual(tok.unk_id, NUM_BASE + 10)
    self.assertEqual(tok.pad_id, NUM_BASE + 11)
    self.assertEqual(tok.stop_tokens, {NUM_BASE + 1, NUM_BASE + 2, NUM_BASE + 3})
    self.assertEqual(tok.special_tokens[f"<|reserved_token_{NUM_BASE + 5}|>"], NUM_BASE + 5)
    self.assertEqual(len(tok.special_tokens), tok.num_reserved_special_tokens)
    self.assertEqual(tok.n_words, NUM_BASE + tok.num_reserved_special_tokens)
    # Plain text round-trips; chat special tokens encode to their ids when allowed. (encode() wraps
    # allowed_special in set(), so pass an explicit set rather than "all".)
    text = "你好 Kimi-K3!"
    self.assertEqual(tok.decode(tok.encode(text)), text)
    ids = tok.encode("<|im_user|>hi[EOS]", allowed_special=set(tok.special_tokens))
    self.assertEqual(ids, [NUM_BASE + 4, ord("h"), ord("i"), NUM_BASE + 1])

  def test_missing_config_warns_and_uses_fallback(self):
    with mock.patch.object(tokenizer_lib.max_logging, "warning") as warn:
      tok = KimiTikTokenTokenizer(self.model_path, add_bos=False, add_eos=False)
    warn.assert_called_once()
    self.assertIn("tokenizer_config.json", warn.call_args[0][0])
    self.assertEqual(tok.bos_id, NUM_BASE + 0)
    self.assertEqual(tok.eos_id, NUM_BASE + 1)
    self.assertEqual(tok.unk_id, NUM_BASE + 254)
    self.assertEqual(tok.pad_id, NUM_BASE + 255)
    self.assertEqual(tok.stop_tokens, {NUM_BASE + 1, NUM_BASE + 2, NUM_BASE + 9})

  def test_malformed_config_warns(self):
    with open(os.path.join(self.tmp, "tokenizer_config.json"), "w", encoding="utf-8") as f:
      f.write("{not json")
    with mock.patch.object(tokenizer_lib.max_logging, "warning") as warn:
      tok = KimiTikTokenTokenizer(self.model_path, add_bos=False, add_eos=False)
    warn.assert_called_once()
    self.assertEqual(tok.pad_id, NUM_BASE + 255)

  def test_llama_tiktoken_unchanged(self):
    path = os.path.join(self.tmp, "tokenizer_llama3.tiktoken")
    _write_tiny_bpe(path)
    tok = build_tokenizer(path, "tiktoken", False, False, None)
    self.assertIs(type(tok), TikTokenTokenizer)
    self.assertEqual(tok.pat_str, TikTokenTokenizer.pat_str)
    self.assertEqual(tok.bos_id, tok.special_tokens["<|begin_of_text|>"])
    self.assertEqual(tok.bos_id, NUM_BASE)
    self.assertEqual(tok.eos_id, NUM_BASE + 1)
    self.assertEqual(tok.pad_id, -1)
    self.assertEqual(tok.stop_tokens, {NUM_BASE + 1, NUM_BASE + 9})

  def test_build_tokenizer_kimi_path_heuristic(self):
    kimi_dir = os.path.join(self.tmp, "Kimi-K3")
    os.makedirs(kimi_dir)
    path = os.path.join(kimi_dir, "tiktoken.model")
    _write_tiny_bpe(path)
    with mock.patch.object(tokenizer_lib.max_logging, "warning"):
      tok = build_tokenizer(path, "tiktoken", False, False, None)
    self.assertIsInstance(tok, KimiTikTokenTokenizer)
    # "kimi" only in a distant ancestor directory does not trigger the Kimi tokenizer.
    other = os.path.join(kimi_dir, "llama", "tokenizer.tiktoken")
    os.makedirs(os.path.dirname(other))
    _write_tiny_bpe(other)
    self.assertIs(type(build_tokenizer(other, "tiktoken", False, False, None)), TikTokenTokenizer)


@requires_kimi_k3_reference
class KimiK3TokenizerParityTest(unittest.TestCase):
  """Encode/decode parity against the HF remote-code tokenizer (tokenization_kimi.py)."""

  def test_parity_with_reference(self):
    try:
      import transformers  # pylint: disable=import-outside-toplevel
    except ImportError:
      self.skipTest("transformers not installed")
    hf_tok = transformers.AutoTokenizer.from_pretrained(KIMI_K3_REFERENCE_DIR, trust_remote_code=True)
    vocab_file = os.path.join(KIMI_K3_REFERENCE_DIR, "tiktoken.model")

    # Copy to a directory without "kimi" in its name: the explicit class must not depend on the path.
    tmp = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, tmp)
    for name in ("tiktoken.model", "tokenizer_config.json"):
      shutil.copy(os.path.join(KIMI_K3_REFERENCE_DIR, name), tmp)
    maxtext_tok = KimiTikTokenTokenizer(os.path.join(tmp, "tiktoken.model"), add_bos=False, add_eos=False)
    maxtext_tok_via_build = build_tokenizer(vocab_file, "tiktoken", False, False, None)
    self.assertIsInstance(maxtext_tok_via_build, KimiTikTokenTokenizer)

    self.assertEqual(maxtext_tok.special_tokens, hf_tok.special_tokens)
    self.assertEqual(maxtext_tok.bos_id, hf_tok.bos_token_id)
    self.assertEqual(maxtext_tok.eos_id, hf_tok.eos_token_id)
    self.assertEqual(maxtext_tok.pad_id, hf_tok.pad_token_id)
    self.assertEqual(maxtext_tok.unk_id, hf_tok.unk_token_id)

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
      self.assertEqual(mt_encoded, hf_encoded, f"Token mismatch on text: {text}")
      self.assertEqual(maxtext_tok_via_build.encode(text), hf_encoded, f"Build tokenizer mismatch on text: {text}")
      self.assertEqual(maxtext_tok.decode(mt_encoded), text)

    chat = "<|end_of_msg|>[BOS]hello 世界[EOS]"
    self.assertEqual(maxtext_tok.encode(chat, allowed_special=set(maxtext_tok.special_tokens)), hf_tok.encode(chat))


if __name__ == "__main__":
  unittest.main()
