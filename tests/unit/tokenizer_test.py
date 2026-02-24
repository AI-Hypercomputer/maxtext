# Copyright 2023–2025 Google LLC
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

"""Tests for tokenizer"""

import numpy as np
from MaxText.globals import MAXTEXT_ASSETS_ROOT
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.trainers.tokenizer import train_tokenizer

import unittest
import pytest
import tensorflow_datasets as tfds
import subprocess
import os


class TokenizerTest(unittest.TestCase):
  """Tests for train_tokenizer.py"""

  @classmethod
  def setUpClass(cls):
    dataset_name = "c4/en:3.0.1"
    dataset_path = "gs://maxtext-dataset"
    cls.vocab_size = 32_768
    cls.max_corpus_chars = 10_000_000
    assets_path = "tests"
    vocab_model_name = "test_tokenizer"
    cls.tokenizer_path = os.path.join(assets_path, vocab_model_name)
    cls.source_tokenizer = input_pipeline_utils.get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.default"),
        "sentencepiece",
        add_bos=False,
        add_eos=False,
    )
    os.environ["TFDS_DATA_DIR"] = dataset_path
    read_config = tfds.ReadConfig(
        shuffle_seed=0,
    )
    train_ds_builder = tfds.builder(dataset_name)
    cls.dataset = train_ds_builder.as_dataset(split="train", read_config=read_config, shuffle_files=True)
    train_tokenizer.train_tokenizer(
        cls.dataset,
        vocab_path=cls.tokenizer_path,
        vocab_size=cls.vocab_size,
        max_corpus_chars=cls.max_corpus_chars,
    )
    cls.test_tokenizer = input_pipeline_utils.get_tokenizer(
        cls.tokenizer_path, "sentencepiece", add_bos=False, add_eos=False
    )

  @classmethod
  def tearDownClass(cls):
    os.remove(cls.tokenizer_path)

  @pytest.mark.tpu_only
  def test_tokenize(self):
    text = "This is a test"
    self.assertTrue(np.array_equal(self.source_tokenizer.encode(text), self.test_tokenizer.encode(text)))

  @pytest.mark.tpu_only
  def test_detokenize(self):
    tokens = [66, 12, 10, 702]
    self.assertEqual(np.asarray(self.source_tokenizer.decode(tokens)), np.asarray(self.test_tokenizer.decode(tokens)))


class TikTokenTest(unittest.TestCase):
  """Tests for train_tokenizer.py"""

  @classmethod
  def setUpClass(cls):
    dataset_name = "c4/en:3.0.1"
    dataset_path = "gs://maxtext-dataset"
    cls.source_tokenizer = input_pipeline_utils.get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer_llama3.tiktoken"),
        "tiktoken",
        add_bos=False,
        add_eos=False,
    )
    os.environ["TFDS_DATA_DIR"] = dataset_path
    read_config = tfds.ReadConfig(
        shuffle_seed=0,
    )
    train_ds_builder = tfds.builder(dataset_name)
    cls.dataset = train_ds_builder.as_dataset(split="train", read_config=read_config, shuffle_files=True)

  @pytest.mark.tpu_only
  def test_tokenize(self):
    text = "This is a test"
    tokens = [2028, 374, 264, 1296]
    self.assertTrue(np.array_equal(self.source_tokenizer.encode(text), tokens))

  @pytest.mark.tpu_only
  def test_detokenize(self):
    tokens = [2028, 374, 264, 1296]
    text = "This is a test"
    self.assertEqual(np.asarray(self.source_tokenizer.decode(tokens)), np.asarray(text))


class HFTokenizerTest(unittest.TestCase):
  """Tests for HFTokenizer"""

  @classmethod
  def setUpClass(cls):
    source = "gs://maxtext-gemma/huggingface/gemma2-2b"
    destination = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers")
    subprocess.run(
        ["gcloud", "storage", "cp", "-R", source, destination],
        check=True,
    )
    cls.hf_tokenizer = input_pipeline_utils.get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "gemma2-2b"), "huggingface", add_bos=False, add_eos=False
    )
    cls.sp_tokenizer = input_pipeline_utils.get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.gemma"), "sentencepiece", add_bos=False, add_eos=False
    )

  @pytest.mark.tpu_only
  def test_tokenize(self):
    text = "This is a test"
    self.assertTrue(np.array_equal(self.hf_tokenizer.encode(text), self.sp_tokenizer.encode(text)))


if __name__ == "__main__":
  unittest.main()
