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

import os
import shutil
import tempfile
import unittest
from unittest import mock

from maxtext.common.gcloud_stub import is_decoupled
from maxtext.configs import pyconfig
from maxtext.configs.types import resolve_tokenizer_path
from maxtext.input_pipeline import input_pipeline_utils
from maxtext.trainers.tokenizer import train_tokenizer
from maxtext.utils.globals import (
    DEFAULT_TOKENIZER_PATH,
    MAXTEXT_ASSETS_ROOT,
    MAXTEXT_PKG_DIR,
)
from tests.utils.test_helpers import ensure_tokenizer_downloaded, get_test_config_path
import numpy as np


class ResolveTokenizerPathTest(unittest.TestCase):
  """End-to-end unit test for controller-to-sidecar and local tokenizer path resolution."""

  def test_tokenizer_path_resolution_end_to_end(self):
    argv = [os.path.join(MAXTEXT_PKG_DIR, "train.py"), get_test_config_path()]
    rel_from_app = "maxtext/src/maxtext/assets/tokenizers/tokenizer.llama2"
    local_llama2 = os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.llama2")

    # 1. Non-colocated: HF_IDS["default"] ("default"), KeyError fallback ("llama2-7b"),
    # and explicit relative paths resolve eagerly on the controller.
    for kwargs in (
        {"model_name": "default"},
        {"model_name": "llama2-7b"},
        {"tokenizer_path": rel_from_app},
    ):
      with self.subTest(colocated=False, **kwargs):
        cfg = pyconfig.initialize(argv, skip_jax_distributed_system=True, colocated_python_data_input=False, **kwargs)
        self.assertEqual(cfg.tokenizer_path, local_llama2)

    # 2. Colocated: controller defers resolution, and the sidecar resolves + loads against its own layout.
    deferred_cfgs = []
    for kwargs, expected_deferred in (
        ({"model_name": "default"}, DEFAULT_TOKENIZER_PATH),
        ({"model_name": "llama2-7b"}, DEFAULT_TOKENIZER_PATH),
        ({"tokenizer_path": rel_from_app}, rel_from_app),
    ):
      with self.subTest(colocated=True, stage="controller", **kwargs):
        cfg = pyconfig.initialize(
            argv,
            skip_jax_distributed_system=True,
            enable_single_controller=True,
            colocated_python_data_input=True,
            **kwargs,
        )
        self.assertEqual(cfg.tokenizer_path, expected_deferred)
        deferred_cfgs.append(cfg)

    orig_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
      sidecar_workdir = os.path.join(tmpdir, "app")
      sidecar_repo = os.path.join(sidecar_workdir, "maxtext")
      sidecar_assets = os.path.join(sidecar_repo, "src", "maxtext", "assets")
      sidecar_tokenizers = os.path.join(sidecar_assets, "tokenizers")
      os.makedirs(sidecar_tokenizers, exist_ok=True)

      sidecar_llama2 = os.path.join(sidecar_tokenizers, "tokenizer.llama2")
      shutil.copyfile(local_llama2, sidecar_llama2)
      sidecar_qwen_dir = os.path.join(sidecar_tokenizers, "qwen3-tokenizer")
      os.makedirs(sidecar_qwen_dir, exist_ok=True)
      custom_dir = os.path.join(sidecar_workdir, "custom_toks")
      os.makedirs(custom_dir, exist_ok=True)
      custom_llama2 = os.path.join(custom_dir, "tokenizer.llama2")
      shutil.copyfile(local_llama2, custom_llama2)

      try:
        os.chdir(sidecar_workdir)
        with mock.patch.dict(os.environ, {"MAXTEXT_ASSETS_ROOT": sidecar_assets}):
          for cfg in deferred_cfgs:
            loaded = input_pipeline_utils.get_tokenizer(
                cfg.tokenizer_path, cfg.tokenizer_type, add_bos=False, add_eos=False
            )
            self.assertGreater(len(loaded.encode("hello world")), 0)

          for raw_path, expected in (
              ("tokenizer.llama2", sidecar_llama2),
              ("assets/tokenizer.llama2", sidecar_llama2),
              ("src/maxtext/assets/tokenizers/tokenizer.llama2", sidecar_llama2),
              ("src/maxtext/assets/tokenizers/qwen3-tokenizer/", sidecar_qwen_dir),
              # Existing custom paths (absolute or relative) resolve to the user's custom file, not assets_root
              (custom_llama2, custom_llama2),
              ("custom_toks/tokenizer.llama2", "custom_toks/tokenizer.llama2"),
              # Missing custom paths stay untouched so loading fails loudly instead of silently using stock assets
              ("/missing/custom/tokenizer.llama2", "/missing/custom/tokenizer.llama2"),
              ("missing_custom/tokenizer.llama2", "missing_custom/tokenizer.llama2"),
              ("gs://bucket/tokenizer.llama2", "gs://bucket/tokenizer.llama2"),
              ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B"),
          ):
            with self.subTest(raw_path=raw_path):
              self.assertEqual(resolve_tokenizer_path(raw_path), expected)

        # When sidecar WORKDIR is /app/maxtext instead of /app, maxtext/src/... still resolves via assets root
        os.chdir(sidecar_repo)
        with mock.patch.dict(os.environ, {"MAXTEXT_ASSETS_ROOT": sidecar_assets}):
          self.assertEqual(resolve_tokenizer_path(rel_from_app), sidecar_llama2)
      finally:
        os.chdir(orig_cwd)


@unittest.skipIf(is_decoupled(), "Bypassed in offline decoupled runs (no GCS/internet)")
class TrainTokenizerTest(unittest.TestCase):
  """Tests for train_tokenizer.py using data from Parquet files"""

  @classmethod
  def setUpClass(cls):
    # the test only use ~10Mb of data, one file is enough, more files cause slow down
    grain_train_files = "gs://maxtext-dataset/hf/c4/c4-train-00000-of-01637.parquet"
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
    dataset_iter = train_tokenizer.build_grain_iterator(grain_train_files, "parquet")
    train_tokenizer.train_tokenizer(
        dataset_iter,
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

  def test_tokenize(self):
    text = "This is a test"
    self.assertTrue(np.array_equal(self.source_tokenizer.encode(text), self.test_tokenizer.encode(text)))

  def test_detokenize(self):
    tokens = [66, 12, 10, 702]
    self.assertEqual(np.asarray(self.source_tokenizer.decode(tokens)), np.asarray(self.test_tokenizer.decode(tokens)))


@unittest.skipIf(is_decoupled(), "Bypassed in offline decoupled runs (no GCS/internet)")
class TikTokenTest(unittest.TestCase):
  """Tests for TikToken"""

  @classmethod
  def setUpClass(cls):
    grain_train_files = "gs://maxtext-dataset/hf/c4/c4-train-00000-of-01637.parquet"
    cls.source_tokenizer = input_pipeline_utils.get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer_llama3.tiktoken"),
        "tiktoken",
        add_bos=False,
        add_eos=False,
    )
    cls.dataset = train_tokenizer.build_grain_iterator(grain_train_files, "parquet")

  def test_tokenize(self):
    text = "This is a test"
    tokens = [2028, 374, 264, 1296]
    self.assertTrue(np.array_equal(self.source_tokenizer.encode(text), tokens))

  def test_detokenize(self):
    tokens = [2028, 374, 264, 1296]
    text = "This is a test"
    self.assertEqual(np.asarray(self.source_tokenizer.decode(tokens)), np.asarray(text))


@unittest.skipIf(is_decoupled(), "Bypassed in offline decoupled runs (no GCS/internet)")
class HFTokenizerTest(unittest.TestCase):
  """Tests for HFTokenizer"""

  @classmethod
  def setUpClass(cls):
    gemma2_path = ensure_tokenizer_downloaded("gemma2-2b", skip_test_on_failure=False)
    cls.hf_tokenizer = input_pipeline_utils.get_tokenizer(gemma2_path, "huggingface", add_bos=False, add_eos=False)
    cls.sp_tokenizer = input_pipeline_utils.get_tokenizer(
        os.path.join(MAXTEXT_ASSETS_ROOT, "tokenizers", "tokenizer.gemma"), "sentencepiece", add_bos=False, add_eos=False
    )

  def test_tokenize(self):
    text = "This is a test"
    self.assertTrue(np.array_equal(self.hf_tokenizer.encode(text), self.sp_tokenizer.encode(text)))


if __name__ == "__main__":
  unittest.main()
