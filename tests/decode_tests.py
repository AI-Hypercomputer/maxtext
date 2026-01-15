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

"""Tests for decode with various configs."""

import io
import os
import unittest

import pytest

from absl.testing import absltest
from contextlib import redirect_stdout

from MaxText.decode import main as decode_main
from MaxText.globals import MAXTEXT_PKG_DIR, MAXTEXT_ASSETS_ROOT


class DecodeTests(unittest.TestCase):
  """Tests decode with various configs."""

  GEMMA_2B_CKPT_PATH = "gs://maxtext-gemma/2b/2025-11-04-04-33//0/items"
  CONFIGS = {
      "base": [  # tests decode
          None,
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "ici_tensor_parallelism=4",
          "max_target_length=128",
          "per_device_batch_size=1",
          rf"tokenizer_path={os.path.join('src', MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
      ],
      "int8": [  # tests decode with int8 quantization
          None,
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "ici_tensor_parallelism=4",
          "max_target_length=128",
          "per_device_batch_size=1",
          "quantization=int8",
          "quantize_kvcache=True",
          rf"tokenizer_path={os.path.join('src', MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
      ],
      "pdb_lt_1": [  # tests decode with per_device_batch_size < 1
          None,
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          "dataset_path=gs://maxtext-dataset",
          "steps=2",
          "enable_checkpointing=False",
          "ici_tensor_parallelism=4",
          "max_target_length=128",
          "per_device_batch_size=.25",
          rf"tokenizer_path={os.path.join('src', MAXTEXT_ASSETS_ROOT, 'tokenizer.llama2')}",
      ],
      "decode_sampling": [
          None,
          os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml"),
          "base_output_directory=gs://runner-maxtext-logs",
          "run_name=runner_test",
          f"load_parameters_path={GEMMA_2B_CKPT_PATH}",
          "per_device_batch_size=1",
          "max_prefill_predict_length=8",
          "max_target_length=16",
          "dataset_type=synthetic",
          "steps=10",
          "async_checkpointing=False",
          "model_name=gemma-2b",
          rf"tokenizer_path={os.path.join('src', MAXTEXT_ASSETS_ROOT, 'tokenizer.gemma')}",
          "attention=dot_product",
          "prompt=I love to",
          "skip_jax_distributed_system=True",
      ],
  }
  SAMPLING_STRATEGY_CONFIG = {
      "greedy": [
          "decode_sampling_strategy=greedy",
      ],
      "weighted": [
          "decode_sampling_strategy=weighted",
          "decode_sampling_temperature=.00001",
      ],
      "nucleus": [
          "decode_sampling_strategy=nucleus",
          "decode_sampling_nucleus_p=0",
      ],
      "topk": [
          "decode_sampling_strategy=topk",
          "decode_sampling_top_k=1",
      ],
  }

  @pytest.mark.tpu_only
  def test_tpu_base(self):
    decode_main(DecodeTests.CONFIGS["base"])

  @pytest.mark.gpu_only
  def test_gpu_base(self):
    decode_main(DecodeTests.CONFIGS["base"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_int8(self):
    decode_main(DecodeTests.CONFIGS["int8"] + ["attention=dot_product"])

  @pytest.mark.gpu_only
  def test_gpu_int8(self):
    decode_main(DecodeTests.CONFIGS["int8"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  def test_tpu_pdb_lt_1(self):
    decode_main(DecodeTests.CONFIGS["pdb_lt_1"])

  @pytest.mark.gpu_only
  def test_gpu_pdb_lt_1(self):
    decode_main(DecodeTests.CONFIGS["pdb_lt_1"] + ["attention=dot_product"])

  @pytest.mark.tpu_only
  @pytest.mark.scheduled_only
  @pytest.mark.skip(reason="Flaky test. Disabled in b/470982884")
  def test_decode_greedy_sampling(self):
    config = DecodeTests.CONFIGS["decode_sampling"] + DecodeTests.SAMPLING_STRATEGY_CONFIG["greedy"]
    captured_out = run_decoding(config)
    expected_output = "Input `I love to` -> ` travel and I love to write"
    assert expected_output in captured_out

  @pytest.mark.tpu_only
  @pytest.mark.scheduled_only
  def test_decode_weighted_sampling(self):
    config = DecodeTests.CONFIGS["decode_sampling"] + DecodeTests.SAMPLING_STRATEGY_CONFIG["weighted"]
    captured_out = run_decoding(config)
    expected_output = "Input `I love to` -> ` travel and I love to write"
    assert expected_output in captured_out

  @pytest.mark.tpu_only
  @pytest.mark.scheduled_only
  def test_decode_nucleus_sampling(self):
    config = DecodeTests.CONFIGS["decode_sampling"] + DecodeTests.SAMPLING_STRATEGY_CONFIG["nucleus"]
    captured_out = run_decoding(config)
    expected_output = "Input `I love to` -> ` travel and I love to write"
    assert expected_output in captured_out

  @pytest.mark.tpu_only
  @pytest.mark.scheduled_only
  @pytest.mark.skip(reason="Flaky test. Disabled in b/470982884")
  def test_decode_topk_sampling(self):
    config = DecodeTests.CONFIGS["decode_sampling"] + DecodeTests.SAMPLING_STRATEGY_CONFIG["topk"]
    captured_out = run_decoding(config)
    expected_output = "Input `I love to` -> ` travel and I love to write"
    assert expected_output in captured_out


def run_decoding(config):
  f = io.StringIO()
  with redirect_stdout(f):
    decode_main(config)
  captured_out = f.getvalue()
  return captured_out


if __name__ == "__main__":
  absltest.main()
