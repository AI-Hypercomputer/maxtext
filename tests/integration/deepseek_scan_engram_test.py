# Copyright 2023-2026 Google LLC
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

"""Unit tests for DeepSeek Engram across scanned decoder layers."""

import unittest
from unittest.mock import patch

import jax.numpy as jnp

import pytest

# The Linen Decoder this test exercised was removed in PR12 (Delete Linen).
# NNX decoder coverage is in tests/unit/nnx_decoders_test.py.
pytestmark = pytest.mark.skip(
    reason="Linen Decoder removed in PR12 (Delete Linen); NNX decoder coverage is in tests/unit/nnx_decoders_test.py"
)


class DummyEmbedding:
  """Dummy embedding layer for testing."""

  def __init__(self, emb_dim: int):
    self.emb_dim = emb_dim

  def __call__(self, x, model_mode):
    return jnp.ones((x.shape[0], x.shape[1], self.emb_dim))


@pytest.mark.integration_test
class TestDeepSeekScanEngram(unittest.TestCase):
  """Test DeepSeek decoder block with scan_layers=True and engram_layers."""

  _COMMON_CONFIG = [
      "run_name=test_deepseek_scan_engram",
      "model_name=deepseek-custom",
      "override_model_config=True",
      "decoder_block=deepseek",
      "scan_layers=True",
      "first_num_dense_layers=5",
      "base_num_decoder_layers=10",
      "num_decoder_layers=10",
      "base_emb_dim=8",
      "base_mlp_dim=8",
      "base_moe_mlp_dim=8",
      "base_num_query_heads=1",
      "base_num_kv_heads=1",
      "head_dim=4",
      "indexer_head_dim=4",
      "qk_nope_head_dim=4",
      "qk_rope_head_dim=4",
      "v_head_dim=4",
      "vocab_size=128",
      "mhc_expansion_rate=4",
      "attention=dot_product",
      "per_device_batch_size=1",
      "max_target_length=8",
      "max_prefill_predict_length=8",
      "enable_checkpointing=False",
      "engram_num_heads=1",
      "engram_head_dim=4",
      "engram_vocab_bases=[128,128]",
      "engram_max_ngram_size=3",
      "engram_kernel_size=4",
      "num_experts=2",
      "num_experts_per_tok=1",
      "hf_access_token=dummy",
      "tokenizer_path=dummy",
  ]

  def _test_engram_pattern(
      self,
      mock_from_pretrained,
      engram_layers_str,
      expected_keys,
      first_num_dense_layers=5,
      base_num_decoder_layers=10,
  ):
    """Helper method to test different engram layer patterns."""
    # The Linen Decoder this exercised was removed in PR12 (Delete Linen);
    # NNX decoder coverage lives in tests/unit/nnx_decoders_test.py.
    del mock_from_pretrained, engram_layers_str, expected_keys, first_num_dense_layers, base_num_decoder_layers
    raise unittest.SkipTest("Linen Decoder removed in PR12 (Delete Linen)")

  @pytest.mark.tpu_only
  @patch("transformers.AutoTokenizer.from_pretrained")
  def test_decoder_init_engram_2_8(self, mock_from_pretrained):
    """Test engram layers at indices 2 and 8."""
    self._test_engram_pattern(
        mock_from_pretrained,
        "1",
        [
            "dense_layers_0_0",
            "dense_layers_engram_1",
            "dense_layers_2_2",
        ],
        first_num_dense_layers=3,
        base_num_decoder_layers=3,
    )

  @pytest.mark.tpu_only
  @patch("transformers.AutoTokenizer.from_pretrained")
  def test_decoder_init_engram_0_5(self, mock_from_pretrained):
    """Test engram layers at indices 0 and 5 - first engram layer of block."""
    self._test_engram_pattern(
        mock_from_pretrained,
        "0,1",
        ["dense_layers_engram_0", "moe_layers_engram_1"],
        first_num_dense_layers=1,
        base_num_decoder_layers=2,
    )

  @pytest.mark.tpu_only
  @patch("transformers.AutoTokenizer.from_pretrained")
  def test_decoder_init_engram_4_9(self, mock_from_pretrained):
    """Test engram layers at indices 4 and 9 - last engram layer of block."""
    self._test_engram_pattern(
        mock_from_pretrained,
        "1,2",
        ["dense_layers_0_0", "dense_layers_engram_1", "moe_layers_engram_2"],
        first_num_dense_layers=2,
        base_num_decoder_layers=3,
    )
