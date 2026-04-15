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

import gc
import os
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.utils.globals import MAXTEXT_PKG_DIR
from maxtext.common.common_types import MODEL_MODE_TRAIN
from maxtext.layers.decoders import Decoder
from maxtext.utils import maxtext_utils
import pytest


class DummyEmbedding:
  """Dummy embedding layer for testing."""

  def __init__(self, emb_dim: int):
    self.emb_dim = emb_dim

  def __call__(self, x, model_mode):
    return jnp.ones((x.shape[0], x.shape[1], self.emb_dim))


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
      "base_emb_dim=64",
      "base_mlp_dim=64",
      "base_moe_mlp_dim=64",
      "base_num_query_heads=2",
      "base_num_kv_heads=2",
      "head_dim=32",
      "indexer_head_dim=32",
      "qk_nope_head_dim=32",
      "qk_rope_head_dim=16",
      "v_head_dim=32",
      "vocab_size=128",
      "mhc_expansion_rate=4",
      "attention=dot_product",
      "per_device_batch_size=1",
      "max_target_length=8",
      "max_prefill_predict_length=8",
      "enable_checkpointing=False",
      "engram_num_heads=1",
      "engram_head_dim=8",
      "engram_vocab_bases=[128,128]",
      "engram_max_ngram_size=3",
      "engram_kernel_size=4",
      "hf_access_token=dummy",
      "tokenizer_path=dummy",
  ]

  def _test_engram_pattern(self, mock_from_pretrained, engram_layers_str, expected_keys):
    """Helper method to test different engram layer patterns."""

    # Setup mock tokenizer
    class MockTokenizer:
      """Mock tokenizer for testing."""

      pad_token_id = 0

      def __len__(self):
        return 128

      def __call__(self, x):
        return jnp.ones_like(x)

      def convert_ids_to_tokens(self, *args, **kwargs):
        return "a"

      def decode(self, *args, **kwargs):
        return "a"

      def batch_decode(self, token_ids, *args, **kwargs):
        return ["a" for _ in token_ids]

    mock_from_pretrained.return_value = MockTokenizer()

    config_path = os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")
    config = pyconfig.initialize([None, config_path] + self._COMMON_CONFIG + [f"engram_layers=[{engram_layers_str}]"])

    devices_array = maxtext_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    decoder = Decoder(
        config=config,
        mesh=mesh,
        model_mode=MODEL_MODE_TRAIN,
    )

    batch_size = config.global_batch_size_to_load
    seq_len = config.max_target_length

    decoder_input_tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    decoder_positions = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    decoder_segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    shared_embedding = DummyEmbedding(emb_dim=config.emb_dim)

    with mesh:
      variables = decoder.init(
          {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1), "aqt": jax.random.PRNGKey(2)},
          shared_embedding=shared_embedding,
          decoder_input_tokens=decoder_input_tokens,
          decoder_positions=decoder_positions,
          decoder_segment_ids=decoder_segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )

    self.assertIn("params", variables)
    params = variables["params"]
    for key in expected_keys:
      self.assertIn(key, params)

    del variables
    del params
    del decoder
    jax.clear_caches()
    gc.collect()

  @pytest.mark.tpu_only
  @patch("transformers.AutoTokenizer.from_pretrained")
  def test_decoder_init_engram_2_8(self, mock_from_pretrained):
    """Test engram layers at indices 2 and 8."""
    self._test_engram_pattern(
        mock_from_pretrained,
        "2,8",
        [
            "dense_layers_0_1",
            "dense_layers_engram_2",
            "dense_layers_3_4",
            "moe_layers_5_7",
            "moe_layers_engram_8",
            "moe_layers_9_9",
        ],
    )

  @pytest.mark.tpu_only
  @patch("transformers.AutoTokenizer.from_pretrained")
  def test_decoder_init_engram_0_5(self, mock_from_pretrained):
    """Test engram layers at indices 0 and 5 - first engram layer of block."""
    self._test_engram_pattern(
        mock_from_pretrained,
        "0,5",
        ["dense_layers_engram_0", "dense_layers_1_4", "moe_layers_engram_5", "moe_layers_6_9"],
    )

  @pytest.mark.tpu_only
  @patch("transformers.AutoTokenizer.from_pretrained")
  def test_decoder_init_engram_4_9(self, mock_from_pretrained):
    """Test engram layers at indices 4 and 9 - last engram layer of block."""
    self._test_engram_pattern(
        mock_from_pretrained,
        "4,9",
        ["dense_layers_0_3", "dense_layers_engram_4", "moe_layers_5_8", "moe_layers_engram_9"],
    )
