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

"""Tests for Attentions."""

import itertools
import os.path
import random
import sys
import unittest
from unittest import mock

import pytest

from absl.testing import parameterized

import numpy as np

from jax.sharding import Mesh, NamedSharding, AxisType, PartitionSpec as P
import jax
import jax.numpy as jnp

from flax import nnx
from flax.linen import partitioning as nn_partitioning

from MaxText import maxtext_utils
from MaxText import max_utils
from MaxText import pyconfig
from MaxText.common_types import (
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    EP_AS_CONTEXT,
    AttentionType,
    ShardMode,
)
from MaxText.globals import MAXTEXT_PKG_DIR
from MaxText.sharding import maybe_shard_with_name
from MaxText.layers.attentions import Attention
from MaxText.layers.attention_op import ChunkedCausalMask, _make_bidirectional_block_mask, _generate_chunk_attention_mask
from MaxText.layers.attention_mla import MLA


class BidirectionalBlockMaskTest(unittest.TestCase):
  """Test for make_bidirectional_block_mask."""

  def test_one_block_mask(self):
    bidirectional_mask = np.asarray([[0, 1, 1, 1, 0, 0]])
    # pylint: disable=protected-access
    block_mask = _make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.asarray(
        [
            [
                [False, False, False, False, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
            ]
        ]
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_two_blocks_mask(self):
    bidirectional_mask = np.asarray([[0, 1, 1, 0, 1, 1]])
    # pylint: disable=protected-access
    block_mask = _make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.asarray(
        [
            [
                [False, False, False, False, False, False],
                [False, True, True, False, False, False],
                [False, True, True, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
            ]
        ]
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_batch_block_masks(self):
    bidirectional_mask = np.asarray([[0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1]])
    # pylint: disable=protected-access
    block_mask = _make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.asarray(
        [
            [
                [False, False, False, False, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, True, True, True, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
            ],
            [
                [False, False, False, False, False, False],
                [False, True, True, False, False, False],
                [False, True, True, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, True, True],
                [False, False, False, False, True, True],
            ],
        ]
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_empty_block_mask(self):
    bidirectional_mask = np.asarray([[0, 0, 0, 0, 0, 0]])
    # pylint: disable=protected-access
    block_mask = _make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.zeros(
        (bidirectional_mask.shape[0], bidirectional_mask.shape[1], bidirectional_mask.shape[1]), dtype=bool
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_full_block_mask(self):
    bidirectional_mask = np.asarray([[1, 1, 1, 1, 1, 1]])
    # pylint: disable=protected-access
    block_mask = _make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.ones(
        (bidirectional_mask.shape[0], bidirectional_mask.shape[1], bidirectional_mask.shape[1]), dtype=bool
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_combine_with_causal_mask(self):
    seq_len = 6
    row_ids = np.arange(seq_len, dtype=np.int32)[:, None]
    col_ids = np.arange(seq_len, dtype=np.int32)[None, :]
    causal_mask = (col_ids <= row_ids)[None, None, None, :, :]
    bidirectional_mask = np.asarray([[0, 1, 1, 1, 0, 0], [0, 1, 1, 0, 1, 1]])
    # pylint: disable=protected-access
    image_mask = _make_bidirectional_block_mask(bidirectional_mask)
    combined_mask = causal_mask | image_mask[:, None, None, ...]
    expected_mask = np.asarray(
        [
            [
                [
                    [
                        [True, False, False, False, False, False],
                        [True, True, True, True, False, False],
                        [True, True, True, True, False, False],
                        [True, True, True, True, False, False],
                        [True, True, True, True, True, False],
                        [True, True, True, True, True, True],
                    ]
                ]
            ],
            [
                [
                    [
                        [True, False, False, False, False, False],
                        [True, True, True, False, False, False],
                        [True, True, True, False, False, False],
                        [True, True, True, True, False, False],
                        [True, True, True, True, True, True],
                        [True, True, True, True, True, True],
                    ]
                ]
            ],
        ]
    )
    np.testing.assert_array_equal(combined_mask, expected_mask)


class ChunkedCausalMaskTest(unittest.TestCase):
  """Test for the ChunkedCausalMask."""

  def test_basic_chunking(self):
    """Tests the mask with a simple chunk size."""
    seq_len = 8
    chunk_size = 4
    mask = ChunkedCausalMask(shape=(seq_len, seq_len), chunk_size=chunk_size)

    # Manually compute the expected mask
    # Causal within chunks (0-3, 4-7)
    expected_mask = np.zeros((seq_len, seq_len), dtype=np.bool_)
    for r in range(seq_len):
      for c in range(seq_len):
        q_chunk = r // chunk_size
        kv_chunk = c // chunk_size
        if q_chunk == kv_chunk and r >= c:
          expected_mask[r, c] = True

    # Get the actual mask by slicing
    actual_mask = mask[:, :]

    np.testing.assert_array_equal(actual_mask, expected_mask)
    # Make sure _generate_chunk_attention_mask also produces the same mask
    # pylint: disable=protected-access
    actual_mask = _generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_full_length_chunk(self):
    """Tests when chunk size equals sequence length (should be causal)."""
    seq_len = 6
    chunk_size = 6  # Same as seq_len
    mask = ChunkedCausalMask(shape=(seq_len, seq_len), chunk_size=chunk_size)

    # Expected mask is a standard lower triangular causal mask
    expected_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.bool_))

    actual_mask = mask[:, :]
    np.testing.assert_array_equal(actual_mask, expected_mask)
    # Make sure _generate_chunk_attention_mask also produces the same mask
    # pylint: disable=protected-access
    actual_mask = _generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_single_token_chunk(self):
    """Tests when chunk size is 1 (only attend to self)."""
    seq_len = 5
    chunk_size = 1
    mask = ChunkedCausalMask(shape=(seq_len, seq_len), chunk_size=chunk_size)

    # Expected mask is just the identity matrix
    expected_mask = np.eye(seq_len, dtype=np.bool_)

    actual_mask = mask[:, :]
    np.testing.assert_array_equal(actual_mask, expected_mask)
    # Make sure _generate_chunk_attention_mask also produces the same mask
    # pylint: disable=protected-access
    actual_mask = _generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_non_square_shape(self):
    """Tests with different query and key sequence lengths."""
    q_len = 6
    kv_len = 8
    chunk_size = 3
    mask = ChunkedCausalMask(shape=(q_len, kv_len), chunk_size=chunk_size)

    # Manually compute expected mask
    expected_mask = np.zeros((q_len, kv_len), dtype=np.bool_)
    for r in range(q_len):
      for c in range(kv_len):
        q_chunk = r // chunk_size
        kv_chunk = c // chunk_size
        if q_chunk == kv_chunk and r >= c:
          expected_mask[r, c] = True

    actual_mask = mask[:, :]
    np.testing.assert_array_equal(actual_mask, expected_mask)
    # Make sure _generate_chunk_attention_mask also produces the same mask
    # pylint: disable=protected-access
    actual_mask = _generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_value_error_on_zero_chunk_size(self):
    """Tests that a ValueError is raised for chunk_size <= 0."""
    with self.assertRaises(ValueError):
      ChunkedCausalMask(shape=(4, 4), chunk_size=0)
    with self.assertRaises(ValueError):
      ChunkedCausalMask(shape=(4, 4), chunk_size=-2)
    with self.assertRaises(ValueError):
      # pylint: disable=protected-access
      _generate_chunk_attention_mask(mask_shape=(4, 4), chunk_size=0)


class AttentionTest(parameterized.TestCase):
  """Test for the Attention"""

  # Note: if you are changing these configs, please make sure to change the configs in
  # context_parallelism.py as well, since we are using the same configs for both
  # tests to get the same mesh and other config
  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "max_prefill_predict_length": 16,
      "max_target_length": 512,
      "sa_block_q": 128,
      "sa_block_kv": 128,
      "sa_block_kv_compute": 128,
      "sa_block_q_dkv": 128,
      "sa_block_kv_dkv": 128,
      "sa_block_kv_dkv_compute": 128,
      "sa_block_q_dq": 128,
      "sa_block_kv_dq": 128,
  }

  def setUp(self):
    """Initializes the configuration for each test"""
    super().setUp()
    jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **self.config_arguments,
    )
    self.cfg = config

    self.rng = jax.random.PRNGKey(0)
    self.nnx_rng = nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42))

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.global_batch_size = self.cfg.global_batch_size_to_train_on
    self.num_kv_heads = self.cfg.num_kv_heads
    self.num_query_heads = self.cfg.num_query_heads
    self.max_target_length = self.cfg.max_target_length
    self.max_prefill_predict_length = self.cfg.max_prefill_predict_length
    self.head_dim = self.cfg.head_dim
    self.embed_dim = self.cfg.base_emb_dim
    self.dtype = self.cfg.dtype
    self.attention_type = self.cfg.attention_type

    dummy_inputs_q = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    dummy_inputs_kv = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    self._attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        attention_type=self.attention_type,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )

  def get_data(self, dtype):
    """get data"""
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )

    decoder_segment_ids = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        self.rng, (self.global_batch_size, self.max_target_length), 0, self.max_target_length
    )

    return lnx, decoder_segment_ids, decoder_positions

  def get_structured_data(self, dtype):
    """get structured data"""
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )

    decoder_positions = jnp.stack(
        [jnp.arange(self.max_target_length, dtype=jnp.int32) for _ in range(self.global_batch_size)]
    )

    decoder_segment_ids = (
        jax.numpy.zeros((self.global_batch_size, self.max_target_length)) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    )

    return lnx, decoder_segment_ids, decoder_positions

  @pytest.mark.tpu_only
  def test_autoregression(self):
    prefill_length = self.cfg.max_prefill_predict_length
    decode_total_length = self.cfg.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(self.dtype)

    mha_full, _ = self._attention_as_mha_generic(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    mha_prefill, _ = self._attention_as_mha_generic(
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )

    self.assertTrue(
        jax.numpy.allclose(mha_prefill, mha_full[:, :prefill_length, :], rtol=1e-02, atol=1e-02, equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      mha_idx, _ = self._attention_as_mha_generic(
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

      mha_full_this_idx = mha_full[:, idx : idx + 1, :]
      self.assertTrue(mha_full_this_idx.shape == mha_idx.shape)
      self.assertTrue(jax.numpy.allclose(mha_full_this_idx, mha_idx, rtol=1e-02, atol=1e-02, equal_nan=False))

  @pytest.mark.tpu_only
  def test_model_mode_prefill_dtype_float32(self):
    self._test_model_mode_prefill_dtype(jnp.float32)

  @pytest.mark.tpu_only
  def test_model_mode_prefill_dtype_bfloat16(self):
    """test model mode prefill for dtype bfloat16"""
    self._test_model_mode_prefill_dtype(jnp.bfloat16)

  def _test_model_mode_prefill_dtype(self, dtype):
    """test model mode prefill for specified dtype"""
    lnx, decoder_segment_ids, decoder_positions = self.get_data(dtype)
    prefill_length = self.cfg.max_prefill_predict_length
    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    dummy_inputs_q = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    dummy_inputs_kv = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=dtype,
        dropout_rate=self.cfg.dropout_rate,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )

    mha_prefill, _ = attention_as_mha_generic(
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )

    self.assertEqual(dtype, mha_prefill.dtype)

  @pytest.mark.tpu_only
  def test_tpu_kernel_attention_mha(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads)

  @pytest.mark.tpu_only
  def test_tpu_kernel_attention_gqa(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads // 2)

  @pytest.mark.tpu_only
  def test_tpu_kernel_attention_mqa(self):
    self.tpu_kernel_attention_helper(1)

  def tpu_kernel_attention_helper(self, num_kv_heads):
    """Test equivalence between dot_product and TPU accelerated"""

    lnx, decoder_segment_ids, decoder_positions = self.get_data(self.dtype)

    dummy_inputs_q = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    dummy_inputs_kv = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        rngs=self.nnx_rng,
    )

    generic_state = nnx.state(attention_as_mha_generic)

    mha_generic_output, _ = attention_as_mha_generic(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    dummy_inputs_q = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    dummy_inputs_kv = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    attention_as_mha_flash = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        mesh=self.mesh,
        attention_kernel="flash",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        rngs=self.nnx_rng,
    )
    nnx.update(attention_as_mha_flash, generic_state)

    mha_generic_flash_output, _ = attention_as_mha_flash(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_output, rtol=1e-01, atol=1e-01, equal_nan=False)
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "cp_no_load_balance",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 1,
          "expert_shard_attention_option": "fsdp",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_with_load_balance",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 1,
          "expert_shard_attention_option": "fsdp",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_ep_no_load_balance",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 2,
          "expert_shard_attention_option": "context",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_ep_with_load_balance",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 2,
          "expert_shard_attention_option": "context",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "ep_no_load_balance",
          "ici_context_parallelism": 1,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 4,
          "expert_shard_attention_option": "context",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "ep_with_load_balance",
          "ici_context_parallelism": 1,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 4,
          "expert_shard_attention_option": "context",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_no_load_balance_explicit",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 1,
          "expert_shard_attention_option": "fsdp",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_with_load_balance_explicit",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 1,
          "expert_shard_attention_option": "fsdp",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_ep_no_load_balance_explicit",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 2,
          "expert_shard_attention_option": "context",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_ep_with_load_balance_explicit",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 2,
          "expert_shard_attention_option": "context",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "ep_no_load_balance_explicit",
          "ici_context_parallelism": 1,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 4,
          "expert_shard_attention_option": "context",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "ep_with_load_balance_explicit",
          "ici_context_parallelism": 1,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 4,
          "expert_shard_attention_option": "context",
          "shard_mode": "explicit",
      },
  )
  # TODO (b/454764135.) : This tests fails with new tokamax kernel
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_context_parallel(
      self,
      ici_context_parallelism,
      context_parallel_load_balance,
      ici_expert_parallelism,
      expert_shard_attention_option,
      shard_mode,
  ):
    """Test equivalence between dot_product and flash attention + context/expert parallelism"""
    num_kv_heads = self.num_kv_heads
    lnx, decoder_segment_ids, decoder_positions = self.get_data(self.dtype)
    # Dot product
    mha_generic_output, _ = self._attention_as_mha_generic(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    generic_state = nnx.state(self._attention_as_mha_generic)

    # Test with Context Parallelism
    cfg_cp = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **self.config_arguments,
        ici_context_parallelism=ici_context_parallelism,
        context_parallel_load_balance=context_parallel_load_balance,
        ici_expert_parallelism=ici_expert_parallelism,
        expert_shard_attention_option=expert_shard_attention_option,
        shard_mode=shard_mode,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    axis_type = AxisType.Explicit if shard_mode == "explicit" else AxisType.Auto
    axis_names = [axis_type for _ in cfg_cp.mesh_axes]
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes, axis_types=tuple(axis_names))
    attention_as_mha_flash_cp = Attention(
        config=cfg_cp,
        num_query_heads=cfg_cp.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=cfg_cp.head_dim,
        max_target_length=cfg_cp.max_target_length,
        max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        mesh=mesh_cp,
        attention_kernel="flash",
        dtype=self.dtype,
        dropout_rate=cfg_cp.dropout_rate,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )
    nnx.update(attention_as_mha_flash_cp, generic_state)

    mha_generic_flash_cp_output = _forward_with_context_expert_parallelism(
        cfg_cp, mesh_cp, attention_as_mha_flash_cp, lnx, decoder_segment_ids, decoder_positions
    )

    # This removes all sharding information and makes them standard NumPy arrays.
    mha_generic_output = jax.device_get(mha_generic_output)
    mha_generic_flash_cp_output = jax.device_get(mha_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_cp_output, rtol=1e-01, atol=1e-01, equal_nan=False),
        msg="Logits from generic dot product and flash attention + context/expert parallelism are not close.\n"
        f"ici_context_parallelism={ici_context_parallelism}, context_parallel_load_balance={context_parallel_load_balance},"
        f" ici_expert_parallelism={ici_expert_parallelism}, expert_shard_attention_option={expert_shard_attention_option}.",
    )

  @pytest.mark.tpu_only
  def test_dot_product_cache_axis_order(self):
    all_axis_orders = tuple(itertools.permutations(range(4)))
    for axis_order in random.choices(all_axis_orders, k=4):
      self.dot_product_attention_helper(prefill_cache_axis_order=axis_order, ar_cache_axis_order=axis_order)
      print(f"passed test for {axis_order=}")

  def dot_product_attention_helper(self, prefill_cache_axis_order, ar_cache_axis_order):
    for compute_axis_order in [(0, 1, 2, 3), (0, 2, 1, 3)]:
      self._dot_product_attention(
          prefill_cache_axis_order,
          ar_cache_axis_order,
          compute_axis_order=compute_axis_order,
      )
      print(f"passed subtest for {compute_axis_order=}")

  def _dot_product_attention(
      self,
      prefill_cache_axis_order,
      ar_cache_axis_order,
      compute_axis_order,
  ):
    """Test equalvant between different layout control in dot_product"""

    rtol, atol = 1e-02, 1e-02

    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=128,
        max_prefill_predict_length=16,
        attention="dot_product",
    )

    prefill_length = config.max_prefill_predict_length
    decode_total_length = config.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(config.dtype)
    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    dummy_inputs_q = jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim))
    dummy_inputs_kv = jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim))
    attention_w_layout = Attention(
        mesh=self.mesh,
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        dtype=config.dtype,
        prefill_cache_axis_order=prefill_cache_axis_order,
        ar_cache_axis_order=ar_cache_axis_order,
        compute_axis_order=compute_axis_order,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )
    attention_w_layout_full, _ = attention_w_layout(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    attention_w_layout_prefill, _ = attention_w_layout(
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )
    self.assertTrue(
        jax.numpy.allclose(attention_w_layout_full[:, :prefill_length, :], attention_w_layout_prefill, equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]

      attention_w_layout_idx, _ = attention_w_layout(
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

      attention_w_layout_full_this_idx = attention_w_layout_full[:, idx : idx + 1, :]
      self.assertTrue(attention_w_layout_full_this_idx.shape == attention_w_layout_idx.shape)
      self.assertTrue(
          jax.numpy.allclose(
              attention_w_layout_full_this_idx, attention_w_layout_idx, rtol=rtol, atol=atol, equal_nan=False
          )
      )

  @pytest.mark.tpu_only
  def test_dot_product_reshape_q(self):
    for compute_axis_order in [(0, 1, 2, 3), (0, 2, 1, 3)]:
      self._dot_product_attention_reshape_q(
          compute_axis_order=compute_axis_order,
      )
      print(f"test passed for compute_axis_order: {compute_axis_order}")

  def _dot_product_attention_reshape_q(self, compute_axis_order):
    """Test equalvant between q and reshape q in dot_product"""

    rtol, atol = 1e-02, 1e-02

    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=128,
        max_prefill_predict_length=16,
        attention="dot_product",
    )

    prefill_length = config.max_prefill_predict_length
    decode_total_length = config.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(config.dtype)

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    dummy_inputs_q = jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim))
    dummy_inputs_kv = jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim))

    attention_wo_reshape_q = Attention(
        mesh=self.mesh,
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        attention_kernel=config.attention,
        dtype=config.dtype,
        compute_axis_order=compute_axis_order,
        reshape_q=False,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )

    attention_w_reshape_q = Attention(
        mesh=self.mesh,
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        attention_kernel=config.attention,
        dtype=config.dtype,
        compute_axis_order=compute_axis_order,
        reshape_q=True,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )

    attention_wo_reshape_q_state = nnx.state(attention_wo_reshape_q)
    nnx.update(attention_w_reshape_q, attention_wo_reshape_q_state)

    attention_wo_reshape_q_full, _ = attention_wo_reshape_q(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    attention_w_reshape_q_full, _ = attention_w_reshape_q(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    attention_wo_reshape_q_prefill, _ = attention_wo_reshape_q(
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )
    self.assertTrue(
        jax.numpy.allclose(
            attention_wo_reshape_q_full[:, :prefill_length, :], attention_wo_reshape_q_prefill, equal_nan=False
        )
    )

    attention_w_reshape_q_prefill, _ = attention_w_reshape_q(
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )
    self.assertTrue(
        jax.numpy.allclose(
            attention_w_reshape_q_full[:, :prefill_length, :], attention_w_reshape_q_prefill, equal_nan=False
        )
    )

    self.assertTrue(jax.numpy.allclose(attention_wo_reshape_q_prefill, attention_w_reshape_q_prefill, equal_nan=False))
    self.assertTrue(
        jax.numpy.allclose(
            attention_wo_reshape_q_full[:, :prefill_length, :],
            attention_w_reshape_q_full[:, :prefill_length, :],
            equal_nan=False,
        )
    )

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]

      attention_wo_reshape_q_idx, _ = attention_wo_reshape_q(
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

      attention_wo_reshape_q_full_this_idx = attention_wo_reshape_q_full[:, idx : idx + 1, :]
      self.assertTrue(attention_wo_reshape_q_full_this_idx.shape == attention_wo_reshape_q_idx.shape)
      self.assertTrue(
          jax.numpy.allclose(
              attention_wo_reshape_q_full_this_idx, attention_wo_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False
          )
      )

      attention_w_reshape_q_idx, _ = attention_w_reshape_q(
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

      attention_w_reshape_q_full_this_idx = attention_w_reshape_q_full[:, idx : idx + 1, :]
      self.assertTrue(attention_w_reshape_q_full_this_idx.shape == attention_w_reshape_q_idx.shape)
      self.assertTrue(
          jax.numpy.allclose(
              attention_w_reshape_q_full_this_idx, attention_w_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False
          )
      )

      self.assertTrue(
          jax.numpy.allclose(attention_w_reshape_q_idx, attention_wo_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False)
      )

  def test_sliding_window_attention(self):
    """Test sliding window attention"""

    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(self.dtype)

    dummy_inputs_q = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    dummy_inputs_kv = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))

    # Global Attention
    global_attn = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        attention_type=AttentionType.GLOBAL,
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.nnx_rng,
    )

    # Attention with sliding window of size 8
    sliding_attn = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        attention_type=AttentionType.LOCAL_SLIDING,
        sliding_window_size=8,
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.nnx_rng,
    )

    # To share parameters, we copy the state from sliding_attn to global_attn.
    sliding_attn_state = nnx.state(sliding_attn)
    nnx.update(global_attn, sliding_attn_state)

    global_attn_output, _ = global_attn(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    sliding_window_output, _ = sliding_attn(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    # Test if sliding window attention is different from global attention
    self.assertFalse(
        jax.numpy.allclose(
            sliding_window_output.astype(jnp.bfloat16), global_attn_output.astype(jnp.bfloat16), rtol=1e-04, atol=1e-04
        )
    )

    # Attention with sliding window of size max_target_length
    # This should be equivalent to global attention.
    sliding_attn_full_window = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        attention_type=AttentionType.LOCAL_SLIDING,
        sliding_window_size=self.max_target_length,
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.nnx_rng,
    )

    nnx.update(sliding_attn_full_window, sliding_attn_state)

    sliding_window_output_full, _ = sliding_attn_full_window(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    print(f"{sliding_window_output_full.astype(jnp.bfloat16)=}")
    print(f"{global_attn_output.astype(jnp.bfloat16)=}")

    # Test if sliding window attention with max_target_length size is the same as global attention
    self.assertTrue(
        jax.numpy.allclose(
            sliding_window_output_full.astype(jnp.bfloat16),
            global_attn_output.astype(jnp.bfloat16),
            rtol=1e-04,
            atol=1e-04,
        )
    )

  @pytest.mark.skip(reason="Requires `vllm-tpu` package which is not yet a MaxText dependency.")
  @pytest.mark.tpu_only
  @mock.patch("tpu_inference.layers.jax.attention_interface.sharded_ragged_paged_attention", create=True)
  def test_forward_serve_vllm(self, mock_sharded_ragged_paged_attention):
    """Tests the forward_serve_vllm method with mocked RPA attention."""
    # Setup config for vLLM RPA
    vllm_config_arguments = self.config_arguments.copy()
    vllm_config_arguments["attention"] = "vllm_rpa"
    vllm_config_arguments["chunk_attn_window_size"] = 128
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **vllm_config_arguments,
    )

    seq_len = self.max_target_length

    # Create Attention instance
    dummy_inputs_q = jnp.ones((self.global_batch_size, seq_len, self.embed_dim))
    dummy_inputs_kv = jnp.ones((self.global_batch_size, seq_len, self.embed_dim))
    attention_vllm = Attention(
        config=config,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=self.dtype,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        rngs=self.nnx_rng,
    )

    # Prepare inputs
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(self.dtype)
    mock_kv_cache = [jnp.ones((1,))]

    mock_attention_metadata = mock.Mock()
    mock_attention_metadata.seq_lens = jnp.array([1] * self.global_batch_size)
    mock_attention_metadata.block_tables = jnp.array([[0]] * self.global_batch_size)
    mock_attention_metadata.query_start_loc = jnp.array(list(range(self.global_batch_size)))
    mock_attention_metadata.request_distribution = jnp.array([self.global_batch_size])

    # Mock the return value of sharded_ragged_paged_attention
    total_tokens = self.global_batch_size * seq_len
    mock_output_shape = (total_tokens, self.num_query_heads, self.head_dim)
    mock_output = jnp.ones(mock_output_shape, dtype=self.dtype)
    mock_updated_kv_cache = [jnp.zeros((1,))]

    mock_callable = mock.Mock(return_value=(mock_output, mock_updated_kv_cache))
    mock_sharded_ragged_paged_attention.return_value = mock_callable

    # Call the attention layer
    output, updated_kv_cache = attention_vllm(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        kv_cache=mock_kv_cache,
        attention_metadata=mock_attention_metadata,
    )

    # Assertions
    mock_sharded_ragged_paged_attention.assert_called_once()
    mock_callable.assert_called_once()
    self.assertEqual(updated_kv_cache, mock_updated_kv_cache)

    # The output of forward_serve_vllm is reshaped back to (batch, seq, ...)
    reshaped_mock_output = mock_output.reshape(self.global_batch_size, seq_len, self.num_query_heads, self.head_dim)
    expected_output = attention_vllm.out_projection(reshaped_mock_output)
    self.assertTrue(jnp.allclose(output, expected_output))
    self.assertEqual(output.shape, (self.global_batch_size, seq_len, self.embed_dim))


class MLATest(parameterized.TestCase):
  """Test for the Multi-Headed Latent Attention"""

  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "max_target_length": 128,
      "max_prefill_predict_length": 16,
      "attention_type": AttentionType.MLA.value,
      "head_dim": 192,
      "q_lora_rank": 10,
      "kv_lora_rank": 20,
      "qk_nope_head_dim": 128,
      "qk_rope_head_dim": 64,
      "v_head_dim": 192,
  }

  def setUp(self):
    """Initializes the configuration for each test"""
    super().setUp()
    jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **self.config_arguments,
    )
    self.cfg = config
    self.rng = jax.random.PRNGKey(0)
    self.nnx_rng = nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42))
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

  def init_mla(self, config_arguments, rope_type):
    """Helper function to initialize MLA with different model names."""
    cfg = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **config_arguments,
        rope_type=rope_type,
    )

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    dummy_inputs_q = jnp.ones((cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.base_emb_dim))
    dummy_inputs_kv = jnp.ones((cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.base_emb_dim))

    mla = MLA(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        mesh=mesh,
        attention_kernel="dot_product",
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        attention_type=cfg.attention_type,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )

    return cfg, mla

  def get_data(self, cfg, dtype):
    """get data"""
    lnx = jax.random.normal(
        self.rng,
        shape=(cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.base_emb_dim),
        dtype=dtype,
    )

    decoder_segment_ids = jax.random.randint(self.rng, (cfg.global_batch_size_to_train_on, cfg.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        self.rng, (cfg.global_batch_size_to_train_on, cfg.max_target_length), 0, cfg.max_target_length
    )

    return lnx, decoder_segment_ids, decoder_positions

  def get_structured_data(self, cfg, dtype):
    """get structured data"""
    lnx = jax.random.normal(
        self.rng,
        shape=(
            cfg.global_batch_size_to_train_on,
            cfg.max_target_length,
            cfg.base_emb_dim,
        ),
        dtype=dtype,
    )

    decoder_positions = jnp.stack(
        [jnp.arange(cfg.max_target_length, dtype=jnp.int32) for _ in range(cfg.global_batch_size_to_train_on)]
    )

    decoder_segment_ids = (
        jax.numpy.zeros((cfg.global_batch_size_to_train_on, cfg.max_target_length)) + DECODING_ACTIVE_SEQUENCE_INDICATOR
    )

    return lnx, decoder_segment_ids, decoder_positions

  @parameterized.named_parameters(
      {"testcase_name": "RoPE_Yarn_Autoregression", "rope_type": "yarn"},
      {"testcase_name": "Default_Autoregression", "rope_type": "default"},
  )
  @pytest.mark.tpu_only
  def test_autoregression(self, rope_type):
    cfg, mla = self.init_mla(self.config_arguments, rope_type)
    prefill_length = cfg.max_prefill_predict_length
    decode_total_length = cfg.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg, cfg.dtype)

    mla_full, _ = mla(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    mla_prefill, _ = mla(
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )

    self.assertTrue(
        jax.numpy.allclose(mla_prefill, mla_full[:, :prefill_length, :], rtol=1e-02, atol=1e-02, equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      mla_idx, _ = mla(
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

      mla_full_this_idx = mla_full[:, idx : idx + 1, :]
      self.assertEqual(mla_full_this_idx.shape, mla_idx.shape)
      # TODO (b/394626702) uncomment last check when decode and kv_cache are implemented for MLA
      # self.assertTrue(jax.numpy.allclose(mla_full_this_idx, mla_idx, rtol=1e-02, atol=1e-02, equal_nan=False))

  def test_projection_initialization(self):
    """Tests that MLA and Attention layers initialize the correct projection weights."""
    # 1. Initialize a standard Attention layer for comparison
    # Create a copy of the arguments and override the attention_type for the base model
    attention_config_args = self.config_arguments.copy()
    attention_config_args["attention_type"] = AttentionType.GLOBAL.value
    attention_cfg = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **attention_config_args,
    )
    dummy_inputs_q = jnp.ones(
        (attention_cfg.global_batch_size_to_train_on, attention_cfg.max_target_length, attention_cfg.base_emb_dim)
    )
    dummy_inputs_kv = jnp.ones(
        (attention_cfg.global_batch_size_to_train_on, attention_cfg.max_target_length, attention_cfg.base_emb_dim)
    )

    base_attention = Attention(
        config=attention_cfg,
        num_query_heads=attention_cfg.num_query_heads,
        num_kv_heads=attention_cfg.num_kv_heads,
        head_dim=attention_cfg.head_dim,
        max_target_length=attention_cfg.max_target_length,
        max_prefill_predict_length=attention_cfg.max_prefill_predict_length,
        inputs_q_shape=dummy_inputs_q.shape,
        inputs_kv_shape=dummy_inputs_kv.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=attention_cfg.dtype,
        rngs=self.nnx_rng,
    )

    # 2. Assert that the base Attention layer HAS all its standard projections
    self.assertTrue(hasattr(base_attention, "query"), "Base Attention should have 'query' projection.")
    self.assertTrue(hasattr(base_attention, "key"), "Base Attention should have 'key' projection.")
    self.assertTrue(hasattr(base_attention, "value"), "Base Attention should have 'value' projection.")
    self.assertTrue(hasattr(base_attention, "out"), "Base Attention should have 'out' projection.")

    # 3. Initialize the MLA layer
    _, mla_layer = self.init_mla(self.config_arguments, rope_type="default")

    # 4. Assert that the MLA layer DOES NOT HAVE the base projections
    self.assertFalse(hasattr(mla_layer, "query"), "MLA should not have 'query' projection.")
    self.assertFalse(hasattr(mla_layer, "key"), "MLA should not have 'key' projection.")
    self.assertFalse(hasattr(mla_layer, "value"), "MLA should not have 'value' projection.")

    # 5. Assert that the MLA layer HAS all of its own specific projections AND the common 'out' projection
    self.assertTrue(hasattr(mla_layer, "wq_a"), "MLA should have 'wq_a' projection.")
    self.assertTrue(hasattr(mla_layer, "wq_b"), "MLA should have 'wq_b' projection.")
    self.assertTrue(hasattr(mla_layer, "wkv_a"), "MLA should have 'wkv_a' projection.")
    self.assertTrue(hasattr(mla_layer, "wkv_b"), "MLA should have 'wkv_b' projection.")
    self.assertTrue(hasattr(mla_layer, "q_norm"), "MLA should have 'q_norm' projection.")
    self.assertTrue(hasattr(mla_layer, "kv_norm"), "MLA should have 'kv_norm' projection.")
    self.assertTrue(hasattr(mla_layer, "out"), "MLA should have 'out' projection.")

  @parameterized.named_parameters(
      {
          "testcase_name": "cp_no_load_balance",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 1,
          "expert_shard_attention_option": "fsdp",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_with_load_balance",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 1,
          "expert_shard_attention_option": "fsdp",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_ep_no_load_balance",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 2,
          "expert_shard_attention_option": "context",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_ep_with_load_balance",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 2,
          "expert_shard_attention_option": "context",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "ep_no_load_balance",
          "ici_context_parallelism": 1,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 4,
          "expert_shard_attention_option": "context",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "ep_with_load_balance",
          "ici_context_parallelism": 1,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 4,
          "expert_shard_attention_option": "context",
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_no_load_balance_explicit",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 1,
          "expert_shard_attention_option": "fsdp",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_with_load_balance_explicit",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 1,
          "expert_shard_attention_option": "fsdp",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_ep_no_load_balance_explicit",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 2,
          "expert_shard_attention_option": "context",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_ep_with_load_balance_explicit",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 2,
          "expert_shard_attention_option": "context",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "ep_no_load_balance_explicit",
          "ici_context_parallelism": 1,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 4,
          "expert_shard_attention_option": "context",
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "ep_with_load_balance_explicit",
          "ici_context_parallelism": 1,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 4,
          "expert_shard_attention_option": "context",
          "shard_mode": "explicit",
      },
  )
  # TODO (b/454764135.) : This tests fails with new tokamax kernel
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_context_parallel(
      self,
      ici_context_parallelism,
      context_parallel_load_balance,
      ici_expert_parallelism,
      expert_shard_attention_option,
      shard_mode,
  ):
    """Test equivalence between dot_product and flash attention + context/expert parallelism"""

    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 512,
        "sa_block_q": 128,
        "sa_block_kv": 128,
        "sa_block_kv_compute": 128,
        "sa_block_q_dkv": 128,
        "sa_block_kv_dkv": 128,
        "sa_block_kv_dkv_compute": 128,
        "sa_block_q_dq": 128,
        "sa_block_kv_dq": 128,
        "attention_type": AttentionType.MLA.value,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "shard_mode": shard_mode,
    }

    cfg, mla = self.init_mla(config_arguments, rope_type="default")
    lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg, cfg.dtype)
    # Dot product
    mla_generic_output, _ = mla(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    generic_state = nnx.state(mla)

    # Test with Context Parallelism
    cfg_cp = pyconfig.initialize(
        [sys.argv[0], os.path.join(MAXTEXT_PKG_DIR, "configs", "base.yml")],
        **config_arguments,
        rope_type=cfg.rope_type,
        ici_context_parallelism=ici_context_parallelism,
        context_parallel_load_balance=context_parallel_load_balance,
        ici_expert_parallelism=ici_expert_parallelism,
        expert_shard_attention_option=expert_shard_attention_option,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    axis_type = AxisType.Explicit if shard_mode == "explicit" else AxisType.Auto
    axis_names = [axis_type for _ in cfg_cp.mesh_axes]
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes, axis_types=tuple(axis_names))
    attention_as_mla_flash_cp = MLA(
        config=cfg_cp,
        num_query_heads=cfg_cp.num_query_heads,
        num_kv_heads=cfg_cp.num_kv_heads,
        head_dim=cfg_cp.head_dim,
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        max_target_length=cfg_cp.max_target_length,
        max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
        mesh=mesh_cp,
        attention_kernel="flash",
        dtype=cfg_cp.dtype,
        dropout_rate=cfg_cp.dropout_rate,
        attention_type=cfg_cp.attention_type,
        q_lora_rank=cfg_cp.q_lora_rank,
        kv_lora_rank=cfg_cp.kv_lora_rank,
        qk_nope_head_dim=cfg_cp.qk_nope_head_dim,
        qk_rope_head_dim=cfg_cp.qk_rope_head_dim,
        v_head_dim=cfg_cp.v_head_dim,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )
    nnx.update(attention_as_mla_flash_cp, generic_state)
    mla_generic_flash_cp_output = _forward_with_context_expert_parallelism(
        cfg_cp, mesh_cp, attention_as_mla_flash_cp, lnx, decoder_segment_ids, decoder_positions
    )

    # This removes all sharding information and makes them standard NumPy arrays.
    mla_generic_output = jax.device_get(mla_generic_output)
    mla_generic_flash_cp_output = jax.device_get(mla_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mla_generic_output, mla_generic_flash_cp_output, rtol=1e-01, atol=1e-01, equal_nan=False),
        msg="MLA Logits from generic dot product and flash attention + context/expert parallelism are not close.\n"
        f"ici_context_parallelism={ici_context_parallelism}, context_parallel_load_balance={context_parallel_load_balance},"
        f" ici_expert_parallelism={ici_expert_parallelism}, expert_shard_attention_option={expert_shard_attention_option}.",
    )


def _forward_with_context_expert_parallelism(cfg_cp, mesh_cp, attention_cp, lnx, decoder_segment_ids, decoder_positions):
  """Get logits from attention under context/expert parallelism."""
  # If load balanced cp, shuffle along seq dim for input
  # This corresponds to the pre-shuffle step in training
  context_parallel_size = cfg_cp.context_parallel_size
  if context_parallel_size > 1 and cfg_cp.context_parallel_load_balance:
    batch = {"inputs": lnx, "inputs_segmentation": decoder_segment_ids, "inputs_position": decoder_positions}
    with mesh_cp:
      reordered_batch = maxtext_utils.get_reorder_callable(context_parallel_size, ShardMode.AUTO)(batch)
    lnx = reordered_batch["inputs"]
    decoder_segment_ids = reordered_batch["inputs_segmentation"]
    decoder_positions = reordered_batch["inputs_position"]
  # apply attention with sharding
  with mesh_cp, nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
    if cfg_cp.expert_shard_attention_option == EP_AS_CONTEXT:
      batch_axis = "activation_batch_no_exp"
      length_axis = "activation_length"
    else:
      batch_axis = "activation_batch"
      length_axis = "activation_length_no_exp"
    lnx_spec = nn_partitioning.logical_to_mesh_axes(
        (batch_axis, length_axis, "activation_embed"), nn_partitioning.get_axis_rules()
    )
    pos_spec = nn_partitioning.logical_to_mesh_axes((batch_axis, length_axis), nn_partitioning.get_axis_rules())
    lnx_sharding = NamedSharding(mesh_cp, lnx_spec)
    pos_sharding = NamedSharding(mesh_cp, pos_spec)

    lnx = jax.device_put(lnx, lnx_sharding)
    decoder_segment_ids = jax.device_put(decoder_segment_ids, pos_sharding)
    decoder_positions = jax.device_put(decoder_positions, pos_sharding)

    attention_cp_output, _ = attention_cp(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

  attention_cp_output = attention_cp_output[0] if isinstance(attention_cp_output, tuple) else attention_cp_output
  # All-gather before re-shuffle to avoid re-order sharding confusion
  repeat_sharding = NamedSharding(mesh_cp, P())
  attention_cp_output = maybe_shard_with_name(attention_cp_output, repeat_sharding, shard_mode=cfg_cp.shard_mode)

  # If load balanced cp, de-shuffle and gather along seq dim for output
  # Note training does not need post-shuffle. Since the target seq is also pre-shuffled, the loss remains correct
  if context_parallel_size > 1 and cfg_cp.context_parallel_load_balance:
    attention_cp_output = max_utils.reorder_sequence(
        tensor=attention_cp_output, cp_size=context_parallel_size, seq_dim=1, to_contiguous=True
    )
  return attention_cp_output


if __name__ == "__main__":
  unittest.main()
