#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for Attentions."""

import itertools
import os.path
import random
import sys
import unittest

import pytest

from absl.testing import parameterized

import numpy as np

from jax.sharding import Mesh
import jax
import jax.numpy as jnp

from flax.core import freeze

from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.common_types import DECODING_ACTIVE_SEQUENCE_INDICATOR, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_PREFILL, MODEL_MODE_TRAIN
from MaxText.globals import PKG_DIR
from MaxText.layers import attentions
from MaxText.layers.attentions import Attention, MLA, ChunkedCausalMask
from MaxText import max_utils
from flax.linen import partitioning as nn_partitioning


class BidirectionalBlockMaskTest(unittest.TestCase):
  """Test for make_bidirectional_block_mask."""

  def test_one_block_mask(self):
    bidirectional_mask = np.asarray([[0, 1, 1, 1, 0, 0]])
    # pylint: disable=protected-access
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
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
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
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
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
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
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
    expected_mask = np.zeros(
        (bidirectional_mask.shape[0], bidirectional_mask.shape[1], bidirectional_mask.shape[1]), dtype=bool
    )
    np.testing.assert_array_equal(block_mask, expected_mask)

  def test_full_block_mask(self):
    bidirectional_mask = np.asarray([[1, 1, 1, 1, 1, 1]])
    # pylint: disable=protected-access
    block_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
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
    image_mask = attentions._make_bidirectional_block_mask(bidirectional_mask)
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
    actual_mask = attentions._generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
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
    actual_mask = attentions._generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
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
    actual_mask = attentions._generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
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
    actual_mask = attentions._generate_chunk_attention_mask(mask_shape=mask.shape, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_value_error_on_zero_chunk_size(self):
    """Tests that a ValueError is raised for chunk_size <= 0."""
    with self.assertRaises(ValueError):
      ChunkedCausalMask(shape=(4, 4), chunk_size=0)
    with self.assertRaises(ValueError):
      ChunkedCausalMask(shape=(4, 4), chunk_size=-2)
    with self.assertRaises(ValueError):
      # pylint: disable=protected-access
      attentions._generate_chunk_attention_mask(mask_shape=(4, 4), chunk_size=0)


class AttentionTest(unittest.TestCase):
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
    super().setUp()
    config = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **self.config_arguments,
    )
    self.cfg = config

    config_cp = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **self.config_arguments,
        ici_context_parallelism=4,  # use context parallelism of 4
        context_parallel_load_balance=False,  # set load_balancing to False such that
        # there's no need for reordering the input/output
    )

    self.cfg_cp = config_cp
    self.rng = jax.random.PRNGKey(0)

    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    devices_array_cp = maxtext_utils.create_device_mesh(self.cfg_cp)  # for context parallelism
    self.mesh_cp = Mesh(devices_array_cp, self.cfg_cp.mesh_axes)  # for context parallelism
    self.global_batch_size = self.cfg.global_batch_size_to_train_on
    self.num_kv_heads = self.cfg.num_kv_heads
    self.num_query_heads = self.cfg.num_query_heads
    self.max_target_length = self.cfg.max_target_length
    self.max_prefill_predict_length = self.cfg.max_prefill_predict_length
    self.head_dim = self.cfg.head_dim
    self.embed_dim = self.cfg.base_emb_dim
    self.dtype = self.cfg.dtype
    self.attention_type = self.cfg.attention_type

    self._attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="self_attention",
        attention_type=self.attention_type,
    )

    self._attention_as_mha_generic_variable = self._attention_as_mha_generic.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
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
  def test_cp_shard_with_load_balance(self):
    self.test_cp_shard_helper(load_balance=True)

  @pytest.mark.tpu_only
  def test_cp_shard_without_load_balance(self):
    self.test_cp_shard_helper(load_balance=False)

  def test_cp_shard_helper(self, load_balance: bool = False):
    """Test equivalence between dot_product and TPU accelerated"""
    num_kv_heads = self.num_kv_heads
    lnx, decoder_segment_ids, decoder_positions = self.get_data(self.dtype)

    mha_generic_output = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    # Test with Context Parallelism

    config_cp = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        **self.config_arguments,
        ici_context_parallelism=4,
        context_parallel_load_balance=load_balance,
    )
    self.cfg_cp = config_cp
    devices_array_cp = maxtext_utils.create_device_mesh(self.cfg_cp)
    self.mesh_cp = Mesh(devices_array_cp, self.cfg_cp.mesh_axes)

    attention_as_mha_flash_cp = Attention(
        config=self.cfg_cp,  # we pass the context parallelism in the config
        num_query_heads=self.cfg_cp.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.cfg_cp.head_dim,
        max_target_length=self.cfg_cp.max_target_length,
        max_prefill_predict_length=self.cfg_cp.max_prefill_predict_length,
        mesh=self.mesh_cp,
        attention_kernel="flash",
        dtype=self.dtype,
        dropout_rate=self.cfg_cp.dropout_rate,
        name="self_attention_cp",
    )
    attention_as_mha_flash_cp_variable = attention_as_mha_flash_cp.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    #### if load balance, need pre-shuffle
    # https://github.com/AI-Hypercomputer/maxtext/blob/2adc3bab1cea2e5a335b9e4c053c868155da2089/MaxText/train.py#L526-L540
    context_parallel_size = self.cfg_cp.ici_context_parallelism
    if context_parallel_size > 1 and self.cfg_cp.context_parallel_load_balance:
      batch = {"inputs": lnx, "inputs_segmentation": decoder_segment_ids, "inputs_position": decoder_positions}
      with self.mesh_cp:
        reorder_fn = max_utils.get_reorder_callable(context_parallel_size)
        reordered_batch = reorder_fn(batch)
      lnx = reordered_batch["inputs"]
      decoder_segment_ids = reordered_batch["inputs_segmentation"]
      decoder_positions = reordered_batch["inputs_position"]
      jax.debug.print("batch {x}", x=batch)
      jax.debug.print("reordered_batch {x}", x=reordered_batch)
      # lnx = max_utils.reorder_sequence(lnx, cp_size=context_parallel_size, seq_dim=1, to_contiguous=False)
      # decoder_positions = max_utils.reorder_sequence(decoder_positions, cp_size=context_parallel_size, seq_dim=1, to_contiguous=False)
      # decoder_segment_ids = max_utils.reorder_sequence(
      #     decoder_segment_ids, cp_size=context_parallel_size, seq_dim=1, to_contiguous=False
      # )
    ####

    with self.mesh_cp, nn_partitioning.axis_rules(self.cfg_cp.logical_axis_rules):
      mha_generic_flash_cp_output = attention_as_mha_flash_cp.apply(
          attention_as_mha_flash_cp_variable,
          lnx,
          lnx,
          decoder_segment_ids=decoder_positions,
          inputs_positions=decoder_segment_ids,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
          rngs={"aqt": self.rng},
      )

    # jax.debug.print("mha_generic_output {x}", x=mha_generic_output)
    # jax.debug.print("mha_generic_flash_cp_output {x}", x=mha_generic_flash_cp_output)

    # deshuffle and gather output from load balanced cp
    if context_parallel_size > 1 and self.cfg_cp.context_parallel_load_balance:
      mha_generic_flash_cp_output = max_utils.reorder_sequence(
          tensor=mha_generic_flash_cp_output, cp_size=context_parallel_size, seq_dim=1, to_contiguous=True
      )
    # Assert that the logits generated by the generic dot product and flash attention+context parallelism are close
    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_cp_output, rtol=1e-01, atol=1e-01, equal_nan=False),
        msg="Logits from generic dot product and flash context parallelism are not close.",
    )

  @pytest.mark.tpu_only
  def test_autoregression(self):
    prefill_length = self.cfg.max_prefill_predict_length
    decode_total_length = self.cfg.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(self.dtype)

    mha_full = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    mha_prefill, output_cache = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )

    self.assertTrue(
        jax.numpy.allclose(mha_prefill, mha_full[:, :prefill_length, :], rtol=1e-02, atol=1e-02, equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      self._attention_as_mha_generic_variable.update(output_cache)
      mha_idx, output_cache = self._attention_as_mha_generic.apply(
          self._attention_as_mha_generic_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
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

    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="self_attention",
    )

    attention_as_mha_generic_variable = attention_as_mha_generic.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_prefill, _ = attention_as_mha_generic.apply(
        attention_as_mha_generic_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
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

    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="self_attention",
    )

    attention_as_mha_generic_variable = attention_as_mha_generic.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_generic_output = attention_as_mha_generic.apply(
        attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_as_mha_flash = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel="flash",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="self_attention",
    )

    attention_as_mha_flash_variable = attention_as_mha_flash.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_generic_flash_output = attention_as_mha_flash.apply(
        attention_as_mha_flash_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_output, rtol=1e-01, atol=1e-01, equal_nan=False)
    )

    # Test with Context Parallelism
    attention_as_mha_flash_cp = Attention(
        config=self.cfg_cp,  # we pass the context parallelism in the config
        num_query_heads=self.cfg_cp.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.cfg_cp.head_dim,
        max_target_length=self.cfg_cp.max_target_length,
        max_prefill_predict_length=self.cfg_cp.max_prefill_predict_length,
        mesh=self.mesh_cp,
        attention_kernel="flash",
        dtype=self.dtype,
        dropout_rate=self.cfg_cp.dropout_rate,
        name="self_attention_cp",
    )
    attention_as_mha_flash_cp_variable = attention_as_mha_flash_cp.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones((self.global_batch_size, self.max_target_length)),
    )

    mha_generic_flash_cp_output = attention_as_mha_flash_cp.apply(
        attention_as_mha_flash_cp_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    # Assert that the logits generated by the generic flash and flash attention+context parallelism are close
    self.assertTrue(
        jax.numpy.allclose(mha_generic_flash_output, mha_generic_flash_cp_output, rtol=1e-01, atol=1e-01, equal_nan=False),
        msg="Logits from generic flash and flash attention+context parallelism are not close.",
    )

    # Assert that the logits generated by the generic dot product and flash attention+context parallelism are close
    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_cp_output, rtol=1e-01, atol=1e-01, equal_nan=False),
        msg="Logits from generic dot product and flash attention+context parallelism are not close.",
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
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
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

    attention_w_layout = Attention(
        mesh=self.mesh,
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        dtype=config.dtype,
        prefill_cache_axis_order=prefill_cache_axis_order,
        ar_cache_axis_order=ar_cache_axis_order,
        compute_axis_order=compute_axis_order,
    )
    attention_w_layout_variable = attention_w_layout.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length)),
    )
    attention_w_layout_full = attention_w_layout.apply(
        attention_w_layout_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_w_layout_prefill, attention_w_layout_output_cache = attention_w_layout.apply(
        attention_w_layout_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    self.assertTrue(
        jax.numpy.allclose(attention_w_layout_full[:, :prefill_length, :], attention_w_layout_prefill, equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):

      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]

      attention_w_layout_variable.update(attention_w_layout_output_cache)
      attention_w_layout_idx, attention_w_layout_output_cache = attention_w_layout.apply(
          attention_w_layout_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )

      attention_w_layout_full_this_idx = attention_w_layout_full[:, idx : idx + 1, :]
      self.assertTrue(attention_w_layout_full_this_idx.shape == attention_w_layout_idx.shape)
      self.assertTrue(
          jax.numpy.allclose(attention_w_layout_full_this_idx, attention_w_layout_idx, rtol=rtol, atol=atol, equal_nan=False)
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
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
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

    attention_wo_reshape_q = Attention(
        mesh=self.mesh,
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        dtype=config.dtype,
        compute_axis_order=compute_axis_order,
        reshape_q=False,
    )
    attention_wo_reshape_q_variable = attention_wo_reshape_q.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length)),
    )

    attention_w_reshape_q = Attention(
        mesh=self.mesh,
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        max_prefill_predict_length=config.max_prefill_predict_length,
        attention_kernel=config.attention,
        dtype=config.dtype,
        compute_axis_order=compute_axis_order,
        reshape_q=True,
    )
    attention_w_reshape_q_variable = attention_w_reshape_q.init(
        {"params": self.rng, "aqt": self.rng},
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length, config.base_emb_dim)),
        jnp.ones((self.global_batch_size, config.max_target_length)),
    )

    attention_wo_reshape_q_full = attention_wo_reshape_q.apply(
        attention_wo_reshape_q_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_w_reshape_q_full = attention_w_reshape_q.apply(
        attention_w_reshape_q_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    attention_wo_reshape_q_prefill, attention_wo_reshape_q_output_cache = attention_wo_reshape_q.apply(
        attention_wo_reshape_q_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    self.assertTrue(
        jax.numpy.allclose(
            attention_wo_reshape_q_full[:, :prefill_length, :], attention_wo_reshape_q_prefill, equal_nan=False
        )
    )

    attention_w_reshape_q_prefill, attention_w_reshape_q_output_cache = attention_w_reshape_q.apply(
        attention_w_reshape_q_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
        rngs={"aqt": self.rng},
        mutable=["cache"],
    )
    self.assertTrue(
        jax.numpy.allclose(attention_w_reshape_q_full[:, :prefill_length, :], attention_w_reshape_q_prefill, equal_nan=False)
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

      attention_wo_reshape_q_variable.update(attention_wo_reshape_q_output_cache)
      attention_wo_reshape_q_idx, attention_wo_reshape_q_output_cache = attention_wo_reshape_q.apply(
          attention_wo_reshape_q_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
      )

      attention_wo_reshape_q_full_this_idx = attention_wo_reshape_q_full[:, idx : idx + 1, :]
      self.assertTrue(attention_wo_reshape_q_full_this_idx.shape == attention_wo_reshape_q_idx.shape)
      self.assertTrue(
          jax.numpy.allclose(
              attention_wo_reshape_q_full_this_idx, attention_wo_reshape_q_idx, rtol=rtol, atol=atol, equal_nan=False
          )
      )

      attention_w_reshape_q_variable.update(attention_w_reshape_q_output_cache)
      attention_w_reshape_q_idx, attention_w_reshape_q_output_cache = attention_w_reshape_q.apply(
          attention_w_reshape_q_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": self.rng},
          mutable=["cache"],
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

    # Global Attention
    global_attn = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="global_attention",
        attention_type=attentions.AttentionType.GLOBAL,
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
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="sliding_window_attention",
        attention_type=attentions.AttentionType.LOCAL_SLIDING,
        sliding_window_size=8,
    )

    # Use freeze to fix the parameters to facilitate the comparison of sliding and global attention.
    attn_variable = freeze(
        sliding_attn.init(
            {"params": self.rng, "aqt": self.rng},
            jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
            jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim)),
            jnp.ones((self.global_batch_size, self.max_target_length)),
        )
    )

    global_attn_output = global_attn.apply(
        attn_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    sliding_window_output = sliding_attn.apply(
        attn_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    # Test if sliding window attention is different from global attention
    self.assertFalse(
        jax.numpy.allclose(
            sliding_window_output.astype(jnp.bfloat16), global_attn_output.astype(jnp.bfloat16), rtol=1e-04, atol=1e-04
        )
    )

    # Attention with sliding window of size max_target_length
    # This should be equivalent to global attension.
    sliding_attn = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name="sliding_window_attention",
        attention_type=attentions.AttentionType.LOCAL_SLIDING,
        sliding_window_size=self.max_target_length,
    )

    sliding_window_output = sliding_attn.apply(
        attn_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": self.rng},
    )

    # Test if sliding window attention with max_target_length size is the same as global attention
    self.assertTrue(
        jax.numpy.allclose(
            sliding_window_output.astype(jnp.bfloat16), global_attn_output.astype(jnp.bfloat16), rtol=1e-04, atol=1e-04
        )
    )


class MLATest(parameterized.TestCase):
  """Test for the Multi-Headed Latent Attention"""

  def init_mla(self, rope_type):
    """Helper function to initialize MLA with different model names."""
    cfg = pyconfig.initialize(
        [sys.argv[0], os.path.join(PKG_DIR, "configs", "base.yml")],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=128,
        max_prefill_predict_length=16,
        attention_type=attentions.AttentionType.MLA.value,
        rope_type=rope_type,
    )
    rng = jax.random.PRNGKey(0)

    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    global_batch_size = cfg.global_batch_size_to_train_on
    num_kv_heads = cfg.num_kv_heads
    num_query_heads = cfg.num_query_heads
    max_target_length = cfg.max_target_length
    max_prefill_predict_length = cfg.max_prefill_predict_length
    head_dim = cfg.head_dim
    embed_dim = cfg.base_emb_dim
    dtype = cfg.dtype
    attention_type = cfg.attention_type

    mla = MLA(
        config=cfg,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        max_prefill_predict_length=max_prefill_predict_length,
        mesh=mesh,
        attention_kernel="dot_product",
        dtype=dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        attention_type=attention_type,
        q_lora_rank=10,
        kv_lora_rank=20,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=192,
    )

    mla_variable = mla.init(
        {"params": rng, "aqt": rng},
        jnp.ones((global_batch_size, max_target_length, embed_dim)),
        jnp.ones((global_batch_size, max_target_length, embed_dim)),
        jnp.ones((global_batch_size, max_target_length)),
    )

    return cfg, mla, mla_variable, rng

  def get_data(self, cfg, rng, dtype):
    """get data"""
    lnx = jax.random.normal(
        rng,
        shape=(cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.base_emb_dim),
        dtype=dtype,
    )

    decoder_segment_ids = jax.random.randint(rng, (cfg.global_batch_size_to_train_on, cfg.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(
        rng, (cfg.global_batch_size_to_train_on, cfg.max_target_length), 0, cfg.max_target_length
    )

    return lnx, decoder_segment_ids, decoder_positions

  def get_structured_data(self, cfg, rng, dtype):
    """get structured data"""
    lnx = jax.random.normal(
        rng,
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
    cfg, mla, mla_variable, rng = self.init_mla(rope_type)
    prefill_length = cfg.max_prefill_predict_length
    decode_total_length = cfg.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg, rng, cfg.dtype)

    mla_full = mla.apply(
        mla_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
        rngs={"aqt": rng},
    )

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    mla_prefill, output_cache = mla.apply(
        mla_variable,
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
        rngs={"aqt": rng},
        mutable=["cache"],
    )

    self.assertTrue(
        jax.numpy.allclose(mla_prefill, mla_full[:, :prefill_length, :], rtol=1e-02, atol=1e-02, equal_nan=False)
    )

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]
      mla_variable.update(output_cache)
      mla_idx, output_cache = mla.apply(
          mla_variable,
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          rngs={"aqt": rng},
          mutable=["cache"],
      )

      mla_full_this_idx = mla_full[:, idx : idx + 1, :]
      self.assertEqual(mla_full_this_idx.shape, mla_idx.shape)
      # TODO (b/394626702) uncomment last check when decode and kv_cache are implemented for MLA
      # self.assertTrue(jax.numpy.allclose(mla_full_this_idx, mla_idx, rtol=1e-02, atol=1e-02, equal_nan=False))


if __name__ == "__main__":
  unittest.main()
