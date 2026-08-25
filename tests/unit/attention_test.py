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
import os
import random
import sys
import types
import unittest
import copy
from unittest import mock

from absl.testing import parameterized
from flax import nnx
from flax.linen import partitioning as nn_partitioning
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.sharding import AxisType, Mesh, NamedSharding
from maxtext.utils import max_utils
from maxtext.utils import maxtext_utils
from maxtext.utils import sharding
from maxtext.common.gcloud_stub import is_decoupled

from maxtext.common.common_types import (
    AttentionType,
    DECODING_ACTIVE_SEQUENCE_INDICATOR,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    DEFAULT_MASK_VALUE,
)
from maxtext.layers.attention_mla import MLA
from maxtext.layers.attention_compressed import CompressedAttention
from maxtext.layers import attention_op
from maxtext.layers.attention_op import (
    AttentionOp,
    BlockCausalMask,
    ChunkedCausalMask,
    _generate_block_causal_attention_mask,
    _generate_chunk_attention_mask,
    _make_bidirectional_block_mask,
    _resolve_attention_type,
)
from maxtext.layers.attentions import Attention
from maxtext.layers import embeddings
from maxtext.kernels.attention import jax_flash_attention
from maxtext.configs import pyconfig
from maxtext.models.qwen3 import Qwen3NextGatedDeltaNet
import numpy as np
import pytest

from tests.utils import attention_test_util
from tests.utils import hlo_test_utils
from tests.utils.test_helpers import get_test_config_path


class JaxFlashAttentionTest(unittest.TestCase):
  """Tests for JAX flash attention."""

  def test_flash_attention_block_masked_soft_cap(self):
    cap = 1.0
    mask_value = -1.0e9
    query = jnp.array([[[[10.0], [10.0]]]], dtype=jnp.float32)
    key = jnp.array([[[[10.0], [0.0]]]], dtype=jnp.float32)
    value = jnp.array([[[[1.0], [3.0]]]], dtype=jnp.float32)
    mask = jnp.array([[True, True], [True, False]])

    output = jax_flash_attention.flash_attention_block_masked(
        query,
        key,
        value,
        segment_ids=None,
        block_kv=2,
        block_q=1,
        mask=mask,
        mask_value=mask_value,
        cap=cap,
    )

    logits = jnp.einsum("bhqd,bhkd->bhqk", query, key)
    logits = jnp.tanh(logits / cap) * cap
    logits = jnp.where(mask[None, None, :, :], logits, mask_value)
    expected = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(logits, axis=-1), value)
    np.testing.assert_allclose(
        np.asarray(output),
        np.asarray(expected),
        rtol=1e-2,
        atol=1e-2,
    )


class SplashLocalMaskTest(unittest.TestCase):
  """Tests for Splash local masks."""

  def test_local_window_matches_dense_mask(self):
    seq_len = 8
    window_size = 3
    mask = splash_attention_mask.CausalMask((seq_len, seq_len)) & splash_attention_mask.LocalMask(
        (seq_len, seq_len),
        window_size=(window_size - 1, window_size),
        offset=0,
    )
    q_sequence = np.arange(seq_len)[:, None]
    kv_sequence = np.arange(seq_len)[None, :]
    expected_mask = (kv_sequence <= q_sequence) & (kv_sequence > q_sequence - window_size)

    np.testing.assert_array_equal(mask[:, :], expected_mask)


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


class BlockCausalMaskTest(unittest.TestCase):
  """Tests the shared dense and Splash block-causal masks."""

  def _make_op(self, sequence_length, *, attention_type=AttentionType.BLOCK_DIFFUSION):
    """Builds a minimal dot-product attention operator."""
    config = types.SimpleNamespace(
        causal_block_size=4,
        context_parallel_load_balance=False,
        context_sharding="context",
    )
    mesh = types.SimpleNamespace(shape={})
    kwargs = {}
    if attention_type is not None:
      kwargs["attention_type"] = attention_type
    return AttentionOp(
        config=config,
        num_query_heads=1,
        num_kv_heads=1,
        max_target_length=sequence_length,
        mesh=mesh,
        attention_kernel="dot_product",
        **kwargs,
    )

  def _make_flash_op(
      self,
      *,
      attention_type=AttentionType.BLOCK_DIFFUSION,
      context_parallel_size=1,
      context_parallel_load_balance=False,
      max_target_length=8,
  ):
    """Builds a minimal flash-attention operator for dispatch tests."""
    config = types.SimpleNamespace(
        causal_block_size=4,
        context_parallel_strategy="all_gather",
        context_parallel_load_balance=context_parallel_load_balance,
        context_sharding="context",
        sa_block_q=4,
        sa_block_kv=4,
        sa_block_kv_compute=4,
        sa_block_q_dkv=4,
        sa_block_kv_dkv=4,
        sa_block_kv_dkv_compute=4,
        sa_block_q_dq=4,
        sa_block_kv_dq=4,
        sa_use_fused_bwd_kernel=True,
        sa_q_layout="HEAD_DIM_MINOR",
        sa_k_layout="HEAD_DIM_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
        use_splash_scheduler=False,
        sa_fuse_reciprocal=False,
        sa_use_base2_exp=False,
        use_tokamax_splash=False,
        use_jax_splash=False,
        cost_estimate_flops_fwd=-1,
        cost_estimate_flops_bwd=-1,
        dq_reduction_steps=-1,
    )
    device = types.SimpleNamespace(platform="cpu")
    mesh = types.SimpleNamespace(
        devices=np.asarray([device] * context_parallel_size, dtype=object),
        shape={"context": context_parallel_size},
    )
    return AttentionOp(
        config=config,
        num_query_heads=1,
        num_kv_heads=1,
        max_target_length=max_target_length,
        mesh=mesh,
        attention_kernel="flash",
        attention_type=attention_type,
    )

  def test_dense_and_splash_masks_match(self):
    sequence_length = 10
    block_size = 4
    query_positions = np.arange(sequence_length)[:, None]
    key_positions = np.arange(sequence_length)[None, :]
    expected = query_positions // block_size >= key_positions // block_size

    splash_mask = BlockCausalMask((sequence_length, sequence_length), block_size)
    dense_mask = _generate_block_causal_attention_mask((sequence_length, sequence_length), block_size)

    np.testing.assert_array_equal(splash_mask[:, :], expected)
    np.testing.assert_array_equal(dense_mask, expected)
    self.assertTrue(expected[1, 3])
    self.assertFalse(expected[3, 4])
    self.assertTrue(expected[4, 3])
    self.assertTrue(expected[8, 9])

  def test_splash_mask_equality_and_hash(self):
    mask = BlockCausalMask((8, 8), 4)
    equivalent_mask = BlockCausalMask((8, 8), 4)

    self.assertEqual(mask, equivalent_mask)
    self.assertEqual(hash(mask), hash(equivalent_mask))
    self.assertNotEqual(mask, object())

  def test_rectangular_dense_mask_respects_query_offset(self):
    dense_mask = _generate_block_causal_attention_mask(
        mask_shape=(3, 8),
        causal_block_size=4,
        q_offset=2,
    )
    query_positions = np.arange(2, 5)[:, None]
    key_positions = np.arange(8)[None, :]
    expected = query_positions // 4 >= key_positions // 4

    np.testing.assert_array_equal(dense_mask, expected)

  def test_dot_product_mask_respects_segment_boundaries(self):
    sequence_length = 12
    query = jnp.zeros((1, sequence_length, 1, 8))
    key = jnp.zeros((1, sequence_length, 1, 8))
    segment_ids = jnp.asarray([[1] * 8 + [2] * 4], dtype=jnp.int32)

    mask = self._make_op(sequence_length).generate_attention_mask(
        query,
        key,
        segment_ids,
        MODEL_MODE_TRAIN,
    )

    positions = np.arange(sequence_length)
    expected = (positions[:, None] // 4 >= positions[None, :] // 4) & (
        np.asarray(segment_ids[0])[:, None] == np.asarray(segment_ids[0])[None, :]
    )
    np.testing.assert_array_equal(np.asarray(mask == 0.0)[0, 0, 0], expected)

  def test_dot_product_mask_uses_original_load_balanced_positions(self):
    sequence_length = 8
    positions = jnp.asarray([[0, 1, 6, 7, 2, 3, 4, 5]], dtype=jnp.int32)
    query = jnp.zeros((1, sequence_length, 1, 8))
    key = jnp.zeros((1, sequence_length, 1, 8))
    segment_ids = jnp.ones((1, sequence_length), dtype=jnp.int32)
    config = types.SimpleNamespace(
        causal_block_size=4,
        context_parallel_load_balance=True,
        context_sharding="context",
    )
    op = AttentionOp(
        config=config,
        num_query_heads=1,
        num_kv_heads=1,
        max_target_length=sequence_length,
        mesh=types.SimpleNamespace(shape={"context": 2}),
        attention_kernel="dot_product",
        attention_type=AttentionType.BLOCK_DIFFUSION,
    )

    mask = op.generate_attention_mask(
        query,
        key,
        segment_ids,
        MODEL_MODE_TRAIN,
        segment_positions=positions,
    )

    expected = np.asarray(positions[0])[:, None] // 4 >= np.asarray(positions[0])[None, :] // 4
    np.testing.assert_array_equal(np.asarray(mask == 0.0)[0, 0, 0], expected)

  def test_autoregressive_mask_is_unchanged(self):
    key_length = 8
    query = jnp.zeros((1, 1, 1, 8))
    key = jnp.zeros((1, key_length, 1, 8))
    segment_ids = jnp.asarray([[1, 1, 0, 1, 0, 0, 1, 0]], dtype=jnp.int32)

    default_mask = self._make_op(key_length, attention_type=None).generate_attention_mask(
        query,
        key,
        segment_ids,
        MODEL_MODE_AUTOREGRESSIVE,
    )
    block_diffusion_mask = self._make_op(key_length).generate_attention_mask(
        query,
        key,
        segment_ids,
        MODEL_MODE_AUTOREGRESSIVE,
    )

    np.testing.assert_array_equal(block_diffusion_mask, default_mask)
    np.testing.assert_array_equal(
        np.asarray(default_mask == 0.0)[0, 0, 0, 0],
        np.asarray(segment_ids[0] == DECODING_ACTIVE_SEQUENCE_INDICATOR),
    )

  def test_invalid_block_size_raises(self):
    with self.assertRaises(ValueError):
      BlockCausalMask((4, 4), 0)
    with self.assertRaises(ValueError):
      _generate_block_causal_attention_mask((4, 4), -1)

  def test_attention_op_rejects_invalid_block_diffusion_configuration(self):
    kwargs = {
        "num_query_heads": 1,
        "num_kv_heads": 1,
        "max_target_length": 8,
        "mesh": types.SimpleNamespace(shape={}),
        "attention_type": AttentionType.BLOCK_DIFFUSION,
    }

    with self.assertRaisesRegex(ValueError, "causal_block_size must be positive"):
      AttentionOp(
          config=types.SimpleNamespace(causal_block_size=0),
          attention_kernel="dot_product",
          **kwargs,
      )
    with self.assertRaisesRegex(ValueError, "supported only by dot_product attention"):
      AttentionOp(
          config=types.SimpleNamespace(causal_block_size=4),
          attention_kernel="paged",
          **kwargs,
      )

  def test_flash_attention_dispatch_on_non_tpu(self):
    query = jnp.zeros((1, 8, 1, 8))
    block_op = self._make_flash_op()

    with self.assertRaisesRegex(ValueError, "supported only by TPU Splash"):
      block_op.apply_attention(
          query,
          query,
          query,
          decoder_segment_ids=None,
          segment_positions=None,
          lengths=None,
          model_mode=MODEL_MODE_TRAIN,
          qk_product_einsum=jnp.einsum,
          wv_product_einsum=jnp.einsum,
      )

    global_op = self._make_flash_op(attention_type=AttentionType.GLOBAL)
    expected = (query, None, None)
    with mock.patch.object(global_op, "apply_attention_dot", return_value=expected) as apply_dot:
      actual = global_op.apply_attention(
          query[:, :1],
          query,
          query,
          decoder_segment_ids=None,
          segment_positions=None,
          lengths=None,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          qk_product_einsum=jnp.einsum,
          wv_product_einsum=jnp.einsum,
      )

    self.assertIs(actual, expected)
    apply_dot.assert_called_once()

  def test_tpu_splash_selects_block_causal_masks(self):
    query = jnp.zeros((1, 8, 1, 8))
    cases = (
        (AttentionType.BLOCK_DIFFUSION, 1, False, BlockCausalMask),
        (AttentionType.BLOCK_DIFFUSION, 2, True, attention_op.LoadBalancedBlockCausalMask),
        (AttentionType.GLOBAL, 2, True, attention_op.LoadBalancedCausalMask),
    )

    for attention_type, cp_size, load_balanced, expected_mask_type in cases:
      with self.subTest(attention_type=attention_type, cp_size=cp_size):
        op = self._make_flash_op(
            attention_type=attention_type,
            context_parallel_size=cp_size,
            context_parallel_load_balance=load_balanced,
        )
        with (
            mock.patch.object(AttentionOp, "_logical_to_mesh_axes", return_value=None),
            mock.patch.object(
                attention_op.splash_attention_mask,
                "MultiHeadMask",
                side_effect=RuntimeError("mask captured"),
            ) as make_multi_head_mask,
            self.assertRaisesRegex(RuntimeError, "mask captured"),
        ):
          op.tpu_flash_attention(query, query, query, decoder_segment_ids=None)

        selected_mask = make_multi_head_mask.call_args.kwargs["masks"][0]
        self.assertIsInstance(selected_mask, expected_mask_type)


class HCAStaticMaskTest(unittest.TestCase):
  """Tests the HCAStaticMask and its equivalence with the compressor mask."""

  def _make_flash_op(
      self,
      *,
      attention_type=AttentionType.COMPRESSED,
      context_parallel_size=1,
      context_parallel_load_balance=False,
      max_target_length=512,
  ):
    config = types.SimpleNamespace(
        causal_block_size=4,
        context_parallel_strategy="all_gather",
        context_parallel_load_balance=context_parallel_load_balance,
        context_sharding="context",
        sa_block_q=128,
        sa_block_kv=128,
        sa_block_kv_compute=128,
        sa_block_q_dkv=128,
        sa_block_kv_dkv=128,
        sa_block_kv_dkv_compute=128,
        sa_block_q_dq=128,
        sa_block_kv_dq=128,
        sa_use_fused_bwd_kernel=True,
        sa_q_layout="HEAD_DIM_MINOR",
        sa_k_layout="HEAD_DIM_MINOR",
        sa_v_layout="HEAD_DIM_MINOR",
        use_splash_scheduler=False,
        sa_fuse_reciprocal=False,
        sa_use_base2_exp=False,
        use_tokamax_splash=False,
        use_jax_splash=False,
        cost_estimate_flops_fwd=-1,
        cost_estimate_flops_bwd=-1,
        dq_reduction_steps=-1,
    )
    device = types.SimpleNamespace(platform="cpu")
    mesh = types.SimpleNamespace(
        devices=np.asarray([device] * context_parallel_size, dtype=object),
        shape={"context": context_parallel_size},
    )
    return AttentionOp(
        config=config,
        num_query_heads=1,
        num_kv_heads=1,
        max_target_length=max_target_length,
        mesh=mesh,
        attention_kernel="flash",
        attention_type=attention_type,
    )

  def test_hca_static_mask_matches_compressor_mask(self):
    test_cases = [
        (512, 128, 128, 0),
        (1024, 128, 128, 64),
        (512, 128, None, 0),  # full causal local attention
        (384, 128, 128, 128),
        (489, 128, 128, 23),  # unaligned sequence length 489
    ]

    for seq_len, compress_ratio, local_window, pad_kv in test_cases:
      with self.subTest(seq_len=seq_len, ratio=compress_ratio, window=local_window, pad=pad_kv):
        comp_len = max(0, seq_len // max(1, compress_ratio))
        total_kv_len = seq_len + pad_kv + comp_len
        shape = (seq_len, total_kv_len)

        mask = attention_op.HCAStaticMask(
            shape=shape,
            local_kv_len=seq_len,
            compressed_kv_len=comp_len,
            pad_kv_total=pad_kv,
            compress_ratio=compress_ratio,
            local_window=local_window,
        )

        op = AttentionOp(
            config=types.SimpleNamespace(
                causal_block_size=4,
                context_parallel_load_balance=False,
                context_sharding="context",
                moba=False,
            ),
            num_query_heads=1,
            num_kv_heads=1,
            max_target_length=seq_len,
            mesh=types.SimpleNamespace(shape={}),
            attention_kernel="dot_product",
            attention_type=AttentionType.COMPRESSED,
            sliding_window_size=local_window,
        )

        position_ids = jnp.arange(seq_len)[None, :]
        usable_len = comp_len * compress_ratio
        if comp_len > 0:
          block_positions = position_ids[:, :usable_len:compress_ratio]
          future_mask = (block_positions[:, None, None, :] + compress_ratio) > (position_ids[:, None, :, None] + 1)
          compressed_causal_mask = jnp.where(future_mask, DEFAULT_MASK_VALUE, 0.0)
        else:
          compressed_causal_mask = jnp.zeros((1, 1, seq_len, 0))

        dummy_q = jnp.zeros((1, seq_len, 1, 128))
        dummy_k = jnp.zeros((1, seq_len + comp_len, 1, 128))
        ref_mask = op.generate_attention_mask(
            dummy_q,
            dummy_k,
            decoder_segment_ids=None,
            model_mode=MODEL_MODE_TRAIN,
            compressed_mask=compressed_causal_mask,
            pad_kv_total=pad_kv,
        )
        expected_mask = np.array(ref_mask[0, 0] == 0.0)

        actual_mask = mask[:, :]
        np.testing.assert_array_equal(actual_mask, expected_mask)

  def test_hca_static_mask_equality_and_hash(self):
    mask1 = attention_op.HCAStaticMask(
        shape=(512, 640),
        local_kv_len=512,
        compressed_kv_len=4,
        pad_kv_total=124,
        compress_ratio=128,
        local_window=128,
    )
    mask2 = attention_op.HCAStaticMask(
        shape=(512, 640),
        local_kv_len=512,
        compressed_kv_len=4,
        pad_kv_total=124,
        compress_ratio=128,
        local_window=128,
    )
    mask_diff_pad = attention_op.HCAStaticMask(
        shape=(512, 640),
        local_kv_len=512,
        compressed_kv_len=4,
        pad_kv_total=0,
        compress_ratio=128,
        local_window=128,
    )

    self.assertEqual(mask1, mask2)
    self.assertEqual(hash(mask1), hash(mask2))
    self.assertNotEqual(mask1, mask_diff_pad)
    self.assertNotEqual(mask1, object())

  def test_zero_division_guard(self):
    mask = attention_op.HCAStaticMask(
        shape=(128, 128),
        local_kv_len=128,
        compressed_kv_len=1,
        compress_ratio=0,
        local_window=128,
    )
    res = mask[:, :]
    self.assertEqual(res.shape, (128, 128))

  def test_load_balanced_cp_raises_on_hca(self):
    op = self._make_flash_op(
        attention_type=AttentionType.COMPRESSED,
        context_parallel_size=2,
        context_parallel_load_balance=True,
        max_target_length=512,
    )
    query = jnp.zeros((1, 1, 512, 128))
    key = jnp.zeros((1, 1, 516, 128))
    with self.assertRaisesRegex(
        ValueError, "Load-balanced context parallelism is currently not supported for DeepSeek-V4 HCA"
    ):
      op.tpu_flash_attention(
          query,
          key,
          key,
          decoder_segment_ids=None,
          compress_ratio=128,
      )

  def test_compressed_flash_missing_or_invalid_compress_ratio_raises(self):
    op = self._make_flash_op(
        attention_type=AttentionType.COMPRESSED,
        max_target_length=512,
    )
    query = jnp.zeros((1, 1, 512, 128))
    key = jnp.zeros((1, 1, 516, 128))
    with self.assertRaisesRegex(ValueError, "compress_ratio must be provided for AttentionType.COMPRESSED"):
      op.tpu_flash_attention(query, key, key, decoder_segment_ids=None, compress_ratio=None)
    with self.assertRaisesRegex(ValueError, "compress_ratio must be provided for AttentionType.COMPRESSED"):
      op.tpu_flash_attention(query, key, key, decoder_segment_ids=None, compress_ratio=0)


class AttentionTypeResolutionTest(unittest.TestCase):

  def test_config_selects_block_diffusion_without_model_dispatch(self):
    config = types.SimpleNamespace(attention_type=AttentionType.BLOCK_DIFFUSION.value)

    self.assertEqual(_resolve_attention_type(config, None), AttentionType.BLOCK_DIFFUSION)
    self.assertEqual(_resolve_attention_type(config, AttentionType.GLOBAL), AttentionType.BLOCK_DIFFUSION)
    self.assertEqual(_resolve_attention_type(config, AttentionType.FULL), AttentionType.FULL)
    self.assertEqual(_resolve_attention_type(config, AttentionType.LOCAL_SLIDING), AttentionType.LOCAL_SLIDING)
    self.assertEqual(
        _resolve_attention_type(types.SimpleNamespace(attention_type=AttentionType.GLOBAL.value), AttentionType.FULL),
        AttentionType.FULL,
    )
    self.assertEqual(_resolve_attention_type(types.SimpleNamespace(), None), AttentionType.GLOBAL)

  def test_attention_op_honors_config_without_overriding_specialized_layers(self):
    config = types.SimpleNamespace(
        attention_type=AttentionType.BLOCK_DIFFUSION.value,
        causal_block_size=4,
    )
    op = AttentionOp(
        config=config,
        num_query_heads=1,
        num_kv_heads=1,
        max_target_length=8,
        mesh=types.SimpleNamespace(shape={}),
        attention_kernel="dot_product",
        attention_type=AttentionType.GLOBAL,
    )

    self.assertEqual(op.attention_type, AttentionType.BLOCK_DIFFUSION)

    full_op = AttentionOp(
        config=config,
        num_query_heads=1,
        num_kv_heads=1,
        max_target_length=8,
        mesh=types.SimpleNamespace(shape={}),
        attention_kernel="dot_product",
        attention_type=AttentionType.FULL,
    )

    self.assertEqual(full_op.attention_type, AttentionType.FULL)


class LoadBalancedMaskTest(unittest.TestCase):
  """Tests for load-balanced Splash masks."""

  def test_load_balanced_local_window(self):
    seq_len = 8
    window_size = 3
    q_sequence = np.asarray([0, 1, 6, 7, 2, 3, 4, 5])
    kv_sequence = np.arange(seq_len)
    causal_mask = attention_op.LoadBalancedCausalMask(shape=(seq_len, seq_len), cp_size=2)
    local_mask = attention_op.LoadBalancedLocalMask(
        shape=(seq_len, seq_len),
        window_size=(window_size - 1, window_size),
        offset=0,
        cp_size=2,
    )

    expected_mask = (kv_sequence[None, :] <= q_sequence[:, None]) & (
        kv_sequence[None, :] > q_sequence[:, None] - window_size
    )

    np.testing.assert_array_equal((causal_mask & local_mask)[:, :], expected_mask)

  def test_load_balanced_chunk_window(self):
    seq_len = 8
    chunk_size = 2
    q_sequence = np.asarray([0, 1, 6, 7, 2, 3, 4, 5])
    kv_sequence = np.arange(seq_len)
    causal_mask = attention_op.LoadBalancedCausalMask(shape=(seq_len, seq_len), cp_size=2)
    chunk_mask = attention_op.LoadBalancedChunkedCausalMask(
        shape=(seq_len, seq_len),
        chunk_size=chunk_size,
        cp_size=2,
    )

    expected_mask = (kv_sequence[None, :] <= q_sequence[:, None]) & (
        q_sequence[:, None] // chunk_size == kv_sequence[None, :] // chunk_size
    )

    np.testing.assert_array_equal((causal_mask & chunk_mask)[:, :], expected_mask)

  def test_load_balanced_block_causal_mask(self):
    sequence_length = 8
    block_size = 2
    mask = attention_op.LoadBalancedBlockCausalMask(
        shape=(sequence_length, sequence_length),
        causal_block_size=block_size,
        cp_size=2,
    )
    query_positions = mask.q_sequence[:, None]
    key_positions = np.arange(sequence_length)[None, :]
    expected = query_positions // block_size >= key_positions // block_size

    np.testing.assert_array_equal(mask[:, :], expected)

  def test_dot_product_local_mask_uses_segment_positions(self):
    config = types.SimpleNamespace(context_parallel_load_balance=True, context_sharding="context")
    mesh = types.SimpleNamespace(shape={"context": 4})
    seq_len = 16
    sliding_window_size = 4
    positions = jnp.asarray(attention_op.LoadBalancedCausalMask(shape=(seq_len, seq_len), cp_size=4).q_sequence[None, :])
    query = jnp.zeros((1, seq_len, 1, 128))
    key = jnp.zeros((1, seq_len, 1, 128))
    decoder_segment_ids = jnp.ones((1, seq_len), dtype=jnp.int32)
    op = AttentionOp(
        config=config,
        num_query_heads=1,
        num_kv_heads=1,
        max_target_length=seq_len,
        mesh=mesh,
        attention_kernel="dot_product",
        attention_type=AttentionType.LOCAL_SLIDING,
        sliding_window_size=sliding_window_size,
    )

    mask = op.generate_attention_mask(
        query,
        key,
        decoder_segment_ids,
        MODEL_MODE_TRAIN,
        segment_positions=positions,
    )

    expected_mask = np.zeros((seq_len, seq_len), dtype=np.bool_)
    for r, q_pos in enumerate(np.asarray(positions[0])):
      for c, kv_pos in enumerate(np.asarray(positions[0])):
        if q_pos - sliding_window_size < kv_pos <= q_pos:
          expected_mask[r, c] = True

    np.testing.assert_array_equal(np.asarray(mask == 0.0)[0, 0, 0], expected_mask)

  def test_dot_product_chunk_mask_uses_segment_positions(self):
    config = types.SimpleNamespace(context_parallel_load_balance=True, context_sharding="context")
    mesh = types.SimpleNamespace(shape={"context": 4})
    seq_len = 16
    chunk_size = 4
    positions = jnp.asarray(attention_op.LoadBalancedCausalMask(shape=(seq_len, seq_len), cp_size=4).q_sequence[None, :])
    query = jnp.zeros((1, seq_len, 1, 128))
    key = jnp.zeros((1, seq_len, 1, 128))
    decoder_segment_ids = jnp.ones((1, seq_len), dtype=jnp.int32)
    op = AttentionOp(
        config=config,
        num_query_heads=1,
        num_kv_heads=1,
        max_target_length=seq_len,
        mesh=mesh,
        attention_kernel="dot_product",
        attention_type=AttentionType.CHUNK,
        chunk_attn_window_size=chunk_size,
    )

    mask = op.generate_attention_mask(
        query,
        key,
        decoder_segment_ids,
        MODEL_MODE_TRAIN,
        segment_positions=positions,
    )

    expected_mask = np.zeros((seq_len, seq_len), dtype=np.bool_)
    for r, q_pos in enumerate(np.asarray(positions[0])):
      for c, kv_pos in enumerate(np.asarray(positions[0])):
        if q_pos >= kv_pos and q_pos // chunk_size == kv_pos // chunk_size:
          expected_mask[r, c] = True

    np.testing.assert_array_equal(np.asarray(mask == 0.0)[0, 0, 0], expected_mask)


class CudnnTePackedSequenceDescriptorTest(unittest.TestCase):
  """Tests packed Transformer Engine attention metadata handling."""

  def _call_te_attention(
      self, sequence_descriptor, config=None, mesh=None, attention_type=AttentionType.GLOBAL, chunk_attn_window_size=None
  ):
    """Runs TE attention with fake Transformer Engine modules."""
    sequence_descriptor.calls = []

    class FakeWrappedAttention:

      def lazy_init(self, *args, **kwargs):  # pylint: disable=unused-argument
        return self

      def __call__(self, *args, **kwargs):
        del args
        return kwargs["sequence_descriptor"]

    def fake_to_nnx(*args, **kwargs):  # pylint: disable=unused-argument
      return FakeWrappedAttention()

    transformer_module = types.ModuleType("transformer_engine.jax.flax.transformer")
    transformer_module.DotProductAttention = mock.Mock()
    attention_module = types.ModuleType("transformer_engine.jax.attention")
    attention_module.SequenceDescriptor = sequence_descriptor
    fake_modules = {
        "transformer_engine": types.ModuleType("transformer_engine"),
        "transformer_engine.jax": types.ModuleType("transformer_engine.jax"),
        "transformer_engine.jax.flax": types.ModuleType("transformer_engine.jax.flax"),
        "transformer_engine.jax.flax.transformer": transformer_module,
        "transformer_engine.jax.attention": attention_module,
    }

    if config is None:
      config = types.SimpleNamespace(
          context_sharding="context",
          context_parallel_strategy="ring",
          context_parallel_load_balance=False,
          packing=True,
          dataset_type="grain",
          max_segments_per_seq=4,
          head_dim=2,
          attention_kernel="cudnn_flash_te",
      )
    if mesh is None:
      mesh = types.SimpleNamespace(shape={"context": 1})
    attention = AttentionOp(
        config=config,
        mesh=mesh,
        attention_kernel="cudnn_flash_te",
        max_target_length=4,
        num_query_heads=2,
        num_kv_heads=2,
        dtype=jnp.float32,
        attention_type=attention_type,
        chunk_attn_window_size=chunk_attn_window_size,
    )
    query = jnp.zeros((1, 4, 2, 2), dtype=jnp.float32)
    key = jnp.zeros((1, 4, 2, 2), dtype=jnp.float32)
    value = jnp.zeros((1, 4, 2, 2), dtype=jnp.float32)
    segment_positions = jnp.arange(4, dtype=jnp.int32)[None, :]

    with (
        mock.patch.dict(sys.modules, fake_modules),
        mock.patch.object(attention_op.nnx_wrappers, "ToNNX", side_effect=fake_to_nnx),
    ):
      output = attention.cudnn_flash_attention(
          query=query,
          key=key,
          value=value,
          decoder_segment_ids=None,
          segment_positions=segment_positions,
      )

    return output, sequence_descriptor.calls

  def test_packed_attention_sequence_descriptor_uses_thd_metadata_with_legacy_fallback(self):
    class SequenceDescriptor:
      calls = []
      reject_thd_kwargs = False

      @classmethod
      def from_segment_ids_and_pos(cls, **kwargs):
        cls.calls.append(kwargs)
        if cls.reject_thd_kwargs and "is_thd" in kwargs:
          raise TypeError("older Transformer Engine does not accept THD metadata")
        return kwargs

    output, descriptor_calls = self._call_te_attention(SequenceDescriptor)

    self.assertEqual(len(descriptor_calls), 2)
    for call in descriptor_calls:
      self.assertTrue(call["is_thd"])
      self.assertFalse(call["is_segment_ids_reordered"])
    self.assertIs(output, descriptor_calls[0])

    SequenceDescriptor.reject_thd_kwargs = True
    output, descriptor_calls = self._call_te_attention(SequenceDescriptor)
    self.assertEqual(len(descriptor_calls), 4)
    self.assertIn("is_thd", descriptor_calls[0])
    self.assertNotIn("is_thd", descriptor_calls[1])
    self.assertIn("is_thd", descriptor_calls[2])
    self.assertNotIn("is_thd", descriptor_calls[3])
    self.assertIs(output, descriptor_calls[1])

  def test_context_parallel_chunk_attention_rejected(self):
    class SequenceDescriptor:
      pass

    config = types.SimpleNamespace(
        context_sharding="context",
        context_parallel_strategy="all_gather",
        context_parallel_load_balance=False,
        packing=False,
        dataset_type="synthetic",
        max_segments_per_seq=1,
        head_dim=2,
        attention_kernel="cudnn_flash_te",
    )
    mesh = types.SimpleNamespace(shape={"context": 2})
    with self.assertRaisesRegex(ValueError, "Chunk attention"):
      self._call_te_attention(
          SequenceDescriptor,
          config=config,
          mesh=mesh,
          attention_type=AttentionType.CHUNK,
          chunk_attn_window_size=2,
      )


class AttentionTest(parameterized.TestCase):
  """Test for the Attention"""

  # Note: if you are changing these configs, please make sure to change the configs in
  # context_parallelism.py as well, since we are using the same configs for both
  # tests to get the same mesh and other config
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
  }

  def setUp(self):
    """Initializes the configuration for each test"""
    super().setUp()
    if not is_decoupled():
      jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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

  def get_packed_data(self, dtype):
    """get packed data"""
    lnx, _, _ = self.get_data(dtype)
    # Uneven segment lengths so boundaries don't line up with splash blocks, reorder chunks, or the CP shard split.
    segment_lengths = (80, 240, 112, 80)
    segment_ids = jnp.concatenate(
        [jnp.full((length,), segment, dtype=jnp.int32) for segment, length in enumerate(segment_lengths, start=1)]
    )
    positions = jnp.concatenate([jnp.arange(length, dtype=jnp.int32) for length in segment_lengths])
    decoder_segment_ids = jnp.broadcast_to(segment_ids, (self.global_batch_size, self.max_target_length))
    decoder_positions = jnp.broadcast_to(positions, (self.global_batch_size, self.max_target_length))

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

    for idx in range(prefill_length, min(prefill_length + 3, decode_total_length)):
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

  @pytest.mark.tpu_only
  def test_tpu_kernel_attention_mha_share_kv(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads, share_kv_projections=True)

  @pytest.mark.tpu_only
  def test_tpu_kernel_attention_gqa_share_kv(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads // 2, share_kv_projections=True)

  def tpu_kernel_attention_helper(self, num_kv_heads, share_kv_projections=False):
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
        share_kv_projections=share_kv_projections,
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
        share_kv_projections=share_kv_projections,
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

  def test_share_kv_projections(self):
    """Test that kv projections are shared."""
    dummy_inputs_q = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    dummy_inputs_kv = jnp.ones((self.global_batch_size, self.max_target_length, self.embed_dim))
    attention_share_kv = Attention(
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
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        share_kv_projections=True,
        rngs=self.nnx_rng,
    )

    self.assertFalse(hasattr(attention_share_kv, "value"))
    self.assertTrue(hasattr(attention_share_kv, "key"))

    # 1. Check NNX state
    state_shared = nnx.state(attention_share_kv)
    self.assertNotIn("value", state_shared)
    self.assertIn("key", state_shared)

    # 2. Forward Pass Verification
    lnx, decoder_segment_ids, decoder_positions = self.get_data(self.dtype)

    output_shared, _ = attention_share_kv(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertEqual(output_shared.shape, (self.global_batch_size, self.max_target_length, self.embed_dim))

    # 3. Equivalence Check with standard unshared Attention
    attention_no_share = Attention(
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
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        share_kv_projections=False,
        rngs=self.nnx_rng,
    )

    # Force unshared layer to copy weights from shared layer, mapping 'key' to 'value'
    attention_no_share.query.kernel[...] = attention_share_kv.query.kernel[...]
    attention_no_share.key.kernel[...] = attention_share_kv.key.kernel[...]
    attention_no_share.value.kernel[...] = attention_share_kv.key.kernel[...]
    attention_no_share.out.kernel[...] = attention_share_kv.out.kernel[...]

    output_no_share, _ = attention_no_share(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertTrue(jax.numpy.allclose(output_shared, output_no_share, rtol=1e-04, atol=1e-04, equal_nan=False))

  @parameterized.named_parameters(
      {
          "testcase_name": "cp_no_load_balance",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 1,
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_with_load_balance",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 1,
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_ep_no_load_balance",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 2,
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_ep_with_load_balance",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 2,
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_no_load_balance_explicit",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 1,
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_with_load_balance_explicit",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 1,
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_ep_no_load_balance_explicit",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 2,
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_ep_with_load_balance_explicit",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 2,
          "shard_mode": "explicit",
      },
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_context_parallel(
      self,
      ici_context_parallelism,
      context_parallel_load_balance,
      ici_expert_parallelism,
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
        [sys.argv[0], get_test_config_path()],
        **self.config_arguments,
        ici_context_parallelism=ici_context_parallelism,
        context_parallel_load_balance=context_parallel_load_balance,
        ici_expert_parallelism=ici_expert_parallelism,
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

    mha_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mha_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    # This removes all sharding information and makes them standard NumPy arrays.
    mha_generic_output = jax.device_get(mha_generic_output)
    mha_generic_flash_cp_output = jax.device_get(mha_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_cp_output, rtol=1e-01, atol=1e-01, equal_nan=False),
        msg="Logits from generic dot product and flash attention + context/expert parallelism are not close.\n"
        f"ici_context_parallelism={ici_context_parallelism}, context_parallel_load_balance={context_parallel_load_balance},"
        f" ici_expert_parallelism={ici_expert_parallelism}.",
    )

  @parameterized.named_parameters(
      {"testcase_name": "no_load_balance", "context_parallel_load_balance": False},
      {"testcase_name": "load_balance", "context_parallel_load_balance": True},
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_packed_all_gather_context_parallel(self, context_parallel_load_balance):
    """Test equivalence between packed dot_product and packed flash attention + all-gather context parallelism."""
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=self.dtype,
    )
    tokens_per_segment = self.max_target_length // 4
    segment_ids = jnp.repeat(jnp.arange(1, 5, dtype=jnp.int32), tokens_per_segment)
    positions = jnp.tile(jnp.arange(tokens_per_segment, dtype=jnp.int32), 4)
    decoder_segment_ids = jnp.broadcast_to(segment_ids, (self.global_batch_size, self.max_target_length))
    decoder_positions = jnp.broadcast_to(positions, (self.global_batch_size, self.max_target_length))
    mha_generic_output, _ = self._attention_as_mha_generic(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    generic_state = nnx.state(self._attention_as_mha_generic)

    cfg_cp = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **self.config_arguments,
        ici_context_parallelism=4,
        context_parallel_strategy="all_gather",
        context_parallel_load_balance=context_parallel_load_balance,
        packing=True,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    attention_as_mha_flash_cp = Attention(
        config=cfg_cp,
        num_query_heads=cfg_cp.num_query_heads,
        num_kv_heads=cfg_cp.num_kv_heads,
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

    mha_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mha_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    self.assertTrue(
        jax.numpy.allclose(
            jax.device_get(mha_generic_output),
            jax.device_get(mha_generic_flash_cp_output),
            rtol=1e-01,
            atol=1e-01,
            equal_nan=False,
        ),
        msg="Logits from packed generic dot product and packed flash attention + all-gather context parallelism "
        f"are not close. context_parallel_load_balance={context_parallel_load_balance}.",
    )

  @parameterized.named_parameters(
      {"testcase_name": "no_load_balance", "context_parallel_load_balance": False, "packing": False},
      {"testcase_name": "load_balance", "context_parallel_load_balance": True, "packing": False},
      {"testcase_name": "packed", "context_parallel_load_balance": False, "packing": True},
      {"testcase_name": "packed_load_balance", "context_parallel_load_balance": True, "packing": True},
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_ring_context_parallel(self, context_parallel_load_balance, packing):
    """Test equivalence between dot_product and flash attention + ring context parallelism"""

    cfg_cp = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **self.config_arguments,
        attention="flash",
        context_parallel_strategy="ring",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=2,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=packing,
        dtype="float32",
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    if packing:
      lnx, decoder_segment_ids, decoder_positions = self.get_packed_data(cfg_cp.dtype)
    else:
      lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=cfg_cp.num_query_heads,
        num_kv_heads=cfg_cp.num_kv_heads,
        head_dim=cfg_cp.head_dim,
        max_target_length=cfg_cp.max_target_length,
        max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=cfg_cp.dtype,
        dropout_rate=cfg_cp.dropout_rate,
        rngs=self.nnx_rng,
    )
    mha_generic_output, _ = attention_as_mha_generic(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    generic_state = nnx.state(attention_as_mha_generic)

    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      attention_as_mha_flash_cp = Attention(
          config=cfg_cp,
          num_query_heads=cfg_cp.num_query_heads,
          num_kv_heads=cfg_cp.num_kv_heads,
          head_dim=cfg_cp.head_dim,
          max_target_length=cfg_cp.max_target_length,
          max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
          inputs_q_shape=lnx.shape,
          inputs_kv_shape=lnx.shape,
          mesh=mesh_cp,
          attention_kernel="flash",
          dtype=cfg_cp.dtype,
          dropout_rate=cfg_cp.dropout_rate,
          model_mode=MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    nnx.update(attention_as_mha_flash_cp, generic_state)

    mha_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mha_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    mha_generic_output = jax.device_get(mha_generic_output)
    mha_generic_flash_cp_output = jax.device_get(mha_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_cp_output, rtol=1e-02, atol=1e-02, equal_nan=False),
        msg="Logits from generic dot product and flash attention + ring context parallelism are not close. "
        f"context_parallel_load_balance={context_parallel_load_balance}, packing={packing}.",
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "no_load_balance",
          "context_parallel_load_balance": False,
          "max_target_length": 512,
          "dq_reduction_steps": 0,
          "ring_scan_unroll": 1,
          "packing": False,
      },
      {
          "testcase_name": "load_balance",
          "context_parallel_load_balance": True,
          "max_target_length": 512,
          "dq_reduction_steps": 0,
          "ring_scan_unroll": 1,
          "packing": False,
      },
      {
          "testcase_name": "load_balance_dq_reduction_unroll",
          "context_parallel_load_balance": True,
          "max_target_length": 1024,
          "dq_reduction_steps": 3,
          "ring_scan_unroll": 2,
          "packing": False,
      },
      {
          "testcase_name": "packed",
          "context_parallel_load_balance": False,
          "max_target_length": 512,
          "dq_reduction_steps": 0,
          "ring_scan_unroll": 1,
          "packing": True,
      },
      {
          "testcase_name": "packed_load_balance",
          "context_parallel_load_balance": True,
          "max_target_length": 512,
          "dq_reduction_steps": 0,
          "ring_scan_unroll": 1,
          "packing": True,
      },
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_ring_context_parallel_grad(
      self,
      context_parallel_load_balance,
      max_target_length,
      dq_reduction_steps,
      ring_scan_unroll,
      packing,
  ):
    """Test gradient equivalence between dot_product and flash attention + ring context parallelism"""

    cfg_cp = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **{**self.config_arguments, "max_target_length": max_target_length},
        attention="flash",
        context_parallel_strategy="ring",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=2,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=packing,
        dtype="float32",
        dq_reduction_steps=dq_reduction_steps,
        ring_scan_unroll=ring_scan_unroll,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    if packing:
      lnx, decoder_segment_ids, decoder_positions = self.get_packed_data(cfg_cp.dtype)
    else:
      lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=cfg_cp.num_query_heads,
        num_kv_heads=cfg_cp.num_kv_heads,
        head_dim=cfg_cp.head_dim,
        max_target_length=cfg_cp.max_target_length,
        max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=cfg_cp.dtype,
        dropout_rate=cfg_cp.dropout_rate,
        rngs=self.nnx_rng,
    )
    generic_state = nnx.state(attention_as_mha_generic)

    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      attention_as_mha_flash_cp = Attention(
          config=cfg_cp,
          num_query_heads=cfg_cp.num_query_heads,
          num_kv_heads=cfg_cp.num_kv_heads,
          head_dim=cfg_cp.head_dim,
          max_target_length=cfg_cp.max_target_length,
          max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
          inputs_q_shape=lnx.shape,
          inputs_kv_shape=lnx.shape,
          mesh=mesh_cp,
          attention_kernel="flash",
          dtype=cfg_cp.dtype,
          dropout_rate=cfg_cp.dropout_rate,
          model_mode=MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    nnx.update(attention_as_mha_flash_cp, generic_state)

    def generic_loss(lnx):
      output, _ = attention_as_mha_generic(
          lnx,
          lnx,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    def ring_loss(lnx):
      if context_parallel_load_balance:
        context_parallel_size = cfg_cp.ici_context_parallelism
        lnx = max_utils.reorder_sequence(lnx, cp_size=context_parallel_size)
        ring_decoder_segment_ids = max_utils.reorder_sequence(decoder_segment_ids, cp_size=context_parallel_size)
        ring_decoder_positions = max_utils.reorder_sequence(decoder_positions, cp_size=context_parallel_size)
      else:
        ring_decoder_segment_ids = decoder_segment_ids
        ring_decoder_positions = decoder_positions
      output, _ = attention_as_mha_flash_cp(
          lnx,
          lnx,
          decoder_segment_ids=ring_decoder_segment_ids,
          inputs_positions=ring_decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    generic_grad = jax.grad(generic_loss)(lnx)
    with jax.set_mesh(mesh_cp), nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      ring_grad = jax.grad(ring_loss)(lnx)
    generic_grad = jax.device_get(generic_grad)
    ring_grad = jax.device_get(ring_grad)

    self.assertTrue(
        jax.numpy.allclose(generic_grad, ring_grad, rtol=1e-02, atol=1e-07, equal_nan=False),
        msg="Input gradients from generic dot product and flash attention + ring context parallelism are not close. "
        f"context_parallel_load_balance={context_parallel_load_balance}, "
        f"dq_reduction_steps={dq_reduction_steps}, ring_scan_unroll={ring_scan_unroll}, packing={packing}.",
    )

  def _ulysses_test_config(self, ici_context_parallelism, packing=False):
    return pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **self.config_arguments,
        attention="flash",
        context_parallel_strategy="ulysses",
        context_parallel_load_balance=False,
        ici_context_parallelism=ici_context_parallelism,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=packing,
        dtype="float32",
    )

  def _ulysses_test_modules(self, cfg_cp, mesh_cp, lnx):
    """Builds the dot-product reference and the Ulysses flash attention modules."""
    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=cfg_cp.num_query_heads,
        num_kv_heads=cfg_cp.num_kv_heads,
        head_dim=cfg_cp.head_dim,
        max_target_length=cfg_cp.max_target_length,
        max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        mesh=self.mesh,
        attention_kernel="dot_product",
        dtype=cfg_cp.dtype,
        dropout_rate=cfg_cp.dropout_rate,
        rngs=self.nnx_rng,
    )
    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      attention_as_mha_flash_cp = Attention(
          config=cfg_cp,
          num_query_heads=cfg_cp.num_query_heads,
          num_kv_heads=cfg_cp.num_kv_heads,
          head_dim=cfg_cp.head_dim,
          max_target_length=cfg_cp.max_target_length,
          max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
          inputs_q_shape=lnx.shape,
          inputs_kv_shape=lnx.shape,
          mesh=mesh_cp,
          attention_kernel="flash",
          dtype=cfg_cp.dtype,
          dropout_rate=cfg_cp.dropout_rate,
          model_mode=MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    return attention_as_mha_generic, attention_as_mha_flash_cp

  @parameterized.named_parameters(
      {"testcase_name": "ulysses_size_2", "ici_context_parallelism": 2, "packing": False},
      {"testcase_name": "ulysses_size_4", "ici_context_parallelism": 4, "packing": False},
      {"testcase_name": "ulysses_size_4_packed", "ici_context_parallelism": 4, "packing": True},
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_ulysses_context_parallel(self, ici_context_parallelism, packing):
    """Test equivalence between dot_product and flash attention + Ulysses context parallelism"""

    cfg_cp = self._ulysses_test_config(ici_context_parallelism, packing=packing)
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    if packing:
      lnx, decoder_segment_ids, decoder_positions = self.get_packed_data(cfg_cp.dtype)
    else:
      lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    attention_as_mha_generic, attention_as_mha_flash_cp = self._ulysses_test_modules(cfg_cp, mesh_cp, lnx)
    mha_generic_output, _ = attention_as_mha_generic(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    nnx.update(attention_as_mha_flash_cp, nnx.state(attention_as_mha_generic))

    mha_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mha_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    mha_generic_output = jax.device_get(mha_generic_output)
    mha_generic_flash_cp_output = jax.device_get(mha_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_cp_output, rtol=1e-02, atol=1e-02, equal_nan=False),
        msg="Logits from generic dot product and flash attention + Ulysses context parallelism are not close. "
        f"ici_context_parallelism={ici_context_parallelism}, packing={packing}.",
    )

  @parameterized.named_parameters(
      {"testcase_name": "ulysses_size_2", "ici_context_parallelism": 2, "packing": False},
      {"testcase_name": "ulysses_size_4", "ici_context_parallelism": 4, "packing": False},
      {"testcase_name": "ulysses_size_4_packed", "ici_context_parallelism": 4, "packing": True},
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_ulysses_context_parallel_grad(self, ici_context_parallelism, packing):
    """Test input-gradient equivalence between dot_product and flash attention + Ulysses context parallelism"""

    cfg_cp = self._ulysses_test_config(ici_context_parallelism, packing=packing)
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    if packing:
      lnx, decoder_segment_ids, decoder_positions = self.get_packed_data(cfg_cp.dtype)
    else:
      lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    attention_as_mha_generic, attention_as_mha_flash_cp = self._ulysses_test_modules(cfg_cp, mesh_cp, lnx)
    nnx.update(attention_as_mha_flash_cp, nnx.state(attention_as_mha_generic))

    def generic_loss(lnx):
      output, _ = attention_as_mha_generic(
          lnx,
          lnx,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    def ulysses_loss(lnx):
      output, _ = attention_as_mha_flash_cp(
          lnx,
          lnx,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    generic_grad = jax.grad(generic_loss)(lnx)
    with jax.set_mesh(mesh_cp), nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      ulysses_grad = jax.grad(ulysses_loss)(lnx)
    generic_grad = jax.device_get(generic_grad)
    ulysses_grad = jax.device_get(ulysses_grad)

    self.assertTrue(
        jax.numpy.allclose(generic_grad, ulysses_grad, rtol=1e-02, atol=1e-07, equal_nan=False),
        msg="Input gradients from generic dot product and flash attention + Ulysses context parallelism are not "
        f"close. ici_context_parallelism={ici_context_parallelism}, packing={packing}.",
    )

  @pytest.mark.tpu_only
  def test_tpu_flash_attention_ulysses_hlo_uses_all_to_all(self):
    """Checks compiled TPU Ulysses attention HLO uses all-to-all collectives."""

    cfg_cp = self._ulysses_test_config(4)
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    _, attention_as_mha_flash_cp = self._ulysses_test_modules(cfg_cp, mesh_cp, lnx)

    def attention_forward(x, pos, seg):
      output, _ = attention_as_mha_flash_cp(
          x,
          x,
          decoder_segment_ids=seg,
          inputs_positions=pos,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return output

    def attention_loss(x, pos, seg):
      return jnp.sum(attention_forward(x, pos, seg).astype(jnp.float32))

    hlo_texts = []
    for lowered_fn in (attention_forward, jax.grad(attention_loss)):
      # The mesh and axis-rules contexts wrap the jit from outside because
      # jax.set_mesh raises inside a traced function, and the output keeps its
      # natural sequence sharding so the only full-sequence gathers in the
      # program are the ones the attention path itself emits.
      with jax.set_mesh(mesh_cp), nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
        input_sharding = NamedSharding(
            mesh_cp,
            nn_partitioning.logical_to_mesh_axes(
                ("activation_batch", "activation_length", "activation_embed"), nn_partitioning.get_axis_rules()
            ),
        )
        metadata_sharding = NamedSharding(
            mesh_cp, nn_partitioning.logical_to_mesh_axes((None, "activation_length"), nn_partitioning.get_axis_rules())
        )
        lowered = jax.jit(lowered_fn).lower(
            jax.device_put(lnx, input_sharding),
            jax.device_put(decoder_positions, metadata_sharding),
            jax.device_put(decoder_segment_ids, metadata_sharding),
        )
        hlo_texts.append(lowered.compile().as_text())

    sequence_lengths = (cfg_cp.max_target_length,)
    for hlo_text in hlo_texts:
      self.assertGreater(len(hlo_test_utils.collective_lines(hlo_text, "all-to-all")), 0)
      self.assertLen(hlo_test_utils.attention_sequence_all_gather_lines(hlo_text, sequence_lengths), 0)
      # The int32 segment-ID gathers are the only intended full-sequence gathers.
      self.assertGreater(
          len(hlo_test_utils.attention_sequence_all_gather_lines(hlo_text, sequence_lengths, dtypes=("s32",))), 0
      )
      self.assertLen(hlo_test_utils.collective_lines(hlo_text, "collective-permute"), 0)

  def _usp_test_config(self, packing=False, context_parallel_load_balance=False):
    return pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **self.config_arguments,
        attention="flash",
        context_parallel_strategy="usp",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=2,
        ici_context_usp_ulysses_parallelism=2,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=packing,
        dtype="float32",
    )

  @parameterized.named_parameters(
      {"testcase_name": "usp_2x2", "context_parallel_load_balance": False, "packing": False},
      {"testcase_name": "usp_2x2_load_balance", "context_parallel_load_balance": True, "packing": False},
      {"testcase_name": "usp_2x2_packed", "context_parallel_load_balance": False, "packing": True},
      {"testcase_name": "usp_2x2_packed_load_balance", "context_parallel_load_balance": True, "packing": True},
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_usp_context_parallel(self, context_parallel_load_balance, packing):
    """Test equivalence between dot_product and flash attention + USP context parallelism"""

    cfg_cp = self._usp_test_config(packing=packing, context_parallel_load_balance=context_parallel_load_balance)
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    if packing:
      lnx, decoder_segment_ids, decoder_positions = self.get_packed_data(cfg_cp.dtype)
    else:
      lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    attention_as_mha_generic, attention_as_mha_flash_cp = self._ulysses_test_modules(cfg_cp, mesh_cp, lnx)
    mha_generic_output, _ = attention_as_mha_generic(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    nnx.update(attention_as_mha_flash_cp, nnx.state(attention_as_mha_generic))

    mha_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mha_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    mha_generic_output = jax.device_get(mha_generic_output)
    mha_generic_flash_cp_output = jax.device_get(mha_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mha_generic_output, mha_generic_flash_cp_output, rtol=1e-02, atol=1e-02, equal_nan=False),
        msg="Logits from generic dot product and flash attention + USP context parallelism are not close. "
        f"context_parallel_load_balance={context_parallel_load_balance}, packing={packing}.",
    )

  @parameterized.named_parameters(
      {"testcase_name": "usp_2x2", "context_parallel_load_balance": False, "packing": False},
      {"testcase_name": "usp_2x2_load_balance", "context_parallel_load_balance": True, "packing": False},
      {"testcase_name": "usp_2x2_packed", "context_parallel_load_balance": False, "packing": True},
      {"testcase_name": "usp_2x2_packed_load_balance", "context_parallel_load_balance": True, "packing": True},
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_usp_context_parallel_grad(self, context_parallel_load_balance, packing):
    """Test input-gradient equivalence between dot_product and flash attention + USP context parallelism"""

    cfg_cp = self._usp_test_config(packing=packing, context_parallel_load_balance=context_parallel_load_balance)
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    if packing:
      lnx, decoder_segment_ids, decoder_positions = self.get_packed_data(cfg_cp.dtype)
    else:
      lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    attention_as_mha_generic, attention_as_mha_flash_cp = self._ulysses_test_modules(cfg_cp, mesh_cp, lnx)
    nnx.update(attention_as_mha_flash_cp, nnx.state(attention_as_mha_generic))

    def generic_loss(lnx):
      output, _ = attention_as_mha_generic(
          lnx,
          lnx,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    def usp_loss(lnx):
      if context_parallel_load_balance:
        context_parallel_size = cfg_cp.ici_context_parallelism
        lnx = max_utils.reorder_sequence(lnx, cp_size=context_parallel_size)
        usp_decoder_segment_ids = max_utils.reorder_sequence(decoder_segment_ids, cp_size=context_parallel_size)
        usp_decoder_positions = max_utils.reorder_sequence(decoder_positions, cp_size=context_parallel_size)
      else:
        usp_decoder_segment_ids = decoder_segment_ids
        usp_decoder_positions = decoder_positions
      output, _ = attention_as_mha_flash_cp(
          lnx,
          lnx,
          decoder_segment_ids=usp_decoder_segment_ids,
          inputs_positions=usp_decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    generic_grad = jax.grad(generic_loss)(lnx)
    with jax.set_mesh(mesh_cp), nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      usp_grad = jax.grad(usp_loss)(lnx)
    generic_grad = jax.device_get(generic_grad)
    usp_grad = jax.device_get(usp_grad)

    self.assertTrue(
        jax.numpy.allclose(generic_grad, usp_grad, rtol=1e-02, atol=1e-07, equal_nan=False),
        msg="Input gradients from generic dot product and flash attention + USP context parallelism are not close. "
        f"context_parallel_load_balance={context_parallel_load_balance}, packing={packing}.",
    )

  @pytest.mark.tpu_only
  def test_tpu_flash_attention_usp_hlo_uses_all_to_all_and_permute(self):
    """Checks compiled TPU USP attention HLO uses all-to-all and collective-permute."""

    cfg_cp = self._usp_test_config()
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg_cp.dtype)
    _, attention_as_mha_flash_cp = self._ulysses_test_modules(cfg_cp, mesh_cp, lnx)

    def attention_forward(x, pos, seg):
      output, _ = attention_as_mha_flash_cp(
          x,
          x,
          decoder_segment_ids=seg,
          inputs_positions=pos,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return output

    def attention_loss(x, pos, seg):
      return jnp.sum(attention_forward(x, pos, seg).astype(jnp.float32))

    hlo_texts = []
    for lowered_fn in (attention_forward, jax.grad(attention_loss)):
      # The mesh and axis-rules contexts wrap the jit from outside because
      # jax.set_mesh raises inside a traced function, and the output keeps its
      # natural sequence sharding so the only full-sequence gathers in the
      # program are the ones the attention path itself emits.
      with jax.set_mesh(mesh_cp), nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
        input_sharding = NamedSharding(
            mesh_cp,
            nn_partitioning.logical_to_mesh_axes(
                ("activation_batch", "activation_length", "activation_embed"), nn_partitioning.get_axis_rules()
            ),
        )
        metadata_sharding = NamedSharding(
            mesh_cp, nn_partitioning.logical_to_mesh_axes((None, "activation_length"), nn_partitioning.get_axis_rules())
        )
        lowered = jax.jit(lowered_fn).lower(
            jax.device_put(lnx, input_sharding),
            jax.device_put(decoder_positions, metadata_sharding),
            jax.device_put(decoder_segment_ids, metadata_sharding),
        )
        hlo_texts.append(lowered.compile().as_text())

    full_sequence_length = cfg_cp.max_target_length
    ring_local_sequence_length = full_sequence_length // cfg_cp.ici_context_parallelism
    # The gradient program legitimately all-gathers the shared input's gradient
    # over the Ulysses axis, so the ring-local length is only checked in the
    # forward program.
    sequence_lengths_per_program = ((full_sequence_length, ring_local_sequence_length), (full_sequence_length,))
    for hlo_text, sequence_lengths in zip(hlo_texts, sequence_lengths_per_program):
      self.assertGreater(len(hlo_test_utils.collective_lines(hlo_text, "all-to-all")), 0)
      self.assertGreater(len(hlo_test_utils.collective_lines(hlo_text, "collective-permute")), 0)
      self.assertLen(hlo_test_utils.attention_sequence_all_gather_lines(hlo_text, sequence_lengths), 0)
      # The int32 segment-ID gather over the Ulysses axis spans one ring-local
      # sequence block; it is the only intended sequence all-gather.
      self.assertGreater(
          len(
              hlo_test_utils.attention_sequence_all_gather_lines(hlo_text, (ring_local_sequence_length,), dtypes=("s32",))
          ),
          0,
      )
      self.assertLen(
          hlo_test_utils.attention_sequence_all_gather_lines(hlo_text, (full_sequence_length,), dtypes=("s32",)), 0
      )

  @pytest.mark.tpu_only
  def test_dot_product_cache_axis_order(self):
    all_axis_orders = tuple(itertools.permutations(range(4)))
    for axis_order in random.choices(all_axis_orders, k=2):
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
        [sys.argv[0], get_test_config_path()],
        **{**self.config_arguments, "attention": "dot_product"},
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
        jax.numpy.allclose(
            attention_w_layout_full[:, :prefill_length, :],
            attention_w_layout_prefill,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
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
        [sys.argv[0], get_test_config_path()],
        **{**self.config_arguments, "attention": "dot_product"},
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
            attention_wo_reshape_q_full[:, :prefill_length, :],
            attention_wo_reshape_q_prefill,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
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
            attention_w_reshape_q_full[:, :prefill_length, :],
            attention_w_reshape_q_prefill,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
    )

    self.assertTrue(
        jax.numpy.allclose(
            attention_wo_reshape_q_prefill,
            attention_w_reshape_q_prefill,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
    )
    self.assertTrue(
        jax.numpy.allclose(
            attention_wo_reshape_q_full[:, :prefill_length, :],
            attention_w_reshape_q_full[:, :prefill_length, :],
            rtol=rtol,
            atol=atol,
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
  @mock.patch("tpu_inference.layers.common.attention_interface.sharded_ragged_paged_attention", create=True)
  def test_forward_serve_vllm(self, mock_sharded_ragged_paged_attention):
    """Tests the forward_serve_vllm method with mocked RPA attention."""
    # Setup config for vLLM RPA
    vllm_config_arguments = self.config_arguments.copy()
    vllm_config_arguments["attention"] = "vllm_rpa"
    vllm_config_arguments["chunk_attn_window_size"] = 128
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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

    mock_sharded_ragged_paged_attention.return_value = (mock_output, mock_updated_kv_cache)

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
    self.assertEqual(updated_kv_cache, mock_updated_kv_cache)

    # The output of forward_serve_vllm is reshaped back to (batch, seq, ...)
    reshaped_mock_output = mock_output.reshape(self.global_batch_size, seq_len, self.num_query_heads, self.head_dim)
    expected_output = attention_vllm.out_projection(reshaped_mock_output)
    self.assertTrue(jnp.allclose(output, expected_output))
    self.assertEqual(output.shape, (self.global_batch_size, seq_len, self.embed_dim))

  @pytest.mark.skip(reason="Requires `vllm-tpu` package which is not yet a MaxText dependency.")
  @pytest.mark.tpu_only
  @mock.patch("tpu_inference.layers.common.attention_interface.sharded_ragged_paged_attention", create=True)
  def test_forward_serve_vllm_batched_rpa(self, mock_sharded_ragged_paged_attention):
    """Tests the forward_serve_vllm method with mocked batched RPA attention."""
    # Setup config for vLLM Batched RPA
    vllm_config_arguments = self.config_arguments.copy()
    vllm_config_arguments["attention"] = "vllm_batched_rpa"
    vllm_config_arguments["chunk_attn_window_size"] = 128
    config = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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

    mock_sharded_ragged_paged_attention.return_value = (mock_output, mock_updated_kv_cache)

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
    self.assertEqual(updated_kv_cache, mock_updated_kv_cache)
    self.assertEqual(os.environ.get("USE_BATCHED_RPA_KERNEL"), "1")

    # The output of forward_serve_vllm is reshaped back to (batch, seq, ...)
    reshaped_mock_output = mock_output.reshape(self.global_batch_size, seq_len, self.num_query_heads, self.head_dim)
    expected_output = attention_vllm.out_projection(reshaped_mock_output)
    self.assertTrue(jnp.allclose(output, expected_output))
    self.assertEqual(output.shape, (self.global_batch_size, seq_len, self.embed_dim))


class MLATest(attention_test_util.MLATestBase):
  """Test for the Multi-Headed Latent Attention"""

  config_arguments = {
      "per_device_batch_size": 1.0,
      "run_name": "test",
      "enable_checkpointing": False,
      "max_target_length": 32,
      "max_prefill_predict_length": 16,
      "attention_type": AttentionType.MLA.value,
      "head_dim": 32,
      "q_lora_rank": 4,
      "kv_lora_rank": 8,
      "qk_nope_head_dim": 16,
      "qk_rope_head_dim": 8,
      "v_head_dim": 32,
      "dtype": "float32",
      "mla_naive_kvcache": False,
  }

  @parameterized.named_parameters(
      {"testcase_name": "RoPE_Yarn_Autoregression", "rope_type": "yarn"},
      {"testcase_name": "Default_Autoregression", "rope_type": "default"},
  )
  @pytest.mark.tpu_only
  def test_mla_autoregression(self, rope_type):
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
      self.assertTrue(jax.numpy.allclose(mla_full_this_idx, mla_idx, rtol=2e-02, atol=2e-02, equal_nan=False))

  @parameterized.named_parameters(
      {"testcase_name": "prefill_less_than_topk", "prefill_len": 4, "target_len": 12},
      {"testcase_name": "prefill_greater_than_topk", "prefill_len": 12, "target_len": 16},
  )
  @pytest.mark.tpu_only
  def test_indexer_autoregression(self, prefill_len, target_len):
    config_arguments = self.config_arguments.copy()
    config_arguments.update(
        {
            "use_indexer": True,
            "indexer_n_heads": 4,
            "indexer_head_dim": 64,
            "indexer_topk": 8,
            "attention": "dot_product",
            "max_target_length": target_len,
            "max_prefill_predict_length": prefill_len,
            "per_device_batch_size": 1,
        }
    )
    cfg, mla = self.init_mla(config_arguments, "yarn")
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
      self.assertTrue(jax.numpy.allclose(mla_full_this_idx, mla_idx, rtol=2e-02, atol=2e-02, equal_nan=False))

  def test_sliced_mla_projections(self):
    config_arguments = self.config_arguments.copy()

    # Enable sliced projections for one config
    config_arguments_sliced = config_arguments.copy()
    config_arguments_sliced["use_sliced_mla_proj"] = True

    cfg_normal, mla_normal = self.init_mla(config_arguments, rope_type="default")
    _, mla_sliced = self.init_mla(config_arguments_sliced, rope_type="default")

    # Sync weights
    nnx.update(mla_sliced, nnx.state(mla_normal))

    # Test TRAIN mode with gradient comparison
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg_normal, cfg_normal.dtype)

    def loss_fn(model, x):
      out, _ = model(
          x,
          x,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(out.astype(jnp.float32) ** 2), out

    (loss_normal, out_normal_train), (grad_model_normal, grad_x_normal) = nnx.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(mla_normal, lnx)

    (loss_sliced, out_sliced_train), (grad_model_sliced, grad_x_sliced) = nnx.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
    )(mla_sliced, lnx)

    self.assertTrue(jnp.allclose(loss_normal, loss_sliced, rtol=1e-05, atol=1e-05, equal_nan=False))
    self.assertTrue(jnp.allclose(out_normal_train, out_sliced_train, rtol=1e-05, atol=1e-05, equal_nan=False))
    self.assertTrue(jnp.allclose(grad_x_normal, grad_x_sliced, rtol=1e-05, atol=1e-05, equal_nan=False))

    grad_model_close = jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y, rtol=1e-05, atol=1e-05, equal_nan=False),
        grad_model_normal,
        grad_model_sliced,
    )
    self.assertTrue(jax.tree_util.tree_all(grad_model_close))

    # Test PREFILL mode followed by AUTOREGRESSIVE mode to test caching
    prefill_length = cfg_normal.max_prefill_predict_length
    decode_total_length = cfg_normal.max_target_length

    # Re-initialize to ensure clean cache
    cfg_normal, mla_normal = self.init_mla(config_arguments, rope_type="default")
    _, mla_sliced = self.init_mla(config_arguments_sliced, rope_type="default")
    nnx.update(mla_sliced, nnx.state(mla_normal))

    lnx_prefill = lnx[:, 0:prefill_length, :]
    decoder_segment_ids_prefill = decoder_segment_ids[:, 0:prefill_length]
    decoder_positions_prefill = decoder_positions[:, 0:prefill_length]

    out_normal_prefill, _ = mla_normal(
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )

    out_sliced_prefill, _ = mla_sliced(
        lnx_prefill,
        lnx_prefill,
        decoder_segment_ids=decoder_segment_ids_prefill,
        inputs_positions=decoder_positions_prefill,
        deterministic=True,
        model_mode=MODEL_MODE_PREFILL,
    )

    self.assertTrue(jnp.allclose(out_normal_prefill, out_sliced_prefill, rtol=1e-05, atol=1e-05, equal_nan=False))

    # Run autoregressive steps
    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]
      decoder_positions_idx = decoder_positions[:, idx : idx + 1]

      out_normal_idx, _ = mla_normal(
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

      out_sliced_idx, _ = mla_sliced(
          lnx_idx,
          lnx_idx,
          inputs_positions=decoder_positions_idx,
          deterministic=True,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

      self.assertTrue(jnp.allclose(out_normal_idx, out_sliced_idx, rtol=1e-05, atol=1e-05, equal_nan=False))

  def test_projection_initialization(self):
    """Tests that MLA and Attention layers initialize the correct projection weights."""
    # 1. Initialize a standard Attention layer for comparison
    # Create a copy of the arguments and override the attention_type for the base model
    attention_config_args = self.config_arguments.copy()
    attention_config_args["attention_type"] = AttentionType.GLOBAL.value
    attention_cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
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
    mla_config_args = self.config_arguments.copy()
    _, mla_layer = self.init_mla(mla_config_args, rope_type="default")

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
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_with_load_balance",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 1,
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_ep_no_load_balance",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 2,
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_ep_with_load_balance",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 2,
          "shard_mode": "auto",
      },
      {
          "testcase_name": "cp_no_load_balance_explicit",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 1,
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_with_load_balance_explicit",
          "ici_context_parallelism": 4,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 1,
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_ep_no_load_balance_explicit",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": False,
          "ici_expert_parallelism": 2,
          "shard_mode": "explicit",
      },
      {
          "testcase_name": "cp_ep_with_load_balance_explicit",
          "ici_context_parallelism": 2,
          "context_parallel_load_balance": True,
          "ici_expert_parallelism": 2,
          "shard_mode": "explicit",
      },
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_context_parallel(
      self,
      ici_context_parallelism,
      context_parallel_load_balance,
      ici_expert_parallelism,
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
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
        rope_type=cfg.rope_type,
        ici_context_parallelism=ici_context_parallelism,
        context_parallel_load_balance=context_parallel_load_balance,
        ici_expert_parallelism=ici_expert_parallelism,
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
    mla_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mla_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    # This removes all sharding information and makes them standard NumPy arrays.
    mla_generic_output = jax.device_get(mla_generic_output)
    mla_generic_flash_cp_output = jax.device_get(mla_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mla_generic_output, mla_generic_flash_cp_output, rtol=1e-01, atol=1e-01, equal_nan=False),
        msg="MLA Logits from generic dot product and flash attention + context/expert parallelism are not close.\n"
        f"ici_context_parallelism={ici_context_parallelism}, context_parallel_load_balance={context_parallel_load_balance},"
        f" ici_expert_parallelism={ici_expert_parallelism}.",
    )

  @parameterized.named_parameters(
      {"testcase_name": "no_load_balance", "context_parallel_load_balance": False},
      {"testcase_name": "load_balance", "context_parallel_load_balance": True},
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_ring_context_parallel(self, context_parallel_load_balance):
    """Test equivalence between dot_product and flash attention + ring context parallelism"""

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
        "attention_type": AttentionType.MLA.value,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "dtype": "float32",
    }

    cfg, mla = self.init_mla(config_arguments, rope_type="default")
    lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg, cfg.dtype)
    mla_generic_output, _ = mla(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    generic_state = nnx.state(mla)

    cfg_cp = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
        attention="flash",
        rope_type=cfg.rope_type,
        context_parallel_strategy="ring",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=2,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=False,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
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
          attention_type=AttentionType(cfg_cp.attention_type),
          q_lora_rank=cfg_cp.q_lora_rank,
          kv_lora_rank=cfg_cp.kv_lora_rank,
          qk_nope_head_dim=cfg_cp.qk_nope_head_dim,
          qk_rope_head_dim=cfg_cp.qk_rope_head_dim,
          v_head_dim=cfg_cp.v_head_dim,
          model_mode=MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    nnx.update(attention_as_mla_flash_cp, generic_state)

    mla_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mla_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    mla_generic_output = jax.device_get(mla_generic_output)
    mla_generic_flash_cp_output = jax.device_get(mla_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mla_generic_output, mla_generic_flash_cp_output, rtol=1e-02, atol=1e-02, equal_nan=False),
        msg="MLA logits from generic dot product and flash attention + ring context parallelism are not close. "
        f"context_parallel_load_balance={context_parallel_load_balance}.",
    )

  @parameterized.named_parameters(
      {"testcase_name": "no_load_balance", "context_parallel_load_balance": False},
      {"testcase_name": "load_balance", "context_parallel_load_balance": True},
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_ring_context_parallel_grad(self, context_parallel_load_balance):
    """Test gradient equivalence between dot_product and flash attention + ring context parallelism"""

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
        "attention_type": AttentionType.MLA.value,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "dtype": "float32",
    }

    cfg, mla = self.init_mla(config_arguments, rope_type="default")
    lnx, decoder_segment_ids, decoder_positions = self.get_data(cfg, cfg.dtype)

    cfg_cp = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
        attention="flash",
        rope_type=cfg.rope_type,
        context_parallel_strategy="ring",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=2,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=False,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
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
          attention_type=AttentionType(cfg_cp.attention_type),
          q_lora_rank=cfg_cp.q_lora_rank,
          kv_lora_rank=cfg_cp.kv_lora_rank,
          qk_nope_head_dim=cfg_cp.qk_nope_head_dim,
          qk_rope_head_dim=cfg_cp.qk_rope_head_dim,
          v_head_dim=cfg_cp.v_head_dim,
          model_mode=MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    nnx.update(attention_as_mla_flash_cp, nnx.state(mla))
    generic_graphdef, generic_state = nnx.split(mla)
    ring_graphdef, ring_state = nnx.split(attention_as_mla_flash_cp)

    def generic_loss(lnx):
      mla_merged = nnx.merge(generic_graphdef, generic_state)
      output, _ = mla_merged(
          lnx,
          lnx,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    def ring_loss(lnx):
      if context_parallel_load_balance:
        context_parallel_size = cfg_cp.ici_context_parallelism
        lnx = max_utils.reorder_sequence(lnx, cp_size=context_parallel_size)
        ring_decoder_segment_ids = max_utils.reorder_sequence(decoder_segment_ids, cp_size=context_parallel_size)
        ring_decoder_positions = max_utils.reorder_sequence(decoder_positions, cp_size=context_parallel_size)
      else:
        ring_decoder_segment_ids = decoder_segment_ids
        ring_decoder_positions = decoder_positions
      ring_merged = nnx.merge(ring_graphdef, ring_state)
      output, _ = ring_merged(
          lnx,
          lnx,
          decoder_segment_ids=ring_decoder_segment_ids,
          inputs_positions=ring_decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    generic_grad = jax.grad(generic_loss)(lnx)
    with jax.set_mesh(mesh_cp), nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      ring_grad = jax.grad(ring_loss)(lnx)
    generic_grad = jax.device_get(generic_grad)
    ring_grad = jax.device_get(ring_grad)

    self.assertTrue(
        jax.numpy.allclose(generic_grad, ring_grad, rtol=1e-02, atol=1e-06, equal_nan=False),
        msg="MLA input gradients from generic dot product and flash attention + ring context parallelism are not close. "
        f"context_parallel_load_balance={context_parallel_load_balance}.",
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "no_lb_cp2",
          "context_parallel_load_balance": False,
          "ici_context_parallelism": 2,
          "indexer_topk": 256,
      },
      {
          "testcase_name": "lb_cp4_smallk",
          "context_parallel_load_balance": True,
          "ici_context_parallelism": 4,
          "indexer_topk": 32,
      },
  )
  @pytest.mark.tpu_only
  def test_tpu_dot_product_context_parallel_with_indexer(
      self, context_parallel_load_balance, ici_context_parallelism=2, indexer_topk=256
  ):
    """Test equivalence between single-device dot_product MLA + Indexer and multi-device dot_product + CP + Indexer"""
    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_target_length": 512,
        "attention_type": AttentionType.MLA.value,
        "use_indexer": True,
        "indexer_loss_scaling_factor": 0.0,
        "indexer_topk": indexer_topk,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "dtype": "float32",
    }

    cfg, mla = self.init_mla({**config_arguments, "attention": "dot_product"}, rope_type="default")
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg, cfg.dtype)
    mla_generic_output, _ = mla(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    generic_state = nnx.state(mla)

    cfg_cp = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
        attention="dot_product",
        rope_type=cfg.rope_type,
        context_parallel_strategy="all_gather",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=ici_context_parallelism,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      attention_as_mla_cp = MLA(
          config=cfg_cp,
          num_query_heads=cfg_cp.num_query_heads,
          num_kv_heads=cfg_cp.num_kv_heads,
          head_dim=cfg_cp.head_dim,
          inputs_q_shape=lnx.shape,
          inputs_kv_shape=lnx.shape,
          max_target_length=cfg_cp.max_target_length,
          max_prefill_predict_length=cfg_cp.max_prefill_predict_length,
          mesh=mesh_cp,
          attention_kernel="dot_product",
          dtype=cfg_cp.dtype,
          dropout_rate=cfg_cp.dropout_rate,
          attention_type=AttentionType(cfg_cp.attention_type),
          q_lora_rank=cfg_cp.q_lora_rank,
          kv_lora_rank=cfg_cp.kv_lora_rank,
          qk_nope_head_dim=cfg_cp.qk_nope_head_dim,
          qk_rope_head_dim=cfg_cp.qk_rope_head_dim,
          v_head_dim=cfg_cp.v_head_dim,
          model_mode=MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    nnx.update(attention_as_mla_cp, generic_state)

    mla_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mla_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    mla_generic_output = jax.device_get(mla_generic_output)
    mla_cp_output = jax.device_get(mla_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mla_generic_output, mla_cp_output, rtol=1e-02, atol=1e-02, equal_nan=False),
        msg=(
            "MLA+Indexer logits from single-device dot product and multi-device dot product context parallelism are"
            f" not close. context_parallel_load_balance={context_parallel_load_balance}."
        ),
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "no_lb_cp2",
          "context_parallel_load_balance": False,
          "ici_context_parallelism": 2,
          "indexer_topk": 256,
      },
      {
          "testcase_name": "lb_cp4_smallk",
          "context_parallel_load_balance": True,
          "ici_context_parallelism": 4,
          "indexer_topk": 32,
      },
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_context_parallel_with_indexer(
      self, context_parallel_load_balance, ici_context_parallelism=2, indexer_topk=256
  ):
    """Test equivalence between dot_product MLA + Indexer and all-gather flash attention + context parallelism + Indexer"""
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
        "attention_type": AttentionType.MLA.value,
        "use_indexer": True,
        "indexer_loss_scaling_factor": 0.1,
        "indexer_topk": indexer_topk,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "dtype": "float32",
    }

    cfg, mla = self.init_mla({**config_arguments, "attention": "dot_product"}, rope_type="default")
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg, cfg.dtype)
    mla_generic_output, _ = mla(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    generic_state = nnx.state(mla)

    cfg_cp = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
        attention="flash",
        rope_type=cfg.rope_type,
        context_parallel_strategy="all_gather",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=ici_context_parallelism,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=False,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
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
          attention_type=AttentionType(cfg_cp.attention_type),
          q_lora_rank=cfg_cp.q_lora_rank,
          kv_lora_rank=cfg_cp.kv_lora_rank,
          qk_nope_head_dim=cfg_cp.qk_nope_head_dim,
          qk_rope_head_dim=cfg_cp.qk_rope_head_dim,
          v_head_dim=cfg_cp.v_head_dim,
          model_mode=MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    nnx.update(attention_as_mla_flash_cp, generic_state)

    mla_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mla_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    mla_generic_output = jax.device_get(mla_generic_output)
    mla_generic_flash_cp_output = jax.device_get(mla_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mla_generic_output, mla_generic_flash_cp_output, rtol=1e-02, atol=1e-02, equal_nan=False),
        msg=(
            "MLA+Indexer logits from generic dot product and flash attention + all-gather context parallelism are not"
            f" close. context_parallel_load_balance={context_parallel_load_balance}."
        ),
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "no_lb_cp2",
          "context_parallel_load_balance": False,
          "ici_context_parallelism": 2,
          "indexer_topk": 256,
      },
      {
          "testcase_name": "lb_cp4_smallk",
          "context_parallel_load_balance": True,
          "ici_context_parallelism": 4,
          "indexer_topk": 32,
      },
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_ring_context_parallel_with_indexer(
      self, context_parallel_load_balance, ici_context_parallelism=2, indexer_topk=256
  ):
    """Test equivalence between dot_product MLA + Indexer and flash attention + ring context parallelism + Indexer"""
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
        "attention_type": AttentionType.MLA.value,
        "use_indexer": True,
        "indexer_loss_scaling_factor": 0.1,
        "indexer_topk": indexer_topk,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "dtype": "float32",
    }

    cfg, mla = self.init_mla({**config_arguments, "attention": "dot_product"}, rope_type="default")
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg, cfg.dtype)
    mla_generic_output, _ = mla(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    generic_state = nnx.state(mla)

    cfg_cp = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
        attention="flash",
        rope_type=cfg.rope_type,
        context_parallel_strategy="ring",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=ici_context_parallelism,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=False,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
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
          attention_type=AttentionType(cfg_cp.attention_type),
          q_lora_rank=cfg_cp.q_lora_rank,
          kv_lora_rank=cfg_cp.kv_lora_rank,
          qk_nope_head_dim=cfg_cp.qk_nope_head_dim,
          qk_rope_head_dim=cfg_cp.qk_rope_head_dim,
          v_head_dim=cfg_cp.v_head_dim,
          model_mode=MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    nnx.update(attention_as_mla_flash_cp, generic_state)

    mla_generic_flash_cp_output = attention_test_util.forward_with_context_expert_parallelism(
        cfg_cp,
        mesh_cp,
        attention_as_mla_flash_cp,
        lnx,
        decoder_segment_ids,
        decoder_positions,
    )

    mla_generic_output = jax.device_get(mla_generic_output)
    mla_generic_flash_cp_output = jax.device_get(mla_generic_flash_cp_output)

    self.assertTrue(
        jax.numpy.allclose(mla_generic_output, mla_generic_flash_cp_output, rtol=1e-02, atol=1e-02, equal_nan=False),
        msg="MLA+Indexer logits from generic dot product and flash attention + ring context parallelism are not close. "
        f"context_parallel_load_balance={context_parallel_load_balance}.",
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "no_lb_cp2",
          "context_parallel_load_balance": False,
          "ici_context_parallelism": 2,
          "indexer_topk": 256,
      },
      {
          "testcase_name": "lb_cp4_smallk",
          "context_parallel_load_balance": True,
          "ici_context_parallelism": 4,
          "indexer_topk": 32,
      },
  )
  @pytest.mark.tpu_only
  def test_tpu_flash_attention_ring_context_parallel_grad_with_indexer(
      self, context_parallel_load_balance, ici_context_parallelism=2, indexer_topk=256
  ):
    """Test gradient equivalence between dot_product and flash attention + ring context parallelism with Indexer"""
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
        "attention_type": AttentionType.MLA.value,
        "use_indexer": True,
        "indexer_loss_scaling_factor": 0.1,
        "indexer_topk": indexer_topk,
        "q_lora_rank": 4,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "dtype": "float32",
    }

    cfg, mla = self.init_mla({**config_arguments, "attention": "dot_product"}, rope_type="default")
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg, cfg.dtype)

    cfg_cp = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
        attention="flash",
        rope_type=cfg.rope_type,
        context_parallel_strategy="ring",
        context_parallel_load_balance=context_parallel_load_balance,
        ici_context_parallelism=ici_context_parallelism,
        use_tokamax_splash=True,
        use_jax_splash=False,
        packing=False,
    )
    devices_array_cp = maxtext_utils.create_device_mesh(cfg_cp)
    mesh_cp = Mesh(devices_array_cp, cfg_cp.mesh_axes)
    with nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
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
          attention_type=AttentionType(cfg_cp.attention_type),
          q_lora_rank=cfg_cp.q_lora_rank,
          kv_lora_rank=cfg_cp.kv_lora_rank,
          qk_nope_head_dim=cfg_cp.qk_nope_head_dim,
          qk_rope_head_dim=cfg_cp.qk_rope_head_dim,
          v_head_dim=cfg_cp.v_head_dim,
          model_mode=MODEL_MODE_PREFILL,
          rngs=self.nnx_rng,
      )
    nnx.update(attention_as_mla_flash_cp, nnx.state(mla))
    generic_graphdef, generic_state = nnx.split(mla)
    ring_graphdef, ring_state = nnx.split(attention_as_mla_flash_cp)

    def generic_loss(lnx):
      mla_merged = nnx.merge(generic_graphdef, generic_state)
      output, _ = mla_merged(
          lnx,
          lnx,
          decoder_segment_ids=decoder_segment_ids,
          inputs_positions=decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    def ring_loss(lnx):
      if context_parallel_load_balance:
        context_parallel_size = cfg_cp.ici_context_parallelism
        lnx = max_utils.reorder_sequence(lnx, cp_size=context_parallel_size)
        ring_decoder_segment_ids = max_utils.reorder_sequence(decoder_segment_ids, cp_size=context_parallel_size)
        ring_decoder_positions = max_utils.reorder_sequence(decoder_positions, cp_size=context_parallel_size)
      else:
        ring_decoder_segment_ids = decoder_segment_ids
        ring_decoder_positions = decoder_positions
      ring_merged = nnx.merge(ring_graphdef, ring_state)
      output, _ = ring_merged(
          lnx,
          lnx,
          decoder_segment_ids=ring_decoder_segment_ids,
          inputs_positions=ring_decoder_positions,
          deterministic=True,
          model_mode=MODEL_MODE_TRAIN,
      )
      return jnp.mean(output.astype(jnp.float32) ** 2)

    generic_grad = jax.grad(generic_loss)(lnx)
    with jax.set_mesh(mesh_cp), nn_partitioning.axis_rules(cfg_cp.logical_axis_rules):
      ring_grad = jax.grad(ring_loss)(lnx)
    generic_grad = jax.device_get(generic_grad)
    ring_grad = jax.device_get(ring_grad)

    self.assertTrue(
        jax.numpy.allclose(generic_grad, ring_grad, rtol=1e-02, atol=1e-06, equal_nan=False),
        msg=(
            "MLA+Indexer input gradients from generic dot product and flash attention + ring context parallelism are"
            f" not close. context_parallel_load_balance={context_parallel_load_balance}."
        ),
    )

  def get_indexer_test_data(self, batch_size, q_len, kv_len, num_heads, head_dim):
    """Helper to generate random data for indexer tests."""
    key_q, key_k, key_is = jax.random.split(self.rng, 3)
    query = jax.random.normal(key_q, (batch_size, q_len, num_heads, head_dim))
    key = jax.random.normal(key_k, (batch_size, kv_len, num_heads, head_dim))
    indexer_score = jax.random.normal(key_is, (batch_size, q_len, kv_len))
    return query, key, indexer_score

  def get_causal_mask_for_indexer(self, batch_size, q_len, kv_len):
    """Helper to generate a causal mask with DEFAULT_MASK_VALUE."""
    row_ids = jnp.arange(q_len)[:, None]
    col_ids = jnp.arange(kv_len)[None, :]
    attention_mask = jnp.where(col_ids <= row_ids, 0.0, DEFAULT_MASK_VALUE)
    attention_mask = jnp.broadcast_to(attention_mask, (batch_size, q_len, kv_len))
    return attention_mask

  def test_indexer_loss(self):
    """Test indexer loss computation."""
    mla_config_args = self.config_arguments.copy()
    mla_config_args["use_indexer"] = True
    mla_config_args["attention"] = "dot_product"
    _, mla = self.init_mla(mla_config_args, rope_type="default")

    batch_size = 2
    q_len = 3
    kv_len = 4
    num_heads = 5
    head_dim = 6
    scaling_factor = 0.5

    query, key, indexer_score = self.get_indexer_test_data(batch_size, q_len, kv_len, num_heads, head_dim)

    # Causal mask
    attention_mask = self.get_causal_mask_for_indexer(batch_size, q_len, kv_len)
    indexer_score += attention_mask

    topk_values, _ = jax.lax.top_k(indexer_score, k=2)
    indexer_mask = mla.indexer.generate_mask(indexer_score, topk_values) + attention_mask

    loss_dense = mla.calculate_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        key=key,
        attention_mask=attention_mask,
        indexer_mask=indexer_mask,
        sparse_loss=False,
        scaling_factor=scaling_factor,
    )

    loss_sparse = mla.calculate_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        key=key,
        attention_mask=attention_mask,
        indexer_mask=indexer_mask,
        sparse_loss=True,
        scaling_factor=scaling_factor,
    )

    np.testing.assert_array_less(0.0, loss_dense)
    np.testing.assert_array_less(0.0, loss_sparse)

  def test_indexer_loss_kl_divergence_zero(self):
    """Test that KL divergence is 0 when target and pred distributions match exactly."""
    mla_config_args = self.config_arguments.copy()
    mla_config_args["use_indexer"] = True
    mla_config_args["attention"] = "dot_product"
    _, mla = self.init_mla(mla_config_args, rope_type="default")

    batch_size = 2
    q_len = 3
    kv_len = 4
    num_heads = 5
    head_dim = 6

    # Setup perfectly matching distributions
    # Make query and key such that einsum yields zeros (so softmax gives uniform distribution over unmasked)
    query = jnp.zeros((batch_size, q_len, num_heads, head_dim))
    key = jnp.zeros((batch_size, kv_len, num_heads, head_dim))

    # Causal mask
    attention_mask = self.get_causal_mask_for_indexer(batch_size, q_len, kv_len)

    # Indexer score matches the shape and is uniform
    indexer_score = jnp.zeros((batch_size, q_len, kv_len)) + attention_mask

    topk_values, _ = jax.lax.top_k(indexer_score, k=2)
    indexer_mask = mla.indexer.generate_mask(indexer_score, topk_values) + attention_mask

    loss = mla.calculate_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        key=key,
        attention_mask=attention_mask,
        indexer_mask=indexer_mask,
        sparse_loss=False,
        scaling_factor=1.0,
    )

    np.testing.assert_allclose(loss, 0.0, atol=1e-5)

  def test_indexer_with_approx_top_k(self):
    """Verify indexer runs with both approx and exact top-k."""
    for use_approx in [False, True]:
      with self.subTest(indexer_use_approx_top_k=use_approx):
        mla_config_args = self.config_arguments.copy()
        mla_config_args["use_indexer"] = True
        mla_config_args["indexer_use_approx_top_k"] = use_approx
        mla_config_args["indexer_topk"] = 4  # Force indexer to run instead of returning early
        mla_config_args["attention"] = "dot_product"

        cfg, mla = self.init_mla(mla_config_args, rope_type="default")

        lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(cfg, cfg.dtype)

        # Run forward pass which triggers indexer
        out, _ = mla(
            lnx,
            lnx,
            decoder_segment_ids=decoder_segment_ids,
            inputs_positions=decoder_positions,
            deterministic=True,
            model_mode=MODEL_MODE_TRAIN,
        )
        self.assertIsNotNone(out)

  def test_approx_top_k_recall(self):
    """Verify that approx_max_k meets the specified recall target compared to exact top_k."""
    jax_rng = jax.random.PRNGKey(0)

    # We need a large enough N to make the approximation meaningful.
    # Use shape [batch=4, queries=16, N=1024]
    batch, queries, N = 4, 16, 1024
    K = 64
    recall_target = 0.95

    # Generate random scores
    scores = jax.random.normal(jax_rng, (batch, queries, N))

    # 1. Run exact Top-K
    _, true_indices = jax.lax.top_k(scores, k=K)  # [batch, queries, K]

    # 2. Run approx Top-K
    _, approx_indices = jax.lax.approx_max_k(scores, k=K, recall_target=recall_target)  # [batch, queries, K]

    # 3. Calculate Recall
    # Broadcast compare true_indices [B, Q, K, 1] and approx_indices [B, Q, 1, K]
    matches = (true_indices[..., None] == approx_indices[..., None, :]).any(axis=-1)  # [B, Q, K]
    num_matches = matches.sum(axis=-1)  # [B, Q]
    actual_recalls = num_matches / K  # [B, Q]
    mean_recall = jnp.mean(actual_recalls)

    print(f"\nApprox Top-K Recall Target: {recall_target}, Actual Mean Recall: {mean_recall:.4f}")

    # Assert that the actual recall is equal or exceeds the target.
    self.assertGreaterEqual(mean_recall, recall_target)

  def test_indexer_gradients(self):
    # Test that gradients do NOT flow back to inputs
    bsz, seqlen = 2, 8
    inputs_positions = jnp.broadcast_to(jnp.arange(seqlen)[None, :], (bsz, seqlen))

    for sparse_training in [False, True]:
      with self.subTest(indexer_sparse_training=sparse_training):
        argv = [
            "",
            get_test_config_path(),
            "run_name=test",
            "attention_type=mla",
            "attention=dot_product",
            "use_indexer=True",
            f"indexer_sparse_training={sparse_training}",
            "max_target_length=16",
            "indexer_topk=4",
            "indexer_n_heads=2",
            "indexer_head_dim=8",
            "emb_dim=16",
            "qk_rope_head_dim=4",
            "q_lora_rank=16",
        ]
        config = pyconfig.initialize(argv)
        rngs = nnx.Rngs(0)
        mesh = jax.sharding.Mesh(jax.devices(), ("data",))
        rope = embeddings.RotaryEmbedding(
            min_timescale=1,
            max_timescale=10000,
            mesh=mesh,
            embedding_dims=config.qk_rope_head_dim,
            fprop_dtype=jnp.float32,
            rngs=rngs,
        )
        rope.interleave = False

        mla = MLA(
            config=config,
            num_query_heads=config.num_query_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            max_target_length=config.max_target_length,
            mesh=mesh,
            attention_kernel="dot_product",
            inputs_q_shape=(bsz, seqlen, config.emb_dim),
            inputs_kv_shape=(bsz, seqlen, config.emb_dim),
            dtype=jnp.float32,
            weight_dtype=jnp.float32,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            rngs=rngs,
        )

        inputs_q = jnp.ones((bsz, seqlen, config.emb_dim))
        inputs_kv = jnp.ones((bsz, seqlen, config.emb_dim))
        low_rank_q = jnp.ones((bsz, seqlen, config.q_lora_rank))

        def full_indexer_loss_fn(inputs_q, inputs_kv, low_rank_q, mla, sparse_training=sparse_training):
          # 1. Main model projections
          # We ignore the low_rank_q returned here and use the explicitly passed one
          # to directly verify its gradients.
          query, _ = mla.mla_query_projection(inputs_q, inputs_positions, MODEL_MODE_TRAIN)
          key, _, _ = mla.mla_kv_projection(inputs_kv, inputs_positions, None, MODEL_MODE_TRAIN, None)

          # 2. Indexer forward
          indexer_mask, _, indexer_score = mla.indexer(
              inputs_q=inputs_q,
              low_rank_q=low_rank_q,
              inputs_kv=inputs_kv,
              inputs_positions=inputs_positions,
          )

          # 3. Calculate full KL loss
          loss = mla.calculate_indexer_loss(
              indexer_score=indexer_score,
              query=query,
              key=key,
              attention_mask=None,
              indexer_mask=indexer_mask,
              sparse_loss=sparse_training,
              scaling_factor=1.0,
          )
          return loss

        # Calculate gradients with respect to input embeddings and low_rank_q
        grad_fn = nnx.grad(full_indexer_loss_fn, argnums=(0, 1, 2))
        grad_q, grad_kv, grad_low_rank_q = grad_fn(inputs_q, inputs_kv, low_rank_q, mla)

        # Gradients should be exactly zero because:
        # a) Indexer inputs are detached in Indexer.__call__
        # b) Main model query/key are detached in calculate_indexer_loss
        self.assertTrue(jnp.all(grad_q == 0.0))
        self.assertTrue(jnp.all(grad_kv == 0.0))
        self.assertTrue(jnp.all(grad_low_rank_q == 0.0))

  def old_generate_mask(self, topk_indices, s, dtype=jnp.float32):
    """Old baseline implementation using pairwise broadcast comparison.

    Retained exclusively in unit tests as a ground-truth reference for cross-checking mathematical equivalence.
    """
    is_topk = (jnp.arange(s) == topk_indices[..., None]).any(axis=-2)
    val_true = jnp.array(0.0, dtype=dtype)
    val_false = jnp.array(DEFAULT_MASK_VALUE, dtype=dtype)
    return jnp.where(is_topk, val_true, val_false)

  def test_generate_mask_threshold_equivalence(self):
    """Verifies that TPU-native threshold cutoff masking matches exact top-k selection when scores are unique."""
    mla_config_args = self.config_arguments.copy()
    mla_config_args["use_indexer"] = True
    mla_config_args["indexer_topk"] = 64
    mla_config_args["attention"] = "dot_product"

    _, mla = self.init_mla(mla_config_args, rope_type="default")

    jax_rng = jax.random.PRNGKey(0)
    b, t, s, k = 2, 128, 1024, 64
    dtype = jnp.float32

    scores = jax.random.normal(jax_rng, (b, t, s), dtype=dtype)
    # Add tiny position-dependent epsilon so all scores are strictly unique (preventing sorting tie-breaker divergence)
    scores = scores + jnp.arange(s, dtype=dtype) * (1e-6 / s)
    topk_values, topk_indices = jax.lax.top_k(scores, k=k)

    # Call original broadcast Indexer logic for cross-checking
    mask_original = self.old_generate_mask(topk_indices, s, dtype=dtype)

    # Call actual optimized Indexer logic
    mask_threshold = mla.indexer.generate_mask(scores, topk_values)

    self.assertTrue(jnp.allclose(mask_original, mask_threshold, atol=1e-5))

  def test_generate_mask_threshold_ties_exact_k(self):
    """Verifies that prefix-sum pruning guarantees exactly k unmasked tokens even with boundary ties."""
    mla_config_args = self.config_arguments.copy()
    mla_config_args["use_indexer"] = True
    mla_config_args["indexer_topk"] = 3
    mla_config_args["attention"] = "dot_product"
    mla_config_args["indexer_mask_exact_topk"] = True

    _, mla = self.init_mla(mla_config_args, rope_type="default")

    k = 3
    dtype = jnp.float32
    scores = jnp.array(
        [
            [
                [0.9, 0.8, 0.5, 0.5, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0],
                [0.5, 0.5, 0.9, 0.8, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0],
            ]
        ],
        dtype=dtype,
    )

    topk_values, topk_indices = jax.lax.top_k(scores, k=k)
    mask = mla.indexer.generate_mask(scores, topk_values)
    mask_original = self.old_generate_mask(topk_indices, s=scores.shape[-1], dtype=dtype)

    val_true = jnp.array(0.0, dtype=dtype)

    self.assertFalse(jnp.isnan(mask).any())
    self.assertEqual(jnp.sum(mask[0, 0] == val_true), 3)  # Exactly 3 tokens (exact k) unmasked
    self.assertEqual(jnp.sum(mask[0, 1] == val_true), 3)  # Exactly 3 tokens (exact k) unmasked

    # Assert equivalence to original broadcast baseline
    self.assertTrue(jnp.allclose(mask_original, mask, atol=1e-5))

    # Assert exact unmasked elements
    np.testing.assert_array_equal(
        mask[0, 0] == val_true,
        [True, True, True, False, False, False, False, False, False, False],
    )
    np.testing.assert_array_equal(
        mask[0, 1] == val_true,
        [True, False, True, True, False, False, False, False, False, False],
    )

  def test_generate_mask_threshold_ties_unsorted(self):
    """Verifies that elements strictly greater than cutoff are preserved even if they appear after ties."""
    mla_config_args = self.config_arguments.copy()
    mla_config_args["use_indexer"] = True
    mla_config_args["indexer_topk"] = 3
    mla_config_args["attention"] = "dot_product"
    mla_config_args["indexer_mask_exact_topk"] = True

    _, mla = self.init_mla(mla_config_args, rope_type="default")

    k = 3
    dtype = jnp.float32
    scores = jnp.array(
        [
            [
                # 0.9 is strictly greater but appears after three 0.5s.
                # If cumsum was used unconditionally on (score >= cutoff), 0.9 would get rank 4 and be masked out!
                # Correct behavior: keep 0.9, and the first two 0.5s to reach exactly k=3.
                [0.5, 0.5, 0.5, 0.9, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0],
            ]
        ],
        dtype=dtype,
    )

    topk_values, topk_indices = jax.lax.top_k(scores, k=k)
    mask = mla.indexer.generate_mask(scores, topk_values)
    mask_original = self.old_generate_mask(topk_indices, s=scores.shape[-1], dtype=dtype)

    val_true = jnp.array(0.0, dtype=dtype)

    self.assertFalse(jnp.isnan(mask).any())
    self.assertEqual(jnp.sum(mask[0, 0] == val_true), 3)  # Exactly 3 tokens unmasked

    # Assert equivalence to original broadcast baseline
    self.assertTrue(jnp.allclose(mask_original, mask, atol=1e-5))

    np.testing.assert_array_equal(
        mask[0, 0] == val_true,
        [True, True, False, True, False, False, False, False, False, False],
    )

  def test_generate_mask_approx_k_overflow(self):
    """Verifies exact-k guarantee when approx_top_k underestimates the threshold."""
    mla_config_args = self.config_arguments.copy()
    mla_config_args["use_indexer"] = True
    mla_config_args["indexer_topk"] = 3
    mla_config_args["attention"] = "dot_product"
    mla_config_args["indexer_mask_exact_topk"] = True

    _, mla = self.init_mla(mla_config_args, rope_type="default")

    dtype = jnp.float32
    scores = jnp.array(
        [
            [
                # Simulating approx_max_k returning an underestimated threshold of 0.5.
                # However, 0.9, 0.8, 0.7, 0.6 (4 elements) are strictly > 0.5.
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.0, -1.0, -2.0, -3.0, -4.0],
            ]
        ],
        dtype=dtype,
    )

    # Artificially supply a threshold of 0.5 at the end
    topk_values = jnp.array([[[1.0, 1.0, 0.5]]], dtype=dtype)
    mask = mla.indexer.generate_mask(scores, topk_values)

    val_true = jnp.array(0.0, dtype=dtype)

    self.assertFalse(jnp.isnan(mask).any())
    self.assertEqual(jnp.sum(mask[0, 0] == val_true), 3)  # Exactly 3 tokens unmasked

    # It should keep the first 3 elements that are > 0.5 which are 0.9, 0.8, 0.7
    np.testing.assert_array_equal(
        mask[0, 0] == val_true,
        [True, True, True, False, False, False, False, False, False, False],
    )

  def test_generate_mask_threshold_ties_raw(self):
    """Verifies that raw thresholding allows more than k unmasked tokens under boundary ties."""
    mla_config_args = self.config_arguments.copy()
    mla_config_args["use_indexer"] = True
    mla_config_args["indexer_topk"] = 3
    mla_config_args["attention"] = "dot_product"
    mla_config_args["indexer_mask_exact_topk"] = False

    _, mla = self.init_mla(mla_config_args, rope_type="default")

    k = 3
    dtype = jnp.float32
    scores = jnp.array(
        [
            [
                [0.9, 0.8, 0.5, 0.5, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0],
                [0.7, 0.6, 0.5, 0.2, 0.1, 0.0, -1.0, -2.0, -3.0, -4.0],
            ]
        ],
        dtype=dtype,
    )

    topk_values, _ = jax.lax.top_k(scores, k=k)
    mask = mla.indexer.generate_mask(scores, topk_values)

    self.assertFalse(jnp.isnan(mask).any())
    self.assertEqual(jnp.sum(mask[0, 0] == 0.0), 4)  # 4 tokens unmasked (tied >= 0.5)
    self.assertEqual(jnp.sum(mask[0, 1] == 0.0), 3)  # Exactly 3 tokens unmasked (no boundary ties)

  def test_generate_mask_sequence_smaller_than_k(self):
    """Verifies that the indexer handles sequence length smaller than or equal to k by returning None."""
    mla_config_args = self.config_arguments.copy()
    mla_config_args["use_indexer"] = True
    mla_config_args["indexer_topk"] = 10  # k = 10
    mla_config_args["attention"] = "dot_product"

    cfg, mla = self.init_mla(mla_config_args, rope_type="default")

    dtype = jnp.float32
    inputs_q = jnp.zeros((1, 5, cfg.emb_dim), dtype=dtype)
    inputs_kv = jnp.zeros((1, 5, cfg.emb_dim), dtype=dtype)  # s = 5 <= k
    low_rank_q = jnp.zeros((1, 5, cfg.q_lora_rank), dtype=dtype)
    inputs_positions = jnp.zeros((1, 5), dtype=jnp.int32)

    mask, indices, score = mla.indexer(
        inputs_q=inputs_q,
        low_rank_q=low_rank_q,
        inputs_kv=inputs_kv,
        inputs_positions=inputs_positions,
    )

    self.assertIsNone(mask)
    self.assertIsNone(indices)
    self.assertIsNone(score)

  def test_mla_indexer_loss_chunking_parity(self):
    """Tests that MLA calculate_indexer_loss produces identically the same loss regardless of head_chunk_size."""
    rng = jax.random.PRNGKey(0)
    batch_size = 2
    q_len = 16
    s_len = 16
    heads = 4
    dim = 8

    # Mock inputs
    indexer_score = jax.random.normal(rng, (batch_size, q_len, s_len))
    query = jax.random.normal(rng, (batch_size, q_len, heads, dim))
    key = jax.random.normal(rng, (batch_size, s_len, heads, dim))
    attention_mask = None
    indexer_mask = jax.random.uniform(rng, (batch_size, q_len, s_len)) < 0.2

    # Initialize a dummy config
    cfg = pyconfig.initialize(
        [
            None,
            "maxtext/configs/base.yml",
            "attention=dot_product",
            "num_query_heads=4",
            "num_kv_heads=4",
            "head_dim=8",
            "indexer_topk=4",
            "attention_type=mla",
        ]
    )

    mla = MLA(
        config=cfg,
        num_query_heads=heads,
        num_kv_heads=heads,
        head_dim=dim,
        dtype=jnp.float32,
        weight_dtype=jnp.float32,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        max_position_embeddings=128,
        original_max_position_embeddings=128,
        max_target_length=128,
        attention_kernel="dot_product",
        mesh=None,
        inputs_q_shape=(2, 16, 8),
        inputs_kv_shape=(2, 16, 8),
        rngs=nnx.Rngs(0),
    )

    # 1. Native Evaluation (No Chunking)
    cfg_dense = copy.deepcopy(cfg)
    object.__setattr__(cfg_dense, "mla_qk_head_chunk_size", 0)
    mla.config = cfg_dense

    loss_native = mla.calculate_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        key=key,
        attention_mask=attention_mask,
        indexer_mask=indexer_mask,
        sparse_loss=True,
        scaling_factor=1.0,
    )

    # 2. Chunked Evaluation
    cfg_chunked = copy.deepcopy(cfg)
    object.__setattr__(cfg_chunked, "mla_qk_head_chunk_size", 2)
    mla.config = cfg_chunked

    loss_chunked = mla.calculate_indexer_loss(
        indexer_score=indexer_score,
        query=query,
        key=key,
        attention_mask=attention_mask,
        indexer_mask=indexer_mask,
        sparse_loss=True,
        scaling_factor=1.0,
    )

    np.testing.assert_allclose(loss_native, loss_chunked, rtol=1e-5, atol=1e-5)


class Qwen3NextGatedDeltaNetTest(unittest.TestCase):
  """Test for the Gated Delta Net in Qwen3-Next"""

  def setUp(self):
    super().setUp()
    self.config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test",
        "enable_checkpointing": False,
        "max_prefill_predict_length": 16,
        "max_target_length": 32,
        "base_emb_dim": 128,  # changed to base_emb_dim so it properly overrides the default 2048
        "gdn_num_value_heads": 4,
        "gdn_num_key_heads": 4,
        "gdn_key_head_dim": 32,
        "gdn_value_head_dim": 32,
        "gdn_conv_kernel_dim": 4,
        "gdn_chunk_size": 16,
        "dtype": "bfloat16",
    }
    self.cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **self.config_arguments,
    )
    devices_array = maxtext_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    self.rng = jax.random.PRNGKey(0)
    self.nnx_rng = nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42))

  def get_structured_data(self, dtype):
    """get structured data for GDN (only requires hidden states)"""
    lnx = jax.random.normal(
        self.rng,
        shape=(self.cfg.global_batch_size_to_train_on, self.cfg.max_target_length, self.cfg.emb_dim),
        dtype=dtype,
    )
    return lnx

  @pytest.mark.cpu_only
  def test_train_path_checks_all_batch_sharding_specs(self):
    """The non-paged GDN path makes every batch-sharded spec shape-compatible."""
    lnx = self.get_structured_data(self.cfg.dtype)
    gdn = Qwen3NextGatedDeltaNet(
        config=self.cfg,
        inputs_shape=lnx.shape,
        mesh=self.mesh,
        dtype=self.cfg.dtype,
        model_mode=MODEL_MODE_TRAIN,
        rngs=self.nnx_rng,
    )

    with mock.patch(
        "maxtext.models.qwen3.remove_incompatible_mesh_axes_from_partition_spec",
        wraps=sharding.remove_incompatible_mesh_axes_from_partition_spec,
    ) as make_compatible:
      output, _ = gdn(lnx, model_mode=MODEL_MODE_TRAIN)

    self.assertEqual(output.shape, lnx.shape)
    self.assertEqual(make_compatible.call_count, 4)
    self.assertEqual([len(call.args[1]) for call in make_compatible.call_args_list], [4, 4, 3, 4])
    self.assertTrue(all(call.kwargs["dims"] == (0,) for call in make_compatible.call_args_list))
    self.assertTrue(all(call.kwargs["allow_remove_axes"] for call in make_compatible.call_args_list))

  @pytest.mark.cpu_only
  @pytest.mark.post_training
  def test_paged_state_truncates_metadata_to_active_requests(self):
    """The paged-state bridge trims maximum-size metadata buffers."""
    gdn_attention = pytest.importorskip("tpu_inference.layers.common.gdn_attention")

    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path("inference/vllm.yml")],
        run_name="paged_gdn_metadata_test",
        enable_checkpointing=False,
        log_config=False,
        base_emb_dim=16,
        gdn_num_value_heads=2,
        gdn_num_key_heads=2,
        gdn_key_head_dim=4,
        gdn_value_head_dim=4,
        gdn_conv_kernel_dim=4,
        gdn_chunk_size=4,
        dtype="float32",
        weight_dtype="float32",
        max_prefill_predict_length=2,
        max_target_length=4,
        per_device_batch_size=1.0,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    hidden_states = jnp.ones((1, 1, cfg.emb_dim), dtype=cfg.dtype)
    gdn = Qwen3NextGatedDeltaNet(
        config=cfg,
        inputs_shape=hidden_states.shape,
        mesh=mesh,
        dtype=cfg.dtype,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        rngs=nnx.Rngs(params=0, dropout=1),
    )

    num_blocks = 2
    key_dim = cfg.gdn_num_key_heads * cfg.gdn_key_head_dim
    value_dim = cfg.gdn_num_value_heads * cfg.gdn_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    conv_state = jnp.zeros((num_blocks, cfg.gdn_conv_kernel_dim - 1, conv_dim), dtype=cfg.dtype)
    recurrent_state = jnp.zeros(
        (num_blocks, cfg.gdn_num_value_heads, cfg.gdn_key_head_dim, cfg.gdn_value_head_dim),
        dtype=cfg.dtype,
    )
    attention_metadata = types.SimpleNamespace(
        padded_num_reqs=1,
        mamba_state_indices=jnp.array([1, 101, 102], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1, 101, 201], dtype=jnp.int32),
        request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        seq_lens=jnp.array([1, 101, 102], dtype=jnp.int32),
    )

    with mock.patch.object(gdn_attention, "run_jax_gdn_attention", autospec=True) as mock_run_gdn:
      mock_run_gdn.return_value = (
          (conv_state, recurrent_state),
          jnp.zeros((hidden_states.shape[1], value_dim), dtype=cfg.dtype),
      )
      output, new_cache = gdn(
          hidden_states,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
          kv_cache=(conv_state, recurrent_state),
          attention_metadata=attention_metadata,
      )

    mock_run_gdn.assert_called_once()
    self.assertEqual(len(mock_run_gdn.call_args.args), 18)
    self.assertEqual(set(mock_run_gdn.call_args.kwargs), {"mesh"})
    self.assertIs(mock_run_gdn.call_args.kwargs["mesh"], mesh)
    np.testing.assert_array_equal(mock_run_gdn.call_args.args[9], jnp.array([1], dtype=jnp.int32))
    np.testing.assert_array_equal(mock_run_gdn.call_args.args[10], jnp.array([0, 1], dtype=jnp.int32))
    np.testing.assert_array_equal(mock_run_gdn.call_args.args[12], jnp.array([1], dtype=jnp.int32))
    self.assertEqual(output.shape, hidden_states.shape)
    self.assertEqual(new_cache[0].shape, conv_state.shape)
    self.assertEqual(new_cache[1].shape, recurrent_state.shape)

  @pytest.mark.tpu_only
  def test_autoregression(self):
    cfg = self.cfg
    prefill_length = cfg.max_prefill_predict_length
    decode_total_length = cfg.max_target_length

    # 1. Init Data
    lnx = self.get_structured_data(cfg.dtype)

    # 2. Init GDN Layer
    gdn = Qwen3NextGatedDeltaNet(
        config=cfg,
        inputs_shape=lnx.shape,
        mesh=self.mesh,
        dtype=cfg.dtype,
        model_mode=MODEL_MODE_PREFILL,
        rngs=self.nnx_rng,
    )

    # 3. Full / Train mode
    gdn_full, _ = gdn(
        lnx,
        model_mode=MODEL_MODE_TRAIN,
    )

    # 4. Prefill mode
    lnx_prefill = lnx[:, 0:prefill_length, :]

    gdn_prefill, _ = gdn(
        lnx_prefill,
        model_mode=MODEL_MODE_PREFILL,
    )

    self.assertTrue(
        jax.numpy.allclose(gdn_prefill, gdn_full[:, :prefill_length, :], rtol=1e-02, atol=1e-02, equal_nan=False)
    )

    # 5. Autoregressive mode
    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx : idx + 1, :]

      gdn_idx, _ = gdn(
          lnx_idx,
          model_mode=MODEL_MODE_AUTOREGRESSIVE,
      )

      gdn_full_this_idx = gdn_full[:, idx : idx + 1, :]
      self.assertEqual(gdn_full_this_idx.shape, gdn_idx.shape)

      self.assertTrue(jax.numpy.allclose(gdn_full_this_idx, gdn_idx, rtol=1e-02, atol=1e-02, equal_nan=False))


class DeepSeekV4AttentionMaskingTest(unittest.TestCase):
  """Tests to validate AttentionOp masking logic for DeepSeek-V4 attention patterns."""

  def setUp(self):
    self.config = pyconfig.initialize([sys.argv[0], "src/maxtext/configs/base.yml"], run_name="test")

  def test_generate_attention_mask_local_sliding(self):
    """Verifies AttentionType.LOCAL_SLIDING enforces both causal and sliding window constraints."""

    # Test with multiple heads and different sequence lengths
    for s_len in [1, 8, 128]:
      op = AttentionOp(
          config=self.config,
          num_query_heads=4,
          num_kv_heads=1,
          max_target_length=256,
          mesh=None,
          attention_kernel="dot_product",
          attention_type=AttentionType.LOCAL_SLIDING,
          sliding_window_size=3,
      )

      batch_size = 1
      q_dummy = jnp.zeros((batch_size, s_len, 1, 128))
      k_dummy = jnp.zeros((batch_size, s_len, 1, 128))

      mask = op.generate_attention_mask(
          query=q_dummy,
          key=k_dummy,
          decoder_segment_ids=None,
          model_mode="train",
      )

      self.assertEqual(mask.shape, (1, 1, 1, s_len, s_len))
      mask_np = np.array(mask)[0, 0, 0]

      # Expected float mask for window_size=3
      # Row 0: [0.0, INF, INF, INF, INF, ...]
      # Row 1: [0.0, 0.0, INF, INF, INF, ...]
      # Row 2: [0.0, 0.0, 0.0, INF, INF, ...]
      # Row 3: [INF, 0.0, 0.0, 0.0, INF, ...]
      if s_len > 1:
        self.assertEqual(mask_np[0, 1], DEFAULT_MASK_VALUE)  # strict causal
      self.assertEqual(mask_np[0, 0], 0.0)

      if s_len >= 4:
        self.assertEqual(mask_np[3, 0], DEFAULT_MASK_VALUE)  # sliding window size=3
        self.assertEqual(mask_np[3, 1], 0.0)

  def test_generate_attention_mask_compressed(self):
    """Verifies AttentionType.COMPRESSED stitches sliding window and float compressed_mask."""

    batch_size = 1
    s_len = 8
    c_len = 2
    kv_len = s_len + c_len

    op = AttentionOp(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        max_target_length=128,
        mesh=None,
        attention_kernel="dot_product",
        attention_type=AttentionType.COMPRESSED,
        sliding_window_size=3,
    )

    q_dummy = jnp.zeros((batch_size, s_len, 1, 128))
    k_dummy = jnp.zeros((batch_size, kv_len, 1, 128))

    # Simulate a compressed float mask [batch, 1, s_len, c_len]
    # In practice, this exactly mirrors what both HCA and CSA output:
    # - HCA emits a simple mask blocking future blocks (batch, 1, seq_len, c_len)
    # - CSA emits a sparse mask where only top-K blocks are 0.0, rest are -inf.
    # We simulate this by making Block 0 invalid (-inf), and Block 1 valid (0.0).
    compressed_mask = np.zeros((batch_size, 1, s_len, c_len), dtype=np.float32)
    compressed_mask[:, :, :, 0] = DEFAULT_MASK_VALUE
    compressed_mask = jnp.array(compressed_mask)

    mask = op.generate_attention_mask(
        query=q_dummy,
        key=k_dummy,
        decoder_segment_ids=None,
        model_mode="train",
        compressed_mask=compressed_mask,
    )

    # Returned float mask should dynamically inherit the dimensionality of compressed_mask
    # Because compressed_mask was 4D, the final mask should also be 4D: [batch, 1, s_len, kv_len]
    self.assertEqual(mask.shape, (batch_size, 1, s_len, kv_len))
    mask_np = np.array(mask)[0, 0]

    # Uncompressed block (first s_len cols) follows sliding window float mask
    self.assertEqual(mask_np[0, 1], DEFAULT_MASK_VALUE)
    self.assertEqual(mask_np[0, 0], 0.0)
    self.assertEqual(mask_np[3, 0], DEFAULT_MASK_VALUE)
    self.assertEqual(mask_np[3, 1], 0.0)

    # Compressed block (last c_len cols) follows compressed_mask strictly
    np.testing.assert_allclose(mask_np[:, s_len], DEFAULT_MASK_VALUE)
    np.testing.assert_allclose(mask_np[:, s_len + 1], 0.0)
    print("Mask logic for uncompressed & compressed attention passed perfectly.")

  def test_generate_attention_mask_compressed_all_modes(self):
    """Verifies AttentionType.COMPRESSED across train, prefill, and autoregressive modes."""
    batch_size = 2
    s_len = 8
    c_len = 2
    kv_len = s_len + c_len

    op = AttentionOp(
        config=self.config,
        num_query_heads=4,
        num_kv_heads=1,
        max_target_length=128,
        mesh=None,
        attention_kernel="dot_product",
        attention_type=AttentionType.COMPRESSED,
        sliding_window_size=3,
    )

    # 1. Training mode (batch_size=2, 4D compressed_mask)
    q_train = jnp.zeros((batch_size, s_len, 1, 128))
    k_train = jnp.zeros((batch_size, kv_len, 1, 128))
    c_mask_4d = jnp.zeros((batch_size, 1, s_len, c_len), dtype=jnp.float32)
    mask_train = op.generate_attention_mask(
        query=q_train,
        key=k_train,
        decoder_segment_ids=None,
        model_mode="train",
        compressed_mask=c_mask_4d,
    )
    self.assertEqual(mask_train.shape, (batch_size, 1, s_len, kv_len))

    # 2. Prefill mode (batch_size=2, 5D compressed_mask with segment_positions)
    c_mask_5d = jnp.zeros((batch_size, 1, 1, s_len, c_len), dtype=jnp.float32)
    seg_pos = jnp.arange(s_len)[None, :].repeat(batch_size, axis=0)
    mask_prefill = op.generate_attention_mask(
        query=q_train,
        key=k_train,
        decoder_segment_ids=None,
        model_mode="prefill",
        compressed_mask=c_mask_5d,
        segment_positions=seg_pos,
    )
    self.assertEqual(mask_prefill.shape, (batch_size, 1, 1, s_len, kv_len))

    # 3. Autoregressive mode (q_seq_len=1, batch_size=2, decoder_segment_ids)
    q_ar = jnp.zeros((batch_size, 1, 1, 128))
    k_ar = jnp.zeros((batch_size, kv_len, 1, 128))
    c_mask_ar = jnp.zeros((batch_size, 1, 1, c_len), dtype=jnp.float32)
    seg_ids = jnp.ones((batch_size, 16), dtype=jnp.int32)
    mask_ar = op.generate_attention_mask(
        query=q_ar,
        key=k_ar,
        decoder_segment_ids=seg_ids,
        model_mode="autoregressive",
        compressed_mask=c_mask_ar,
    )
    self.assertEqual(mask_ar.shape, (batch_size, 1, 1, 1, kv_len))

    # 4. Compressed mask is None (fallback to uncompressed mask shape)
    mask_none = op.generate_attention_mask(
        query=q_train,
        key=k_train,
        decoder_segment_ids=None,
        model_mode="train",
        compressed_mask=None,
    )
    self.assertEqual(mask_none.ndim, 4)
    self.assertEqual(mask_none.shape[-1], kv_len)


class CompressedAttentionTest(parameterized.TestCase):
  """Parity and compilation tests for CompressedAttention (DeepSeek-V4)."""

  def setUp(self):
    """Setup test dependencies and configuration."""
    super().setUp()
    if not is_decoupled():
      jax.config.update("jax_remove_size_one_mesh_axis_from_type", True)

  @parameterized.named_parameters(
      {"testcase_name": "csa_ratio4_dot_product", "compress_ratio": 4, "attention_kernel": "dot_product"},
      {"testcase_name": "hca_ratio128_dot_product", "compress_ratio": 128, "attention_kernel": "dot_product"},
  )
  def test_compressed_attention_run(self, compress_ratio, attention_kernel):
    self._run_compressed_attention(compress_ratio, attention_kernel)

  @parameterized.named_parameters(
      {"testcase_name": "csa_ratio4_flash", "compress_ratio": 4, "attention_kernel": "flash"},
      {"testcase_name": "hca_ratio128_flash", "compress_ratio": 128, "attention_kernel": "flash"},
  )
  @pytest.mark.tpu_only
  def test_compressed_attention_flash(self, compress_ratio, attention_kernel):
    self._run_compressed_attention(compress_ratio, attention_kernel)

  @parameterized.named_parameters(
      {"testcase_name": "csa_ratio4", "compress_ratio": 4},
      {"testcase_name": "hca_ratio128", "compress_ratio": 128},
  )
  @pytest.mark.tpu_only
  def test_compressed_attention_flash_vs_dot_product(self, compress_ratio):
    """Direct forward-value numerical equivalence between dot_product and flash attention."""
    out_dot = self._run_compressed_attention(compress_ratio, "dot_product")
    out_flash = self._run_compressed_attention(compress_ratio, "flash")
    np.testing.assert_allclose(np.array(out_flash), np.array(out_dot), rtol=1e-2, atol=1e-2)

  @pytest.mark.tpu_only
  def test_hca_flash_vs_dot_product_unaligned_489(self):
    """Direct forward-value numerical equivalence between dot_product and flash attention for S=489."""
    out_dot = self._run_compressed_attention(128, "dot_product", seq_len=489)
    out_flash = self._run_compressed_attention(128, "flash", seq_len=489)
    np.testing.assert_allclose(np.array(out_flash), np.array(out_dot), rtol=1e-2, atol=1e-2)

  @pytest.mark.tpu_only
  def test_hca_flash_vs_dot_product_packed_crossing_window(self):
    """Verifies numerical equivalence between dot_product and flash attention on packed sequences crossing compression windows."""
    l1, l2 = 200, 312
    total_len = l1 + l2
    compress_ratio = 128
    cfg_dot = self._get_test_config(
        max_target_length=total_len,
        compress_ratio=compress_ratio,
        attention_kernel="dot_product",
    )
    attn_dot = self._create_compressed_attention_layer(cfg_dot, compress_ratio=compress_ratio, attention_kernel="dot_product")

    cfg_flash = self._get_test_config(
        max_target_length=total_len,
        compress_ratio=compress_ratio,
        attention_kernel="flash",
    )
    attn_flash = self._create_compressed_attention_layer(cfg_flash, compress_ratio=compress_ratio, attention_kernel="flash")

    batch_size = cfg_dot.global_batch_size_to_train_on

    # Generate random inputs and packed metadata
    x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, total_len, cfg_dot.base_emb_dim))
    pos = jnp.broadcast_to(
        jnp.concatenate([jnp.arange(l1, dtype=jnp.int32), jnp.arange(l2, dtype=jnp.int32)], axis=0)[None, :],
        (batch_size, total_len),
    )
    seg = jnp.broadcast_to(
        jnp.concatenate([jnp.ones(l1, dtype=jnp.int32), jnp.full(l2, 2, dtype=jnp.int32)], axis=0)[None, :],
        (batch_size, total_len),
    )

    out_dot, _ = attn_dot(
        x,
        x,
        decoder_segment_ids=seg,
        inputs_positions=pos,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    out_flash, _ = attn_flash(
        x,
        x,
        decoder_segment_ids=seg,
        inputs_positions=pos,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    np.testing.assert_allclose(
        np.array(out_flash),
        np.array(out_dot),
        rtol=1e-2,
        atol=1e-2,
        err_msg="Static flash attention does not match dot_product on packed sequence crossing compression window.",
    )

  def _get_test_config(self, max_target_length, compress_ratio, attention_kernel):
    """Initializes and returns a MaxTextConfig for document packing tests."""
    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test_packing_equivalence",
        "enable_checkpointing": False,
        "ici_fsdp_parallelism": 1,
        "ici_data_parallelism": -1,
        "ici_tensor_parallelism": 1,
        "ici_autoregressive_parallelism": 1,
        "max_target_length": max_target_length,
        "max_prefill_predict_length": max_target_length,
        "attention_type": AttentionType.COMPRESSED.value,
        "head_dim": 128,
        "q_lora_rank": 256,
        "kv_lora_rank": 256,
        "dtype": "float32",
        "use_tokamax_splash": True,
        "o_groups": 2,
        "o_lora_rank": 256,
        "compressed_rope_max_timescale": 160000,
        "rope_max_timescale": 10000,
        "qk_rope_head_dim": 64,
        "base_num_kv_heads": 1,
        "base_num_query_heads": 16,
    }
    return pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
    )

  def _create_compressed_attention_layer(self, cfg, compress_ratio, attention_kernel):
    """Instantiates a CompressedAttention layer with test configuration."""
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)
    return CompressedAttention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        inputs_q_shape=(cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.base_emb_dim),
        inputs_kv_shape=(cfg.global_batch_size_to_train_on, cfg.max_target_length, cfg.base_emb_dim),
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        mesh=mesh,
        attention_kernel=attention_kernel,
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        attention_type=AttentionType(cfg.attention_type),
        q_lora_rank=cfg.q_lora_rank,
        compress_ratio=compress_ratio,
        rngs=nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42)),
    )

  @parameterized.named_parameters(
      {
          "testcase_name": "csa_dot_product",
          "compress_ratio": 4,
          "attention_kernel": "dot_product",
          "l1": 32,
          "l2": 32,
      },
      {
          "testcase_name": "csa_flash",
          "compress_ratio": 4,
          "attention_kernel": "flash",
          "l1": 64,
          "l2": 64,
      },
      {
          "testcase_name": "hca_dot_product",
          "compress_ratio": 128,
          "attention_kernel": "dot_product",
          "l1": 128,
          "l2": 128,
      },
      {
          "testcase_name": "hca_flash",
          "compress_ratio": 128,
          "attention_kernel": "flash",
          "l1": 256,
          "l2": 256,
      },
  )
  @pytest.mark.tpu_only
  def test_packed_vs_unpacked_equivalence(self, compress_ratio, attention_kernel, l1, l2):
    """Asserts bitwise/numerical equivalence between packed and independent unpacked forward passes."""
    total_len = l1 + l2

    cfg = self._get_test_config(
        max_target_length=total_len,
        compress_ratio=compress_ratio,
        attention_kernel=attention_kernel,
    )
    attn = self._create_compressed_attention_layer(cfg, compress_ratio=compress_ratio, attention_kernel=attention_kernel)
    batch_size = cfg.global_batch_size_to_train_on

    # Generate distinct random tokens for Document 1 and Document 2
    key1, key2 = jax.random.split(jax.random.PRNGKey(42))
    x1 = jax.random.normal(key1, (batch_size, l1, cfg.base_emb_dim))
    x2 = jax.random.normal(key2, (batch_size, l2, cfg.base_emb_dim))

    pos1 = jnp.broadcast_to(jnp.arange(l1, dtype=jnp.int32)[None, :], (batch_size, l1))
    pos2 = jnp.broadcast_to(jnp.arange(l2, dtype=jnp.int32)[None, :], (batch_size, l2))
    seg1 = jnp.ones((batch_size, l1), dtype=jnp.int32)
    seg2 = jnp.ones((batch_size, l2), dtype=jnp.int32)

    # --- 1. UNPACKED (INDEPENDENT) PASSES ---
    out1_unpacked, _ = attn(
        x1,
        x1,
        decoder_segment_ids=seg1,
        inputs_positions=pos1,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    out2_unpacked, _ = attn(
        x2,
        x2,
        decoder_segment_ids=seg2,
        inputs_positions=pos2,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )
    expected_unpacked = jnp.concatenate([out1_unpacked, out2_unpacked], axis=1)  # [B, L1 + L2, D]

    # --- 2. PACKED (CONCATENATED) PASS ---
    x_packed = jnp.concatenate([x1, x2], axis=1)
    pos_packed = jnp.concatenate([pos1, pos2], axis=1)
    seg_packed = jnp.concatenate([jnp.full_like(seg1, 1), jnp.full_like(seg2, 2)], axis=1)

    out_packed, _ = attn(
        x_packed,
        x_packed,
        decoder_segment_ids=seg_packed,
        inputs_positions=pos_packed,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    # --- 3. ASSERT EXACT NUMERICAL EQUIVALENCE ---
    # Document 1 outputs must match
    np.testing.assert_allclose(
        np.array(out_packed[:, :l1, :]),
        np.array(out1_unpacked),
        rtol=5e-3,
        atol=5e-3,
        err_msg="Document 1 output in packed sequence does not match unpacked execution.",
    )

    # Document 2 outputs must match
    np.testing.assert_allclose(
        np.array(out_packed[:, l1:, :]),
        np.array(out2_unpacked),
        rtol=5e-3,
        atol=5e-3,
        err_msg="Document 2 output in packed sequence does not match unpacked execution.",
    )

    # Full concatenated sequence must match
    np.testing.assert_allclose(
        np.array(out_packed),
        np.array(expected_unpacked),
        rtol=5e-3,
        atol=5e-3,
    )

    # --- 4. ADVERSARIAL LEAKAGE CHECK ---
    # Mutating Document 1 by +1000.0 must have zero effect on Document 2 in the packed pass
    x_packed_corrupted = x_packed.at[:, :l1, :].set(x_packed[:, :l1, :] + 1000.0)
    out_packed_corrupted, _ = attn(
        x_packed_corrupted,
        x_packed_corrupted,
        decoder_segment_ids=seg_packed,
        inputs_positions=pos_packed,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    np.testing.assert_allclose(
        np.array(out_packed_corrupted[:, l1:, :]),
        np.array(out2_unpacked),
        rtol=5e-3,
        atol=5e-3,
        err_msg="Adversarial corruption in Doc 1 leaked into Doc 2 in packed sequence.",
    )

  def _run_compressed_attention(self, compress_ratio, attention_kernel, seq_len=None):
    """Runs CompressedAttention forward pass with specified compression ratio and kernel."""
    target_length = seq_len if seq_len is not None else 512
    # Setup test config
    config_arguments = {
        "per_device_batch_size": 1.0,
        "run_name": "test_compressed",
        "enable_checkpointing": False,
        "ici_fsdp_parallelism": 1,
        "ici_data_parallelism": -1,
        "ici_tensor_parallelism": 1,
        "ici_autoregressive_parallelism": 1,
        "max_target_length": target_length,
        "max_prefill_predict_length": target_length,
        "attention_type": AttentionType.COMPRESSED.value,
        "head_dim": 128,
        "q_lora_rank": 256,
        "kv_lora_rank": 256,
        "dtype": "float32",
        "use_tokamax_splash": True,
        "o_groups": 2,
        "o_lora_rank": 256,
        "compressed_rope_max_timescale": 160000,
        "rope_max_timescale": 10000,
        "qk_rope_head_dim": 64,
        "base_num_kv_heads": 1,
        "base_num_query_heads": 16,
    }
    cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        **config_arguments,
    )
    devices_array = maxtext_utils.create_device_mesh(cfg)
    mesh = Mesh(devices_array, cfg.mesh_axes)

    batch_size = cfg.global_batch_size_to_train_on
    cur_seq_len = cfg.max_target_length
    embed_dim = cfg.base_emb_dim

    # Inputs shape: [batch, seq_len, embed_dim]
    lnx = jax.random.normal(
        jax.random.PRNGKey(0),
        shape=(batch_size, cur_seq_len, embed_dim),
        dtype=jnp.float32,
    )
    decoder_positions = jnp.stack([jnp.arange(cur_seq_len, dtype=jnp.int32) for _ in range(batch_size)])
    decoder_segment_ids = jnp.ones((batch_size, cur_seq_len), dtype=jnp.int32)

    # Instantiate CompressedAttention
    attn = CompressedAttention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        inputs_q_shape=lnx.shape,
        inputs_kv_shape=lnx.shape,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        mesh=mesh,
        attention_kernel=attention_kernel,
        dtype=cfg.dtype,
        dropout_rate=cfg.dropout_rate,
        attention_type=AttentionType(cfg.attention_type),
        q_lora_rank=cfg.q_lora_rank,
        compress_ratio=compress_ratio,
        rngs=nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42)),
    )

    # Run forward pass (train mode)
    output, _ = attn(
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=MODEL_MODE_TRAIN,
    )

    self.assertEqual(output.shape, (batch_size, cur_seq_len, embed_dim))
    return output


class KVHeadShardingTest(parameterized.TestCase):
  """Tests that KV heads must divide the mesh axes that shard `kv_heads`.

  Attention heads are atomic under tensor parallelism, so a mesh that shards
  `kv_heads` more ways than there are heads has to be rejected. The mesh is
  faked here to keep the test hermetic on a single-device host; only
  `mesh.shape` is consulted when resolving logical axes onto mesh axes.
  """

  _KV_KERNEL_AXES = ("embed", "kv_heads", "kv_head_dim")
  _NUM_KV_HEADS = 2
  _EMBED_DIM = 16
  _INDIVISIBLE = r"num_kv_heads \(2\).*must be divisible"
  _INDIVISIBLE_BY_4 = r"num_kv_heads \(2\).*must be divisible by 4"

  def setUp(self):
    """Builds an attention layer with two KV heads on a single-device mesh."""
    super().setUp()
    self.cfg = pyconfig.initialize(
        [sys.argv[0], get_test_config_path()],
        per_device_batch_size=1.0,
        run_name="test",
        enable_checkpointing=False,
        max_target_length=128,
    )
    self.inputs_kv_shape = (1, self.cfg.max_target_length, self._EMBED_DIM)
    mesh = Mesh(maxtext_utils.create_device_mesh(self.cfg), self.cfg.mesh_axes)
    # A single-device mesh shards nothing, so construction always succeeds; each
    # test then swaps in a fake mesh to exercise the sharding check.
    self.attention = Attention(
        config=self.cfg,
        num_query_heads=self._NUM_KV_HEADS * 2,
        num_kv_heads=self._NUM_KV_HEADS,
        head_dim=self.cfg.head_dim,
        max_target_length=self.cfg.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        inputs_q_shape=self.inputs_kv_shape,
        inputs_kv_shape=self.inputs_kv_shape,
        mesh=mesh,
        attention_kernel="dot_product",
        dtype=self.cfg.dtype,
        dropout_rate=self.cfg.dropout_rate,
        attention_type=self.cfg.attention_type,
        model_mode=MODEL_MODE_PREFILL,
        rngs=nnx.Rngs(params=0, dropout=jax.random.PRNGKey(42)),
    )

  def _set_mesh_shape(self, **mesh_shape):
    """Replaces the attention mesh with one reporting `mesh_shape`."""
    self.attention.mesh = types.SimpleNamespace(shape=mesh_shape)

  def _use_ulysses(self):
    """Returns a context manager putting the layer on Ulysses context parallelism.

    Patched into the config's flat dictionary rather than assigned as an
    attribute, because `HyperParameters` is read-only after initialization.
    Not built through `pyconfig` either, because a genuine Ulysses config
    additionally demands TPU hardware, flash attention with Tokamax Splash and
    several other options that are irrelevant here.
    """
    return mock.patch.dict(
        self.attention.config.get_keys(),
        {"context_parallel_strategy": "ulysses"},
    )

  @parameterized.named_parameters(
      # `kv_heads` maps to tensor x tensor_sequence x autoregressive, so each of
      # those axes, and their product, constrains the KV head count.
      ("tensor", {"tensor": 4}),
      ("tensor_sequence", {"tensor_sequence": 4}),
      ("autoregressive", {"autoregressive": 4}),
      (
          "product_of_axes",
          {"tensor": 2, "tensor_sequence": 2, "autoregressive": 2},
      ),
  )
  def test_indivisible_kv_heads_rejected(self, mesh_shape):
    self._set_mesh_shape(**mesh_shape)
    with self.assertRaisesRegex(ValueError, self._INDIVISIBLE):
      self.attention.init_kv_w(inputs_kv_shape=self.inputs_kv_shape)

  @parameterized.named_parameters(
      ("exactly_divisible", {"tensor": 2}),
      ("size_one_axes_ignored", {"tensor": 2, "tensor_sequence": 1}),
      # fsdp shards `embed`, not `kv_heads`, so it places no constraint.
      ("axis_that_does_not_shard_kv_heads", {"fsdp": 4}),
      ("unsharded", {}),
  )
  def test_divisible_kv_heads_accepted(self, mesh_shape):
    # The validator is called directly rather than through `init_kv_w`, which
    # would go on to initialize parameters against the fake mesh.
    self._set_mesh_shape(**mesh_shape)
    self.attention._validate_kv_head_sharding(self._KV_KERNEL_AXES)  # pylint: disable=protected-access

  def test_replicated_kernel_axes_skip_validation(self):
    """A replicated KV projection is unconstrained even on an over-sharded mesh."""
    self._set_mesh_shape(tensor=4)
    self.attention._validate_kv_head_sharding((None, None, None))  # pylint: disable=protected-access

  def test_context_axis_ignored_without_ulysses(self):
    """Only Ulysses shards KV heads over the context axis."""
    self._set_mesh_shape(context=4)
    self.attention._validate_kv_head_sharding(self._KV_KERNEL_AXES)  # pylint: disable=protected-access

  def test_ulysses_context_axis_rejected(self):
    """Ulysses shards KV heads over context via all-to-all, not a logical rule."""
    self._set_mesh_shape(context=4)
    with (
        self._use_ulysses(),
        self.assertRaisesRegex(ValueError, self._INDIVISIBLE_BY_4),
    ):
      self.attention.init_kv_w(inputs_kv_shape=self.inputs_kv_shape)

  def test_ulysses_multiplies_with_tensor_parallelism(self):
    """The binding constraint is the product of the context and tensor axes.

    Two KV heads clear `context`=2 and `tensor`=1 individually, and `types.py`
    only checks the context factor, so the combined degree of 4 is caught here.
    """
    self._set_mesh_shape(context=2, tensor=2)
    with (
        self._use_ulysses(),
        self.assertRaisesRegex(ValueError, self._INDIVISIBLE_BY_4),
    ):
      self.attention.init_kv_w(inputs_kv_shape=self.inputs_kv_shape)

  def test_ulysses_divisible_accepted(self):
    self._set_mesh_shape(context=2)
    with self._use_ulysses():
      self.attention._validate_kv_head_sharding(self._KV_KERNEL_AXES)  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
