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
import random
import sys
import types
import unittest
from unittest import mock

from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.sharding import AxisType, Mesh
from maxtext.utils import maxtext_utils
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
from maxtext.layers import attention_op
from maxtext.layers.attention_op import (
    AttentionOp,
    ChunkedCausalMask,
    _generate_chunk_attention_mask,
    _make_bidirectional_block_mask,
)
from maxtext.layers.attentions import Attention
from maxtext.layers import embeddings
from maxtext.kernels.attention import jax_flash_attention
from maxtext.configs import pyconfig
from maxtext.models.qwen3 import Qwen3NextGatedDeltaNet
import numpy as np
import pytest

from tests.utils import attention_test_util
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
        rtol=2e-3,
        atol=2e-3,
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
  @mock.patch("tpu_inference.layers.jax.attention_interface.sharded_ragged_paged_attention", create=True)
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

    topk_indices = jnp.array([[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]])
    indexer_mask = mla.indexer.generate_mask(topk_indices, kv_len) + attention_mask

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

    topk_indices = jnp.array([[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]])
    indexer_mask = mla.indexer.generate_mask(topk_indices, kv_len) + attention_mask

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


if __name__ == "__main__":
  unittest.main()
