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

import sys
import unittest

import common_types

from flax.core import freeze
import jax
import jax.numpy as jnp
import max_utils
import numpy as np
import pytest

import pyconfig

from layers import attentions
from layers import embeddings

Mesh = jax.sharding.Mesh
Attention = attentions.Attention


class AttentionTest(unittest.TestCase):
  """Test for the Attention """
  def setUp(self):
    super().setUp()
    pyconfig.initialize([sys.argv[0], 'configs/base.yml'], per_device_batch_size = 1.0, run_name='test', enable_checkpointing=False, max_target_length=128, max_prefill_predict_length=16 )
    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(0)

    devices_array = max_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

    self.global_batch_size = self.cfg.global_batch_size_to_train_on
    self.num_kv_heads = self.cfg.num_kv_heads
    self.num_query_heads = self.cfg.num_query_heads
    self.max_target_length = self.cfg.max_target_length
    self.head_dim = self.cfg.head_dim
    self.embed_dim = self.cfg.base_emb_dim
    self.dtype = self.cfg.dtype

    self._attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel = "dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name='self_attention',
    )

    self._attention_as_mha_generic_variable = self._attention_as_mha_generic.init(
        {'params': self.rng, 'aqt': self.rng},
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length)),
    )

  def get_data(self, dtype):
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )

    decoder_segment_ids = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, 4)
    decoder_positions = jax.random.randint(self.rng, (self.global_batch_size, self.max_target_length), 0, self.max_target_length)

    return lnx, decoder_segment_ids, decoder_positions

  def get_structured_data(self, dtype):
    lnx = jax.random.normal(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )

    decoder_positions = jnp.stack([
          jnp.arange(self.max_target_length, dtype=jnp.int32)
          for _ in range(self.global_batch_size)
    ])

    decoder_segment_ids = jax.numpy.zeros((self.global_batch_size, self.max_target_length))\
                          + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR

    return lnx, decoder_segment_ids, decoder_positions
  
  @pytest.mark.tpu
  def test_autoregression(self):
    prefill_length = self.cfg.max_prefill_predict_length
    decode_total_length = self.cfg.max_target_length
    lnx, decoder_segment_ids, decoder_positions = self.get_structured_data(
        self.dtype)
    
    mha_full = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={'aqt': self.rng},
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
        model_mode=common_types.MODEL_MODE_PREFILL,
        rngs={'aqt': self.rng},
        mutable=["cache"]
    )

    self.assertTrue(
        jax.numpy.allclose(
            mha_prefill, mha_full[:,:prefill_length,:], rtol=1e-02, atol=1e-02, equal_nan=False
        )
    )

    for idx in range(prefill_length, decode_total_length):
      lnx_idx = lnx[:, idx:idx+1, :]
      decoder_positions_idx = decoder_positions[:, idx:idx+1]
      self._attention_as_mha_generic_variable.update(output_cache)
      mha_idx, output_cache = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx_idx,
        lnx_idx,
        inputs_positions=decoder_positions_idx,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
        rngs={'aqt': self.rng},
        mutable=["cache"]
      )

      mha_full_this_idx = mha_full[:,idx:idx+1,:]
      self.assertTrue(
        mha_full_this_idx.shape == mha_idx.shape
      )
      self.assertTrue(
        jax.numpy.allclose(
            mha_full_this_idx, mha_idx, rtol=1e-02, atol=1e-02, equal_nan=False
        )
      )

  @pytest.mark.tpu
  def test_tpu_kernel_attention_mha(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads)

  @pytest.mark.tpu
  def test_tpu_kernel_attention_gqa(self):
    self.tpu_kernel_attention_helper(self.num_kv_heads // 2)

  @pytest.mark.tpu
  def test_tpu_kernel_attention_mqa(self):
    self.tpu_kernel_attention_helper(1)

  def tpu_kernel_attention_helper(self, num_kv_heads):
    """Test equalvant between dot_product and TPU accelerated"""

    lnx, decoder_segment_ids, decoder_positions = self.get_data(
        self.dtype)

    attention_as_mha_generic = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel = "dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name='self_attention',
    )

    attention_as_mha_generic_variable = attention_as_mha_generic.init(
        {'params': self.rng, 'aqt': self.rng},
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length)),
    )

    mha_generic_output = attention_as_mha_generic.apply(
        attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={'aqt': self.rng},
    )

    attention_as_mha_flash = Attention(
        config=self.cfg,
        num_query_heads=self.num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=self.head_dim,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.cfg.max_prefill_predict_length,
        mesh=self.mesh,
        attention_kernel = "flash",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name='self_attention',
    )

    attention_as_mha_flash_variable = attention_as_mha_flash.init(
        {'params': self.rng, 'aqt': self.rng},
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length)),
    )

    mha_generic_flash_output = attention_as_mha_flash.apply(
        attention_as_mha_flash_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        model_mode=common_types.MODEL_MODE_TRAIN,
        rngs={'aqt': self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(
            mha_generic_output, mha_generic_flash_output, rtol=1e-01, atol=1e-01, equal_nan=False
        )
    )

if __name__ == '__main__':
  unittest.main()
