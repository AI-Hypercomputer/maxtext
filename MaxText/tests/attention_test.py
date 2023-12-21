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
FlashMultiHeadDotProductAttention = attentions.FlashMultiHeadDotProductAttention
MultiHeadDotProductAttention = attentions.MultiHeadDotProductAttention
MultiQueryDotProductAttention = attentions.MultiQueryDotProductAttention
GroupedQueryDotProductAttention = attentions.GroupedQueryDotProductAttention
LLaMARotaryEmbedding = embeddings.LLaMARotaryEmbedding


class AttentionTest(unittest.TestCase):
  """Test for the Attention """
  def setUp(self):
    super().setUp()
    pyconfig.initialize([sys.argv[0], 'configs/base.yml'], per_device_batch_size = 1.0, run_name='test', enable_checkpointing=False)
    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(0)

    devices_array = max_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

    self.global_batch_size = self.cfg.global_batch_size_to_train_on
    self.num_heads = self.cfg.base_num_heads
    self.max_target_length = self.cfg.max_target_length
    self.head_dim = self.cfg.head_dim
    self.embed_dim = self.cfg.base_emb_dim
    self.dtype = self.cfg.dtype

    self.mha_attention = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        mesh=self.mesh,
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name='self_attention',
    )

    self.mha_variable = self.mha_attention.init(
        {'params': self.rng, 'aqt': self.rng},
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length)),
    )

    self.flash_attention = FlashMultiHeadDotProductAttention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        mesh=self.mesh,
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name='self_attention',
        max_target_length=self.max_target_length
    )

    self.flash_variable = self.flash_attention.init(
        {'params': self.rng, 'aqt': self.rng},
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length)),
    )

  def get_data(self, dtype):
    lnx = jax.random.uniform(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=dtype,
    )
    decoder_segment_ids = jnp.ones(
        shape=(self.global_batch_size, self.max_target_length), dtype=np.int32
    )

    def batch_positions():
      return [
          jnp.arange(self.max_target_length, dtype=jnp.int32)
          for _ in range(self.global_batch_size)
      ]

    if self.global_batch_size > 1:
      decoder_positions = jnp.stack(batch_positions())

    return lnx , decoder_segment_ids, decoder_positions

  @pytest.mark.tpu
  def test_attention(self):
    """Test equalvant between MHA and Flash MHA."""

    lnx, decoder_segment_ids, decoder_positions = self.get_data(
        self.dtype)

    mha_output = self.mha_attention.apply(
        self.mha_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        decode=False,
        rngs={'aqt': self.rng},
    )
    flash_output = self.flash_attention.apply(
        self.flash_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        deterministic=True,
        decode=False,
        rngs={'aqt': self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(
            flash_output, mha_output, rtol=1e-01, atol=1e-01, equal_nan=False
        )
    )

if __name__ == '__main__':
  unittest.main()
