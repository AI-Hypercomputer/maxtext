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
Attention = attentions.Attention
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
    self.num_kv_heads = self.cfg.num_kv_heads
    self.num_query_heads = self.cfg.num_query_heads
    self.max_target_length = self.cfg.max_target_length
    self.head_dim = self.cfg.head_dim
    self.embed_dim = self.cfg.base_emb_dim
    self.dtype = self.cfg.dtype

    self._attention_as_mha_generic = Attention(
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
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

  @pytest.mark.tpu
  def test_attention(self):
    """Test equalvant between MHA and Flash MHA."""

    lnx, decoder_segment_ids, decoder_positions = self.get_data(
        self.dtype)
    
    mha_generic_output = self._attention_as_mha_generic.apply(
        self._attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        decode=False,
        rngs={'aqt': self.rng},
    )

    attention_as_mha_flash = Attention(
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
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
        decode=False,
        rngs={'aqt': self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(
            mha_generic_output, mha_generic_flash_output, rtol=1e-01, atol=1e-01, equal_nan=False
        )
    )

  @pytest.mark.tpu
  def test_multiquery_attention(self):
    attention_as_mqa = Attention(
        num_query_heads=self.num_query_heads,
        num_kv_heads=1,
        head_dim=self.head_dim,
        mesh=self.mesh,
        attention_kernel = "dot_product",
        dtype=self.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name='self_attention',
    )

    attention_as_mqa_variable = attention_as_mqa.init(
        {'params': self.rng, 'aqt': self.rng},
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length)),
    )

    lnx, decoder_segment_ids, decoder_positions = self.get_data(
        self.dtype)

    mqa_generic_output = attention_as_mqa.apply(
        attention_as_mqa_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        decode=False,
        rngs={'aqt': self.rng},
    )

    attention_as_mha_generic_variable = self._attention_as_mha_generic.init(
        {'params': self.rng, 'aqt': self.rng},
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length)),
    )

    new_key_kernel = jax.numpy.repeat(attention_as_mqa_variable['params']['key']['kernel'].value, self.num_kv_heads, axis=1)
    attention_as_mha_generic_variable['params']['key']['kernel'] = attention_as_mha_generic_variable['params']['key']['kernel'].replace(value = new_key_kernel)
    new_value_kernel = jax.numpy.repeat(attention_as_mqa_variable['params']['value']['kernel'].value, self.num_kv_heads, axis=1)
    attention_as_mha_generic_variable['params']['value']['kernel'] = attention_as_mha_generic_variable['params']['value']['kernel'].replace(value = new_value_kernel)
    new_out_kernel = attention_as_mqa_variable['params']['out']['kernel'].value
    attention_as_mha_generic_variable['params']['out']['kernel'] = attention_as_mha_generic_variable['params']['out']['kernel'].replace(value = new_out_kernel)
    new_query_kernel = attention_as_mqa_variable['params']['query']['kernel'].value
    attention_as_mha_generic_variable['params']['query']['kernel'] = attention_as_mha_generic_variable['params']['query']['kernel'].replace(value = new_query_kernel)

    mha_generic = self._attention_as_mha_generic.apply(
        attention_as_mha_generic_variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_positions,
        inputs_positions=decoder_segment_ids,
        deterministic=True,
        decode=False,
        rngs={'aqt': self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(
            mqa_generic_output, mha_generic, rtol=1e-06, atol=1e-06, equal_nan=False
        )
    )

if __name__ == '__main__':
  unittest.main()
