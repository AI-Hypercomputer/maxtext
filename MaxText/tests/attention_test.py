"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

""" Tests for Attention """
import jax
import unittest
import jax.numpy as jnp
import max_utils
from jax.sharding import Mesh
from layers import attentions
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest

import pyconfig
import sys

MultiHeadDotProductAttention = attentions.MultiHeadDotProductAttention


class AttentionTest(unittest.TestCase):
  """Test for the Attention """
  def setUp(self):
    super().setUp()
    pyconfig.initialize([sys.argv[0], 'configs/base.yml'], per_device_batch_size = 1.0, run_name='test', enable_checkpointing=False)
    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(0)

    devices_array = max_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)
    
    self.BS = self.cfg.global_batch_size_to_train_on
    self.NUM_HEADS = self.cfg.base_num_heads
    self.MAX_TARGET_LENGTH = self.cfg.max_target_length
    self.HEAD_DIM = self.cfg.head_dim
    self.BASE_EMB_DIM = self.cfg.base_emb_dim
    
    # initialize attention
    self.attention = MultiHeadDotProductAttention(
        num_heads=self.NUM_HEADS,
        dtype=self.cfg.dtype,
        head_dim=self.HEAD_DIM,
        dropout_rate=self.cfg.dropout_rate,
        name='self_attention',
        config=self.cfg,
        mesh = self.mesh)
    self.variable = self.attention.init({'params': self.rng, 'aqt': self.rng}, jnp.ones((self.BS, self.MAX_TARGET_LENGTH, self.BASE_EMB_DIM)), 
                            jnp.ones((self.BS, self.MAX_TARGET_LENGTH, self.BASE_EMB_DIM)), 'flash')


  def get_decoder_mask(self):
    a = jnp.stack([jnp.tri(self.MAX_TARGET_LENGTH, dtype = 'bfloat16')[jnp.newaxis,:] for _ in range(self.BS)])
    return a

  def get_data(self):
    lnx = jax.random.uniform(self.rng, shape = (self.BS, self.MAX_TARGET_LENGTH, self.BASE_EMB_DIM),dtype = 'bfloat16')
    decoder_segment_ids = jnp.ones(shape = (self.BS, self.MAX_TARGET_LENGTH), dtype = np.int32)
    def batch_positions():
      return [jnp.arange(self.MAX_TARGET_LENGTH, dtype=jnp.int32) for _ in range(self.BS)]
    if self.BS > 1:
      decoder_positions = jnp.stack(batch_positions())
    decoder_mask = self.get_decoder_mask()
    return lnx, decoder_mask, decoder_segment_ids, decoder_positions

  @pytest.mark.tpu
  def test_attention(self):
    lnx, decoder_mask, decoder_segment_ids, decoder_positions = self.get_data()

    mha_output = self.attention.apply(
            self.variable,
            lnx,
            lnx,
            decoder_segment_ids = decoder_segment_ids,
            attention_type='mha',
            inputs_positions = decoder_positions,
            mask = decoder_mask,
            bias = None,
            deterministic=True,
            decode=False,
            rngs={'aqt': self.rng}
    )
    flash_output = self.attention.apply(
            self.variable,
            lnx,
            lnx,
            decoder_segment_ids = decoder_segment_ids,
            attention_type='flash',
            inputs_positions = decoder_positions,
            mask = decoder_mask,
            bias = None,
            deterministic=True,
            decode=False,
            rngs={'aqt': self.rng}
    )
    self.assertTrue(jax.numpy.allclose(flash_output, mha_output, rtol=1e-01, atol=1e-01, equal_nan=False))


if __name__ == '__main__':
  unittest.main()