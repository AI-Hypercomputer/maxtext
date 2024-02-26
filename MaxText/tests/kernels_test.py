#  Copyright 2024 Google LLC
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

"""Tests for Kernels."""

import unittest

from flax.core import freeze
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import pytest
from kernels.ragged_attention import mqa_reference, ragged_mqa

class RaggedAttentionTest(unittest.TestCase):
  def _assert_allclose(self, a, b, **kwargs):
    if a.dtype == jnp.bfloat16:
      a = a.astype(np.float32)
    if b.dtype == jnp.bfloat16:
      b = b.astype(np.float32)
    np.testing.assert_allclose(a, b, **kwargs)

  """Test for Ragged Attention """
  def setUp(self):
    super().setUp()
    # TODO(patemotter): make more flexible/use config
    self.rng = jax.random.PRNGKey(0)
    self.batch_size = 96
    self.num_kv_heads = 16
    self.num_query_heads = 16
    self.max_target_length = 1024
    self.head_dim = 256
    self.bk = 128
    self.dtype = jnp.bfloat16
    self.lengths = jnp.array([10,100,512,256] * 24, dtype=jnp.int32) 
  

  def create_mqa_qkv(self):
    k1, k2, k3 = random.split(self.rng, 3)
    q = random.normal(k1, (self.batch_size, self.num_query_heads, self.head_dim), dtype=self.dtype) 
    k = random.normal(k2, (self.batch_size, self.max_target_length, self.head_dim), dtype=self.dtype)  
    v = random.normal(k3, (self.batch_size, self.max_target_length, self.head_dim), dtype=self.dtype)  
    return q, k, v
  

  def create_mha_qkv(self, q, k, v):
    # Expand to represent the shapes used for MHA
    k = jnp.expand_dims(k, axis=2) 
    v = jnp.expand_dims(v, axis=2)  
    k *= jnp.ones((self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim)) 
    v *= jnp.ones((self.batch_size, self.max_target_length, self.num_kv_heads, self.head_dim)) 

    # Swap the num_kv_heads and max_target_length dimensions to make Mosaic happy with last two dims
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)
    return q, k, v


  @pytest.mark.tpu
  def test_autoregression(self):
    q, k, v = self.create_mqa_qkv()

    # Compare the ragged MQA kernel directly with the reference version of MQA
    ragged_mqa_output, (ragged_mqa_max_logits, ragged_mqa_denom) = ragged_mqa(q, k, v, self.lengths, bk=self.bk)
    mqa_ref_output, (mqa_ref_max_logits, mqa_ref_denom) = mqa_reference(q, k, v, self.lengths)
    self._assert_allclose(ragged_mqa_output, mqa_ref_output, atol=3e-2, rtol=3e-2)
    self._assert_allclose(ragged_mqa_max_logits, mqa_ref_max_logits, atol=1e-3, rtol=1e-3)
    self._assert_allclose(ragged_mqa_denom, mqa_ref_denom, atol=1e-2, rtol=1e-3)

    # Compare the vmapped version of the MQA kernel to original 
    q_mha, k_mha, v_mha = self.create_mha_qkv(q, k, v)
    vmap_ragged_mqa = jax.vmap(ragged_mqa, in_axes=[None, 1, 1, None])
    vmap_ragged_output, (vmap_ragged_logits, vmap_ragged_denom) = vmap_ragged_mqa(q_mha, k_mha, v_mha, self.lengths)
    self._assert_allclose(ragged_mqa_output, vmap_ragged_output[0], atol=3e-2, rtol=3e-2)
    self._assert_allclose(mqa_ref_max_logits, vmap_ragged_logits[0], atol=1e-3, rtol=1e-3)
    self._assert_allclose(mqa_ref_denom, vmap_ragged_denom[0], atol=1e-3, rtol=1e-3)
