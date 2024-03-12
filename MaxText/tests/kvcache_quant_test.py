"""
 Copyright 2024 Google LLC

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

""" Tests for the quantizations """
from jax import numpy as jnp
# from jax import random, lax
# from flax import linen as nn
import pyconfig
from layers import attentions, quantizations
import unittest

_ARRAY = [
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  [[10., 12.], [13., 14.], [15., 16.]],
  [[20., 22.], [23., 24.], [25., 26.]],
  [[30., 32.], [33., 34.], [35., 36.]],
  ]

_VALUE = [
  [[64, 127], [96, 127], [106, 127]],
  [[106, 127], [118, 127], [120, 127]],
  [[116, 127], [122, 127], [123, 127]],
  [[120, 127], [124, 127], [124, 127]]
  ]

_SCALE = [
  [[ 2.], [ 4.], [ 6.]],
  [[12.], [14.], [16.]],
  [[22.], [24.], [26.]],
  [[32.], [34.], [36.]]
  ]

_KV = jnp.array(_ARRAY, dtype=jnp.float32)
_KV_VAL = jnp.array(_VALUE, dtype=jnp.int8)
_KV_SCALE = jnp.array(_SCALE, dtype=jnp.int8)


def _getAttentionOp(quantize_kv:bool=False, num_query_heads=3, num_kv_heads=1):
  return attentions.AttentionOp(
    None,
    max_target_length=10,
    max_prefill_predict_length=5,
    float32_qk_product=False,
    float32_logits=False,
    quant=None,
    quantize_kv=quantize_kv,
    num_query_heads=num_query_heads,
    num_kv_heads=num_kv_heads,
    dropout_rate = 0,
    dtype=jnp.bfloat16
)

class KVCacheQuantTest(unittest.TestCase):
  """Tests for KV cache quantization."""

  def test_configure_quantization_is_true(self):
    pyconfig.initialize([None, "configs/base.yml"], quantize_kvcache=True, enable_checkpointing=False)
    config = pyconfig.config
    quantize_kv = quantizations.configure_kv_quantization(config)
    self.assertTrue(quantize_kv)

  def test_configure_quantization_is_false(self):
    pyconfig.initialize([None, "configs/base.yml"], enable_checkpointing=False)
    config = pyconfig.config
    quantize_kv = quantizations.configure_kv_quantization(config)
    self.assertFalse(quantize_kv)
    pyconfig.initialize([None, "configs/base.yml"], quantize_kvcache=False, enable_checkpointing=False)
    config = pyconfig.config
    quantize_kv = quantizations.configure_kv_quantization(config)
    self.assertFalse(quantize_kv)

  def test_kv_quantize(self):
    kv_val, kv_scale = quantizations.quantize_kv(_KV)
    self.assertTrue(jnp.array_equal(kv_val, _KV_VAL))
    self.assertTrue(jnp.array_equal(kv_scale, _KV_SCALE))

  def test_kv_unquantize(self):
    kv = quantizations.unquantize_kv(_KV_VAL, _KV_SCALE, jnp.float32)
    self.assertEqual(kv.dtype, jnp.float32)
    self.assertTrue(jnp.array_equal(jnp.rint(kv), jnp.rint(_KV)))






if __name__ == '__main__':
  unittest.main()

