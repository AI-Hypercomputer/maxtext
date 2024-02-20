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
from jax import random, lax
from flax import linen as nn
import numpy as np
import pyconfig
from layers import quantizations
import unittest

class QuantTestModule(nn.Module):
  """Test module for einsum."""
  quantization: quantizations.AqtQuantization

  @nn.compact
  def __call__(self, inputs):
    identity = jnp.identity(2, dtype=inputs.dtype)
    einsum = jnp.einsum
    dot_general = lax.dot_general
    if self.quantization:
      einsum = self.quantization.einsum()
      dot_general_cls = self.quantization.dot_general_cls()
      dot_general = dot_general_cls()
    res_einsum = einsum('bc,ab->ac', inputs, identity)
    res_dg = dot_general(inputs, inputs, (((), ()), ((), ())), precision=None)
    return res_einsum, res_dg

def _configure_quantization(quant_str="", mode_str='train'):
  pyconfig.initialize([None, "configs/base.yml"], enable_checkpointing=False, quantization=quant_str)
  config = pyconfig.config
  quant = quantizations.configure_quantization(config, mode_str)
  return quant

def _apply(quant_str=""):
  quant = _configure_quantization(quant_str)
  test_module = QuantTestModule(quant)
  rng = random.PRNGKey(0)
  variables = test_module.init({'params': rng}, jnp.ones((2, 2)))
  inputs = jnp.ones((2, 2))
  res_einsum, res_dg = test_module.apply(variables, inputs, rngs={'params': random.PRNGKey(0)})
  return inputs, res_einsum, res_dg


class QuantizationTest(unittest.TestCase):
  """Tests for quantization."""

  def test_in_quant_mode(self):
    quant = _configure_quantization(quant_str="int8", mode_str='convert')
    self.assertTrue(quantizations.in_convert_mode(quant))
    self.assertFalse(quantizations.in_serve_mode(quant))


  def test_configure_quantization_is_null(self):
    for quant_mode in ["train", "serve", "convert"]:
      quant = _configure_quantization(quant_str="", mode_str=quant_mode)
      self.assertEqual(quant, None)

  def test_configure_quantization_is_int8(self):
    for quant_mode in ["train", "serve", "convert"]:
      quant = _configure_quantization(quant_str="int8", mode_str=quant_mode)
      self.assertNotEqual(quant, None)
      self.assertEqual(quant.quant_dg, quantizations.AQT_INT8_CONFIG)

  def test_aqt_quantization(self):
    # Without quantization
    inputs, res_einsum, res_dg = _apply()
    self.assertTrue(jnp.array_equal(inputs, res_einsum))
    self.assertEqual(res_einsum.dtype, np.dtype(np.float32))
    self.assertTrue(jnp.array_equal(inputs, res_dg[0][0]))
    self.assertEqual(res_dg.dtype, np.dtype(np.float32))

    # With int8 quantization
    inputs, res_einsum, res_dg = _apply(quant_str="int8")
    self.assertTrue(jnp.greater(jnp.max(inputs), jnp.max(res_einsum)))
    self.assertEqual(res_einsum.dtype, np.dtype(np.float32))
    self.assertTrue(jnp.greater(jnp.max(inputs), jnp.max(res_dg[0][0])))
    #self.assertEqual(res_dg.dtype, np.dtype(np.float32))

if __name__ == '__main__':
  unittest.main()
