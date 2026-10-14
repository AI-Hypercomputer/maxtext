# Copyright 2023–2026 Google LLC
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

"""Tests for the native JAX MX recipes, quantization=jax_mxfp8 and jax_mxfp4."""

import sys
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import linears
from maxtext.layers import quantizations
from tests.utils.test_helpers import get_test_config_path


def _supports(mode: str) -> bool:
  try:
    jax.nn.get_scaled_dot_general_config(mode)
  except ValueError:
    return False
  return True


def _modes() -> list[str]:
  return [mode for mode in ("mxfp8", "mxfp4") if _supports(mode)]


def _config(**kwargs):
  """A tiny dense model whose contracting extents are all multiples of 32, with jax_mxfp8 by default."""
  init_kwargs = {
      "run_name": "test",
      "dataset_type": "synthetic",
      "enable_checkpointing": False,
      "enable_goodput_recording": False,
      "steps": 1,
      "per_device_batch_size": 1,
      "use_qwix_quantization": False,
      "skip_jax_distributed_system": True,
      "base_emb_dim": 64,
      "base_num_query_heads": 2,
      "base_num_kv_heads": 2,
      "head_dim": 32,
      "base_mlp_dim": 128,
      "base_num_decoder_layers": 1,
      "max_target_length": 32,
      "quantization": "jax_mxfp8",
  } | kwargs
  return pyconfig.initialize([sys.argv[0], get_test_config_path()], **init_kwargs)


class JaxMxQuantizationTest(unittest.TestCase):
  """The quantization object and the layer it builds."""

  def test_configure_quantization(self):
    for mode in _modes():
      with self.subTest(mode=mode):
        quant = quantizations.configure_quantization(_config(quantization=f"jax_{mode}"))
        self.assertIsInstance(quant, quantizations.JaxBlockScaledQuantization)
        self.assertEqual(quant.mode, mode)

  def test_layer_matches_scaled_dot_general(self):
    x = jax.random.normal(jax.random.key(1), (2, 32, 64), jnp.float32).astype(jnp.bfloat16)
    for mode in _modes():
      with self.subTest(mode=mode):
        self._check_layer_matches_scaled_dot_general(mode, x)

  def _check_layer_matches_scaled_dot_general(self, mode, x):
    """DenseGeneral with the MX backend computes exactly what jax.nn.scaled_dot_general computes."""
    layer = linears.DenseGeneral(
        in_features_shape=64,
        out_features_shape=128,
        weight_dtype=jnp.float32,
        dtype=jnp.bfloat16,
        quant=quantizations.JaxBlockScaledQuantization(mode=mode),
        rngs=nnx.Rngs(0),
    )
    kernel = layer.kernel[...].astype(jnp.bfloat16)
    config = jax.nn.get_scaled_dot_general_config(mode)

    def reference(lhs):
      return jax.nn.scaled_dot_general(
          lhs, kernel, (((2,), (0,)), ((), ())), preferred_element_type=jnp.bfloat16, configs=[config, config, config]
      )

    def loss(fn, lhs):
      return fn(lhs).astype(jnp.float32).sum()

    y = layer(x)
    self.assertEqual(y.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(y, np.float32), np.asarray(reference(x), np.float32))
    dx = jax.grad(loss, argnums=1)(layer, x)
    dx_ref = jax.grad(loss, argnums=1)(reference, x)
    np.testing.assert_array_equal(np.asarray(dx, np.float32), np.asarray(dx_ref, np.float32))


class JaxMxConfigValidationTest(unittest.TestCase):
  """Settings that would silently leave GEMMs unquantized or fail obscurely in JAX are rejected at config time."""

  def assert_rejected(self, regex, **kwargs):
    # pydantic's ValidationError, which pyconfig raises, is a ValueError.
    with self.assertRaisesRegex(ValueError, regex):
      _config(**kwargs)

  def test_qwix_flag(self):
    self.assert_rejected("does not support use_qwix_quantization=True", use_qwix_quantization=True)

  def test_explicit_sharding(self):
    self.assert_rejected("does not support shard_mode='explicit'", shard_mode="explicit")

  def test_moe(self):
    self.assert_rejected("does not support MoE models", num_experts=8, num_experts_per_tok=2, base_moe_mlp_dim=128)


if __name__ == "__main__":
  unittest.main()
