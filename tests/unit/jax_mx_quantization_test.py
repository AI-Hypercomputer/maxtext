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


MODES = ("mxfp8", "mxfp4")


def _supports(mode: str) -> bool:
  """Whether the installed JAX knows the MX mode (mxfp4 needs jax-ml/jax#41341)."""
  try:
    jax.nn.get_scaled_dot_general_config(mode)
  except (AttributeError, ValueError):
    return False
  return True


def _modes() -> list[str]:
  return [mode for mode in MODES if _supports(mode)]


def _config(**kwargs):
  """A tiny dense model with jax_mxfp8 by default."""
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
    # Configuration does not touch JAX, so both modes are tested whatever JAX is installed.
    for mode in MODES:
      with self.subTest(mode=mode):
        quant = quantizations.configure_quantization(_config(quantization=f"jax_{mode}"))
        self.assertIsInstance(quant, quantizations.JaxBlockScaledQuantization)
        self.assertEqual(quant.mode, mode)

  def test_unsupported_mode_fails_at_construction(self):
    with self.assertRaises(ValueError):
      quantizations.JaxBlockScaledDotGeneral(mode="not_a_mode")

  def test_contracting_dims_checked_per_gemm(self):
    """Each GEMM's contracting dim is checked on the layer's real shapes."""
    if not _supports("mxfp8"):
      self.skipTest("the installed JAX has no mxfp8 config")
    cases = {
        "forward": {"in_features": 48, "out_features": 64, "tokens": (2, 32)},
        "input gradient": {"in_features": 64, "out_features": 48, "tokens": (2, 32)},
        "weight gradient": {"in_features": 64, "out_features": 64, "tokens": (2, 24)},
    }
    for gemm, case in cases.items():
      with self.subTest(gemm=gemm):
        layer = linears.DenseGeneral(
            in_features_shape=case["in_features"],
            out_features_shape=case["out_features"],
            weight_dtype=jnp.float32,
            dtype=jnp.bfloat16,
            quant=quantizations.JaxBlockScaledQuantization(mode="mxfp8"),
            rngs=nnx.Rngs(0),
        )
        x = jnp.zeros((*case["tokens"], case["in_features"]), jnp.bfloat16)
        with self.assertRaisesRegex(ValueError, rf"the {gemm} \(.*\) GEMM contracts over 48 elements"):
          layer(x)

  def test_layer_matches_scaled_dot_general(self):
    if not _modes():
      self.skipTest("the installed JAX supports no MX mode")
    x = jax.random.normal(jax.random.key(1), (2, 32, 64), jnp.float32).astype(jnp.bfloat16)
    for mode in _modes():
      with self.subTest(mode=mode):
        self._check_layer_matches_scaled_dot_general(mode, x)
    if set(_modes()) == set(MODES):
      with self.subTest(mode="mxfp4", gradient_mode="mxfp8"):
        self._check_layer_matches_scaled_dot_general("mxfp4", x, gradient_mode="mxfp8")

  def _check_layer_matches_scaled_dot_general(self, mode, x, gradient_mode=None):
    """DenseGeneral with the MX backend computes exactly what jax.nn.scaled_dot_general computes."""
    layer = linears.DenseGeneral(
        in_features_shape=64,
        out_features_shape=128,
        weight_dtype=jnp.float32,
        dtype=jnp.bfloat16,
        quant=quantizations.JaxBlockScaledQuantization(mode=mode, gradient_mode=gradient_mode),
        rngs=nnx.Rngs(0),
    )
    kernel = layer.kernel[...].astype(jnp.bfloat16)
    config = jax.nn.get_scaled_dot_general_config(mode)
    gradient_config = jax.nn.get_scaled_dot_general_config(gradient_mode or mode)

    def reference(lhs, rhs=kernel):
      return jax.nn.scaled_dot_general(
          lhs,
          rhs,
          (((2,), (0,)), ((), ())),
          preferred_element_type=jnp.bfloat16,
          configs=[config, config, gradient_config],
      )

    def loss(fn, lhs):
      return (fn(lhs).astype(jnp.float32) ** 2).sum()

    y = layer(x)
    self.assertEqual(y.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(y, np.float32), np.asarray(reference(x), np.float32))
    dx = jax.grad(loss, argnums=1)(layer, x)
    dx_ref = jax.grad(loss, argnums=1)(reference, x)
    np.testing.assert_array_equal(np.asarray(dx, np.float32), np.asarray(dx_ref, np.float32))
    dk = nnx.grad(lambda m: loss(m, x))(layer)["kernel"][...]
    dk_ref = jax.grad(lambda w: loss(lambda lhs: reference(lhs, w), x))(kernel)
    np.testing.assert_array_equal(np.asarray(dk, np.float32), np.asarray(dk_ref, np.float32))


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

  def test_fp8_weight_dtype(self):
    self.assert_rejected("weight_dtype must not be fp8", weight_dtype="float8_e4m3fn")


if __name__ == "__main__":
  unittest.main()
