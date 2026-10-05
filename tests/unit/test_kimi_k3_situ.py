# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Kimi-K3 SiTU (Shifted-Truncated Unit) activation functions."""

import os
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch

_ORIG_MATMUL_PRECISION = None


def setUpModule():
  """Sets up high precision matmul and mocks reference modules."""
  global _ORIG_MATMUL_PRECISION
  _ORIG_MATMUL_PRECISION = jax.config.jax_default_matmul_precision
  jax.config.update("jax_default_matmul_precision", "highest")

  ref_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../kimi-k3-hf-reference"))
  if os.path.exists(ref_dir) and ref_dir not in sys.path:
    sys.path.insert(0, ref_dir)

  import types  # pylint: disable=import-outside-toplevel
  import transformers.utils.generic  # pylint: disable=import-outside-toplevel

  if not hasattr(transformers.utils.generic, "OutputRecorder"):

    class OutputRecorder:

      def __init__(self, *args, **kwargs):
        pass

    transformers.utils.generic.OutputRecorder = OutputRecorder
  if not hasattr(transformers.utils.generic, "check_model_inputs"):
    transformers.utils.generic.check_model_inputs = lambda *args, **kwargs: None

  for mod in ["fla", "fla.modules", "fla.ops", "fla.ops.kda", "fla.ops.utils", "fla.ops.utils.index", "fla.utils"]:
    if mod not in sys.modules:
      m = types.ModuleType(mod)
      sys.modules[mod] = m
      m.FusedRMSNormGated = None
      m.ShortConvolution = None
      m.chunk_kda = None
      m.fused_recurrent_kda = None
      m.prepare_cu_seqlens_from_mask = None
      m.prepare_lens_from_mask = None
      m.tensor_cache = lambda f: f


def tearDownModule():
  if _ORIG_MATMUL_PRECISION is not None:
    jax.config.update("jax_default_matmul_precision", _ORIG_MATMUL_PRECISION)


from maxtext.layers.linears import situ_gate, situ_linear, _convert_to_activation_function
from tests.utils.kimi_k3_reference import requires_kimi_k3_reference


class KimiK3SituActivationTest(unittest.TestCase):
  """Tests validating SiTU activations mathematical properties and reference parity."""

  def test_situ_activation_values_and_registration(self):
    """Verifies SiTU activation mathematical properties and conversion registration."""
    # Test zero
    self.assertEqual(float(situ_gate(jnp.array(0.0))), 0.0)
    self.assertEqual(float(situ_linear(jnp.array(0.0))), 0.0)

    # Test values
    x = jnp.array([1.0, 2.0, -1.0])
    expected_gate = 4.0 * jnp.tanh(x / 4.0) * jax.nn.sigmoid(x)
    expected_linear = 25.0 * jnp.tanh(x / 25.0)
    np.testing.assert_allclose(situ_gate(x), expected_gate, rtol=1e-6)
    np.testing.assert_allclose(situ_linear(x), expected_linear, rtol=1e-6)

    # Test registration
    fn_gate = _convert_to_activation_function("situ_gate")
    fn_linear = _convert_to_activation_function("situ_linear")
    self.assertEqual(fn_gate, situ_gate)
    self.assertEqual(fn_linear, situ_linear)

  @requires_kimi_k3_reference
  def test_situ_activation_parity_against_reference(self):
    """Verifies numerical parity of SiTU activation against PyTorch SituAndMul."""
    import modeling_kimi_linear  # pylint: disable=import-outside-toplevel

    torch.manual_seed(42)
    pt_act = modeling_kimi_linear.SituAndMul(beta=4.0, linear_beta=25.0)
    x_pt = torch.randn(4, 16, 64, dtype=torch.float32)
    y_pt = pt_act(x_pt).detach().numpy()

    gate_jax = x_pt[..., :32].numpy()
    up_jax = x_pt[..., 32:].numpy()
    y_jax = np.array(situ_gate(jnp.array(gate_jax)) * situ_linear(jnp.array(up_jax)))

    np.testing.assert_allclose(y_jax, y_pt, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
