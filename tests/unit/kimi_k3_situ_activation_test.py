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

"""Unit tests for the Kimi-K3 SiTU activation functions (situ_gate / situ_linear)."""

import ast
import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers.linears import situ_gate, situ_linear, _convert_to_activation_function
from tests.utils.kimi_k3_reference import KIMI_K3_REFERENCE_DIR, requires_kimi_k3_reference

try:
  import torch  # pylint: disable=import-error
except ImportError:  # pragma: no cover - torch is optional for the hermetic tests.
  torch = None


def _load_reference_situ_and_mul():
  """Loads HF `SituAndMul` from the reference without importing `modeling_kimi_linear`.

  Importing the full reference module needs `fla` and a recent `transformers`; extracting just
  the class source keeps this test free of sys.modules / sys.path side effects.
  """
  path = os.path.join(KIMI_K3_REFERENCE_DIR, "modeling_kimi_linear.py")
  with open(path, encoding="utf-8") as f:
    tree = ast.parse(f.read())
  cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SituAndMul")
  namespace = {"torch": torch, "nn": torch.nn}
  exec(compile(ast.Module(body=[cls], type_ignores=[]), path, "exec"), namespace)  # pylint: disable=exec-used
  return namespace["SituAndMul"]


class KimiK3SituActivationTest(unittest.TestCase):
  """Tests validating SiTU activation values, properties and reference parity."""

  def test_registration(self):
    self.assertIs(_convert_to_activation_function("situ_gate"), situ_gate)
    self.assertIs(_convert_to_activation_function("situ_linear"), situ_linear)

  def test_golden_values(self):
    """Hard-coded values computed independently (Python math) for beta=4 / linear_beta=25."""
    x_gate = jnp.array([0.0, 1.0, -2.0, 10.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        situ_gate(x_gate), [0.0, 0.7161995568672704, -0.22034286184779028, 3.9462780318602517], rtol=1e-6, atol=1e-7
    )
    x_lin = jnp.array([0.0, 1.0, 100.0, -50.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        situ_linear(x_lin), [0.0, 0.9994670077790893, 24.983232493476677, -24.100689501895424], rtol=1e-6, atol=1e-7
    )

  def test_asymptotes_and_gradients(self):
    big = jnp.array([1e4, -1e4], dtype=jnp.float32)
    np.testing.assert_allclose(situ_linear(big), [25.0, -25.0], rtol=1e-6)
    np.testing.assert_allclose(situ_gate(big), [4.0, 0.0], atol=1e-6)
    small = jnp.linspace(-0.1, 0.1, 11, dtype=jnp.float32)
    np.testing.assert_allclose(situ_linear(small), small, atol=1e-5)
    for fn in (situ_gate, situ_linear):
      grads = jax.vmap(jax.grad(fn))(jnp.array([-1e3, -30.0, 0.0, 30.0, 1e3], dtype=jnp.float32))
      self.assertTrue(bool(jnp.all(jnp.isfinite(grads))))

  def test_bf16_preserves_dtype_and_matches_fp32(self):
    x = jax.random.normal(jax.random.PRNGKey(0), (256,), dtype=jnp.float32) * 20.0
    x_bf16 = x.astype(jnp.bfloat16)
    for fn in (situ_gate, situ_linear):
      y = fn(x_bf16)
      self.assertEqual(y.dtype, jnp.bfloat16)
      # fp32 compute + single cast back: result equals the fp32 result rounded to bf16.
      expected = fn(x_bf16.astype(jnp.float32)).astype(jnp.bfloat16)
      np.testing.assert_array_equal(np.asarray(y, np.float32), np.asarray(expected, np.float32))

  @unittest.skipIf(torch is None, "torch is not installed")
  @requires_kimi_k3_reference
  def test_parity_against_reference(self):
    """Numerical parity against HF SituAndMul, covering the soft-cap regime and bf16."""
    situ_and_mul = _load_reference_situ_and_mul()(beta=4.0, linear_beta=25.0)
    torch.manual_seed(42)
    x_pt = torch.randn(4, 16, 64, dtype=torch.float32) * 20.0
    for dtype_pt, dtype_jax, tol in ((torch.float32, jnp.float32, 1e-5), (torch.bfloat16, jnp.bfloat16, 2e-2)):
      x = x_pt.to(dtype_pt)
      y_pt = situ_and_mul(x).float().numpy()
      x_jax = jnp.asarray(x.float().numpy()).astype(dtype_jax)
      y_jax = situ_gate(x_jax[..., :32]) * situ_linear(x_jax[..., 32:])
      self.assertEqual(y_jax.dtype, dtype_jax)
      np.testing.assert_allclose(np.asarray(y_jax, np.float32), y_pt, rtol=tol, atol=tol)


if __name__ == "__main__":
  unittest.main()
