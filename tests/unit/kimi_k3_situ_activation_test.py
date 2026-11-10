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
import json
import os
import sys
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import numpy as np

from maxtext.configs import pyconfig
from maxtext.layers import linears
from maxtext.layers.linears import situ_and_mul, situ_gate, situ_linear, _convert_to_activation_function
from maxtext.utils import maxtext_utils
from tests.utils.kimi_k3_reference import KIMI_K3_REFERENCE_DIR, requires_kimi_k3_reference
from tests.utils.test_helpers import get_test_config_path

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


def _np_situ_and_mul(gate, up, cotangent, beta=4.0, linear_beta=25.0):
  """Hermetic float64 NumPy oracle for HF `SituAndMul`: returns (y, d_gate, d_up) for the VJP."""
  g = np.asarray(gate, np.float64)
  u = np.asarray(up, np.float64)
  ct = np.asarray(cotangent, np.float64)
  sig = 1.0 / (1.0 + np.exp(-g))
  tg = np.tanh(g / beta)
  situ_a = beta * tg * sig
  d_situ_a = (1.0 - tg**2) * sig + beta * tg * sig * (1.0 - sig)
  tu = np.tanh(u / linear_beta)
  lin = linear_beta * tu
  d_lin = 1.0 - tu**2
  return situ_a * lin, ct * d_situ_a * lin, ct * situ_a * d_lin


def _situ_inputs(dtype, shape=(8, 128)):
  """Gate/up/cotangent inputs spanning the soft-cap regime ([-30, 30] and beyond)."""
  k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
  gate = (jax.random.normal(k1, shape, jnp.float32) * 15.0).astype(dtype)
  up = (jax.random.normal(k2, shape, jnp.float32) * 40.0).astype(dtype)
  ct = jax.random.normal(k3, shape, jnp.float32).astype(dtype)
  return gate, up, ct


# Tolerances vs the float64 oracle. bf16 results are rounded once (half a bf16 ulp ~ 2e-3 relative). The fp32
# atol covers fp32 cancellation in `1 - tanh(x / beta)**2` (abs error ~1e-5) in the gate gradient.
_SITU_TOLS = {jnp.float32: {"rtol": 1e-5, "atol": 5e-5}, jnp.bfloat16: {"rtol": 4e-3, "atol": 1e-4}}


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

  def test_situ_and_mul_forward_and_vjp_match_numpy_oracle(self):
    """Hermetic (CI) check of the combined gate-times-up math and its VJP, fp32 and bf16."""
    for dtype in (jnp.float32, jnp.bfloat16):
      with self.subTest(dtype=dtype.__name__):
        gate, up, ct = _situ_inputs(dtype)
        y, vjp_fn = jax.vjp(situ_and_mul, gate, up)
        d_gate, d_up = vjp_fn(ct)
        for arr in (y, d_gate, d_up):
          self.assertEqual(arr.dtype, dtype)
        y_ref, d_gate_ref, d_up_ref = _np_situ_and_mul(
            np.asarray(gate, np.float32), np.asarray(up, np.float32), np.asarray(ct, np.float32)
        )
        tol = _SITU_TOLS[dtype]
        np.testing.assert_allclose(np.asarray(y, np.float32), y_ref, **tol)
        np.testing.assert_allclose(np.asarray(d_gate, np.float32), d_gate_ref, **tol)
        np.testing.assert_allclose(np.asarray(d_up, np.float32), d_up_ref, **tol)

  def test_composed_fp32_matches_numpy_oracle(self):
    """situ_gate(gate) * situ_linear(up) in fp32 (and its VJP) is the same math as the fused helper."""
    gate, up, ct = _situ_inputs(jnp.float32)
    y, vjp_fn = jax.vjp(lambda g, u: situ_gate(g) * situ_linear(u), gate, up)
    d_gate, d_up = vjp_fn(ct)
    y_ref, d_gate_ref, d_up_ref = _np_situ_and_mul(gate, up, ct)
    for got, want in ((y, y_ref), (d_gate, d_gate_ref), (d_up, d_up_ref)):
      np.testing.assert_allclose(np.asarray(got), want, **_SITU_TOLS[jnp.float32])

  def test_tolerance_rejects_perturbed_betas(self):
    """Negative control: a +1% beta error must fail the tolerances used above (incl. bf16)."""
    for dtype in (jnp.float32, jnp.bfloat16):
      gate, up, ct = _situ_inputs(dtype)
      y_ref, _, _ = _np_situ_and_mul(np.asarray(gate, np.float32), np.asarray(up, np.float32), np.asarray(ct, np.float32))
      for kwargs in ({"gate_beta": 4.04}, {"linear_beta": 25.25}):
        with self.subTest(dtype=dtype.__name__, **kwargs):
          y = situ_and_mul(gate, up, **kwargs)
          with self.assertRaises(AssertionError):
            np.testing.assert_allclose(np.asarray(y, np.float32), y_ref, **_SITU_TOLS[dtype])

  def test_situ_and_mul_bf16_rounds_once(self):
    """bf16 output equals the fp32 result cast to bf16 once (HF `SituAndMul` semantics)."""
    gate, up, _ = _situ_inputs(jnp.bfloat16)
    y = situ_and_mul(gate, up)
    expected = situ_and_mul(gate.astype(jnp.float32), up.astype(jnp.float32)).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(y, np.float32), np.asarray(expected, np.float32))
    self.assertEqual(situ_and_mul(gate, up, dtype=jnp.float32).dtype, jnp.float32)

  def test_situ_and_mul_linear_beta_none(self):
    """`linear_beta=None` leaves the up branch untransformed, as in HF."""
    gate, up, _ = _situ_inputs(jnp.float32)
    np.testing.assert_allclose(situ_and_mul(gate, up, linear_beta=None), situ_gate(gate) * up, rtol=1e-6, atol=1e-6)

  def test_check_grads(self):
    """Numerical-vs-autodiff gradient check (first and second order) in fp32."""
    x = jnp.array([-12.0, -3.0, -0.5, 0.0, 0.7, 2.5, 9.0, 40.0], jnp.float32)
    check_grads(situ_gate, (x,), order=2, modes=["fwd", "rev"], atol=1e-2, rtol=1e-2, eps=1e-2)
    check_grads(situ_linear, (x,), order=2, modes=["fwd", "rev"], atol=1e-2, rtol=1e-2, eps=1e-2)
    check_grads(situ_and_mul, (x, x[::-1]), order=1, modes=["fwd", "rev"], atol=1e-2, rtol=1e-2, eps=1e-2)

  def test_nonpositive_beta_rejected(self):
    x = jnp.ones((2,), jnp.float32)
    for beta in (0.0, -1.0):
      with self.assertRaises(ValueError):
        situ_gate(x, beta=beta)
      with self.assertRaises(ValueError):
        situ_linear(x, beta=beta)
      with self.assertRaises(ValueError):
        situ_and_mul(x, x, gate_beta=beta)
      with self.assertRaises(ValueError):
        situ_and_mul(x, x, linear_beta=beta)

  def test_mlp_block_uses_fused_situ_and_mul(self):
    """MlpBlock with the SiTU pair multiplies in fp32 and rounds once (fused and unfused wi)."""
    for fused_mlp in (False, True):
      with self.subTest(fused_mlp=fused_mlp):
        cfg = pyconfig.initialize(
            [sys.argv[0], get_test_config_path()],
            per_device_batch_size=1.0,
            run_name="test",
            enable_checkpointing=False,
            max_target_length=128,
            fused_mlp=fused_mlp,
        )
        mesh = jax.sharding.Mesh(maxtext_utils.create_device_mesh(cfg), cfg.mesh_axes)
        layer = linears.MlpBlock(
            config=cfg,
            mesh=mesh,
            in_features=16,
            intermediate_dim=32,
            activations=("situ_gate", "situ_linear"),
            intermediate_dropout_rate=0.0,
            dtype=jnp.bfloat16,
            rngs=nnx.Rngs(params=0, dropout=1),
        )
        inputs = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 16), jnp.float32).astype(jnp.bfloat16) * 8
        with mock.patch.object(linears, "situ_and_mul", wraps=linears.situ_and_mul) as spy:
          out = layer(inputs, deterministic=True)
        spy.assert_called_once()
        self.assertEqual(out.shape, (2, 3, 16))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  @unittest.skipIf(torch is None, "torch is not installed")
  @requires_kimi_k3_reference
  def test_parity_against_reference(self):
    """Numerical parity against HF SituAndMul, covering the soft-cap regime and bf16."""
    with open(os.path.join(KIMI_K3_REFERENCE_DIR, "config.json"), encoding="utf-8") as f:
      hf_config = json.load(f)
    hf_config = hf_config.get("text_config", hf_config)
    beta = hf_config["activation_situ_beta"]
    linear_beta = hf_config["activation_situ_linear_beta"]
    self.assertEqual((beta, linear_beta), (linears.KIMI_K3_SITU_BETA, linears.KIMI_K3_SITU_LINEAR_BETA))
    ref_situ_and_mul = _load_reference_situ_and_mul()(beta=beta, linear_beta=linear_beta)
    torch.manual_seed(42)
    x_pt = torch.randn(4, 16, 64, dtype=torch.float32) * 20.0
    for dtype_pt, dtype_jax in ((torch.float32, jnp.float32), (torch.bfloat16, jnp.bfloat16)):
      x = x_pt.to(dtype_pt)
      y_pt = ref_situ_and_mul(x).float().numpy()
      x_jax = jnp.asarray(x.float().numpy()).astype(dtype_jax)
      y_jax = situ_and_mul(x_jax[..., :32], x_jax[..., 32:], gate_beta=beta, linear_beta=linear_beta)
      self.assertEqual(y_jax.dtype, dtype_jax)
      np.testing.assert_allclose(np.asarray(y_jax, np.float32), y_pt, **_SITU_TOLS[dtype_jax])


if __name__ == "__main__":
  unittest.main()
